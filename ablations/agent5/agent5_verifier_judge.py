from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional

from sage_common import (
    SYMPTOMS,
    generate_text,
    load_llm,
    safe_json_loads,
)


JUDGE_PROMPT = """You are the final clinical extraction judge in VERGE.

You are NOT performing a new extraction and you do NOT have the clinical note.
Adjudicate only from the structured extraction, verification, refinement, and
loop records provided below. Do not invent new patient facts.

1. ORIGINAL CLAIM FROM AGENT 2
Target finding: {feature}
Original prediction: {original_prediction}
Original evidence: "{original_evidence}"
Original note section: {original_section}
Original experiencer: {original_experiencer}
Original assertion status: {original_assertion}
Original family relation: {original_relation}
Original cancer type: {original_cancer_type}

2. FINAL CLAIM STATE AFTER THE VERIFY--REFINE LOOP
Final current prediction: {current_prediction}
Final current evidence: "{current_evidence}"
Final note section: {current_section}
Final experiencer: {current_experiencer}
Final assertion status: {current_assertion}
Final family relation: {current_relation}
Final cancer type: {current_cancer_type}

3. FINAL AGENT 3 VERIFIER ASSESSMENT
Verifier verdict: {verifier_verdict}
Grounding level: {grounding_level}
Exact match: {exact_match}
Normalized exact match: {normalized_match}
Token-subsequence match: {token_subsequence}
ROUGE-L: {rouge_l}
Modified BLEU: {bleu}
BERTScore precision: {bertscore}
Issues found: {issues_list}
Verifier reasoning: {verifier_reasoning}
Verifier assessment applies directly to final claim: {verification_applies_to_final_claim}

4. MOST RECENT AGENT 4 REFINEMENT
Refinement available: {refinement_available}
Corrected prediction: {corrected_prediction}
Corrected evidence: "{corrected_evidence}"
Corrected note section: {corrected_section}
Corrected experiencer: {corrected_experiencer}
Corrected assertion status: {corrected_assertion}
Corrected family relation: {corrected_relation}
Corrected cancer type: {corrected_cancer_type}
Change type: {change_type}
Evidence grounded after refinement: {refinement_evidence_grounded}
Correction reasoning: {correction_reasoning}
Operational refinement error: {refinement_operational_error}

5. COMPLETE VERIFY--REFINE HISTORY
{history_text}

6. LOOP STATUS
Label history: {label_history}
Total refinement rounds: {total_refinement_rounds}
Total label flips: {flip_count}
Label stable: {label_stable}
Exit reason: {exit_reason}
Oscillation/human-review condition: {oscillation}
Deterministic human-review reasons: {forced_review_reasons}

7. AUXILIARY EXTRACTOR LABEL-PROBABILITY AUDIT
P(Yes): {p_yes}
P(No): {p_no}
Normalized P(Yes): {p_yes_norm}
Binary label entropy: {label_entropy}

DECISION RULES
- Use the complete recorded extraction--verification--refinement trail.
- Give greatest weight to grounded evidence and the final verification state.
- Do not infer a positive finding from diagnoses, medications, procedures,
  laboratory results, tests, risk factors, or other indirect information.
- A positive final label should have a defensible evidence trail. Strong or
  moderate grounding with passed clinical validation supports a positive label.
- A negative label is appropriate when valid direct positive evidence was not
  established or when a proposed positive correction was rejected.
- A recovered negative should become Yes only when the recorded recovery is
  grounded and directly supports the target finding.
- For family history of colorectal cancer, a positive decision requires a
  biological family relation and colorectal/colon/rectal/bowel cancer
  specificity in the recorded evidence trail.
- If the final Agent 3 verdict is VERIFIED, do not reverse the final loop claim
  without a concrete contradiction in the recorded trail.
- If the loop exited LABEL_STABLE while Agent 3 still requested REFINE,
  adjudicate from the verifier problems, the Refiner's unchanged correction,
  and the complete history.
- Extractor label probabilities are auxiliary uncertainty information only.
  They must not independently determine the final label.
- If deterministic human-review reasons are present, still provide a binary
  Yes/No final label, but the system will force for_human_review=True
  regardless of your requested value.
- Be conservative when the recorded trail does not establish a positive claim.

Return JSON only:
{{
  "final_label": "Yes" or "No",
  "confidence": "HIGH" or "MODERATE" or "LOW",
  "decision_source": "extractor" or "verifier" or "refiner",
  "explanation": "<one concise sentence explaining the final decision>",
  "for_human_review": true or false
}}"""


def _optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _strict_bool(value: Any, default: bool = False) -> bool:
    parsed = _optional_bool(value)
    return default if parsed is None else parsed


def _judge_schema_valid(parsed: Any) -> bool:
    if not isinstance(parsed, dict):
        return False

    label = str(parsed.get("final_label", "") or "").strip()
    if label not in {"Yes", "No"}:
        return False

    confidence = str(parsed.get("confidence", "") or "").strip().upper()
    if confidence not in {"HIGH", "MODERATE", "LOW"}:
        return False

    source = str(parsed.get("decision_source", "") or "").strip().lower()
    if source not in {"extractor", "verifier", "refiner"}:
        return False

    explanation = str(parsed.get("explanation", "") or "").strip()
    if not explanation:
        return False

    if _optional_bool(parsed.get("for_human_review")) is None:
        return False

    return True


def _empty_probability_audit(reason: str = "") -> Dict[str, Any]:
    return {
        "valid": False,
        "chosen_label": "",
        "p_yes": None,
        "p_no": None,
        "p_yes_norm": None,
        "p_no_norm": None,
        "yes_minus_no_logit_margin": None,
        "binary_entropy": None,
        "probability_scope": "",
        "reason": reason,
    }


def _binary_choice_probability_audit(
    prompt: str,
    response: str,
    chosen_label: str,
    field_name: str = "final_label",
) -> Dict[str, Any]:
    chosen_label = str(chosen_label or "").strip()
    if chosen_label not in {"Yes", "No"}:
        return _empty_probability_audit("invalid chosen label")

    response = str(response or "")
    pattern = re.compile(
        rf'"{re.escape(field_name)}"\s*:\s*"',
        flags=re.IGNORECASE,
    )
    match = pattern.search(response)
    if match is None:
        return _empty_probability_audit(
            f"could not locate JSON field {field_name!r} in raw generation"
        )

    generated_prefix = response[:match.end()]

    try:
        import torch

        tokenizer, model = load_llm()

        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"

        base = formatted + generated_prefix

        yes_ids = tokenizer(
            base + "Yes",
            add_special_tokens=True,
        )["input_ids"]
        no_ids = tokenizer(
            base + "No",
            add_special_tokens=True,
        )["input_ids"]

        common = 0
        limit = min(len(yes_ids), len(no_ids))
        while common < limit and yes_ids[common] == no_ids[common]:
            common += 1

        if common == 0 or common >= len(yes_ids) or common >= len(no_ids):
            return _empty_probability_audit(
                "could not establish first divergent Yes/No token"
            )

        yes_token_id = int(yes_ids[common])
        no_token_id = int(no_ids[common])

        if yes_token_id == no_token_id:
            return _empty_probability_audit(
                "counterfactual Yes/No tokens did not diverge"
            )

        prefix_ids = yes_ids[:common]
        if not prefix_ids:
            return _empty_probability_audit("empty aligned generation prefix")

        input_ids = torch.tensor(
            [prefix_ids],
            dtype=torch.long,
            device=model.device,
        )
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits[0, -1]

        yes_logit = float(logits[yes_token_id].item())
        no_logit = float(logits[no_token_id].item())

        probs = torch.softmax(logits, dim=-1)
        p_yes = float(probs[yes_token_id].item())
        p_no = float(probs[no_token_id].item())

        denom = p_yes + p_no
        if denom <= 0:
            return _empty_probability_audit(
                "non-positive binary probability denominator"
            )

        p_yes_norm = p_yes / denom
        p_no_norm = p_no / denom

        eps = 1e-12
        entropy = -(
            p_yes_norm * math.log(max(p_yes_norm, eps))
            + p_no_norm * math.log(max(p_no_norm, eps))
        )

        yes_suffix_len = len(yes_ids) - common
        no_suffix_len = len(no_ids) - common
        scope = (
            "complete_label"
            if yes_suffix_len == 1 and no_suffix_len == 1
            else "first_divergent_token_preference"
        )

        return {
            "valid": True,
            "chosen_label": chosen_label,
            "p_yes": p_yes,
            "p_no": p_no,
            "p_yes_norm": p_yes_norm,
            "p_no_norm": p_no_norm,
            "yes_minus_no_logit_margin": yes_logit - no_logit,
            "binary_entropy": entropy,
            "probability_scope": scope,
            "yes_token_id": yes_token_id,
            "no_token_id": no_token_id,
            "first_divergent_token_position": common,
            "reason": "",
        }

    except Exception as error:
        return _empty_probability_audit(
            f"probability audit failed: {error}"
        )


def _safe_text(value: Any) -> str:
    """Convert optional values to stable prompt text."""
    if value is None:
        return ""
    return str(value)


def _valid_binary_label(
    value: Any,
    fallback: str = "No",
) -> str:
    text = str(value or "").strip().lower()
    if text == "yes":
        return "Yes"
    if text == "no":
        return "No"
    return fallback if fallback in {"Yes", "No"} else "No"


def _format_metric(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_probability(
    label_probs: Optional[Dict[str, float]],
    key: str,
) -> str:
    if not label_probs:
        return "N/A"

    value = label_probs.get(key, -1)

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if numeric < 0:
        return "N/A"

    return f"{numeric:.6f}"


def _history_operational_failure(
    history: List[Dict[str, Any]],
) -> bool:
    for record in history:
        if not isinstance(record, dict):
            continue

        if _strict_bool(record.get("operational_error", False)):
            return True

        if _strict_bool(record.get("refinement_operational_error", False)):
            return True

        if _strict_bool(record.get("verification_operational_error", False)):
            return True

        action = str(record.get("refiner_action", "") or "").upper()
        if action == "REFINEMENT_ERROR":
            return True

        issues = record.get("issues_found", [])
        if isinstance(issues, list):
            if any(
                str(issue).startswith(
                    ("VERIFICATION_ERROR:", "VERIFICATION_OPERATIONAL_ERROR:")
                )
                for issue in issues
            ):
                return True

    return False


def _derive_forced_review_reasons(
    loop_info: Dict[str, Any],
    verification: Dict[str, Any],
    refinement: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> List[str]:
    reasons: List[str] = []

    oscillation = _strict_bool(
        loop_info.get("human_review_oscillation", False)
    )
    exit_reason = str(
        loop_info.get("exit_reason", "") or ""
    ).upper()

    if oscillation or exit_reason == "MAX_ROUNDS_LABEL_OSCILLATION":
        reasons.append("MAX_ROUNDS_LABEL_OSCILLATION")

    if _strict_bool(loop_info.get("operational_failure", False)):
        reasons.append("LOOP_OPERATIONAL_FAILURE")

    if _strict_bool(refinement.get("operational_error", False)):
        reasons.append("REFINEMENT_OPERATIONAL_FAILURE")

    if (
        str(refinement.get("change_type", "") or "").upper()
        == "REFINEMENT_ERROR"
    ):
        reasons.append("REFINEMENT_ERROR")

    if _history_operational_failure(history):
        reasons.append("HISTORY_OPERATIONAL_FAILURE")

    grounding_level = str(
        verification.get("grounding_level", "") or ""
    ).upper()

    issues = verification.get("issues_found", [])
    if grounding_level == "ERROR":
        reasons.append("VERIFICATION_ERROR")

    if _strict_bool(verification.get("operational_error", False)):
        reasons.append("VERIFICATION_OPERATIONAL_FAILURE")

    if isinstance(issues, list) and any(
        str(issue).startswith(
            ("VERIFICATION_ERROR:", "VERIFICATION_OPERATIONAL_ERROR:")
        )
        for issue in issues
    ):
        reasons.append("VERIFICATION_ERROR")

    return list(dict.fromkeys(reasons))


def _history_to_text(
    history: List[Dict[str, Any]],
) -> str:
    if not history:
        return "No refinement rounds were recorded."

    try:
        return json.dumps(
            history,
            ensure_ascii=False,
            indent=2,
        )
    except Exception:
        return str(history)


def _default_decision_source(
    original_prediction: str,
    current_prediction: str,
    verification: Dict[str, Any],
    refinement: Dict[str, Any],
) -> str:
    """Choose a valid fallback source without inventing unsupported enums."""
    if refinement and _strict_bool(refinement.get("refined", False)):
        return "refiner"

    if verification:
        return "verifier"

    return "extractor"


def judge_claim(
    structured_claim: Dict[str, Any],
    verification: Optional[Dict[str, Any]] = None,
    refinement: Optional[Dict[str, Any]] = None,
    loop_info: Optional[Dict[str, Any]] = None,
    label_probs: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    original_claim = dict(structured_claim or {})
    loop = dict(loop_info or {})

    feature = str(original_claim.get("feature", ""))

    original_prediction = _valid_binary_label(
        original_claim.get("prediction", "No"),
        fallback="No",
    )

    original_evidence = str(
        original_claim.get("evidence_quote", "") or ""
    )


    final_claim_raw = loop.get("final_claim")
    final_claim = (
        dict(final_claim_raw)
        if isinstance(final_claim_raw, dict)
        else dict(original_claim)
    )

    current_prediction = _valid_binary_label(
        final_claim.get("prediction", original_prediction),
        fallback=original_prediction,
    )

    current_evidence = str(
        final_claim.get("evidence_quote", "") or ""
    )

    loop_verification = loop.get("final_verification")
    final_verification = (
        dict(loop_verification)
        if isinstance(loop_verification, dict)
        else dict(verification or {})
    )

    loop_refinement = loop.get("final_refinement")
    final_refinement = (
        dict(loop_refinement)
        if isinstance(loop_refinement, dict)
        else dict(refinement or {})
    )

    history_raw = loop.get("history", [])
    history: List[Dict[str, Any]] = (
        list(history_raw)
        if isinstance(history_raw, list)
        else []
    )

    label_history_raw = loop.get(
        "label_history",
        [original_prediction],
    )
    label_history = (
        list(label_history_raw)
        if isinstance(label_history_raw, list)
        else [original_prediction]
    )

    total_refinement_rounds = int(
        loop.get(
            "total_refinement_rounds",
            loop.get("total_rounds", 0),
        )
        or 0
    )

    flip_count = int(
        loop.get(
            "flip_count",
            loop.get("consecutive_flips", 0),
        )
        or 0
    )

    label_stable = _strict_bool(
        loop.get("label_stable", False)
    )

    exit_reason = str(
        loop.get("exit_reason", "") or ""
    )

    oscillation = _strict_bool(
        loop.get("human_review_oscillation", False)
    )

    verifier_verdict = str(
        final_verification.get("verdict", "") or ""
    )

    grounding_level = str(
        final_verification.get("grounding_level", "") or ""
    )

    grounding_scores = final_verification.get(
        "grounding_scores",
        {},
    )
    if not isinstance(grounding_scores, dict):
        grounding_scores = {}

    issues = final_verification.get(
        "issues_found",
        [],
    )
    if not isinstance(issues, list):
        issues = [str(issues)]

    verifier_reasoning = str(
        final_verification.get("verdict_reasoning", "") or ""
    )

    refinement_available = bool(final_refinement)

    corrected_prediction = _valid_binary_label(
        final_refinement.get(
            "corrected_prediction",
            current_prediction,
        ),
        fallback=current_prediction,
    )

    corrected_evidence = str(
        final_refinement.get("corrected_evidence", "") or ""
    )

    change_type = str(
        final_refinement.get("change_type", "UNCHANGED")
        or "UNCHANGED"
    )

    correction_reasoning = str(
        final_refinement.get("correction_reasoning", "") or ""
    )

    refinement_operational_error = _strict_bool(
        final_refinement.get("operational_error", False)
    )

    forced_review_reasons = _derive_forced_review_reasons(
        loop_info=loop,
        verification=final_verification,
        refinement=final_refinement,
        history=history,
    )

    force_human_review = bool(forced_review_reasons)

    missing_loop_state = (
        bool(loop_info)
        and not isinstance(loop.get("final_claim"), dict)
    )
    if missing_loop_state:
        forced_review_reasons.append("MISSING_FINAL_CLAIM_STATE")
        force_human_review = True

    if not final_verification:
        forced_review_reasons.append("MISSING_FINAL_VERIFICATION")
        force_human_review = True

    forced_review_reasons = list(
        dict.fromkeys(forced_review_reasons)
    )

    history_text = _history_to_text(history)

    issues_list = (
        "; ".join(str(issue) for issue in issues)
        if issues
        else "none"
    )

    forced_review_text = (
        "; ".join(forced_review_reasons)
        if forced_review_reasons
        else "none"
    )

    prompt = JUDGE_PROMPT.format(
        feature=feature,

        original_prediction=original_prediction,
        original_evidence=original_evidence or "(none)",
        original_section=_safe_text(
            original_claim.get("note_section", "")
        ) or "unknown",
        original_experiencer=_safe_text(
            original_claim.get("experiencer", "")
        ) or "unknown",
        original_assertion=_safe_text(
            original_claim.get("assertion_status", "")
        ) or "unknown",
        original_relation=_safe_text(
            original_claim.get("relation_found", "")
        ) or "(none)",
        original_cancer_type=_safe_text(
            original_claim.get("cancer_type_found", "")
        ) or "(none)",

        current_prediction=current_prediction,
        current_evidence=current_evidence or "(none)",
        current_section=_safe_text(
            final_claim.get("note_section", "")
        ) or "unknown",
        current_experiencer=_safe_text(
            final_claim.get("experiencer", "")
        ) or "unknown",
        current_assertion=_safe_text(
            final_claim.get("assertion_status", "")
        ) or "unknown",
        current_relation=_safe_text(
            final_claim.get("relation_found", "")
        ) or "(none)",
        current_cancer_type=_safe_text(
            final_claim.get("cancer_type_found", "")
        ) or "(none)",

        verifier_verdict=verifier_verdict or "(missing)",
        grounding_level=grounding_level or "(missing)",
        exact_match=grounding_scores.get(
            "exact_match",
            "N/A",
        ),
        normalized_match=grounding_scores.get(
            "normalized_match",
            "N/A",
        ),
        token_subsequence=grounding_scores.get(
            "token_subsequence",
            "N/A",
        ),
        rouge_l=_format_metric(
            grounding_scores.get("rouge_l")
        ),
        bleu=_format_metric(
            grounding_scores.get("bleu")
        ),
        bertscore=_format_metric(
            grounding_scores.get("bertscore")
        ),
        issues_list=issues_list,
        verifier_reasoning=verifier_reasoning or "(none)",
        verification_applies_to_final_claim=_strict_bool(
            loop.get("final_verification_applies_to_final_claim", True),
            default=True,
        ),

        refinement_available=refinement_available,
        corrected_prediction=corrected_prediction,
        corrected_evidence=corrected_evidence or "(none)",
        corrected_section=_safe_text(
            final_refinement.get("corrected_section", "")
        ) or "(none)",
        corrected_experiencer=_safe_text(
            final_refinement.get("corrected_experiencer", "")
        ) or "(none)",
        corrected_assertion=_safe_text(
            final_refinement.get("corrected_assertion", "")
        ) or "(none)",
        corrected_relation=_safe_text(
            final_refinement.get("relation_found", "")
        ) or "(none)",
        corrected_cancer_type=_safe_text(
            final_refinement.get("cancer_type_found", "")
        ) or "(none)",
        change_type=change_type,
        refinement_evidence_grounded=final_refinement.get(
            "evidence_grounded",
            "N/A",
        ),
        correction_reasoning=correction_reasoning or "(none)",
        refinement_operational_error=refinement_operational_error,

        history_text=history_text,

        label_history=json.dumps(
            label_history,
            ensure_ascii=False,
        ),
        total_refinement_rounds=total_refinement_rounds,
        flip_count=flip_count,
        label_stable=label_stable,
        exit_reason=exit_reason or "(not recorded)",
        oscillation=oscillation,
        forced_review_reasons=forced_review_text,

        p_yes=_format_probability(
            label_probs,
            "p_yes",
        ),
        p_no=_format_probability(
            label_probs,
            "p_no",
        ),
        p_yes_norm=_format_probability(
            label_probs,
            "p_yes_norm",
        ),
        label_entropy=_format_probability(
            label_probs,
            "label_entropy",
        ),
    )

    fallback_source = _default_decision_source(
        original_prediction=original_prediction,
        current_prediction=current_prediction,
        verification=final_verification,
        refinement=final_refinement,
    )

    result: Dict[str, Any] = {
        "feature": feature,
        "original_prediction": original_prediction,
        "current_prediction": current_prediction,
        "final_label": current_prediction,
        "confidence": "LOW",
        "decision_source": fallback_source,
        "explanation": "",
        "for_human_review": force_human_review,
        "forced_human_review": force_human_review,
        "forced_review_reasons": forced_review_reasons,
        "verifier_verdict": verifier_verdict,
        "grounding_level": grounding_level,
        "refiner_change": change_type,
        "total_refinement_rounds": total_refinement_rounds,
        "flip_count": flip_count,
        "label_history": label_history,
        "label_stable": label_stable,
        "exit_reason": exit_reason,
        "human_review_oscillation": oscillation,
        "history": history,
        "label_probs": label_probs or {},
        "judge_generated_label": "",
        "judge_label_probs": {},
        "judge_primary_schema_valid": False,
        "judge_retry_used": False,
        "judge_retry_schema_valid": False,
        "judge_parse_success": False,
        "judge_operational_error": False,
    }

    response = ""
    parsed = None
    primary_schema_valid = False
    retry_used = False

    try:
        response = generate_text(
            prompt,
            max_new_tokens=192,
        )
        parsed = safe_json_loads(response)
        primary_schema_valid = _judge_schema_valid(parsed)
    except Exception:
        parsed = None
        primary_schema_valid = False

    result["judge_primary_schema_valid"] = primary_schema_valid

    if not primary_schema_valid:
        retry_used = True
        retry_prompt = (
            prompt
            + "\n\nIMPORTANT RETRY: Return exactly one valid JSON object "
              "with all five required fields and no additional text. "
              "final_label must be Yes or No; confidence must be HIGH, "
              "MODERATE, or LOW; decision_source must be extractor, verifier, "
              "or refiner; explanation must be non-empty; for_human_review "
              "must be a JSON boolean."
        )
        try:
            response = generate_text(
                retry_prompt,
                max_new_tokens=192,
            )
            parsed = safe_json_loads(response)
        except Exception:
            parsed = None

    retry_schema_valid = _judge_schema_valid(parsed)

    result["judge_retry_used"] = retry_used
    result["judge_retry_schema_valid"] = retry_schema_valid

    if not retry_schema_valid:
        result["judge_operational_error"] = True
        result["for_human_review"] = True
        result["forced_human_review"] = True

        reason = (
            "JUDGE_SCHEMA_ERROR"
            if isinstance(parsed, dict)
            else "JUDGE_PARSE_ERROR"
        )
        if reason not in result["forced_review_reasons"]:
            result["forced_review_reasons"].append(reason)

        result["explanation"] = (
            "Judge generation did not produce the required structured schema; "
            "the final loop label was retained."
        )
        return result

    result["judge_parse_success"] = True

    proposed_label = _valid_binary_label(
        parsed.get("final_label", ""),
        fallback=current_prediction,
    )

    result["judge_generated_label"] = proposed_label
    result["judge_label_probs"] = _binary_choice_probability_audit(
        prompt=(
            prompt
            if not retry_used
            else retry_prompt
        ),
        response=response,
        chosen_label=proposed_label,
        field_name="final_label",
    )

    result["final_label"] = proposed_label

    confidence = str(
        parsed.get("confidence", "LOW") or "LOW"
    ).strip().upper()
    if confidence in {"HIGH", "MODERATE", "LOW"}:
        result["confidence"] = confidence

    decision_source = str(
        parsed.get("decision_source", fallback_source)
        or fallback_source
    ).strip().lower()
    if decision_source in {
        "extractor",
        "verifier",
        "refiner",
    }:
        result["decision_source"] = decision_source

    result["explanation"] = str(
        parsed.get("explanation", "") or ""
    ).strip()

    judge_requests_review = _strict_bool(
        parsed.get("for_human_review", False)
    )

    # Deterministic flags can only be added to, never cleared.
    result["for_human_review"] = (
        judge_requests_review
        or force_human_review
    )

    return result

def judge_all_claims(
    structured_claims: Dict[str, Dict[str, Any]],
    verifications: Optional[
        Dict[str, Dict[str, Any]]
    ] = None,
    refinements: Optional[
        Dict[str, Dict[str, Any]]
    ] = None,
    all_loop_info: Optional[
        Dict[str, Dict[str, Any]]
    ] = None,
    all_label_probs: Optional[
        Dict[str, Dict[str, float]]
    ] = None,
) -> Dict[str, Dict[str, Any]]:
    decisions: Dict[str, Dict[str, Any]] = {}

    verification_map = verifications or {}
    refinement_map = refinements or {}
    loop_map = all_loop_info or {}
    probability_map = all_label_probs or {}

    for feature in SYMPTOMS:
        claim = structured_claims.get(feature)
        loop_info = loop_map.get(feature, {})
        verification = verification_map.get(feature, {})
        refinement = refinement_map.get(feature, {})
        label_probs = probability_map.get(feature)

        if claim is None:
            decisions[feature] = {
                "feature": feature,
                "original_prediction": "",
                "current_prediction": "No",
                "final_label": "No",
                "confidence": "LOW",
                "decision_source": "extractor",
                "explanation": "Missing Agent 2 structured claim.",
                "for_human_review": True,
                "forced_human_review": True,
                "forced_review_reasons": [
                    "MISSING_AGENT2_CLAIM"
                ],
                "verifier_verdict": "",
                "grounding_level": "",
                "refiner_change": "",
                "total_refinement_rounds": 0,
                "flip_count": 0,
                "label_history": [],
                "label_stable": False,
                "exit_reason": "",
                "human_review_oscillation": False,
                "history": [],
                "label_probs": label_probs or {},
                "judge_generated_label": "",
                "judge_label_probs": {},
                "judge_primary_schema_valid": False,
                "judge_retry_used": False,
                "judge_retry_schema_valid": False,
                "judge_parse_success": False,
                "judge_operational_error": True,
            }
            continue

        try:
            decisions[feature] = judge_claim(
                structured_claim=claim,
                verification=verification,
                refinement=refinement,
                loop_info=loop_info,
                label_probs=label_probs,
            )
        except Exception as error:
            final_claim = (
                loop_info.get("final_claim", {})
                if isinstance(loop_info, dict)
                else {}
            )

            fallback_label = _valid_binary_label(
                (
                    final_claim.get("prediction")
                    if isinstance(final_claim, dict)
                    else None
                ),
                fallback=_valid_binary_label(
                    claim.get("prediction", "No"),
                    fallback="No",
                ),
            )

            decisions[feature] = {
                "feature": feature,
                "original_prediction": _valid_binary_label(
                    claim.get("prediction", "No"),
                    fallback="No",
                ),
                "current_prediction": fallback_label,
                "final_label": fallback_label,
                "confidence": "LOW",
                "decision_source": "verifier",
                "explanation": (
                    f"Judge failed: {str(error)}"
                ),
                "for_human_review": True,
                "forced_human_review": True,
                "forced_review_reasons": [
                    "UNHANDLED_JUDGE_ERROR"
                ],
                "verifier_verdict": "",
                "grounding_level": "",
                "refiner_change": "",
                "total_refinement_rounds": 0,
                "flip_count": 0,
                "label_history": [],
                "label_stable": False,
                "exit_reason": "",
                "human_review_oscillation": False,
                "history": [],
                "label_probs": label_probs or {},
                "judge_generated_label": "",
                "judge_label_probs": {},
                "judge_primary_schema_valid": False,
                "judge_retry_used": False,
                "judge_retry_schema_valid": False,
                "judge_parse_success": False,
                "judge_operational_error": True,
            }

    return decisions


def compute_improvement_summary(
    pre_verifications: Dict[str, Dict[str, Any]],
    post_verifications: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    improved = 0
    degraded = 0
    unchanged = 0
    total = 0
    details: List[Dict[str, Any]] = []

    for feature in SYMPTOMS:
        pre = pre_verifications.get(feature, {})
        post = post_verifications.get(feature, {})

        pre_verdict = str(
            pre.get("verdict", "") or ""
        ).upper()
        post_verdict = str(
            post.get("verdict", "") or ""
        ).upper()

        if pre_verdict not in {"VERIFIED", "REFINE"}:
            continue
        if post_verdict not in {"VERIFIED", "REFINE"}:
            continue

        total += 1

        if (
            pre_verdict == "REFINE"
            and post_verdict == "VERIFIED"
        ):
            improved += 1
            change = "IMPROVED"
        elif (
            pre_verdict == "VERIFIED"
            and post_verdict == "REFINE"
        ):
            degraded += 1
            change = "DEGRADED"
        else:
            unchanged += 1
            change = "UNCHANGED"

        pre_grounding = pre.get(
            "grounding_scores",
            {},
        )
        if not isinstance(pre_grounding, dict):
            pre_grounding = {}

        post_grounding = post.get(
            "grounding_scores",
            {},
        )
        if not isinstance(post_grounding, dict):
            post_grounding = {}

        details.append(
            {
                "feature": feature,
                "pre_verdict": pre_verdict,
                "post_verdict": post_verdict,
                "change": change,
                "pre_bleu": pre_grounding.get(
                    "bleu",
                    0,
                ),
                "post_bleu": post_grounding.get(
                    "bleu",
                    0,
                ),
                "pre_rouge_l": pre_grounding.get(
                    "rouge_l",
                    0,
                ),
                "post_rouge_l": post_grounding.get(
                    "rouge_l",
                    0,
                ),
                "pre_grounding_level": pre.get(
                    "grounding_level",
                    pre_grounding.get(
                        "grounding_level",
                        "",
                    ),
                ),
                "post_grounding_level": post.get(
                    "grounding_level",
                    post_grounding.get(
                        "grounding_level",
                        "",
                    ),
                ),
            }
        )

    return {
        "total_compared": total,
        "improved": improved,
        "degraded": degraded,
        "unchanged": unchanged,
        "details": details,
    }
