from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Dict, Optional

from sage_common import (
    SYMPTOMS,
    generate_text,
    load_llm,
    safe_json_loads,
)


VALID_ASSERTIONS = {
    "affirmed",
    "negated",
    "hypothetical",
    "planned",
    "historical",
    "absent",
    "unknown",
    "",
}

VALID_EXPERIENCERS = {
    "patient",
    "family_member",
    "other",
    "unknown",
    "",
}

CRC_TERMS = (
    "colorectal cancer",
    "colon cancer",
    "rectal cancer",
    "bowel cancer",
    "colorectal carcinoma",
    "colon carcinoma",
    "rectal carcinoma",
    "bowel carcinoma",
    "crc",
)

GENERIC_RELATION_TERMS = (
    "mother",
    "father",
    "parent",
    "sister",
    "brother",
    "sibling",
    "grandmother",
    "grandfather",
    "aunt",
    "uncle",
    "son",
    "daughter",
    "relative",
    "family member",
    "first-degree relative",
    "first degree relative",
    "maternal",
    "paternal",
)


def _usable_quote(text: Any) -> bool:
    """Match Agent 3: >5 non-whitespace characters are required."""
    value = str(text or "")
    return len(re.sub(r"\s+", "", value)) > 5


def _normalize_text(text: Any) -> str:
    """Normalize unicode, case, and whitespace for evidence grounding."""
    value = unicodedata.normalize("NFKD", str(text or ""))
    value = value.lower().strip()
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    value = value.replace("\u2018", "'").replace("\u2019", "'")
    return re.sub(r"\s+", " ", value)


def _evidence_grounded_in_note(
    evidence: str,
    note_text: str,
) -> bool:
    """
    Accept evidence only when it is an exact or normalized substring
    of the FULL clinical note.
    """
    quote = str(evidence or "").strip()
    note = str(note_text or "")

    if not _usable_quote(quote) or not note:
        return False

    if quote in note:
        return True

    normalized_quote = _normalize_text(quote)
    normalized_note = _normalize_text(note)

    return bool(normalized_quote) and normalized_quote in normalized_note


def _is_family_history(feature: str) -> bool:
    return feature == "Family history of colorectal cancer"


def _has_family_relation(relation: str) -> bool:
    value = _normalize_text(relation)
    if not value:
        return False
    return any(term in value for term in GENERIC_RELATION_TERMS)


def _has_crc_type(cancer_type: str) -> bool:
    value = _normalize_text(cancer_type)
    if not value:
        return False
    return any(term in value for term in CRC_TERMS)


def _clean_enum(
    value: Any,
    allowed: set[str],
    default: str = "",
) -> str:
    cleaned = str(value or "").strip().lower()
    return cleaned if cleaned in allowed else default



def _positive_semantic_failures(
    feature: str,
    evidence: str,
    experiencer: str,
    assertion: str,
    relation_found: str,
    cancer_type_found: str,
) -> list[str]:
    failures: list[str] = []

    if assertion not in {"affirmed", "historical"}:
        failures.append(
            "positive assertion must be affirmed or historical"
        )

    if _is_family_history(feature):
        if experiencer != "family_member":
            failures.append(
                "family-history positive must have experiencer=family_member"
            )

        if not _has_family_relation(relation_found):
            failures.append(
                "family-history positive lacks a recognized biological relation"
            )

        if not _has_crc_type(cancer_type_found):
            failures.append(
                "family-history positive lacks colorectal-cancer specificity"
            )

        # The LLM-generated metadata itself is not enough. The exact grounded
        # evidence must also contain recognizable family-relation and CRC terms.
        if not _has_family_relation(evidence):
            failures.append(
                "grounded evidence does not itself support a family relation"
            )

        if not _has_crc_type(evidence):
            failures.append(
                "grounded evidence does not itself support colorectal cancer"
            )

    else:
        if experiencer != "patient":
            failures.append(
                "patient symptom positive must have experiencer=patient"
            )

    return failures



def _recovery_alias_for_quote(
    feature: str,
    quote: str,
) -> str:
    """Infer the frozen study alias represented in a grounded candidate quote."""
    from sage_common import DIRECT_ALIASES

    def norm(x: str) -> str:
        return " ".join(str(x or "").strip().lower().split())

    feature_n = norm(feature)
    quote_n = norm(quote)

    aliases = []

    for key, values in DIRECT_ALIASES.items():
        if norm(key) == feature_n:
            aliases = list(values or [])
            break

    hits = []

    for alias in aliases:
        alias_n = norm(alias)
        if alias_n and alias_n in quote_n:
            hits.append((len(alias_n), str(alias)))

    if hits:
        hits.sort(reverse=True)
        return hits[0][1]

    if feature_n == "family history of colorectal cancer":
        return "__FAMILY_RELATION_PLUS_CRC__"

    return str(feature or "").strip()


def _recovery_structural_failures(
    feature: str,
    quote: str,
    note_text: str,
):
    """Apply the frozen V2 clinical-evidence policy to one grounded candidate."""
    from clinical_evidence_policy import (
        candidate_context_hard_failures,
        candidate_structural_hard_failures_v2,
    )

    alias = _recovery_alias_for_quote(
        feature,
        quote,
    )

    failures = list(
        candidate_context_hard_failures(
            feature,
            alias,
            quote,
        )
        or []
    )

    failures.extend(
        candidate_structural_hard_failures_v2(
            feature,
            alias,
            quote,
            note_text,
        )
        or []
    )

    return alias, sorted(set(failures))


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
    field_name: str = "corrected_prediction",
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
            return _empty_probability_audit(
                "empty aligned generation prefix"
            )

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


REFINE_PROMPT = """You are a clinical extraction refiner.

Agent 3 found one or more problems with the current extraction claim.
Re-examine the FULL clinical note and produce a corrected extraction.

FULL CLINICAL NOTE:
{note_text}

CURRENT CLAIM:
  Target finding: {feature}
  Current prediction: {prediction}
  Current evidence: "{evidence_quote}"
  Current note section: {note_section}
  Current experiencer: {experiencer}
  Current assertion status: {assertion_status}
  Current family relation: {relation_found}
  Current cancer type: {cancer_type_found}

VERIFIER PROBLEMS:
{issues_list}

YOUR TASK:
Correct the claim using ONLY information explicitly supported by the clinical
note.

RULES:
1. Re-read the full note, focusing on the verifier problems above.

2. If the current claim is positive but the cited evidence is unfaithful,
   negated, hypothetical, planned, about the wrong experiencer, or only
   indirectly inferred, search the note for valid direct evidence of
   "{feature}".

3. If the current claim is negative and Agent 3 recovered possible missed
   affirmative evidence, verify that the recovered evidence is genuinely
   present in the note, affirmed, about the correct experiencer, and directly
   documents the target finding.

4. Do NOT infer a finding solely from a diagnosis, medication, procedure,
   laboratory result, test, risk factor, or other indirect clinical information.

5. If valid direct evidence cannot be identified, return
   corrected_prediction="No".

6. If valid direct evidence is identified, return
   corrected_prediction="Yes" and copy an EXACT supporting quote from the
   clinical note.

7. For "Family history of colorectal cancer", corrected_prediction="Yes"
   requires BOTH:
      a. a biological family relation; and
      b. colorectal, colon, rectal, or bowel cancer.
   Personal cancer history, unspecified family cancer, another cancer type,
   Lynch syndrome alone, FAP alone, genetic testing alone, or hereditary-risk
   discussion alone is insufficient.

8. Preserve the correct metadata for the corrected claim:
      - note section
      - experiencer
      - assertion status
      - family relation, when applicable
      - cancer type, when applicable

9. Be conservative. Do not return Yes unless the evidence is explicitly
   supported by the note.

Return JSON only:
{{
  "corrected_prediction": "Yes" or "No",
  "corrected_evidence": "<exact text from note>" or "",
  "corrected_section": "<note section>" or "none",
  "corrected_experiencer": "patient" or "family_member" or "other" or "",
  "corrected_assertion": "affirmed" or "negated" or "hypothetical" or
                         "planned" or "historical" or "absent" or "unknown",
  "correction_reasoning": "<brief explanation of the correction>",
  "relation_found": "<biological family relation if applicable>" or "",
  "cancer_type_found": "<cancer type if applicable>" or ""
}}"""


POSITIVE_RESCUE_PROMPT = """You are performing a recall-protective second review
before an existing positive clinical finding is allowed to change to No.

FULL CLINICAL NOTE:
{note_text}

TARGET FINDING: {feature}
CURRENT POSITIVE EVIDENCE THAT FAILED REVIEW:
"{failed_evidence}"

WHY THE PROPOSED POSITIVE FAILED:
{failure_reason}

TASK:
Search the ENTIRE note for a DIFFERENT, direct, affirmative mention of the same
target finding. This is a rescue search, not a request to force a positive label.
Return rescue_found=true only when the note contains direct evidence that would
support a Yes claim for the correct experiencer.

RULES:
1. The rescue evidence must be copied EXACTLY from the note.
2. Do not use negated, hypothetical, planned, conditional, or merely possible
   mentions as affirmative evidence.
3. Do not infer the target from diagnoses, medications, procedures, laboratory
   results, tests, risk factors, or treatment indications.
4. For patient symptoms/findings, the experiencer must be the patient.
5. For family history of colorectal cancer, rescue_found=true requires BOTH a
   biological family relation and colorectal/colon/rectal/bowel cancer in the
   supporting evidence. Personal cancer history, unspecified family cancer,
   another cancer type, Lynch syndrome alone, FAP alone, or genetic-risk
   discussion alone is insufficient.
6. If no valid alternative evidence exists, return rescue_found=false.

Return JSON only:
{{
  "rescue_found": true or false,
  "rescue_evidence": "<exact quote from note>" or "",
  "rescue_section": "<note section>" or "none",
  "rescue_experiencer": "patient" or "family_member" or "other" or "",
  "rescue_assertion": "affirmed" or "historical" or "negated" or
                      "hypothetical" or "planned" or "absent" or "unknown",
  "relation_found": "<biological family relation if applicable>" or "",
  "cancer_type_found": "<cancer type if applicable>" or "",
  "rescue_reasoning": "<brief explanation>"
}}"""


def _attempt_positive_rescue(
    feature: str,
    full_note: str,
    failed_evidence: str,
    failure_reason: str,
) -> Optional[Dict[str, Any]]:
    prompt = POSITIVE_RESCUE_PROMPT.format(
        note_text=str(full_note or ""),
        feature=str(feature or ""),
        failed_evidence=str(failed_evidence or "") or "(none)",
        failure_reason=str(failure_reason or "") or "(not specified)",
    )

    try:
        response = generate_text(prompt, max_new_tokens=384)
        parsed = safe_json_loads(response)
    except Exception:
        return None

    if not isinstance(parsed, dict):
        return None

    rescue_flag = parsed.get("rescue_found", False)
    if isinstance(rescue_flag, str):
        rescue_flag = rescue_flag.strip().lower() in {"true", "yes", "1"}
    if not bool(rescue_flag):
        return None

    rescue_evidence = str(parsed.get("rescue_evidence", "") or "").strip()
    if not _evidence_grounded_in_note(rescue_evidence, full_note):
        return None

    rescue_alias, structural_failures = _recovery_structural_failures(
        feature=feature,
        quote=rescue_evidence,
        note_text=full_note,
    )

    if structural_failures:
        return None

    rescue_section = str(parsed.get("rescue_section", "") or "").strip().lower()
    rescue_experiencer = _clean_enum(
        parsed.get("rescue_experiencer", ""),
        VALID_EXPERIENCERS,
        default="",
    )
    rescue_assertion = _clean_enum(
        parsed.get("rescue_assertion", ""),
        VALID_ASSERTIONS,
        default="unknown",
    )
    relation_found = str(parsed.get("relation_found", "") or "").strip().lower()
    cancer_type_found = str(parsed.get("cancer_type_found", "") or "").strip().lower()

    semantic_failures = _positive_semantic_failures(
        feature=feature,
        evidence=rescue_evidence,
        experiencer=rescue_experiencer,
        assertion=rescue_assertion,
        relation_found=relation_found,
        cancer_type_found=cancer_type_found,
    )
    if semantic_failures:
        return None

    return {
        "evidence": rescue_evidence,
        "section": rescue_section or "none",
        "experiencer": rescue_experiencer,
        "assertion": rescue_assertion,
        "relation_found": relation_found,
        "cancer_type_found": cancer_type_found,
        "reasoning": str(parsed.get("rescue_reasoning", "") or "").strip(),
    }


def _base_result(
    structured_claim: Dict[str, Any],
    verification: Dict[str, Any],
    label_probs: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Create a complete before/after audit record."""
    prediction = str(structured_claim.get("prediction", "")).strip()
    evidence = str(structured_claim.get("evidence_quote", "") or "").strip()

    return {
        "feature": str(structured_claim.get("feature", "")),
        "original_prediction": prediction,
        "original_evidence": evidence,
        "original_section": str(
            structured_claim.get("note_section", "") or ""
        ),
        "original_experiencer": str(
            structured_claim.get("experiencer", "") or ""
        ),
        "original_assertion": str(
            structured_claim.get("assertion_status", "") or ""
        ),
        "original_relation_found": str(
            structured_claim.get("relation_found", "") or ""
        ),
        "original_cancer_type_found": str(
            structured_claim.get("cancer_type_found", "") or ""
        ),
        "verifier_verdict": str(
            verification.get("verdict", "") or ""
        ),
        "verifier_issues": list(
            verification.get("issues_found", []) or []
        ),
        "refined": False,
        "corrected_prediction": prediction,
        "corrected_evidence": evidence,
        "corrected_section": str(
            structured_claim.get("note_section", "") or ""
        ),
        "corrected_experiencer": str(
            structured_claim.get("experiencer", "") or ""
        ),
        "corrected_assertion": str(
            structured_claim.get("assertion_status", "") or ""
        ),
        "correction_reasoning": "",
        "relation_found": str(
            structured_claim.get("relation_found", "") or ""
        ),
        "cancer_type_found": str(
            structured_claim.get("cancer_type_found", "") or ""
        ),
        "change_type": "UNCHANGED",
        "evidence_grounded": None,
        "correction_valid": True,
        "operational_error": False,
        "label_probs": label_probs or {},
        "refiner_generated_prediction": "",
        "refiner_label_probs": {},
        "positive_rescue_attempted": False,
        "positive_rescue_found": False,
        "positive_rescue_evidence": "",
        "positive_rescue_reasoning": "",
    }


def _mark_error(
    result: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    result["refined"] = False
    result["change_type"] = "REFINEMENT_ERROR"
    result["correction_valid"] = False
    result["operational_error"] = True
    result["correction_reasoning"] = reason
    return result


def _determine_change_type(
    original_prediction: str,
    corrected_prediction: str,
    original_evidence: str,
    corrected_evidence: str,
    metadata_changed: bool,
) -> str:
    """Classify the refinement outcome for auditing."""
    if corrected_prediction != original_prediction:
        if original_prediction == "Yes" and corrected_prediction == "No":
            return "REJECTED"
        if original_prediction == "No" and corrected_prediction == "Yes":
            return "RECOVERED"
        return "LABEL_CHANGED"

    if corrected_evidence != original_evidence:
        return "EVIDENCE_CORRECTED"

    if metadata_changed:
        return "METADATA_CORRECTED"

    return "CONFIRMED"


def refine_claim(
    structured_claim: Dict[str, Any],
    verification: Dict[str, Any],
    note_text: str,
    label_probs: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    result = _base_result(
        structured_claim=structured_claim,
        verification=verification,
        label_probs=label_probs,
    )

    feature = str(structured_claim.get("feature", ""))
    prediction = str(structured_claim.get("prediction", "")).strip()
    evidence = str(
        structured_claim.get("evidence_quote", "") or ""
    ).strip()
    verdict = str(verification.get("verdict", "") or "").strip()
    issues = list(verification.get("issues_found", []) or [])
    full_note = str(note_text or "")

    if verdict == "VERIFIED":
        result["correction_reasoning"] = (
            "No refinement performed because Agent 3 verified the claim."
        )
        return result

    if verdict != "REFINE":
        return _mark_error(
            result,
            f"Unexpected Agent 3 verdict {verdict!r}; expected VERIFIED or REFINE.",
        )

    if prediction not in {"Yes", "No"}:
        return _mark_error(
            result,
            f"Invalid current prediction {prediction!r}; expected Yes or No.",
        )

    if not full_note.strip():
        return _mark_error(
            result,
            "Clinical note is empty; refinement cannot be performed.",
        )

    issues_text = "\n".join(
        f"  - {issue}" for issue in issues
    )
    if not issues_text:
        issues_text = "  - No specific verifier issue text was provided."

    prompt = REFINE_PROMPT.format(
        note_text=full_note,
        feature=feature,
        prediction=prediction,
        evidence_quote=evidence or "(no evidence)",
        note_section=str(
            structured_claim.get("note_section", "") or "unknown"
        ),
        experiencer=str(
            structured_claim.get("experiencer", "") or "unknown"
        ),
        assertion_status=str(
            structured_claim.get("assertion_status", "") or "unknown"
        ),
        relation_found=str(
            structured_claim.get("relation_found", "") or ""
        ),
        cancer_type_found=str(
            structured_claim.get("cancer_type_found", "") or ""
        ),
        issues_list=issues_text,
    )

    try:
        response = generate_text(
            prompt,
            max_new_tokens=384,
        )
        parsed = safe_json_loads(response)
    except Exception as error:
        return _mark_error(
            result,
            f"Refinement generation failed: {str(error)}",
        )

    if not isinstance(parsed, dict):
        return _mark_error(
            result,
            "Refinement response could not be parsed as a JSON object.",
        )

    corrected_prediction = str(
        parsed.get("corrected_prediction", "")
    ).strip()

    if corrected_prediction not in {"Yes", "No"}:
        return _mark_error(
            result,
            "Refinement returned an invalid corrected_prediction; "
            "expected Yes or No.",
        )

    refiner_generated_prediction = corrected_prediction

    refiner_label_probs = _binary_choice_probability_audit(
        prompt=prompt,
        response=response,
        chosen_label=refiner_generated_prediction,
        field_name="corrected_prediction",
    )

    corrected_evidence = str(
        parsed.get("corrected_evidence", "") or ""
    ).strip()

    corrected_section = str(
        parsed.get("corrected_section", "") or ""
    ).strip().lower()

    corrected_experiencer = _clean_enum(
        parsed.get("corrected_experiencer", ""),
        VALID_EXPERIENCERS,
        default="",
    )

    corrected_assertion = _clean_enum(
        parsed.get("corrected_assertion", ""),
        VALID_ASSERTIONS,
        default="unknown",
    )

    correction_reasoning = str(
        parsed.get("correction_reasoning", "") or ""
    ).strip()

    relation_found = str(
        parsed.get("relation_found", "") or ""
    ).strip().lower()

    cancer_type_found = str(
        parsed.get("cancer_type_found", "") or ""
    ).strip().lower()

    evidence_grounded = _evidence_grounded_in_note(
        corrected_evidence,
        full_note,
    )

    if corrected_prediction == "Yes" and not evidence_grounded:
        corrected_prediction = "No"
        corrected_evidence = ""
        corrected_section = corrected_section or "none"
        corrected_experiencer = corrected_experiencer or ""
        corrected_assertion = "absent"
        relation_found = ""
        cancer_type_found = ""

        deterministic_reason = (
            "The proposed positive correction was rejected because its "
            "supporting evidence did not pass exact or normalized grounding "
            "against the full clinical note."
        )
        correction_reasoning = (
            f"{correction_reasoning} {deterministic_reason}"
        ).strip()

    if (
        corrected_prediction == "Yes"
        and _is_family_history(feature)
    ):
        valid_relation = _has_family_relation(relation_found)
        valid_crc = _has_crc_type(cancer_type_found)

        if not (valid_relation and valid_crc):
            corrected_prediction = "No"
            corrected_evidence = ""
            corrected_section = corrected_section or "none"
            corrected_experiencer = ""
            corrected_assertion = "absent"
            relation_found = ""
            cancer_type_found = ""

            missing_parts = []
            if not valid_relation:
                missing_parts.append("biological family relation")
            if not valid_crc:
                missing_parts.append("colorectal-cancer specificity")

            deterministic_reason = (
                "The proposed family-history positive correction was rejected "
                "because it lacked "
                + " and ".join(missing_parts)
                + "."
            )
            correction_reasoning = (
                f"{correction_reasoning} {deterministic_reason}"
            ).strip()

 
    if corrected_prediction == "Yes":
        semantic_failures = _positive_semantic_failures(
            feature=feature,
            evidence=corrected_evidence,
            experiencer=corrected_experiencer,
            assertion=corrected_assertion,
            relation_found=relation_found,
            cancer_type_found=cancer_type_found,
        )

        if semantic_failures:
            corrected_prediction = "No"
            corrected_evidence = ""
            corrected_section = corrected_section or "none"
            corrected_experiencer = ""
            corrected_assertion = "absent"
            relation_found = ""
            cancer_type_found = ""

            deterministic_reason = (
                "The proposed positive correction was rejected by "
                "deterministic semantic safeguards: "
                + "; ".join(semantic_failures)
                + "."
            )
            correction_reasoning = (
                f"{correction_reasoning} {deterministic_reason}"
            ).strip()

    if (
        prediction == "Yes"
        and corrected_prediction == "No"
        and not bool(structured_claim.get("positive_rescue_used", False))
    ):
        result["positive_rescue_attempted"] = True
        rescue = _attempt_positive_rescue(
            feature=feature,
            full_note=full_note,
            failed_evidence=evidence,
            failure_reason=correction_reasoning,
        )

        if rescue is not None:
            corrected_prediction = "Yes"
            corrected_evidence = rescue["evidence"]
            corrected_section = rescue["section"]
            corrected_experiencer = rescue["experiencer"]
            corrected_assertion = rescue["assertion"]
            relation_found = rescue["relation_found"]
            cancer_type_found = rescue["cancer_type_found"]

            result["positive_rescue_found"] = True
            result["positive_rescue_evidence"] = corrected_evidence
            result["positive_rescue_reasoning"] = rescue["reasoning"]

            correction_reasoning = (
                f"{correction_reasoning} Recall-protective rescue found "
                f"alternative direct evidence: {rescue['reasoning']}"
            ).strip()

    if corrected_prediction == "No":
        if corrected_evidence:
            no_evidence_grounded = _evidence_grounded_in_note(
                corrected_evidence,
                full_note,
            )
            if not no_evidence_grounded:
                corrected_evidence = ""
        corrected_section = corrected_section or "none"

        if corrected_assertion not in {
            "negated",
            "absent",
            "historical",
            "unknown",
        }:
            corrected_assertion = "absent"

    final_evidence_grounded = (
        _evidence_grounded_in_note(
            corrected_evidence,
            full_note,
        )
        if corrected_evidence
        else None
    )

    original_section = str(
        structured_claim.get("note_section", "") or ""
    ).strip().lower()
    original_experiencer = str(
        structured_claim.get("experiencer", "") or ""
    ).strip().lower()
    original_assertion = str(
        structured_claim.get("assertion_status", "") or ""
    ).strip().lower()
    original_relation = str(
        structured_claim.get("relation_found", "") or ""
    ).strip().lower()
    original_cancer_type = str(
        structured_claim.get("cancer_type_found", "") or ""
    ).strip().lower()

    metadata_changed = any(
        [
            corrected_section != original_section,
            corrected_experiencer != original_experiencer,
            corrected_assertion != original_assertion,
            relation_found != original_relation,
            cancer_type_found != original_cancer_type,
        ]
    )

    result.update(
        {
            "refined": True,
            "corrected_prediction": corrected_prediction,
            "corrected_evidence": corrected_evidence,
            "corrected_section": corrected_section,
            "corrected_experiencer": corrected_experiencer,
            "corrected_assertion": corrected_assertion,
            "correction_reasoning": correction_reasoning,
            "relation_found": relation_found,
            "cancer_type_found": cancer_type_found,
            "refiner_generated_prediction": refiner_generated_prediction,
            "refiner_label_probs": refiner_label_probs,
            "evidence_grounded": final_evidence_grounded,
            "correction_valid": True,
            "operational_error": False,
        }
    )

    result["change_type"] = _determine_change_type(
        original_prediction=prediction,
        corrected_prediction=corrected_prediction,
        original_evidence=evidence,
        corrected_evidence=corrected_evidence,
        metadata_changed=metadata_changed,
    )

    return result


def refine_all_claims(
    structured_claims: Dict[str, Dict[str, Any]],
    verifications: Dict[str, Dict[str, Any]],
    note_text: str,
    all_label_probs: Optional[
        Dict[str, Dict[str, float]]
    ] = None,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}

    for feature in SYMPTOMS:
        claim = structured_claims.get(feature)
        verification = verifications.get(feature)

        if claim is None or verification is None:
            continue

        label_probs = (all_label_probs or {}).get(feature)

        try:
            results[feature] = refine_claim(
                structured_claim=claim,
                verification=verification,
                note_text=note_text,
                label_probs=label_probs,
            )
        except Exception as error:
            fallback = _base_result(
                structured_claim=claim,
                verification=verification,
                label_probs=label_probs,
            )
            results[feature] = _mark_error(
                fallback,
                f"Unhandled refinement error: {str(error)}",
            )

    return results
