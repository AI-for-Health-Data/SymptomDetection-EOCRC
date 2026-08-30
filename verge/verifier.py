from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional

from sage_common import (
    SYMPTOMS,
    generate_text,
    safe_json_loads,
)

_TOKEN_PATTERN = re.compile(r"\w+|\S")


def _tokenize(text: Any) -> List[str]:
    """Lower-case tokenization used by the lexical grounding metrics."""
    return _TOKEN_PATTERN.findall(str(text or "").lower())


def _normalize_text(text: Any) -> str:
    """Unicode/case/whitespace normalization for substring grounding."""
    value = unicodedata.normalize("NFKD", str(text or ""))
    value = value.lower().strip()
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    value = value.replace("\u2018", "'").replace("\u2019", "'")
    return re.sub(r"\s+", " ", value)


def exact_match(quote: str, note: str) -> bool:
    """Return True when the stripped quote is a verbatim substring."""
    if not quote or not note:
        return False
    return quote.strip() in note


def normalized_match(quote: str, note: str) -> bool:
    """Return True after unicode/case/whitespace normalization."""
    q = _normalize_text(quote)
    n = _normalize_text(note)
    return bool(q) and q in n


def token_subsequence_match(
    quote: str,
    note: str,
    min_tokens: int = 4,
) -> bool:
    """Return True when quote tokens occur contiguously in the note."""
    q_tokens = _tokenize(quote)
    n_tokens = _tokenize(note)

    if len(q_tokens) < min_tokens or len(q_tokens) > len(n_tokens):
        return False

    width = len(q_tokens)
    for start in range(len(n_tokens) - width + 1):
        if n_tokens[start : start + width] == q_tokens:
            return True
    return False


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 using an LCS implementation with O(len(hypothesis)) memory."""
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    previous = [0] * (len(hyp_tokens) + 1)

    for ref_token in ref_tokens:
        current = [0]
        for j, hyp_token in enumerate(hyp_tokens, start=1):
            if ref_token == hyp_token:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[j - 1]))
        previous = current

    lcs_len = previous[-1]
    precision = lcs_len / len(hyp_tokens)
    recall = lcs_len / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def _modified_ngram_precision(
    reference_tokens: List[str],
    hypothesis_tokens: List[str],
    n: int,
) -> float:
    """Modified n-gram precision used by BLEU, without brevity penalty."""
    if len(hypothesis_tokens) < n:
        return 0.0

    hyp_ngrams = Counter(
        zip(*[hypothesis_tokens[offset:] for offset in range(n)])
    )
    ref_ngrams = Counter(
        zip(*[reference_tokens[offset:] for offset in range(n)])
    )

    matches = sum(
        min(count, ref_ngrams[ngram])
        for ngram, count in hyp_ngrams.items()
    )
    total = sum(hyp_ngrams.values())

    return 0.0 if total == 0 else matches / total


def compute_bleu_no_bp(
    reference: str,
    hypothesis: str,
    max_n: int = 4,
) -> float:
    """BLEU geometric mean of modified precisions without brevity penalty."""
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    effective_max_n = min(max_n, len(hyp_tokens))
    log_precisions: List[float] = []

    for n in range(1, effective_max_n + 1):
        precision = _modified_ngram_precision(
            ref_tokens,
            hyp_tokens,
            n,
        )
        if precision == 0.0:
            return 0.0
        log_precisions.append(math.log(precision))

    return math.exp(sum(log_precisions) / len(log_precisions))


def compute_bertscore_single(
    reference: str,
    hypothesis: str,
) -> Optional[float]:
    """
    Compute BERTScore precision using roberta-large.

    Returns None when bert_score is unavailable or the metric cannot be
    computed. Other grounding metrics remain available in that case.
    """
    if not reference or not hypothesis:
        return None

    try:
        from bert_score import score as bert_score_fn

        precision, _, _ = bert_score_fn(
            [hypothesis],
            [reference],
            lang="en",
            model_type="roberta-large",
            rescale_with_baseline=False,
            verbose=False,
        )
        return float(precision[0])
    except Exception:
        return None


def compute_grounding_scores(
    evidence_quote: str,
    note_text: str,
    skip_bertscore: bool = False,
) -> Dict[str, Any]:
    """
    Compute hierarchical grounding metrics against the FULL note_text.

    A quote with <=5 non-whitespace characters is treated as unusable.
    """
    result: Dict[str, Any] = {
        "exact_match": False,
        "normalized_match": False,
        "token_subsequence": False,
        "rouge_l": 0.0,
        "bleu": 0.0,
        "bertscore": None,
        "grounding_level": "NOT_APPLICABLE",
    }

    quote = str(evidence_quote or "").strip()
    note = str(note_text or "")

    if len(quote) <= 5:
        return result

    result["exact_match"] = exact_match(quote, note)
    result["normalized_match"] = normalized_match(quote, note)
    result["token_subsequence"] = token_subsequence_match(quote, note)
    result["rouge_l"] = round(compute_rouge_l(note, quote), 4)
    result["bleu"] = round(compute_bleu_no_bp(note, quote), 4)

    if not skip_bertscore:
        bertscore = compute_bertscore_single(note, quote)
        if bertscore is not None:
            result["bertscore"] = round(bertscore, 4)

    # Deterministic grounding hierarchy.
    if (
        result["exact_match"]
        or result["normalized_match"]
        or result["token_subsequence"]
    ):
        level = "STRONG"
    elif (
        result["rouge_l"] >= 0.50
        or result["bleu"] >= 0.30
        or (
            result["bertscore"] is not None
            and result["bertscore"] >= 0.88
        )
    ):
        level = "MODERATE"
    elif (
        result["rouge_l"] >= 0.20
        or result["bleu"] >= 0.10
        or (
            result["bertscore"] is not None
            and result["bertscore"] >= 0.85
        )
    ):
        level = "WEAK"
    else:
        level = "NONE"

    result["grounding_level"] = level
    return result


VALIDATION_PROMPT_POSITIVE = """You are a clinical claim verifier.
Assess this POSITIVE extraction claim against the FULL clinical note.

CLINICAL NOTE:
{note_text}

CLAIM:
The extractor says "{feature}" is PRESENT.

EVIDENCE CITED:
"{evidence_quote}"

NOTE SECTION:
{note_section}

EXPERIENCER:
{experiencer}

Evaluate each applicable dimension independently.

1. SOURCE FAITHFULNESS
Does the evidence quote exist in, or accurately represent, information in the
clinical note? Paraphrasing and typo correction are acceptable only when the
clinical meaning is preserved.

2. CONTEXTUAL VALIDITY
Is the finding affirmed rather than negated, hypothetical, or merely planned?
Does it refer to the correct experiencer? Is its temporal interpretation
appropriate for the claim?

3. DIRECT SUPPORT
Does the note directly document the target finding itself? Do NOT infer the
finding solely from a diagnosis, medication, procedure, laboratory result,
test, risk factor, or other indirect information.

4. RELATION SPECIFICITY
For "Family history of colorectal cancer", does the evidence identify a
biological family relation? For all other findings, mark this not applicable.

5. COLORECTAL-CANCER SPECIFICITY
For "Family history of colorectal cancer", does the evidence specifically
identify colorectal, colon, rectal, or bowel cancer? A different cancer type,
unspecified cancer, Lynch syndrome alone, FAP alone, genetic testing, or
hereditary-risk discussion is insufficient. For all other findings, mark this
not applicable.

Return JSON only:
{{
  "source_faithfulness": {{
    "pass": true or false,
    "reason": "..."
  }},
  "contextual_validity": {{
    "pass": true or false,
    "reason": "..."
  }},
  "direct_support": {{
    "pass": true or false,
    "reason": "..."
  }},
  "relation_specificity": {{
    "pass": true or false,
    "reason": "...",
    "relation": "..." or "not_applicable"
  }},
  "cancer_specificity": {{
    "pass": true or false,
    "reason": "...",
    "cancer_type": "..." or "not_applicable"
  }}
}}"""


VALIDATION_PROMPT_NEGATIVE = """You are a clinical claim verifier.
Assess this NEGATIVE extraction claim against the FULL clinical note.

CLINICAL NOTE:
{note_text}

CLAIM:
The extractor says "{feature}" is NOT PRESENT / NOT DOCUMENTED.

DENIAL OR EVIDENCE CITED, IF ANY:
"{evidence_quote}"

Search the full note for affirmative evidence that "{feature}" is actually
documented and may have been missed by the extractor.

Consider:
- direct mentions using standard or non-standard clinical terminology;
- clinically equivalent descriptions of the target finding;
- documentation in any note section.

Do NOT count:
- medications, procedures, diagnoses, laboratory results, tests, or risk
  factors as indirect evidence of a symptom;
- negated or denied findings;
- hypothetical or planned findings;
- findings about the wrong experiencer;
- for family history of colorectal cancer, personal history of cancer,
  unspecified family cancer, a non-colorectal cancer, Lynch syndrome alone,
  FAP alone, genetic testing alone, or hereditary-risk discussion alone.

If affirmative evidence is found, return an EXACT quote from the clinical note.
Do not invent or paraphrase the recovered quote.

Return JSON only:
{{
  "missed_evidence_found": true or false,
  "evidence_quote": "<exact text from note>" or "",
  "evidence_section": "<note section>" or "",
  "source_faithfulness": {{
    "pass": true or false,
    "reason": "..."
  }},
  "contextual_validity": {{
    "pass": true or false,
    "reason": "..."
  }},
  "direct_support": {{
    "pass": true or false,
    "reason": "..."
  }},
  "relation_specificity": {{
    "pass": true or false,
    "reason": "...",
    "relation": "..." or "not_applicable"
  }},
  "cancer_specificity": {{
    "pass": true or false,
    "reason": "...",
    "cancer_type": "..." or "not_applicable"
  }}
}}"""


def _optional_bool(value: Any) -> Optional[bool]:
    """Parse booleans without converting missing/None values to False."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None

    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def _dimension_from_parsed(
    parsed: Dict[str, Any],
    key: str,
) -> Dict[str, Any]:
    """Extract one pass/reason dimension while preserving unresolved status."""
    value = parsed.get(key)

    if not isinstance(value, dict):
        return {"pass": None, "reason": ""}

    return {
        "pass": _optional_bool(value.get("pass")),
        "reason": str(value.get("reason", "") or ""),
    }


def _recovered_quote_is_grounded(
    quote: str,
    note_text: str,
) -> bool:
    """
    Negative recovery hard check.

    A recovered affirmative quote is accepted only when it is a verbatim or
    normalized substring of the full note.
    """
    quote = str(quote or "").strip()
    if len(quote) <= 5:
        return False

    return exact_match(quote, note_text) or normalized_match(quote, note_text)



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



def _find_structurally_valid_direct_recovery(
    feature: str,
    note_text: str,
    max_candidates: int = 3,
) -> Dict[str, Any]:
    from clinical_evidence_policy import (
        candidate_context_hard_failures,
        candidate_structural_hard_failures_v2,
        evaluate_positive_evidence,
        find_affirmed_direct_candidates,
    )

    full_note = str(note_text or "")

    candidates = find_affirmed_direct_candidates(
        feature=feature,
        note_text=full_note,
        max_candidates=max_candidates,
    )

    audit = []

    for index, candidate in enumerate(candidates, 1):
        quote = str(
            candidate.get("quote", "") or ""
        ).strip()

        if not quote:
            continue

        alias = str(
            candidate.get("alias", "")
            or candidate.get("matched_alias", "")
            or ""
        ).strip()

        if not alias:
            alias = _recovery_alias_for_quote(
                feature,
                quote,
            )

        assertion = str(
            candidate.get("assertion", "") or ""
        ).strip()

        section = str(
            candidate.get("section", "")
            or candidate.get("note_section", "")
            or ""
        ).strip()

        evidence_policy = evaluate_positive_evidence(
            feature=feature,
            evidence=quote,
            note_text=full_note,
        )

        failures = list(
            evidence_policy.get(
                "hard_failures",
                [],
            )
            or []
        )

        failures.extend(
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
                full_note,
            )
            or []
        )

        failures = sorted(set(failures))

        record = {
            "candidate_index": index,
            "quote": quote,
            "alias": alias,
            "assertion": assertion,
            "section": section,
            "hard_failures": failures,
            "accepted": not bool(failures),
        }

        audit.append(record)

        if not failures:
            return {
                "accepted": record,
                "audit": audit,
            }

    return {
        "accepted": None,
        "audit": audit,
    }


def validate_claim_with_llm(
    structured_claim: Dict[str, Any],
    note_text: str,
) -> Dict[str, Any]:
    """
    Run LLM validation against the FULL note_text.

    Positive and negative claims use different prompts but return one common
    validation schema.
    """
    feature = str(structured_claim.get("feature", ""))
    prediction = str(structured_claim.get("prediction", "")).strip()
    evidence_quote = str(structured_claim.get("evidence_quote", "") or "")
    note_section = str(structured_claim.get("note_section", "") or "")
    experiencer = str(structured_claim.get("experiencer", "") or "")
    full_note = str(note_text or "")

    result: Dict[str, Any] = {
        "source_faithfulness": {"pass": None, "reason": ""},
        "contextual_validity": {"pass": None, "reason": ""},
        "direct_support": {"pass": None, "reason": ""},
        "relation_specificity": {
            "pass": None,
            "reason": "",
            "relation": "not_applicable",
        },
        "cancer_specificity": {
            "pass": None,
            "reason": "",
            "cancer_type": "not_applicable",
        },
        "missed_evidence_found_raw": False,
        "missed_evidence_found": False,
        "recovered_quote": "",
        "recovered_section": "",
        "recovered_quote_grounded": False,
        "recovered_alias": "",
        "recovered_structural_failures": [],
        "recovered_structural_pass": False,
        "recovery_source": "",
        "direct_candidate_audit": [],
        "llm_response_parsed": False,
    }

    if prediction == "No":
        direct_recovery = _find_structurally_valid_direct_recovery(
            feature=feature,
            note_text=full_note,
            max_candidates=3,
        )

        result["direct_candidate_audit"] = direct_recovery.get(
            "audit",
            [],
        )

        accepted = direct_recovery.get("accepted")

        if isinstance(accepted, dict):
            recovered_quote = str(
                accepted.get("quote", "") or ""
            ).strip()

            recovered_alias = str(
                accepted.get("alias", "") or ""
            ).strip()

            recovered_section = str(
                accepted.get("section", "") or ""
            ).strip()

            result["missed_evidence_found_raw"] = True
            result["missed_evidence_found"] = True
            result["recovered_quote"] = recovered_quote
            result["recovered_section"] = recovered_section
            result["recovered_quote_grounded"] = True
            result["recovered_alias"] = recovered_alias
            result["recovered_structural_failures"] = []
            result["recovered_structural_pass"] = True
            result["recovery_source"] = (
                "deterministic_direct_candidate"
            )

            return result

    if prediction == "Yes":
        prompt = VALIDATION_PROMPT_POSITIVE.format(
            note_text=full_note,
            feature=feature,
            evidence_quote=evidence_quote or "(no evidence cited)",
            note_section=note_section or "unknown",
            experiencer=experiencer or "unknown",
        )
    elif prediction == "No":
        prompt = VALIDATION_PROMPT_NEGATIVE.format(
            note_text=full_note,
            feature=feature,
            evidence_quote=evidence_quote or "(no evidence cited)",
        )
    else:
        return result

    try:
        response = generate_text(prompt, max_new_tokens=384)
        parsed = safe_json_loads(response)
    except Exception:
        return result

    if not isinstance(parsed, dict):
        return result

    result["llm_response_parsed"] = True

    result["source_faithfulness"] = _dimension_from_parsed(
        parsed,
        "source_faithfulness",
    )
    result["contextual_validity"] = _dimension_from_parsed(
        parsed,
        "contextual_validity",
    )

    if "direct_support" in parsed:
        result["direct_support"] = _dimension_from_parsed(
            parsed,
            "direct_support",
        )
    else:
        result["direct_support"] = _dimension_from_parsed(
            parsed,
            "clinical_plausibility",
        )

    relation = parsed.get("relation_specificity")
    if isinstance(relation, dict):
        result["relation_specificity"] = {
            "pass": _optional_bool(relation.get("pass")),
            "reason": str(relation.get("reason", "") or ""),
            "relation": str(
                relation.get("relation", "not_applicable")
                or "not_applicable"
            ),
        }

    cancer = parsed.get("cancer_specificity")
    if isinstance(cancer, dict):
        result["cancer_specificity"] = {
            "pass": _optional_bool(cancer.get("pass")),
            "reason": str(cancer.get("reason", "") or ""),
            "cancer_type": str(
                cancer.get("cancer_type", "not_applicable")
                or "not_applicable"
            ),
        }

    if prediction == "No":
        raw_found = bool(parsed.get("missed_evidence_found", False))
        recovered_quote = str(parsed.get("evidence_quote", "") or "").strip()
        recovered_section = str(
            parsed.get("evidence_section", "") or ""
        ).strip()

        quote_grounded = (
            raw_found
            and _recovered_quote_is_grounded(
                recovered_quote,
                full_note,
            )
        )

        recovered_alias = ""
        structural_failures = []

        if quote_grounded:
            (
                recovered_alias,
                structural_failures,
            ) = _recovery_structural_failures(
                feature=feature,
                quote=recovered_quote,
                note_text=full_note,
            )

        structural_pass = (
            quote_grounded
            and not structural_failures
        )

        result["missed_evidence_found_raw"] = raw_found
        result["recovered_quote_grounded"] = quote_grounded
        result["recovered_alias"] = recovered_alias
        result["recovered_structural_failures"] = structural_failures
        result["recovered_structural_pass"] = structural_pass

        if structural_pass:
            result["recovery_source"] = "llm_fallback"
        elif quote_grounded:
            result["recovery_source"] = "llm_fallback_rejected"

        # Only a grounded candidate that also passes the frozen structural
        # clinical-evidence policy may route a negative claim to REFINE.
        result["recovered_quote"] = (
            recovered_quote if structural_pass else ""
        )
        result["recovered_section"] = (
            recovered_section if structural_pass else ""
        )
        result["missed_evidence_found"] = structural_pass

    return result


def _failed_or_unresolved(
    validation: Dict[str, Any],
    key: str,
) -> bool:
    """Applicable validation dimensions must explicitly pass."""
    value = validation.get(key, {})
    return not isinstance(value, dict) or value.get("pass") is not True


def compute_overall_verdict(
    structured_claim: Dict[str, Any],
    grounding: Dict[str, Any],
    validation: Dict[str, Any],
    label_probs: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Deterministically return VERIFIED or REFINE.

    label_probs is accepted for audit compatibility only and does NOT alter
    routing.
    """
    prediction = str(structured_claim.get("prediction", "")).strip()
    is_family_history = bool(structured_claim.get("is_family_history", False))
    issues: List[str] = []

    if prediction == "Yes":
        grounding_level = grounding.get(
            "grounding_level",
            "NOT_APPLICABLE",
        )

        if grounding_level == "NOT_APPLICABLE":
            issues.append(
                "MISSING_EVIDENCE: no usable evidence quote is available"
            )
        elif grounding_level == "NONE":
            issues.append(
                "UNGROUNDED_EVIDENCE: evidence does not sufficiently match "
                "the clinical note"
            )
        elif grounding_level == "WEAK":
            issues.append(
                "WEAK_GROUNDING: evidence only weakly matches the clinical note"
            )

        if _failed_or_unresolved(validation, "source_faithfulness"):
            reason = validation.get("source_faithfulness", {}).get(
                "reason",
                "",
            )
            issues.append(
                "SOURCE_UNFAITHFUL_OR_UNRESOLVED: "
                + (reason or "source faithfulness did not explicitly pass")
            )

        if _failed_or_unresolved(validation, "contextual_validity"):
            reason = validation.get("contextual_validity", {}).get(
                "reason",
                "",
            )
            issues.append(
                "CONTEXT_INVALID_OR_UNRESOLVED: "
                + (reason or "clinical context did not explicitly pass")
            )

        if _failed_or_unresolved(validation, "direct_support"):
            reason = validation.get("direct_support", {}).get(
                "reason",
                "",
            )
            issues.append(
                "DIRECT_SUPPORT_FAILED_OR_UNRESOLVED: "
                + (reason or "direct support did not explicitly pass")
            )

        if is_family_history:
            if _failed_or_unresolved(
                validation,
                "relation_specificity",
            ):
                reason = validation.get(
                    "relation_specificity",
                    {},
                ).get("reason", "")
                issues.append(
                    "RELATION_SPECIFICITY_FAILED_OR_UNRESOLVED: "
                    + (
                        reason
                        or "biological family relation did not explicitly pass"
                    )
                )

            if _failed_or_unresolved(
                validation,
                "cancer_specificity",
            ):
                reason = validation.get(
                    "cancer_specificity",
                    {},
                ).get("reason", "")
                issues.append(
                    "CRC_SPECIFICITY_FAILED_OR_UNRESOLVED: "
                    + (
                        reason
                        or "colorectal-cancer specificity did not explicitly pass"
                    )
                )

        if issues:
            verdict = "REFINE"
            reasoning = "; ".join(issues)
        else:
            verdict = "VERIFIED"
            reasoning = (
                "Grounding is STRONG or MODERATE and all applicable "
                "clinical-validation criteria explicitly passed"
            )

    elif prediction == "No":
        if validation.get("missed_evidence_found") is True:
            recovered_quote = validation.get("recovered_quote", "")
            recovered_section = validation.get(
                "recovered_section",
                "",
            )
            issues.append(
                "MISSED_GROUNDED_AFFIRMATIVE_EVIDENCE: "
                f"{recovered_section or 'note'}: "
                f"'{str(recovered_quote)[:160]}'"
            )
            verdict = "REFINE"
            reasoning = (
                "Grounded affirmative evidence was recovered for a negative "
                "claim"
            )
        else:
            verdict = "VERIFIED"
            reasoning = (
                "No grounded affirmative evidence was recovered for the "
                "negative claim"
            )


    else:
        issues.append(
            f"INVALID_PREDICTION: expected Yes/No, received {prediction!r}"
        )
        verdict = "REFINE"
        reasoning = "Invalid prediction requires correction"

    return {
        "verdict": verdict,
        "issues_found": issues,
        "verdict_reasoning": reasoning,
        "label_probs_audit_only": label_probs or {},
    }

def verify_claim(
    structured_claim: Dict[str, Any],
    note_text: str,
    skip_bertscore: bool = False,
    label_probs: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Verify one structured claim against the FULL clinical note.

    Returns only VERIFIED or REFINE.
    """
    feature = str(structured_claim.get("feature", ""))
    prediction = str(structured_claim.get("prediction", "")).strip()
    evidence_quote = str(
        structured_claim.get("evidence_quote", "") or ""
    )

    grounding = compute_grounding_scores(
        evidence_quote=evidence_quote,
        note_text=note_text,
        skip_bertscore=skip_bertscore,
    )

    validation = validate_claim_with_llm(
        structured_claim=structured_claim,
        note_text=note_text,
    )

    verdict_info = compute_overall_verdict(
        structured_claim=structured_claim,
        grounding=grounding,
        validation=validation,
        label_probs=label_probs,
    )

    return {
        "feature": feature,
        "prediction": prediction,
        "grounding_scores": grounding,
        "grounding_level": grounding["grounding_level"],
        "validation_results": validation,
        "label_probs": label_probs or {},
        "issues_found": verdict_info["issues_found"],
        "verdict": verdict_info["verdict"],
        "verdict_reasoning": verdict_info["verdict_reasoning"],
    }


def verify_all_claims(
    structured_claims: Dict[str, Dict[str, Any]],
    note_text: str,
    skip_bertscore: bool = False,
    all_label_probs: Optional[
        Dict[str, Dict[str, float]]
    ] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Verify all seven claims for one note.

    Operational exceptions are represented as REFINE, never ESCALATE.
    """
    results: Dict[str, Dict[str, Any]] = {}

    for feature in SYMPTOMS:
        claim = structured_claims.get(feature)
        if claim is None:
            continue

        label_probs = (all_label_probs or {}).get(feature)

        try:
            results[feature] = verify_claim(
                structured_claim=claim,
                note_text=note_text,
                skip_bertscore=skip_bertscore,
                label_probs=label_probs,
            )
        except Exception as error:
            results[feature] = {
                "feature": feature,
                "prediction": claim.get("prediction", ""),
                "grounding_scores": {},
                "grounding_level": "ERROR",
                "validation_results": {},
                "label_probs": label_probs or {},
                "issues_found": [
                    f"VERIFICATION_ERROR: {str(error)}"
                ],
                "verdict": "REFINE",
                "verdict_reasoning": (
                    "Verification failed operationally and requires correction"
                ),
            }

    return results
