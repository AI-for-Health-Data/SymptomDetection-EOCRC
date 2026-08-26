from __future__ import annotations

import json
from typing import Any, Dict

from sage_common import (
    SYMPTOMS,
    generate_text,
    safe_json_loads,
)


UNUSABLE_INFERENCE_VALUES = {
    "",
    "n/a",
    "na",
    "none",
    "no inference",
    "not reported",
    "not mentioned",
    "not documented",
    "not documented in the note",
    "unknown",
}


STRUCTURER_PROMPT = """You are VERGE Agent 2, the Claim Composer.

Your task is representation standardization only.

You do NOT have the clinical note.
You must NOT reconsider or change the extractor's binary prediction.
You must NOT invent new patient evidence.
You must NOT add clinical facts that are absent from the supplied inference.

The extractor evidence below is the only patient-specific text available
to you. The evidence_quote in your response MUST copy that inference exactly.

Return these fields:
- evidence_quote
- note_section
- experiencer
- assertion_status
- relation_found
- cancer_type_found

Rules:
- evidence_quote: copy the supplied inference EXACTLY.
- note_section: HPI, ROS, Assessment, Plan, PE, PMH, FH, CC,
  Medications, Diagnosis, Impression, Other, Unknown, or none.
- experiencer: patient, family_member, other, unknown, or empty.
- assertion_status: affirmed, negated, hypothetical, planned,
  historical, absent, unknown, or empty.
- relation_found: family relation explicitly mentioned in the supplied
  inference; otherwise empty.
- cancer_type_found: cancer type explicitly mentioned in the supplied
  inference; otherwise empty.
- Do not infer a family relation or cancer type that is not explicitly
  present in the supplied inference.
- Return one flat JSON object only. No markdown or explanation.

TARGET SYMPTOM: {feature}
FIXED EXTRACTOR PREDICTION: {prediction}
FIXED EXTRACTOR CONFIDENCE: {confidence}
EXTRACTOR INFERENCE:
{inference_json}
"""


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _usable_inference(value: Any) -> bool:
    text = _clean(value)

    if len(text) <= 5:
        return False

    low = text.lower()

    if low in UNUSABLE_INFERENCE_VALUES:
        return False

    if (
        low.startswith("not documented")
        or low.startswith("not mentioned")
        or low.startswith("no documentation")
    ):
        return False

    return True


def _coerce_confidence(value: Any) -> int:
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Agent 1 confidence is not numeric: {value!r}"
        )

    if not x.is_integer() or not 1 <= x <= 5:
        raise ValueError(
            f"Agent 1 confidence must be integer 1-5: {value!r}"
        )

    return int(x)


def _prefer_agent1(
    agent1_value: Any,
    llm_value: Any,
    default: str = "",
) -> str:
    """
    Preserve Agent 1 metadata whenever present.
    Use Agent 2 interpretation only to fill a missing Agent 1 field.
    """
    a1 = _clean(agent1_value)

    if a1:
        return a1

    llm = _clean(llm_value)

    if llm:
        return llm

    return default


def _build_structured_claim(
    *,
    feature: str,
    prediction: str,
    confidence: int,
    inference: str,
    evidence_quote: str,
    note_section: str,
    experiencer: str,
    assertion_status: str,
    relation_found: str,
    cancer_type_found: str,
    structurer_source: str,
    structurer_operational_error: str = "",
) -> Dict[str, Any]:

    return {
        "feature": feature,
        "prediction": prediction,
        "confidence": confidence,
        "original_inference": inference,
        "evidence_quote": _clean(evidence_quote),
        "note_section": _clean(note_section).lower(),
        "experiencer": _clean(experiencer).lower(),
        "assertion_status": _clean(assertion_status).lower(),
        "relation_found": _clean(relation_found).lower(),
        "cancer_type_found": _clean(cancer_type_found).lower(),
        "has_evidence": bool(
            _clean(evidence_quote)
            and len(_clean(evidence_quote)) > 5
        ),
        "is_family_history": (
            feature == "Family history of colorectal cancer"
        ),
        "structurer_source": structurer_source,
        "structurer_operational_error": (
            _clean(structurer_operational_error)
        ),
    }


def structure_single_claim(
    feature: str,
    prediction: str,
    confidence: Any,
    inference: str,
    note_section: str = "",
    experiencer: str = "",
    assertion_status: str = "",
) -> Dict[str, Any]:

    prediction = _clean(prediction)

    # Never silently change an Agent 1 label.
    if prediction not in {"Yes", "No"}:
        raise ValueError(
            f"Invalid Agent 1 label for {feature}: {prediction!r}"
        )

    confidence = _coerce_confidence(confidence)
    inference = _clean(inference)

    usable = _usable_inference(inference)

    evidence_quote = inference if usable else ""


    if not usable:

        return _build_structured_claim(
            feature=feature,
            prediction=prediction,
            confidence=confidence,
            inference=inference,
            evidence_quote="",
            note_section=(
                note_section
                or ("none" if prediction == "No" else "unknown")
            ),
            experiencer=(
                experiencer
                or ("" if prediction == "No" else "patient")
            ),
            assertion_status=(
                assertion_status
                or ("absent" if prediction == "No" else "unknown")
            ),
            relation_found="",
            cancer_type_found="",
            structurer_source="fallback_no_usable_inference",
        )

    prompt = STRUCTURER_PROMPT.format(
        feature=feature,
        prediction=prediction,
        confidence=confidence,
        inference_json=json.dumps(
            inference,
            ensure_ascii=False,
        ),
    )

    try:
        response = generate_text(
            prompt,
            max_new_tokens=256,
        )

        parsed = safe_json_loads(response)

        if not isinstance(parsed, dict):
            raise ValueError(
                "Claim Composer returned unparseable JSON."
            )


        return _build_structured_claim(
            feature=feature,
            prediction=prediction,
            confidence=confidence,
            inference=inference,
            evidence_quote=inference,
            note_section=_prefer_agent1(
                note_section,
                parsed.get("note_section"),
                "unknown",
            ),
            experiencer=_prefer_agent1(
                experiencer,
                parsed.get("experiencer"),
                "",
            ),
            assertion_status=_prefer_agent1(
                assertion_status,
                parsed.get("assertion_status"),
                "unknown",
            ),
            relation_found=parsed.get(
                "relation_found",
                "",
            ),
            cancer_type_found=parsed.get(
                "cancer_type_found",
                "",
            ),
            structurer_source="llm",
        )

    except Exception as error:

        return _build_structured_claim(
            feature=feature,
            prediction=prediction,
            confidence=confidence,
            inference=inference,
            evidence_quote=inference,
            note_section=(
                note_section
                or ("none" if prediction == "No" else "unknown")
            ),
            experiencer=(
                experiencer
                or ("" if prediction == "No" else "patient")
            ),
            assertion_status=(
                assertion_status
                or (
                    "negated"
                    if prediction == "No"
                    else "affirmed"
                )
            ),
            relation_found="",
            cancer_type_found="",
            structurer_source="fallback_llm_failure",
            structurer_operational_error=str(error),
        )


def structure_all_claims(
    extractor_output: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:

    if not isinstance(extractor_output, dict):
        raise TypeError(
            "extractor_output must be a dictionary."
        )

    structured: Dict[str, Dict[str, Any]] = {}

    for feature in SYMPTOMS:

        prediction = extractor_output.get(feature)

        if prediction not in {"Yes", "No"}:
            raise ValueError(
                f"Missing/invalid Agent 1 label for "
                f"{feature}: {prediction!r}"
            )

        confidence = extractor_output.get(
            f"{feature} confidence"
        )

        inference = extractor_output.get(
            f"{feature} inference",
            "",
        )

        note_section = extractor_output.get(
            f"{feature} note_section",
            "",
        )

        experiencer = extractor_output.get(
            f"{feature} experiencer",
            "",
        )

        assertion_status = extractor_output.get(
            f"{feature} assertion_status",
            "",
        )

        structured[feature] = structure_single_claim(
            feature=feature,
            prediction=prediction,
            confidence=confidence,
            inference=inference,
            note_section=note_section,
            experiencer=experiencer,
            assertion_status=assertion_status,
        )

    return structured
