from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from sage_common import SYMPTOMS, load_llm
from agent3_unified_verifier import (
    verify_claim,
    _find_structurally_valid_direct_recovery,
)
from agent4_refiner import refine_claim
from agent5_verifier_judge import judge_claim


DEFAULT_A1 = (
    "/lustre/smuexa01/client/users/nikkieh/utsw/runs/"
    "verge_agent1_grade_ready_v2/agent1_outputs_structural_repaired.csv"
)

DEFAULT_A2 = (
    "/lustre/smuexa01/client/users/nikkieh/utsw/runs/"
    "verge_agent2_20260816/full/agent2_claims_pair_level.csv"
)

DEFAULT_A3 = (
    "/lustre/smuexa01/client/users/nikkieh/utsw/runs/"
    "verge_agent3_20260816/recovered_full/"
    "agent3_verification_pair_level.csv"
)

DEFAULT_OUTDIR = (
    "/lustre/smuexa01/client/users/nikkieh/utsw/runs/"
    "verge_continuation_20260817"
)

MAX_REFINE_ROUNDS = 5
CHECKPOINT_EVERY_NOTES = 5


def parse_jsonish(value: Any, default: Any = None) -> Any:
    if default is None:
        default = {}

    if isinstance(value, (dict, list)):
        return value

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    text = str(value).strip()
    if not text:
        return default

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return default


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    return False


def clean_scalar(value: Any) -> Any:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return value


def normalize_id(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass

    return obj

def build_agent2_claim(row: pd.Series) -> Dict[str, Any]:
    feature = str(clean_scalar(row.get("feature", ""))).strip()

    return {
        "feature": feature,
        "prediction": str(clean_scalar(row.get("prediction", ""))).strip(),
        "confidence": clean_scalar(row.get("confidence", "")),
        "original_inference": str(
            clean_scalar(row.get("original_inference", ""))
        ),
        "evidence_quote": str(
            clean_scalar(row.get("evidence_quote", ""))
        ),
        "note_section": str(
            clean_scalar(row.get("note_section", ""))
        ),
        "experiencer": str(
            clean_scalar(row.get("experiencer", ""))
        ),
        "assertion_status": str(
            clean_scalar(row.get("assertion_status", ""))
        ),
        "relation_found": str(
            clean_scalar(row.get("relation_found", ""))
        ),
        "cancer_type_found": str(
            clean_scalar(row.get("cancer_type_found", ""))
        ),
        "has_evidence": safe_bool(row.get("has_evidence", False)),
        "is_family_history": (
            feature == "Family history of colorectal cancer"
        ),
        "structurer_source": str(
            clean_scalar(row.get("structurer_source", ""))
        ),
        "structurer_operational_error": safe_bool(
            row.get("structurer_operational_error", False)
        ),
    }


def build_initial_agent3_verification(row: pd.Series) -> Dict[str, Any]:
    grounding = parse_jsonish(row.get("grounding_scores", ""), {})
    validation = parse_jsonish(row.get("validation_results", ""), {})
    issues = parse_jsonish(row.get("issues_found", ""), [])

    if not isinstance(grounding, dict):
        grounding = {}
    if not isinstance(validation, dict):
        validation = {}
    if not isinstance(issues, list):
        issues = [str(issues)] if issues else []

    return {
        "feature": str(clean_scalar(row.get("feature", ""))).strip(),
        "prediction": str(clean_scalar(row.get("prediction", ""))).strip(),
        "grounding_scores": grounding,
        "grounding_level": str(
            clean_scalar(row.get("grounding_level", ""))
        ),
        "validation_results": validation,
        "label_probs": parse_jsonish(row.get("label_probs", ""), {}),
        "issues_found": issues,
        "verdict": str(clean_scalar(row.get("verdict", ""))).strip(),
        "verdict_reasoning": str(
            clean_scalar(row.get("verdict_reasoning", ""))
        ),
        "operational_error": safe_bool(
            row.get("operational_error", False)
        ),
    }


def build_refined_claim(
    previous_claim: Dict[str, Any],
    refinement: Dict[str, Any],
) -> Dict[str, Any]:
    feature = str(previous_claim.get("feature", ""))
    corrected_prediction = str(
        refinement.get(
            "corrected_prediction",
            previous_claim.get("prediction", ""),
        )
        or ""
    ).strip()

    corrected_evidence = str(
        refinement.get(
            "corrected_evidence",
            previous_claim.get("evidence_quote", ""),
        )
        or ""
    )

    return {
        "feature": feature,
        "prediction": corrected_prediction,
        "confidence": previous_claim.get("confidence", ""),
        "original_inference": previous_claim.get("original_inference", ""),
        "evidence_quote": corrected_evidence,
        "note_section": str(
            refinement.get(
                "corrected_section",
                previous_claim.get("note_section", ""),
            )
            or ""
        ),
        "experiencer": str(
            refinement.get(
                "corrected_experiencer",
                previous_claim.get("experiencer", ""),
            )
            or ""
        ),
        "assertion_status": str(
            refinement.get(
                "corrected_assertion",
                previous_claim.get("assertion_status", ""),
            )
            or ""
        ),
        "relation_found": str(
            refinement.get(
                "relation_found",
                previous_claim.get("relation_found", ""),
            )
            or ""
        ),
        "cancer_type_found": str(
            refinement.get(
                "cancer_type_found",
                previous_claim.get("cancer_type_found", ""),
            )
            or ""
        ),
        "has_evidence": bool(str(corrected_evidence).strip()),
        "is_family_history": (
            feature == "Family history of colorectal cancer"
        ),
        "structurer_source": previous_claim.get(
            "structurer_source",
            "",
        ),
        "structurer_operational_error": previous_claim.get(
            "structurer_operational_error",
            False,
        ),
        # Explicit provenance marker for downstream audit.
        "claim_state_source": "agent4_refinement",
    }


def verification_operational_error(
    verification: Dict[str, Any],
) -> bool:
    if not isinstance(verification, dict):
        return True

    if safe_bool(verification.get("operational_error", False)):
        return True

    if str(
        verification.get("grounding_level", "") or ""
    ).strip().upper() == "ERROR":
        return True

    issues = verification.get("issues_found", [])
    if not isinstance(issues, list):
        issues = [issues]

    prefixes = (
        "VERIFICATION_ERROR:",
        "VERIFICATION_OPERATIONAL_ERROR:",
    )

    return any(
        str(issue).strip().upper().startswith(
            tuple(p.upper() for p in prefixes)
        )
        for issue in issues
    )


def refinement_operational_error(
    refinement: Dict[str, Any],
) -> bool:
    if not isinstance(refinement, dict):
        return True

    if safe_bool(refinement.get("operational_error", False)):
        return True

    return (
        str(refinement.get("change_type", "") or "")
        .strip()
        .upper()
        == "REFINEMENT_ERROR"
    )


def make_verification_error(
    claim: Dict[str, Any],
    reason: str,
    label_probs: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    return {
        "feature": claim.get("feature", ""),
        "prediction": claim.get("prediction", ""),
        "grounding_scores": {},
        "grounding_level": "ERROR",
        "validation_results": {},
        "label_probs": label_probs or {},
        "issues_found": [f"VERIFICATION_ERROR: {reason}"],
        "verdict": "REFINE",
        "verdict_reasoning": reason,
        "operational_error": True,
    }


def make_refinement_error(
    claim: Dict[str, Any],
    verification: Dict[str, Any],
    reason: str,
    label_probs: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    return {
        "feature": claim.get("feature", ""),
        "original_prediction": claim.get("prediction", ""),
        "original_evidence": claim.get("evidence_quote", ""),
        "verifier_verdict": verification.get("verdict", ""),
        "verifier_issues": verification.get("issues_found", []),
        "refined": False,
        "corrected_prediction": claim.get("prediction", ""),
        "corrected_evidence": claim.get("evidence_quote", ""),
        "corrected_section": claim.get("note_section", ""),
        "corrected_experiencer": claim.get("experiencer", ""),
        "corrected_assertion": claim.get("assertion_status", ""),
        "correction_reasoning": reason,
        "relation_found": claim.get("relation_found", ""),
        "cancer_type_found": claim.get("cancer_type_found", ""),
        "change_type": "REFINEMENT_ERROR",
        "evidence_grounded": None,
        "correction_valid": False,
        "operational_error": True,
        "label_probs": label_probs or {},
        "refiner_generated_prediction": "",
        "refiner_label_probs": {},
    }

def continue_verify_refine_from_frozen_agent3(
    original_claim: Dict[str, Any],
    initial_verification: Dict[str, Any],
    note_text: str,
    label_probs: Optional[Dict[str, float]] = None,
    skip_bertscore: bool = False,
    max_rounds: int = MAX_REFINE_ROUNDS,
) -> Dict[str, Any]:
    if max_rounds != 5:
        raise ValueError(
            "VERGE final protocol requires exactly 5 maximum "
            "Agent 4 refinement rounds."
        )

    full_note = str(note_text or "")
    if not full_note.strip():
        raise RuntimeError(
            "Full clinical note is missing; continuation cannot run."
        )

    feature = str(original_claim.get("feature", ""))
    current_claim = dict(original_claim)
    current_label = str(
        current_claim.get("prediction", "")
    ).strip()

    if current_label not in {"Yes", "No"}:
        raise RuntimeError(
            f"Invalid frozen Agent 2 label for {feature}: {current_label!r}"
        )

    current_verification = dict(initial_verification or {})
    final_verification = dict(current_verification)
    final_refinement: Dict[str, Any] = {}

    history: List[Dict[str, Any]] = []
    label_history: List[str] = [current_label]

    total_refinement_rounds = 0
    flip_count = 0
    label_stable = False
    operational_failure = False
    human_review_oscillation = False
    reached_max_refinement_bound = False
    final_bound_verification_verdict = ""
    exit_reason = ""

    initial_verification_error = verification_operational_error(
        current_verification
    )
    operational_failure |= initial_verification_error

    history.append(
        {
            "stage": "INITIAL_AGENT3",
            "round": 0,
            "entering_label": current_label,
            "verification": json_safe(current_verification),
            "verification_operational_error": initial_verification_error,
            "operational_error": initial_verification_error,
            "exiting_label": current_label,
            "label_changed": False,
        }
    )

    initial_verdict = str(
        current_verification.get("verdict", "")
    ).strip().upper()
    if current_label == "No":
        try:
            direct_recovery = (
                _find_structurally_valid_direct_recovery(
                    feature=feature,
                    note_text=full_note,
                    max_candidates=3,
                )
            )
            direct_recovery_error = False
            direct_recovery_error_reason = ""
        except Exception as error:
            direct_recovery = {
                "accepted": None,
                "audit": [],
            }
            direct_recovery_error = True
            direct_recovery_error_reason = str(error)
            operational_failure = True

        accepted_direct = direct_recovery.get(
            "accepted"
        )

        history.append(
            {
                "stage": "FROZEN_NO_DIRECT_RECOVERY_SCAN",
                "round": 0,
                "entering_label": current_label,
                "candidate_found": isinstance(
                    accepted_direct,
                    dict,
                ),
                "accepted_candidate": json_safe(
                    accepted_direct
                ),
                "candidate_audit": json_safe(
                    direct_recovery.get(
                        "audit",
                        [],
                    )
                ),
                "operational_error":
                    direct_recovery_error,
                "error":
                    direct_recovery_error_reason,
                "exiting_label": current_label,
                "label_changed": False,
            }
        )

        if isinstance(accepted_direct, dict):
            try:
                current_verification = verify_claim(
                    structured_claim=current_claim,
                    note_text=full_note,
                    skip_bertscore=skip_bertscore,
                    label_probs=label_probs,
                )
            except Exception as error:
                current_verification = (
                    make_verification_error(
                        current_claim,
                        (
                            "Selective recall-V2 Agent 3 "
                            f"recheck failed: {error}"
                        ),
                        label_probs=label_probs,
                    )
                )

            final_verification = dict(
                current_verification
            )

            selective_verification_error = (
                verification_operational_error(
                    current_verification
                )
            )
            operational_failure |= (
                selective_verification_error
            )

            initial_verdict = str(
                current_verification.get(
                    "verdict",
                    "",
                )
            ).strip().upper()

            history.append(
                {
                    "stage":
                        "SELECTIVE_RECALL_V2_AGENT3_RECHECK",
                    "round": 0,
                    "entering_label": current_label,
                    "accepted_candidate": json_safe(
                        accepted_direct
                    ),
                    "verification": json_safe(
                        current_verification
                    ),
                    "verification_operational_error":
                        selective_verification_error,
                    "operational_error":
                        selective_verification_error,
                    "exiting_label": current_label,
                    "label_changed": False,
                }
            )

    if initial_verdict == "VERIFIED":
        return {
            "feature": feature,
            "final_claim": current_claim,
            "final_verification": current_verification,
            "final_refinement": {},
            "history": history,
            "label_history": label_history,
            "total_refinement_rounds": 0,
            "flip_count": 0,
            "label_stable": False,
            "exit_reason": "VERIFIED",
            "human_review_oscillation": False,
            "operational_failure": operational_failure,
            "reached_max_refinement_bound": False,
            "final_bound_verification_verdict": "",
            "final_verification_applies_to_final_claim": True,
        }

    if initial_verdict != "REFINE":
        operational_failure = True
        return {
            "feature": feature,
            "final_claim": current_claim,
            "final_verification": current_verification,
            "final_refinement": {},
            "history": history,
            "label_history": label_history,
            "total_refinement_rounds": 0,
            "flip_count": 0,
            "label_stable": False,
            "exit_reason": "INVALID_INITIAL_VERIFIER_STATE",
            "human_review_oscillation": False,
            "operational_failure": True,
            "reached_max_refinement_bound": False,
            "final_bound_verification_verdict": "",
            "final_verification_applies_to_final_claim": True,
        }

    for round_num in range(1, max_rounds + 1):
        total_refinement_rounds += 1

        pre_refine_claim = dict(current_claim)
        pre_refine_label = current_label
        pre_refine_verification = dict(current_verification)

        try:
            refinement = refine_claim(
                structured_claim=current_claim,
                verification=current_verification,
                note_text=full_note,     # FULL NOTE; never sliced
                label_probs=label_probs, # AUDIT ONLY
            )
        except Exception as error:
            refinement = make_refinement_error(
                current_claim,
                current_verification,
                f"Unhandled Agent 4 exception: {error}",
                label_probs=label_probs,
            )

        final_refinement = dict(refinement)
        ref_error = refinement_operational_error(refinement)
        operational_failure |= ref_error

        if ref_error:
            history.append(
                {
                    "stage": "REFINEMENT_ROUND",
                    "round": round_num,
                    "entering_claim": json_safe(pre_refine_claim),
                    "entering_label": pre_refine_label,
                    "pre_refinement_verification": json_safe(
                        pre_refine_verification
                    ),
                    "refinement": json_safe(refinement),
                    "refinement_operational_error": True,
                    "verification_operational_error": (
                        verification_operational_error(
                            pre_refine_verification
                        )
                    ),
                    "operational_error": True,
                    "exiting_claim": json_safe(current_claim),
                    "exiting_label": current_label,
                    "label_changed": False,
                    "post_refinement_verification": None,
                }
            )
            exit_reason = "REFINEMENT_ERROR"
            break

        corrected_label = str(
            refinement.get("corrected_prediction", "")
        ).strip()

        if corrected_label not in {"Yes", "No"}:
            operational_failure = True
            history.append(
                {
                    "stage": "REFINEMENT_ROUND",
                    "round": round_num,
                    "entering_claim": json_safe(pre_refine_claim),
                    "entering_label": pre_refine_label,
                    "pre_refinement_verification": json_safe(
                        pre_refine_verification
                    ),
                    "refinement": json_safe(refinement),
                    "refinement_operational_error": True,
                    "operational_error": True,
                    "error": (
                        "Agent 4 produced an invalid corrected_prediction."
                    ),
                    "exiting_claim": json_safe(current_claim),
                    "exiting_label": current_label,
                    "label_changed": False,
                    "post_refinement_verification": None,
                }
            )
            exit_reason = "REFINEMENT_ERROR"
            break

        promoted_claim = build_refined_claim(
            current_claim,
            refinement,
        )

        label_changed = corrected_label != current_label

        current_claim = promoted_claim
        current_label = corrected_label
        label_history.append(current_label)

        if label_changed:
            flip_count += 1

        round_record: Dict[str, Any] = {
            "stage": "REFINEMENT_ROUND",
            "round": round_num,
            "entering_claim": json_safe(pre_refine_claim),
            "entering_label": pre_refine_label,
            "pre_refinement_verification": json_safe(
                pre_refine_verification
            ),
            "refinement": json_safe(refinement),
            "refinement_operational_error": False,
            "verification_operational_error": (
                verification_operational_error(
                    pre_refine_verification
                )
            ),
            "exiting_claim": json_safe(current_claim),
            "exiting_label": current_label,
            "label_changed": label_changed,
            "post_refinement_verification": None,
            "operational_error": False,
        }

        if not label_changed:
            label_stable = True
            exit_reason = "LABEL_STABLE"
            history.append(round_record)
            break

        try:
            post_verification = verify_claim(
                structured_claim=current_claim,
                note_text=full_note,     
                skip_bertscore=skip_bertscore,
                label_probs=label_probs,  
            )
        except Exception as error:
            post_verification = make_verification_error(
                current_claim,
                f"Unhandled Agent 3 exception after Agent 4: {error}",
                label_probs=label_probs,
            )

        post_ver_error = verification_operational_error(
            post_verification
        )
        operational_failure |= post_ver_error

        final_verification = dict(post_verification)
        current_verification = dict(post_verification)

        round_record["post_refinement_verification"] = json_safe(
            post_verification
        )
        round_record["post_verification_operational_error"] = (
            post_ver_error
        )
        round_record["operational_error"] = (
            round_record["verification_operational_error"]
            or post_ver_error
        )

        history.append(round_record)

        post_verdict = str(
            post_verification.get("verdict", "")
        ).strip().upper()

        if post_verdict == "VERIFIED":
            if round_num == max_rounds:
                reached_max_refinement_bound = True
                final_bound_verification_verdict = "VERIFIED"
                exit_reason = "MAX_ROUNDS_FINAL_VERIFIED"
            else:
                exit_reason = "VERIFIED"
            break

        if post_verdict != "REFINE":
            operational_failure = True
            exit_reason = "INVALID_REVERIFICATION_STATE"
            break

        if round_num == max_rounds:
            reached_max_refinement_bound = True
            final_bound_verification_verdict = "REFINE"
            human_review_oscillation = True

            if post_ver_error:
                exit_reason = "MAX_ROUNDS_FINAL_VERIFICATION_ERROR"
            else:
                exit_reason = "MAX_ROUNDS_LABEL_OSCILLATION"
            break

    if not exit_reason:
        operational_failure = True
        exit_reason = "CONTROLLER_ERROR"

    final_verification_applies_to_final_claim = (
        exit_reason != "LABEL_STABLE"
    )

    return {
        "feature": feature,
        "final_claim": current_claim,
        "final_verification": final_verification,
        "final_refinement": final_refinement,
        "history": history,
        "label_history": label_history,
        "total_refinement_rounds": total_refinement_rounds,
        "flip_count": flip_count,
        "label_stable": label_stable,
        "exit_reason": exit_reason,
        "human_review_oscillation": human_review_oscillation,
        "operational_failure": operational_failure,
        "reached_max_refinement_bound": reached_max_refinement_bound,
        "final_bound_verification_verdict": (
            final_bound_verification_verdict
        ),
        "final_verification_applies_to_final_claim": (
            final_verification_applies_to_final_claim
        ),
    }


def load_and_audit_inputs(
    agent1_path: str,
    agent2_path: str,
    agent3_path: str,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    a1 = pd.read_csv(agent1_path)
    a2 = pd.read_csv(agent2_path)
    a3 = pd.read_csv(agent3_path)

    required_a1 = {
        "_run_row_index",
        "NOTE_ID",
        "PAT_ID",
        "Clean_note_text",
        "label_probs",
    }
    required_a2 = {
        "_run_row_index",
        "NOTE_ID",
        "PAT_ID",
        "feature",
        "prediction",
    }
    required_a3 = {
        "_run_row_index",
        "NOTE_ID",
        "PAT_ID",
        "feature",
        "prediction",
        "verdict",
        "grounding_level",
        "grounding_scores",
        "validation_results",
        "issues_found",
        "operational_error",
    }

    for name, df, required in [
        ("Agent 1", a1, required_a1),
        ("Agent 2", a2, required_a2),
        ("Agent 3", a3, required_a3),
    ]:
        missing = sorted(required - set(df.columns))
        if missing:
            raise RuntimeError(
                f"{name} input is missing columns: {missing}"
            )

    frozen_a3_errors = a3["operational_error"].map(safe_bool)
    if frozen_a3_errors.any():
        bad = a3.loc[
            frozen_a3_errors,
            ["_run_row_index", "feature", "verdict", "issues_found"],
        ]
        raise RuntimeError(
            "Frozen Agent 3 input contains operational failures; "
            "use the recovered_full canonical file.\n"
            + bad.head(20).to_string(index=False)
        )

    if not a1["_run_row_index"].is_unique:
        raise RuntimeError(
            "Agent 1 _run_row_index must be unique."
        )

    a2_counts = a2.groupby("_run_row_index").size()
    a3_counts = a3.groupby("_run_row_index").size()

    if not (a2_counts == 7).all():
        raise RuntimeError(
            "Agent 2 does not contain exactly seven claims per note."
        )
    if not (a3_counts == 7).all():
        raise RuntimeError(
            "Agent 3 does not contain exactly seven claims per note."
        )

    if a2.duplicated(["_run_row_index", "feature"]).any():
        raise RuntimeError(
            "Duplicate Agent 2 note-feature pairs detected."
        )
    if a3.duplicated(["_run_row_index", "feature"]).any():
        raise RuntimeError(
            "Duplicate Agent 3 note-feature pairs detected."
        )

    merged = a2.merge(
        a3,
        on=["_run_row_index", "feature"],
        how="outer",
        suffixes=("_a2", "_a3"),
        validate="one_to_one",
        indicator=True,
    )

    if not (merged["_merge"] == "both").all():
        bad = merged.loc[
            merged["_merge"] != "both",
            ["_run_row_index", "feature", "_merge"],
        ]
        raise RuntimeError(
            "Agent 2 / Agent 3 pair mismatch:\n"
            + bad.head(20).to_string(index=False)
        )

    merged = merged.drop(columns=["_merge"])

    pred_mismatch = (
        merged["prediction_a2"].astype(str).str.strip()
        != merged["prediction_a3"].astype(str).str.strip()
    )
    if pred_mismatch.any():
        bad = merged.loc[
            pred_mismatch,
            [
                "_run_row_index",
                "feature",
                "prediction_a2",
                "prediction_a3",
            ],
        ]
        raise RuntimeError(
            "Frozen Agent 2 and Agent 3 labels disagree:\n"
            + bad.head(20).to_string(index=False)
        )

    note_cols = [
        "_run_row_index",
        "NOTE_ID",
        "PAT_ID",
        "Clean_note_text",
        "label_probs",
    ]

    merged = merged.merge(
        a1[note_cols],
        on="_run_row_index",
        how="left",
        validate="many_to_one",
        suffixes=("", "_a1"),
    )

    if merged["Clean_note_text"].isna().any():
        raise RuntimeError(
            "One or more continuation claims have no clinical note."
        )

    if (
        merged["Clean_note_text"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise RuntimeError(
            "One or more continuation claims have an empty clinical note."
        )

    for source_col in [
        "NOTE_ID_a2",
        "NOTE_ID_a3",
        "NOTE_ID",
    ]:
        if source_col not in merged.columns:
            continue

    for a, b in [
        ("NOTE_ID_a2", "NOTE_ID"),
        ("NOTE_ID_a3", "NOTE_ID"),
        ("PAT_ID_a2", "PAT_ID"),
        ("PAT_ID_a3", "PAT_ID"),
    ]:
        if a in merged.columns and b in merged.columns:
            mismatch = [
                i
                for i, (x, y) in enumerate(
                    zip(merged[a], merged[b])
                )
                if normalize_id(x) != normalize_id(y)
            ]
            if mismatch:
                raise RuntimeError(
                    f"Identifier mismatch between {a} and {b}; "
                    f"first row index: {mismatch[0]}"
                )

    by_note = (
        merged.groupby("_run_row_index")["feature"]
        .apply(lambda s: set(map(str, s)))
    )
    expected = set(SYMPTOMS)
    bad_notes = [
        idx
        for idx, features in by_note.items()
        if features != expected
    ]
    if bad_notes:
        raise RuntimeError(
            "Feature coverage mismatch for run_row_index values: "
            f"{bad_notes[:20]}"
        )

    hashes = {
        "agent1_sha256": sha256_file(agent1_path),
        "agent2_sha256": sha256_file(agent2_path),
        "agent3_sha256": sha256_file(agent3_path),
    }

    return merged, hashes


def save_checkpoint(
    outdir: Path,
    pair_rows: List[Dict[str, Any]],
    audit_rows: List[Dict[str, Any]],
    completed_note_indices: List[Any],
) -> None:
    pd.DataFrame(pair_rows).to_csv(
        outdir / "continuation_checkpoint_pair_level.csv",
        index=False,
    )

    with open(
        outdir / "continuation_checkpoint_audit.json",
        "w",
    ) as f:
        json.dump(
            json_safe(audit_rows),
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        outdir / "continuation_checkpoint_state.json",
        "w",
    ) as f:
        json.dump(
            {
                "completed_run_row_indices": list(
                    completed_note_indices
                )
            },
            f,
            indent=2,
        )


def run(args: argparse.Namespace) -> None:
    start_time = time.time()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 82)
    print("VERGE CONTINUATION — FROZEN A2 + FROZEN INITIAL A3 -> A4<->A3 -> A5")
    print("=" * 82)
    print("Agent 1 note/probability source:", args.agent1_output)
    print("Frozen Agent 2 pairs:", args.agent2_pairs)
    print("Frozen initial Agent 3 pairs:", args.agent3_pairs)
    print("Output directory:", outdir)
    print("Clinical note policy: FULL Clean_note_text — NO CHARACTER TRUNCATION")
    print("Initial Agent 2 rerun: NO")
    print("Initial Agent 3 rerun: NO")
    print("Maximum Agent 4 rounds: 5")
    print("Fifth label-changing correction gets final Agent 3 check: YES")
    print("Label-stable Agent 4 correction promoted to final_claim: YES")
    print("Operational failures sticky / force review: YES")
    print("Agent 1 label probabilities route pipeline: NO (audit only)")
    print("=" * 82)

    merged, input_hashes = load_and_audit_inputs(
        args.agent1_output,
        args.agent2_pairs,
        args.agent3_pairs,
    )

    note_indices = list(
        merged["_run_row_index"].drop_duplicates()
    )

    if args.max_notes is not None:
        note_indices = note_indices[: args.max_notes]
        print(
            f"\n[SMOKE MODE] Processing {len(note_indices)} notes."
        )

    print("\nLoading shared LLM once...")
    load_llm()
    print("Shared LLM loaded.")

    pair_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    completed_note_indices: List[Any] = []

    exit_counts: Counter = Counter()
    final_label_counts: Counter = Counter()
    initial_verdict_counts: Counter = Counter()
    refiner_change_counts: Counter = Counter()

    total_refine_rounds = 0
    total_flips = 0
    operational_failure_claims = 0
    human_review_claims = 0
    label_stable_claims = 0
    max_bound_claims = 0
    agent4_probability_audits = 0
    agent4_probability_invalid = 0
    agent5_probability_audits = 0
    agent5_probability_invalid = 0

    for note_number, run_idx in enumerate(
        note_indices,
        start=1,
    ):
        block = merged.loc[
            merged["_run_row_index"] == run_idx
        ].copy()

        if len(block) != 7:
            raise RuntimeError(
                f"run_row_index={run_idx}: expected 7 pairs, "
                f"found {len(block)}"
            )

        full_note = str(block.iloc[0]["Clean_note_text"])
        note_probs = parse_jsonish(
            block.iloc[0].get("label_probs", ""),
            {},
        )
        if not isinstance(note_probs, dict):
            note_probs = {}

        note_decisions: List[Dict[str, Any]] = []

        for feature in SYMPTOMS:
            rows = block.loc[
                block["feature"].astype(str) == feature
            ]

            if len(rows) != 1:
                raise RuntimeError(
                    f"run_row_index={run_idx}, feature={feature}: "
                    f"expected exactly 1 pair, found {len(rows)}"
                )

            row = rows.iloc[0]

            a2_row = pd.Series(
                {
                    c[:-3] if c.endswith("_a2") else c: row[c]
                    for c in row.index
                    if c.endswith("_a2")
                    or c
                    in {
                        "_run_row_index",
                        "feature",
                        "confidence",
                        "original_inference",
                        "evidence_quote",
                        "note_section",
                        "experiencer",
                        "assertion_status",
                        "relation_found",
                        "cancer_type_found",
                        "has_evidence",
                        "is_family_history",
                        "structurer_source",
                        "structurer_operational_error",
                    }
                }
            )
            a2_row["feature"] = feature
            a2_row["prediction"] = row["prediction_a2"]

            original_claim = build_agent2_claim(a2_row)

            a3_row = pd.Series(
                {
                    "feature": feature,
                    "prediction": row["prediction_a3"],
                    "grounding_scores": row.get(
                        "grounding_scores",
                        row.get("grounding_scores_a3", ""),
                    ),
                    "grounding_level": row.get(
                        "grounding_level",
                        row.get("grounding_level_a3", ""),
                    ),
                    "validation_results": row.get(
                        "validation_results",
                        row.get("validation_results_a3", ""),
                    ),
                    "label_probs": row.get(
                        "label_probs_a3",
                        row.get("label_probs", ""),
                    ),
                    "issues_found": row.get(
                        "issues_found",
                        row.get("issues_found_a3", ""),
                    ),
                    "verdict": row.get(
                        "verdict",
                        row.get("verdict_a3", ""),
                    ),
                    "verdict_reasoning": row.get(
                        "verdict_reasoning",
                        row.get("verdict_reasoning_a3", ""),
                    ),
                    "operational_error": row.get(
                        "operational_error",
                        row.get("operational_error_a3", False),
                    ),
                }
            )

            initial_verification = (
                build_initial_agent3_verification(a3_row)
            )

            feature_probs = note_probs.get(feature)
            if not isinstance(feature_probs, dict):
                feature_probs = {}

            initial_verdict = str(
                initial_verification.get("verdict", "")
            ).upper()
            initial_verdict_counts[initial_verdict] += 1

            loop_info = (
                continue_verify_refine_from_frozen_agent3(
                    original_claim=original_claim,
                    initial_verification=initial_verification,
                    note_text=full_note,
                    label_probs=feature_probs,
                    skip_bertscore=args.skip_bertscore,
                    max_rounds=MAX_REFINE_ROUNDS,
                )
            )

            try:
                decision = judge_claim(
                    structured_claim=original_claim,
                    loop_info=loop_info,
                    label_probs=feature_probs,
                )
            except Exception as error:
                fallback_label = str(
                    loop_info.get(
                        "final_claim",
                        original_claim,
                    ).get(
                        "prediction",
                        original_claim.get("prediction", "No"),
                    )
                )
                if fallback_label not in {"Yes", "No"}:
                    fallback_label = "No"

                decision = {
                    "feature": feature,
                    "original_prediction": original_claim.get(
                        "prediction", ""
                    ),
                    "current_prediction": fallback_label,
                    "final_label": fallback_label,
                    "confidence": "LOW",
                    "decision_source": "verifier",
                    "explanation": (
                        f"Unhandled Agent 5 exception: {error}"
                    ),
                    "for_human_review": True,
                    "forced_human_review": True,
                    "forced_review_reasons": [
                        "UNHANDLED_JUDGE_ERROR"
                    ],
                    "judge_generated_label": "",
                    "judge_label_probs": {},
                    "judge_primary_schema_valid": False,
                    "judge_retry_used": False,
                    "judge_retry_schema_valid": False,
                    "judge_parse_success": False,
                    "judge_operational_error": True,
                }

            if (
                safe_bool(
                    loop_info.get("operational_failure", False)
                )
                or safe_bool(
                    loop_info.get(
                        "human_review_oscillation",
                        False,
                    )
                )
            ):
                decision["for_human_review"] = True
                decision["forced_human_review"] = True

            for history_record in loop_info.get("history", []):
                if not isinstance(history_record, dict):
                    continue
                if history_record.get("stage") != "REFINEMENT_ROUND":
                    continue
                refinement_record = history_record.get("refinement", {})
                if not isinstance(refinement_record, dict):
                    refinement_record = {}
                agent4_probability_audits += 1
                ref_prob = refinement_record.get("refiner_label_probs", {})
                if (
                    not isinstance(ref_prob, dict)
                    or not safe_bool(ref_prob.get("valid", False))
                ):
                    agent4_probability_invalid += 1

            agent5_probability_audits += 1
            judge_prob = decision.get("judge_label_probs", {})
            if (
                not isinstance(judge_prob, dict)
                or not safe_bool(judge_prob.get("valid", False))
            ):
                agent5_probability_invalid += 1

            final_claim = loop_info.get("final_claim", {})
            final_verification = loop_info.get(
                "final_verification",
                {},
            )
            final_refinement = loop_info.get(
                "final_refinement",
                {},
            )

            exit_reason = str(
                loop_info.get("exit_reason", "")
            )
            exit_counts[exit_reason] += 1

            total_refine_rounds += int(
                loop_info.get(
                    "total_refinement_rounds",
                    0,
                )
                or 0
            )
            total_flips += int(
                loop_info.get("flip_count", 0) or 0
            )

            if loop_info.get("label_stable"):
                label_stable_claims += 1

            if loop_info.get("reached_max_refinement_bound"):
                max_bound_claims += 1

            if loop_info.get("operational_failure"):
                operational_failure_claims += 1

            if decision.get("for_human_review"):
                human_review_claims += 1

            final_label = str(
                decision.get("final_label", "")
            )
            final_label_counts[final_label] += 1

            change_type = str(
                final_refinement.get(
                    "change_type",
                    "",
                )
                or ""
            )
            if change_type:
                refiner_change_counts[change_type] += 1

            final_grounding = final_verification.get(
                "grounding_scores",
                {},
            )
            if not isinstance(final_grounding, dict):
                final_grounding = {}

            pair_row = {
                "_run_row_index": run_idx,
                "NOTE_ID": row.get(
                    "NOTE_ID",
                    row.get("NOTE_ID_a2", ""),
                ),
                "PAT_ID": row.get(
                    "PAT_ID",
                    row.get("PAT_ID_a2", ""),
                ),
                "feature": feature,

                "agent2_prediction": original_claim.get(
                    "prediction", ""
                ),
                "initial_agent3_verdict": initial_verdict,

                "final_loop_prediction": final_claim.get(
                    "prediction", ""
                ),
                "final_loop_evidence": final_claim.get(
                    "evidence_quote", ""
                ),
                "final_loop_section": final_claim.get(
                    "note_section", ""
                ),
                "final_loop_experiencer": final_claim.get(
                    "experiencer", ""
                ),
                "final_loop_assertion": final_claim.get(
                    "assertion_status", ""
                ),

                "final_agent3_verdict": final_verification.get(
                    "verdict", ""
                ),
                "final_grounding_level": final_verification.get(
                    "grounding_level", ""
                ),
                "final_bleu": final_grounding.get("bleu", ""),
                "final_rouge_l": final_grounding.get(
                    "rouge_l", ""
                ),
                "final_bertscore": final_grounding.get(
                    "bertscore", ""
                ),

                "final_refiner_change": change_type,
                "agent4_generated_prediction": final_refinement.get(
                    "refiner_generated_prediction", ""
                ),
                "agent4_label_probs": json.dumps(
                    final_refinement.get("refiner_label_probs", {}),
                    ensure_ascii=False,
                ),
                "total_refinement_rounds": loop_info.get(
                    "total_refinement_rounds",
                    0,
                ),
                "flip_count": loop_info.get(
                    "flip_count",
                    0,
                ),
                "label_history": json.dumps(
                    loop_info.get("label_history", []),
                    ensure_ascii=False,
                ),
                "label_stable": loop_info.get(
                    "label_stable",
                    False,
                ),
                "exit_reason": exit_reason,
                "reached_max_refinement_bound": loop_info.get(
                    "reached_max_refinement_bound",
                    False,
                ),
                "final_bound_verification_verdict": loop_info.get(
                    "final_bound_verification_verdict",
                    "",
                ),
                "human_review_oscillation": loop_info.get(
                    "human_review_oscillation",
                    False,
                ),
                "operational_failure": loop_info.get(
                    "operational_failure",
                    False,
                ),
                "final_verification_applies_to_final_claim": (
                    loop_info.get(
                        "final_verification_applies_to_final_claim",
                        False,
                    )
                ),

                "agent5_generated_label": decision.get(
                    "judge_generated_label", ""
                ),
                "agent5_label_probs": json.dumps(
                    decision.get("judge_label_probs", {}),
                    ensure_ascii=False,
                ),
                "agent5_final_label": decision.get(
                    "final_label", ""
                ),
                "agent5_confidence": decision.get(
                    "confidence", ""
                ),
                "agent5_decision_source": decision.get(
                    "decision_source", ""
                ),
                "agent5_explanation": decision.get(
                    "explanation", ""
                ),
                "for_human_review": decision.get(
                    "for_human_review",
                    False,
                ),
                "forced_human_review": decision.get(
                    "forced_human_review",
                    False,
                ),
                "forced_review_reasons": json.dumps(
                    decision.get(
                        "forced_review_reasons",
                        [],
                    ),
                    ensure_ascii=False,
                ),
                "judge_parse_success": decision.get(
                    "judge_parse_success",
                    False,
                ),
                "judge_operational_error": decision.get(
                    "judge_operational_error",
                    False,
                ),

                # Extractor uncertainty is audit-only.
                "extractor_label_probs": json.dumps(
                    feature_probs,
                    ensure_ascii=False,
                ),
            }

            pair_rows.append(pair_row)

            audit_rows.append(
                {
                    "_run_row_index": run_idx,
                    "NOTE_ID": pair_row["NOTE_ID"],
                    "PAT_ID": pair_row["PAT_ID"],
                    "feature": feature,
                    "original_agent2_claim": json_safe(
                        original_claim
                    ),
                    "frozen_initial_agent3_verification": (
                        json_safe(initial_verification)
                    ),
                    "loop_info": json_safe(loop_info),
                    "agent5_decision": json_safe(decision),
                    "extractor_label_probs": json_safe(
                        feature_probs
                    ),
                }
            )

            note_decisions.append(decision)

        completed_note_indices.append(run_idx)

        if (
            note_number % CHECKPOINT_EVERY_NOTES == 0
            or note_number == len(note_indices)
        ):
            save_checkpoint(
                outdir,
                pair_rows,
                audit_rows,
                completed_note_indices,
            )

            print(
                f"  checkpoint {note_number}/{len(note_indices)} "
                f"| claims={len(pair_rows)} "
                f"| refine_rounds={total_refine_rounds} "
                f"| flips={total_flips} "
                f"| human_review={human_review_claims}",
                flush=True,
            )

    pair_df = pd.DataFrame(pair_rows)

    expected_claims = len(note_indices) * 7
    if len(pair_df) != expected_claims:
        raise RuntimeError(
            f"Expected {expected_claims} final pair rows, "
            f"found {len(pair_df)}"
        )

    pair_path = outdir / "verge_final_pair_level.csv"
    pair_df.to_csv(pair_path, index=False)

    note_rows = []
    for run_idx, group in pair_df.groupby(
        "_run_row_index",
        sort=False,
    ):
        note_rows.append(
            {
                "_run_row_index": run_idx,
                "NOTE_ID": group.iloc[0]["NOTE_ID"],
                "PAT_ID": group.iloc[0]["PAT_ID"],
                "n_claims": len(group),
                "n_final_yes": int(
                    (group["agent5_final_label"] == "Yes").sum()
                ),
                "n_final_no": int(
                    (group["agent5_final_label"] == "No").sum()
                ),
                "n_human_review": int(
                    group["for_human_review"]
                    .map(safe_bool)
                    .sum()
                ),
                "n_operational_failure": int(
                    group["operational_failure"]
                    .map(safe_bool)
                    .sum()
                ),
                "total_refinement_rounds": int(
                    group["total_refinement_rounds"]
                    .fillna(0)
                    .astype(int)
                    .sum()
                ),
                "total_flips": int(
                    group["flip_count"]
                    .fillna(0)
                    .astype(int)
                    .sum()
                ),
            }
        )

    note_path = outdir / "verge_final_note_level.csv"
    pd.DataFrame(note_rows).to_csv(
        note_path,
        index=False,
    )

    audit_path = outdir / "verge_final_audit.json"
    with audit_path.open("w") as f:
        json.dump(
            json_safe(audit_rows),
            f,
            indent=2,
            ensure_ascii=False,
        )

    elapsed = time.time() - start_time

    summary = {
        "notes_processed": len(note_indices),
        "claims_processed": len(pair_df),
        "expected_claims": expected_claims,
        "max_refinement_rounds": MAX_REFINE_ROUNDS,

        "initial_agent3_verdict_counts": dict(
            initial_verdict_counts
        ),
        "exit_reason_counts": dict(exit_counts),
        "refiner_change_counts": dict(
            refiner_change_counts
        ),
        "agent5_final_label_counts": dict(
            final_label_counts
        ),

        "total_refinement_rounds": total_refine_rounds,
        "total_label_flips": total_flips,
        "label_stable_claims": label_stable_claims,
        "reached_max_refinement_bound_claims": (
            max_bound_claims
        ),
        "operational_failure_claims": (
            operational_failure_claims
        ),
        "human_review_claims": human_review_claims,

        "agent4_probability_audits": agent4_probability_audits,
        "agent4_probability_invalid": agent4_probability_invalid,
        "agent5_probability_audits": agent5_probability_audits,
        "agent5_probability_invalid": agent5_probability_invalid,

        "clinical_note_policy": (
            "FULL Clean_note_text; no character truncation"
        ),
        "initial_agent2_rerun": False,
        "initial_agent3_rerun": False,
        "label_probabilities_route_pipeline": False,

        "input_hashes": input_hashes,
        "controller_sha256": sha256_file(__file__),
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60.0,
    }

    summary_path = outdir / "verge_final_summary.json"
    with summary_path.open("w") as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 82)
    print("VERGE CONTINUATION COMPLETE")
    print("=" * 82)
    print(f"{'notes_processed':42s}: {len(note_indices)}")
    print(f"{'claims_processed':42s}: {len(pair_df)}")
    print(f"{'expected_claims':42s}: {expected_claims}")
    print(f"{'total_refinement_rounds':42s}: {total_refine_rounds}")
    print(f"{'total_label_flips':42s}: {total_flips}")
    print(f"{'label_stable_claims':42s}: {label_stable_claims}")
    print(f"{'max_bound_claims':42s}: {max_bound_claims}")
    print(
        f"{'operational_failure_claims':42s}: "
        f"{operational_failure_claims}"
    )
    print(f"{'human_review_claims':42s}: {human_review_claims}")
    print(f"{'agent4_probability_audits':42s}: {agent4_probability_audits}")
    print(f"{'agent4_probability_invalid':42s}: {agent4_probability_invalid}")
    print(f"{'agent5_probability_audits':42s}: {agent5_probability_audits}")
    print(f"{'agent5_probability_invalid':42s}: {agent5_probability_invalid}")
    print(f"{'runtime_minutes':42s}: {elapsed/60.0:.3f}")

    print("\nExit reasons:")
    for key, value in sorted(exit_counts.items()):
        print(f"  {key:42s}: {value}")

    print("\nAgent 5 final labels:")
    for key, value in sorted(final_label_counts.items()):
        print(f"  {key:42s}: {value}")

    print("\nOutputs:")
    print(pair_path)
    print(note_path)
    print(audit_path)
    print(summary_path)
    print("\nSUCCESS: VERGE continuation completed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Continue frozen VERGE Agent 2 + initial Agent 3 outputs "
            "through bounded Agent 4<->Agent 3 refinement and Agent 5."
        )
    )

    parser.add_argument(
        "--agent1_output",
        default=DEFAULT_A1,
    )
    parser.add_argument(
        "--agent2_pairs",
        default=DEFAULT_A2,
    )
    parser.add_argument(
        "--agent3_pairs",
        default=DEFAULT_A3,
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTDIR,
    )
    parser.add_argument(
        "--max_notes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--skip_bertscore",
        action="store_true",
        help=(
            "Diagnostic only. Final experiment should leave "
            "BERTScore enabled."
        ),
    )

    args = parser.parse_args()

    for path in [
        args.agent1_output,
        args.agent2_pairs,
        args.agent3_pairs,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    run(args)


if __name__ == "__main__":
    main()
