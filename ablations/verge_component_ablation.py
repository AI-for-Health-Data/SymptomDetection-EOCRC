import argparse
import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional
import run_verge_continuation as rvc


ORIGINAL_BUILD_INITIAL_A3 = rvc.build_initial_agent3_verification
ORIGINAL_VERIFY_CLAIM = rvc.verify_claim
ORIGINAL_CONTINUE = rvc.continue_verify_refine_from_frozen_agent3
ORIGINAL_JUDGE = rvc.judge_claim


def _initial_history(
    original_claim: Dict[str, Any],
    initial_verification: Dict[str, Any],
):
    label = str(original_claim.get("prediction", "")).strip()

    verification_error = rvc.verification_operational_error(
        initial_verification
    )

    history = [
        {
            "stage": "INITIAL_AGENT3",
            "round": 0,
            "entering_label": label,
            "verification": rvc.json_safe(initial_verification),
            "verification_operational_error": verification_error,
            "operational_error": verification_error,
            "exiting_label": label,
            "label_changed": False,
        }
    ]

    return label, verification_error, history


def _base_loop_result(
    original_claim: Dict[str, Any],
    initial_verification: Dict[str, Any],
    *,
    final_claim: Optional[Dict[str, Any]] = None,
    final_refinement: Optional[Dict[str, Any]] = None,
    history=None,
    label_history=None,
    total_refinement_rounds: int = 0,
    flip_count: int = 0,
    label_stable: bool = False,
    exit_reason: str = "",
    operational_failure: bool = False,
    final_verification_applies_to_final_claim: bool = True,
):
    feature = str(original_claim.get("feature", ""))

    if final_claim is None:
        final_claim = dict(original_claim)

    if final_refinement is None:
        final_refinement = {}

    if history is None:
        _, _, history = _initial_history(
            original_claim,
            initial_verification,
        )

    if label_history is None:
        label_history = [
            str(original_claim.get("prediction", "")).strip()
        ]

    return {
        "feature": feature,
        "final_claim": dict(final_claim),
        "final_verification": dict(initial_verification),
        "final_refinement": dict(final_refinement),
        "history": history,
        "label_history": label_history,
        "total_refinement_rounds": int(total_refinement_rounds),
        "flip_count": int(flip_count),
        "label_stable": bool(label_stable),
        "exit_reason": exit_reason,
        "human_review_oscillation": False,
        "operational_failure": bool(operational_failure),
        "reached_max_refinement_bound": False,
        "final_bound_verification_verdict": "",
        "final_verification_applies_to_final_claim": bool(
            final_verification_applies_to_final_claim
        ),
    }


def judge_without_agent5(
    structured_claim: Dict[str, Any],
    loop_info: Optional[Dict[str, Any]] = None,
    label_probs: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    loop = dict(loop_info or {})

    final_claim = loop.get("final_claim", {})
    if not isinstance(final_claim, dict):
        final_claim = {}

    label = str(
        final_claim.get(
            "prediction",
            structured_claim.get("prediction", ""),
        )
    ).strip()

    if label not in {"Yes", "No"}:
        label = str(
            structured_claim.get("prediction", "No")
        ).strip()

    if label not in {"Yes", "No"}:
        label = "No"

    final_refinement = loop.get("final_refinement", {})
    if not isinstance(final_refinement, dict):
        final_refinement = {}

    if final_refinement:
        source = "refiner"
    else:
        source = "verifier"

    forced_review = bool(
        rvc.safe_bool(loop.get("operational_failure", False))
        or rvc.safe_bool(
            loop.get("human_review_oscillation", False)
        )
    )

    reasons = []
    if rvc.safe_bool(
        loop.get("human_review_oscillation", False)
    ):
        reasons.append("MAX_ROUNDS_LABEL_OSCILLATION")

    if rvc.safe_bool(
        loop.get("operational_failure", False)
    ):
        reasons.append("UPSTREAM_OPERATIONAL_FAILURE")

    return {
        "feature": structured_claim.get("feature", ""),
        "original_prediction": structured_claim.get(
            "prediction", ""
        ),
        "current_prediction": label,
        "final_label": label,
        "confidence": "NOT_APPLICABLE",
        "decision_source": source,
        "explanation": (
            "Agent 5 ablated; final verify-refine loop label retained."
        ),
        "for_human_review": forced_review,
        "forced_human_review": forced_review,
        "forced_review_reasons": reasons,
        "judge_generated_label": "",
        "judge_label_probs": {},
        "judge_primary_schema_valid": True,
        "judge_retry_used": False,
        "judge_retry_schema_valid": False,
        "judge_parse_success": True,
        "judge_operational_error": False,
    }


def continue_without_refiner(
    original_claim: Dict[str, Any],
    initial_verification: Dict[str, Any],
    note_text: str,
    label_probs: Optional[Dict[str, Any]] = None,
    skip_bertscore: bool = False,
    max_rounds: int = 5,
):
    current_label, initial_error, history = _initial_history(
        original_claim,
        initial_verification,
    )

    verdict = str(
        initial_verification.get("verdict", "")
    ).strip().upper()

    if verdict == "VERIFIED":
        exit_reason = "VERIFIED"
    elif verdict == "REFINE":
        exit_reason = "REFINER_ABLATED"
    else:
        exit_reason = "INVALID_INITIAL_VERIFIER_STATE"
        initial_error = True

    return _base_loop_result(
        original_claim,
        initial_verification,
        final_claim=original_claim,
        final_refinement={},
        history=history,
        label_history=[current_label],
        total_refinement_rounds=0,
        flip_count=0,
        label_stable=False,
        exit_reason=exit_reason,
        operational_failure=initial_error,
        final_verification_applies_to_final_claim=True,
    )


def continue_without_reverification(
    original_claim: Dict[str, Any],
    initial_verification: Dict[str, Any],
    note_text: str,
    label_probs: Optional[Dict[str, Any]] = None,
    skip_bertscore: bool = False,
    max_rounds: int = 5,
):
    current_label, initial_error, history = _initial_history(
        original_claim,
        initial_verification,
    )

    verdict = str(
        initial_verification.get("verdict", "")
    ).strip().upper()

    if verdict == "VERIFIED":
        return _base_loop_result(
            original_claim,
            initial_verification,
            final_claim=original_claim,
            final_refinement={},
            history=history,
            label_history=[current_label],
            total_refinement_rounds=0,
            flip_count=0,
            label_stable=False,
            exit_reason="VERIFIED",
            operational_failure=initial_error,
            final_verification_applies_to_final_claim=True,
        )

    if verdict != "REFINE":
        return _base_loop_result(
            original_claim,
            initial_verification,
            final_claim=original_claim,
            final_refinement={},
            history=history,
            label_history=[current_label],
            total_refinement_rounds=0,
            flip_count=0,
            label_stable=False,
            exit_reason="INVALID_INITIAL_VERIFIER_STATE",
            operational_failure=True,
            final_verification_applies_to_final_claim=True,
        )

    try:
        refinement = rvc.refine_claim(
            structured_claim=original_claim,
            verification=initial_verification,
            note_text=str(note_text or ""),
            label_probs=label_probs,
        )
    except Exception as error:
        refinement = rvc.make_refinement_error(
            original_claim,
            initial_verification,
            f"Unhandled Agent 4 exception: {error}",
            label_probs=label_probs,
        )

    refinement = dict(refinement or {})

    ref_error = rvc.refinement_operational_error(
        refinement
    )

    if ref_error:
        round_record = {
            "stage": "REFINEMENT_ROUND",
            "round": 1,
            "entering_claim": rvc.json_safe(original_claim),
            "entering_label": current_label,
            "pre_refinement_verification": rvc.json_safe(
                initial_verification
            ),
            "refinement": rvc.json_safe(refinement),
            "refinement_operational_error": True,
            "verification_operational_error": initial_error,
            "operational_error": True,
            "exiting_claim": rvc.json_safe(original_claim),
            "exiting_label": current_label,
            "label_changed": False,
            "post_refinement_verification": None,
        }

        history.append(round_record)

        return _base_loop_result(
            original_claim,
            initial_verification,
            final_claim=original_claim,
            final_refinement=refinement,
            history=history,
            label_history=[current_label],
            total_refinement_rounds=1,
            flip_count=0,
            label_stable=False,
            exit_reason="REFINEMENT_ERROR",
            operational_failure=True,
            final_verification_applies_to_final_claim=True,
        )

    corrected_label = str(
        refinement.get("corrected_prediction", "")
    ).strip()

    if corrected_label not in {"Yes", "No"}:
        raise RuntimeError(
            "Agent 4 produced invalid corrected_prediction "
            f"in no_reverification ablation: {corrected_label!r}"
        )

    promoted_claim = rvc.build_refined_claim(
        original_claim,
        refinement,
    )

    changed = corrected_label != current_label

    round_record = {
        "stage": "REFINEMENT_ROUND",
        "round": 1,
        "entering_claim": rvc.json_safe(original_claim),
        "entering_label": current_label,
        "pre_refinement_verification": rvc.json_safe(
            initial_verification
        ),
        "refinement": rvc.json_safe(refinement),
        "refinement_operational_error": False,
        "verification_operational_error": initial_error,
        "operational_error": initial_error,
        "exiting_claim": rvc.json_safe(promoted_claim),
        "exiting_label": corrected_label,
        "label_changed": changed,
        "post_refinement_verification": None,
    }

    history.append(round_record)

    return _base_loop_result(
        original_claim,
        initial_verification,
        final_claim=promoted_claim,
        final_refinement=refinement,
        history=history,
        label_history=[current_label, corrected_label],
        total_refinement_rounds=1,
        flip_count=int(changed),
        label_stable=not changed,
        exit_reason="REVERIFICATION_ABLATED",
        operational_failure=initial_error,
        # Agent 3 verified the pre-refinement claim, not the promoted one.
        final_verification_applies_to_final_claim=False,
    )


def _remove_positive_clinical_validation(
    verification: Dict[str, Any],
):
    result = copy.deepcopy(verification or {})

    if rvc.verification_operational_error(result):
        return result

    grounding = str(
        result.get("grounding_level", "")
    ).strip().upper()

    result["validation_results"] = {
        "clinical_validation_disabled": True
    }

    if grounding in {"STRONG", "MODERATE"}:
        result["verdict"] = "VERIFIED"
        result["issues_found"] = []
        result["verdict_reasoning"] = (
            "Ablation: positive claim accepted using textual "
            "grounding only; clinical-validation criteria disabled."
        )
    else:
        result["verdict"] = "REFINE"
        result["issues_found"] = [
            "GROUNDING_BELOW_VERIFICATION_THRESHOLD"
        ]
        result["verdict_reasoning"] = (
            "Ablation: positive claim failed grounding-only "
            "verification; clinical-validation criteria disabled."
        )

    return result


def build_initial_no_clinical_validation(row):
    result = ORIGINAL_BUILD_INITIAL_A3(row)

    prediction = str(
        row.get("prediction", "")
    ).strip()

    if prediction == "Yes":
        result = _remove_positive_clinical_validation(
            result
        )

    return result


def verify_no_clinical_validation(
    structured_claim: Dict[str, Any],
    note_text: str,
    skip_bertscore: bool = False,
    label_probs: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    prediction = str(
        structured_claim.get("prediction", "")
    ).strip()

    result = ORIGINAL_VERIFY_CLAIM(
        structured_claim=structured_claim,
        note_text=note_text,
        skip_bertscore=skip_bertscore,
        label_probs=label_probs,
    )

    if prediction == "Yes":
        result = _remove_positive_clinical_validation(
            result
        )

    return result


def _disable_negative_recovery_in_verification(
    verification: Dict[str, Any],
):
    result = copy.deepcopy(verification or {})

    if rvc.verification_operational_error(result):
        return result

    result["validation_results"] = {
        "missed_evidence_found": False,
        "negative_recovery_disabled": True,
    }
    result["issues_found"] = []
    result["verdict"] = "VERIFIED"
    result["verdict_reasoning"] = (
        "Ablation: negative-recovery pathway disabled."
    )

    return result


def build_initial_no_negative_recovery(row):
    result = ORIGINAL_BUILD_INITIAL_A3(row)

    prediction = str(
        row.get("prediction", "")
    ).strip()

    if prediction == "No":
        result = _disable_negative_recovery_in_verification(
            result
        )

    return result


def verify_no_negative_recovery(
    structured_claim: Dict[str, Any],
    note_text: str,
    skip_bertscore: bool = False,
    label_probs: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    prediction = str(
        structured_claim.get("prediction", "")
    ).strip()

    if prediction != "No":
        return ORIGINAL_VERIFY_CLAIM(
            structured_claim=structured_claim,
            note_text=note_text,
            skip_bertscore=skip_bertscore,
            label_probs=label_probs,
        )

    return {
        "feature": structured_claim.get("feature", ""),
        "prediction": "No",
        "grounding_scores": {},
        "grounding_level": "NOT_APPLICABLE",
        "validation_results": {
            "missed_evidence_found": False,
            "negative_recovery_disabled": True,
        },
        "issues_found": [],
        "verdict": "VERIFIED",
        "verdict_reasoning": (
            "Ablation: negative-recovery pathway disabled."
        ),
        "operational_error": False,
        "label_probs": label_probs or {},
    }


def configure_ablation(name: str):
    # Restore production functions first.
    rvc.build_initial_agent3_verification = ORIGINAL_BUILD_INITIAL_A3
    rvc.verify_claim = ORIGINAL_VERIFY_CLAIM
    rvc.continue_verify_refine_from_frozen_agent3 = ORIGINAL_CONTINUE
    rvc.judge_claim = ORIGINAL_JUDGE

    if name == "no_final_judge":
        rvc.judge_claim = judge_without_agent5

    elif name == "no_refiner":
        rvc.continue_verify_refine_from_frozen_agent3 = (
            continue_without_refiner
        )

    elif name == "no_reverification":
        rvc.continue_verify_refine_from_frozen_agent3 = (
            continue_without_reverification
        )

    elif name == "no_clinical_validation":
        rvc.build_initial_agent3_verification = (
            build_initial_no_clinical_validation
        )
        rvc.verify_claim = verify_no_clinical_validation

    elif name == "no_negative_recovery":
        rvc.build_initial_agent3_verification = (
            build_initial_no_negative_recovery
        )
        rvc.verify_claim = verify_no_negative_recovery

    else:
        raise ValueError(f"Unknown ablation: {name}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ablation",
        required=True,
        choices=[
            "no_final_judge",
            "no_refiner",
            "no_reverification",
            "no_clinical_validation",
            "no_negative_recovery",
        ],
    )

    parser.add_argument(
        "--agent1_output",
        default=rvc.DEFAULT_A1,
    )
    parser.add_argument(
        "--agent2_pairs",
        default=rvc.DEFAULT_A2,
    )
    parser.add_argument(
        "--agent3_pairs",
        default=rvc.DEFAULT_A3,
    )

    parser.add_argument(
        "--output_dir",
        default=None,
    )

    parser.add_argument(
        "--max_notes",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--skip_bertscore",
        action="store_true",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = (
            "/lustre/smuexa01/client/users/nikkieh/utsw/runs/"
            "verge_ablation_20260818/"
            f"{args.ablation}"
        )

    Path(args.output_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in (
        args.agent1_output,
        args.agent2_pairs,
        args.agent3_pairs,
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    configure_ablation(args.ablation)

    print("=" * 80)
    print("VERGE ABLATION")
    print("Ablation:", args.ablation)
    print("Output:", args.output_dir)
    print("=" * 80)

    rvc.run(args)


if __name__ == "__main__":
    main()
