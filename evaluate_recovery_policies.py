import argparse
from typing import Optional

import numpy as np
import pandas as pd


SYMPTOMS = [
    "Abdominal pain",
    "Rectal bleeding",
    "Rectal pain",
    "Diarrhea",
    "Constipation",
    "Weight loss",
    "Family history of colorectal cancer",
]

FH_SYMPTOM = "Family history of colorectal cancer"


def canon_id(value) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def parse_binary(value) -> Optional[int]:
    if pd.isna(value):
        return None

    if isinstance(value, (int, float, np.integer, np.floating)):
        if float(value) == 1.0:
            return 1
        if float(value) == 0.0:
            return 0
        return None

    text = str(value).strip().lower()

    positive = {
        "1", "1.0", "yes", "y", "true", "positive",
        "present", "keep", "supported"
    }

    negative = {
        "0", "0.0", "no", "n", "false", "negative",
        "absent", "reject", "drop", "contradicted"
    }

    if text in positive:
        return 1

    if text in negative:
        return 0

    return None


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def score_policy(scored: pd.DataFrame, prediction_col: str, policy_name: str):
    pred = scored[prediction_col].fillna(0).astype(int)
    gold = scored["gold"].astype(int)

    tp = int(((pred == 1) & (gold == 1)).sum())
    fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())
    tn = int(((pred == 0) & (gold == 0)).sum())

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    fh_mask = scored["symptom"] == FH_SYMPTOM
    fh_pred = pred[fh_mask]
    fh_gold = gold[fh_mask]

    fh_predictions = int((fh_pred == 1).sum())
    fh_tp = int(((fh_pred == 1) & (fh_gold == 1)).sum())
    fh_fp = int(((fh_pred == 1) & (fh_gold == 0)).sum())
    fh_gold_negative = int((fh_gold == 0).sum())

    fh_fp_rate = safe_divide(fh_fp, fh_gold_negative)
    fh_false_discovery_rate = safe_divide(fh_fp, fh_predictions)

    return {
        "Policy": policy_name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "FH Predictions": fh_predictions,
        "FH TP": fh_tp,
        "FH FP": fh_fp,
        "FH FP Rate": fh_fp_rate,
        "FH False Discovery Rate": fh_false_discovery_rate,
    }


def score_reinstatements(
    df: pd.DataFrame,
    eligibility_col: str,
    name: str,
):
    eligible = df[df[eligibility_col] == 1].copy()

    tp = int((eligible["gold"] == 1).sum())
    fp = int((eligible["gold"] == 0).sum())

    return {
        "Recovery Policy": name,
        "Reinstated": len(eligible),
        "Reinstated TP": tp,
        "Reinstated FP": fp,
        "Reinstatement Precision": safe_divide(tp, tp + fp),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Agent 3 as a verifier-error recovery layer and compare "
            "it with constrained and threshold-based recovery policies."
        )
    )

    parser.add_argument(
        "--adjudicated",
        required=True,
        help="adjudicated_*_v2.csv",
    )

    parser.add_argument(
        "--gold",
        required=True,
        help="Wide gold-label CSV",
    )

    parser.add_argument(
        "--gold_id",
        default="NOTE_ID",
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=0.66,
        help="BLEU threshold for threshold-based recovery.",
    )

    parser.add_argument(
        "--out_prefix",
        default="recovery",
    )

    args = parser.parse_args()

    pred = pd.read_csv(args.adjudicated)
    gold = pd.read_csv(args.gold)

    required = {
        "note_id",
        "symptom",
        "label_before",
        "verifier_keep",
        "adjudicator_keep",
        "nli_rejected",
        "grade_rejected",
        "grade_bleu",
    }

    missing = required - set(pred.columns)

    if missing:
        raise SystemExit(
            f"[ERROR] Missing adjudicated columns: {sorted(missing)}"
        )

    if args.gold_id not in gold.columns:
        raise SystemExit(
            f"[ERROR] Gold ID column '{args.gold_id}' not found."
        )

    pred["note_id"] = pred["note_id"].map(canon_id)
    pred["symptom"] = pred["symptom"].astype(str).str.strip()

    binary_columns = [
        "label_before",
        "verifier_keep",
        "adjudicator_keep",
        "nli_rejected",
        "grade_rejected",
    ]

    for column in binary_columns:
        pred[column] = pred[column].map(parse_binary).fillna(0).astype(int)

    pred["grade_bleu"] = pd.to_numeric(
        pred["grade_bleu"],
        errors="coerce",
    ).fillna(0.0)

    duplicate_keys = pred.duplicated(
        subset=["note_id", "symptom"],
        keep=False,
    )

    if duplicate_keys.any():
        conflicts = (
            pred.loc[duplicate_keys]
            .groupby(["note_id", "symptom"])[binary_columns]
            .nunique()
            .max(axis=1)
        )

        if (conflicts > 1).any():
            raise SystemExit(
                "[ERROR] Conflicting duplicate note-symptom rows found."
            )

        pred = pred.drop_duplicates(
            subset=["note_id", "symptom"],
            keep="last",
        )

    available_symptoms = [
        symptom for symptom in SYMPTOMS
        if symptom in gold.columns
    ]

    gold_long = gold.melt(
        id_vars=[args.gold_id],
        value_vars=available_symptoms,
        var_name="symptom",
        value_name="gold_raw",
    )

    gold_long["note_id"] = gold_long[args.gold_id].map(canon_id)
    gold_long["gold"] = gold_long["gold_raw"].map(parse_binary)

    gold_long = gold_long[
        gold_long["gold"].notna()
    ].copy()

    gold_long["gold"] = gold_long["gold"].astype(int)

    scored = gold_long[
        ["note_id", "symptom", "gold"]
    ].merge(
        pred[
            [
                "note_id",
                "symptom",
                "label_before",
                "verifier_keep",
                "adjudicator_keep",
                "nli_rejected",
                "grade_rejected",
                "grade_bleu",
            ]
        ],
        on=["note_id", "symptom"],
        how="left",
        validate="one_to_one",
    )

    fill_binary = [
        "label_before",
        "verifier_keep",
        "adjudicator_keep",
        "nli_rejected",
        "grade_rejected",
    ]

    for column in fill_binary:
        scored[column] = scored[column].fillna(0).astype(int)

    scored["grade_bleu"] = scored["grade_bleu"].fillna(0.0)


    # Agent 1 / Experiment 7 extractor.
    scored["extractor"] = scored["label_before"]

    # Agent 2 only.
    scored["verifier_only"] = scored["verifier_keep"]

    # Agent 3 
    scored["agent3_unrestricted_reinstate"] = (
        (scored["verifier_keep"] == 0)
        & (scored["adjudicator_keep"] == 1)
    ).astype(int)

    scored["full_agent3_unrestricted"] = (
        (scored["verifier_keep"] == 1)
        | (scored["agent3_unrestricted_reinstate"] == 1)
    ).astype(int)

    # Constrained Agent 3:
    # only recover NLI-gate rejections, never grounding-gate rejections.
    scored["agent3_nli_only_reinstate"] = (
        (scored["verifier_keep"] == 0)
        & (scored["adjudicator_keep"] == 1)
        & (scored["nli_rejected"] == 1)
        & (scored["grade_rejected"] == 0)
    ).astype(int)

    scored["full_agent3_nli_only"] = (
        (scored["verifier_keep"] == 1)
        | (scored["agent3_nli_only_reinstate"] == 1)
    ).astype(int)

    # Simple non-LLM baseline:
    # reinstate NLI-gate rejections using BLEU alone.
    scored["threshold_reinstate"] = (
        (scored["verifier_keep"] == 0)
        & (scored["nli_rejected"] == 1)
        & (scored["grade_rejected"] == 0)
        & (scored["grade_bleu"] >= args.tau)
    ).astype(int)

    scored["full_threshold"] = (
        (scored["verifier_keep"] == 1)
        | (scored["threshold_reinstate"] == 1)
    ).astype(int)

    policies = [
        ("extractor", "Experiment 7 / Extractor"),
        ("verifier_only", "Verifier only"),
        (
            "full_agent3_unrestricted",
            "Verifier + Agent 3 unrestricted",
        ),
        (
            "full_agent3_nli_only",
            "Verifier + Agent 3 NLI-only",
        ),
        (
            "full_threshold",
            f"Verifier + BLEU threshold ({args.tau:.2f})",
        ),
    ]

    policy_results = []

    for column, name in policies:
        policy_results.append(
            score_policy(scored, column, name)
        )

    policy_summary = pd.DataFrame(policy_results)

    recovery_results = [
        score_reinstatements(
            scored,
            "agent3_unrestricted_reinstate",
            "Agent 3 unrestricted",
        ),
        score_reinstatements(
            scored,
            "agent3_nli_only_reinstate",
            "Agent 3 NLI-only",
        ),
        score_reinstatements(
            scored,
            "threshold_reinstate",
            f"BLEU threshold {args.tau:.2f}",
        ),
    ]

    recovery_summary = pd.DataFrame(recovery_results)

    agent3_reinstated = scored[
        scored["agent3_unrestricted_reinstate"] == 1
    ].copy()

    gate_rows = []

    gate_definitions = [
        (
            "NLI gate only",
            (agent3_reinstated["nli_rejected"] == 1)
            & (agent3_reinstated["grade_rejected"] == 0),
        ),
        (
            "Grounding gate only",
            (agent3_reinstated["grade_rejected"] == 1)
            & (agent3_reinstated["nli_rejected"] == 0),
        ),
        (
            "Both gates",
            (agent3_reinstated["nli_rejected"] == 1)
            & (agent3_reinstated["grade_rejected"] == 1),
        ),
        (
            "Neither/other",
            (agent3_reinstated["nli_rejected"] == 0)
            & (agent3_reinstated["grade_rejected"] == 0),
        ),
    ]

    for gate_name, mask in gate_definitions:
        subset = agent3_reinstated[mask]

        tp = int((subset["gold"] == 1).sum())
        fp = int((subset["gold"] == 0).sum())

        gate_rows.append(
            {
                "Rejection Type": gate_name,
                "Reinstated": len(subset),
                "TP": tp,
                "FP": fp,
                "Precision": safe_divide(tp, tp + fp),
            }
        )

    gate_summary = pd.DataFrame(gate_rows)

    display_columns = [
        "Policy",
        "Precision",
        "Recall",
        "F1",
        "FP",
        "FH Predictions",
        "FH FP",
        "FH FP Rate",
    ]

    print("\n" + "=" * 112)
    print("VERIFIER-ERROR RECOVERY OPERATING POINTS")
    print("=" * 112)

    print(
        policy_summary[display_columns].to_string(
            index=False,
            formatters={
                "Precision": lambda x: f"{x:.4f}",
                "Recall": lambda x: f"{x:.4f}",
                "F1": lambda x: f"{x:.4f}",
                "FH FP Rate": lambda x: f"{x:.4f}",
            },
        )
    )

    print("\n" + "-" * 112)
    print("RECOVERY-STAGE PERFORMANCE")
    print("-" * 112)

    print(
        recovery_summary.to_string(
            index=False,
            formatters={
                "Reinstatement Precision": lambda x: f"{x:.4f}",
            },
        )
    )

    print("\n" + "-" * 112)
    print("AGENT 3 REINSTATEMENTS BY VERIFIER REJECTION TYPE")
    print("-" * 112)

    print(
        gate_summary.to_string(
            index=False,
            formatters={
                "Precision": lambda x: (
                    f"{x:.4f}" if pd.notna(x) else "n/a"
                ),
            },
        )
    )

    policy_path = f"{args.out_prefix}_policy_summary.csv"
    recovery_path = f"{args.out_prefix}_recovery_summary.csv"
    gate_path = f"{args.out_prefix}_agent3_by_gate.csv"
    pair_path = f"{args.out_prefix}_pair_level.csv"

    policy_summary.to_csv(policy_path, index=False)
    recovery_summary.to_csv(recovery_path, index=False)
    gate_summary.to_csv(gate_path, index=False)
    scored.to_csv(pair_path, index=False)

    print("\nFiles written:")
    print(f"  {policy_path}")
    print(f"  {recovery_path}")
    print(f"  {gate_path}")
    print(f"  {pair_path}")
    print("=" * 112)


if __name__ == "__main__":
    main()
