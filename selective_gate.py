from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

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


def _flag_true(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}



def load_gold(path: str, id_col: str) -> Optional[Dict[Tuple[str, str], str]]:
    """Return {(note_id, symptom): 'yes'/'no'} from a wide gold CSV, or None.

    Assumes gold has one row per note keyed by id_col and one column per
    symptom holding Yes/No/blank. Blank -> excluded (returned as absent key).
    """
    gold = pd.read_csv(path)
    if id_col not in gold.columns:
        raise SystemExit(f"[ERROR] gold id column '{id_col}' not found")
    lookup: Dict[Tuple[str, str], str] = {}
    for _, row in gold.iterrows():
        nid = str(row[id_col]).strip()
        if nid.endswith(".0"):
            nid = nid[:-2]
        for symptom in SYMPTOMS:
            if symptom not in gold.columns:
                continue
            val = row[symptom]
            if pd.isna(val) or str(val).strip() == "":
                continue
            text = str(val).strip().lower()

            positive_values = {
                "1", "1.0", "yes", "y", "true", "positive", "pos", "present"
            }
            negative_values = {
                "0", "0.0", "no", "n", "false", "negative", "neg", "absent"
            }

            if text in positive_values:
                lookup[(nid, symptom)] = "yes"
            elif text in negative_values:
                lookup[(nid, symptom)] = "no"
    return lookup


def _canon(nid) -> str:
    s = str(nid).strip()
    return s[:-2] if s.endswith(".0") else s


def risk_coverage_sweep(
    rejected: pd.DataFrame,
    gold: Optional[Dict[Tuple[str, str], str]],
) -> pd.DataFrame:
    """For each threshold, reinstate NLI-gate rejections with BLEU >= tau and
    report how many are reinstated and (if gold) how many are correct.

    Coverage here = fraction of NLI-gate rejections auto-reinstated.
    Risk = fraction of reinstated pairs that are gold-negative (false reinstate).
    """
    # Only NLI-gate rejections are threshold-eligible for reinstatement.
    eligible = rejected[
        rejected["nli_rejected"].apply(_flag_true)
        & ~rejected["grade_rejected"].apply(_flag_true)
    ].copy()

    rows: List[dict] = []
    for tau in [round(t, 2) for t in np.arange(0.0, 1.01, 0.05)]:
        reinstated = eligible[eligible["grade_bleu"] >= tau]
        n_reinstated = len(reinstated)
        coverage = n_reinstated / len(eligible) if len(eligible) else 0.0
        record = {
            "tau": tau,
            "n_eligible": len(eligible),
            "n_reinstated": n_reinstated,
            "coverage": round(coverage, 4),
        }
        if gold is not None and n_reinstated:
            labels = [
                gold.get((_canon(r["note_id"]), r["symptom"]))
                for _, r in reinstated.iterrows()
            ]
            scored = [x for x in labels if x is not None]
            tp = sum(1 for x in scored if x == "yes")
            fp = sum(1 for x in scored if x == "no")
            record["reinstated_scored"] = len(scored)
            record["reinstated_TP"] = tp
            record["reinstated_FP"] = fp
            record["reinstate_precision"] = (
                round(tp / (tp + fp), 4) if (tp + fp) else np.nan
            )
            record["risk_fp_rate"] = (
                round(fp / (tp + fp), 4) if (tp + fp) else np.nan
            )
        rows.append(record)
    return pd.DataFrame(rows)


def three_bucket_policy(
    rejected: pd.DataFrame,
    tau_keep: float,
    tau_flag: float,
    gold: Optional[Dict[Tuple[str, str], str]],
) -> Tuple[pd.DataFrame, dict]:
    """Assign each verifier-rejected positive to KEEP / FLAG / REJECT.

    KEEP   : NLI-gate rejection with grounding >= tau_keep (auto-reinstate).
    FLAG   : grounding in [tau_flag, tau_keep) OR any grounding-gate rejection
             whose grounding >= tau_flag -> route to human review.
    REJECT : grounding < tau_flag -> drop.
    """
    assignments: List[str] = []
    for _, r in rejected.iterrows():
        g = float(r["grade_bleu"])
        is_grounding_gate = _flag_true(r["grade_rejected"])
        if is_grounding_gate:
            # ungrounded-by-construction: never auto-keep; flag if borderline.
            bucket = "FLAG" if g >= tau_flag else "REJECT"
        else:
            if g >= tau_keep:
                bucket = "KEEP"
            elif g >= tau_flag:
                bucket = "FLAG"
            else:
                bucket = "REJECT"
        assignments.append(bucket)
    out = rejected.copy()
    out["bucket"] = assignments

    summary = {
        "KEEP": int((out["bucket"] == "KEEP").sum()),
        "FLAG": int((out["bucket"] == "FLAG").sum()),
        "REJECT": int((out["bucket"] == "REJECT").sum()),
        "total_rejected": len(out),
    }
    if gold is not None:
        for bucket in ["KEEP", "FLAG", "REJECT"]:
            sub = out[out["bucket"] == bucket]
            labels = [
                gold.get((_canon(r["note_id"]), r["symptom"]))
                for _, r in sub.iterrows()
            ]
            scored = [x for x in labels if x is not None]
            tp = sum(1 for x in scored if x == "yes")  # truly positive
            fp = sum(1 for x in scored if x == "no")  # truly negative
            summary[f"{bucket}_true_pos"] = tp
            summary[f"{bucket}_true_neg"] = fp
    return out, summary

def compare_to_llm(
    rejected: pd.DataFrame,
    adjudicated: pd.DataFrame,
    gold: Optional[Dict[Tuple[str, str], str]],
    tau_keep: float,
) -> None:
    """Compare the LLM adjudicator's reinstatements to a pure BLEU>=tau_keep
    threshold on the same NLI-gate-eligible pool."""
    adj = adjudicated.copy()
    # normalize id/symptom for join
    id_col = "note_id" if "note_id" in adj.columns else "NOTE_ID"
    sym_col = "symptom" if "symptom" in adj.columns else "Symptom"
    keep_col = None
    for c in ["adjudicator_keep", "Adjudicator_Keep"]:
        if c in adj.columns:
            keep_col = c
            break
    if keep_col is None:
        print("[compare] no adjudicator_keep column; skipping LLM comparison")
        return
    adj["_key"] = list(zip(adj[id_col].map(_canon), adj[sym_col]))

    eligible = rejected[
        rejected["nli_rejected"].apply(_flag_true)
        & ~rejected["grade_rejected"].apply(_flag_true)
    ].copy()

    eligible["_key"] = list(
        zip(eligible["note_id"].map(_canon), eligible["symptom"])
    )

    eligible_keys = set(eligible["_key"])

    llm_reinstate = {
        key
        for key, keep in zip(adj["_key"], adj[keep_col])
        if _flag_true(keep) and key in eligible_keys
    }

    thr_reinstate = set(
        eligible.loc[
            eligible["grade_bleu"] >= tau_keep,
            "_key",
        ]
    )
    print(f"  Eligible NLI-gate pairs   : {len(eligible_keys)}")
    print(f"  LLM reinstated in pool    : {len(llm_reinstate)}")
    print(f"  Threshold reinstated      : {len(thr_reinstate)}")

    both = llm_reinstate & thr_reinstate
    llm_only = llm_reinstate - thr_reinstate
    thr_only = thr_reinstate - llm_reinstate

    print("\n" + "-" * 64)
    print(f"LLM ADJUDICATOR vs THRESHOLD (BLEU >= {tau_keep}) on NLI-gate pool")
    print("-" * 64)
    print(f"  Reinstated by both        : {len(both)}")
    print(f"  LLM only (not threshold)  : {len(llm_only)}")
    print(f"  Threshold only (not LLM)  : {len(thr_only)}")

    if gold is not None:
        def score(keys):
            s = [gold.get(k) for k in keys]
            s = [x for x in s if x is not None]
            tp = sum(1 for x in s if x == "yes")
            fp = sum(1 for x in s if x == "no")
            return tp, fp

        for name, keys in [
            ("both", both),
            ("LLM-only", llm_only),
            ("threshold-only", thr_only),
        ]:
            tp, fp = score(keys)
            prec = f"{tp/(tp+fp):.3f}" if (tp + fp) else "n/a"
            print(
                f"  {name:<16} correct(TP)={tp:>3}  wrong(FP)={fp:>3}  "
                f"precision={prec}"
            )
        print(
            "\n  Read: if 'LLM-only' reinstatements are mostly FP and "
            "'threshold-only' mostly TP,\n  the LLM adjudicator is doing worse "
            "than a threshold and can be dropped."
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Selective grounding-threshold gate (Geifman & El-Yaniv "
        "2017 reject option) with a KEEP/FLAG/REJECT three-bucket policy."
    )
    ap.add_argument("--verified", required=True, help="verify_*_v2.csv")
    ap.add_argument("--adjudicated", default=None, help="adjudicated_*_v2.csv")
    ap.add_argument("--gold", default=None)
    ap.add_argument("--gold_id", default="NOTE_ID")
    ap.add_argument("--tau_keep", type=float, default=0.66,
                    help="Auto-reinstate at/above this grounding BLEU (HIGH).")
    ap.add_argument("--tau_flag", type=float, default=0.33,
                    help="Flag-for-review at/above this (MODERATE); below=reject.")
    ap.add_argument("--out_prefix", default="selective")
    a = ap.parse_args()

    v = pd.read_csv(a.verified)
    required = {"note_id", "symptom", "grade_bleu", "nli_rejected",
                "grade_rejected", "verifier_keep"}
    missing = required - set(v.columns)
    if missing:
        raise SystemExit(f"[ERROR] verified file missing columns: {missing}")

    v["verifier_keep"] = v["verifier_keep"].apply(_flag_true)
    rejected = v[~v["verifier_keep"]].copy()

    gold = load_gold(a.gold, a.gold_id) if a.gold else None

    if gold is not None:
        gold_yes = sum(value == "yes" for value in gold.values())
        gold_no = sum(value == "no" for value in gold.values())

        print(f"  Gold lookup size    : {len(gold)}")
        print(f"  Gold positive labels: {gold_yes}")
        print(f"  Gold negative labels: {gold_no}")

        rejected_labels = [
            gold.get((_canon(row["note_id"]), row["symptom"]))
            for _, row in rejected.iterrows()
        ]

        print(
            "  Rejected gold labels: "
            f"yes={sum(x == 'yes' for x in rejected_labels)}, "
            f"no={sum(x == 'no' for x in rejected_labels)}, "
            f"missing={sum(x is None for x in rejected_labels)}"
        )
    print("=" * 64)
    print("SELECTIVE GROUNDING GATE (reject option; Geifman & El-Yaniv 2017)")
    print("=" * 64)
    print(f"  Verified file       : {a.verified}")
    print(f"  Verifier-rejected   : {len(rejected)}")
    print(f"    NLI-gate          : {int(rejected['nli_rejected'].apply(_flag_true).sum())}")
    print(f"    grounding-gate    : {int(rejected['grade_rejected'].apply(_flag_true).sum())}")
    print(f"  tau_keep (KEEP)     : {a.tau_keep}")
    print(f"  tau_flag (FLAG)     : {a.tau_flag}")
    print(f"  Gold scoring        : {'ON' if gold else 'OFF'}")
    print("=" * 64)

    # Q1: risk-coverage sweep
    sweep = risk_coverage_sweep(rejected, gold)
    sweep_path = f"{a.out_prefix}_risk_coverage.csv"
    sweep.to_csv(sweep_path, index=False)
    print("\nRISK-COVERAGE SWEEP (NLI-gate reinstatement by BLEU threshold)")
    print(sweep.to_string(index=False))
    print(f"\nWrote {sweep_path}")

    # Q2: three-bucket policy
    buckets, summary = three_bucket_policy(
        rejected, a.tau_keep, a.tau_flag, gold
    )
    bucket_path = f"{a.out_prefix}_three_bucket.csv"
    buckets.to_csv(bucket_path, index=False)
    print("\n" + "-" * 64)
    print("THREE-BUCKET POLICY  (KEEP / FLAG-for-review / REJECT)")
    print("-" * 64)
    print(f"  KEEP   (auto-reinstate) : {summary['KEEP']}")
    print(f"  FLAG   (human review)   : {summary['FLAG']}")
    print(f"  REJECT (drop)           : {summary['REJECT']}")
    if gold is not None:
        print("\n  Gold composition of each bucket (of scored pairs):")
        for b in ["KEEP", "FLAG", "REJECT"]:
            tp = summary.get(f"{b}_true_pos", 0)
            fp = summary.get(f"{b}_true_neg", 0)
            tot = tp + fp
            prec = f"{tp/tot:.3f}" if tot else "n/a"
            print(
                f"    {b:<7} truly-positive={tp:>3}  truly-negative={fp:>3}  "
                f"(purity={prec})"
            )
        print(
            "\n  Read: KEEP should be high-purity positive (safe to auto-keep);"
            "\n  REJECT should be high-purity negative (safe to drop);"
            "\n  FLAG is the uncertain middle correctly routed to a human."
        )
    print(f"\nWrote {bucket_path}")

    # Q1b: LLM vs threshold, if adjudicated file provided
    if a.adjudicated:
        adj = pd.read_csv(a.adjudicated)
        compare_to_llm(rejected, adj, gold, a.tau_keep)

    print("\n" + "=" * 64)
    print("DONE. Key files:")
    print(f"  {sweep_path}   (risk-coverage curve)")
    print(f"  {bucket_path}  (per-pair KEEP/FLAG/REJECT)")
    print("=" * 64)


if __name__ == "__main__":
    main()
