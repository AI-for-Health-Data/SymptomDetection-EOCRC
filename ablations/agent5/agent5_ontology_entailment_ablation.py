import json
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd


A5 = Path(
    "runs/verge_entailment_v1_20260823/full/"
    "entailment_results.csv"
)

BASE = Path(
    "runs/verge_recall_v2_20260821/full/"
    "verge_final_pair_level.csv"
)

GOLD = Path(
    "runs/verge_agent1_grade_ready_v2/"
    "agent1_gold_scored_pairs.csv"
)

OUT = Path(
    "runs/verge_entailment_v1_20260823/evaluation"
)

OUT.mkdir(parents=True, exist_ok=True)


def nid(x):
    x = str(x or "").strip()
    return x[:-2] if x.endswith(".0") else x


def feat(x):
    return re.sub(
        r"\s+",
        " ",
        str(x or "").strip().lower().replace("_", " "),
    )


def lab(x):
    x = str(x or "").strip().lower()

    if x in {"yes", "1", "1.0", "true", "positive"}:
        return 1

    if x in {"no", "0", "0.0", "false", "negative"}:
        return 0

    return None


def truthy(x):
    return str(x or "").strip().lower() in {
        "true", "1", "1.0", "yes"
    }


def compute_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=int)

    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else np.nan
    )

    specificity = tn / (tn + fp) if tn + fp else np.nan
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    balanced_accuracy = (recall + specificity) / 2

    den = math.sqrt(
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )

    mcc = (
        (tp * tn - fp * fn) / den
        if den
        else np.nan
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
    }

gold = pd.read_csv(
    GOLD,
    dtype=str,
    keep_default_na=False,
)

base = pd.read_csv(
    BASE,
    dtype=str,
    keep_default_na=False,
)

a5 = pd.read_csv(
    A5,
    dtype=str,
    keep_default_na=False,
)

assert len(a5) == 430, f"Expected 430 Agent-5 rows, found {len(a5)}"

gold["PAT_KEY"] = gold["PAT_ID"].map(nid)
gold["NOTE_KEY"] = gold["NOTE_ID"].map(nid)
gold["FEATURE_KEY"] = gold["Symptom"].map(feat)
gold["gold"] = gold["Gold"].map(lab)

gold = gold[
    gold["gold"].notna()
].copy()

assert len(gold) == 4033, len(gold)

base["PAT_KEY"] = base["PAT_ID"].map(nid)
base["NOTE_KEY"] = base["NOTE_ID"].map(nid)
base["FEATURE_KEY"] = base["feature"].map(feat)
base["verge4"] = base["final_loop_prediction"].map(lab)

a5["PAT_KEY"] = a5["PAT_ID"].map(nid)
a5["NOTE_KEY"] = a5["NOTE_ID"].map(nid)
a5["FEATURE_KEY"] = a5["feature"].map(feat)
a5["accepted"] = a5["accepted_recovery"].map(truthy)

df = gold[
    [
        "PAT_KEY",
        "NOTE_KEY",
        "FEATURE_KEY",
        "gold",
    ]
].merge(
    base[
        [
            "PAT_KEY",
            "NOTE_KEY",
            "FEATURE_KEY",
            "verge4",
        ]
    ],
    on=[
        "PAT_KEY",
        "NOTE_KEY",
        "FEATURE_KEY",
    ],
    how="left",
    validate="one_to_one",
)

assert df["verge4"].notna().all()

m4 = compute_metrics(
    df["gold"],
    df["verge4"],
)

expected_current = {
    "TP": 493,
    "FP": 88,
    "FN": 209,
    "TN": 3243,
}

observed_current = {
    k: m4[k]
    for k in ["TP", "FP", "FN", "TN"]
}

assert observed_current == expected_current, (
    f"Current VERGE mismatch: {observed_current}"
)

accepted = a5[
    a5["accepted"]
][
    [
        "PAT_KEY",
        "NOTE_KEY",
        "FEATURE_KEY",
        "accepted_evidence_quote",
    ]
].copy()

assert len(accepted) == 42, (
    f"Expected 42 accepted recoveries, found {len(accepted)}"
)

accepted["recovered"] = 1

df = df.merge(
    accepted,
    on=[
        "PAT_KEY",
        "NOTE_KEY",
        "FEATURE_KEY",
    ],
    how="left",
    validate="one_to_one",
)

df["recovered"] = (
    df["recovered"]
    .fillna(0)
    .astype(int)
)

bad = df[
    (df["recovered"] == 1)
    &
    (df["verge4"] != 0)
]

assert len(bad) == 0, (
    f"{len(bad)} accepted recoveries were not provisional No claims"
)


df["verge5"] = np.where(
    df["recovered"] == 1,
    1,
    df["verge4"],
)


m5 = compute_metrics(
    df["gold"],
    df["verge5"],
)

rec = df[
    df["recovered"] == 1
].copy()

tp_recovered = int(
    (rec["gold"] == 1).sum()
)

fp_added = int(
    (rec["gold"] == 0).sum()
)

recovery_precision = (
    tp_recovered / len(rec)
    if len(rec)
    else np.nan
)


print("=" * 100)
print("ONTOLOGY-CONDITIONED FIVE-AGENT VERGE")
print("=" * 100)

print("\nCURRENT VERGE:")
print(
    json.dumps(
        m4,
        indent=2,
    )
)

print("\nONTOLOGY-ENTAILMENT VERGE:")
print(
    json.dumps(
        m5,
        indent=2,
    )
)

print("\n" + "=" * 100)
print("AGENT 5 INCREMENTAL RECOVERY")
print("=" * 100)

print("Accepted recoveries :", len(rec))
print("TP recovered        :", tp_recovered)
print("FP added            :", fp_added)
print(
    "Recovery precision  :",
    f"{recovery_precision:.6f}"
)


print("\n" + "=" * 100)
print("FROZEN DEVELOPMENT TARGET")
print("=" * 100)

print(
    "Recall >= 0.730    :",
    m5["recall"] >= 0.730
)

print(
    "Precision >= 0.840 :",
    m5["precision"] >= 0.840
)

print(
    "F1 > current VERGE :",
    m5["f1"] > m4["f1"]
)

print(
    "MCC > current VERGE:",
    m5["mcc"] > m4["mcc"]
)

def posnum(col):
    return pd.to_numeric(
        a5[col],
        errors="coerce",
    ).fillna(0)


print("\n" + "=" * 100)
print("AGENT 5 RECOVERY FUNNEL")
print("=" * 100)

print(
    "Candidates examined              :",
    len(a5)
)

print(
    "Discovery parsed                 :",
    int(
        a5["discovery_parse_success"]
        .map(truthy)
        .sum()
    )
)

print(
    "Cases with grounded candidates   :",
    int(
        posnum(
            "grounded_candidate_count"
        ).gt(0).sum()
    )
)

print(
    "Cases with direct entailment     :",
    int(
        posnum(
            "entailment_direct_support_count"
        ).gt(0).sum()
    )
)

print(
    "Accepted after Agent-3 verify    :",
    int(a5["accepted"].sum())
)

rows = []

for feature, g in df.groupby(
    "FEATURE_KEY"
):

    before = compute_metrics(
        g["gold"],
        g["verge4"],
    )

    after = compute_metrics(
        g["gold"],
        g["verge5"],
    )

    rr = g[
        g["recovered"] == 1
    ]

    rows.append({
        "feature": feature,

        "TP_recovered": int(
            (rr["gold"] == 1).sum()
        ),

        "FP_added": int(
            (rr["gold"] == 0).sum()
        ),

        "before_precision":
            before["precision"],

        "after_precision":
            after["precision"],

        "before_recall":
            before["recall"],

        "after_recall":
            after["recall"],

        "before_f1":
            before["f1"],

        "after_f1":
            after["f1"],
    })


per = pd.DataFrame(rows)

print("\n" + "=" * 100)
print("PER-FINDING RECOVERY")
print("=" * 100)

print(
    per.to_string(
        index=False
    )
)


patient_df = df.copy()

groups = {
    pat: g.index.to_numpy()
    for pat, g in patient_df.groupby(
        "PAT_KEY"
    )
}

patients = np.asarray(
    list(groups.keys()),
    dtype=object,
)

rng = np.random.default_rng(42)

boot_rows = []

for b in range(2000):

    sampled_patients = rng.choice(
        patients,
        size=len(patients),
        replace=True,
    )

    sampled_idx = np.concatenate(
        [
            groups[p]
            for p in sampled_patients
        ]
    )

    bs = patient_df.loc[
        sampled_idx
    ]

    x = compute_metrics(
        bs["gold"],
        bs["verge4"],
    )

    y = compute_metrics(
        bs["gold"],
        bs["verge5"],
    )

    boot_rows.append({
        "replicate": b,
        "delta_precision":
            y["precision"] - x["precision"],
        "delta_recall":
            y["recall"] - x["recall"],
        "delta_f1":
            y["f1"] - x["f1"],
        "delta_mcc":
            y["mcc"] - x["mcc"],
    })


boot = pd.DataFrame(
    boot_rows
)

boot_summary = []

for metric in [
    "precision",
    "recall",
    "f1",
    "mcc",
]:

    values = boot[
        f"delta_{metric}"
    ].dropna()

    boot_summary.append({
        "metric": metric,
        "estimate":
            m5[metric] - m4[metric],
        "ci_low":
            float(
                values.quantile(0.025)
            ),
        "ci_high":
            float(
                values.quantile(0.975)
            ),
    })


boot_summary = pd.DataFrame(
    boot_summary
)

print("\n" + "=" * 100)
print("PATIENT-CLUSTER BOOTSTRAP")
print("ONTOLOGY-ENTAILMENT MINUS CURRENT VERGE")
print("=" * 100)

print(
    boot_summary.to_string(
        index=False
    )
)

df.to_csv(
    OUT / "entailment_final_labeled_pairs.csv",
    index=False,
)

per.to_csv(
    OUT / "entailment_per_finding.csv",
    index=False,
)

boot.to_csv(
    OUT / "entailment_bootstrap_replicates.csv",
    index=False,
)

boot_summary.to_csv(
    OUT / "entailment_bootstrap_summary.csv",
    index=False,
)


summary = {
    "current_verge": m4,
    "ontology_entailment_verge": m5,

    "accepted_recoveries":
        int(len(rec)),

    "tp_recovered":
        tp_recovered,

    "fp_added":
        fp_added,

    "recovery_precision":
        recovery_precision,

    "goal_recall_ge_073":
        bool(
            m5["recall"] >= 0.730
        ),

    "goal_precision_ge_084":
        bool(
            m5["precision"] >= 0.840
        ),

    "goal_f1_gt_current":
        bool(
            m5["f1"] > m4["f1"]
        ),

    "goal_mcc_gt_current":
        bool(
            m5["mcc"] > m4["mcc"]
        ),
}


with open(
    OUT / "entailment_evaluation_summary.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


print("\nSaved evaluation to:")
print(OUT)
