import ast
import json
import math
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

PRED = Path(
    "runs/verge_recall_v2_20260821/full/"
    "verge_final_pair_level.csv"
)

GOLD = Path(
    "runs/verge_agent1_grade_ready_v2/"
    "agent1_gold_scored_pairs.csv"
)

OUT = Path(
    "runs/verge_final_primary_20260822"
)

OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 2000
SEED = 42


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

    if x in {
        "yes", "1", "1.0", "true", "positive"
    }:
        return 1

    if x in {
        "no", "0", "0.0", "false", "negative"
    }:
        return 0

    return None


def truthy(x):
    return str(x or "").strip().lower() in {
        "true", "1", "1.0", "yes", "y"
    }


def parse_jsonish(x, default=None):
    if default is None:
        default = []

    if isinstance(x, type(default)):
        return x

    s = str(x or "").strip()

    if not s:
        return default

    for loader in (json.loads, ast.literal_eval):
        try:
            y = loader(s)
            if isinstance(y, type(default)):
                return y
        except Exception:
            pass

    return default


def metrics(y, p):
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
        if den else np.nan
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

pred = pd.read_csv(
    PRED,
    dtype=str,
    keep_default_na=False,
)

gold = pd.read_csv(
    GOLD,
    dtype=str,
    keep_default_na=False,
)

pred["PAT_KEY"] = pred["PAT_ID"].map(nid)
pred["NOTE_KEY"] = pred["NOTE_ID"].map(nid)
pred["FEATURE_KEY"] = pred["feature"].map(feat)

gold["PAT_KEY"] = gold["PAT_ID"].map(nid)
gold["NOTE_KEY"] = gold["NOTE_ID"].map(nid)
gold["FEATURE_KEY"] = gold["Symptom"].map(feat)

gold_small = gold[
    [
        "PAT_KEY",
        "NOTE_KEY",
        "FEATURE_KEY",
        "Gold",
        "Extractor",
    ]
].copy()

df = pred.merge(
    gold_small,
    on=[
        "PAT_KEY",
        "NOTE_KEY",
        "FEATURE_KEY",
    ],
    how="left",
    validate="one_to_one",
)

df["gold_binary"] = df["Gold"].map(lab)
df["extractor_binary"] = df["Extractor"].map(lab)
df["verge_binary"] = df["final_loop_prediction"].map(lab)
df["judge_binary"] = df["agent5_final_label"].map(lab)
df["verge_final_label"] = df["final_loop_prediction"]
df["verge_final_evidence"] = df["final_loop_evidence"]
df["verge_final_section"] = df["final_loop_section"]
df["verge_final_experiencer"] = df["final_loop_experiencer"]
df["verge_final_assertion"] = df["final_loop_assertion"]

df["primary_prediction_source"] = (
    "bounded_verifier_refiner_loop"
)

df["agent5_role"] = "audit_only_non_overriding"

df.to_csv(
    OUT / "verge_primary_pair_level_all.csv",
    index=False,
)

labeled = df[
    df["gold_binary"].notna()
].copy()

assert len(labeled) == 4033, len(labeled)
assert labeled["verge_binary"].notna().all()
assert labeled["extractor_binary"].notna().all()

labeled.to_csv(
    OUT / "verge_primary_pair_level_labeled.csv",
    index=False,
)


systems = {
    "Extractor": "extractor_binary",
    "VERGE": "verge_binary",
    "Judge_override_ablation": "judge_binary",
}

overall = []

for name, col in systems.items():
    usable = labeled[col].notna()

    mm = metrics(
        labeled.loc[usable, "gold_binary"].astype(int),
        labeled.loc[usable, col].astype(int),
    )

    overall.append({
        "system": name,
        **mm,
        "N": int(usable.sum()),
    })

overall_df = pd.DataFrame(overall)

overall_df.to_csv(
    OUT / "overall_metrics.csv",
    index=False,
)

per_rows = []

for feature, g in labeled.groupby("FEATURE_KEY"):
    for name, col in {
        "Extractor": "extractor_binary",
        "VERGE": "verge_binary",
    }.items():

        usable = g[col].notna()

        mm = metrics(
            g.loc[usable, "gold_binary"].astype(int),
            g.loc[usable, col].astype(int),
        )

        per_rows.append({
            "feature": feature,
            "system": name,
            **mm,
            "N": int(usable.sum()),
        })

per_df = pd.DataFrame(per_rows)

per_df.to_csv(
    OUT / "per_finding_metrics.csv",
    index=False,
)

changed = labeled[
    labeled["extractor_binary"]
    != labeled["verge_binary"]
].copy()

changed["change_quality"] = np.where(
    (
        changed["extractor_binary"]
        != changed["gold_binary"]
    )
    &
    (
        changed["verge_binary"]
        == changed["gold_binary"]
    ),
    "corrective",
    np.where(
        (
            changed["extractor_binary"]
            == changed["gold_binary"]
        )
        &
        (
            changed["verge_binary"]
            != changed["gold_binary"]
        ),
        "degrading",
        "other",
    ),
)

changed["direction"] = np.where(
    (
        changed["extractor_binary"] == 0
    )
    &
    (
        changed["verge_binary"] == 1
    ),
    "No_to_Yes",
    "Yes_to_No",
)

changed.to_csv(
    OUT / "extractor_to_verge_changes.csv",
    index=False,
)

change_summary = (
    changed.groupby(
        [
            "FEATURE_KEY",
            "change_quality",
        ]
    )
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

for c in ["corrective", "degrading", "other"]:
    if c not in change_summary:
        change_summary[c] = 0

change_summary["net"] = (
    change_summary["corrective"]
    - change_summary["degrading"]
)

change_summary.to_csv(
    OUT / "change_summary_by_finding.csv",
    index=False,
)

judge_changed = labeled[
    labeled["judge_binary"].notna()
    &
    (
        labeled["judge_binary"]
        != labeled["verge_binary"]
    )
].copy()

judge_changed["quality"] = np.where(
    (
        judge_changed["verge_binary"]
        != judge_changed["gold_binary"]
    )
    &
    (
        judge_changed["judge_binary"]
        == judge_changed["gold_binary"]
    ),
    "corrective",
    np.where(
        (
            judge_changed["verge_binary"]
            == judge_changed["gold_binary"]
        )
        &
        (
            judge_changed["judge_binary"]
            != judge_changed["gold_binary"]
        ),
        "degrading",
        "other",
    ),
)

judge_changed.to_csv(
    OUT / "judge_override_ablation_cases.csv",
    index=False,
)

judge_summary = {
    "overrides": int(len(judge_changed)),
    "corrective": int(
        (judge_changed["quality"] == "corrective").sum()
    ),
    "degrading": int(
        (judge_changed["quality"] == "degrading").sum()
    ),
}

judge_summary["net"] = (
    judge_summary["corrective"]
    - judge_summary["degrading"]
)

UNRESOLVED = {
    "MAX_ROUNDS_LABEL_OSCILLATION",
    "REFINEMENT_ERROR",
    "INVALID_INITIAL_VERIFIER_STATE",
}

def row_review(r):
    if str(r.get("exit_reason", "")).strip().upper() in UNRESOLVED:
        return True

    for c in [
        "for_human_review",
        "forced_human_review",
        "human_review_oscillation",
        "operational_failure",
    ]:
        if c in r.index and truthy(r[c]):
            return True

    return False


df["verge_human_review"] = df.apply(
    row_review,
    axis=1,
)

labeled_review = df[
    df["gold_binary"].notna()
].copy()

labeled_review["loop_correct"] = (
    labeled_review["verge_binary"]
    == labeled_review["gold_binary"]
)

review_stats = []

for flag, g in labeled_review.groupby(
    "verge_human_review"
):
    review_stats.append({
        "human_review": bool(flag),
        "N": int(len(g)),
        "errors": int((~g["loop_correct"]).sum()),
        "error_rate": float(
            (~g["loop_correct"]).mean()
        ),
    })

pd.DataFrame(review_stats).to_csv(
    OUT / "human_review_analysis.csv",
    index=False,
)

process = {
    "claims_processed": int(len(df)),
    "labeled_claims": int(len(labeled)),
    "human_review_all_claims": int(
        df["verge_human_review"].sum()
    ),
    "human_review_labeled_claims": int(
        labeled_review["verge_human_review"].sum()
    ),
    "operational_failure_claims": int(
        sum(
            truthy(x)
            for x in df.get(
                "operational_failure",
                pd.Series([""] * len(df)),
            )
        )
    ),
    "exit_reasons": {
        str(k): int(v)
        for k, v in
        df["exit_reason"].value_counts().items()
    },
}

if "total_refinement_rounds" in df.columns:
    vals = pd.to_numeric(
        df["total_refinement_rounds"],
        errors="coerce",
    ).fillna(0)

    process["total_refinement_rounds"] = int(
        vals.sum()
    )

if "flip_count" in df.columns:
    vals = pd.to_numeric(
        df["flip_count"],
        errors="coerce",
    ).fillna(0)

    process["total_label_flips"] = int(
        vals.sum()
    )

if "history" in df.columns:
    scans = 0
    candidate_hits = 0
    selective_rechecks = 0

    for x in df["history"]:
        hist = parse_jsonish(x, [])

        for h in hist:
            if not isinstance(h, dict):
                continue

            stage = str(
                h.get("stage", "")
            ).strip()

            if stage == "FROZEN_NO_DIRECT_RECOVERY_SCAN":
                scans += 1
                if truthy(h.get("candidate_found", False)):
                    candidate_hits += 1

            if stage == "SELECTIVE_RECALL_V2_AGENT3_RECHECK":
                selective_rechecks += 1

    process["negative_direct_scans"] = scans
    process["negative_direct_candidate_hits"] = candidate_hits
    process["selective_agent3_rechecks"] = selective_rechecks


patient_groups = {
    pat: np.asarray(idx, dtype=int)
    for pat, idx in
    labeled.groupby("PAT_KEY").groups.items()
}

patients = np.asarray(
    list(patient_groups.keys()),
    dtype=object,
)

rng = np.random.default_rng(SEED)

boot_rows = []

for b in range(N_BOOT):
    sampled = rng.choice(
        patients,
        size=len(patients),
        replace=True,
    )

    idx = np.concatenate(
        [patient_groups[p] for p in sampled]
    )

    bs = labeled.loc[idx]

    ex = metrics(
        bs["gold_binary"].astype(int),
        bs["extractor_binary"].astype(int),
    )

    vg = metrics(
        bs["gold_binary"].astype(int),
        bs["verge_binary"].astype(int),
    )

    row = {"replicate": b}

    for metric in [
        "precision",
        "recall",
        "f1",
        "mcc",
    ]:
        row[f"extractor_{metric}"] = ex[metric]
        row[f"verge_{metric}"] = vg[metric]
        row[f"delta_{metric}"] = (
            vg[metric] - ex[metric]
        )

    boot_rows.append(row)

boot = pd.DataFrame(boot_rows)

boot.to_csv(
    OUT / "patient_cluster_bootstrap_replicates.csv",
    index=False,
)

boot_summary = []

point_ex = metrics(
    labeled["gold_binary"].astype(int),
    labeled["extractor_binary"].astype(int),
)

point_vg = metrics(
    labeled["gold_binary"].astype(int),
    labeled["verge_binary"].astype(int),
)

for metric in [
    "precision",
    "recall",
    "f1",
    "mcc",
]:
    for system, prefix, point in [
        ("Extractor", "extractor", point_ex[metric]),
        ("VERGE", "verge", point_vg[metric]),
    ]:
        values = boot[
            f"{prefix}_{metric}"
        ].dropna()

        boot_summary.append({
            "metric": metric,
            "system": system,
            "estimate": point,
            "ci_low": float(
                np.percentile(values, 2.5)
            ),
            "ci_high": float(
                np.percentile(values, 97.5)
            ),
        })

    values = boot[
        f"delta_{metric}"
    ].dropna()

    boot_summary.append({
        "metric": metric,
        "system": "VERGE_minus_Extractor",
        "estimate": (
            point_vg[metric]
            - point_ex[metric]
        ),
        "ci_low": float(
            np.percentile(values, 2.5)
        ),
        "ci_high": float(
            np.percentile(values, 97.5)
        ),
    })

boot_summary_df = pd.DataFrame(
    boot_summary
)

boot_summary_df.to_csv(
    OUT / "bootstrap_summary.csv",
    index=False,
)

ex_correct = (
    labeled["extractor_binary"]
    == labeled["gold_binary"]
)

vg_correct = (
    labeled["verge_binary"]
    == labeled["gold_binary"]
)

corrective = int(
    ((~ex_correct) & vg_correct).sum()
)

degrading = int(
    (ex_correct & (~vg_correct)).sum()
)

n_discordant = corrective + degrading

try:
    from scipy.stats import binomtest

    mcnemar_p = float(
        binomtest(
            corrective,
            n_discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )
except Exception:
    k = min(corrective, degrading)

    tail = sum(
        math.comb(n_discordant, i)
        * (0.5 ** n_discordant)
        for i in range(k + 1)
    )

    mcnemar_p = min(
        1.0,
        2.0 * tail,
    )

mcnemar = {
    "corrective": corrective,
    "degrading": degrading,
    "discordant": n_discordant,
    "exact_two_sided_p": mcnemar_p,
}

with open(
    OUT / "mcnemar.json",
    "w",
) as f:
    json.dump(
        mcnemar,
        f,
        indent=2,
    )


FEATURES = sorted(
    labeled["FEATURE_KEY"].unique()
)

note_rows = []

for note_id, g in labeled.groupby("NOTE_KEY"):

    if (
        g["FEATURE_KEY"].nunique()
        != len(FEATURES)
    ):
        continue

    g = g.set_index("FEATURE_KEY")

    if not all(
        f in g.index
        for f in FEATURES
    ):
        continue

    y = np.asarray(
        [
            int(g.loc[f, "gold_binary"])
            for f in FEATURES
        ]
    )

    for system, col in [
        ("Extractor", "extractor_binary"),
        ("VERGE", "verge_binary"),
    ]:
        p = np.asarray(
            [
                int(g.loc[f, col])
                for f in FEATURES
            ]
        )

        inter = int(
            ((y == 1) & (p == 1)).sum()
        )
        union = int(
            ((y == 1) | (p == 1)).sum()
        )

        jaccard = (
            inter / union
            if union
            else 1.0
        )

        note_rows.append({
            "NOTE_ID": note_id,
            "system": system,
            "exact_match": int(
                np.array_equal(y, p)
            ),
            "jaccard": jaccard,
            "hamming_loss": float(
                (y != p).mean()
            ),
        })

note_df = pd.DataFrame(note_rows)

note_df.to_csv(
    OUT / "note_level_metrics_by_note.csv",
    index=False,
)

note_summary = (
    note_df.groupby("system")
    .agg(
        notes=("NOTE_ID", "nunique"),
        exact_match_accuracy=(
            "exact_match",
            "mean",
        ),
        mean_jaccard=("jaccard", "mean"),
        hamming_loss=("hamming_loss", "mean"),
    )
    .reset_index()
)

note_summary.to_csv(
    OUT / "note_level_summary.csv",
    index=False,
)

summary = {
    "primary_system": (
        "VERGE bounded Verifier-Refiner loop"
    ),
    "agent5_role": "audit_only_non_overriding",
    "overall": {
        row["system"]: {
            k: (
                int(v)
                if k in {
                    "TP", "FP", "FN", "TN", "N"
                }
                else float(v)
            )
            for k, v in row.items()
            if k != "system"
        }
        for row in overall
    },
    "extractor_to_verge_changes": {
        "total": int(len(changed)),
        "corrective": int(
            (
                changed["change_quality"]
                == "corrective"
            ).sum()
        ),
        "degrading": int(
            (
                changed["change_quality"]
                == "degrading"
            ).sum()
        ),
        "no_to_yes": int(
            (
                changed["direction"]
                == "No_to_Yes"
            ).sum()
        ),
        "yes_to_no": int(
            (
                changed["direction"]
                == "Yes_to_No"
            ).sum()
        ),
    },
    "judge_override_ablation": judge_summary,
    "process": process,
    "mcnemar": mcnemar,
}

with open(
    OUT / "final_summary.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


print("=" * 90)
print("FINAL VERGE PRIMARY RESULTS")
print("=" * 90)

print(overall_df.to_string(index=False))

print("\nExtractor -> VERGE changes:")
print(summary["extractor_to_verge_changes"])

print("\nAgent 5 override ablation:")
print(judge_summary)

print("\nProcess:")
print(json.dumps(process, indent=2))

print("\nBootstrap:")
print(
    boot_summary_df.to_string(
        index=False
    )
)

print("\nMcNemar:")
print(mcnemar)

print("\nNote-level:")
print(
    note_summary.to_string(
        index=False
    )
)

print("\nSaved final package to:", OUT)
