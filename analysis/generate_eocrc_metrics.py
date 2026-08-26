from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu, rankdata, spearmanr

import eval_common as ec


EXPERIMENTS_DEFAULT = ["E1", "E3", "E7", "MULTI"]

SYMPTOMS = [
    "Abdominal pain",
    "Rectal bleeding",
    "Rectal pain",
    "Diarrhea",
    "Constipation",
    "Weight loss",
    "Family history of colorectal cancer",
]

SYMPTOM_ALIASES = {
    "abdominal pain": "Abdominal pain",
    "rectal bleeding": "Rectal bleeding",
    "rectal pain": "Rectal pain",
    "diarrhea": "Diarrhea",
    "constipation": "Constipation",
    "weight loss": "Weight loss",
    "family history crc": "Family history of colorectal cancer",
    "family history of crc": "Family history of colorectal cancer",
    "family history of colorectal cancer":
        "Family history of colorectal cancer",
}


def normalize_note_id(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text or None


def normalize_symptom(value):
    text = str(value).strip()
    return SYMPTOM_ALIASES.get(text.lower(), text)


def normalize_binary(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value) if int(value) in (0, 1) else np.nan
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return np.nan
        return int(value) if value in (0.0, 1.0) else np.nan

    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1", "positive", "present"}:
        return 1
    if text in {"no", "n", "false", "0", "negative", "absent"}:
        return 0
    return np.nan


def choose_column(columns, candidates):
    lookup = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    shape = np.broadcast_shapes(numerator.shape, denominator.shape)
    output = np.full(shape, np.nan, dtype=float)
    np.divide(
        numerator,
        denominator,
        out=output,
        where=denominator != 0,
    )
    return output


def metrics_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    tp = counts[..., 0]
    fp = counts[..., 1]
    fn = counts[..., 2]
    tn = counts[..., 3]

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * tp, 2 * tp + fp + fn)
    specificity = safe_divide(tn, tn + fp)
    npv = safe_divide(tn, tn + fn)
    accuracy = safe_divide(tp + tn, tp + fp + fn + tn)
    balanced_accuracy = (recall + specificity) / 2

    denominator = np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = safe_divide(tp * tn - fp * fn, denominator)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "npv": npv,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
    }


def confusion_index(gold, prediction):
    if gold == 1 and prediction == 1:
        return 0  # TP
    if gold == 0 and prediction == 1:
        return 1  # FP
    if gold == 1 and prediction == 0:
        return 2  # FN
    return 3      # TN


def percentile_ci(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return (
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    )


def bootstrap_pvalue(delta):
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) == 0:
        return np.nan
    lower = (np.sum(delta <= 0) + 1) / (len(delta) + 1)
    upper = (np.sum(delta >= 0) + 1) / (len(delta) + 1)
    return float(min(1.0, 2 * min(lower, upper)))


def holm_adjust(values):
    pvalues = np.asarray(values, dtype=float)
    adjusted = np.full_like(pvalues, np.nan)
    valid = np.where(np.isfinite(pvalues))[0]
    if len(valid) == 0:
        return adjusted

    ordered = valid[np.argsort(pvalues[valid])]
    running = 0.0
    total = len(ordered)

    for rank, index in enumerate(ordered):
        candidate = (total - rank) * pvalues[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)

    return adjusted


def auc_from_scores(correctness, scores):
    correctness = np.asarray(correctness, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    correctness = correctness[valid]
    scores = scores[valid]

    n_positive = int((correctness == 1).sum())
    n_negative = int((correctness == 0).sum())

    if n_positive == 0 or n_negative == 0:
        return np.nan

    ranks = rankdata(scores, method="average")
    positive_rank_sum = ranks[correctness == 1].sum()

    return float(
        (
            positive_rank_sum
            - n_positive * (n_positive + 1) / 2
        )
        / (n_positive * n_negative)
    )


def load_gold(path):
    dataframe = pd.read_csv(path)

    note_column = choose_column(
        dataframe.columns,
        ["note_id", "NOTE_ID"],
    )
    symptom_column = choose_column(
        dataframe.columns,
        ["symptom", "finding"],
    )
    gold_column = choose_column(
        dataframe.columns,
        ["gold", "gold_label", "gold_raw", "truth", "reference"],
    )

    if not note_column or not symptom_column or not gold_column:
        raise SystemExit(
            f"Could not identify note, symptom, and gold columns in {path}.\n"
            f"Columns: {dataframe.columns.tolist()}"
        )

    gold = pd.DataFrame(
        {
            "note_id": dataframe[note_column].map(normalize_note_id),
            "symptom": dataframe[symptom_column].map(normalize_symptom),
            "gold": dataframe[gold_column].map(normalize_binary),
        }
    )

    gold = gold[
        gold["note_id"].notna()
        & gold["symptom"].isin(SYMPTOMS)
        & gold["gold"].notna()
    ].copy()

    gold["gold"] = gold["gold"].astype(int)

    duplicated = gold.duplicated(
        ["note_id", "symptom"],
        keep=False,
    )

    if duplicated.any():
        conflicts = (
            gold.loc[duplicated]
            .groupby(["note_id", "symptom"])["gold"]
            .nunique()
        )
        conflicts = conflicts[conflicts > 1]

        if len(conflicts):
            raise SystemExit(
                "Conflicting duplicate gold labels:\n"
                + conflicts.head(20).to_string()
            )

        gold = gold.drop_duplicates(["note_id", "symptom"])

    rank = {symptom: index for index, symptom in enumerate(SYMPTOMS)}
    gold["_rank"] = gold["symptom"].map(rank)

    return (
        gold.sort_values(["note_id", "_rank"])
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


def extract_confidence(raw, long_dataframe):
    long_confidence_column = choose_column(
        long_dataframe.columns,
        ["confidence", "conf_num", "confidence_num"],
    )

    if long_confidence_column:
        return pd.DataFrame(
            {
                "note_id":
                    long_dataframe["note_id"].map(normalize_note_id),
                "symptom":
                    long_dataframe["symptom"].map(normalize_symptom),
                "confidence": pd.to_numeric(
                    long_dataframe[long_confidence_column],
                    errors="coerce",
                ),
            }
        ).drop_duplicates(["note_id", "symptom"])

    note_column = choose_column(
        raw.columns,
        ["NOTE_ID", "note_id"],
    )

    if not note_column:
        return pd.DataFrame(
            columns=["note_id", "symptom", "confidence"]
        )

    pieces = []

    for symptom in SYMPTOMS:
        confidence_column = choose_column(
            raw.columns,
            [
                f"{symptom} Conf_num",
                f"{symptom} confidence",
                f"{symptom}_confidence",
                f"{symptom} confidence_num",
            ],
        )

        if not confidence_column:
            continue

        pieces.append(
            pd.DataFrame(
                {
                    "note_id":
                        raw[note_column].map(normalize_note_id),
                    "symptom": symptom,
                    "confidence": pd.to_numeric(
                        raw[confidence_column],
                        errors="coerce",
                    ),
                }
            )
        )

    if not pieces:
        return pd.DataFrame(
            columns=["note_id", "symptom", "confidence"]
        )

    return (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates(["note_id", "symptom"])
    )


def load_experiment(exp_key, base_dir, gold):
    raw = ec.load_experiment(exp_key, base_dir)
    long_dataframe = ec.to_long(raw, exp_key)

    required = {"note_id", "symptom", "answer"}
    missing = required.difference(long_dataframe.columns)

    if missing:
        raise SystemExit(
            f"{exp_key}: ec.to_long() is missing {sorted(missing)}"
        )

    columns = ["note_id", "symptom", "answer"]

    for optional in ["evidence", "bleu", "note_text"]:
        if optional in long_dataframe.columns:
            columns.append(optional)

    prediction = long_dataframe[columns].copy()
    prediction["note_id"] = prediction["note_id"].map(
        normalize_note_id
    )
    prediction["symptom"] = prediction["symptom"].map(
        normalize_symptom
    )
    prediction["pred"] = prediction["answer"].map(
        normalize_binary
    )

    duplicated = prediction.duplicated(
        ["note_id", "symptom"],
        keep=False,
    )

    if duplicated.any():
        raise SystemExit(
            f"{exp_key}: duplicate note-symptom predictions:\n"
            + prediction.loc[
                duplicated,
                ["note_id", "symptom"],
            ].head(20).to_string(index=False)
        )

    confidence = extract_confidence(raw, prediction)

    merge_columns = [
        column
        for column in [
            "note_id",
            "symptom",
            "pred",
            "evidence",
            "bleu",
            "note_text",
        ]
        if column in prediction.columns
    ]

    aligned = gold.merge(
        prediction[merge_columns],
        on=["note_id", "symptom"],
        how="left",
        validate="one_to_one",
    )

    if not confidence.empty:
        aligned = aligned.merge(
            confidence,
            on=["note_id", "symptom"],
            how="left",
            validate="one_to_one",
        )
    else:
        aligned["confidence"] = np.nan

    if "evidence" not in aligned:
        aligned["evidence"] = ""
    if "bleu" not in aligned:
        aligned["bleu"] = np.nan
    if "note_text" not in aligned:
        aligned["note_text"] = ""

    aligned["bleu"] = pd.to_numeric(
        aligned["bleu"],
        errors="coerce",
    )
    aligned["valid_prediction"] = aligned["pred"].notna()
    aligned["correct"] = (
        aligned["valid_prediction"]
        & aligned["pred"].eq(aligned["gold"])
    )

    return raw, aligned


def build_count_cube(
    aligned,
    notes,
    symptoms,
    valid_mask=None,
    prediction_values=None,
):
    note_lookup = {
        note_id: index
        for index, note_id in enumerate(notes)
    }
    symptom_lookup = {
        symptom: index
        for index, symptom in enumerate(symptoms)
    }

    cube = np.zeros(
        (len(notes), len(symptoms), 4),
        dtype=np.int16,
    )
    valid_grid = np.zeros(
        (len(notes), len(symptoms)),
        dtype=np.int8,
    )

    if valid_mask is None:
        valid_mask = aligned["valid_prediction"].to_numpy()

    if prediction_values is None:
        prediction_values = aligned["pred"].to_numpy(dtype=float)

    valid_mask = np.asarray(valid_mask, dtype=bool)
    prediction_values = np.asarray(prediction_values, dtype=float)
    gold_values = aligned["gold"].to_numpy(dtype=int)
    note_values = aligned["note_id"].to_numpy()
    symptom_values = aligned["symptom"].to_numpy()

    for row_index in np.where(valid_mask)[0]:
        note_index = note_lookup[note_values[row_index]]
        symptom_index = symptom_lookup[symptom_values[row_index]]
        gold_value = int(gold_values[row_index])
        prediction_value = int(prediction_values[row_index])

        cube[
            note_index,
            symptom_index,
            confusion_index(gold_value, prediction_value),
        ] += 1

        valid_grid[note_index, symptom_index] += 1

    return cube, valid_grid


def bootstrap_metrics(cube, valid_grid, bootstrap_weights, gold_per_note):
    n_bootstrap = bootstrap_weights.shape[0]
    n_symptoms = cube.shape[1]

    symptom_counts = (
        bootstrap_weights @ cube.reshape(cube.shape[0], -1)
    ).reshape(n_bootstrap, n_symptoms, 4)

    micro_counts = symptom_counts.sum(axis=1)
    micro = metrics_from_counts(micro_counts)
    symptom_metrics = metrics_from_counts(symptom_counts)

    symptom_support = (
        symptom_counts[..., 0]
        + symptom_counts[..., 2]
    )

    weighted_f1 = safe_divide(
        np.nansum(
            symptom_metrics["f1"] * symptom_support,
            axis=1,
        ),
        symptom_support.sum(axis=1),
    )

    valid_per_note = valid_grid.sum(axis=1)

    results = {
        **micro,
        "coverage": safe_divide(
            bootstrap_weights @ valid_per_note,
            bootstrap_weights @ gold_per_note,
        ),
        "macro_precision": np.nanmean(
            symptom_metrics["precision"],
            axis=1,
        ),
        "macro_recall": np.nanmean(
            symptom_metrics["recall"],
            axis=1,
        ),
        "macro_f1": np.nanmean(
            symptom_metrics["f1"],
            axis=1,
        ),
        "weighted_f1": weighted_f1,
    }

    return results, symptom_metrics


def summarize_experiment(exp_key, aligned, symptoms):
    per_symptom_rows = []

    for symptom in symptoms:
        gold_count = int((aligned["symptom"] == symptom).sum())
        group = aligned[
            (aligned["symptom"] == symptom)
            & aligned["valid_prediction"]
        ]

        counts = np.zeros(4, dtype=int)

        for gold_value, prediction_value in zip(
            group["gold"].astype(int),
            group["pred"].astype(int),
        ):
            counts[
                confusion_index(gold_value, prediction_value)
            ] += 1

        metrics = metrics_from_counts(counts)
        metrics = {
            key: float(np.asarray(value))
            for key, value in metrics.items()
        }

        tp, fp, fn, tn = counts.tolist()

        per_symptom_rows.append(
            {
                "experiment": exp_key,
                "symptom": symptom,
                "n_gold_pairs": gold_count,
                "n_scored": len(group),
                "coverage": len(group) / gold_count,
                "gold_positive": tp + fn,
                "predicted_positive": tp + fp,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                **metrics,
            }
        )

    per_symptom = pd.DataFrame(per_symptom_rows)

    total_counts = per_symptom[
        ["tp", "fp", "fn", "tn"]
    ].sum().to_numpy()

    overall_metrics = metrics_from_counts(total_counts)
    overall_metrics = {
        key: float(np.asarray(value))
        for key, value in overall_metrics.items()
    }

    tp, fp, fn, tn = total_counts.astype(int).tolist()
    support = per_symptom["gold_positive"].to_numpy(dtype=float)

    weighted_f1 = (
        np.nansum(per_symptom["f1"].to_numpy() * support)
        / support.sum()
        if support.sum()
        else np.nan
    )

    exact_results = []

    for _, group in aligned.groupby("note_id", sort=False):
        if (
            len(group) == len(symptoms)
            and group["valid_prediction"].all()
            and set(group["symptom"]) == set(symptoms)
        ):
            exact_results.append(
                bool(
                    (
                        group["pred"].astype(int)
                        == group["gold"].astype(int)
                    ).all()
                )
            )

    subset_accuracy = (
        float(np.mean(exact_results))
        if exact_results
        else np.nan
    )

    overall = {
        "experiment": exp_key,
        "n_gold_pairs": len(aligned),
        "n_scored": int(aligned["valid_prediction"].sum()),
        "missing_predictions": int(
            (~aligned["valid_prediction"]).sum()
        ),
        "coverage": float(aligned["valid_prediction"].mean()),
        "n_notes": aligned["note_id"].nunique(),
        "gold_positive": tp + fn,
        "predicted_positive": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        **overall_metrics,
        "macro_precision": float(
            np.nanmean(per_symptom["precision"])
        ),
        "macro_recall": float(
            np.nanmean(per_symptom["recall"])
        ),
        "macro_f1": float(
            np.nanmean(per_symptom["f1"])
        ),
        "weighted_f1": float(weighted_f1),
        "hamming_loss": float(
            1 - overall_metrics["accuracy"]
        ),
        "complete_notes": len(exact_results),
        "subset_accuracy_complete_notes": subset_accuracy,
    }

    return overall, per_symptom


def confidence_metrics(exp_key, aligned):
    positive = aligned[
        aligned["valid_prediction"]
        & aligned["pred"].eq(1)
        & aligned["confidence"].notna()
    ].copy()

    if positive.empty:
        return pd.DataFrame(), pd.DataFrame()

    positive["correct_positive"] = positive["gold"].eq(1).astype(int)

    summary_rows = []
    bin_rows = []

    scopes = [("ALL", positive)] + [
        (
            symptom,
            positive[positive["symptom"] == symptom],
        )
        for symptom in SYMPTOMS
    ]

    for scope, group in scopes:
        if group.empty:
            continue

        scores = group["confidence"].to_numpy(dtype=float)
        correctness = group[
            "correct_positive"
        ].to_numpy(dtype=int)

        rho = np.nan
        rho_p = np.nan

        if len(group) >= 3 and np.nanstd(scores) > 0:
            correlation = spearmanr(scores, correctness)
            rho = float(correlation.statistic)
            rho_p = float(correlation.pvalue)

        summary_rows.append(
            {
                "experiment": exp_key,
                "scope": scope,
                "n_yes": len(group),
                "overall_precision": float(correctness.mean()),
                "mean_confidence": float(np.nanmean(scores)),
                "percent_confidence_5": float(
                    np.mean(scores == 5)
                ),
                "spearman_rho": rho,
                "spearman_p": rho_p,
                "correctness_auroc": auc_from_scores(
                    correctness,
                    scores,
                ),
            }
        )

        for confidence_level in [1, 2, 3, 4, 5]:
            confidence_group = group[
                group["confidence"] == confidence_level
            ]

            bin_rows.append(
                {
                    "experiment": exp_key,
                    "scope": scope,
                    "confidence": confidence_level,
                    "n": len(confidence_group),
                    "correct": int(
                        confidence_group[
                            "correct_positive"
                        ].sum()
                    ),
                    "precision": (
                        float(
                            confidence_group[
                                "correct_positive"
                            ].mean()
                        )
                        if len(confidence_group)
                        else np.nan
                    ),
                }
            )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(bin_rows),
    )


def grounding_metrics(exp_key, aligned):
    if not hasattr(ec, "grounding_check"):
        return pd.DataFrame()

    positive = aligned[
        aligned["valid_prediction"]
        & aligned["pred"].eq(1)
    ].copy()

    if positive.empty:
        return pd.DataFrame()

    positive["evidence"] = (
        positive["evidence"]
        .fillna("")
        .astype(str)
    )
    positive["nontrivial_evidence"] = ~positive["evidence"].map(
        ec.is_trivial_inference
    )
    positive["correct_positive"] = positive["gold"].eq(1)

    verdicts = []

    for _, row in positive.iterrows():
        if not row["nontrivial_evidence"]:
            verdicts.append("trivial_or_missing")
            continue

        result = ec.grounding_check(
            row["note_text"],
            row["evidence"],
        )
        verdicts.append(str(result.get("verdict", "unknown")))

    positive["grounding_verdict"] = verdicts

    rows = []
    scopes = [("ALL", positive)] + [
        (
            symptom,
            positive[positive["symptom"] == symptom],
        )
        for symptom in SYMPTOMS
    ]

    for scope, group in scopes:
        if group.empty:
            continue

        true_positive_bleu = group.loc[
            group["correct_positive"],
            "bleu",
        ].dropna()

        false_positive_bleu = group.loc[
            ~group["correct_positive"],
            "bleu",
        ].dropna()

        supported = group["grounding_verdict"].isin(
            ["exact", "normalized", "fuzzy"]
        )

        mann_whitney_p = np.nan

        if (
            len(true_positive_bleu)
            and len(false_positive_bleu)
        ):
            mann_whitney_p = float(
                mannwhitneyu(
                    true_positive_bleu,
                    false_positive_bleu,
                    alternative="two-sided",
                ).pvalue
            )

        rows.append(
            {
                "experiment": exp_key,
                "scope": scope,
                "n_positive": len(group),
                "n_nontrivial_evidence": int(
                    group["nontrivial_evidence"].sum()
                ),
                "evidence_coverage": float(
                    group["nontrivial_evidence"].mean()
                ),
                "det_supported_rate": float(supported.mean()),
                "exact_rate": float(
                    group["grounding_verdict"].eq("exact").mean()
                ),
                "normalized_rate": float(
                    group["grounding_verdict"].eq(
                        "normalized"
                    ).mean()
                ),
                "fuzzy_rate": float(
                    group["grounding_verdict"].eq("fuzzy").mean()
                ),
                "mean_bleu": float(group["bleu"].mean()),
                "median_bleu": float(group["bleu"].median()),
                "mean_bleu_tp": (
                    float(true_positive_bleu.mean())
                    if len(true_positive_bleu)
                    else np.nan
                ),
                "mean_bleu_fp": (
                    float(false_positive_bleu.mean())
                    if len(false_positive_bleu)
                    else np.nan
                ),
                "bleu_gap_tp_minus_fp": (
                    float(
                        true_positive_bleu.mean()
                        - false_positive_bleu.mean()
                    )
                    if (
                        len(true_positive_bleu)
                        and len(false_positive_bleu)
                    )
                    else np.nan
                ),
                "bleu_tp_vs_fp_mannwhitney_p":
                    mann_whitney_p,
            }
        )

    return pd.DataFrame(rows)


def llm_judge_metrics(exp_key, gold):
    paths = [
        Path(
            f"judge_results_{exp_key}_full/"
            f"judge_{exp_key}.csv"
        ),
        Path(
            f"judge_results_{exp_key.upper()}_full/"
            f"judge_{exp_key.upper()}.csv"
        ),
    ]

    path = next((item for item in paths if item.exists()), None)

    if path is None:
        return pd.DataFrame()

    dataframe = pd.read_csv(path)

    if not {
        "note_id",
        "symptom",
        "evidence_in_note",
    }.issubset(dataframe.columns):
        return pd.DataFrame()

    dataframe["note_id"] = dataframe["note_id"].map(
        normalize_note_id
    )
    dataframe["symptom"] = dataframe["symptom"].map(
        normalize_symptom
    )

    dataframe = dataframe.merge(
        gold,
        on=["note_id", "symptom"],
        how="left",
        validate="many_to_one",
    )

    if "label" in dataframe:
        dataframe["pred"] = dataframe["label"].map(
            normalize_binary
        )
    else:
        dataframe["pred"] = 1

    dataframe["correct_positive"] = (
        dataframe["gold"].eq(1)
        & dataframe["pred"].eq(1)
    )
    dataframe["judge_in_note"] = dataframe[
        "evidence_in_note"
    ].map({"Yes": True, "No": False})

    for column in [
        "grounding_faithfulness",
        "label_support",
    ]:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    rows = []
    scope_groups = [("ALL", dataframe)] + [
        (
            symptom,
            dataframe[dataframe["symptom"] == symptom],
        )
        for symptom in SYMPTOMS
    ]

    for scope, scope_dataframe in scope_groups:
        for correctness_scope, group in [
            ("ALL", scope_dataframe),
            (
                "TP",
                scope_dataframe[
                    scope_dataframe["correct_positive"]
                ],
            ),
            (
                "FP",
                scope_dataframe[
                    ~scope_dataframe["correct_positive"]
                ],
            ),
        ]:
            if group.empty:
                continue

            rows.append(
                {
                    "experiment": exp_key,
                    "scope": scope,
                    "correctness_scope": correctness_scope,
                    "n": len(group),
                    "parse_rate": (
                        float(group["judge_parse_ok"].mean())
                        if "judge_parse_ok" in group
                        else np.nan
                    ),
                    "evidence_in_note_rate": float(
                        group["judge_in_note"].mean()
                    ),
                    "mean_grounding_faithfulness": float(
                        group[
                            "grounding_faithfulness"
                        ].mean()
                    ),
                    "faithfulness_4_5_rate": float(
                        group[
                            "grounding_faithfulness"
                        ].ge(4).mean()
                    ),
                    "mean_label_support": float(
                        group["label_support"].mean()
                    ),
                    "label_support_4_5_rate": float(
                        group["label_support"].ge(4).mean()
                    ),
                    "label_support_1_2_rate": float(
                        group["label_support"].le(2).mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def write_latex_tables(
    output_directory,
    overall,
    paired,
):
    overall_table = overall[
        [
            "experiment",
            "coverage",
            "precision",
            "recall",
            "f1",
            "macro_f1",
            "specificity",
            "balanced_accuracy",
            "mcc",
        ]
    ].copy()

    overall_table.columns = [
        "Experiment",
        "Coverage",
        "Precision",
        "Recall",
        "$F_1$",
        "Macro-$F_1$",
        "Specificity",
        "Balanced Acc.",
        "MCC",
    ]

    overall_table.to_latex(
        output_directory / "table_overall_metrics.tex",
        index=False,
        escape=False,
        float_format=lambda value: f"{value:.3f}",
        caption="Overall performance across experiments.",
        label="tab:overall_metrics",
    )

    comparison = paired[
        (paired["experiment_a"] == "E7")
        & (paired["experiment_b"] == "MULTI")
    ][
        [
            "metric",
            "estimate_a",
            "estimate_b",
            "delta_b_minus_a",
            "ci_low",
            "ci_high",
            "bootstrap_p_holm",
        ]
    ].copy()

    if not comparison.empty:
        comparison.columns = [
            "Metric",
            "E7",
            "Multi-agent",
            "$\\Delta$",
            "95\\% CI low",
            "95\\% CI high",
            "Holm-adjusted $p$",
        ]

        comparison.to_latex(
            output_directory / "table_e7_vs_multi.tex",
            index=False,
            escape=False,
            float_format=lambda value: f"{value:.4f}",
            caption=(
                "Paired comparison of E7 and the final "
                "multi-agent pipeline."
            ),
            label="tab:e7_multi",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=EXPERIMENTS_DEFAULT,
    )
    parser.add_argument("--base_dir", default=".")
    parser.add_argument(
        "--gold_file",
        default="multiagent_naive_pair_level.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="metrics_E1_E3_E7_MULTI",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=5000,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    experiment_mapping = getattr(ec, "EXPERIMENTS", {})

    missing_experiments = [
        experiment
        for experiment in args.experiments
        if experiment not in experiment_mapping
    ]

    if missing_experiments:
        raise SystemExit(
            "Missing from eval_common.EXPERIMENTS: "
            + ", ".join(missing_experiments)
        )

    gold = load_gold(
        Path(args.base_dir) / args.gold_file
    )

    notes = sorted(gold["note_id"].unique())
    symptoms = [
        symptom
        for symptom in SYMPTOMS
        if symptom in set(gold["symptom"])
    ]

    note_lookup = {
        note_id: index
        for index, note_id in enumerate(notes)
    }

    gold_per_note = np.zeros(
        len(notes),
        dtype=np.int16,
    )

    for note_id, count in gold.groupby("note_id").size().items():
        gold_per_note[note_lookup[note_id]] = int(count)

    random_generator = np.random.default_rng(args.seed)

    bootstrap_weights = random_generator.multinomial(
        len(notes),
        np.full(len(notes), 1 / len(notes)),
        size=args.bootstraps,
    ).astype(np.int16)

    overall_rows = []
    per_symptom_frames = []
    overall_ci_rows = []
    per_symptom_ci_rows = []

    aligned_by_experiment = {}
    bootstrap_by_experiment = {}

    confidence_summary_frames = []
    confidence_bin_frames = []
    grounding_frames = []
    judge_frames = []

    for experiment in args.experiments:
        print(f"Processing {experiment}...")

        raw, aligned = load_experiment(
            experiment,
            args.base_dir,
            gold,
        )

        aligned_by_experiment[experiment] = aligned

        overall, per_symptom = summarize_experiment(
            experiment,
            aligned,
            symptoms,
        )

        overall_rows.append(overall)
        per_symptom_frames.append(per_symptom)

        cube, valid_grid = build_count_cube(
            aligned,
            notes,
            symptoms,
        )

        bootstrap_results, bootstrap_symptom_results = (
            bootstrap_metrics(
                cube,
                valid_grid,
                bootstrap_weights,
                gold_per_note,
            )
        )

        bootstrap_by_experiment[experiment] = bootstrap_results

        for metric in [
            "coverage",
            "precision",
            "recall",
            "f1",
            "specificity",
            "npv",
            "accuracy",
            "balanced_accuracy",
            "mcc",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
        ]:
            low, high = percentile_ci(
                bootstrap_results[metric]
            )

            overall_ci_rows.append(
                {
                    "experiment": experiment,
                    "metric": metric,
                    "estimate": overall[metric],
                    "ci_low": low,
                    "ci_high": high,
                    "n_bootstrap": args.bootstraps,
                    "bootstrap_unit": "note",
                }
            )

        per_symptom_lookup = per_symptom.set_index("symptom")

        for symptom_index, symptom in enumerate(symptoms):
            for metric in [
                "precision",
                "recall",
                "f1",
                "specificity",
                "npv",
                "accuracy",
                "balanced_accuracy",
                "mcc",
            ]:
                low, high = percentile_ci(
                    bootstrap_symptom_results[
                        metric
                    ][:, symptom_index]
                )

                per_symptom_ci_rows.append(
                    {
                        "experiment": experiment,
                        "symptom": symptom,
                        "metric": metric,
                        "estimate":
                            per_symptom_lookup.loc[
                                symptom,
                                metric,
                            ],
                        "ci_low": low,
                        "ci_high": high,
                        "n_bootstrap": args.bootstraps,
                        "bootstrap_unit": "note",
                    }
                )

        confidence_summary, confidence_bins = (
            confidence_metrics(
                experiment,
                aligned,
            )
        )

        if not confidence_summary.empty:
            confidence_summary_frames.append(
                confidence_summary
            )

        if not confidence_bins.empty:
            confidence_bin_frames.append(confidence_bins)

        grounding = grounding_metrics(
            experiment,
            aligned,
        )

        if not grounding.empty:
            grounding_frames.append(grounding)

        judge = llm_judge_metrics(
            experiment,
            gold,
        )

        if not judge.empty:
            judge_frames.append(judge)

    overall_dataframe = pd.DataFrame(overall_rows)

    per_symptom_dataframe = pd.concat(
        per_symptom_frames,
        ignore_index=True,
    )

    overall_ci_dataframe = pd.DataFrame(
        overall_ci_rows
    )

    per_symptom_ci_dataframe = pd.DataFrame(
        per_symptom_ci_rows
    )

    paired_rows = []
    mcnemar_rows = []

    paired_metrics = [
        "precision",
        "recall",
        "f1",
        "specificity",
        "npv",
        "accuracy",
        "balanced_accuracy",
        "mcc",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]

    for experiment_a, experiment_b in itertools.combinations(
        args.experiments,
        2,
    ):
        aligned_a = aligned_by_experiment[experiment_a]
        aligned_b = aligned_by_experiment[experiment_b]

        key_columns = ["note_id", "symptom", "gold"]

        if not aligned_a[key_columns].equals(
            aligned_b[key_columns]
        ):
            raise SystemExit(
                f"Pair alignment differs between "
                f"{experiment_a} and {experiment_b}."
            )

        common_scored = (
            aligned_a["valid_prediction"].to_numpy()
            & aligned_b["valid_prediction"].to_numpy()
        )

        predictions_a = aligned_a["pred"].to_numpy(dtype=float)
        predictions_b = aligned_b["pred"].to_numpy(dtype=float)
        gold_values = aligned_a["gold"].to_numpy(dtype=int)

        cube_a, valid_grid = build_count_cube(
            aligned_a,
            notes,
            symptoms,
            valid_mask=common_scored,
            prediction_values=predictions_a,
        )

        cube_b, _ = build_count_cube(
            aligned_b,
            notes,
            symptoms,
            valid_mask=common_scored,
            prediction_values=predictions_b,
        )

        bootstrap_a, _ = bootstrap_metrics(
            cube_a,
            valid_grid,
            bootstrap_weights,
            gold_per_note,
        )

        bootstrap_b, _ = bootstrap_metrics(
            cube_b,
            valid_grid,
            bootstrap_weights,
            gold_per_note,
        )

        counts_a = cube_a.sum(axis=(0, 1))
        counts_b = cube_b.sum(axis=(0, 1))

        actual_a = {
            key: float(np.asarray(value))
            for key, value in metrics_from_counts(
                counts_a
            ).items()
        }

        actual_b = {
            key: float(np.asarray(value))
            for key, value in metrics_from_counts(
                counts_b
            ).items()
        }

        symptom_counts_a = cube_a.sum(axis=0)
        symptom_counts_b = cube_b.sum(axis=0)

        symptom_metrics_a = metrics_from_counts(
            symptom_counts_a
        )
        symptom_metrics_b = metrics_from_counts(
            symptom_counts_b
        )

        support_a = (
            symptom_counts_a[:, 0]
            + symptom_counts_a[:, 2]
        )
        support_b = (
            symptom_counts_b[:, 0]
            + symptom_counts_b[:, 2]
        )

        actual_a.update(
            {
                "macro_precision": float(
                    np.nanmean(
                        symptom_metrics_a["precision"]
                    )
                ),
                "macro_recall": float(
                    np.nanmean(
                        symptom_metrics_a["recall"]
                    )
                ),
                "macro_f1": float(
                    np.nanmean(symptom_metrics_a["f1"])
                ),
                "weighted_f1": float(
                    np.nansum(
                        symptom_metrics_a["f1"]
                        * support_a
                    )
                    / support_a.sum()
                ),
            }
        )

        actual_b.update(
            {
                "macro_precision": float(
                    np.nanmean(
                        symptom_metrics_b["precision"]
                    )
                ),
                "macro_recall": float(
                    np.nanmean(
                        symptom_metrics_b["recall"]
                    )
                ),
                "macro_f1": float(
                    np.nanmean(symptom_metrics_b["f1"])
                ),
                "weighted_f1": float(
                    np.nansum(
                        symptom_metrics_b["f1"]
                        * support_b
                    )
                    / support_b.sum()
                ),
            }
        )

        for metric in paired_metrics:
            delta_bootstrap = (
                bootstrap_b[metric]
                - bootstrap_a[metric]
            )

            low, high = percentile_ci(delta_bootstrap)

            paired_rows.append(
                {
                    "experiment_a": experiment_a,
                    "experiment_b": experiment_b,
                    "n_common_scored_pairs": int(
                        common_scored.sum()
                    ),
                    "metric": metric,
                    "estimate_a": actual_a[metric],
                    "estimate_b": actual_b[metric],
                    "delta_b_minus_a":
                        actual_b[metric]
                        - actual_a[metric],
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_p":
                        bootstrap_pvalue(
                            delta_bootstrap
                        ),
                    "n_bootstrap": args.bootstraps,
                }
            )

        correctness_a = (
            predictions_a[common_scored].astype(int)
            == gold_values[common_scored]
        )

        correctness_b = (
            predictions_b[common_scored].astype(int)
            == gold_values[common_scored]
        )

        a_correct_b_wrong = int(
            np.sum(correctness_a & ~correctness_b)
        )

        a_wrong_b_correct = int(
            np.sum(~correctness_a & correctness_b)
        )

        discordant = (
            a_correct_b_wrong
            + a_wrong_b_correct
        )

        mcnemar_p = (
            float(
                binomtest(
                    a_correct_b_wrong,
                    n=discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
            if discordant
            else np.nan
        )

        mcnemar_rows.append(
            {
                "experiment_a": experiment_a,
                "experiment_b": experiment_b,
                "n_common_scored_pairs": int(
                    common_scored.sum()
                ),
                "a_correct_b_wrong":
                    a_correct_b_wrong,
                "a_wrong_b_correct":
                    a_wrong_b_correct,
                "discordant_pairs": discordant,
                "mcnemar_exact_p": mcnemar_p,
            }
        )

    paired_dataframe = pd.DataFrame(paired_rows)

    paired_dataframe["bootstrap_p_holm"] = np.nan

    for metric, indices in paired_dataframe.groupby(
        "metric"
    ).groups.items():
        paired_dataframe.loc[
            indices,
            "bootstrap_p_holm",
        ] = holm_adjust(
            paired_dataframe.loc[
                indices,
                "bootstrap_p",
            ].to_numpy()
        )

    mcnemar_dataframe = pd.DataFrame(mcnemar_rows)

    mcnemar_dataframe["mcnemar_p_holm"] = holm_adjust(
        mcnemar_dataframe[
            "mcnemar_exact_p"
        ].to_numpy()
    )

    overall_dataframe.to_csv(
        output_directory / "overall_metrics.csv",
        index=False,
    )

    overall_ci_dataframe.to_csv(
        output_directory / "overall_bootstrap_ci.csv",
        index=False,
    )

    per_symptom_dataframe.to_csv(
        output_directory / "per_symptom_metrics.csv",
        index=False,
    )

    per_symptom_ci_dataframe.to_csv(
        output_directory / "per_symptom_bootstrap_ci.csv",
        index=False,
    )

    paired_dataframe.to_csv(
        output_directory
        / "paired_bootstrap_comparisons.csv",
        index=False,
    )

    mcnemar_dataframe.to_csv(
        output_directory / "mcnemar_tests.csv",
        index=False,
    )

    if confidence_summary_frames:
        pd.concat(
            confidence_summary_frames,
            ignore_index=True,
        ).to_csv(
            output_directory
            / "confidence_yes_summary.csv",
            index=False,
        )

    if confidence_bin_frames:
        pd.concat(
            confidence_bin_frames,
            ignore_index=True,
        ).to_csv(
            output_directory
            / "confidence_yes_bins.csv",
            index=False,
        )

    if grounding_frames:
        pd.concat(
            grounding_frames,
            ignore_index=True,
        ).to_csv(
            output_directory
            / "deterministic_grounding_summary.csv",
            index=False,
        )

    if judge_frames:
        pd.concat(
            judge_frames,
            ignore_index=True,
        ).to_csv(
            output_directory / "llm_judge_summary.csv",
            index=False,
        )

    write_latex_tables(
        output_directory,
        overall_dataframe,
        paired_dataframe,
    )

    metadata = {
        "experiments": args.experiments,
        "gold_file": args.gold_file,
        "n_gold_pairs": len(gold),
        "n_gold_notes": gold["note_id"].nunique(),
        "symptoms": symptoms,
        "bootstrap_replicates": args.bootstraps,
        "bootstrap_unit": "note",
        "seed": args.seed,
        "primary_metric_scope":
            "clinician-gold-scored pairs with valid predictions",
        "paired_metric_scope":
            "pairs scored by both compared experiments",
        "paired_delta_definition":
            "experiment_b minus experiment_a",
    }

    (
        output_directory / "run_metadata.json"
    ).write_text(json.dumps(metadata, indent=2))

    print("\nOverall metrics")
    print(
        overall_dataframe[
            [
                "experiment",
                "n_scored",
                "coverage",
                "tp",
                "fp",
                "fn",
                "tn",
                "precision",
                "recall",
                "f1",
                "macro_f1",
                "specificity",
                "balanced_accuracy",
                "mcc",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nSaved all outputs to: "
        f"{output_directory}"
    )


if __name__ == "__main__":
    main()
