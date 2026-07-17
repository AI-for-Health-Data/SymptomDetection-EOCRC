from __future__ import annotations
import argparse
import os
import re
import sys
import json
import numpy as np
import pandas as pd


SYMPTOMS = [
    "Abdominal pain", "Rectal bleeding", "Rectal pain",
    "Diarrhea", "Constipation", "Weight loss",
    "Family history of colorectal cancer",
]

EXP_FILES = {
    "E1": "experiment1_note_level_metrics.csv",
    "E2": "exp_aliases_only_note_level_metrics.csv",
    "E3": "exp_ros_only_note_level_metrics.csv",
    "E4": "experiment4_note_level_metrics.csv",
    "E5": "experiment3_note_level_metrics.csv",
    "E6": "exp6_note_level_metrics.csv",
    "E7": "exp7_note_level_metrics.csv",
    "E8": "exp8v2_note_level_metrics.csv",
    "E9": "exp9_ce_note_level_metrics.csv",
    "E3b": "exp_ros_section_note_level_metrics.csv",
}
EXP_ORDER = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]

JOIN_KEY = "NOTE_ID"
GOLD_FH_HINT = "family history of colorectal cancer"


def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def map_label(value, blank_policy="missing"):
    if value is None:
        return 0 if blank_policy == "no" else np.nan
    s = str(value).strip().lower()
    if s in ("", "nan", "none", "n/a", "na", "not assessed", "not evaluated",
             "unknown", "?"):
        return 0 if blank_policy == "no" else np.nan
    if s in ("yes", "y", "1", "1.0", "true", "t", "present", "positive", "+"):
        return 1
    if s.startswith("y"):
        return 1
    if s in ("no", "n", "0", "0.0", "false", "f", "absent", "negative", "-",
             "denies", "denied"):
        return 0
    if s.startswith("n"):
        return 0
    return 1


def resolve_symptom_columns(df_cols, symptoms):
    norm_to_actual = {}
    for c in df_cols:
        norm_to_actual.setdefault(norm_key(c), c)
    mapping, missing = {}, []
    for sym in symptoms:
        k = norm_key(sym)
        if k in norm_to_actual:
            mapping[sym] = norm_to_actual[k]
        else:
            missing.append(sym)
    return mapping, missing


def _is_precleaned(gold, log):
    canonical = set(SYMPTOMS)
    if not canonical.issubset(set(gold.columns)):
        return False
    if gold[JOIN_KEY].duplicated().sum() != 0:
        return False
    sample_col = SYMPTOMS[0]
    vals = set(gold[sample_col].astype(str).str.strip().str.lower().unique())
    if not vals.issubset({"0", "1", "0.0", "1.0", "", "nan", "none"}):
        log.append("[gold] canonical columns + unique keys found, but cells are "
                   "not 0/1 -- treating as RAW export (will collapse + map).")
        return False
    return True


def load_gold(path, sheet, blank_policy, log):
    if path.lower().endswith((".xlsx", ".xls")):
        gold = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
        gold = gold.astype(str)
    else:
        gold = pd.read_csv(path, dtype=str, keep_default_na=False)

    log.append(f"[gold] loaded {path}  rows={len(gold)}  cols={len(gold.columns)}")

    if JOIN_KEY not in gold.columns:
        nk = {norm_key(c): c for c in gold.columns}
        if norm_key(JOIN_KEY) in nk:
            gold = gold.rename(columns={nk[norm_key(JOIN_KEY)]: JOIN_KEY})
        else:
            sys.exit(f"FATAL: gold file has no {JOIN_KEY} column. "
                     f"Columns: {list(gold.columns)}")
    gold[JOIN_KEY] = (gold[JOIN_KEY].astype(str)
                      .str.replace(r"\.0$", "", regex=True).str.strip())

    if _is_precleaned(gold, log):
        log.append("[gold] PRE-CLEANED file detected: 1 row/NOTE_ID, canonical "
                   "0/1 columns. Skipping internal collapse + label mapping "
                   "(trusting clean_gold_labels.py output).")
        g = pd.DataFrame({JOIN_KEY: gold[JOIN_KEY].values})
        for sym in SYMPTOMS:
            num = pd.to_numeric(gold[sym], errors="coerce")
            g[sym] = num.map(lambda x: np.nan if pd.isna(x) else int(x))
        for sym in SYMPTOMS:
            pos = int((g[sym] == 1).sum()); neg = int((g[sym] == 0).sum())
            mis = int(g[sym].isna().sum())
            log.append(f"[gold] {sym:38s} pos={pos:4d} neg={neg:4d} missing={mis:4d}")
        return g

    log.append("[gold] RAW export mode: collapsing keep='last' + mapping labels. "
               "WARNING: lossy for notes whose lines disagree -- prefer "
               "clean_gold_labels.py output.")
    dups = gold[JOIN_KEY].duplicated().sum()
    if dups:
        log.append(f"[gold] WARNING: {dups} duplicate {JOIN_KEY} rows; keeping last")
        gold = gold.drop_duplicates(subset=[JOIN_KEY], keep="last")

    colmap, missing = resolve_symptom_columns(gold.columns, SYMPTOMS)
    if missing:
        sys.exit(f"FATAL: gold file missing symptom columns for: {missing}\n"
                 f"Available columns: {list(gold.columns)}")
    log.append("[gold] symptom column map: " +
               json.dumps({s: colmap[s] for s in SYMPTOMS}, ensure_ascii=False))

    g = pd.DataFrame({JOIN_KEY: gold[JOIN_KEY].values})
    raw_value_counts = {}
    for sym in SYMPTOMS:
        col = colmap[sym]
        mapped = gold[col].map(lambda v: map_label(v, blank_policy))
        g[sym] = mapped.values
        vc = gold[col].astype(str).str.strip().str.lower().value_counts().head(8)
        raw_value_counts[sym] = vc.to_dict()
    log.append("[gold] raw value distributions (top): " +
               json.dumps(raw_value_counts, ensure_ascii=False)[:1500])

    for sym in SYMPTOMS:
        pos = int((g[sym] == 1).sum()); neg = int((g[sym] == 0).sum())
        mis = int(g[sym].isna().sum())
        log.append(f"[gold] {sym:38s} pos={pos:4d} neg={neg:4d} missing={mis:4d}")
    return g


def load_predictions(exp, base_dir, log):
    path = os.path.join(base_dir, EXP_FILES[exp])
    if not os.path.exists(path):
        log.append(f"[pred] {exp}: FILE NOT FOUND {path} -- skipping")
        return None
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    if JOIN_KEY not in df.columns:
        nk = {norm_key(c): c for c in df.columns}
        if norm_key(JOIN_KEY) in nk:
            df = df.rename(columns={nk[norm_key(JOIN_KEY)]: JOIN_KEY})
        else:
            log.append(f"[pred] {exp}: no {JOIN_KEY} column -- skipping")
            return None

    df[JOIN_KEY] = (df[JOIN_KEY].astype(str)
                    .str.replace(r"\.0$", "", regex=True).str.strip())
    df = df.drop_duplicates(subset=[JOIN_KEY], keep="last")

    colmap, missing = resolve_symptom_columns(df.columns, SYMPTOMS)
    if missing:
        log.append(f"[pred] {exp}: missing prediction columns {missing} -- skipping")
        return None

    p = pd.DataFrame({JOIN_KEY: df[JOIN_KEY].values})
    for sym in SYMPTOMS:
        p[sym] = df[colmap[sym]].map(lambda v: map_label(v, "missing")).values
    log.append(f"[pred] {exp}: rows={len(p)}")
    return p


def confusion(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = (2 * prec * rec / (prec + rec)
          if (prec == prec and rec == rec and (prec + rec) > 0) else np.nan)
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else np.nan
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, n=tp + fp + fn + tn,
                precision=prec, recall=rec, f1=f1,
                specificity=spec, accuracy=acc)


def evaluate(gold, base_dir, out_dir, log, min_join_rate=0.5):
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    fh_crosstab_rows = []

    for exp in EXP_ORDER:
        pred = load_predictions(exp, base_dir, log)
        if pred is None:
            continue

        merged = pred.merge(gold, on=JOIN_KEY, how="inner",
                            suffixes=("_pred", "_gold"))
        join_rate = len(merged) / max(len(pred), 1)
        log.append(f"[join] {exp}: pred={len(pred)} gold={len(gold)} "
                   f"joined={len(merged)} ({join_rate:.1%})")
        if join_rate < min_join_rate:
            log.append(f"[join] {exp}: WARNING join rate < {min_join_rate:.0%} "
                       f"-- inspect before trusting.")

        for sym in SYMPTOMS:
            yt = merged[f"{sym}_gold"].to_numpy(dtype=float)
            yp = merged[f"{sym}_pred"].to_numpy(dtype=float)
            mask = ~np.isnan(yt) & ~np.isnan(yp)
            yt2, yp2 = yt[mask].astype(int), yp[mask].astype(int)
            if len(yt2) == 0:
                continue
            m = confusion(yt2, yp2)
            m.update(exp=exp, symptom=sym, n_scored=int(len(yt2)),
                     n_excluded=int((~mask).sum()))
            all_rows.append(m)

            if norm_key(sym) == GOLD_FH_HINT:
                pos_pred = (yp2 == 1)
                tp_fh = int(np.sum(pos_pred & (yt2 == 1)))
                fp_fh = int(np.sum(pos_pred & (yt2 == 0)))
                fh_crosstab_rows.append(dict(
                    exp=exp, fh_pos_predictions=int(pos_pred.sum()),
                    gold_confirmed_TP=tp_fh, gold_false_FP=fp_fh,
                    fp_rate_among_positives=(fp_fh / pos_pred.sum()
                                             if pos_pred.sum() else np.nan)))

    res = pd.DataFrame(all_rows)
    if res.empty:
        sys.exit("No experiments scored. Check files / join key.")

    res = res[["exp", "symptom", "n_scored", "n_excluded",
               "tp", "fp", "fn", "tn",
               "precision", "recall", "f1", "specificity", "accuracy"]]
    res.to_csv(os.path.join(out_dir, "gold_per_symptom.csv"), index=False)

    summ_rows = []
    for exp in EXP_ORDER:
        sub = res[res["exp"] == exp]
        if sub.empty:
            continue
        TP, FP, FN, TN = sub[["tp", "fp", "fn", "tn"]].sum()
        micro_p = TP / (TP + FP) if (TP + FP) else np.nan
        micro_r = TP / (TP + FN) if (TP + FN) else np.nan
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                    if (micro_p == micro_p and micro_r == micro_r
                        and micro_p + micro_r > 0) else np.nan)
        summ_rows.append(dict(
            exp=exp,
            macro_precision=sub["precision"].mean(),
            macro_recall=sub["recall"].mean(),
            macro_f1=sub["f1"].mean(),
            micro_precision=micro_p, micro_recall=micro_r, micro_f1=micro_f1,
            total_TP=int(TP), total_FP=int(FP),
            total_FN=int(FN), total_TN=int(TN)))
    summ = pd.DataFrame(summ_rows)
    summ.to_csv(os.path.join(out_dir, "gold_summary.csv"), index=False)

    fh = pd.DataFrame(fh_crosstab_rows)
    if not fh.empty:
        fh.to_csv(os.path.join(out_dir, "gold_family_history_contamination.csv"),
                  index=False)

    with open(os.path.join(out_dir, "gold_eval_log.txt"), "w") as f:
        f.write("\n".join(log))

    print("\n" + "=" * 70)
    print("GOLD-STANDARD SUMMARY (macro / micro F1 per experiment)")
    print("=" * 70)
    summ_show = summ.copy()
    for c in summ_show.columns:
        if summ_show[c].dtype.kind == "f":
            summ_show[c] = summ_show[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "nan")
    print(summ_show.to_string(index=False))
    if not fh.empty:
        print("\n" + "=" * 70)
        print("FAMILY-HISTORY: were 'contaminated' positives actually FALSE?")
        print("=" * 70)
        print(fh.to_string(index=False))
    print(f"\nWrote: {out_dir}/gold_per_symptom.csv, gold_summary.csv, "
          f"gold_family_history_contamination.csv, gold_eval_log.txt")
    return res, summ, fh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="clinician CSV/XLSX (raw or pre-cleaned)")
    ap.add_argument("--sheet", default=None, help="xlsx sheet name/index")
    ap.add_argument("--base_dir", default=".", help="dir with *_note_level_metrics.csv")
    ap.add_argument("--blank", choices=["missing", "no"], default="missing",
                    help="raw mode only: how to treat blank gold cells")
    ap.add_argument("--out_dir", default="gold_results")
    ap.add_argument("--min_join_rate", type=float, default=0.5)
    args = ap.parse_args()

    log = []
    log.append("=== gold_eval run ===")
    log.append(f"gold file: {args.gold}")
    log.append(f"blank policy (raw mode only): {args.blank}")
    try:
        sheet = int(args.sheet) if (args.sheet and args.sheet.isdigit()) else args.sheet
    except Exception:
        sheet = args.sheet

    gold = load_gold(args.gold, sheet, args.blank, log)
    evaluate(gold, args.base_dir, args.out_dir, log, args.min_join_rate)


if __name__ == "__main__":
    main()
