from __future__ import annotations
import argparse
import re
import sys
import pandas as pd
import numpy as np

JOIN_KEY = "NOTE_ID"


SYMPTOM_COLS = {
    "Abdominal Pain ":                      "Abdominal pain",
    "Rectal Bleeding":                      "Rectal bleeding",
    "Rectal Pain ":                         "Rectal pain",
    "Diarrhea":                             "Diarrhea",
    "Constipation ":                        "Constipation",
    "Weight Loss":                          "Weight loss",
    "Family History of Colorectal Cancer":  "Family history of colorectal cancer",
}
FH_RAW = "Family History of Colorectal Cancer"

POS_TOKENS = ("yes", "y", "1", "1.0", "true", "t", "present", "positive", "+")
NEG_TOKENS = ("no", "n", "0", "0.0", "false", "f", "absent", "negative", "-")
CRC_PAT = re.compile(r"colorectal|colon|rectal|crc|bowel")


def norm(v: str) -> str:
    return re.sub(r"\s+", " ", str(v).strip().lower())


def cell_polarity(raw_value: str, is_fh: bool, fh_mode: str):
    """Return 1 (positive), 0 (negative), or np.nan (blank/unknown) for one cell."""
    s = norm(raw_value)
    if s in ("", "nan", "none"):
        return np.nan
    if s in NEG_TOKENS:
        return 0
    if s in POS_TOKENS:
        return 1
    if is_fh:
        if fh_mode == "strict":
            return 1 if CRC_PAT.search(s) else 0   
        return 1                                    
    return 1


def collapse_series(polarities: pd.Series, rule: str):
    """Collapse a note's per-line polarities (with NaN for blanks) to one label."""
    vals = polarities.dropna().tolist()
    if not vals:
        return np.nan
    if rule == "any_pos":
        return 1 if any(v == 1 for v in vals) else 0
    if rule == "first":
        return int(vals[0])
    if rule == "last":
        return int(vals[-1])
    if rule == "majority":
        pos = sum(v == 1 for v in vals)
        neg = sum(v == 0 for v in vals)
        return 1 if pos >= neg else 0          # tie -> positive
    raise ValueError(f"unknown collapse rule: {rule}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="new_data_labels.csv")
    ap.add_argument("--exp",  default="rebuilt_notes_by_noteid.csv")
    ap.add_argument("--collapse", default="any_pos",
                    choices=["any_pos", "last", "first", "majority"])
    ap.add_argument("--fh_mode", default="lenient",
                    choices=["lenient", "strict"])
    ap.add_argument("--out", default="gold_clean_by_noteid.csv")
    args = ap.parse_args()

    gold = pd.read_csv(args.gold, dtype=str, keep_default_na=False)
    if "LINE" not in gold.columns:
        gold["LINE"] = 1

    missing = [c for c in SYMPTOM_COLS if c not in gold.columns]
    if missing:
        sys.exit(f"ERROR: gold file missing expected columns: {missing}\n"
                 f"Found: {list(gold.columns)}")

    # drop EXACT duplicate rows (pure export duplication) 
    dup_subset = [JOIN_KEY, "LINE"] + list(SYMPTOM_COLS.keys())
    before = len(gold)
    gold = gold.drop_duplicates(subset=dup_subset, keep="first").copy()
    print(f"[1] dropped {before - len(gold)} exact-duplicate rows "
          f"({before} -> {len(gold)})")

    # per-cell polarity 
    for raw, short in SYMPTOM_COLS.items():
        is_fh = (raw == FH_RAW)
        gold[f"__pol__{short}"] = gold[raw].map(
            lambda v: cell_polarity(v, is_fh, args.fh_mode))

    # collapse to one row per NOTE_ID 
    # sort by LINE so first/last are meaningful
    gold["LINE_n"] = pd.to_numeric(gold["LINE"], errors="coerce").fillna(0)
    gold = gold.sort_values([JOIN_KEY, "LINE_n"])

    records = []
    n_conflict = {short: 0 for short in SYMPTOM_COLS.values()}
    for nid, grp in gold.groupby(JOIN_KEY, sort=False):
        rec = {JOIN_KEY: nid}
        for short in SYMPTOM_COLS.values():
            pol = grp[f"__pol__{short}"]
            # count notes whose non-blank lines disagree (for the audit log)
            nun = pol.dropna().nunique()
            if nun > 1:
                n_conflict[short] += 1
            rec[short] = collapse_series(pol, args.collapse)
        records.append(rec)

    clean = pd.DataFrame(records)

    # align to experiment note set 
    exp = pd.read_csv(args.exp, dtype={JOIN_KEY: str})
    exp_ids = set(exp[JOIN_KEY].astype(str).str.strip())
    clean[JOIN_KEY] = clean[JOIN_KEY].astype(str).str.strip()
    n_before = len(clean)
    clean = clean[clean[JOIN_KEY].isin(exp_ids)].reset_index(drop=True)

    print(f"[2] collapse rule = '{args.collapse}', fh_mode = '{args.fh_mode}'")
    print(f"[3] notes with cross-line label conflicts (pre-collapse):")
    for short, c in n_conflict.items():
        print(f"      {short:38s} {c:4d}")
    print(f"[4] aligned to experiment set: {n_before} -> {len(clean)} notes "
          f"(exp has {len(exp_ids)})")

    miss = exp_ids - set(clean[JOIN_KEY])
    if miss:
        print(f"    WARNING: {len(miss)} experiment notes have NO gold row!")

    # positive counts per symptom 
    print("\n[5] gold POSITIVE counts per symptom (this configuration):")
    for short in SYMPTOM_COLS.values():
        pos = int((clean[short] == 1).sum())
        neg = int((clean[short] == 0).sum())
        nan = int(clean[short].isna().sum())
        print(f"      {short:38s} pos={pos:4d}  neg={neg:4d}  blank={nan:3d}")

    clean.to_csv(args.out, index=False)
    print(f"\nSaved cleaned gold -> {args.out}")
    print("Columns:", list(clean.columns))


if __name__ == "__main__":
    main()
