import argparse
import pandas as pd


def norm_yes(x) -> bool:
    return str(x).strip().lower().startswith("y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics",  required=True)
    ap.add_argument("--verified", required=True)
    ap.add_argument("--keep_col", default="final_keep",
                    help="Column to use as keep decision: "
                         "'final_keep' (after adjudicator) or "
                         "'verifier_keep' (verifier only)")
    ap.add_argument("--id_col",   default="NOTE_ID")
    ap.add_argument("--verified_id_col", default="note_id")
    ap.add_argument("--out",      default="postverify_note_level_metrics.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.metrics)
    vf = pd.read_csv(a.verified)

    if a.id_col not in df.columns:
        raise SystemExit(f"[ERROR] '{a.id_col}' not in metrics; "
                         f"have {list(df.columns)[:12]}")
    vid = a.verified_id_col if a.verified_id_col in vf.columns else a.id_col
    if vid not in vf.columns:
        raise SystemExit(f"[ERROR] no id column in verified; "
                         f"have {list(vf.columns)}")

    # Determine keep column
    keep_col = a.keep_col
    if keep_col not in vf.columns:
        # Fallback
        for fallback in ["final_keep", "verifier_keep"]:
            if fallback in vf.columns:
                keep_col = fallback
                break
        else:
            raise SystemExit(f"[ERROR] no keep column found in verified file")
    print(f"Using keep column: '{keep_col}'")

    # Identify rejections
    rej = vf[~vf[keep_col].astype(str).str.lower().isin(["true", "1", "yes"])]

    # Count GRADE rejections if available
    grade_rej = 0
    if "grade_rejected" in vf.columns:
        grade_rej = int(vf["grade_rejected"].astype(str)
                        .str.lower().isin(["true","1"]).sum())

    print(f"Total verdicts: {len(vf)}")
    print(f"Rejections to apply: {len(rej)}")
    if grade_rej:
        print(f"  (of which {grade_rej} by GRADE gate)")

    # Build flip map
    flip_by_note = {}
    for _, r in rej.iterrows():
        flip_by_note.setdefault(str(r[vid]), set()).add(r["symptom"])

    df[a.id_col] = df[a.id_col].astype(str)
    n_flips = 0; n_already_no = 0; n_missing = 0

    for i, row in df.iterrows():
        syms = flip_by_note.get(row[a.id_col])
        if not syms: continue
        for s in syms:
            if s not in df.columns:
                n_missing += 1; continue
            if norm_yes(row[s]):
                df.at[i, s] = "No"
                n_flips += 1
            else:
                n_already_no += 1

    df.to_csv(a.out, index=False)
    print(f"\nFlipped {n_flips} labels Yes → No.")
    if n_already_no: print(f"  ({n_already_no} already No)")
    if n_missing:    print(f"  [WARN] {n_missing} symptom columns missing")
    print(f"Wrote {a.out}")
    print(f"\nTo score:")
    print(f"  mkdir -p gold_multiagent")
    print(f"  cp {a.out} gold_multiagent/exp7_note_level_metrics.csv")
    print(f"  python gold_eval.py --gold gold_clean_anypos_lenient.csv \\")
    print(f"      --base_dir gold_multiagent --out_dir gold_multiagent_results")


if __name__ == "__main__":
    main()
