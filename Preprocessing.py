import pandas as pd

notes_df = pd.read_csv("rebuilt_notes_by_noteid.csv")

total          = len(notes_df)
n_patients     = notes_df["PAT_ID"].nunique() if "PAT_ID" in notes_df.columns else "N/A"
n_empty        = (notes_df["Clean_note_text"].str.strip() == "").sum()
n_very_short   = (notes_df["Clean_note_text"].str.len() < 50).sum()
n_usable       = (notes_df["Clean_note_text"].str.len() >= 50).sum()
avg_len        = int(notes_df["Clean_note_text"].str.len().mean())
max_len        = int(notes_df["Clean_note_text"].str.len().max())
min_len        = int(notes_df["Clean_note_text"].str.len().min())

print("=" * 60)
print("NOTE QUALITY CHECK — before Experiment 1")
print("=" * 60)
print(f"  Total rows in CSV:        {total:>6,}")
print(f"  Unique patients:          {n_patients:>6}")
print(f"  Empty notes:              {n_empty:>6,}  ← should be 0")
print(f"  Very short (<50 chars):   {n_very_short:>6,}  ← will be dropped")
print(f"  USABLE notes (≥50 chars): {n_usable:>6,}  ← Experiment 1 input")
print(f"  Avg note length:          {avg_len:>6,} chars")
print(f"  Min note length:          {min_len:>6,} chars")
print(f"  Max note length:          {max_len:>6,} chars")

if "PAT_ID" in notes_df.columns:
    npp = notes_df.groupby("PAT_ID").size()
    print(f"\n  Notes per patient:")
    print(f"    Min:    {npp.min()}")
    print(f"    Median: {npp.median():.0f}")
    print(f"    Max:    {npp.max()}")
    print(f"    Mean:   {npp.mean():.1f}")

lengths = notes_df["Clean_note_text"].str.len()
print(f"\n  Length distribution:")
print(f"    < 100 chars:          {(lengths < 100).sum():>5,}")
print(f"    100 – 500 chars:      {((lengths >= 100) & (lengths < 500)).sum():>5,}")
print(f"    500 – 2000 chars:     {((lengths >= 500) & (lengths < 2000)).sum():>5,}")
print(f"    2000 – 5000 chars:    {((lengths >= 2000) & (lengths < 5000)).sum():>5,}")
print(f"    5000 – 12000 chars:   {((lengths >= 5000) & (lengths < 12000)).sum():>5,}")
print(f"    > 12000 chars:        {(lengths >= 12000).sum():>5,}  ← may need truncation")

print(f"\n  Missing values:")
for col in ["PAT_ID", "PAT_ENC_CSN_ID", "NOTE_ID",
            "DATE_OF_SERVIC_DTTM", "Clean_note_text"]:
    if col in notes_df.columns:
        n_null = notes_df[col].isna().sum()
        print(f"    {col:<28} {n_null:>5,} null")

if "DATE_OF_SERVIC_DTTM" in notes_df.columns:
    notes_df["DATE_OF_SERVIC_DTTM"] = pd.to_datetime(
        notes_df["DATE_OF_SERVIC_DTTM"], errors="coerce"
    )
    date_min = notes_df["DATE_OF_SERVIC_DTTM"].min()
    date_max = notes_df["DATE_OF_SERVIC_DTTM"].max()
    print(f"\n  Date range: {date_min.date()} → {date_max.date()}")

print(f"\n{'─'*60}")
print("SAMPLE NOTE (longest usable note):")
print(f"{'─'*60}")
longest_idx = lengths.idxmax()
sample = notes_df.loc[longest_idx]
print(f"PAT_ID:    {sample.get('PAT_ID', 'N/A')}")
print(f"NOTE_ID:   {sample.get('NOTE_ID', 'N/A')}")
print(f"Length:    {len(str(sample['Clean_note_text'])):,} chars")
print(f"\nFirst 500 chars:")
print(str(sample["Clean_note_text"])[:500])

print(f"\n{'='*60}")
print(f"EXPERIMENT 1 WILL PROCESS: {n_usable:,} notes")
if n_very_short > 0:
    print(f"NOTE: {n_very_short} notes under 50 chars will be dropped in Exp 1 setup")
if (lengths > 12000).sum() > 0:
    print(f"NOTE: {(lengths > 12000).sum()} notes exceed 12,000 chars — "
          f"consider setting MAX_NOTE_CHARS = 12000")
print("=" * 60)
