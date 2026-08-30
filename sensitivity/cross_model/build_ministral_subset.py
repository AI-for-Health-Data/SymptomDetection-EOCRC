from pathlib import Path
import re
import numpy as np
import pandas as pd

SEED = 314159

PRIMARY_ROOT = Path(
    "runs/verge_final_primary_20260822"
)

OUT = Path(
    "runs/verge_m2_ministral3_14b_20260824/subset"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def norm_name(x):
    return (
        str(x)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def find_col(df, names):
    lookup = {
        norm_name(c): c
        for c in df.columns
    }

    for name in names:
        key = norm_name(name)

        if key in lookup:
            return lookup[key]

    return None


def label01(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip().lower()

    if s in {
        "1", "1.0", "yes", "true",
        "positive", "present"
    }:
        return 1

    if s in {
        "0", "0.0", "no", "false",
        "negative", "absent"
    }:
        return 0

    raise ValueError(
        f"Unrecognized label {x!r}"
    )

candidates = []

for p in PRIMARY_ROOT.rglob("*.csv"):

    try:
        header = pd.read_csv(
            p,
            nrows=0,
        )
    except Exception:
        continue

    pat = find_col(
        header,
        ["PAT_ID", "patient_id"],
    )

    note = find_col(
        header,
        ["NOTE_ID", "note_id"],
    )

    symptom = find_col(
        header,
        ["Symptom", "feature", "target_symptom"],
    )

    gold = find_col(
        header,
        ["Gold", "gold_label", "reference_label"],
    )

    extractor = find_col(
        header,
        ["Extractor", "extractor_prediction"],
    )

    verge = find_col(
        header,
        [
            "final_loop_prediction",
            "VERGE",
            "verge_prediction",
            "final_prediction",
        ],
    )

    if all(
        [
            pat,
            note,
            symptom,
            gold,
            extractor,
            verge,
        ]
    ):
        candidates.append(
            (
                p,
                pat,
                note,
                symptom,
                gold,
                extractor,
                verge,
            )
        )


if not candidates:
    raise RuntimeError(
        "Could not identify finalized VERGE CSV."
    )


selected = None

for item in candidates:

    p, pat, note, symptom, gold, extractor, verge = item

    d = pd.read_csv(
        p,
        low_memory=False,
    )

    if d[gold].notna().sum() != 4033:
        continue

    tmp = d[
        [pat, note, symptom, gold, extractor, verge]
    ].copy()

    try:
        g = tmp[gold].map(label01)
        e = tmp[extractor].map(label01)
        v = tmp[verge].map(label01)
    except Exception:
        continue

    labeled = g.notna()

    changes = (
        e[labeled].astype(int)
        != v[labeled].astype(int)
    ).sum()

    if changes == 133:
        selected = item
        break


if selected is None:
    raise RuntimeError(
        "No candidate reproduced the frozen "
        "4033-pair / 133-change primary result."
    )


p, pat, note, symptom, gold, extractor, verge = selected

print("PRIMARY FILE:", p)

df = pd.read_csv(
    p,
    low_memory=False,
)

df = df[
    [pat, note, symptom, gold, extractor, verge]
].copy()

df.columns = [
    "PAT_ID",
    "NOTE_ID",
    "Symptom",
    "Gold_raw",
    "Extractor_raw",
    "Llama_VERGE_raw",
]

df = df[
    df["Gold_raw"].notna()
].copy()

df["Gold"] = (
    df["Gold_raw"]
    .map(label01)
    .astype(int)
)

df["Extractor"] = (
    df["Extractor_raw"]
    .map(label01)
    .astype(int)
)

df["Llama_VERGE"] = (
    df["Llama_VERGE_raw"]
    .map(label01)
    .astype(int)
)

assert len(df) == 4033

assert not df.duplicated(
    ["PAT_ID", "NOTE_ID", "Symptom"]
).any()


df["changed"] = (
    df["Extractor"]
    != df["Llama_VERGE"]
)

df["reference_improving"] = (
    (df["Extractor"] != df["Gold"])
    &
    (df["Llama_VERGE"] == df["Gold"])
)

df["reference_degrading"] = (
    (df["Extractor"] == df["Gold"])
    &
    (df["Llama_VERGE"] != df["Gold"])
)

df["shared_error"] = (
    (df["Extractor"] != df["Gold"])
    &
    (df["Llama_VERGE"] == df["Extractor"])
)


assert int(df["changed"].sum()) == 133
assert int(df["reference_improving"].sum()) == 98
assert int(df["reference_degrading"].sum()) == 35


def keyset(frame):
    return set(
        zip(
            frame["PAT_ID"].astype(str),
            frame["NOTE_ID"].astype(str),
            frame["Symptom"].astype(str),
        )
    )

parts = []

for i, (symptom_name, group) in enumerate(
    df.groupby(
        "Symptom",
        sort=True,
    )
):

    prevalence = float(
        group["Gold"].mean()
    )

    n_pos = int(
        round(50 * prevalence)
    )

    n_neg = 50 - n_pos

    positives = group[
        group["Gold"] == 1
    ].sample(
        n=n_pos,
        random_state=SEED + i * 10,
    )

    negatives = group[
        group["Gold"] == 0
    ].sample(
        n=n_neg,
        random_state=SEED + i * 10 + 1,
    )

    parts.append(
        pd.concat(
            [positives, negatives]
        )
    )


representative = pd.concat(
    parts
)

assert len(representative) == 350

changed = df[
    df["changed"]
].copy()

assert len(changed) == 133

shared_pool = df[
    df["shared_error"]
].copy()

assert len(shared_pool) >= 100

shared_sample = shared_pool.sample(
    n=100,
    random_state=SEED + 999,
)


rep_keys = keyset(representative)
changed_keys = keyset(changed)
shared_keys = keyset(shared_sample)

union_keys = (
    rep_keys
    | changed_keys
    | shared_keys
)


def row_key(row):
    return (
        str(row["PAT_ID"]),
        str(row["NOTE_ID"]),
        str(row["Symptom"]),
    )


manifest = df[
    df.apply(
        lambda r:
            row_key(r) in union_keys,
        axis=1,
    )
].copy()


manifest["in_representative_350"] = (
    manifest.apply(
        lambda r:
            row_key(r) in rep_keys,
        axis=1,
    )
)

manifest["in_all_changed_133"] = (
    manifest.apply(
        lambda r:
            row_key(r) in changed_keys,
        axis=1,
    )
)

manifest["in_shared_error_100"] = (
    manifest.apply(
        lambda r:
            row_key(r) in shared_keys,
        axis=1,
    )
)


manifest["change_category"] = "unchanged"

manifest.loc[
    manifest["reference_improving"],
    "change_category",
] = "reference_improving"

manifest.loc[
    manifest["reference_degrading"],
    "change_category",
] = "reference_degrading"


manifest = manifest.sort_values(
    [
        "Symptom",
        "PAT_ID",
        "NOTE_ID",
    ]
).reset_index(drop=True)


columns = [
    "PAT_ID",
    "NOTE_ID",
    "Symptom",
    "Gold",
    "Extractor",
    "Llama_VERGE",
    "changed",
    "reference_improving",
    "reference_degrading",
    "shared_error",
    "in_representative_350",
    "in_all_changed_133",
    "in_shared_error_100",
    "change_category",
]


manifest[
    columns
].to_csv(
    OUT
    / "m2_ministral_subset_manifest.csv",
    index=False,
)


summary = (
    "VERGE M2 MINISTRAL-3-14B SUBSET\n"
    f"seed={SEED}\n"
    f"primary_file={p}\n"
    f"representative={int(manifest['in_representative_350'].sum())}\n"
    f"all_changed={int(manifest['in_all_changed_133'].sum())}\n"
    f"shared_error={int(manifest['in_shared_error_100'].sum())}\n"
    f"unique_union={len(manifest)}\n"
    f"reference_improving_changed={int(changed['reference_improving'].sum())}\n"
    f"reference_degrading_changed={int(changed['reference_degrading'].sum())}\n"
)


with open(
    OUT
    / "m2_ministral_subset_summary.txt",
    "w",
) as f:
    f.write(summary)

    f.write(
        "\nRepresentative distribution:\n"
    )

    f.write(
        representative.groupby(
            ["Symptom", "Gold"]
        ).size().to_string()
    )

    f.write("\n")


print(summary)

print(
    "WROTE:",
    OUT
    / "m2_ministral_subset_manifest.csv",
)
