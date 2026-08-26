import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import medspacy
from medspacy.ner import TargetRule

from sage_common import DIRECT_ALIASES

GOLD = Path(
    "runs/verge_agent1_grade_ready_v2/"
    "agent1_gold_scored_pairs.csv"
)

NOTES = Path("rebuilt_notes_by_noteid.csv")

UMLS = Path("all_umls_concepts.csv")

OUT = Path(
    "runs/medspacy_context_baseline_20260824"
)
OUT.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "abdominal pain",
    "rectal bleeding",
    "rectal pain",
    "diarrhea",
    "constipation",
    "weight loss",
    "family history of colorectal cancer",
]

def nid(x):
    x = str(x or "").strip()
    return x[:-2] if x.endswith(".0") else x


def norm(x):
    x = str(x or "").strip().lower()
    x = x.replace("_", " ")
    x = re.sub(r"\s+", " ", x)
    return x


def label01(x):
    x = norm(x)

    if x in {
        "1", "1.0", "yes", "true", "positive"
    }:
        return 1

    if x in {
        "0", "0.0", "no", "false", "negative"
    }:
        return 0

    return None


def safe_ext(ent, name):
    try:
        return bool(getattr(ent._, name))
    except Exception:
        return False


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

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else np.nan
    )

    accuracy = (
        (tp + tn) / (tp + fp + fn + tn)
    )

    balanced_accuracy = (
        (recall + specificity) / 2
    )

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

notes = pd.read_csv(
    NOTES,
    dtype=str,
    keep_default_na=False,
)

gold["PAT_KEY"] = gold["PAT_ID"].map(nid)
gold["NOTE_KEY"] = gold["NOTE_ID"].map(nid)
gold["FEATURE_KEY"] = gold["Symptom"].map(norm)
gold["gold"] = gold["Gold"].map(label01)

gold = gold[
    gold["gold"].notna()
].copy()

assert len(gold) == 4033, (
    f"Expected 4033 labeled pairs; got {len(gold)}"
)

notes["PAT_KEY"] = notes["PAT_ID"].map(nid)
notes["NOTE_KEY"] = notes["NOTE_ID"].map(nid)

notes = (
    notes[
        [
            "PAT_KEY",
            "NOTE_KEY",
            "Clean_note_text",
        ]
    ]
    .drop_duplicates(
        ["PAT_KEY", "NOTE_KEY"]
    )
    .copy()
)

print("Labeled pairs :", len(gold))
print("Unique notes  :", len(notes))

terms_by_feature = {
    f: set([f])
    for f in FEATURES
}


alias_norm = {
    norm(k): v
    for k, v in DIRECT_ALIASES.items()
}

for feature in FEATURES:

    aliases = alias_norm.get(
        feature,
        [],
    )

    if isinstance(aliases, str):
        aliases = [aliases]

    for term in aliases:
        term = norm(term)

        if len(term) >= 3:
            terms_by_feature[
                feature
            ].add(term)


if UMLS.exists():

    umls = pd.read_csv(
        UMLS,
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "symptom_group",
        "umls_name",
    }

    missing = required - set(umls.columns)

    if missing:
        raise RuntimeError(
            f"Missing UMLS columns: {missing}"
        )

    for _, r in umls.iterrows():

        feature = norm(
            r["symptom_group"]
        )

        term = norm(
            r["umls_name"]
        )

        if (
            feature in terms_by_feature
            and len(term) >= 3
        ):
            terms_by_feature[
                feature
            ].add(term)

else:
    raise FileNotFoundError(
        "all_umls_concepts.csv not found"
    )

term_features = defaultdict(set)

for feature, terms in terms_by_feature.items():
    for term in terms:
        term_features[term].add(feature)


ambiguous_terms = {
    term
    for term, features in term_features.items()
    if len(features) > 1
}


for feature in FEATURES:

    canonical = feature

    terms_by_feature[feature] = {
        t
        for t in terms_by_feature[feature]
        if (
            t == canonical
            or t not in ambiguous_terms
        )
    }


print("\nVocabulary sizes:")

for feature in FEATURES:
    print(
        f"{feature:40s}",
        len(terms_by_feature[feature]),
    )

print(
    "\nAmbiguous shared terms excluded:",
    len(ambiguous_terms),
)

nlp = medspacy.load()

print("\nmedspaCy pipes:")
print(nlp.pipe_names)

if "medspacy_target_matcher" not in nlp.pipe_names:
    raise RuntimeError(
        "medspacy_target_matcher not loaded"
    )

if "medspacy_context" not in nlp.pipe_names:
    raise RuntimeError(
        "medspacy_context not loaded"
    )


target_matcher = nlp.get_pipe(
    "medspacy_target_matcher"
)


# Category -> task feature
category_to_feature = {}


FAMILY_FEATURE = (
    "family history of colorectal cancer"
)

family_lexical_cues = re.compile(
    r"\b("
    r"family|familial|relative|"
    r"mother|father|parent|"
    r"sister|brother|sibling|"
    r"grandmother|grandfather|"
    r"grandparent|aunt|uncle|"
    r"daughter|son"
    r")\b",
    re.I,
)


rules = []

for i, feature in enumerate(FEATURES):

    for j, term in enumerate(
        sorted(
            terms_by_feature[feature],
            key=lambda x: (-len(x), x),
        )
    ):

        if feature == FAMILY_FEATURE:

            if family_lexical_cues.search(term):
                category = "FH_DIRECT"

            else:
                category = "FH_CRC_CONTEXT"

        else:
            category = (
                "SYM_"
                + re.sub(
                    r"[^A-Z0-9]+",
                    "_",
                    feature.upper(),
                ).strip("_")
            )

        category_to_feature[
            category
        ] = feature

        rules.append(
            TargetRule(
                term,
                category,
            )
        )


target_matcher.add(rules)

print(
    "Target rules added:",
    len(rules),
)

note_predictions = []

for idx, row in notes.iterrows():

    text = str(
        row["Clean_note_text"] or ""
    )

    doc = nlp(text)

    pred = {
        feature: 0
        for feature in FEATURES
    }

    evidence = {
        feature: []
        for feature in FEATURES
    }

    for ent in doc.ents:

        category = ent.label_

        if category not in category_to_feature:
            continue

        feature = category_to_feature[
            category
        ]

        negated = safe_ext(
            ent,
            "is_negated",
        )

        uncertain = safe_ext(
            ent,
            "is_uncertain",
        )

        hypothetical = safe_ext(
            ent,
            "is_hypothetical",
        )

        family = safe_ext(
            ent,
            "is_family",
        )


        if (
            negated
            or uncertain
            or hypothetical
        ):
            continue


        if feature != FAMILY_FEATURE:
            if family:
                continue

        else:

            if (
                category
                == "FH_CRC_CONTEXT"
                and not family
            ):
                continue


        pred[feature] = 1

        if len(
            evidence[feature]
        ) < 5:
            evidence[
                feature
            ].append(ent.text)


    for feature in FEATURES:

        note_predictions.append({
            "PAT_KEY":
                row["PAT_KEY"],

            "NOTE_KEY":
                row["NOTE_KEY"],

            "FEATURE_KEY":
                feature,

            "medspacy_prediction":
                pred[feature],

            "medspacy_evidence":
                " || ".join(
                    evidence[feature]
                ),
        })


pred = pd.DataFrame(
    note_predictions
)


df = gold[
    [
        "PAT_KEY",
        "NOTE_KEY",
        "FEATURE_KEY",
        "gold",
    ]
].merge(
    pred,
    on=[
        "PAT_KEY",
        "NOTE_KEY",
        "FEATURE_KEY",
    ],
    how="left",
    validate="one_to_one",
)


if df[
    "medspacy_prediction"
].isna().any():
    bad = df[
        df[
            "medspacy_prediction"
        ].isna()
    ]

    raise RuntimeError(
        f"{len(bad)} labeled pairs missing predictions"
    )


df[
    "medspacy_prediction"
] = df[
    "medspacy_prediction"
].astype(int)

overall = compute_metrics(
    df["gold"],
    df["medspacy_prediction"],
)

print("\n" + "=" * 100)
print("MEDSPACY + CONTEXT BASELINE")
print("=" * 100)

print(
    json.dumps(
        overall,
        indent=2,
    )
)


rows = []

for feature, g in df.groupby(
    "FEATURE_KEY"
):

    m = compute_metrics(
        g["gold"],
        g["medspacy_prediction"],
    )

    rows.append({
        "feature": feature,
        **m,
    })


per_feature = pd.DataFrame(
    rows
).sort_values(
    "feature"
)

print("\nPER-SYMPTOM:")
print(
    per_feature.to_string(
        index=False
    )
)


df.to_csv(
    OUT /
    "medspacy_context_predictions.csv",
    index=False,
)

per_feature.to_csv(
    OUT /
    "medspacy_context_per_symptom.csv",
    index=False,
)


with open(
    OUT /
    "medspacy_context_overall.json",
    "w",
) as f:

    json.dump(
        {
            "system":
                "medspaCy TargetMatcher + ConText",

            "medspacy_version":
                getattr(
                    medspacy,
                    "__version__",
                    "unknown",
                ),

            "n_labeled_pairs":
                int(len(df)),

            "terminology":
                (
                    "Frozen VERGE direct aliases "
                    "+ UMLS-derived terms"
                ),

            "gold_tuning":
                False,

            "uses_llm":
                False,

            "uses_rag":
                False,

            "uses_verge_verifier":
                False,

            "uses_verge_refiner":
                False,

            "historical_mentions_excluded":
                False,

            "overall":
                overall,
        },
        f,
        indent=2,
    )


with open(
    OUT /
    "medspacy_vocabulary.txt",
    "w",
) as f:

    for feature in FEATURES:

        f.write(
            f"\n[{feature}]\n"
        )

        for term in sorted(
            terms_by_feature[feature]
        ):
            f.write(
                term + "\n"
            )


print("\nSaved to:")
print(OUT)
