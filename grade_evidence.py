import re
from collections import Counter

GRADE_HIGH_KEYWORDS = [
    "systematic review",
    "meta-analysis",
    "meta analysis",
    "cochrane review",
    "cochrane database",
    "pooled analysis",
    "pooled data",
    "randomized controlled trial",
    "randomised controlled trial",
    "rct",
    "randomized trial",
    "randomised trial",
    "double-blind",
    "double blind",
    "placebo-controlled",
    "phase iii",
    "phase 3",
]

GRADE_MODERATE_KEYWORDS = [
    "clinical trial",
    "controlled trial",
    "phase ii",
    "phase 2",
    "open-label",
    "single-blind",
    "non-inferiority",
    "cohort study",
    "prospective cohort",
    "retrospective cohort",
    "longitudinal study",
    "population-based study",
    "large-scale study",
    "multicenter study",
    "multicentre study",
    "registry-based",
    "nationwide study",
    "observational study",
    "cross-sectional",
]

GRADE_LOW_KEYWORDS = [
    "case-control",
    "case control",
    "case series",
    "retrospective analysis",
    "narrative review",
    "literature review",
    "review article",
    "review of the literature",
    "small study",
    "pilot study",
    "single-center",
    "single-centre",
    "single center",
    "uncontrolled",
    "non-randomized",
    "non-randomised",
]

GRADE_VERY_LOW_KEYWORDS = [
    "case report",
    "case presentation",
    "expert opinion",
    "expert consensus",
    "editorial",
    "letter to the editor",
    "correspondence",
    "commentary",
    "hypothesis",
    "theoretical",
    "animal study",
    "in vitro",
    "cell line",
    "anecdotal",
    "personal experience",
]

SOURCE_TIER_ORDER = {
    "HIGH": 0,
    "MODERATE": 1,
    "UNCLASSIFIED": 2,
    "LOW": 3,
    "VERY_LOW": 4,
}

ALL_TIERS = list(SOURCE_TIER_ORDER.keys())


def keyword_present(text: str, keyword: str) -> bool:
    """Match a study-design keyword using phrase boundaries."""
    text = str(text or "").lower()
    keyword = str(keyword or "").lower().strip()
    if not keyword:
        return False
    pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
    return bool(re.search(pattern, text))


def contains_any_keyword(text: str, keywords) -> bool:
    return any(keyword_present(text, keyword) for keyword in keywords)


def classify_passage_grade(text: str) -> str:
    """
    Assign a conservative GRADE-inspired source-design tier.

    VERY_LOW and LOW indicators override stronger designs, because an
    affirmative weak-design marker is the signal we most want to act
    on. HIGH is checked before MODERATE so that a specific phrase such
    as "randomized controlled trial" is not swallowed by the broader
    "controlled trial" keyword.

    Returns UNCLASSIFIED when no design marker is present. This is an
    absence of evidence about the design, not evidence of a weak
    design, and the caller must decide how to treat it.
    """
    text = str(text or "")

    if contains_any_keyword(text, GRADE_VERY_LOW_KEYWORDS):
        return "VERY_LOW"
    if contains_any_keyword(text, GRADE_LOW_KEYWORDS):
        return "LOW"
    if contains_any_keyword(text, GRADE_HIGH_KEYWORDS):
        return "HIGH"
    if contains_any_keyword(text, GRADE_MODERATE_KEYWORDS):
        return "MODERATE"

    return "UNCLASSIFIED"


def classify_abstract_grades(abstracts: dict) -> dict:
    """
    Preferred path when parent-abstract text is available.

    abstracts: {abstract_id: full_abstract_text}
    returns:   {abstract_id: tier}

    Design keywords live in the abstract's methods sentences, so
    classifying at the abstract level and inheriting the tier down to
    that abstract's chunks yields far fewer UNCLASSIFIED passages than
    classifying each sentence chunk in isolation.
    """
    return {
        abstract_id: classify_passage_grade(text)
        for abstract_id, text in abstracts.items()
    }


def tag_passages_with_grade(passages, abstract_tiers=None):
    """
    Add a source-design tier to every retrieved passage.

    If abstract_tiers is supplied and a passage carries an
    "abstract_id" (or "pmid") key, the parent abstract's tier is used.
    Otherwise the tier is derived from the chunk text itself.
    """
    tagged = []
    for passage in passages:
        copied = dict(passage)
        tier = None
        if abstract_tiers:
            parent = copied.get("abstract_id", copied.get("pmid"))
            if parent is not None:
                tier = abstract_tiers.get(parent)
        if tier is None:
            tier = classify_passage_grade(copied.get("text", ""))
        copied["grade"] = tier
        tagged.append(copied)
    return tagged


def sort_passages_by_grade(passages):
    """Sort by source tier, then by descending retrieval score."""
    return sorted(
        passages,
        key=lambda passage: (
            SOURCE_TIER_ORDER.get(
                passage.get("grade", "UNCLASSIFIED"),
                SOURCE_TIER_ORDER["UNCLASSIFIED"],
            ),
            -float(passage.get("score", 0.0)),
        ),
    )


def format_passages_with_grade(
    passages,
    max_passages: int = 5,
    label: str = "SOURCE TIER",
) -> str:
    """
    Format source-tier-tagged passages for the LLM prompt.

    `label` controls the prefix shown to the model. The default is
    "SOURCE TIER" rather than "GRADE" so the prompt does not imply a
    formal GRADE certainty rating.
    """
    if not passages:
        return "  No relevant passages retrieved."

    selected = sort_passages_by_grade(passages)[:max_passages]
    lines = []
    for index, passage in enumerate(selected, start=1):
        tier = passage.get("grade", "UNCLASSIFIED")
        text = str(passage.get("text", ""))[:250]
        lines.append(f"  [Evidence {index} | {label}: {tier}] {text}")
    return "\n".join(lines)


def tier_distribution(passages) -> dict:
    """Convenience: count tiers across a passage list."""
    counts = Counter(p.get("grade", "UNCLASSIFIED") for p in passages)
    return {tier: counts.get(tier, 0) for tier in ALL_TIERS}


if __name__ == "__main__":
    test_passages = [
        {"text": "A systematic review of randomized controlled trials "
                 "evaluated screening.", "score": 0.85},
        {"text": "This case report describes a patient with Lynch "
                 "syndrome.", "score": 0.90},
        {"text": "A prospective cohort study evaluated family history "
                 "of colorectal cancer.", "score": 0.75},
        {"text": "Rectal bleeding was the most common presenting "
                 "symptom in patients under 50.", "score": 0.80},
    ]
    tagged = tag_passages_with_grade(test_passages)
    for passage in tagged:
        print(f"{passage['grade']:<13} {passage['score']}  "
              f"{passage['text'][:60]}")
    print()
    print("distribution:", tier_distribution(tagged))
    print()
    print(format_passages_with_grade(tagged))
