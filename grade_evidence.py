import re
import math
from collections import Counter

GRADE_HIGH_KEYWORDS = [
    "systematic review", "meta-analysis", "meta analysis",
    "cochrane review", "cochrane database",
    "pooled analysis", "pooled data",
    "randomized controlled trial", "randomised controlled trial",
    "rct", "randomized trial", "randomised trial",
    "double-blind", "double blind", "placebo-controlled",
    "phase iii", "phase 3",
]

GRADE_MODERATE_KEYWORDS = [
    "clinical trial", "controlled trial", "phase ii", "phase 2",
    "open-label", "single-blind", "non-inferiority",
    "cohort study", "prospective cohort", "retrospective cohort",
    "longitudinal study", "population-based study",
    "large-scale study", "multicenter study", "multicentre study",
    "registry-based", "nationwide study",
    "observational study", "cross-sectional",
]

GRADE_LOW_KEYWORDS = [
    "case-control", "case control",
    "case series", "retrospective analysis",
    "narrative review", "literature review",
    "review article", "review of the literature",
    "small study", "pilot study",
    "single-center", "single-centre", "single center",
    "uncontrolled", "non-randomized", "non-randomised",
]

GRADE_VERY_LOW_KEYWORDS = [
    "case report", "case presentation",
    "expert opinion", "expert consensus",
    "editorial", "letter to the editor", "correspondence",
    "commentary", "hypothesis", "theoretical",
    "animal study", "in vitro", "cell line",
    "anecdotal", "personal experience",
]


def classify_passage_grade(text: str) -> str:
    text_lower = text.lower()

    for keyword in GRADE_HIGH_KEYWORDS:
        if keyword in text_lower:
            return "HIGH"

    for keyword in GRADE_MODERATE_KEYWORDS:
        if keyword in text_lower:
            return "MODERATE"

    for keyword in GRADE_VERY_LOW_KEYWORDS:
        if keyword in text_lower:
            return "VERY_LOW"

    for keyword in GRADE_LOW_KEYWORDS:
        if keyword in text_lower:
            return "LOW"

    return "LOW"


def tag_passages_with_grade(passages: list) -> list:
    for p in passages:
        p["grade"] = classify_passage_grade(p.get("text", ""))
    return passages


def sort_passages_by_grade(passages: list) -> list:
    grade_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "VERY_LOW": 3}
    return sorted(
        passages,
        key=lambda p: (grade_order.get(p.get("grade", "LOW"), 2),
                       -p.get("score", 0))
    )


def format_passages_with_grade(passages: list, max_passages: int = 5) -> str:
    if not passages:
        return "  No relevant passages retrieved."

    # Sort by GRADE then score
    sorted_passages = sort_passages_by_grade(passages)[:max_passages]

    lines = []
    for i, p in enumerate(sorted_passages):
        grade = p.get("grade", "LOW")
        text = p.get("text", "")[:250]
        lines.append(f"  [Evidence {i+1} | GRADE: {grade}] {text}")

    return "\n".join(lines)


def filter_very_low(passages: list) -> list:
    return [p for p in passages if p.get("grade") != "VERY_LOW"]


GROUNDING_HIGH     = 0.66   
GROUNDING_MODERATE = 0.33   

def grounding_grade(bleu: float) -> str:
    if bleu >= GROUNDING_HIGH:
        return "HIGH"
    if bleu >= GROUNDING_MODERATE:
        return "MODERATE"
    return "LOW"


def bleu_no_bp(hypothesis: str, reference: str, max_n: int = 4) -> float:
    _tok = re.compile(r"\w+|\S")
    hyp_tokens = _tok.findall(hypothesis.lower())
    ref_tokens = _tok.findall(reference.lower())
    if not hyp_tokens:
        return 0.0

    precisions = []
    for n in range(1, min(max_n, len(hyp_tokens)) + 1):
        hyp_ng = Counter(
            tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1))
        ref_ng = Counter(
            tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1))
        clipped = sum(min(c, ref_ng[ng]) for ng, c in hyp_ng.items())
        total = sum(hyp_ng.values())
        if total == 0 or clipped == 0:
            return 0.0
        precisions.append(clipped / total)

    return math.exp(sum(math.log(p) for p in precisions) / len(precisions))


def grade_summary(passages: list) -> dict:
    counts = {"HIGH": 0, "MODERATE": 0, "LOW": 0, "VERY_LOW": 0}
    for p in passages:
        grade = p.get("grade", "LOW")
        if grade in counts:
            counts[grade] += 1
    counts["total"] = sum(counts.values())
    return counts


if __name__ == "__main__":
    test_passages = [
        {"text": "A systematic review of 15 randomized controlled trials "
                 "found that colonoscopy screening reduces CRC mortality by 30%.",
         "score": 0.85},
        {"text": "In this case report, a 32-year-old male presented with "
                 "Lynch syndrome and early-onset colorectal cancer.",
         "score": 0.72},
        {"text": "A large prospective cohort study of 50,000 patients showed "
                 "that family history of CRC doubles the risk of early onset.",
         "score": 0.68},
        {"text": "The authors hypothesize that germline mutations in MLH1 and "
                 "MSH2 may contribute to hereditary colorectal cancer risk.",
         "score": 0.55},
        {"text": "NCCN guidelines recommend colonoscopy every 1-2 years for "
                 "patients with Lynch syndrome starting at age 20-25.",
         "score": 0.90},
    ]

    tagged = tag_passages_with_grade(test_passages)
    for p in tagged:
        print(f"  GRADE={p['grade']:<10} score={p['score']:.2f}  {p['text'][:80]}...")

    print(f"\n  Summary: {grade_summary(tagged)}")

    print(f"\n  Formatted for prompt:")
    print(format_passages_with_grade(tagged))

    # Test grounding GRADE
    print(f"\n  Grounding GRADE tests:")
    for bleu_val in [0.85, 0.50, 0.20, 0.05, 0.0]:
        print(f"    BLEU={bleu_val:.2f} → {grounding_grade(bleu_val)}")
