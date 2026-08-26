from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List

from sage_common import DIRECT_ALIASES, find_phrase_spans

ELIGIBLE_POSITIVE_ASSERTIONS = {
    "AFFIRMED",
    "HISTORICAL",
}

NEGATION_PATTERNS = [
    re.compile(r"\bno\s+history\s+of\b", re.I),
    re.compile(r"\bcurrently\s+does\s+not\s+have\b", re.I),
    re.compile(r"\bdoes\s+not\s+have\b", re.I),
    re.compile(r"\bnegative\s+for\b", re.I),
    re.compile(r"\bdenies?\b", re.I),
    re.compile(r"\bdenied\b", re.I),
    re.compile(r"\bwithout\b", re.I),
    re.compile(r"\bnot\b", re.I),
    re.compile(r"\bno\b", re.I),
    re.compile(r"\babsent\b", re.I),
    re.compile(r"\bfree\s+of\b", re.I),
    re.compile(r"\blacks?\b", re.I),
]

POST_NEGATION_PATTERNS = [
    re.compile(r"\bnot\s+present\b", re.I),
    re.compile(r"\babsent\b", re.I),
    re.compile(r"\bdenied\b", re.I),
    re.compile(r"\bnegative\b", re.I),
]

HYPOTHETICAL_PATTERNS = [
    re.compile(r"\bpossible\b", re.I),
    re.compile(r"\bpossibly\b", re.I),
    re.compile(r"\bmay\s+have\b", re.I),
    re.compile(r"\bmight\s+have\b", re.I),
    re.compile(r"\bcould\s+have\b", re.I),
    re.compile(r"\bsuspected\b", re.I),
    re.compile(r"\bsuspicion\s+of\b", re.I),
    re.compile(r"\brule\s+out\b", re.I),
    re.compile(r"\br/o\b", re.I),
    re.compile(r"\bif\s+(?:he|she|they|patient|pt)\s+develops?\b", re.I),
]

PLANNED_PATTERNS = [
    re.compile(r"\bplan(?:ned)?\s+to\b", re.I),
    re.compile(r"\bwill\s+(?:evaluate|assess|screen|test)\s+for\b", re.I),
    re.compile(r"\bscreen(?:ing)?\s+for\b", re.I),
]

HISTORICAL_PATTERNS = [
    re.compile(r"\bhx\b", re.I),
    re.compile(r"\bh/o\b", re.I),
    re.compile(r"\bhistory\s+of\b", re.I),
    re.compile(r"\bprevious(?:ly)?\b", re.I),
    re.compile(r"\bprior\b", re.I),
    re.compile(r"\bpast\b", re.I),
    re.compile(r"\bresolved\b", re.I),
    re.compile(r"\bwas\s+treated\s+for\b", re.I),
]

CONTRAST_MARKERS = (
    " but ",
    " however ",
    " except ",
    " although ",
    " yet ",
)

FAMILY_RELATION_PATTERNS = [
    re.compile(
        r"\b(?:mother|father|parent|sister|brother|sibling|"
        r"grandmother|grandfather|aunt|uncle|son|daughter|relative|"
        r"family member|first[- ]degree relative)\b",
        re.I,
    ),
    re.compile(r"\b(?:mgf|mgm|pgf|pgm)\b", re.I),
    re.compile(
        r"\b(?:maternal|paternal)\s+"
        r"(?:mother|father|grandmother|grandfather|aunt|uncle)\b",
        re.I,
    ),
]

CRC_PATTERNS = [
    re.compile(r"\bcolorectal\s+(?:cancer|carcinoma|ca)\b", re.I),
    re.compile(r"\bcolon\s+(?:cancer|carcinoma|ca)\b", re.I),
    re.compile(r"\brectal\s+(?:cancer|carcinoma|ca)\b", re.I),
    re.compile(r"\bbowel\s+(?:cancer|carcinoma|ca)\b", re.I),
    re.compile(r"\bcrc\b", re.I),
]

def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.lower().strip()
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return re.sub(r"\s+", " ", text)


def evidence_grounded_exact_or_normalized(
    evidence: str,
    note_text: str,
) -> bool:
    evidence = str(evidence or "").strip()
    note_text = str(note_text or "")

    if not evidence or not note_text:
        return False

    if evidence in note_text:
        return True

    e = normalize_text(evidence)
    n = normalize_text(note_text)

    return bool(e) and e in n


def has_family_relation(text: str) -> bool:
    value = str(text or "")
    return any(p.search(value) for p in FAMILY_RELATION_PATTERNS)


def has_crc_type(text: str) -> bool:
    value = str(text or "")
    return any(p.search(value) for p in CRC_PATTERNS)


def _trim_left_scope(
    text: str,
    start: int,
    max_chars: int = 140,
) -> str:
    lo = max(0, start - max_chars)
    prefix = text[lo:start]

    boundary = max(
        prefix.rfind("\n"),
        prefix.rfind("."),
        prefix.rfind(";"),
    )
    if boundary >= 0:
        prefix = prefix[boundary + 1:]

    lower = prefix.lower()
    best = -1
    best_len = 0

    for marker in CONTRAST_MARKERS:
        pos = lower.rfind(marker)
        if pos > best:
            best = pos
            best_len = len(marker)

    if best >= 0:
        prefix = prefix[best + best_len:]

    return prefix


def _right_scope(
    text: str,
    end: int,
    max_chars: int = 70,
) -> str:
    hi = min(len(text), end + max_chars)
    suffix = text[end:hi]

    stops = [
        x for x in (
            suffix.find("\n"),
            suffix.find("."),
            suffix.find(";"),
        )
        if x >= 0
    ]

    if stops:
        suffix = suffix[:min(stops)]

    return suffix


def classify_target_mention(
    text: str,
    start: int,
    end: int,
) -> str:
    """
    Classify one direct target mention.

    Returns:
      AFFIRMED
      HISTORICAL
      NEGATED
      HYPOTHETICAL
      PLANNED
    """
    text = str(text or "")

    before = _trim_left_scope(text, start)
    after = _right_scope(text, end)

    local_lo = max(0, start - 100)
    local_hi = min(len(text), end + 100)
    local = text[local_lo:local_hi]

    # Explicit negation has highest priority.
    if any(p.search(before) for p in NEGATION_PATTERNS):
        return "NEGATED"

    if any(p.search(after) for p in POST_NEGATION_PATTERNS):
        return "NEGATED"

    if any(p.search(local) for p in PLANNED_PATTERNS):
        return "PLANNED"

    if any(p.search(local) for p in HYPOTHETICAL_PATTERNS):
        return "HYPOTHETICAL"

    if any(p.search(local) for p in HISTORICAL_PATTERNS):
        return "HISTORICAL"

    return "AFFIRMED"

def _context_quote(
    text: str,
    start: int,
    end: int,
    left: int = 100,
    right: int = 100,
) -> str:
    lo = max(0, start - left)
    hi = min(len(text), end + right)

    left_text = text[lo:start]
    boundary = max(
        left_text.rfind("\n"),
        left_text.rfind("."),
        left_text.rfind(";"),
    )
    if boundary >= 0:
        lo = lo + boundary + 1

    right_text = text[end:hi]
    stops = [
        x for x in (
            right_text.find("\n"),
            right_text.find("."),
            right_text.find(";"),
        )
        if x >= 0
    ]
    if stops:
        hi = end + min(stops) + 1

    return text[lo:hi].strip()


def direct_target_mentions(
    feature: str,
    text: str,
) -> List[Dict[str, Any]]:
    text = str(text or "")
    aliases = list(DIRECT_ALIASES.get(feature, []))
    aliases.sort(key=len, reverse=True)

    raw = []

    for alias in aliases:
        for start, end in find_phrase_spans(alias, text):
            raw.append({
                "alias": alias,
                "start": start,
                "end": end,
            })

    raw.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

    selected = []

    for item in raw:
        overlap = any(
            not (
                item["end"] <= kept["start"]
                or item["start"] >= kept["end"]
            )
            for kept in selected
        )
        if overlap:
            continue

        assertion = classify_target_mention(
            text,
            item["start"],
            item["end"],
        )

        quote = _context_quote(
            text,
            item["start"],
            item["end"],
        )

        selected.append({
            **item,
            "assertion": assertion,
            "quote": quote,
        })

    return selected



def _family_history_composite_candidates(
    note_text: str,
    max_candidates: int = 3,
) -> List[Dict[str, Any]]:
    text = str(note_text or "")
    candidates: List[Dict[str, Any]] = []
    seen_quotes = set()

    for crc_pattern in CRC_PATTERNS:
        for match in crc_pattern.finditer(text):
            start = match.start()
            end = match.end()

            assertion = classify_target_mention(
                text,
                start,
                end,
            )

            if assertion not in ELIGIBLE_POSITIVE_ASSERTIONS:
                continue

            quote = _context_quote(
                text,
                start,
                end,
                left=140,
                right=140,
            )

            if not quote:
                continue

            if not has_family_relation(quote):
                continue

            if not has_crc_type(quote):
                continue

            normalized_quote = normalize_text(quote)
            if normalized_quote in seen_quotes:
                continue
            seen_quotes.add(normalized_quote)

            candidates.append(
                {
                    "alias": "__FAMILY_RELATION_PLUS_CRC__",
                    "start": start,
                    "end": end,
                    "assertion": assertion,
                    "quote": quote,
                }
            )

            if len(candidates) >= max_candidates:
                return candidates

    return candidates


def find_affirmed_direct_candidates(
    feature: str,
    note_text: str,
    max_candidates: int = 3,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    seen_quotes = set()

    for mention in direct_target_mentions(feature, note_text):
        if mention["assertion"] not in ELIGIBLE_POSITIVE_ASSERTIONS:
            continue

        if feature == "Family history of colorectal cancer":
            if not has_family_relation(mention["quote"]):
                continue
            if not has_crc_type(mention["quote"]):
                continue

        normalized_quote = normalize_text(mention["quote"])
        if normalized_quote in seen_quotes:
            continue
        seen_quotes.add(normalized_quote)

        candidates.append(mention)

        if len(candidates) >= max_candidates:
            return candidates

    if feature == "Family history of colorectal cancer":
        remaining = max_candidates - len(candidates)

        for mention in _family_history_composite_candidates(
            note_text,
            max_candidates=remaining,
        ):
            normalized_quote = normalize_text(mention["quote"])

            if normalized_quote in seen_quotes:
                continue
            seen_quotes.add(normalized_quote)

            candidates.append(mention)

            if len(candidates) >= max_candidates:
                break

    return candidates


def evaluate_positive_evidence(
    feature: str,
    evidence: str,
    note_text: str,
) -> Dict[str, Any]:
    result = {
        "grounded": False,
        "direct_alias_found": False,
        "eligible_direct_alias_found": False,
        "assertions": [],
        "family_relation_found": None,
        "crc_type_found": None,
        "hard_failures": [],
    }

    result["grounded"] = evidence_grounded_exact_or_normalized(
        evidence,
        note_text,
    )

    if not result["grounded"]:
        result["hard_failures"].append("EVIDENCE_NOT_GROUNDED")
        return result

    mentions = direct_target_mentions(feature, evidence)

    result["direct_alias_found"] = bool(mentions)
    result["assertions"] = [
        m["assertion"] for m in mentions
    ]

    eligible = [
        m for m in mentions
        if m["assertion"] in ELIGIBLE_POSITIVE_ASSERTIONS
    ]

    result["eligible_direct_alias_found"] = bool(eligible)

    if mentions and not eligible:
        result["hard_failures"].append(
            "NO_ELIGIBLE_AFFIRMATIVE_TARGET_MENTION"
        )

    if feature == "Family history of colorectal cancer":
        relation_ok = has_family_relation(evidence)
        crc_ok = has_crc_type(evidence)

        result["family_relation_found"] = relation_ok
        result["crc_type_found"] = crc_ok

        if not relation_ok:
            result["hard_failures"].append(
                "FAMILY_RELATION_NOT_EXPLICIT"
            )

        if not crc_ok:
            result["hard_failures"].append(
                "CRC_TYPE_NOT_EXPLICIT"
            )

    return result

_TEST_ORDER_CONTEXT = re.compile(
    r"\b(?:poc|fobt|fit|fecal fit|occult blood test|"
    r"order(?:ed)?|ordering|screen(?:ing)?|lab(?:s|oratory)?|"
    r"stool culture|test(?:ing)?|econsult)\b",
    re.I,
)

_EXPLICIT_POSITIVE_RESULT = re.compile(
    r"\b(?:positive|pos|detected|heme positive|"
    r"occult blood positive|fit positive|fobt positive)\b",
    re.I,
)

_ADVICE_CUES = re.compile(
    r"\b(?:avoid|prevent|encourage|recommend(?:ed|s)?|"
    r"advis(?:e|ed)|counsel(?:ed|ing)?|goal)\b",
    re.I,
)

_CONDITIONAL_CUES = re.compile(
    r"\b(?:go to (?:the )?er if|return if|call if|"
    r"seek care if|if symptoms?|if .* develops?|"
    r"may lead to)\b",
    re.I,
)

_WEIGHT_LOSS_PURPOSE = re.compile(
    r"\b(?:for weight loss|for wt loss|diet/weight loss|"
    r"using .* for weight loss|use .* for weight loss|"
    r"encourage weight loss|encourage wt loss)\b",
    re.I,
)

_BOWEL_CONTEXT = re.compile(
    r"\b(?:stool|stools|bowel|bowel movement|bowel movements|"
    r"\bbm\b|\bbms\b|defecat|toilet|constipat)\b",
    re.I,
)


def _first_alias_position(
    alias: str,
    quote: str,
) -> int:

    if not alias or alias.startswith("__"):
        return -1

    m = re.search(
        r"(?<!\w)" + re.escape(str(alias)) + r"(?!\w)",
        str(quote or ""),
        re.I,
    )
    return m.start() if m else -1


def candidate_context_hard_failures(
    feature: str,
    alias: str,
    quote: str,
) -> List[str]:
    failures: List[str] = []

    feature = str(feature or "")
    alias = str(alias or "").strip().lower()
    quote = str(quote or "")
    q = normalize_text(quote)

    if not q:
        return ["EMPTY_CANDIDATE"]

    pos = _first_alias_position(alias, quote)
    before = (
        _trim_left_scope(
            quote,
            pos,
            max_chars=60,
        )
        if pos >= 0
        else quote
    )
    after = (
        quote[pos:min(len(quote), pos + 120)]
        if pos >= 0
        else quote
    )

 
    if (
        feature == "Rectal bleeding"
        and alias == "occult blood"
        and _TEST_ORDER_CONTEXT.search(quote)
        and not _EXPLICIT_POSITIVE_RESULT.search(quote)
    ):
        failures.append("TEST_OR_ORDER_ONLY")


    if pos >= 0 and _ADVICE_CUES.search(before):
        failures.append("ADVICE_OR_PREVENTION_CONTEXT")

    if _CONDITIONAL_CUES.search(quote):
        failures.append("CONDITIONAL_OR_HYPOTHETICAL_CONTEXT")

    if alias and not alias.startswith("__"):
        if re.search(
            re.escape(alias) + r"\s*[:=\-]\s*none\b",
            q,
            re.I,
        ):
            failures.append("EXPLICIT_NONE_CONTEXT")


    if alias == "straining" and not _BOWEL_CONTEXT.search(quote):
        failures.append("STRAINING_NOT_BOWEL_CONTEXT")


    if feature == "Weight loss":
        if _WEIGHT_LOSS_PURPOSE.search(quote):
            failures.append("WEIGHT_LOSS_GOAL_OR_PURPOSE")

    return sorted(set(failures))


_STRUCTURAL_SECTION_RE = re.compile(
    r"\b("
    r"chief complaint|"
    r"history of present illness|hpi|"
    r"interval history|interval hx|"
    r"review of systems|review of symptoms|ros|"
    r"patient active problem list|active problem list|problem list|"
    r"past medical history|pmh|"
    r"family history|"
    r"assessment and plan|assessment/plan|assessment|plan|"
    r"physical exam|physical examination|"
    r"objective|subjective"
    r")\b",
    re.I,
)

_STRONG_NEGATION_CUE_RE = re.compile(
    r"\b("
    r"negative\s+for|"
    r"neg\s+for|"
    r"denies|denied|"
    r"without|"
    r"no\s+evidence\s+of|"
    r"pertinent\s+negatives?\s+(?:include|includes|included)"
    r")\b",
    re.I,
)

_NEGATION_SCOPE_BREAK_RE = re.compile(
    r"(?:[.;\n]|"
    r"\bbut\b|"
    r"\bhowever\b|"
    r"\bexcept\b|"
    r"\balthough\b|"
    r"\bwhereas\b)",
    re.I,
)

_MARKED_ROS_RE = re.compile(
    r"review\s+of\s+(?:systems|symptoms)"
    r".{0,160}?"
    r"positives?\s+(?:are\s+)?(?:in\s+)?bold",
    re.I | re.S,
)

_PROBLEM_LIST_HEADERS = {
    "patient active problem list",
    "active problem list",
    "problem list",
}

_BROAD_RECOVERY_ALIASES = {
    "frequent bowel movement",
    "frequent bowel movements",
    "hard stool",
    "hard stools",
}

_DIARRHEA_CORROBORATION_RE = re.compile(
    r"\b("
    r"diarrhea|"
    r"loose\s+(?:stool|stools|bowel movements?|bms?)|"
    r"watery\s+(?:stool|stools|bowel movements?|bms?)|"
    r"liquid\s+(?:stool|stools|bowel movements?|bms?)|"
    r"runny\s+(?:stool|stools)"
    r")\b",
    re.I,
)

_CONSTIPATION_CORROBORATION_RE = re.compile(
    r"\b("
    r"constipation|constipated|"
    r"straining|"
    r"difficult(?:y)?\s+(?:passing|with)\s+"
    r"(?:stool|stools|bowel movements?|bms?)|"
    r"infrequent\s+(?:stool|stools|bowel movements?|bms?)|"
    r"no\s+(?:bowel movement|bm)\b|"
    r"stool\s+softener|"
    r"miralax|senna|laxative"
    r")\b",
    re.I,
)


def _whitespace_flexible_pattern(text):
    """Build a regex that tolerates whitespace differences."""
    tokens = [
        re.escape(x)
        for x in re.split(r"\s+", str(text or "").strip())
        if x
    ]
    if not tokens:
        return None
    return re.compile(r"\s+".join(tokens), re.I)


def _locate_candidate_span(note, quote, alias):
    note = str(note or "")
    quote = str(quote or "").strip()
    alias = str(alias or "").strip()

    if quote:
        pat = _whitespace_flexible_pattern(quote)
        if pat is not None:
            m = pat.search(note)
            if m:
                return m.start(), m.end()

    if alias and not alias.startswith("__"):
        pat = _whitespace_flexible_pattern(alias)
        if pat is not None:
            m = pat.search(note)
            if m:
                return m.start(), m.end()

    return None, None


def _locate_alias_near_candidate(note, alias, start, end):
    alias = str(alias or "").strip()

    if (
        not alias
        or alias.startswith("__")
        or start is None
        or end is None
    ):
        return start, end

    lo = max(0, start - 80)
    hi = min(len(note), end + 80)

    chunk = note[lo:hi]
    pat = _whitespace_flexible_pattern(alias)

    if pat is None:
        return start, end

    matches = list(pat.finditer(chunk))
    if not matches:
        return start, end

    target_center = (start + end) / 2.0

    best = min(
        matches,
        key=lambda m: abs(
            (lo + m.start() + lo + m.end()) / 2.0
            - target_center
        ),
    )

    return lo + best.start(), lo + best.end()


def _nearest_structural_section(note, position):
    if position is None:
        return ""

    matches = list(
        _STRUCTURAL_SECTION_RE.finditer(
            note[:position]
        )
    )

    if not matches:
        return ""

    return matches[-1].group(1).strip().lower()


def _candidate_is_plus_marked(note, start):
    if start is None:
        return False

    left = note[max(0, start - 8):start]

    return bool(
        re.search(
            r"\+\s*$",
            left,
        )
    )


def _candidate_in_marked_ros(note, start):
    if start is None:
        return False

    before = note[max(0, start - 900):start]

    matches = list(
        _MARKED_ROS_RE.finditer(before)
    )

    if not matches:
        return False

    last = matches[-1]
    distance = len(before) - last.end()

    return distance <= 700


def _negation_list_applies(note, start):
    if start is None:
        return False

    before = note[max(0, start - 400):start]

    matches = list(
        _STRONG_NEGATION_CUE_RE.finditer(before)
    )

    if matches:
        last = matches[-1]
        suffix = before[last.end():]

        if (
            len(suffix) <= 300
            and not _NEGATION_SCOPE_BREAK_RE.search(suffix)
        ):
            return True

    very_local = note[max(0, start - 35):start]

    if re.search(
        r"\bno\s+(?:current\s+)?$",
        very_local,
        re.I,
    ):
        return True

    return False


def _broad_alias_has_corroboration(
    feature,
    alias,
    note,
    start,
    end,
):
    alias_n = re.sub(
        r"\s+",
        " ",
        str(alias or "").strip().lower(),
    )

    if alias_n not in _BROAD_RECOVERY_ALIASES:
        return True

    if start is None:
        return False

    lo = max(0, start - 180)
    hi = min(
        len(note),
        (end if end is not None else start) + 220,
    )

    ctx = note[lo:hi]

    feature_n = re.sub(
        r"\s+",
        " ",
        str(feature or "").strip().lower(),
    )

    if feature_n == "diarrhea":
        return bool(
            _DIARRHEA_CORROBORATION_RE.search(ctx)
        )

    if feature_n == "constipation":
        return bool(
            _CONSTIPATION_CORROBORATION_RE.search(ctx)
        )

    return True


def candidate_structural_hard_failures(
    feature,
    alias,
    quote,
    note,
):
    failures = []

    note = str(note or "")
    quote = str(quote or "")
    alias = str(alias or "")

    start, end = _locate_candidate_span(
        note,
        quote,
        alias,
    )

    if start is None:
        return ["CANDIDATE_LOCATION_UNRESOLVED"]

    alias_start, alias_end = _locate_alias_near_candidate(
        note,
        alias,
        start,
        end,
    )


    if _negation_list_applies(
        note,
        alias_start,
    ):
        failures.append(
            "NEGATED_LIST_SCOPE"
        )


    if (
        _candidate_in_marked_ros(
            note,
            alias_start,
        )
        and not _candidate_is_plus_marked(
            note,
            alias_start,
        )
    ):
        failures.append(
            "UNMARKED_ITEM_IN_POLARITY_MARKED_ROS"
        )


    nearest_section = _nearest_structural_section(
        note,
        alias_start,
    )

    if nearest_section in _PROBLEM_LIST_HEADERS:
        failures.append(
            "PROBLEM_LIST_ONLY_CONTEXT"
        )


    if not _broad_alias_has_corroboration(
        feature,
        alias,
        note,
        alias_start,
        alias_end,
    ):
        failures.append(
            "BROAD_ALIAS_WITHOUT_CORROBORATION"
        )

    return sorted(set(failures))


_V2_BROAD_DIARRHEA_ALIASES = {
    "frequent bowel movement",
    "frequent bowel movements",
}

_V2_BROAD_CONSTIPATION_ALIASES = {
    "hard stool",
    "hard stools",
}

_V2_DIARRHEA_POSITIVE_CORROBORATION = re.compile(
    r"\b(?:"
    r"loose\s+(?:stool|stools|bowel movements?|bms?)|"
    r"watery\s+(?:stool|stools|bowel movements?|bms?)|"
    r"liquid\s+(?:stool|stools|bowel movements?|bms?)|"
    r"runny\s+(?:stool|stools)|"
    r"diarrhea"
    r")\b",
    re.I,
)

_V2_DIARRHEA_NEGATED = re.compile(
    r"\b(?:"
    r"no|denies|denied|without|"
    r"negative\s+for|neg\s+for"
    r")\b"
    r".{0,80}\bdiarrhea\b",
    re.I | re.S,
)

_V2_CONSTIPATION_CORROBORATION = re.compile(
    r"\b(?:"
    r"constipation|constipated|"
    r"straining|"
    r"infrequent\s+(?:stool|stools|bowel movements?|bms?)|"
    r"difficult(?:y)?\s+(?:passing|with)\s+"
    r"(?:stool|stools|bowel movements?|bms?)|"
    r"no\s+(?:bowel movement|bm)\b|"
    r"stool\s+softener|"
    r"miralax|senna|laxative"
    r")\b",
    re.I,
)

_V2_ROS_NEGATIVE_UNLESS_MARKED = re.compile(
    r"\b(?:"
    r"ros|review\s+of\s+systems|review\s+of\s+symptoms"
    r")\b"
    r".{0,500}?"
    r"\bnegative\s+unless\s+(?:bold|bolded|marked)\b",
    re.I | re.S,
)

_V2_REMOTE_DIAGNOSTIC_HISTORY = re.compile(
    r"\b(?:"
    r"pmh|past\s+medical\s+history|history\s+of"
    r")\b"
    r".{0,260}?"
    r"\b(?:diagnosed|diagnosis)\b"
    r".{0,160}?"
    r"\bafter\s+presenting\s+with\b",
    re.I | re.S,
)

_V2_FH_CONCERN_ONLY = re.compile(
    r"\b(?:"
    r"mother|father|mom|dad|"
    r"sister|brother|sibling|"
    r"grandmother|grandfather|"
    r"grandparent|gm|gf|mgm|mgf|pgm|pgf|"
    r"aunt|uncle|daughter|son"
    r")\b"
    r".{0,80}?"
    r"\b(?:"
    r"concerned|concern|worried|worry|"
    r"fearful|fears?"
    r")\b"
    r".{0,120}?"
    r"\b(?:"
    r"colon|colorectal|rectal|bowel"
    r")\b"
    r".{0,20}?"
    r"\b(?:cancer|carcinoma|ca)\b",
    re.I | re.S,
)


def _v2_sentence_scope(note, start, end):
    note = str(note or "")

    if start is None:
        return "", 0

    floor = max(0, start - 1400)

    left = note[floor:start]

    boundaries = [
        left.rfind("."),
        left.rfind(";"),
        left.rfind("\n"),
    ]

    cut = max(boundaries)

    if cut >= 0:
        scope_start = floor + cut + 1
    else:
        scope_start = floor

    ceiling = min(
        len(note),
        (end if end is not None else start) + 700,
    )

    right = note[
        end if end is not None else start:
        ceiling
    ]

    right_boundaries = [
        x for x in (
            right.find("."),
            right.find(";"),
            right.find("\n"),
        )
        if x >= 0
    ]

    if right_boundaries:
        scope_end = (
            end if end is not None else start
        ) + min(right_boundaries) + 1
    else:
        scope_end = ceiling

    return note[scope_start:scope_end], scope_start


def _v2_long_negation_applies(note, alias_start):
    if alias_start is None:
        return False

    scope, scope_start = _v2_sentence_scope(
        note,
        alias_start,
        alias_start,
    )

    relative = alias_start - scope_start

    if relative < 0:
        return False

    before = scope[:relative]

    matches = list(
        _STRONG_NEGATION_CUE_RE.finditer(before)
    )

    if not matches:
        return False

    last = matches[-1]

    between = before[last.end():]

    if re.search(
        r"\b(?:but|however|except|although|whereas)\b",
        between,
        re.I,
    ):
        return False

    return True


def _v2_unmarked_negative_ros(note, alias_start):
    if alias_start is None:
        return False

    before = note[
        max(0, alias_start - 1000):
        alias_start
    ]

    matches = list(
        _V2_ROS_NEGATIVE_UNLESS_MARKED.finditer(before)
    )

    if not matches:
        return False

    last = matches[-1]

    if len(before) - last.end() > 750:
        return False

    return not _candidate_is_plus_marked(
        note,
        alias_start,
    )


def _v2_broad_alias_valid(
    feature,
    alias,
    note,
    alias_start,
    alias_end,
):
    alias_n = re.sub(
        r"\s+",
        " ",
        str(alias or "").strip().lower(),
    )

    feature_n = re.sub(
        r"\s+",
        " ",
        str(feature or "").strip().lower(),
    )

    if alias_start is None:
        return False

    lo = max(0, alias_start - 220)
    hi = min(
        len(note),
        (
            alias_end
            if alias_end is not None
            else alias_start
        ) + 320,
    )

    ctx = note[lo:hi]

    if (
        feature_n == "diarrhea"
        and alias_n in _V2_BROAD_DIARRHEA_ALIASES
    ):

        if _V2_DIARRHEA_NEGATED.search(ctx):
            return False

        matches = list(
            _V2_DIARRHEA_POSITIVE_CORROBORATION.finditer(ctx)
        )

        for m in matches:
            text = m.group(0).lower()

            if "frequent bowel" in text:
                continue

            if (
                text == "diarrhea"
                and _V2_DIARRHEA_NEGATED.search(ctx)
            ):
                continue

            return True

        return False

    if (
        feature_n == "constipation"
        and alias_n in _V2_BROAD_CONSTIPATION_ALIASES
    ):
        return bool(
            _V2_CONSTIPATION_CORROBORATION.search(ctx)
        )

    return True


def _v2_remote_diagnostic_history(
    note,
    alias_start,
    alias_end,
):

    if alias_start is None:
        return False

    lo = max(0, alias_start - 420)
    hi = min(
        len(note),
        (
            alias_end
            if alias_end is not None
            else alias_start
        ) + 260,
    )

    ctx = note[lo:hi]

    return bool(
        _V2_REMOTE_DIAGNOSTIC_HISTORY.search(ctx)
    )


def _v2_fh_concern_only(
    feature,
    alias,
    note,
    start,
    end,
):
    feature_n = re.sub(
        r"\s+",
        " ",
        str(feature or "").strip().lower(),
    )

    if (
        feature_n
        != "family history of colorectal cancer"
    ):
        return False

    if start is None:
        return False

    lo = max(0, start - 180)
    hi = min(
        len(note),
        (
            end
            if end is not None
            else start
        ) + 220,
    )

    ctx = note[lo:hi]

    return bool(
        _V2_FH_CONCERN_ONLY.search(ctx)
    )


def candidate_structural_hard_failures_v2(
    feature,
    alias,
    quote,
    note,
):
    failures = set(
        candidate_structural_hard_failures(
            feature,
            alias,
            quote,
            note,
        )
    )

    note = str(note or "")

    start, end = _locate_candidate_span(
        note,
        quote,
        alias,
    )

    if start is None:
        failures.add(
            "CANDIDATE_LOCATION_UNRESOLVED"
        )
        return sorted(failures)

    alias_start, alias_end = (
        _locate_alias_near_candidate(
            note,
            alias,
            start,
            end,
        )
    )

    if _v2_long_negation_applies(
        note,
        alias_start,
    ):
        failures.add(
            "LONG_NEGATED_LIST_SCOPE"
        )

    if _v2_unmarked_negative_ros(
        note,
        alias_start,
    ):
        failures.add(
            "UNMARKED_ITEM_IN_NEGATIVE_UNLESS_BOLDED_ROS"
        )

    if not _v2_broad_alias_valid(
        feature,
        alias,
        note,
        alias_start,
        alias_end,
    ):
        failures.add(
            "BROAD_ALIAS_WITHOUT_STRICT_CORROBORATION"
        )

    if _v2_remote_diagnostic_history(
        note,
        alias_start,
        alias_end,
    ):
        failures.add(
            "REMOTE_DIAGNOSTIC_HISTORY_ONLY"
        )

    if _v2_fh_concern_only(
        feature,
        alias,
        note,
        start,
        end,
    ):
        failures.add(
            "FAMILY_MEMBER_CONCERN_NOT_FAMILY_HISTORY"
        )

    return sorted(failures)
