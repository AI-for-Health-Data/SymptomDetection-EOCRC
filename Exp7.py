from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from bert_score import score as bert_score_fn

    HAVE_BERTSCORE = True
except ModuleNotFoundError:
    HAVE_BERTSCORE = False


PATH = os.environ.get("EOCRC_NOTES", "rebuilt_notes_by_noteid.csv")
HF_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_CACHE_DIR = os.environ.get(
    "HF_CACHE_DIR", "/lustre/smuexa01/client/users/nikkieh/hf_cache"
)
UMLS_JSON = os.environ.get("UMLS_JSON", "umls_synonyms.json")
UMLS_CACHE = os.environ.get("UMLS_CACHE", "umls_concept_cache.json")
CHUNKS_PATH = os.environ.get(
    "PUBMED_CHUNKS", "pubmed_chunks_sentences.json"
)

MEDCPT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"

OUT_RAW = os.environ.get("EXP7_RAW", "exp7_outputs_raw.csv")
OUT_METRICS = os.environ.get(
    "EXP7_METRICS", "exp7_note_level_metrics.csv"
)
OUT_CHECKPOINT = os.environ.get("EXP7_CHECKPOINT", "exp7_checkpoint.csv")
OUT_SUMMARY = os.environ.get("EXP7_SUMMARY", "exp7_summary_table.csv")
OUT_VALID = os.environ.get("EXP7_VALID", "exp7_valid_outputs.csv")
OUT_SCHEMA_FAIL = os.environ.get(
    "EXP7_SCHEMA_FAIL", "exp7_schema_failures.csv"
)
OUT_PARSE_FAIL = os.environ.get(
    "EXP7_PARSE_FAIL", "exp7_parse_failures.json"
)

DENSE_TOP_K = 32
RRF_K = 60
SOURCE_CANDIDATE_K = 10
DENSE_KEEP_K = 2
BLEU_THRESHOLD = 0.10
NOTE_CHAR_LIMIT = 12_000
MAX_INPUT_TOKENS = 12_000
MAIN_MAX_NEW_TOKENS = 1_536
RETRY_MAX_NEW_TOKENS = 512
DENIAL_CONTEXT = 40
CHECKPOINT_EVERY = 25
MAX_GROUP_B_IN_QUERY = 3
MAX_GROUP_B_IN_PROMPT = 10

ALL_SOURCE_TIERS = [
    "HIGH",
    "MODERATE",
    "UNCLASSIFIED",
    "LOW",
    "VERY_LOW",
]
ALLOWED_SOURCE_TIERS = {"HIGH", "MODERATE", "UNCLASSIFIED"}
SOURCE_TIER_PRIORITY = {
    "HIGH": 0,
    "MODERATE": 1,
    "UNCLASSIFIED": 2,
}

SYMPTOMS = [
    "Abdominal pain",
    "Rectal bleeding",
    "Rectal pain",
    "Diarrhea",
    "Constipation",
    "Weight loss",
    "Family history of colorectal cancer",
]
SYMPTOM_SPECS = [
    (symptom, f"{symptom} confidence", f"{symptom} inference")
    for symptom in SYMPTOMS
]

EXPECTED_OUTPUT_KEYS: List[str] = []
for _symptom in SYMPTOMS[:-1]:
    _duration = f"Duration of {_symptom.lower()}"
    EXPECTED_OUTPUT_KEYS.extend(
        [
            _symptom,
            f"{_symptom} confidence",
            f"{_symptom} inference",
            _duration,
            f"{_duration} confidence",
            f"{_duration} inference",
        ]
    )
EXPECTED_OUTPUT_KEYS.extend(
    [
        "Family history of colorectal cancer",
        "Family history of colorectal cancer confidence",
        "Family history of colorectal cancer inference",
        "Other comments",
        "Other comments confidence",
        "Other comments inference",
    ]
)

BAD_INFERENCE_VALS = {
    "",
    "n/a",
    "na",
    "not mentioned",
    "none",
    "no inference",
    "not reported",
    "not applicable",
    "not stated",
    "not mentioned outside ros",
    "yes",
    "no",
    "present",
    "absent",
    "only in ros-negative context",
    "only ros-negative, nowhere else",
}

DIRECT_ALIASES: Dict[str, List[str]] = {
    "Abdominal pain": [
        "abdominal pain",
        "abd pain",
        "abd pn",
        "abdo pain",
        "stomach pain",
        "stomach ache",
        "belly pain",
        "epigastric pain",
        "epigastric discomfort",
        "epigastric tenderness",
        "abdominal tenderness",
        "tender abdomen",
        "ruq pain",
        "right upper quadrant pain",
        "luq pain",
        "left upper quadrant pain",
        "rlq pain",
        "right lower quadrant pain",
        "llq pain",
        "left lower quadrant pain",
        "periumbilical pain",
        "umbilical pain",
        "suprapubic pain",
        "pelvic pain",
        "lower abdominal pain",
        "upper abdominal pain",
        "abdominal cramping",
        "colicky pain",
        "colicky abdominal pain",
        "sharp abdominal pain",
        "dull abdominal pain",
        "abdominal discomfort",
        "abdominal soreness",
        "c/o abdominal pain",
        "complains of abdominal pain",
        "reports abdominal pain",
        "pain in abdomen",
    ],
    "Rectal bleeding": [
        "rectal bleeding",
        "bleeding per rectum",
        "blood per rectum",
        "blood from rectum",
        "blood in stool",
        "bloody stool",
        "blood on stool",
        "stool with blood",
        "streaks of blood",
        "blood streaked stool",
        "blood on toilet paper",
        "blood when wiping",
        "hematochezia",
        "haematochezia",
        "brbpr",
        "bright red blood per rectum",
        "maroon stools",
        "melena",
        "black tarry stools",
        "positive fobt",
        "positive fit",
        "occult blood",
        "heme positive stool",
    ],
    "Rectal pain": [
        "rectal pain",
        "pain in rectum",
        "painful rectum",
        "anal pain",
        "pain in anus",
        "anorectal pain",
        "proctalgia",
        "proctalgia fugax",
        "rectal discomfort",
        "anal discomfort",
        "rectal soreness",
        "anal soreness",
        "pain with bowel movement",
        "painful bowel movement",
        "painful defecation",
        "dyschezia",
        "pain during defecation",
        "pain after bowel movement",
        "tenesmus",
        "rectal pressure",
        "perianal pain",
        "perirectal pain",
    ],
    "Diarrhea": [
        "diarrhea",
        "diarrhoea",
        "loose stools",
        "loose stool",
        "watery stools",
        "watery stool",
        "liquid stool",
        "runny stool",
        "frequent stools",
        "frequent bowel movements",
        "increased bowel movements",
        "increased stool frequency",
        "multiple loose bms",
        "loose bm",
        "watery bm",
        "explosive diarrhea",
        "bristol 6",
        "bristol 7",
    ],
    "Constipation": [
        "constipation",
        "constipated",
        "hard stools",
        "hard stool",
        "infrequent stools",
        "infrequent bowel movements",
        "decreased stool frequency",
        "decreased bowel movements",
        "no bm",
        "no bowel movement",
        "no stool for",
        "difficulty passing stool",
        "straining",
        "incomplete evacuation",
        "obstipation",
        "fecal impaction",
        "stool impaction",
        "bristol 1",
        "bristol 2",
        "pellet stools",
    ],
    "Weight loss": [
        "weight loss",
        "wt loss",
        "lost weight",
        "losing weight",
        "weight down",
        "weight decreased",
        "decreased weight",
        "unintentional weight loss",
        "unexplained weight loss",
        "involuntary weight loss",
        "clothes fitting looser",
        "cachexia",
        "wasting",
        "cachectic",
    ],
    "Family history of colorectal cancer": [
        "family history of colorectal cancer",
        "family history colorectal cancer",
        "family history of colon cancer",
        "family history colon cancer",
        "family history of rectal cancer",
        "family history of bowel cancer",
        "fh colon cancer",
        "fhx colon cancer",
        "fhx crc",
        "fh crc",
        "crc in family",
        "colon cancer in family",
        "colon cancer runs in family",
        "mother had colon cancer",
        "father had colon cancer",
        "sister had colon cancer",
        "brother had colon cancer",
        "parent had colon cancer",
        "relative with colon cancer",
        "first degree relative with colon cancer",
        "fh bowel cancer",
    ],
}


RETRIEVAL_EXTRA_TERMS: Dict[str, List[str]] = {
    "Abdominal pain": [
        "cramping",
        "bloating",
        "abdominal bloating",
        "distension",
        "abdominal distension",
        "dyspepsia",
        "indigestion",
        "fullness",
        "abdominal pressure",
        "gas pain",
        "heartburn",
        "acid reflux",
        "gerd symptoms",
        "gastritis",
    ],
    "Rectal bleeding": [
        "rectal hemorrhage",
        "rectorrhagia",
        "hemorrhoids with bleeding",
        "anal fissure bleeding",
    ],
    "Rectal pain": [
        "hemorrhoid pain",
        "thrombosed hemorrhoid",
        "anal fissure pain",
        "fissure pain",
    ],
    "Diarrhea": [
        "the runs",
        "fecal urgency",
        "bowel urgency",
        "soft stools",
        "mushy stools",
        "gastroenteritis",
        "colitis with diarrhea",
    ],
    "Constipation": [
        "retained stool",
        "stool burden",
        "slow transit constipation",
        "scybalous stools",
        "strains with bm",
    ],
    "Weight loss": [
        "malnutrition",
        "anorexia",
        "loss of appetite",
        "decreased appetite",
        "poor weight gain",
        "failure to thrive",
    ],
    "Family history of colorectal cancer": [
        "lynch syndrome",
        "hnpcc",
        "familial adenomatous polyposis",
        "fap",
        "hereditary colorectal cancer",
    ],
}


VERY_LOW_SOURCE_PATTERNS: List[Tuple[str, str]] = [
    (r"\bcase report\b", "case report"),
    (r"\bcase reports\b", "case reports"),
    (r"\bexpert opinion\b", "expert opinion"),
    (r"\beditorial\b", "editorial"),
    (r"\bcommentary\b", "commentary"),
    (r"\bletter to the editor\b", "letter to the editor"),
    (r"\banimal study\b", "animal study"),
    (r"\banimal model\b", "animal model"),
    (r"\bin vitro\b", "in vitro"),
]
LOW_SOURCE_PATTERNS: List[Tuple[str, str]] = [
    (r"\bcase series\b", "case series"),
    (r"\bcase-control\b", "case-control"),
    (r"\bcase control\b", "case control"),
    (r"\bnarrative review\b", "narrative review"),
    (r"\bliterature review\b", "literature review"),
    (r"\bpilot study\b", "pilot study"),
    (r"\bsingle[- ]center\b", "single-center"),
    (r"\buncontrolled study\b", "uncontrolled study"),
]
HIGH_SOURCE_PATTERNS: List[Tuple[str, str]] = [
    (r"\bsystematic review\b", "systematic review"),
    (r"\bmeta-analysis\b", "meta-analysis"),
    (r"\bmeta analysis\b", "meta analysis"),
    (r"\brandomi[sz]ed controlled trial\b", "randomized controlled trial"),
    (r"\bplacebo-controlled\b", "placebo-controlled"),
    (r"\bphase iii\b", "phase III"),
    (r"\bphase 3\b", "phase 3"),
]
MODERATE_SOURCE_PATTERNS: List[Tuple[str, str]] = [
    (r"\bnon[- ]randomi[sz]ed clinical trial\b", "non-randomized trial"),
    (r"\bprospective cohort\b", "prospective cohort"),
    (r"\bretrospective cohort\b", "retrospective cohort"),
    (r"\bcohort study\b", "cohort study"),
    (r"\blongitudinal study\b", "longitudinal study"),
    (r"\bpopulation-based\b", "population-based"),
    (r"\bregistry-based\b", "registry-based"),
    (r"\bobservational study\b", "observational study"),
    (r"\bcross-sectional\b", "cross-sectional"),
]


def classify_source_tier(text: str) -> Tuple[str, Optional[str]]:
    """Assign a GRADE-inspired source-design tier to a sentence chunk."""
    value = str(text).lower()
    for patterns, tier in [
        (VERY_LOW_SOURCE_PATTERNS, "VERY_LOW"),
        (LOW_SOURCE_PATTERNS, "LOW"),
        (HIGH_SOURCE_PATTERNS, "HIGH"),
        (MODERATE_SOURCE_PATTERNS, "MODERATE"),
    ]:
        for pattern, marker in patterns:
            if re.search(pattern, value, flags=re.IGNORECASE):
                return tier, marker
    return "UNCLASSIFIED", None


def require_file(path: str, description: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} not found: {path}")


def phrase_in_text(term: str, text: str) -> bool:
    """Boundary-aware phrase matching; avoids raw-substring false matches."""
    term_value = str(term).strip().lower()
    text_value = str(text).lower()
    if not term_value or len(term_value) < 3:
        return False
    pattern = r"(?<!\w)" + re.escape(term_value) + r"(?!\w)"
    return bool(re.search(pattern, text_value))


def find_phrase_spans(term: str, text: str) -> Iterable[Tuple[int, int]]:
    term_value = str(term).strip().lower()
    text_value = str(text).lower()
    if not term_value or len(term_value) < 3:
        return []
    pattern = re.compile(r"(?<!\w)" + re.escape(term_value) + r"(?!\w)")
    return [(match.start(), match.end()) for match in pattern.finditer(text_value)]


PRE_NEGATION = [
    r"\bno\b",
    r"\bdenies?\b",
    r"\bnegative for\b",
    r"\bwithout\b",
    r"\babsence of\b",
    r"\bno history of\b",
    r"\bnot\b",
]
POST_NEGATION = [
    r"\babsent\b",
    r"\bdenied\b",
    r"\bnegative\b",
    r"\bnot present\b",
    r"\bresolved\b",
]


def mention_is_negated(
    text_lower: str,
    start: int,
    end: int,
    window: int = DENIAL_CONTEXT,
) -> bool:
    before = text_lower[max(0, start - window) : start]
    after = text_lower[end : min(len(text_lower), end + window)]
    pre = any(re.search(pattern, before) for pattern in PRE_NEGATION)
    post = any(re.search(pattern, after) for pattern in POST_NEGATION)
    return pre or post


def load_umls_synonyms() -> Dict[str, List[str]]:
    require_file(UMLS_JSON, "UMLS synonym file")
    with open(UMLS_JSON, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    lowered = {
        symptom: [str(term).lower() for term in terms]
        for symptom, terms in data.items()
    }
    total = sum(len(values) for values in lowered.values())
    print(f"UMLS synonyms: {total} terms across {len(lowered)} symptoms")
    return lowered


def load_concept_cache() -> Tuple[Dict[str, Any], Dict[str, str]]:
    require_file(UMLS_CACHE, "UMLS concept cache")
    with open(UMLS_CACHE, "r", encoding="utf-8") as handle:
        cache = json.load(handle)
    n_group_a = sum(
        1 for value in cache.values() if value.get("is_group_a", False)
    )
    n_group_b = len(cache) - n_group_a
    print(
        f"UMLS cache: {len(cache)} concepts "
        f"(GROUP A={n_group_a}, GROUP B={n_group_b})"
    )
    group_b_lookup = {
        str(term).lower(): str(value.get("name", term))
        for term, value in cache.items()
        if not value.get("is_group_a", False)
    }
    return cache, group_b_lookup


def build_vocabularies(
    umls_synonyms: Mapping[str, Sequence[str]],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build separate direct-note and broad-retrieval vocabularies."""
    direct_vocab: Dict[str, List[str]] = {}
    retrieval_vocab: Dict[str, List[str]] = {}
    for symptom in SYMPTOMS:
        direct = list(
            dict.fromkeys(
                str(term).lower()
                for term in DIRECT_ALIASES.get(symptom, [])
                if str(term).strip()
            )
        )
        extras = [
            str(term).lower()
            for term in RETRIEVAL_EXTRA_TERMS.get(symptom, [])
            if str(term).strip()
        ]
        umls = [
            str(term).lower()
            for term in umls_synonyms.get(symptom, [])
            if str(term).strip()
        ]
        direct_vocab[symptom] = direct
        retrieval_vocab[symptom] = list(dict.fromkeys(direct + extras + umls))

    direct_total = sum(len(values) for values in direct_vocab.values())
    retrieval_total = sum(len(values) for values in retrieval_vocab.values())
    direct_unique = len(
        {term for values in direct_vocab.values() for term in values}
    )
    retrieval_unique = len(
        {term for values in retrieval_vocab.values() for term in values}
    )
    print(
        f"Direct vocabulary:    {direct_total} entries "
        f"({direct_unique} unique) — note matching + prompt"
    )
    print(
        f"Retrieval vocabulary: {retrieval_total} entries "
        f"({retrieval_unique} unique) — MedCPT queries + relevance"
    )
    return direct_vocab, retrieval_vocab


def _chunk_to_text(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, Mapping):
        for key in ["text", "chunk", "sentence", "content", "abstract"]:
            if key in chunk and chunk[key] is not None:
                return str(chunk[key])
    return str(chunk)


def load_chunks() -> List[str]:
    require_file(CHUNKS_PATH, "PubMed sentence-chunk corpus")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as handle:
        raw_chunks = json.load(handle)
    if not isinstance(raw_chunks, list):
        raise ValueError("PubMed chunk file must contain a JSON list.")
    chunks = [_chunk_to_text(chunk).strip() for chunk in raw_chunks]
    chunks = [chunk for chunk in chunks if chunk]
    if not chunks:
        raise ValueError("PubMed chunk corpus is empty after normalization.")
    print(f"Sentence chunks loaded: {len(chunks)}")
    return chunks


def load_notes() -> pd.DataFrame:
    require_file(PATH, "Clinical notes CSV")
    notes_df = pd.read_csv(PATH)
    if "Clean_note_text" not in notes_df.columns:
        raise ValueError("Clean_note_text column is missing from the notes CSV.")
    notes_df["Clean_note_text"] = (
        notes_df["Clean_note_text"].fillna("").astype(str)
    )
    for column in [
        "DATE_OF_SERVIC_DTTM",
        "SPEC_NOTE_TIME_DTTM",
        "CONTACT_DATE",
    ]:
        if column in notes_df.columns:
            notes_df[column] = pd.to_datetime(notes_df[column], errors="coerce")
    sort_columns = [
        column
        for column in [
            "PAT_ID",
            "PAT_ENC_CSN_ID",
            "DATE_OF_SERVIC_DTTM",
            "SPEC_NOTE_TIME_DTTM",
            "CONTACT_NUM",
            "NOTE_ID",
        ]
        if column in notes_df.columns
    ]
    if sort_columns:
        notes_df = notes_df.sort_values(
            sort_columns, kind="stable"
        ).reset_index(drop=True)
    notes_df = notes_df[
        notes_df["Clean_note_text"].str.strip().ne("")
    ].reset_index(drop=True)
    notes_df["_run_row_index"] = np.arange(len(notes_df))
    print(f"Notes loaded: {len(notes_df)}")
    return notes_df


class MedCPTEncoder:
    def __init__(self, cache_dir: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading MedCPT Article Encoder: {MEDCPT_ARTICLE_MODEL}")
        self.article_tokenizer = AutoTokenizer.from_pretrained(
            MEDCPT_ARTICLE_MODEL, cache_dir=cache_dir
        )
        self.article_encoder = AutoModel.from_pretrained(
            MEDCPT_ARTICLE_MODEL, cache_dir=cache_dir
        ).to(self.device)
        self.article_encoder.eval()

        print(f"Loading MedCPT Query Encoder: {MEDCPT_QUERY_MODEL}")
        self.query_tokenizer = AutoTokenizer.from_pretrained(
            MEDCPT_QUERY_MODEL, cache_dir=cache_dir
        )
        self.query_encoder = AutoModel.from_pretrained(
            MEDCPT_QUERY_MODEL, cache_dir=cache_dir
        ).to(self.device)
        self.query_encoder.eval()
        print(f"MedCPT encoders ready on {self.device}")

    def _encode(
        self,
        model: Any,
        tokenizer: Any,
        texts: Sequence[str],
        batch_size: int,
        max_length: int = 512,
    ) -> np.ndarray:
        all_vectors: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            with self.torch.no_grad():
                tokens = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                output = model(**tokens)
                vectors = output.last_hidden_state[:, 0, :]
                vectors = vectors / (
                    vectors.norm(dim=1, keepdim=True) + 1e-8
                )
                all_vectors.append(vectors.cpu().float().numpy())
        if not all_vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(all_vectors)

    def encode_articles(
        self, texts: Sequence[str], batch_size: int = 64
    ) -> np.ndarray:
        return self._encode(
            self.article_encoder,
            self.article_tokenizer,
            texts,
            batch_size=batch_size,
        )

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(
            self.query_encoder,
            self.query_tokenizer,
            texts,
            batch_size=32,
        )


def build_dense_index(
    encoder: MedCPTEncoder,
    chunks: Sequence[str],
    batch_size: int = 128,
) -> np.ndarray:
    print(f"Encoding {len(chunks)} chunks with MedCPT Article Encoder...")
    start = time.time()
    vectors = encoder.encode_articles(chunks, batch_size=batch_size)
    elapsed_minutes = (time.time() - start) / 60
    print(
        f"Dense index built: {vectors.shape} in "
        f"{elapsed_minutes:.1f} minutes"
    )
    return vectors


def scan_note_for_umls_concepts(
    note_text: str,
    direct_vocab: Mapping[str, Sequence[str]],
    group_b_lookup: Mapping[str, str],
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    note_lower = note_text.lower()
    group_a: Dict[str, List[str]] = {}
    for symptom in SYMPTOMS:
        found = [
            term
            for term in direct_vocab.get(symptom, [])
            if phrase_in_text(term, note_lower)
        ]
        if found:
            group_a[symptom] = found[:5]

    group_b: Dict[str, str] = {}
    for term, concept_name in group_b_lookup.items():
        if phrase_in_text(term, note_lower) and concept_name not in group_b:
            group_b[concept_name] = term
    return group_a, group_b


def build_query_variants(
    note_text: str,
    symptom: str,
    direct_vocab: Mapping[str, Sequence[str]],
    retrieval_vocab: Mapping[str, Sequence[str]],
    group_b: Mapping[str, str],
) -> Dict[str, str]:
    """Construct distinct query formulations for within-symptom RRF."""
    note_lower = note_text.lower()
    note_matches = [
        term
        for term in direct_vocab.get(symptom, [])
        if phrase_in_text(term, note_lower)
    ][:2]

    direct_set = set(direct_vocab.get(symptom, []))
    broader_terms = [
        term
        for term in retrieval_vocab.get(symptom, [])
        if term not in direct_set
    ][:3]
    group_b_names = list(group_b.keys())[:MAX_GROUP_B_IN_QUERY]

    variants: Dict[str, str] = {
        "base": f"{symptom} colorectal cancer clinical presentation",
        "note_anchored": " ".join(
            list(dict.fromkeys([symptom] + note_matches + group_b_names))
        ),
        "ontology_expanded": " ".join(
            list(
                dict.fromkeys(
                    [symptom] + broader_terms + ["colorectal cancer"]
                )
            )
        ),
    }

    # Remove duplicate query strings while preserving deterministic names/order.
    unique: Dict[str, str] = {}
    seen: set[str] = set()
    for name, query in variants.items():
        normalized = " ".join(query.split()).strip()
        if normalized and normalized.lower() not in seen:
            unique[name] = normalized
            seen.add(normalized.lower())
    return unique


def retrieve_with_rrf(
    encoder: MedCPTEncoder,
    chunk_vectors: np.ndarray,
    chunks: Sequence[str],
    query_variants: Mapping[str, Mapping[str, str]],
    retrieval_vocab: Mapping[str, Sequence[str]],
    top_k: int = DENSE_TOP_K,
    candidate_k: int = SOURCE_CANDIDATE_K,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Retrieve each query variant independently and fuse rankings per symptom.

    Returns
    -------
    per_symptom_candidates:
        Up to candidate_k fused, phrase-relevant passages per symptom.
    retrieval_audit:
        Query texts, fallback status, and fused-list metadata.
    """
    flat_queries: List[str] = []
    flat_keys: List[Tuple[str, str]] = []
    for symptom in SYMPTOMS:
        variants = query_variants.get(symptom, {})
        for variant_name, query_text in variants.items():
            flat_keys.append((symptom, variant_name))
            flat_queries.append(query_text)

    if not flat_queries:
        return {symptom: [] for symptom in SYMPTOMS}, {}

    query_vectors = encoder.encode_queries(flat_queries)
    all_scores = chunk_vectors @ query_vectors.T

    ranked_lists: Dict[str, Dict[str, List[int]]] = defaultdict(dict)
    score_maps: Dict[str, Dict[str, Dict[int, float]]] = defaultdict(dict)

    for column_index, (symptom, variant_name) in enumerate(flat_keys):
        scores = all_scores[:, column_index]
        top_indices = np.argsort(scores)[::-1][:top_k]
        ranked_lists[symptom][variant_name] = [
            int(index) for index in top_indices
        ]
        score_maps[symptom][variant_name] = {
            int(index): float(scores[index]) for index in top_indices
        }

    per_symptom_candidates: Dict[str, List[Dict[str, Any]]] = {}
    retrieval_audit: Dict[str, Any] = {}

    for symptom in SYMPTOMS:
        fused_scores: Dict[int, float] = defaultdict(float)
        ranks_by_chunk: Dict[int, Dict[str, int]] = defaultdict(dict)
        cosines_by_chunk: Dict[int, Dict[str, float]] = defaultdict(dict)

        for variant_name, ranked_indices in ranked_lists.get(symptom, {}).items():
            for zero_based_rank, chunk_index in enumerate(ranked_indices):
                fused_scores[chunk_index] += 1.0 / (
                    RRF_K + zero_based_rank + 1
                )
                ranks_by_chunk[chunk_index][variant_name] = zero_based_rank + 1
                cosines_by_chunk[chunk_index][variant_name] = score_maps[
                    symptom
                ][variant_name][chunk_index]

        fused_indices = sorted(
            fused_scores,
            key=lambda index: (
                fused_scores[index],
                max(cosines_by_chunk[index].values()),
            ),
            reverse=True,
        )

        fused_passages: List[Dict[str, Any]] = []
        for fused_rank, chunk_index in enumerate(fused_indices, start=1):
            cosine_values = list(cosines_by_chunk[chunk_index].values())
            fused_passages.append(
                {
                    "text": chunks[chunk_index],
                    "chunk_idx": int(chunk_index),
                    "rrf_score": float(fused_scores[chunk_index]),
                    "rrf_rank": fused_rank,
                    "max_cosine": float(max(cosine_values)),
                    "mean_cosine": float(np.mean(cosine_values)),
                    "query_ranks": ranks_by_chunk[chunk_index],
                    "query_cosines": cosines_by_chunk[chunk_index],
                }
            )

        symptom_terms = retrieval_vocab.get(symptom, [])
        relevant = [
            passage
            for passage in fused_passages
            if any(
                phrase_in_text(term, passage["text"])
                for term in symptom_terms
            )
        ]
        relevance_fallback = len(relevant) == 0
        selected_pool = relevant if relevant else fused_passages
        candidates = selected_pool[:candidate_k]
        per_symptom_candidates[symptom] = candidates

        retrieval_audit[symptom] = {
            "query_variants": dict(query_variants.get(symptom, {})),
            "n_query_variants": len(query_variants.get(symptom, {})),
            "n_fused_unique_chunks": len(fused_passages),
            "n_phrase_relevant": len(relevant),
            "phrase_relevance_fallback": relevance_fallback,
            "n_candidates_for_source_filter": len(candidates),
        }

    return per_symptom_candidates, retrieval_audit


def prepare_source_filtered_evidence(
    per_symptom_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Tuple[str, Dict[str, Any], Counter, Counter, int]:
    """
    Apply source-design classification, filtering, prioritization, and audit.

    HIGH, MODERATE, and UNCLASSIFIED are eligible.
    LOW and VERY_LOW are excluded.
    """
    evidence_blocks: List[str] = []
    source_tier_audit: Dict[str, Any] = {}
    classified_counter: Counter = Counter()
    included_counter: Counter = Counter()
    total_included = 0

    for symptom in SYMPTOMS:
        tagged: List[Dict[str, Any]] = []
        for candidate_position, raw_passage in enumerate(
            per_symptom_candidates.get(symptom, []), start=1
        ):
            passage = dict(raw_passage)
            tier, matched_marker = classify_source_tier(passage.get("text", ""))
            passage["source_tier"] = tier
            passage["source_marker"] = matched_marker
            passage["eligible"] = tier in ALLOWED_SOURCE_TIERS
            passage["candidate_position"] = candidate_position
            passage["included"] = False
            classified_counter[tier] += 1
            tagged.append(passage)

        eligible = [passage for passage in tagged if passage["eligible"]]
        eligible.sort(
            key=lambda passage: (
                SOURCE_TIER_PRIORITY[passage["source_tier"]],
                -float(passage.get("rrf_score", 0.0)),
                -float(passage.get("max_cosine", 0.0)),
            )
        )
        included = eligible[:DENSE_KEEP_K]
        included_ids = {id(passage) for passage in included}
        for passage in tagged:
            if id(passage) in included_ids:
                passage["included"] = True
                included_counter[passage["source_tier"]] += 1
                total_included += 1

        audit_rows: List[Dict[str, Any]] = []
        for passage in tagged:
            audit_rows.append(
                {
                    "candidate_position": passage.get("candidate_position"),
                    "chunk_idx": passage.get("chunk_idx"),
                    "rrf_rank": passage.get("rrf_rank"),
                    "rrf_score": passage.get("rrf_score"),
                    "max_cosine": passage.get("max_cosine"),
                    "mean_cosine": passage.get("mean_cosine"),
                    "query_ranks": passage.get("query_ranks", {}),
                    "source_tier": passage.get("source_tier"),
                    "source_marker": passage.get("source_marker"),
                    "eligible": passage.get("eligible", False),
                    "included": passage.get("included", False),
                    "text": passage.get("text", ""),
                }
            )
        source_tier_audit[symptom] = audit_rows

        if included:
            lines = [f"[{symptom}]"]
            for evidence_index, passage in enumerate(included, start=1):
                lines.append(
                    "  [Evidence {index}; SOURCE TIER={tier}; "
                    "RRF={rrf:.6f}; max cosine={cosine:.4f}] {text}".format(
                        index=evidence_index,
                        tier=passage["source_tier"],
                        rrf=float(passage.get("rrf_score", 0.0)),
                        cosine=float(passage.get("max_cosine", 0.0)),
                        text=str(passage.get("text", "")).strip(),
                    )
                )
            evidence_blocks.append("\n".join(lines))
        else:
            evidence_blocks.append(
                f"[{symptom}]\n  No eligible source-tier passage retained."
            )

    evidence_section = "\n\n".join(evidence_blocks)
    return (
        evidence_section,
        source_tier_audit,
        classified_counter,
        included_counter,
        total_included,
    )


def split_note_ros(note_text: str) -> Tuple[str, str]:
    note_lower = note_text.lower()
    ros_header_patterns = [
        r"(?im)^\s*review of systems\s*:??\s*$",
        r"(?im)^\s*ros\s*:??\s*$",
        r"(?im)^\s*r\.o\.s\.\s*:??\s*$",
        r"(?im)^\s*systems review\s*:??\s*$",
        r"(?im)^\s*review of system\s*:??\s*$",
    ]
    starts: List[int] = []
    for pattern in ros_header_patterns:
        match = re.search(pattern, note_text)
        if match:
            starts.append(match.start())
    if not starts:
        # Conservative fallback for inline ROS labels.
        inline = [
            note_lower.find(header)
            for header in ["review of systems", "ros:", "r.o.s."]
            if note_lower.find(header) >= 0
        ]
        if not inline:
            return note_text, ""
        ros_start = min(inline)
    else:
        ros_start = min(starts)

    next_header_pattern = re.compile(
        r"(?im)^\s*(assessment(?: and plan)?|plan|physical exam(?:ination)?|"
        r"medications|allergies|vital signs|past medical history|"
        r"social history|family history|objective|subjective|hpi|"
        r"history of present illness|diagnosis|impression)\s*:??\s*$"
    )
    following = next_header_pattern.search(note_text, pos=ros_start + 5)
    ros_end = following.start() if following else len(note_text)
    non_ros = note_text[:ros_start] + note_text[ros_end:]
    ros_section = note_text[ros_start:ros_end]
    return non_ros, ros_section


def symptom_present_outside_ros(
    note_text: str,
    symptom: str,
    direct_vocab: Mapping[str, Sequence[str]],
) -> bool:
    non_ros, _ = split_note_ros(note_text)
    non_ros_lower = non_ros.lower()
    for term in direct_vocab.get(symptom, []):
        for start, end in find_phrase_spans(term, non_ros_lower):
            if not mention_is_negated(non_ros_lower, start, end):
                return True
    return False


FH_MEMBER_PATTERN = (
    r"\b(?:mother|father|parent|sister|brother|sibling|relative|"
    r"family member|first[- ]degree relative|dad|mom|grandmother|"
    r"grandfather|aunt|uncle|son|daughter)\b"
)
FH_CRC_PATTERN = (
    r"(?:\bcrc\b|\b(?:colorectal|colon|rectal|bowel)\b.{0,20}"
    r"\b(?:cancer|ca|carcinoma)\b)"
)
FH_PATTERNS = [
    rf"{FH_MEMBER_PATTERN}[^.;]{{0,100}}{FH_CRC_PATTERN}",
    rf"{FH_CRC_PATTERN}[^.;]{{0,100}}{FH_MEMBER_PATTERN}",
]


def explicit_family_history_crc(note_text: str) -> bool:
    """Audit only. This function does not change the model label."""
    non_ros, _ = split_note_ros(note_text)
    text = non_ros.lower()
    return any(re.search(pattern, text) for pattern in FH_PATTERNS)

def build_alias_section(direct_vocab: Mapping[str, Sequence[str]]) -> str:
    lines = []
    for symptom in SYMPTOMS:
        terms = list(direct_vocab.get(symptom, []))[:15]
        lines.append(f"{symptom}: {', '.join(terms)}")
    return "\n".join(lines)


def build_group_b_section(group_b: Mapping[str, str]) -> str:
    if not group_b:
        return "  None detected."
    return "\n".join(
        f"  - {name} (matched note phrase: '{surface_form}')"
        for name, surface_form in list(group_b.items())[:MAX_GROUP_B_IN_PROMPT]
    )


def build_main_prompt(
    note_text: str,
    alias_section: str,
    group_b_section: str,
    evidence_section: str,
) -> str:
    key_listing = "\n".join(f'"{key}"' for key in EXPECTED_OUTPUT_KEYS)
    return f"""You are an experienced gastroenterology clinician.
Extract seven target findings from the patient's clinical note.

PATIENT-EVIDENCE REQUIREMENT:
- Patient-specific labels must be based ONLY on the Patient NOTE TEXT.
- Retrieved biomedical passages and terminology lists may help recognize
  equivalent wording, but they CANNOT establish or negate a patient symptom.
- Every positive inference must quote or closely paraphrase evidence from the
  Patient NOTE TEXT. Never quote or paraphrase retrieved literature as the
  patient inference.
- Do not make external assumptions from diagnoses, medications, risk factors,
  or general medical knowledge.

REVIEW OF SYSTEMS (ROS) RULE:
- ROS may contain templated negatives that conflict with narrative sections.
- If ROS denies a symptom but Chief Complaint, HPI, Assessment/Plan, or
  Diagnosis affirms it, answer "Yes" and cite the NON-ROS evidence.
- If the note outside ROS explicitly denies the symptom, answer "No" with high
  confidence and cite the NON-ROS denial.
- If the symptom appears only as a negative in ROS and nowhere else, answer
  "No" with confidence 2.
- Absence of a denial is not evidence that a symptom is present.

FAMILY-HISTORY RULE:
- Answer "Yes" only when the Patient NOTE TEXT explicitly links a biological
  family member to colorectal, colon, rectal, or bowel cancer.
- The patient's personal cancer history is not family history.
- Lynch syndrome, FAP, genetic testing, hereditary-risk discussion, or a
  generic family history of unspecified cancer is insufficient by itself.

OUTPUT RULES:
- Every symptom label must be exactly "Yes" or "No".
- Every confidence must be an integer from 1 to 5.
- Every inference, duration, and Other comments field must be a string.
- Use "N/A" when duration or Other comments are not reported.
- Output ONE flat JSON object. Do not use nested dictionaries.
- Include every key listed below exactly once.
- Output only JSON, without prose or markdown fences.

REQUIRED KEYS:
{key_listing}

PATIENT NOTE TEXT:
<NOTE>
{note_text}
</NOTE>

TERMINOLOGY GUIDANCE:
These direct aliases help recognize equivalent note wording. They are not
patient evidence by themselves.
{alias_section}

RELATED UMLS CONCEPTS DETECTED IN THE NOTE:
These are terminology hints only. A related concept does not establish a
specific target finding.
{group_b_section}

EXTERNAL BIOMEDICAL CONTEXT:
The passages below were selected using MedCPT retrieval, within-symptom RRF,
and a GRADE-inspired source-design screening heuristic. SOURCE TIER labels are
retrieval-screening categories, not formal GRADE certainty ratings.

These passages are terminology context ONLY. They cannot determine a patient
label and cannot be used as an inference excerpt.
{evidence_section}
""".strip()


def build_grounding_retry_prompt(
    note_text: str,
    parsed: Mapping[str, Any],
    weak_symptoms: Sequence[str],
) -> str:
    current: Dict[str, Any] = {}
    required_keys: List[str] = []
    for symptom in weak_symptoms:
        current[symptom] = {
            "answer": parsed.get(symptom),
            "confidence": parsed.get(f"{symptom} confidence"),
            "inference": parsed.get(f"{symptom} inference"),
        }
        required_keys.extend(
            [
                symptom,
                f"{symptom} confidence",
                f"{symptom} inference",
            ]
        )
    return f"""Review only the symptoms listed below.
Use ONLY the patient note. Do not use retrieved literature, terminology lists,
medical assumptions, related diagnoses, medications, or general knowledge.

For each symptom:
- Answer "Yes" only if the note explicitly supports the symptom for this
  patient.
- The inference must quote or closely paraphrase an exact note statement.
- If no patient-specific evidence exists, answer "No" with confidence 2 and
  inference "Not documented in the note".
- For family history of colorectal cancer, "Yes" requires a family member
  linked to colorectal, colon, rectal, or bowel cancer.
- Output one flat JSON object containing exactly these keys:
{json.dumps(required_keys, ensure_ascii=False)}

CURRENT PREDICTIONS REQUIRING REVIEW:
{json.dumps(current, ensure_ascii=False)}

PATIENT NOTE:
<NOTE>
{note_text}
</NOTE>
""".strip()


def build_missing_block_prompt(
    note_text: str, missing_symptoms: Sequence[str]
) -> str:
    required_keys: List[str] = []
    for symptom in missing_symptoms:
        required_keys.extend(
            [
                symptom,
                f"{symptom} confidence",
                f"{symptom} inference",
            ]
        )
    return f"""The following symptom blocks were omitted from a prior JSON
extraction. Regenerate only these blocks using ONLY the patient note.

For each symptom:
- Answer exactly "Yes" or "No".
- Answer "Yes" only if the note explicitly supports the finding.
- Copy or closely paraphrase note evidence in the inference field.
- If no evidence exists, answer "No" with confidence 2 and inference
  "Not documented in the note".
- Confidence must be an integer from 1 to 5.
- Output one flat JSON object containing exactly these keys and nothing else:
{json.dumps(required_keys, ensure_ascii=False)}

PATIENT NOTE:
<NOTE>
{note_text}
</NOTE>
""".strip()


def build_full_note_only_repair_prompt(note_text: str) -> str:
    """Last-resort parse repair; explicitly audited when used."""
    key_listing = "\n".join(f'"{key}"' for key in EXPECTED_OUTPUT_KEYS)
    return f"""Extract all seven target findings using ONLY the patient note.
Return one compact, flat, valid JSON object.

Rules:
- Each symptom label must be exactly "Yes" or "No".
- "Yes" requires explicit patient-specific note evidence.
- Inference must quote or closely paraphrase the note.
- Confidence must be an integer from 1 to 5.
- Use "N/A" for unreported durations and Other comments.
- Do not use nested dictionaries.
- Output only JSON.

Required keys:
{key_listing}

PATIENT NOTE:
<NOTE>
{note_text}
</NOTE>
""".strip()


def strip_code_fences(text: str) -> str:
    value = str(text).strip()
    if value.startswith("```"):
        value = re.sub(
            r"^```(?:json)?", "", value, flags=re.IGNORECASE
        ).strip()
        if value.endswith("```"):
            value = value[:-3].strip()
    return value


def extract_first_json(text: str) -> str:
    """Extract the first balanced JSON object, respecting quoted strings."""
    value = strip_code_fences(text)
    start = value.find("{")
    if start < 0:
        return value
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(value)):
        character = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return value[start : index + 1]
    return value[start:]


def safe_json_loads(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    raw = extract_first_json(str(text))
    attempts = [raw]

    normalized = (
        raw.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    normalized = re.sub(
        r':\s*N/A(\s*[,\n}\]])', r': "N/A"\1', normalized
    )
    normalized = re.sub(
        r':\s*None(\s*[,\n}\]])', r': "None"\1', normalized
    )
    normalized = re.sub(r",\s*([}\]])", r"\1", normalized)
    normalized = re.sub(
        r'("|\d|true|false|null)\s*\n(\s*")', r"\1,\n\2", normalized
    )
    attempts.append(normalized)

    missing_comma_fix = re.sub(
        r'("|\d|true|false|null)(\s*")', r"\1,\2", normalized
    )
    attempts.append(missing_comma_fix)

    for candidate in attempts:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, candidate
        except (json.JSONDecodeError, TypeError):
            continue
    return None, raw


def parse_saved_dict(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed, _ = safe_json_loads(text)
    if isinstance(parsed, dict):
        return parsed
    try:
        literal = ast.literal_eval(text)
        return literal if isinstance(literal, dict) else None
    except (ValueError, SyntaxError):
        return None


def normalize_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"yes", "1", "true", "present", "y"}:
        return "Yes"
    if normalized in {
        "no",
        "0",
        "false",
        "absent",
        "n",
        "not mentioned",
        "not mentioned outside ros",
    }:
        return "No"
    return None


def flatten_parsed(parsed: Any) -> Any:
    """
    Recover nested model outputs without inventing labels.

    A symptom label is recovered only from an explicit answer/presence/result
    field. Confidence and inference values are never converted into labels.
    """
    if not isinstance(parsed, dict):
        return parsed
    result = dict(parsed)
    for symptom in SYMPTOMS:
        value = parsed.get(symptom)
        confidence_key = f"{symptom} confidence"
        inference_key = f"{symptom} inference"

        if isinstance(value, str):
            normalized = normalize_label(value)
            if normalized is not None:
                result[symptom] = normalized
            continue

        if not isinstance(value, dict):
            continue

        explicit_answer: Optional[str] = None
        candidate_keys = [
            symptom,
            symptom.lower(),
            "answer",
            "presence",
            "result",
            "label",
        ]
        for key in candidate_keys:
            if key in value:
                explicit_answer = normalize_label(value[key])
                if explicit_answer is not None:
                    break
        if explicit_answer is not None:
            result[symptom] = explicit_answer
        else:
            result.pop(symptom, None)

        if confidence_key not in result or pd.isna(result.get(confidence_key)):
            for key, nested_value in value.items():
                if "confidence" in str(key).lower():
                    result[confidence_key] = nested_value
                    break

        current_inference = result.get(inference_key)
        current_bad = (
            not isinstance(current_inference, str)
            or current_inference.strip().lower() in BAD_INFERENCE_VALS
            or len(current_inference.strip()) < 5
        )
        if current_bad:
            for key, nested_value in value.items():
                if "inference" in str(key).lower() and isinstance(
                    nested_value, str
                ):
                    result[inference_key] = nested_value
                    break
    return result


def to_num(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else np.nan


def missing_symptom_blocks(parsed: Any) -> List[str]:
    if not isinstance(parsed, dict):
        return list(SYMPTOMS)
    missing = []
    for symptom in SYMPTOMS:
        if normalize_label(parsed.get(symptom)) is None:
            missing.append(symptom)
    return missing


def complete_auxiliary_fields(parsed: Any) -> Any:
    """Normalize auxiliary fields only; never create or change a symptom label.

    The model sometimes emits confidence 0 for an unavailable duration or for
    an empty Other-comments field. The schema permits confidence values 1--5,
    and the prompt defines confidence 1 for unavailable auxiliary information.
    Therefore, missing or out-of-range auxiliary confidences are normalized to
    1. This function never infers a symptom label from an auxiliary field.
    """
    if not isinstance(parsed, dict):
        return parsed

    result = dict(parsed)

    def normalize_aux_confidence(value: Any) -> int | float:
        confidence = to_num(value)
        if np.isnan(confidence) or confidence < 1 or confidence > 5:
            return 1
        # Preserve valid integral values as integers in the saved JSON.
        if float(confidence).is_integer():
            return int(confidence)
        return confidence

    for symptom in SYMPTOMS[:-1]:
        duration = f"Duration of {symptom.lower()}"
        duration_confidence = f"{duration} confidence"
        duration_inference = f"{duration} inference"

        duration_value = result.get(duration)
        if not isinstance(duration_value, str) or not duration_value.strip():
            result[duration] = "N/A"

        result[duration_confidence] = normalize_aux_confidence(
            result.get(duration_confidence)
        )

        inference_value = result.get(duration_inference)
        if not isinstance(inference_value, str) or not inference_value.strip():
            result[duration_inference] = "N/A"

    comments_value = result.get("Other comments")
    if not isinstance(comments_value, str) or not comments_value.strip():
        result["Other comments"] = "N/A"

    result["Other comments confidence"] = normalize_aux_confidence(
        result.get("Other comments confidence")
    )

    comments_inference = result.get("Other comments inference")
    if not isinstance(comments_inference, str) or not comments_inference.strip():
        result["Other comments inference"] = "N/A"

    return result


def missing_expected_keys(parsed: Any) -> List[str]:
    if not isinstance(parsed, dict):
        return list(EXPECTED_OUTPUT_KEYS)
    return [key for key in EXPECTED_OUTPUT_KEYS if key not in parsed]


def invalid_output_values(parsed: Any) -> List[str]:
    if not isinstance(parsed, dict):
        return ["Output is not a JSON object."]
    issues: List[str] = []
    for symptom, confidence_key, inference_key in SYMPTOM_SPECS:
        label = normalize_label(parsed.get(symptom))
        if label is None:
            issues.append(f"{symptom}: invalid label {parsed.get(symptom)!r}")
        confidence = to_num(parsed.get(confidence_key))
        if np.isnan(confidence) or confidence < 1 or confidence > 5:
            issues.append(
                f"{confidence_key}: invalid confidence "
                f"{parsed.get(confidence_key)!r}"
            )
        if not isinstance(parsed.get(inference_key), str):
            issues.append(f"{inference_key}: inference is not a string")

    for key in EXPECTED_OUTPUT_KEYS:
        if key.endswith(" confidence"):
            confidence = to_num(parsed.get(key))
            if np.isnan(confidence) or confidence < 1 or confidence > 5:
                issue = f"{key}: invalid confidence {parsed.get(key)!r}"
                if issue not in issues:
                    issues.append(issue)
        elif key not in SYMPTOMS and key not in [
            f"{symptom} inference" for symptom in SYMPTOMS
        ]:
            # Duration/comment fields should be strings.
            if not isinstance(parsed.get(key), str):
                issues.append(f"{key}: value is not a string")
    return issues

_TOKEN_PATTERN = re.compile(r"\w+|\S")


def simple_tokenize(text: Any) -> List[str]:
    if not isinstance(text, str):
        text = "" if text is None or pd.isna(text) else str(text)
    return _TOKEN_PATTERN.findall(text.lower())


def modified_precision(
    reference_tokens: Sequence[str], hypothesis_tokens: Sequence[str], n: int
) -> float:
    if len(hypothesis_tokens) < n:
        return 0.0
    hypothesis_ngrams = Counter(
        zip(*[hypothesis_tokens[offset:] for offset in range(n)])
    )
    reference_ngrams = Counter(
        zip(*[reference_tokens[offset:] for offset in range(n)])
    )
    matches = sum(
        min(count, reference_ngrams[ngram])
        for ngram, count in hypothesis_ngrams.items()
    )
    total = sum(hypothesis_ngrams.values())
    return 0.0 if total == 0 else matches / total


def compute_bleu_no_bp(reference: Any, hypothesis: Any, max_n: int = 4) -> float:
    reference_tokens = simple_tokenize(reference)
    hypothesis_tokens = simple_tokenize(hypothesis)
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    effective_max_n = min(max_n, len(hypothesis_tokens))
    log_precisions: List[float] = []
    for n in range(1, effective_max_n + 1):
        precision = modified_precision(
            reference_tokens, hypothesis_tokens, n
        )
        if precision == 0.0:
            return 0.0
        log_precisions.append(math.log(precision))
    return math.exp(sum(log_precisions) / len(log_precisions))


def compute_bertscore_batch(
    references: Sequence[str], hypotheses: Sequence[str]
) -> List[float]:
    if not references:
        return []
    if not HAVE_BERTSCORE:
        return [np.nan] * len(references)
    precision, _, _ = bert_score_fn(
        list(hypotheses),
        list(references),
        lang="en",
        model_type="roberta-large",
        rescale_with_baseline=False,
        batch_size=16,
        verbose=False,
    )
    return [float(value) for value in precision]

def load_llama(model_id: str = HF_MODEL_ID) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading extractor model: {model_id}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {total_memory:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=HF_CACHE_DIR
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
    )
    model.eval()
    print("LLaMA extractor loaded")
    return tokenizer, model


def generate(
    prompt: str,
    tokenizer: Any,
    model: Any,
    max_new_tokens: int = MAIN_MAX_NEW_TOKENS,
) -> Tuple[str, bool, bool, int, int]:
    import torch

    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"

    full_token_count = len(
        tokenizer(formatted, add_special_tokens=True)["input_ids"]
    )
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(model.device)
    used_token_count = int(inputs["input_ids"].shape[1])
    input_truncated = full_token_count > used_token_count

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][used_token_count:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    hit_cap = len(new_tokens) >= max_new_tokens
    unbalanced = text.count("{") != text.count("}")
    output_truncated = bool(hit_cap and unbalanced)
    return (
        text,
        output_truncated,
        input_truncated,
        full_token_count,
        used_token_count,
    )


def maybe_truncate(text: Any, max_chars: int = NOTE_CHAR_LIMIT) -> str:
    value = "" if text is None else str(text)
    if len(value) <= max_chars:
        return value
    separator = "\n...[middle of note elided]...\n"
    available = max_chars - len(separator)
    first_half = available // 2
    second_half = available - first_half
    return value[:first_half] + separator + value[-second_half:]


def _row_identifier(row: pd.Series, fallback_index: int) -> Any:
    if "NOTE_ID" in row and pd.notna(row.get("NOTE_ID")):
        return row.get("NOTE_ID")
    return int(row.get("_run_row_index", fallback_index))


def run_inference(
    notes_df: pd.DataFrame,
    encoder: MedCPTEncoder,
    chunk_vectors: np.ndarray,
    chunks: Sequence[str],
    direct_vocab: Mapping[str, Sequence[str]],
    retrieval_vocab: Mapping[str, Sequence[str]],
    group_b_lookup: Mapping[str, str],
    tokenizer: Any,
    model: Any,
    on_schema_failure: str = "warn",
    resume: bool = False,
) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("EXPERIMENT 7 — FULL MedCPT + WITHIN-SYMPTOM RRF RAG EXTRACTOR")
    print(
        f"  Retrieval: top-{DENSE_TOP_K} per query variant; "
        f"RRF k={RRF_K}; classify up to {SOURCE_CANDIDATE_K}; "
        f"retain up to {DENSE_KEEP_K}/symptom"
    )
    print(
        f"  Source tiers admitted: {sorted(ALLOWED_SOURCE_TIERS)}; "
        "LOW and VERY_LOW excluded"
    )
    print("  Patient note: before aliases/UMLS/retrieved context")
    print("  Retrieval label override: NONE")
    print("  Prior-visit memory: NONE")
    print("  Deterministic family-history label gate: NONE (audit only)")
    print("  Decoding: greedy (do_sample=False)")
    print("=" * 78)

    alias_section = build_alias_section(direct_vocab)
    rows: List[Dict[str, Any]] = []
    processed_ids: set[str] = set()

    if resume and os.path.exists(OUT_CHECKPOINT):
        existing = pd.read_csv(OUT_CHECKPOINT)
        if "exp_output_dict" in existing.columns:
            existing["exp_output_dict"] = existing["exp_output_dict"].apply(
                parse_saved_dict
            )
        rows = existing.to_dict("records")
        # _run_row_index is stable even when NOTE_ID is read as float/string.
        if "_run_row_index" in existing.columns:
            processed_ids = {
                str(int(value))
                for value in existing["_run_row_index"].dropna().tolist()
            }
        elif "NOTE_ID" in existing.columns:
            processed_ids = {
                str(value)
                for value in existing["NOTE_ID"].dropna().tolist()
            }
        print(
            f"[RESUME] Loaded {len(rows)} checkpoint rows from "
            f"{OUT_CHECKPOINT}"
        )

    self_corrections = 0
    block_repairs = 0
    compact_retries = 0
    full_note_only_repairs = 0
    nested_recovered = 0
    schema_failures = 0
    token_truncations = 0
    note_char_truncations = 0
    notes_with_zero_evidence = 0
    tier_counter: Counter = Counter()
    included_counter: Counter = Counter()
    parse_failure_log: List[Dict[str, Any]] = []

    iterator = tqdm(
        notes_df.iterrows(),
        total=len(notes_df),
        desc="Exp7 MedCPT+RRF RAG",
    )

    for dataframe_index, row in iterator:
        note_identifier = _row_identifier(row, dataframe_index)
        resume_identifier = str(int(row.get("_run_row_index", dataframe_index)))
        if resume and resume_identifier in processed_ids:
            continue

        original_note_text = str(row.get("Clean_note_text", ""))
        note_char_truncated = len(original_note_text) > NOTE_CHAR_LIMIT
        note_text = maybe_truncate(original_note_text)
        if note_char_truncated:
            note_char_truncations += 1

        group_a, group_b = scan_note_for_umls_concepts(
            note_text, direct_vocab, group_b_lookup
        )
        query_variants = {
            symptom: build_query_variants(
                note_text,
                symptom,
                direct_vocab,
                retrieval_vocab,
                group_b,
            )
            for symptom in SYMPTOMS
        }
        per_symptom_candidates, retrieval_audit = retrieve_with_rrf(
            encoder,
            chunk_vectors,
            chunks,
            query_variants,
            retrieval_vocab,
        )
        (
            evidence_section,
            source_tier_audit,
            note_tier_counts,
            note_included_counts,
            n_included_this_note,
        ) = prepare_source_filtered_evidence(per_symptom_candidates)
        tier_counter.update(note_tier_counts)
        included_counter.update(note_included_counts)
        if n_included_this_note == 0:
            notes_with_zero_evidence += 1

        group_b_section = build_group_b_section(group_b)
        prompt = build_main_prompt(
            note_text,
            alias_section,
            group_b_section,
            evidence_section,
        )

        content = ""
        initial_generation_text = ""
        initial_raw_json = ""
        raw_json = ""
        output_truncated = False
        input_truncated = False
        full_tokens = 0
        used_tokens = 0
        compact_retry_used = False
        full_note_only_repair_used = False

        try:
            (
                content,
                output_truncated,
                input_truncated,
                full_tokens,
                used_tokens,
            ) = generate(prompt, tokenizer, model)
        except Exception as error:
            print(f"  Initial generation error for {note_identifier}: {error}")

        initial_generation_text = content
        parsed, raw_json = safe_json_loads(content)
        initial_raw_json = raw_json

        if input_truncated:
            token_truncations += 1

        # Compact full-RAG retry for truncated or malformed JSON.
        if output_truncated or not isinstance(parsed, dict):
            compact_retry_used = True
            compact_retries += 1
            compact_prompt = prompt + (
                "\n\nIMPORTANT FORMAT RETRY: Keep every inference under "
                "15 words. Return one compact flat JSON object only."
            )
            try:
                (
                    retry_content,
                    _,
                    retry_input_truncated,
                    retry_full_tokens,
                    retry_used_tokens,
                ) = generate(
                    compact_prompt,
                    tokenizer,
                    model,
                    max_new_tokens=MAIN_MAX_NEW_TOKENS,
                )
                retry_parsed, retry_raw = safe_json_loads(retry_content)
                if isinstance(retry_parsed, dict):
                    content = retry_content
                    parsed = retry_parsed
                    raw_json = retry_raw
                    input_truncated = retry_input_truncated
                    full_tokens = retry_full_tokens
                    used_tokens = retry_used_tokens
            except Exception as error:
                print(f"  Compact retry error for {note_identifier}: {error}")

        # Last-resort parse repair is note-only and is explicitly audited.
        if not isinstance(parsed, dict):
            full_note_only_repair_used = True
            full_note_only_repairs += 1
            try:
                repair_content, _, _, _, _ = generate(
                    build_full_note_only_repair_prompt(note_text),
                    tokenizer,
                    model,
                    max_new_tokens=MAIN_MAX_NEW_TOKENS,
                )
                repair_parsed, repair_raw = safe_json_loads(repair_content)
                if isinstance(repair_parsed, dict):
                    parsed = repair_parsed
                    raw_json = repair_raw
            except Exception as error:
                print(
                    f"  Full note-only repair error for {note_identifier}: "
                    f"{error}"
                )

        if not isinstance(parsed, dict):
            parse_failure_log.append(
                {
                    "note_id": note_identifier,
                    "len_chars": len(content),
                    "brace_open": content.count("{"),
                    "brace_close": content.count("}"),
                    "prompt_tokens_full": full_tokens,
                    "prompt_tokens_used": used_tokens,
                    "input_truncated": input_truncated,
                    "compact_retry_used": compact_retry_used,
                    "full_note_only_repair_used": full_note_only_repair_used,
                    "output_tail": content[-500:],
                }
            )

        if isinstance(parsed, dict):
            had_nested = any(
                isinstance(parsed.get(symptom), dict) for symptom in SYMPTOMS
            )
            parsed = flatten_parsed(parsed)
            if had_nested:
                nested_recovered += 1

        # Targeted note-only repair for omitted symptom blocks.
        if isinstance(parsed, dict):
            missing_blocks = missing_symptom_blocks(parsed)
            if missing_blocks:
                block_repairs += 1
                for start in range(0, len(missing_blocks), 4):
                    batch = missing_blocks[start : start + 4]
                    try:
                        repair_content, _, _, _, _ = generate(
                            build_missing_block_prompt(note_text, batch),
                            tokenizer,
                            model,
                            max_new_tokens=RETRY_MAX_NEW_TOKENS,
                        )
                        repair_parsed, _ = safe_json_loads(repair_content)
                        if not isinstance(repair_parsed, dict):
                            continue
                        repair_parsed = flatten_parsed(repair_parsed)
                        for symptom in batch:
                            for key in [
                                symptom,
                                f"{symptom} confidence",
                                f"{symptom} inference",
                            ]:
                                if key in repair_parsed:
                                    parsed[key] = repair_parsed[key]
                    except Exception as error:
                        print(
                            f"  Missing-block repair error for "
                            f"{note_identifier}: {error}"
                        )

        # Note-only grounding retry for weakly grounded Yes predictions.
        if isinstance(parsed, dict):
            weak_symptoms: List[str] = []
            for symptom, _, inference_key in SYMPTOM_SPECS:
                if normalize_label(parsed.get(symptom)) != "Yes":
                    continue
                inference = parsed.get(inference_key)
                inference_missing = (
                    not isinstance(inference, str)
                    or inference.strip().lower() in BAD_INFERENCE_VALS
                    or len(inference.strip()) <= 5
                )
                if inference_missing:
                    weak_symptoms.append(symptom)
                    continue
                if compute_bleu_no_bp(note_text, inference) < BLEU_THRESHOLD:
                    weak_symptoms.append(symptom)

            if weak_symptoms:
                self_corrections += 1
                for start in range(0, len(weak_symptoms), 4):
                    batch = weak_symptoms[start : start + 4]
                    try:
                        retry_content, _, _, _, _ = generate(
                            build_grounding_retry_prompt(note_text, parsed, batch),
                            tokenizer,
                            model,
                            max_new_tokens=RETRY_MAX_NEW_TOKENS,
                        )
                        retry_parsed, _ = safe_json_loads(retry_content)
                        if not isinstance(retry_parsed, dict):
                            continue
                        retry_parsed = flatten_parsed(retry_parsed)
                        for symptom in batch:
                            for key in [
                                symptom,
                                f"{symptom} confidence",
                                f"{symptom} inference",
                            ]:
                                if key in retry_parsed:
                                    parsed[key] = retry_parsed[key]
                    except Exception as error:
                        print(
                            f"  Grounding retry error for "
                            f"{note_identifier}: {error}"
                        )

        parsed = complete_auxiliary_fields(parsed)
        missing_keys = missing_expected_keys(parsed)
        invalid_values = invalid_output_values(parsed)
        schema_valid = bool(
            isinstance(parsed, dict)
            and len(missing_keys) == 0
            and len(invalid_values) == 0
        )
        if not schema_valid:
            schema_failures += 1

        # Audits do not modify labels.
        fh_audit = {
            "predicted_label": (
                normalize_label(parsed.get("Family history of colorectal cancer"))
                if isinstance(parsed, dict)
                else None
            ),
            "explicit_family_member_crc_pattern": explicit_family_history_crc(
                note_text
            ),
            "label_modified_by_rule": False,
        }
        grounding_flags: Dict[str, Any] = {}
        if isinstance(parsed, dict):
            for symptom in SYMPTOMS:
                if normalize_label(parsed.get(symptom)) == "Yes":
                    grounding_flags[symptom] = {
                        "predicted_yes": True,
                        "note_match_outside_ros": symptom_present_outside_ros(
                            note_text, symptom, direct_vocab
                        ),
                    }

        output_row = row.to_dict()
        output_row["exp_output_text_initial"] = initial_generation_text
        output_row["exp_output_raw_initial"] = initial_raw_json
        output_row["exp_output_raw"] = (
            json.dumps(parsed, ensure_ascii=False)
            if isinstance(parsed, dict)
            else raw_json
        )
        output_row["exp_output_dict"] = parsed
        output_row["group_a_found"] = json.dumps(
            group_a, ensure_ascii=False
        )
        output_row["group_b_found"] = json.dumps(
            group_b, ensure_ascii=False
        )
        output_row["query_variants"] = json.dumps(
            query_variants, ensure_ascii=False
        )
        output_row["retrieval_audit"] = json.dumps(
            retrieval_audit, ensure_ascii=False
        )
        output_row["source_tier_audit"] = json.dumps(
            source_tier_audit, ensure_ascii=False
        )
        output_row["n_passages_included"] = n_included_this_note
        output_row["fh_audit"] = json.dumps(fh_audit, ensure_ascii=False)
        output_row["grounding_flags"] = json.dumps(
            grounding_flags, ensure_ascii=False
        )
        output_row["compact_retry_used"] = compact_retry_used
        output_row["full_note_only_repair_used"] = full_note_only_repair_used
        output_row["missing_output_keys"] = json.dumps(
            missing_keys, ensure_ascii=False
        )
        output_row["invalid_output_values"] = json.dumps(
            invalid_values, ensure_ascii=False
        )
        output_row["schema_valid"] = schema_valid
        output_row["note_chars_original"] = len(original_note_text)
        output_row["note_chars_used"] = len(note_text)
        output_row["note_char_truncated"] = note_char_truncated
        output_row["prompt_tokens_full"] = full_tokens
        output_row["prompt_tokens_used"] = used_tokens
        output_row["input_truncated"] = input_truncated
        rows.append(output_row)

        if len(rows) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_csv(OUT_CHECKPOINT, index=False)
            print(
                f"  [{len(rows)}/{len(notes_df)}] "
                f"compact_retries={compact_retries} "
                f"full_repairs={full_note_only_repairs} "
                f"block_repairs={block_repairs} "
                f"grounding_retries={self_corrections} "
                f"schema_failures={schema_failures} "
                f"evidence_included={sum(included_counter.values())}"
            )

    experiment_df = pd.DataFrame(rows)
    experiment_df.to_csv(OUT_RAW, index=False)

    if parse_failure_log:
        with open(OUT_PARSE_FAIL, "w", encoding="utf-8") as handle:
            json.dump(parse_failure_log, handle, indent=2, ensure_ascii=False)
    elif os.path.exists(OUT_PARSE_FAIL):
        os.remove(OUT_PARSE_FAIL)

    schema_mask = experiment_df["schema_valid"].fillna(False).astype(bool)
    valid_df = experiment_df.loc[schema_mask].copy()
    valid_df.to_csv(OUT_VALID, index=False)
    if not schema_mask.all():
        failure_columns = [
            column
            for column in [
                "NOTE_ID",
                "_run_row_index",
                "missing_output_keys",
                "invalid_output_values",
                "exp_output_raw_initial",
                "exp_output_raw",
            ]
            if column in experiment_df.columns
        ]
        experiment_df.loc[~schema_mask, failure_columns].to_csv(
            OUT_SCHEMA_FAIL, index=False
        )

    print("\n" + "=" * 78)
    print("RUN SUMMARY")
    print("=" * 78)
    print(f"Rows written:             {len(experiment_df)} -> {OUT_RAW}")
    print(f"Schema-valid rows:        {int(schema_mask.sum())} -> {OUT_VALID}")
    print(
        f"Parse failures remaining: "
        f"{sum(not isinstance(value, dict) for value in experiment_df['exp_output_dict'])}"
    )
    print(f"Schema failures:          {int((~schema_mask).sum())}")
    print(f"Compact full-RAG retries: {compact_retries}")
    print(f"Full note-only repairs:   {full_note_only_repairs}")
    print(f"Missing-block repairs:    {block_repairs}")
    print(f"Grounding retries:        {self_corrections}")
    print(f"Nested outputs recovered: {nested_recovered}")
    print(f"Input token truncations:  {token_truncations}/{len(experiment_df)}")
    print(
        f"Note-char truncations:    "
        f"{note_char_truncations}/{len(experiment_df)}"
    )

    print("\n" + "-" * 78)
    print("SOURCE-TIER AUDIT")
    print("-" * 78)
    print(f"{'Tier':<16}{'Classified':>14}{'Included':>12}")
    for tier in ALL_SOURCE_TIERS:
        print(
            f"{tier:<16}{tier_counter.get(tier, 0):>14}"
            f"{included_counter.get(tier, 0):>12}"
        )
    total_classified = sum(tier_counter.values())
    total_included = sum(included_counter.values())
    print(f"{'TOTAL':<16}{total_classified:>14}{total_included:>12}")
    print(
        f"Notes with zero included evidence: "
        f"{notes_with_zero_evidence}/{len(experiment_df)}"
    )
    if total_included == 0:
        raise RuntimeError(
            "No passages entered any prompt. This run is not RAG. "
            "Inspect source-tier classification and allowed tiers."
        )

    if not schema_mask.all():
        message = (
            f"{int((~schema_mask).sum())} notes have incomplete schemas. "
            f"See {OUT_SCHEMA_FAIL}."
        )
        if on_schema_failure == "raise":
            raise RuntimeError(message)
        print(f"[SCHEMA WARNING] {message}")

    return experiment_df


def run_metrics(experiment_df: pd.DataFrame, skip_bertscore: bool = False) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("EVIDENCE-GROUNDING METRICS — EXPERIMENT 7")
    print("=" * 78)

    metric_df = experiment_df.copy()
    if "exp_output_dict" not in metric_df.columns:
        source_column = (
            "exp_output_raw" if "exp_output_raw" in metric_df.columns else None
        )
        if source_column is None:
            raise ValueError("No parsed or raw output column is available.")
        metric_df["exp_output_dict"] = metric_df[source_column].apply(
            parse_saved_dict
        )
    else:
        metric_df["exp_output_dict"] = metric_df["exp_output_dict"].apply(
            parse_saved_dict
        )

    metric_df["exp_output_dict"] = metric_df["exp_output_dict"].apply(
        lambda value: complete_auxiliary_fields(flatten_parsed(value))
        if isinstance(value, dict)
        else None
    )

    for symptom, confidence_key, inference_key in SYMPTOM_SPECS:
        metric_df[symptom] = metric_df["exp_output_dict"].apply(
            lambda value, key=symptom: (
                normalize_label(value.get(key))
                if isinstance(value, dict)
                else None
            )
        )
        metric_df[confidence_key] = metric_df["exp_output_dict"].apply(
            lambda value, key=confidence_key: (
                value.get(key, np.nan) if isinstance(value, dict) else np.nan
            )
        )
        metric_df[inference_key] = metric_df["exp_output_dict"].apply(
            lambda value, key=inference_key: (
                value.get(key, "") if isinstance(value, dict) else ""
            )
        )
        metric_df[f"{symptom} Conf_num"] = metric_df[confidence_key].apply(
            to_num
        )

    print("Computing BLEU without brevity penalty...")
    for symptom, _, inference_key in SYMPTOM_SPECS:
        bleu_values = []
        for _, row in metric_df.iterrows():
            inference = row[inference_key]
            if (
                not isinstance(inference, str)
                or inference.strip().lower() in BAD_INFERENCE_VALS
                or len(inference.strip()) < 5
            ):
                bleu_values.append(0.0)
            else:
                bleu_values.append(
                    compute_bleu_no_bp(row["Clean_note_text"], inference)
                )
        metric_df[f"{symptom} BLEU_noBP"] = bleu_values
    print("BLEU complete.")

    for symptom, _, _ in SYMPTOM_SPECS:
        metric_df[f"{symptom} BERT_P"] = np.nan

    if HAVE_BERTSCORE and not skip_bertscore:
        print("Computing BERTScore precision...")
        for symptom, _, inference_key in SYMPTOM_SPECS:
            indices: List[int] = []
            references: List[str] = []
            hypotheses: List[str] = []
            for row_index, row in metric_df.iterrows():
                inference = row[inference_key]
                if (
                    not isinstance(inference, str)
                    or inference.strip().lower() in BAD_INFERENCE_VALS
                    or len(inference.strip()) < 5
                ):
                    continue
                indices.append(row_index)
                references.append(str(row["Clean_note_text"]))
                hypotheses.append(inference)
            scores = compute_bertscore_batch(references, hypotheses)
            for row_index, score in zip(indices, scores):
                metric_df.at[row_index, f"{symptom} BERT_P"] = score
        print("BERTScore complete.")
    elif skip_bertscore:
        print("BERTScore skipped by command-line option.")
    else:
        print("bert_score package not installed; BERTScore skipped.")

    metric_df.to_csv(OUT_METRICS, index=False)

    summary_rows: List[Dict[str, Any]] = []
    for symptom, _, _ in SYMPTOM_SPECS:
        confidence_column = f"{symptom} Conf_num"
        bleu_column = f"{symptom} BLEU_noBP"
        bert_column = f"{symptom} BERT_P"
        yes_mask = metric_df[symptom] == "Yes"
        no_mask = metric_df[symptom] == "No"
        labelled_mask = yes_mask | no_mask
        yes_bleu = metric_df.loc[yes_mask, bleu_column].mean()
        no_bleu = metric_df.loc[no_mask, bleu_column].mean()
        summary_rows.append(
            {
                "Symptom": symptom,
                "Labelled_Count": int(labelled_mask.sum()),
                "Yes_Count": int(yes_mask.sum()),
                "No_Count": int(no_mask.sum()),
                "Yes_Conf_Mean": metric_df.loc[
                    yes_mask, confidence_column
                ].mean(),
                "Yes_Conf_SD": metric_df.loc[
                    yes_mask, confidence_column
                ].std(),
                "No_Conf_Mean": metric_df.loc[
                    no_mask, confidence_column
                ].mean(),
                "No_Conf_SD": metric_df.loc[
                    no_mask, confidence_column
                ].std(),
                "Yes_BLEU_Mean": yes_bleu,
                "Yes_BLEU_SD": metric_df.loc[yes_mask, bleu_column].std(),
                "No_BLEU_Mean": no_bleu,
                "No_BLEU_SD": metric_df.loc[no_mask, bleu_column].std(),
                "BLEU_Gap_Yes_Minus_No": yes_bleu - no_bleu,
                "Yes_BERTP_Mean": metric_df.loc[
                    yes_mask, bert_column
                ].mean(),
                "Yes_BERTP_SD": metric_df.loc[yes_mask, bert_column].std(),
                "No_BERTP_Mean": metric_df.loc[no_mask, bert_column].mean(),
                "No_BERTP_SD": metric_df.loc[no_mask, bert_column].std(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    print("\n" + "-" * 78)
    print("DETECTION COUNTS AND GROUNDING GAPS")
    print("-" * 78)
    print(
        f"{'Symptom':<44}{'Labelled':>10}{'Yes':>7}{'No':>7}{'Gap':>10}"
    )
    for _, summary in summary_df.iterrows():
        gap = summary["BLEU_Gap_Yes_Minus_No"]
        flag = "  <-- LOW GAP WARNING" if pd.notna(gap) and gap < 0.10 else ""
        print(
            f"{summary['Symptom']:<44}"
            f"{int(summary['Labelled_Count']):>10}"
            f"{int(summary['Yes_Count']):>7}"
            f"{int(summary['No_Count']):>7}"
            f"{gap:>10.3f}{flag}"
        )
    print(f"\nMetrics written to: {OUT_METRICS}")
    print(f"Summary written to: {OUT_SUMMARY}")
    return summary_df

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 7: full MedCPT + within-symptom RRF RAG "
            "extractor with GRADE-inspired source-design filtering"
        )
    )
    parser.add_argument(
        "--phase",
        choices=["inference", "metrics", "all"],
        default="all",
    )
    parser.add_argument("--model", default=HF_MODEL_ID)
    parser.add_argument("--max_notes", type=int, default=None)
    parser.add_argument(
        "--on_schema_failure",
        choices=["warn", "raise"],
        default="warn",
    )
    parser.add_argument(
        "--skip_bertscore",
        action="store_true",
        help="Skip BERTScore to reduce runtime/memory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"Resume from {OUT_CHECKPOINT} when it exists.",
    )
    parser.add_argument(
        "--article_batch_size", type=int, default=128
    )
    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("EXPERIMENT 7 CONFIGURATION")
    print("=" * 78)
    print("  Model:          LLaMA-3.1-8B-Instruct")
    print("  Decoding:       greedy (do_sample=False)")
    print("  Components:     direct aliases + UMLS + ROS + MedCPT + RRF")
    print(
        f"  Retrieval:      top-{DENSE_TOP_K} per query variant; "
        f"within-symptom RRF k={RRF_K}"
    )
    print(
        f"  Passage flow:   up to {SOURCE_CANDIDATE_K} candidates -> "
        f"source filter -> up to {DENSE_KEEP_K}/symptom"
    )
    print(
        f"  Source tiers:   admit {sorted(ALLOWED_SOURCE_TIERS)}; "
        "exclude LOW/VERY_LOW"
    )
    print("  Patient labels: note evidence only; no programmatic RAG override")
    print("  FH handling:    prompt rule + audit only; no deterministic gate")
    print(f"  Max note chars: {NOTE_CHAR_LIMIT:,}")
    print(f"  Max input tokens:{MAX_INPUT_TOKENS:,}")
    print(f"  Raw output:     {OUT_RAW}")
    print(f"  Metrics output: {OUT_METRICS}")
    print("=" * 78)

    umls_synonyms = load_umls_synonyms()
    direct_vocab, retrieval_vocab = build_vocabularies(umls_synonyms)
    _, group_b_lookup = load_concept_cache()
    chunks = load_chunks()
    notes_df = load_notes()
    if args.max_notes is not None:
        if args.max_notes <= 0:
            raise ValueError("--max_notes must be a positive integer.")
        notes_df = notes_df.head(args.max_notes).copy()
        print(f"[TEST MODE] Running first {len(notes_df)} notes")

    experiment_df: Optional[pd.DataFrame] = None

    if args.phase in {"inference", "all"}:
        encoder = MedCPTEncoder(HF_CACHE_DIR)
        chunk_vectors = build_dense_index(
            encoder, chunks, batch_size=args.article_batch_size
        )
        tokenizer, model = load_llama(args.model)
        experiment_df = run_inference(
            notes_df=notes_df,
            encoder=encoder,
            chunk_vectors=chunk_vectors,
            chunks=chunks,
            direct_vocab=direct_vocab,
            retrieval_vocab=retrieval_vocab,
            group_b_lookup=group_b_lookup,
            tokenizer=tokenizer,
            model=model,
            on_schema_failure=args.on_schema_failure,
            resume=args.resume,
        )

    if args.phase == "metrics":
        require_file(OUT_RAW, "Experiment 7 raw output")
        experiment_df = pd.read_csv(OUT_RAW)

    if args.phase in {"metrics", "all"}:
        if experiment_df is None:
            raise RuntimeError("No Experiment 7 output is available for metrics.")
        run_metrics(experiment_df, skip_bertscore=args.skip_bertscore)


if __name__ == "__main__":
    main()
