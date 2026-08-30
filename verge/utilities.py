from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


HF_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_CACHE_DIR = os.environ.get(
    "HF_CACHE_DIR", "/lustre/smuexa01/client/users/nikkieh/hf_cache"
)
REVIEWER_MAX_NEW_TOKENS = 512
SCANNER_MAX_NEW_TOKENS = 384

SYMPTOMS = [
    "Abdominal pain", "Rectal bleeding", "Rectal pain",
    "Diarrhea", "Constipation", "Weight loss",
    "Family history of colorectal cancer",
]

SYMPTOM_SYMPTOMS = SYMPTOMS[:-1]   

METADATA_KEYS = ["note_section", "experiencer", "assertion_status"]


DIRECT_ALIASES: Dict[str, List[str]] = {
    "Abdominal pain": [
        "abdominal pain","abd pain","abd pn","abdo pain","stomach pain",
        "stomach ache","belly pain","epigastric pain","epigastric discomfort",
        "epigastric tenderness","abdominal tenderness","tender abdomen",
        "ruq pain","right upper quadrant pain","luq pain",
        "left upper quadrant pain","rlq pain","right lower quadrant pain",
        "llq pain","left lower quadrant pain","periumbilical pain",
        "umbilical pain","suprapubic pain","pelvic pain",
        "lower abdominal pain","upper abdominal pain","abdominal cramping",
        "colicky pain","colicky abdominal pain","sharp abdominal pain",
        "dull abdominal pain","abdominal discomfort","abdominal soreness",
        "c/o abdominal pain","complains of abdominal pain",
        "reports abdominal pain","pain in abdomen",
    ],
    "Rectal bleeding": [
        "rectal bleeding","bleeding per rectum","blood per rectum",
        "blood from rectum","blood in stool","bloody stool",
        "blood on stool","stool with blood","streaks of blood",
        "blood streaked stool","blood on toilet paper","blood when wiping",
        "hematochezia","haematochezia","brbpr",
        "bright red blood per rectum","maroon stools","melena",
        "black tarry stools","positive fobt","positive fit",
        "occult blood","heme positive stool",
    ],
    "Rectal pain": [
        "rectal pain","pain in rectum","painful rectum","anal pain",
        "pain in anus","anorectal pain","proctalgia","proctalgia fugax",
        "rectal discomfort","anal discomfort","rectal soreness",
        "anal soreness","pain with bowel movement","painful bowel movement",
        "painful defecation","dyschezia","pain during defecation",
        "pain after bowel movement","tenesmus","rectal pressure",
        "perianal pain","perirectal pain",
    ],
    "Diarrhea": [
        "diarrhea","diarrhoea","loose stools","loose stool",
        "watery stools","watery stool","liquid stool","runny stool",
        "frequent stools","frequent bowel movements",
        "increased bowel movements","increased stool frequency",
        "multiple loose bms","loose bm","watery bm",
        "explosive diarrhea","bristol 6","bristol 7",
    ],
    "Constipation": [
        "constipation","constipated","hard stools","hard stool",
        "infrequent stools","infrequent bowel movements",
        "decreased stool frequency","decreased bowel movements",
        "no bm","no bowel movement","no stool for",
        "difficulty passing stool","straining","incomplete evacuation",
        "obstipation","fecal impaction","stool impaction",
        "bristol 1","bristol 2","pellet stools",
    ],
    "Weight loss": [
        "weight loss","wt loss","lost weight","losing weight",
        "weight down","weight decreased","decreased weight",
        "unintentional weight loss","unexplained weight loss",
        "involuntary weight loss","clothes fitting looser",
        "cachexia","wasting","cachectic",
    ],
    "Family history of colorectal cancer": [
        "family history of colorectal cancer",
        "family history colorectal cancer",
        "family history of colon cancer","family history colon cancer",
        "family history of rectal cancer","family history of bowel cancer",
        "fh colon cancer","fhx colon cancer","fhx crc","fh crc",
        "crc in family","colon cancer in family",
        "colon cancer runs in family","mother had colon cancer",
        "father had colon cancer","sister had colon cancer",
        "brother had colon cancer","parent had colon cancer",
        "relative with colon cancer",
        "first degree relative with colon cancer","fh bowel cancer",
    ],
}


PRE_NEGATION_CUES = [
    "no","not","denies","deny","denied","without","negative","neg",
    "absent","none","no evidence of","ruled out","rules out","r/o",
    "unremarkable","normal","within normal limits","wnl","non","never",
    "neither","nor","free of","lacks","failed to reveal",
    "no signs of","no symptoms of","no complaints of","no history of",
]

POST_NEGATION_CUES = [
    "absent","denied","negative","not present","resolved",
    "unremarkable","normal","negative for",
]

NEGATION_PRE_PATTERNS = [
    re.compile(r"\b" + re.escape(cue) + r"\b", re.IGNORECASE)
    for cue in PRE_NEGATION_CUES
]

NEGATION_POST_PATTERNS = [
    re.compile(r"\b" + re.escape(cue) + r"\b", re.IGNORECASE)
    for cue in POST_NEGATION_CUES
]

INFERENCE_INDICATORS: Dict[str, Dict[str, List[str]]] = {
    "Abdominal pain": {
        "medication": ["acetaminophen","ibuprofen","naproxen","opioid",
                        "tramadol","morphine","hydrocodone","oxycodone",
                        "ketorolac","analgesic","pain medication"],
        "diagnosis":  ["ibs","irritable bowel","crohn","colitis",
                        "diverticulitis","appendicitis","pancreatitis",
                        "cholecystitis","gastritis","peptic ulcer"],
        "procedure":  ["ct abdomen","abdominal ct","abdominal ultrasound",
                        "kub","abdominal x-ray"],
    },
    "Rectal bleeding": {
        "medication": ["iron supplement","ferrous sulfate"],
        "diagnosis":  ["hemorrhoids","anal fissure","diverticulosis",
                        "colorectal cancer","polyp","angiodysplasia"],
        "procedure":  ["colonoscopy","sigmoidoscopy","egd","endoscopy",
                        "biopsy"],
        "lab":        ["hemoglobin","hematocrit","hgb","hct","cbc",
                        "iron studies","ferritin","fobt","fit test"],
    },
    "Rectal pain": {
        "medication": ["lidocaine","hydrocortisone cream","sitz bath"],
        "diagnosis":  ["hemorrhoids","thrombosed hemorrhoid","anal fissure",
                        "perianal abscess","fistula"],
        "procedure":  ["rectal exam","anoscopy","proctoscopy"],
    },
    "Diarrhea": {
        "medication": ["loperamide","imodium","pepto-bismol","bismuth",
                        "kaolin","antidiarrheal","oral rehydration",
                        "ondansetron"],
        "diagnosis":  ["ibs","colitis","c. diff","cdiff",
                        "clostridium difficile","gastroenteritis",
                        "celiac","malabsorption"],
        "procedure":  ["stool culture","stool studies","c diff test",
                        "ova and parasites"],
        "lab":        ["stool culture","c diff toxin","fecal calprotectin"],
    },
    "Constipation": {
        "medication": ["docusate","senna","miralax","polyethylene glycol",
                        "bisacodyl","lactulose","magnesium citrate",
                        "laxative","stool softener","enema","suppository",
                        "fiber supplement","psyllium","methylcellulose"],
        "diagnosis":  ["ileus","bowel obstruction","megacolon",
                        "hypothyroidism"],
        "procedure":  ["abdominal x-ray","kub"],
    },
    "Weight loss": {
        "medication": ["appetite stimulant","megace","megestrol",
                        "dronabinol","nutritional supplement","ensure",
                        "boost"],
        "diagnosis":  ["malnutrition","cachexia","anorexia","cancer",
                        "malignancy","hyperthyroidism","diabetes"],
        "lab":        ["albumin","prealbumin","bmi","weight"],
    },
    "Family history of colorectal cancer": {
        "diagnosis":  ["lynch syndrome","hnpcc","fap",
                        "familial adenomatous polyposis",
                        "hereditary nonpolyposis"],
        "procedure":  ["genetic testing","genetic counseling",
                        "mismatch repair","microsatellite instability"],
    },
}

# Family-member keywords for H3 and H8
FAMILY_KEYWORDS = [
    "mother","father","parent","sister","brother","sibling","relative",
    "family member","first-degree relative","first degree relative",
    "dad","mom","grandmother","grandfather","aunt","uncle",
    "son","daughter","cousin","niece","nephew",
]

FAMILY_KEYWORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in FAMILY_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

CRC_KEYWORD_PATTERN = re.compile(
    r"\b(?:colorectal|colon|rectal|bowel)\b.{0,20}"
    r"\b(?:cancer|ca|carcinoma|malignancy)\b"
    r"|\bcrc\b",
    re.IGNORECASE,
)

CANCER_KEYWORD_PATTERN = re.compile(
    r"\b(?:cancer|carcinoma|malignancy|tumor|tumour|neoplasm|"
    r"ca|oncology|chemo|radiation|chemotherapy)\b",
    re.IGNORECASE,
)

# Section identification patterns
SECTION_PATTERNS = {
    "hpi": re.compile(
        r"(?:history of present illness|hpi|chief complaint|cc|"
        r"presenting complaint|reason for visit)", re.I
    ),
    "ros": re.compile(
        r"(?:review of systems|ros|r\.o\.s\.?|systems review)", re.I
    ),
    "assessment": re.compile(
        r"(?:assessment|impression|assessment and plan|a/p|a&p)", re.I
    ),
    "plan": re.compile(r"(?:\bplan\b|treatment plan)", re.I),
    "pmh": re.compile(
        r"(?:past medical history|pmh|medical history|pmhx)", re.I
    ),
    "fh": re.compile(r"(?:family history|fh|fhx|family hx)", re.I),
    "pe": re.compile(
        r"(?:physical exam|physical examination|pe|exam|vitals)", re.I
    ),
    "medications": re.compile(
        r"(?:medications|meds|current medications|home medications|"
        r"medication list)", re.I
    ),
}

NARRATIVE_SECTIONS = {"hpi", "cc", "chief complaint", "assessment",
                       "plan", "a/p", "assessment and plan",
                       "hospital course", "diagnosis", "impression"}

TEMPLATE_SECTIONS = {"ros", "review of systems"}

def phrase_in_text(term: str, text: str) -> bool:
    term_val = str(term).strip().lower()
    text_val = str(text).lower()
    if not term_val or len(term_val) < 3:
        return False
    pattern = r"(?<!\w)" + re.escape(term_val) + r"(?!\w)"
    return bool(re.search(pattern, text_val))


def find_phrase_spans(term: str, text: str) -> List[Tuple[int, int]]:
    term_val = str(term).strip().lower()
    text_val = str(text).lower()
    if not term_val or len(term_val) < 3:
        return []
    pat = re.compile(r"(?<!\w)" + re.escape(term_val) + r"(?!\w)")
    return [(m.start(), m.end()) for m in pat.finditer(text_val)]

def _normalize_for_match(text: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", str(text))
    t = t.lower().strip()
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2018", "'").replace("\u2019", "'")
    t = re.sub(r"\s+", " ", t)
    return t


def _tokenize_simple(text: str) -> List[str]:
    return re.findall(r"\w+|\S", text.lower())


def grounding_match(
    quote: str,
    note_text: str,
    fuzzy_threshold: float = 0.90,
    min_tokens_for_subsequence: int = 4,
) -> Dict[str, Any]:
    """Four-level grounding cascade.

    Returns dict with: grounded (bool), match_level (1-4 or None),
    best_fuzzy_score (float), matched_position ([start, end] or None),
    matched_text (str or None).
    """
    result = {
        "grounded": False,
        "match_level": None,
        "best_fuzzy_score": 0.0,
        "matched_position": None,
        "matched_text": None,
    }

    if not quote or not note_text:
        return result

    quote_str = str(quote).strip()
    note_str = str(note_text)

    if len(quote_str) < 3:
        return result

    idx = note_str.find(quote_str)
    if idx >= 0:
        result.update({
            "grounded": True, "match_level": 1,
            "best_fuzzy_score": 1.0,
            "matched_position": [idx, idx + len(quote_str)],
            "matched_text": note_str[idx:idx + len(quote_str)],
        })
        return result

    q_norm = _normalize_for_match(quote_str)
    n_norm = _normalize_for_match(note_str)
    idx2 = n_norm.find(q_norm)
    if idx2 >= 0:
        result.update({
            "grounded": True, "match_level": 2,
            "best_fuzzy_score": 1.0,
            "matched_position": [idx2, idx2 + len(q_norm)],
            "matched_text": q_norm,
        })
        return result

    q_tokens = _tokenize_simple(quote_str)
    n_tokens = _tokenize_simple(note_str)
    if len(q_tokens) >= min_tokens_for_subsequence and len(n_tokens) >= len(q_tokens):
        for i in range(len(n_tokens) - len(q_tokens) + 1):
            if n_tokens[i:i + len(q_tokens)] == q_tokens:
                result.update({
                    "grounded": True, "match_level": 3,
                    "best_fuzzy_score": 1.0,
                    "matched_position": [i, i + len(q_tokens)],
                    "matched_text": " ".join(q_tokens),
                })
                return result

    if len(q_tokens) >= 3:
        q_set = Counter(q_tokens)
        q_len = len(q_tokens)
        window_min = max(1, int(q_len * 0.8))
        window_max = min(len(n_tokens), int(q_len * 1.2))
        best_score = 0.0
        best_pos = -1

        for wsize in range(window_min, window_max + 1):
            for i in range(len(n_tokens) - wsize + 1):
                w_tokens = n_tokens[i:i + wsize]
                w_set = Counter(w_tokens)
                intersection = sum((q_set & w_set).values())
                union = sum((q_set | w_set).values())
                if union > 0:
                    jaccard = intersection / union
                    if jaccard > best_score:
                        best_score = jaccard
                        best_pos = i

        result["best_fuzzy_score"] = best_score
        if best_score >= fuzzy_threshold:
            result.update({
                "grounded": True, "match_level": 4,
                "matched_position": [best_pos, best_pos + q_len],
                "matched_text": " ".join(
                    n_tokens[best_pos:best_pos + q_len]
                ),
            })
            return result

    return result


def check_negation_deterministic(
    quote: str,
    note_text: str,
    feature: str,
    window: int = 40,
) -> Dict[str, Any]:
    """Layers 1-2 of H4: negation cue scan + structural analysis."""
    result = {
        "negation_cues_found": [],
        "scope_contains_target": False,
        "structural_classification": "UNKNOWN",
    }

    text_lower = note_text.lower() if note_text else ""
    quote_lower = quote.lower() if quote else ""

    if not quote_lower:
        return result

    aliases = DIRECT_ALIASES.get(feature, [])
    target_spans = []
    for alias in aliases:
        target_spans.extend(find_phrase_spans(alias, quote_lower))

    if not target_spans:
        target_spans = find_phrase_spans(feature.lower(), quote_lower)

    if not target_spans:
        result["structural_classification"] = "NO_TARGET_FOUND"
        return result

    result["scope_contains_target"] = True

    for start, end in target_spans:
        before = quote_lower[max(0, start - window):start]
        after = quote_lower[end:min(len(quote_lower), end + window)]

        for pattern in NEGATION_PRE_PATTERNS:
            m = pattern.search(before)
            if m:
                result["negation_cues_found"].append(m.group())

        for pattern in NEGATION_POST_PATTERNS:
            m = pattern.search(after)
            if m:
                result["negation_cues_found"].append(m.group())

    result["negation_cues_found"] = list(set(result["negation_cues_found"]))

    if result["negation_cues_found"]:
        if len(result["negation_cues_found"]) >= 2:
            result["structural_classification"] = "POSSIBLE_DOUBLE_NEGATION"
        else:
            result["structural_classification"] = "NEGATED"
    else:
        result["structural_classification"] = "AFFIRMED"

    return result


TEMPLATE_INDICATORS = [
    re.compile(r"[+\-]\s*\w+.*[+\-]\s*\w+", re.I),          
    re.compile(r"(?:positive|negative) for:?\s", re.I),         
    re.compile(r"denies:?\s+\w+(?:,\s*\w+){2,}", re.I),      
    re.compile(r"[\u2611\u2610\u2612☑☐]", re.I),               
    re.compile(r"\[x\]|\[ \]", re.I),                           
]


def detect_template_score(text: str) -> int:
    """Count template indicators in a text passage."""
    score = 0
    for pat in TEMPLATE_INDICATORS:
        if pat.search(text):
            score += 1
    # High density of symptoms in short text
    if len(text) < 200:
        symptom_mentions = len(re.findall(
            r"\b(?:pain|bleeding|nausea|vomiting|diarrhea|constipation|"
            r"fever|chills|fatigue|headache|cough|sob|dyspnea)\b",
            text, re.I
        ))
        if symptom_mentions >= 5:
            score += 1
    return score

def split_note_ros(note_text: str) -> Tuple[str, str]:
    note_lower = note_text.lower()
    ros_patterns = [
        r"(?im)^\s*review of systems\s*:??\s*$",
        r"(?im)^\s*ros\s*:??\s*$",
        r"(?im)^\s*r\.o\.s\.\s*:??\s*$",
        r"(?im)^\s*systems review\s*:??\s*$",
    ]
    starts = []
    for pat in ros_patterns:
        m = re.search(pat, note_text)
        if m:
            starts.append(m.start())

    if not starts:
        inline = [
            note_lower.find(h)
            for h in ["review of systems", "ros:", "r.o.s."]
            if note_lower.find(h) >= 0
        ]
        if not inline:
            return note_text, ""
        ros_start = min(inline)
    else:
        ros_start = min(starts)

    next_pat = re.compile(
        r"(?im)^\s*(assessment(?: and plan)?|plan|physical exam(?:ination)?|"
        r"medications|allergies|vital signs|past medical history|"
        r"social history|family history|objective|subjective|hpi|"
        r"history of present illness|diagnosis|impression)\s*:??\s*$"
    )
    following = next_pat.search(note_text, pos=ros_start + 5)
    ros_end = following.start() if following else len(note_text)
    non_ros = note_text[:ros_start] + note_text[ros_end:]
    ros_section = note_text[ros_start:ros_end]
    return non_ros, ros_section


_LOADED_MODEL = None
_LOADED_TOKENIZER = None


def load_llm():
    """Load the shared LLM once. Returns (tokenizer, model)."""
    global _LOADED_MODEL, _LOADED_TOKENIZER
    if _LOADED_MODEL is not None:
        return _LOADED_TOKENIZER, _LOADED_MODEL

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading shared LLM: {HF_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_ID, trust_remote_code=True, cache_dir=HF_CACHE_DIR
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True, cache_dir=HF_CACHE_DIR,
    )
    model.eval()
    _LOADED_MODEL = model
    _LOADED_TOKENIZER = tokenizer
    print("Shared LLM ready.")
    return tokenizer, model


def generate_text(
    prompt: str,
    max_new_tokens: int = REVIEWER_MAX_NEW_TOKENS,
) -> str:
    """Generate greedy text from the shared LLM."""
    import torch
    tokenizer, model = load_llm()

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

    context_candidates = []
    model_context = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(model_context, int) and 0 < model_context < 10**9:
        context_candidates.append(model_context)
    tokenizer_context = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_context, int) and 0 < tokenizer_context < 10**9:
        context_candidates.append(tokenizer_context)
    context_limit = min(context_candidates) if context_candidates else 131_072
    max_prompt_tokens = max(1, context_limit - int(max_new_tokens))

    if full_token_count > max_prompt_tokens:
        raise RuntimeError(
            "Full Verifier/Refiner prompt exceeds the model context window: "
            f"{full_token_count} prompt tokens > {max_prompt_tokens} allowed "
            f"with max_new_tokens={max_new_tokens}. "
            "Tokenizer truncation is disabled to preserve the whole clinical note."
        )

    inputs = tokenizer(
        formatted, return_tensors="pt", truncation=False,
    ).to(model.device)
    used = int(inputs["input_ids"].shape[1])

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][used:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse JSON from LLM output."""
    raw = str(text).strip()
    # Strip code fences
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.I).strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    # Find first JSON object
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                raw = raw[start:i+1]
                break

    # Normalize
    raw = raw.replace("\u201c",'"').replace("\u201d",'"')
    raw = re.sub(r':\s*N/A(\s*[,\n}\]])', r': "N/A"\1', raw)
    raw = re.sub(r':\s*None(\s*[,\n}\]])', r': "None"\1', raw)
    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def compute_bleu_no_bp(reference: str, hypothesis: str, max_n: int = 4) -> float:
    ref_tokens = _tokenize_simple(str(reference))
    hyp_tokens = _tokenize_simple(str(hypothesis))
    if not ref_tokens or not hyp_tokens:
        return 0.0
    eff_n = min(max_n, len(hyp_tokens))
    log_precs = []
    for n in range(1, eff_n + 1):
        if len(hyp_tokens) < n:
            return 0.0
        hyp_ng = Counter(zip(*[hyp_tokens[o:] for o in range(n)]))
        ref_ng = Counter(zip(*[ref_tokens[o:] for o in range(n)]))
        matches = sum(min(c, ref_ng[ng]) for ng, c in hyp_ng.items())
        total = sum(hyp_ng.values())
        if total == 0 or matches == 0:
            return 0.0
        log_precs.append(math.log(matches / total))
    return math.exp(sum(log_precs) / len(log_precs))
