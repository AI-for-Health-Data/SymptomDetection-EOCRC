from grade_evidence import tag_passages_with_grade, format_passages_with_grade
import os, re, json, math, time, argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ["TRANSFORMERS_NO_TF"]   = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"]               = "0"
os.environ["USE_FLAX"]             = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from bert_score import score as bert_score_fn
    HAVE_BERTSCORE = True
except ModuleNotFoundError:
    HAVE_BERTSCORE = False


PATH         = "rebuilt_notes_by_noteid.csv"
HF_MODEL_ID  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_CACHE_DIR = "/lustre/smuexa01/client/users/nikkieh/hf_cache"
UMLS_JSON    = "umls_synonyms.json"
UMLS_CACHE   = "umls_concept_cache.json"
CHUNKS_PATH  = "pubmed_chunks_sentences.json"

MEDCPT_QUERY_MODEL   = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"

DENSE_TOP_K    = 32
DENSE_KEEP_K   = 5
RRF_K          = 60
BLEU_THRESHOLD = 0.10
RAG_SCORE_MIN  = 0.30
NOTE_MATCH_MIN = 2
DENIAL_CONTEXT = 25

SYMPTOMS = [
    "Abdominal pain", "Rectal bleeding", "Rectal pain",
    "Diarrhea", "Constipation", "Weight loss",
    "Family history of colorectal cancer",
]
SYMPTOM_SPECS = [(s, f"{s} confidence", f"{s} inference") for s in SYMPTOMS]

BAD_INFERENCE_VALS = {
    "", "n/a", "na", "not mentioned", "none", "no inference",
    "not reported", "not applicable", "not stated",
    "not mentioned outside ros", "yes", "no", "present", "absent",
    "only in ros-negative context", "only ros-negative, nowhere else",
}

DENIAL_PREFIXES = [
    "no ","denies ","negative for ","without ",
    "absent ","none ","not ","deny ","no history of ",
]

MANUAL_ALIASES = {
    "Abdominal pain": [
        "abdominal pain","abd pain","abd pn","abdo pain","stomach pain",
        "stomach ache","stomachache","tummy pain","tummy ache","belly pain",
        "bellyache","gut pain","abdominal discomfort","abdominal soreness",
        "abdominal tenderness","tender abdomen","tenderness in abdomen",
        "epigastric pain","epigastric discomfort","epigastric tenderness",
        "ruq pain","right upper quadrant pain","luq pain",
        "left upper quadrant pain","rlq pain","right lower quadrant pain",
        "llq pain","left lower quadrant pain","periumbilical pain",
        "peri-umbilical pain","umbilical pain","suprapubic pain","pelvic pain",
        "lower abdominal pain","upper abdominal pain","cramping",
        "abdominal cramping","colicky pain","colicky abdominal pain",
        "sharp abdominal pain","dull abdominal pain","burning abdominal pain",
        "gnawing pain","pressure in abdomen","fullness","abdominal pressure",
        "bloating","abdominal bloating","distension","abdominal distension",
        "gas pain","gassy","flatulence with pain","dyspepsia","indigestion",
        "heartburn","acid reflux","gerd symptoms","gastritis","peptic ulcer",
        "ulcer pain","c/o abdominal pain","complains of abdominal pain",
        "reports abdominal pain","pain in abdomen",
    ],
    "Rectal bleeding": [
        "rectal bleeding","bleeding per rectum","blood per rectum",
        "blood from rectum","rectal hemorrhage","rectal haemorrhage",
        "rectorrhagia","blood in stool","bloody stool","blood on stool",
        "stool with blood","streaks of blood","blood streaked stool",
        "blood on toilet paper","blood when wiping","hematochezia",
        "haematochezia","brbpr","bright red blood per rectum","maroon stools",
        "melena","black tarry stools","positive fobt","positive fit",
        "occult blood","heme positive stool","hemorrhoids with bleeding",
        "haemorrhoids with bleeding","anal fissure bleeding",
        "fissure with bleeding",
    ],
    "Rectal pain": [
        "rectal pain","pain in rectum","painful rectum","anal pain",
        "pain in anus","anorectal pain","proctalgia","proctalgia fugax",
        "rectal discomfort","anal discomfort","rectal soreness","anal soreness",
        "pain with bowel movement","painful bowel movement",
        "painful defecation","dyschezia","odynochezia",
        "pain during defecation","pain after bowel movement","tenesmus",
        "rectal pressure","feels pressure in rectum","anal fissure pain",
        "fissure pain","hemorrhoid pain","thrombosed hemorrhoid",
        "perianal pain","perirectal pain",
    ],
    "Diarrhea": [
        "diarrhea","diarrhoea","d+","loose stools","loose stool",
        "watery stools","watery stool","liquid stool","runny stool",
        "the runs","frequent stools","frequent bowel movements",
        "increased bowel movements","increased stool frequency",
        "multiple loose bms","loose bm","watery bm","urgent bowel movements",
        "fecal urgency","bowel urgency","explosive diarrhea",
        "bristol 6","bristol 7","soft stools","mushy stools",
        "gastroenteritis","enteritis","colitis with diarrhea",
        "travelers diarrhea","c diff","c. diff","clostridioides difficile",
    ],
    "Constipation": [
        "constipation","constipated","hard stools","hard stool",
        "infrequent stools","infrequent bowel movements",
        "decreased stool frequency","decreased bowel movements",
        "no bm","no bowel movement","no stool for",
        "difficulty passing stool","difficulty stooling",
        "straining","strains with bm","incomplete evacuation",
        "obstipation","fecal impaction","stool impaction",
        "retained stool","stool burden","slow transit constipation",
        "bristol 1","bristol 2","pellet stools","scybalous stools",
    ],
    "Weight loss": [
        "weight loss","wt loss","lost weight","losing weight",
        "weight down","weight decreased","decreased weight",
        "unintentional weight loss","unexplained weight loss",
        "involuntary weight loss","clothes fitting looser",
        "poor weight gain","failure to thrive",
        "cachexia","wasting","cachectic","malnutrition",
        "anorexia","loss of appetite","decreased appetite",
    ],
    "Family history of colorectal cancer": [
        "family history of colorectal cancer","family history colorectal cancer",
        "family history of colon cancer","family history colon cancer",
        "fh colon cancer","fhx colon cancer","fhx crc","fh crc",
        "crc in family","colon ca in family","colon cancer runs in family",
        "mother had colon cancer","father had colon cancer",
        "sister had colon cancer","brother had colon cancer",
        "first degree relative with colon cancer","fdr with colon cancer",
        "lynch syndrome","hnpcc","familial adenomatous polyposis","fap",
        "hereditary colorectal cancer","family history of bowel cancer",
        "fh bowel cancer",
    ],
}


def load_umls_synonyms():
    with open(UMLS_JSON) as f:
        data = json.load(f)
    lowered = {s: [t.lower() for t in terms] for s, terms in data.items()}
    total = sum(len(v) for v in lowered.values())
    print(f"UMLS synonyms (Component C): {total} terms across {len(lowered)} symptoms")
    return lowered

def load_concept_cache():
    if not os.path.exists(UMLS_CACHE):
        print(f"ERROR: {UMLS_CACHE} not found.")
        exit(1)
    with open(UMLS_CACHE) as f:
        cache = json.load(f)
    n_ga = sum(1 for d in cache.values() if d.get("is_group_a", False))
    n_gb = len(cache) - n_ga
    print(f"UMLS cache: {len(cache)} concepts (GROUP A={n_ga}, GROUP B={n_gb})")
    group_b_lookup = {
        term: data.get("name", term)
        for term, data in cache.items()
        if not data.get("is_group_a", False)
    }
    return cache, group_b_lookup

def build_merged_vocabulary(umls_synonyms):
    merged = {}
    for symptom in SYMPTOMS:
        umls_terms   = umls_synonyms.get(symptom, [])
        manual_terms = [t.lower() for t in MANUAL_ALIASES.get(symptom, [])]
        combined = list(dict.fromkeys(manual_terms + umls_terms))
        merged[symptom] = combined
    total  = sum(len(v) for v in merged.values())
    unique = len(set(t for v in merged.values() for t in v))
    print(f"Merged vocabulary (B+C): {unique} unique surface forms ({total} with overlaps)")
    return merged

def load_chunks():
    if not os.path.exists(CHUNKS_PATH):
        print(f"ERROR: {CHUNKS_PATH} not found.")
        exit(1)
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    print(f"Sentence chunks loaded: {len(chunks)}")
    return chunks

def load_notes():
    notes_df = pd.read_csv(PATH)
    for col in ["DATE_OF_SERVIC_DTTM", "SPEC_NOTE_TIME_DTTM", "CONTACT_DATE"]:
        if col in notes_df.columns:
            notes_df[col] = pd.to_datetime(notes_df[col], errors="coerce")
    sort_cols = [c for c in [
        "PAT_ID","PAT_ENC_CSN_ID","DATE_OF_SERVIC_DTTM",
        "SPEC_NOTE_TIME_DTTM","CONTACT_NUM","NOTE_ID"
    ] if c in notes_df.columns]
    if sort_cols:
        notes_df = notes_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    notes_df = notes_df[
        notes_df["Clean_note_text"].str.strip().ne("")
    ].reset_index(drop=True)
    print(f"Notes loaded: {len(notes_df)}")
    return notes_df


# MEDCPT ENCODER
class MedCPTEncoder:
    def __init__(self, cache_dir):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch  = torch
        print(f"Loading MedCPT Article Encoder ({MEDCPT_ARTICLE_MODEL})...")
        self.art_tok = AutoTokenizer.from_pretrained(
            MEDCPT_ARTICLE_MODEL, cache_dir=cache_dir)
        self.art_enc = AutoModel.from_pretrained(
            MEDCPT_ARTICLE_MODEL, cache_dir=cache_dir).to(self.device)
        self.art_enc.eval()
        print(f"Loading MedCPT Query Encoder ({MEDCPT_QUERY_MODEL})...")
        self.qry_tok = AutoTokenizer.from_pretrained(
            MEDCPT_QUERY_MODEL, cache_dir=cache_dir)
        self.qry_enc = AutoModel.from_pretrained(
            MEDCPT_QUERY_MODEL, cache_dir=cache_dir).to(self.device)
        self.qry_enc.eval()
        print("MedCPT encoders ready")

    def _encode(self, model, tokenizer, texts, batch_size=64, max_len=512):
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            with self.torch.no_grad():
                toks = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt"
                ).to(self.device)
                out  = model(**toks)
                vecs = out.last_hidden_state[:, 0, :]        
                vecs = vecs / (vecs.norm(dim=1, keepdim=True) + 1e-8)  # L2
                all_vecs.append(vecs.cpu().float().numpy())
        return np.vstack(all_vecs)

    def encode_articles(self, texts, batch_size=64):
        return self._encode(self.art_enc, self.art_tok, texts, batch_size)

    def encode_queries(self, texts):
        return self._encode(self.qry_enc, self.qry_tok, texts, batch_size=32)

    def cosine_scores(self, query_vec, article_vecs):
        return article_vecs @ query_vec.T

def build_dense_index(encoder, chunks, batch_size=64):
    print(f"Encoding {len(chunks)} chunks with MedCPT Article Encoder...")
    print("  (this runs once, ~2-5 min on A100)")
    t0 = time.time()
    chunk_vecs = encoder.encode_articles(chunks, batch_size=batch_size)
    elapsed = (time.time() - t0) / 60
    print(f"Dense index built: {chunk_vecs.shape} in {elapsed:.1f} min")
    return chunk_vecs

# UMLS CONCEPT SCANNING
def scan_note_for_umls_concepts(note_text, merged_vocab, group_b_lookup):
    note_lower = note_text.lower()
    group_a = {}
    for symptom in SYMPTOMS:
        found = [t for t in merged_vocab[symptom] if t in note_lower]
        if found:
            group_a[symptom] = found[:5]
    group_b = {}
    for term, concept_name in group_b_lookup.items():
        if term in note_lower and concept_name not in group_b:
            group_b[concept_name] = term
    return group_a, group_b

def build_dense_query(note_text, symptom, merged_vocab, group_b):
    note_lower   = note_text.lower()
    note_found   = [t for t in merged_vocab[symptom] if t in note_lower][:3]
    umls_terms   = merged_vocab[symptom][:3]
    gb_names     = list(group_b.keys())[:4]
    parts = []
    parts.extend(note_found[:2] if note_found else [symptom.lower()])
    for t in umls_terms:
        if t not in parts and len(parts) < 5:
            parts.append(t)
    parts.extend(gb_names[:3])
    parts.append("colorectal cancer")
    return " ".join(parts)

# MEDCPT RETRIEVAL WITH RRF
def retrieve_dense(encoder, chunk_vecs, chunks, queries, merged_vocab,
                   top_k=DENSE_TOP_K, keep_k=DENSE_KEEP_K):
    query_texts = list(queries.values())
    query_vecs  = encoder.encode_queries(query_texts)
    all_scores  = chunk_vecs @ query_vecs.T                  # (N, 7)

    per_symptom_passages = {}
    for i, (symptom, query) in enumerate(queries.items()):
        sym_scores = all_scores[:, i]
        top_idx    = np.argsort(sym_scores)[::-1][:top_k]
        passages   = [{"text": chunks[j], "score": float(sym_scores[j]),
                       "rank": r} for r, j in enumerate(top_idx)]
        symptom_terms = merged_vocab.get(symptom, [])[:30]
        relevant = [p for p in passages
                    if any(t in p["text"].lower() for t in symptom_terms)]
        per_symptom_passages[symptom] = (
            relevant[:keep_k] if relevant else passages[:keep_k])

    # Global RRF
    rrf_scores = defaultdict(float)
    for i, symptom in enumerate(queries.keys()):
        sym_scores = all_scores[:, i]
        top_idx    = np.argsort(sym_scores)[::-1][:top_k]
        for rank, idx in enumerate(top_idx):
            rrf_scores[int(idx)] += 1.0 / (RRF_K + rank + 1)

    top_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:keep_k]
    top_passages = [{"text": chunks[idx], "score": score,
                     "chunk_idx": idx} for idx, score in top_rrf]
    return per_symptom_passages, top_passages

def format_passages(passages, symptom=""):
    if not passages:
        return f"  No relevant passages retrieved."
    return "\n".join(
        f"  [Evidence {i+1}] {p['text'][:250]}"
        for i, p in enumerate(passages))

# ROS PROGRAMMATIC ISOLATION
def split_note_ros(note_text):
    note_lower  = note_text.lower()
    ros_headers = ["review of systems","ros:","ros ","r.o.s.",
                   "systems review","review of system"]
    ros_start   = len(note_lower)
    for h in ros_headers:
        idx = note_lower.find(h)
        if 0 <= idx < ros_start:
            ros_start = idx
    if ros_start == len(note_lower):
        return note_text, ""
    next_headers = ["assessment","plan:","physical exam","medications",
                    "allergies","vital signs","physical examination",
                    "past medical","social history","family history",
                    "objective","subjective","hpi","history of present"]
    ros_end = len(note_lower)
    for h in next_headers:
        idx = note_lower.find(h, ros_start + 5)
        if ros_start < idx < ros_end:
            ros_end = idx
    non_ros    = note_text[:ros_start] + note_text[ros_end:]
    ros_section = note_text[ros_start:ros_end]
    return non_ros, ros_section

def symptom_present_outside_ros(note_text, symptom, merged_vocab):
    non_ros, _ = split_note_ros(note_text)
    note_lower  = non_ros.lower()
    for term in merged_vocab.get(symptom, []):
        idx = 0
        while True:
            idx = note_lower.find(term, idx)
            if idx == -1:
                break
            ctx = note_lower[max(0, idx - DENIAL_CONTEXT):idx]
            if not any(p in ctx for p in DENIAL_PREFIXES):
                return True
            idx += 1
    return False

PROMPT_TEMPLATE = """You are an experienced gastroenterology clinician.
Analyze the patient's clinical note and extract information
ONLY from the note text (no external assumptions).

REVIEW OF SYSTEMS (ROS) RULE:
- The ROS section often contains templated negatives (e.g., "no abdominal pain",
  "negative for diarrhea") that can conflict with the rest of the note.
- Do NOT use ROS-negative statements as evidence to answer "No".
- If ROS says "no X" but other parts (Chief Complaint, HPI, Assessment/Plan,
  Diagnosis) indicate X, then answer "Yes" using NON-ROS evidence for inference.
- If the symptom is not mentioned outside ROS at all, answer "No" with
  low confidence (2), and inference can be "Not mentioned outside ROS".
- If the note (outside ROS) explicitly denies a symptom
  (e.g., in HPI: "patient denies rectal bleeding"), then answer "No"
  with high confidence and use that NON-ROS denial as inference.

GROUP B: EXTRA UMLS CLINICAL CONCEPTS FOUND IN THIS NOTE:
{GROUP_B_SECTION}

RETRIEVED BIOMEDICAL EVIDENCE (MedCPT semantic retrieval, GRADE-rated):
Passages are labeled with their evidence quality level following the GRADE
framework: HIGH (systematic reviews, RCTs), MODERATE (cohort studies),
LOW (narrative reviews), VERY_LOW (case reports, expert opinion).
Prioritize higher-GRADE evidence when interpreting ambiguous findings.
IMPORTANT: Cite ONLY from the NOTE TEXT below, NOT from these passages.
{EVIDENCE_SECTION}

SYNONYM GUIDANCE: 
{ALIAS_SECTION}

Rules:
- Clearly present outside ROS -> "Yes" (conf 4-5)
- Explicitly denied outside ROS -> "No" (conf 4-5)
- Only ROS-negative, nowhere else -> "No" (conf 2)
- Use "N/A" for duration if not reported.
- Output ONLY valid JSON. No prose, no markdown.

=== CRITICAL JSON FORMAT RULES ===
Output ONE flat JSON object. Every key maps to a string or number directly.
Do NOT use nested objects or sub-dictionaries as values.

CORRECT format example:
{{"Abdominal pain": "Yes", "Abdominal pain confidence": 4,
  "Abdominal pain inference": "patient reports crampy abdominal pain x 3 weeks",
  "Duration of abdominal pain": "3 weeks", ...}}

WRONG format (DO NOT DO THIS):
{{"Abdominal pain": {{"confidence": 4, "inference": "yes"}} }}
{{"Abdominal pain": {{"abdominal pain": "yes", "abdominal pain confidence": 4}} }}
===================================

Return JSON with exactly these keys (all 7 symptoms):
"Abdominal pain", "Abdominal pain confidence", "Abdominal pain inference",
"Duration of abdominal pain", "Duration of abdominal pain confidence", "Duration of abdominal pain inference",
"Rectal bleeding", "Rectal bleeding confidence", "Rectal bleeding inference",
"Duration of rectal bleeding", "Duration of rectal bleeding confidence", "Duration of rectal bleeding inference",
"Rectal pain", "Rectal pain confidence", "Rectal pain inference",
"Duration of rectal pain", "Duration of rectal pain confidence", "Duration of rectal pain inference",
"Diarrhea", "Diarrhea confidence", "Diarrhea inference",
"Duration of diarrhea", "Duration of diarrhea confidence", "Duration of diarrhea inference",
"Constipation", "Constipation confidence", "Constipation inference",
"Duration of constipation", "Duration of constipation confidence", "Duration of constipation inference",
"Weight loss", "Weight loss confidence", "Weight loss inference",
"Duration of weight loss", "Duration of weight loss confidence", "Duration of weight loss inference",
"Family history of colorectal cancer", "Family history of colorectal cancer confidence",
"Family history of colorectal cancer inference",
"Other comments", "Other comments confidence", "Other comments inference"

Patient NOTE TEXT:
<<NOTE_TEXT>>""".strip()

def build_alias_section(merged_vocab):
    lines = []
    for symptom in SYMPTOMS:
        terms = merged_vocab[symptom]
        lines.append(f"{symptom}: {', '.join(terms[:40])}")
    return "\n".join(lines)

def strip_code_fences(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
        if s.endswith("```"): s = s[:-3].strip()
    return s

def extract_first_json(s):
    s     = strip_code_fences(s)
    start = s.find("{")
    if start == -1: return s
    depth = 0
    for i in range(start, len(s)):
        if   s[i] == "{": depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0: return s[start:i+1]
    return s

def safe_json_loads(s):
    s0 = extract_first_json(s)
    try: return json.loads(s0), s0
    except: pass
    s1 = s0
    s1 = re.sub(r'("|\d|true|false|null)\s*\n(\s*")', r'\1,\n\2', s1)
    s1 = re.sub(r",\s*([}\]])", r"\1", s1)
    s1 = (s1.replace("\u201c",'"').replace("\u201d",'"')
             .replace("\u2018","'").replace("\u2019","'"))
    s1 = re.sub(r':\s*N/A(\s*[,\n}\]])',  r': "N/A"\1', s1)
    s1 = re.sub(r':\s*None(\s*[,\n}\]])', r': "None"\1', s1)
    try: return json.loads(s1), s1
    except: pass
    s2 = re.sub(r'("|\d|true|false|null)(\s*")', r'\1,\2', s1)
    try: return json.loads(s2), s2
    except: return None, s0

def flatten_parsed(d):
    if not isinstance(d, dict):
        return d
    result = dict(d)

    for sym in SYMPTOMS:
        val       = d.get(sym)
        sym_lower = sym.lower()
        conf_key  = sym + " confidence"
        inf_key   = sym + " inference"

        # Already flat string — normalize
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ("yes","1","true","present","y"):
                result[sym] = "Yes"
            elif v in ("no","0","false","absent","n",
                       "not mentioned outside ros","not mentioned"):
                result[sym] = "No"
            continue

        if not isinstance(val, dict):
            continue  # int, None, etc — skip

        inner  = val
        answer = None

        for try_key in [sym, sym_lower, "answer", "presence", "result"]:
            if try_key in inner:
                v = str(inner[try_key]).strip().lower()
                if v in ("yes","1","true","present","y"):
                    answer = "Yes"; break
                elif v in ("no","0","false","absent","n"):
                    answer = "No";  break

        if not answer:
            inner_conf = None
            for k, v in inner.items():
                if "confidence" in k.lower() and sym_lower in k.lower():
                    try: inner_conf = float(v)
                    except: pass
                    break
            if inner_conf is not None:
                answer = "Yes" if inner_conf >= 4 else "No"

        if not answer:
            for k, v in inner.items():
                if "inference" in k.lower() and sym_lower in k.lower():
                    v_str = str(v).strip().lower()
                    if v_str == "yes":
                        answer = "Yes"; break
                    elif v_str == "no":
                        answer = "No";  break

        result[sym] = answer if answer else "No"

        outer_conf = result.get(conf_key)
        if outer_conf is None or (isinstance(outer_conf, float) and
                                   np.isnan(outer_conf)):
            for k, v in inner.items():
                if "confidence" in k.lower() and sym_lower in k.lower():
                    result[conf_key] = v
                    break

        outer_inf = result.get(inf_key, "")
        if (not isinstance(outer_inf, str)
                or outer_inf.strip().lower() in BAD_INFERENCE_VALS
                or len(outer_inf.strip()) < 10):
            for k, v in inner.items():
                if "inference" in k.lower() and sym_lower in k.lower():
                    v_str = str(v).strip()
                    if (v_str.lower() not in BAD_INFERENCE_VALS
                            and len(v_str) > 8):
                        result[inf_key] = v_str
                    break
    return result

def normalize_answer(x):
    if pd.isna(x): return ""
    return str(x).strip().lower()

def to_num(x):
    if pd.isna(x): return np.nan
    m = re.search(r"-?\d+(?:\.\d+)?", str(x))
    return float(m.group()) if m else np.nan

_token_pat = re.compile(r"\w+|\S")

def simple_tokenize(text):
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    return _token_pat.findall(text.lower())

def modified_precision(ref_tokens, hyp_tokens, n):
    if len(hyp_tokens) < n: return 0.0
    hyp_ngrams = Counter(zip(*[hyp_tokens[i:] for i in range(n)]))
    ref_ngrams  = Counter(zip(*[ref_tokens[i:]  for i in range(n)]))
    match = sum(min(c, ref_ngrams[ng]) for ng, c in hyp_ngrams.items())
    total = sum(hyp_ngrams.values())
    return 0.0 if total == 0 else match / total

def compute_bleu_no_bp(ref, hyp, max_n=4):
    ref_toks = simple_tokenize(ref)
    hyp_toks = simple_tokenize(hyp)
    if not ref_toks or not hyp_toks: return 0.0
    max_n    = min(max_n, len(hyp_toks))
    log_precs = []
    for n in range(1, max_n + 1):
        p = modified_precision(ref_toks, hyp_toks, n)
        if p == 0.0: return 0.0
        log_precs.append(math.log(p))
    return math.exp(sum(log_precs) / len(log_precs))

def compute_bertscore_batch(refs, hyps):
    if not refs: return []
    P, R, F1 = bert_score_fn(
        hyps, refs, lang="en",
        rescale_with_baseline=False,
        batch_size=16, verbose=False)
    return [float(x) for x in P]


def load_llama(model_id=HF_MODEL_ID):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"\nLoading: {model_id}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {round(torch.cuda.get_device_properties(0).total_memory/1e9,1)} GB")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=HF_CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="float16", device_map="auto",
        trust_remote_code=True, cache_dir=HF_CACHE_DIR)
    model.eval()
    print("LLaMA loaded")
    return tokenizer, model

def generate(prompt, tokenizer, model, max_new_tokens=1536):
    import torch
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = tokenizer(
        formatted, return_tensors="pt",
        truncation=True, max_length=7000).to(model.device)   # leave room for output
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # truncation flag: hit the cap AND JSON braces unbalanced
    hit_cap = (len(new_tokens) >= max_new_tokens)
    unbalanced = text.count("{") != text.count("}")
    return text, (hit_cap and unbalanced)

def maybe_truncate(text, max_chars=12000):
    if text is None: return ""
    t = str(text)
    if len(t) <= max_chars: return t
    return t[:max_chars//2] + "\n...\n" + t[-(max_chars//2):]

def detect_dense_contradiction(llm_output, passages, symptom,
                                merged_vocab, note_text):
    if str(llm_output.get(symptom, "")).strip().lower() != "no":
        return False, None
    note_lower   = note_text.lower()
    all_terms    = merged_vocab.get(symptom, [])
    note_matches = [t for t in all_terms if t in note_lower]
    if len(note_matches) < NOTE_MATCH_MIN:
        return False, None
    non_denied = []
    for term in note_matches:
        idx = 0
        while True:
            idx = note_lower.find(term, idx)
            if idx == -1: break
            ctx = note_lower[max(0, idx - DENIAL_CONTEXT):idx]
            if not any(p in ctx for p in DENIAL_PREFIXES):
                non_denied.append(term)
                break
            idx += 1
    if not non_denied:
        return False, None
    for p in passages:
        if p.get("score", 0) >= RAG_SCORE_MIN:
            return True, p["text"][:200]
    return False, None


def run_inference(notes_df, encoder, chunk_vecs, chunks,
                  merged_vocab, group_b_lookup, tokenizer, model):
    print("\n" + "="*65)
    print("EXPERIMENT 7 — ROS + UMLS + MedCPT RAG (CORRECTED)")
    print("  Retrieval:  MedCPT dense semantic (cosine similarity)")
    print("  Fusion:     RRF across 7 symptom queries (k=60)")
    print("  Override:   cosine >= 0.30 + neuro-symbolic ROS gate")
    print("  JSON:       Flat format enforced with example in prompt")
    print("  Recovery:   flatten_parsed() on any remaining nested output")
    print("="*65)

    alias_section  = build_alias_section(merged_vocab)
    patient_memory = {}
    rows           = []
    rag_overrides  = 0
    self_corrections = 0
    nested_recovered = 0

    for idx, row in tqdm(notes_df.iterrows(), total=len(notes_df),
                         desc="Exp7 MedCPT RAG (fixed)"):
        note_text = maybe_truncate(row["Clean_note_text"])
        pat_id    = row.get("PAT_ID", None)

        prior_context = ""
        if pat_id and pat_id in patient_memory:
            prior_pos = [s for s, v in patient_memory[pat_id].items()
                         if v == "yes"]
            if prior_pos:
                prior_context = (
                    f"PRIOR VISIT: Patient previously reported: "
                    f"{', '.join(prior_pos)}.\n\n")

        group_a, group_b = scan_note_for_umls_concepts(
            note_text, merged_vocab, group_b_lookup)

        queries = {}
        for symptom in SYMPTOMS:
            queries[symptom] = build_dense_query(
                note_text, symptom, merged_vocab, group_b)

        per_symptom_passages, top_passages = retrieve_dense(
            encoder, chunk_vecs, chunks, queries, merged_vocab)

        if group_b:
            gb_lines = [
                f"  - {name} (found as: '{form}')"
                for name, form in list(group_b.items())[:15]
            ]
            group_b_section = prior_context + "\n".join(gb_lines)
        else:
            group_b_section = prior_context + "  None detected."

        # FUSION FIX: inject each symptom's OWN retrieved passages, not the global RRF pool
        _ev_blocks = []
        for _sym in SYMPTOMS:
            _ps = per_symptom_passages.get(_sym, [])
            if _ps:
                _ps = tag_passages_with_grade(_ps)
                _ev_blocks.append(f"[{_sym}]\n" + format_passages_with_grade(_ps))
        evidence_section = ("\n".join(_ev_blocks) if _ev_blocks
                            else "  No relevant passages retrieved.")
        prompt = PROMPT_TEMPLATE
        prompt = prompt.replace("{GROUP_B_SECTION}",  group_b_section)
        prompt = prompt.replace("{EVIDENCE_SECTION}", evidence_section)
        prompt = prompt.replace("{ALIAS_SECTION}",    alias_section)
        prompt = prompt.replace("<<NOTE_TEXT>>",      note_text)

        try:
            content, truncated = generate(prompt, tokenizer, model)
        except Exception as e:
            content = ""
            truncated = False        # <-- ADD THIS LINE
            print(f"  Error note {idx}: {e}")

        if truncated:
            # output was cut off mid-JSON; retry once asking for BRIEF inferences
            brief = prompt + ("\n\nIMPORTANT: keep each 'inference' under 15 words so the "
                              "full JSON fits. Output ONLY compact valid JSON.")
            content, _ = generate(brief, tokenizer, model, max_new_tokens=1536)

        parsed, raw_json = safe_json_loads(content)

        if not parsed:
            globals().setdefault("_PARSE_FAIL_LOG", [])
            globals()["_PARSE_FAIL_LOG"].append({
                "note_id": row.get("NOTE_ID", idx),
                "len_chars": len(content),
                "brace_open": content.count("{"), "brace_close": content.count("}"),
                "tail": content[-120:].replace("\n"," ")})

        if parsed and isinstance(parsed, dict):
            needs_flatten = any(
                isinstance(parsed.get(sym), dict) for sym in SYMPTOMS)
            if needs_flatten:
                parsed = flatten_parsed(parsed)
                nested_recovered += 1

        if parsed:
            weak = []
            for symptom, conf_key, inf_key in SYMPTOM_SPECS:
                # Only check Yes predictions — No predictions have
                # low BLEU by design (short denial phrases)
                if str(parsed.get(symptom,"")).strip().lower() != "yes":
                    continue
                inf = parsed.get(inf_key, "")
                if isinstance(inf, str) and "[DENSE-OVERRIDE]" in inf:
                    inf = inf.replace("[DENSE-OVERRIDE] ","")
                if (isinstance(inf, str)
                        and inf.strip().lower() not in BAD_INFERENCE_VALS
                        and len(inf.strip()) > 5):
                    if compute_bleu_no_bp(note_text, inf) < BLEU_THRESHOLD:
                        weak.append(symptom)

            if weak and len(weak) <= 4:
                self_corrections += 1
                gb_keys = list(group_b.keys())[:3]
                retry_queries = {}
                for symptom in SYMPTOMS:
                    if symptom in weak:
                        retry_queries[symptom] = (
                            symptom.lower() + " " +
                            " ".join(gb_keys) +
                            " colorectal cancer EOCRC early onset "
                            "clinical presentation patient documented")
                    else:
                        retry_queries[symptom] = queries[symptom]

                _, retry_passages = retrieve_dense(
                    encoder, chunk_vecs, chunks,
                    retry_queries, merged_vocab)
                retry_passages = tag_passages_with_grade(retry_passages)
                retry_evidence = format_passages_with_grade(retry_passages)

                retry_prompt = PROMPT_TEMPLATE
                retry_prompt = retry_prompt.replace(
                    "{GROUP_B_SECTION}", group_b_section)
                retry_prompt = retry_prompt.replace(
                    "{EVIDENCE_SECTION}", retry_evidence)
                retry_prompt = retry_prompt.replace(
                    "{ALIAS_SECTION}", alias_section)
                retry_prompt = retry_prompt.replace(
                    "<<NOTE_TEXT>>", note_text)

                try:
                    rc, _ = generate(retry_prompt, tokenizer, model)
                    rp, rr = safe_json_loads(rc)
                    if rp:
                        if isinstance(rp, dict):
                            needs_flatten = any(
                                isinstance(rp.get(sym), dict)
                                for sym in SYMPTOMS)
                            if needs_flatten:
                                rp = flatten_parsed(rp)
                        parsed   = rp
                        raw_json = rr
                except Exception as e:
                    print(f"  Retry error {idx}: {e}")

        if parsed:
            for symptom in SYMPTOMS:
                contr, evidence = detect_dense_contradiction(
                    parsed, per_symptom_passages.get(symptom, []),
                    symptom, merged_vocab, note_text)
                if contr:
                    rag_overrides += 1
                    parsed[symptom] = "Yes"
                    parsed[f"{symptom} confidence"] = 3
                    parsed[f"{symptom} inference"]  = (
                        f"[DENSE-OVERRIDE] {evidence[:150]}")

        if parsed and pat_id:
            if pat_id not in patient_memory:
                patient_memory[pat_id] = {}
            for symptom in SYMPTOMS:
                ans = normalize_answer(parsed.get(symptom, ""))
                if ans in ("yes", "no"):
                    patient_memory[pat_id][symptom] = ans

        out = row.to_dict()
        out["exp_output_raw"]  = raw_json
        out["exp_output_dict"] = parsed
        out["group_a_found"]   = json.dumps(group_a)
        out["group_b_found"]   = json.dumps(group_b)
        rows.append(out)

        if (idx + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv("exp7_checkpoint.csv", index=False)
            print(f"  [{idx+1}/{len(notes_df)}] "
                  f"overrides={rag_overrides} "
                  f"corrections={self_corrections} "
                  f"nested_recovered={nested_recovered}")

    exp_df = pd.DataFrame(rows)
    exp_df.to_csv(os.environ.get("EXP7_RAW","exp7_outputs_raw.csv"), index=False)
    n_failed = exp_df["exp_output_dict"].apply(
        lambda x: not isinstance(x, dict)).sum()

    print(f"\nexp7_outputs_raw.csv ({len(exp_df)} rows)")
    print(f"Parse failures:    {n_failed}/{len(exp_df)}")
    print(f"Dense overrides:   {rag_overrides}")
    print(f"Self-corrections:  {self_corrections}")
    print(f"Nested recovered:  {nested_recovered}")

    valid_df = exp_df[
        exp_df["exp_output_dict"].apply(lambda x: isinstance(x, dict))
    ].copy().reset_index(drop=True)
    print(f"Valid rows: {len(valid_df)}/{len(exp_df)}")
    _fl = globals().get("_PARSE_FAIL_LOG", [])
    if _fl:
        import json as _json
        _json.dump(_fl, open("exp7_parse_failures.json","w"), indent=2)
        _ntrunc = sum(1 for x in _fl if x["brace_open"] != x["brace_close"])
        print(f"[PARSE] {len(_fl)} failures; {_ntrunc} look truncated (unbalanced braces)")
    return valid_df

def run_metrics(valid_df):
    print("\n" + "="*65)
    print("METRICS — Experiment 7: MedCPT RAG (corrected)")
    print("="*65)

    metric_df = valid_df.copy()

    for symptom, conf_key, inf_key in SYMPTOM_SPECS:
        metric_df[symptom]  = metric_df["exp_output_dict"].apply(
            lambda d, s=symptom:  d.get(s, "")     if isinstance(d, dict) else "")
        metric_df[conf_key] = metric_df["exp_output_dict"].apply(
            lambda d, k=conf_key: d.get(k, np.nan) if isinstance(d, dict) else np.nan)
        metric_df[inf_key]  = metric_df["exp_output_dict"].apply(
            lambda d, k=inf_key:  d.get(k, "")     if isinstance(d, dict) else "")
        metric_df[f"{symptom} Conf_num"] = metric_df[conf_key].apply(to_num)

    print("Computing BLEU...")
    for symptom, _, inf_key in SYMPTOM_SPECS:
        bleu_col = f"{symptom} BLEU_noBP"
        vals = []
        for _, row in metric_df.iterrows():
            hyp = row[inf_key]
            if isinstance(hyp, str) and "[DENSE-OVERRIDE]" in hyp:
                hyp = hyp.replace("[DENSE-OVERRIDE] ", "")
            # Skip trivial inference values
            if (not isinstance(hyp, str)
                    or hyp.strip().lower() in BAD_INFERENCE_VALS
                    or len(hyp.strip()) < 10):
                vals.append(0.0)
                continue
            ref = row["Clean_note_text"]
            vals.append(compute_bleu_no_bp(ref, hyp))
        metric_df[bleu_col] = vals
    print("BLEU done.")

    for symptom, _, _ in SYMPTOM_SPECS:
        metric_df[f"{symptom} BERT_P"] = np.nan

    if HAVE_BERTSCORE:
        print("Computing BERTScore...")
        for symptom, _, inf_key in SYMPTOM_SPECS:
            bert_col  = f"{symptom} BERT_P"
            idxs, refs, hyps = [], [], []
            for i, row in metric_df.iterrows():
                hyp = row[inf_key]
                if isinstance(hyp, str) and "[DENSE-OVERRIDE]" in hyp:
                    hyp = hyp.replace("[DENSE-OVERRIDE] ", "")
                ref = row["Clean_note_text"]
                if (not isinstance(hyp, str)
                        or hyp.strip().lower() in BAD_INFERENCE_VALS
                        or len(hyp.strip()) < 10):
                    continue
                idxs.append(i); refs.append(ref); hyps.append(hyp)
            if idxs:
                bert_vals = compute_bertscore_batch(refs, hyps)
                for i, val in zip(idxs, bert_vals):
                    metric_df.at[i, bert_col] = val
        print("BERTScore done.")

    metric_df.to_csv(os.environ.get("EXP7_METRICS","exp7_note_level_metrics.csv"), index=False)

    summary_rows = []
    for symptom, conf_key, inf_key in SYMPTOM_SPECS:
        conf_col = f"{symptom} Conf_num"
        bleu_col = f"{symptom} BLEU_noBP"
        bert_col = f"{symptom} BERT_P"
        labels   = metric_df[symptom].astype(str).str.strip().str.lower()
        yes_mask = labels == "yes"
        no_mask  = labels == "no"
        summary_rows.append({
            "Symptom":    symptom,
            "Yes_Count":  int(yes_mask.sum()),
            "No_Count":   int(no_mask.sum()),
            "Yes_Conf_Mean": metric_df.loc[yes_mask, conf_col].mean(),
            "Yes_Conf_SD":   metric_df.loc[yes_mask, conf_col].std(),
            "No_Conf_Mean":  metric_df.loc[no_mask,  conf_col].mean(),
            "No_Conf_SD":    metric_df.loc[no_mask,  conf_col].std(),
            "Yes_BLEU_Mean": metric_df.loc[yes_mask, bleu_col].mean(),
            "Yes_BLEU_SD":   metric_df.loc[yes_mask, bleu_col].std(),
            "No_BLEU_Mean":  metric_df.loc[no_mask,  bleu_col].mean(),
            "No_BLEU_SD":    metric_df.loc[no_mask,  bleu_col].std(),
            "Yes_BERTP_Mean": metric_df.loc[yes_mask, bert_col].mean(),
            "Yes_BERTP_SD":   metric_df.loc[yes_mask, bert_col].std(),
            "No_BERTP_Mean":  metric_df.loc[no_mask,  bert_col].mean(),
            "No_BERTP_SD":    metric_df.loc[no_mask,  bert_col].std(),
            "All_Conf_Mean": metric_df[conf_col].mean(),
            "All_Conf_SD":   metric_df[conf_col].std(),
            "All_BLEU_Mean": metric_df[bleu_col].mean(),
            "All_BLEU_SD":   metric_df[bleu_col].std(),
            "All_BERTP_Mean": metric_df[bert_col].mean(),
            "All_BERTP_SD":   metric_df[bert_col].std(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("exp7_summary_table.csv", index=False)

    print("\n" + "="*65)
    print("DETECTION COUNTS (Exp 7 corrected)")
    print("="*65)
    print(f"{'Symptom':<46} {'Yes':>6} {'No':>6}")
    print("-"*60)
    for _, r in summary_df.iterrows():
        print(f"  {r['Symptom']:<44} {int(r['Yes_Count']):>6} {int(r['No_Count']):>6}")
    print(f"\n  TOTAL YES: {int(summary_df['Yes_Count'].sum())}")

    print("\n" + "="*65)
    print("STRATIFIED METRICS Yes vs No (Exp 7 corrected)")
    print("="*65)
    hdr = f"{'Symptom':<44} {'Lbl':>3} {'N':>5}  {'Conf':>5} {'SD':>4}  {'BLEU':>5} {'SD':>5}  {'BERT-P':>6} {'SD':>6}"
    print(hdr)
    print("-"*len(hdr))
    for _, r in summary_df.iterrows():
        sym = r["Symptom"]
        print(f"  {sym:<42} Yes {int(r['Yes_Count']):>5}  "
              f"{r['Yes_Conf_Mean']:>5.2f} {r['Yes_Conf_SD']:>4.2f}  "
              f"{r['Yes_BLEU_Mean']:>5.3f} {r['Yes_BLEU_SD']:>5.3f}  "
              f"{r['Yes_BERTP_Mean']:>6.4f} {r['Yes_BERTP_SD']:>6.4f}")
        print(f"  {'':42}  No {int(r['No_Count']):>5}  "
              f"{r['No_Conf_Mean']:>5.2f} {r['No_Conf_SD']:>4.2f}  "
              f"{r['No_BLEU_Mean']:>5.3f} {r['No_BLEU_SD']:>5.3f}  "
              f"{r['No_BERTP_Mean']:>6.4f} {r['No_BERTP_SD']:>6.4f}")
        gap = r['Yes_BLEU_Mean'] - r['No_BLEU_Mean']
        flag = " <-- CONTAMINATION RISK" if gap < 0.10 else ""
        print(f"  {'':42}     {'':5}  {'':5} {'':4}  Gap={gap:>5.3f}{flag}")
        print()
    return summary_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp7 MedCPT RAG — corrected flat JSON")
    parser.add_argument("--phase", default="all",
                        choices=["inference","metrics","all"])
    parser.add_argument("--model", default=HF_MODEL_ID)
    parser.add_argument("--max_notes", type=int, default=None)
    args = parser.parse_args()

    print("\n" + "="*65)
    print("EXPERIMENT 7 — ROS + UMLS + MedCPT RAG (CORRECTED)")
    print("  Component A: LLaMA-3.1-8B-Instruct (JSON schema)")
    print("  Component B: Manual aliases (225 terms)")
    print("  Component C: UMLS synonyms (189 terms) -> 369 merged")
    print("  Component D: ROS Rules R1-R4 (prompt + programmatic)")
    print("  Component F: MedCPT dense retrieval (k=32, RRF k=60)")
    print("  FIX: Flat JSON enforced in prompt + flatten_parsed recovery")
    print("  FIX: BLEU self-correction limited to Yes predictions only")
    print("="*65)

    umls_synonyms        = load_umls_synonyms()
    merged_vocab         = build_merged_vocabulary(umls_synonyms)
    cache, group_b_lookup = load_concept_cache()
    chunks               = load_chunks()
    notes_df             = load_notes()

    if args.max_notes:
        notes_df = notes_df.head(args.max_notes)
        print(f"[TEST MODE] Running on {args.max_notes} notes")

    valid_df = None
    if args.phase in ("inference", "all"):
        encoder    = MedCPTEncoder(HF_CACHE_DIR)
        chunk_vecs = build_dense_index(encoder, chunks, batch_size=128)
        tokenizer, model = load_llama(args.model)
        valid_df   = run_inference(
            notes_df, encoder, chunk_vecs, chunks,
            merged_vocab, group_b_lookup, tokenizer, model)
        valid_df.to_csv("exp7_valid_outputs.csv", index=False)

    if args.phase == "metrics":
        raw_df = pd.read_csv("exp7_outputs_raw.csv")
        raw_df["exp_output_dict"] = raw_df["exp_output_raw"].apply(
            lambda x: flatten_parsed(safe_json_loads(str(x))[0])
            if pd.notna(x) else None)
        raw_df["exp_output_dict"] = raw_df["exp_output_dict"].apply(
            lambda x: x if isinstance(x, dict) else None)
        valid_df = raw_df[
            raw_df["exp_output_dict"].notna()
        ].copy().reset_index(drop=True)
        print(f"Valid rows loaded: {len(valid_df)}")

    if args.phase in ("metrics", "all") and valid_df is not None:
        run_metrics(valid_df)
