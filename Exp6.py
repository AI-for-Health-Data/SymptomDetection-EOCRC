import os, re, json, math, time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from bert_score import score as bert_score
    HAVE_BERTSCORE = True
except ModuleNotFoundError:
    HAVE_BERTSCORE = False

try:
    from rank_bm25 import BM25Okapi
    HAVE_BM25 = True
except ModuleNotFoundError:
    HAVE_BM25 = False
    print("ERROR: rank_bm25 not installed.")

os.environ["TRANSFORMERS_NO_TF"]   = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"]               = "0"
os.environ["USE_FLAX"]             = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PATH         = "rebuilt_notes_by_noteid.csv"
HF_MODEL_ID  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_CACHE_DIR = "/lustre/smuexa01/client/users/nikkieh/hf_cache"
UMLS_JSON    = "umls_synonyms.json"
UMLS_CACHE   = "umls_concept_cache.json"
CORPUS_PATH  = "pubmed_corpus.json"
CHUNKS_PATH  = "pubmed_chunks_sentences.json"

BM25_TOP_K     = 5
BM25_KEEP_K    = 3
BLEU_THRESHOLD = 0.10
RAG_SCORE_MIN  = 12.0
NOTE_MATCH_MIN = 2
DENIAL_CONTEXT = 20

SYMPTOMS = [
    "Abdominal pain", "Rectal bleeding", "Rectal pain",
    "Diarrhea", "Constipation", "Weight loss",
    "Family history of colorectal cancer",
]
SYMPTOM_SPECS = [(s, f"{s} confidence", f"{s} inference") for s in SYMPTOMS]

BAD_INFERENCE_VALS = {
    "", "n/a", "na", "not mentioned", "none", "no inference",
    "not reported", "not applicable", "not stated",
}

DENIAL_PREFIXES = [
    "no ","denies ","negative for ","without ",
    "absent ","none ","not ","deny ","no history of ",
]

SYMPTOM_CUI_FAMILIES = {
    "C0000737","C0152171","C0232503","C0694868","C1963065",
    "C0085584","C0232491","C0232492","C0232493","C0232494",
    "C0267596","C0018932","C0018937","C1321898","C0025209",
    "C0267615","C0267614",
    "C0034886","C0085606","C0085644","C0232607","C0232608",
    "C0011991","C0152164","C0860904","C0232726","C0232727",
    "C0009806","C0687720","C0232720","C0232721",
    "C1262477","C0043096","C0085295","C0003123","C0162429",
    "C0241889","C0332265","C0728708","C1553497",
}

# MANUAL ALIASES — GROUP A (225 terms, same as E1-E5)
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

RAG_PLACEHOLDERS = {
    "Abdominal pain":                      "RAG_ABDOMINAL_PAIN",
    "Rectal bleeding":                     "RAG_RECTAL_BLEEDING",
    "Rectal pain":                         "RAG_RECTAL_PAIN",
    "Diarrhea":                            "RAG_DIARRHEA",
    "Constipation":                        "RAG_CONSTIPATION",
    "Weight loss":                         "RAG_WEIGHT_LOSS",
    "Family history of colorectal cancer": "RAG_FAMILY_HISTORY",
}


# LOAD UMLS SYNONYMS
def load_umls_synonyms():
    with open(UMLS_JSON) as f:
        data = json.load(f)
    lowered = {s: [t.lower() for t in terms] for s, terms in data.items()}
    total = sum(len(v) for v in lowered.values())
    print(f"UMLS synonyms (GROUP A): {total} terms across {len(lowered)} symptoms")
    return lowered

def load_concept_cache():
    if not os.path.exists(UMLS_CACHE):
        print(f"ERROR: {UMLS_CACHE} not found.")
        print("Run: $PYTHON build_umls_concept_cache.py")
        print("(takes ~30-60 min, one-time only)")
        exit(1)

    with open(UMLS_CACHE) as f:
        cache = json.load(f)

    n_total   = len(cache)
    n_group_a = sum(1 for d in cache.values() if d.get("is_group_a", False))
    n_group_b = n_total - n_group_a

    print(f"UMLS concept cache loaded: {n_total} concepts")
    print(f"  GROUP A (our 7 symptoms): {n_group_a}")
    print(f"  GROUP B (extra concepts): {n_group_b}")

    # Build fast lookup sets
    # group_b_terms: set of surface forms → concept name
    group_b_lookup = {}
    for term, data in cache.items():
        if not data.get("is_group_a", False):
            group_b_lookup[term] = data.get("name", term)

    return cache, group_b_lookup

def scan_note_for_umls_concepts(note_text, umls_synonyms,
                                 cache, group_b_lookup):
    note_lower = note_text.lower()

    group_a = {}
    for symptom in SYMPTOMS:
        umls_terms   = umls_synonyms.get(symptom, [])
        manual_terms = [t.lower() for t in MANUAL_ALIASES.get(symptom, [])]
        all_terms    = list(set(umls_terms + manual_terms))
        found        = [t for t in all_terms if t in note_lower]
        if found:
            group_a[symptom] = found[:5]

    group_b = {}
    for term, concept_name in group_b_lookup.items():
        if term in note_lower:
            # Use UMLS concept name as the key (deduplicate by concept)
            if concept_name not in group_b:
                group_b[concept_name] = term

    return group_a, group_b


# BUILD RAG QUERY FROM GROUP B CONCEPTS
def build_rag_query_from_group_b(note_text, symptom,
                                  umls_synonyms, group_b):
    note_lower   = note_text.lower()

    umls_terms   = umls_synonyms.get(symptom, [])
    manual_terms = [t.lower() for t in MANUAL_ALIASES.get(symptom, [])]
    note_found   = [t for t in set(umls_terms + manual_terms)
                    if t in note_lower][:3]

    umls_canonical = umls_terms[:3]
    gb_names = list(group_b.keys())[:4]
    parts = []
    if note_found:
        parts.extend(note_found[:2])
    else:
        parts.append(symptom.lower())

    for t in umls_canonical:
        if t not in parts and len(parts) < 5:
            parts.append(t)

    if gb_names:
        parts.extend(gb_names[:3])

    parts.append("colorectal cancer")
    query = " ".join(parts)

    return query, note_found, gb_names


# SENTENCE-LEVEL CHUNKING
def chunk_by_sentence(text, max_sentences=4, overlap=1):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if not sentences:
        return [text.strip()] if len(text.strip()) > 50 else []
    chunks, start = [], 0
    while start < len(sentences):
        end   = min(start + max_sentences, len(sentences))
        chunk = " ".join(sentences[start:end])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        if end == len(sentences):
            break
        start += max_sentences - overlap
    return chunks


def build_chunks():
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH) as f:
            chunks = json.load(f)
        print(f"Sentence chunks loaded: {len(chunks)}")
        return chunks
    if not os.path.exists(CORPUS_PATH):
        print(f"ERROR: {CORPUS_PATH} not found. Run fetch_corpus.py first.")
        exit(1)
    with open(CORPUS_PATH) as f:
        abstracts = json.load(f)
    all_chunks = []
    for abstract in abstracts:
        all_chunks.extend(chunk_by_sentence(abstract))
    with open(CHUNKS_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"Saved {len(all_chunks)} sentence chunks → {CHUNKS_PATH}")
    return all_chunks

def build_bm25_index(chunks):
    print(f"Building BM25 index over {len(chunks)} chunks...")
    tokenized = [c.lower().split() for c in chunks]
    bm25      = BM25Okapi(tokenized)
    print("BM25 index ready")
    return bm25


# RETRIEVE PUBMED PASSAGES

def retrieve_for_symptom(bm25, chunks, query, symptom,
                          umls_synonyms, k=BM25_TOP_K, keep=BM25_KEEP_K):
    tokens  = query.lower().split()
    scores  = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:k]
    passages = [{"text": chunks[i], "score": float(scores[i])}
                for i in top_idx if scores[i] > 0]

    symptom_terms = (
        [t for t in umls_synonyms.get(symptom, [])]
        + [t.lower() for t in MANUAL_ALIASES.get(symptom, [])]
    )
    relevant = [p for p in passages
                if any(t in p["text"].lower() for t in symptom_terms[:25])]

    return (relevant if relevant else passages[:keep])[:keep]


def format_passages(passages, symptom):
    if not passages:
        return f"  No relevant passages retrieved for {symptom}."
    return "\n".join(
        f"  [Evidence {i+1}] {p['text'][:250]}"
        for i, p in enumerate(passages)
    )

def detect_rag_contradiction(llm_output, passages, symptom,
                              umls_synonyms, note_text):
    if str(llm_output.get(symptom,"")).strip().lower() != "no":
        return False, None
    note_lower = note_text.lower()
    all_terms  = (
        [t for t in umls_synonyms.get(symptom,[])]
        + [t.lower() for t in MANUAL_ALIASES.get(symptom,[])]
    )

    # Condition 1: >= NOTE_MATCH_MIN symptom terms in note
    note_matches = [t for t in all_terms if t in note_lower]
    if len(note_matches) < NOTE_MATCH_MIN:
        return False, None

    # Condition 2: at least 1 match NOT in denial context
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

    # Condition 3: passage has high BM25 score
    for p in passages:
        if p["score"] >= RAG_SCORE_MIN:
            return True, p["text"][:200]
    return False, None


PROMPT_TEMPLATE = """
You are an experienced gastroenterology clinician.
Analyze the patient's clinical note and extract information
ONLY from the note text (no external assumptions).

IMPORTANT RULE ABOUT ROS:
- ROS often contains templated negatives (e.g. "no abdominal pain")
  that conflict with the rest of the note.
- Do NOT use ROS-negative statements as evidence to answer "No".
- If ROS says "no X" but HPI / Chief Complaint / Assessment show X,
  answer "Yes" using non-ROS evidence.
- If explicitly denied OUTSIDE ROS: answer "No" (conf 4-5).
- If only in ROS-negative context: answer "No" (conf 2).

GROUP B: EXTRA UMLS CLINICAL CONCEPTS FOUND IN THIS NOTE:
These are all UMLS-identified clinical concepts found in this note
beyond our 7 target symptoms. They provide clinical context.
{GROUP_B_SECTION}

RETRIEVED PUBMED EVIDENCE (per symptom, grounded in GROUP B context):
Use these passages to interpret ambiguous findings.
Cite ONLY from the NOTE TEXT — not from these passages.

Abdominal pain:
{RAG_ABDOMINAL_PAIN}

Rectal bleeding:
{RAG_RECTAL_BLEEDING}

Rectal pain:
{RAG_RECTAL_PAIN}

Diarrhea:
{RAG_DIARRHEA}

Constipation:
{RAG_CONSTIPATION}

Weight loss:
{RAG_WEIGHT_LOSS}

Family history of colorectal cancer:
{RAG_FAMILY_HISTORY}

SYNONYM GUIDANCE:
{ALIAS_SECTION}

Rules:
- Clearly present outside ROS → "Yes" (conf 4-5)
- Explicitly denied outside ROS → "No" (conf 4-5)
- Only ROS-negative, nowhere else → "No" (conf 2)
- Output ONLY valid JSON. No prose, no markdown.

Return JSON:
"Abdominal pain","Abdominal pain confidence","Abdominal pain inference",
"Duration of abdominal pain","Duration of abdominal pain confidence","Duration of abdominal pain inference",
"Rectal bleeding","Rectal bleeding confidence","Rectal bleeding inference",
"Duration of rectal bleeding","Duration of rectal bleeding confidence","Duration of rectal bleeding inference",
"Rectal pain","Rectal pain confidence","Rectal pain inference",
"Duration of rectal pain","Duration of rectal pain confidence","Duration of rectal pain inference",
"Diarrhea","Diarrhea confidence","Diarrhea inference",
"Duration of diarrhea","Duration of diarrhea confidence","Duration of diarrhea inference",
"Constipation","Constipation confidence","Constipation inference",
"Duration of constipation","Duration of constipation confidence","Duration of constipation inference",
"Weight loss","Weight loss confidence","Weight loss inference",
"Duration of weight loss","Duration of weight loss confidence","Duration of weight loss inference",
"Family history of colorectal cancer","Family history of colorectal cancer confidence","Family history of colorectal cancer inference",
"Other comments","Other comments confidence","Other comments inference"

Patient NOTE TEXT:
<<NOTE_TEXT>>
""".strip()


def build_alias_section():
    lines = []
    for symptom in SYMPTOMS:
        aliases = MANUAL_ALIASES.get(symptom, [])
        seen, deduped = set(), []
        for a in aliases:
            if a.lower() not in seen:
                seen.add(a.lower())
                deduped.append(a)
        lines.append(f"{symptom}: {', '.join(deduped[:40])}")
    return "\n".join(lines)


def load_notes():
    notes_df = pd.read_csv(PATH)
    for col in ["DATE_OF_SERVIC_DTTM","SPEC_NOTE_TIME_DTTM","CONTACT_DATE"]:
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


def maybe_truncate(text, max_chars=None):
    if text is None: return ""
    t = str(text)
    if max_chars is None or len(t) <= max_chars: return t
    return t[:max_chars//2] + "\n...\n" + t[-(max_chars//2):]


def strip_code_fences(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?","",s,flags=re.IGNORECASE).strip()
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
    s1 = re.sub(r'("|\d|true|false|null)\s*\n(\s*")',r'\1,\n\2',s1)
    s1 = re.sub(r",\s*([}\]])",r"\1",s1)
    s1 = (s1.replace("\u201c",'"').replace("\u201d",'"')
             .replace("\u2018","'").replace("\u2019","'"))
    s1 = re.sub(r':\s*N/A(\s*[,\n}\]])',  r': "N/A"\1', s1)
    s1 = re.sub(r':\s*None(\s*[,\n}\]])', r': "None"\1',s1)
    s1 = re.sub(r':\s*n/a(\s*[,\n}\]])',  r': "N/A"\1', s1)
    try: return json.loads(s1), s1
    except: pass
    s2 = re.sub(r'("|\d|true|false|null)(\s*")',r'\1,\2',s1)
    try: return json.loads(s2), s2
    except: return None, s0


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
    match = sum(min(c, ref_ngrams[ng]) for ng,c in hyp_ngrams.items())
    total = sum(hyp_ngrams.values())
    return 0.0 if total == 0 else match / total


def compute_bleu_no_bp(ref, hyp, max_n=4):
    ref_toks = simple_tokenize(ref)
    hyp_toks = simple_tokenize(hyp)
    if not ref_toks or not hyp_toks: return 0.0
    max_n = min(max_n, len(hyp_toks))
    log_precs = []
    for n in range(1, max_n+1):
        p = modified_precision(ref_toks, hyp_toks, n)
        if p == 0.0: return 0.0
        log_precs.append(math.log(p))
    return math.exp(sum(log_precs)/len(log_precs))


def compute_bertscore_batch(refs, hyps):
    if not refs: return []
    P, R, F1 = bert_score(hyps, refs, lang="en",
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
        model_id, torch_dtype="float16", device_map="auto",
        trust_remote_code=True, cache_dir=HF_CACHE_DIR)
    model.eval()
    print("Model loaded")
    return tokenizer, model


def generate(prompt, tokenizer, model, max_new_tokens=1024):
    import torch
    messages = [{"role":"user","content":prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = tokenizer(formatted, return_tensors="pt",
                       truncation=True, max_length=8192).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# AGENTIC INFERENCE

def run_agentic_inference(notes_df, bm25, chunks, umls_synonyms,
                          cache, group_b_lookup, tokenizer, model):
    print("\n" + "="*65)
    print("AGENTIC INFERENCE — Experiment 6: BM25 RAG (Full UMLS)")
    print("  Step 1: Scan each note for ALL UMLS concepts in cache")
    print("  Step 2: Split into GROUP A (symptoms) + GROUP B (extra)")
    print("  Step 3: Build RAG query FROM GROUP B concepts")
    print("  Step 4: Retrieve PubMed passages about GROUP B + EOCRC")
    print("  Step 5: LLM gets note + evidence about GROUP B findings")
    print("  +VERIFY: BLEU self-correction if evidence not grounded")
    print("="*65)

    alias_section    = build_alias_section()
    patient_memory   = {}
    rows             = []
    rag_overrides    = 0
    self_corrections = 0
    group_b_freq     = defaultdict(int)

    for idx, row in tqdm(notes_df.iterrows(), total=len(notes_df),
                         desc="Exp6 BM25 RAG"):
        note_text = maybe_truncate(row["Clean_note_text"])
        pat_id    = row.get("PAT_ID", None)

        prior_context = ""
        if pat_id and pat_id in patient_memory:
            prior_pos = [s for s,v in patient_memory[pat_id].items()
                         if v == "yes"]
            if prior_pos:
                prior_context = (
                    f"PRIOR VISIT: Patient previously reported: "
                    f"{', '.join(prior_pos)}.\n\n")

        group_a, group_b = scan_note_for_umls_concepts(
            note_text, umls_synonyms, cache, group_b_lookup)

        for c in group_b:
            group_b_freq[c] += 1

        per_symptom_passages = {}
        for symptom in SYMPTOMS:
            query, note_found, gb_names = build_rag_query_from_group_b(
                note_text, symptom, umls_synonyms, group_b)
            passages = retrieve_for_symptom(
                bm25, chunks, query, symptom, umls_synonyms)
            per_symptom_passages[symptom] = passages

        if group_b:
            gb_lines = [
                f"  - {name} (found as: '{form}')"
                for name, form in list(group_b.items())[:15]
            ]
            group_b_section = prior_context + "\n".join(gb_lines)
        else:
            group_b_section = prior_context + "  None detected."

        prompt = PROMPT_TEMPLATE
        prompt = prompt.replace("{GROUP_B_SECTION}",  group_b_section)
        prompt = prompt.replace("{ALIAS_SECTION}",    alias_section)
        prompt = prompt.replace("<<NOTE_TEXT>>",      note_text)
        for symptom in SYMPTOMS:
            ph = "{" + RAG_PLACEHOLDERS[symptom] + "}"
            prompt = prompt.replace(
                ph, format_passages(per_symptom_passages[symptom], symptom))

        try:
            content = generate(prompt, tokenizer, model)
        except Exception as e:
            content = ""
            print(f"  Error {idx}: {e}")

        parsed, raw_json = safe_json_loads(content)

        if parsed:
            weak = []
            for symptom, conf_key, inf_key in SYMPTOM_SPECS:
                inf = parsed.get(inf_key, "")
                if (isinstance(inf, str)
                        and inf.strip().lower() not in BAD_INFERENCE_VALS):
                    if compute_bleu_no_bp(note_text, inf) < BLEU_THRESHOLD:
                        weak.append(symptom)

            if weak and len(weak) <= 4:
                self_corrections += 1
                gb_keys  = list(group_b.keys())[:3]
                retry_prompt = PROMPT_TEMPLATE
                retry_prompt = retry_prompt.replace(
                    "{GROUP_B_SECTION}", group_b_section)
                retry_prompt = retry_prompt.replace(
                    "{ALIAS_SECTION}", alias_section)
                retry_prompt = retry_prompt.replace(
                    "<<NOTE_TEXT>>", note_text)
                for symptom in SYMPTOMS:
                    ph = "{" + RAG_PLACEHOLDERS[symptom] + "}"
                    if symptom in weak:
                        exp_q = (symptom.lower() + " "
                                 + " ".join(gb_keys)
                                 + " colorectal cancer EOCRC early onset")
                        exp_p = retrieve_for_symptom(
                            bm25, chunks, exp_q, symptom, umls_synonyms)
                        retry_prompt = retry_prompt.replace(
                            ph, format_passages(exp_p, symptom))
                    else:
                        retry_prompt = retry_prompt.replace(
                            ph, format_passages(
                                per_symptom_passages[symptom], symptom))
                try:
                    rc = generate(retry_prompt, tokenizer, model)
                    rp, rr = safe_json_loads(rc)
                    if rp:
                        parsed   = rp
                        raw_json = rr
                except Exception as e:
                    print(f"  Retry error {idx}: {e}")

        if parsed:
            for symptom in SYMPTOMS:
                contr, evidence = detect_rag_contradiction(
                    parsed, per_symptom_passages[symptom],
                    symptom, umls_synonyms, note_text)
                if contr:
                    rag_overrides += 1
                    parsed[symptom] = "Yes"
                    parsed[f"{symptom} confidence"] = 3
                    parsed[f"{symptom} inference"]  = (
                        f"[RAG-OVERRIDE] {evidence[:150]}")

        if parsed and pat_id:
            if pat_id not in patient_memory:
                patient_memory[pat_id] = {}
            for symptom in SYMPTOMS:
                ans = normalize_answer(parsed.get(symptom,""))
                if ans in ("yes","no"):
                    patient_memory[pat_id][symptom] = ans

        out = row.to_dict()
        out["exp_output_raw"]  = raw_json
        out["exp_output_dict"] = parsed
        out["group_a_found"]   = json.dumps(group_a)
        out["group_b_found"]   = json.dumps(group_b)
        rows.append(out)

        if (idx+1) % 50 == 0:
            pd.DataFrame(rows).to_csv("exp6_checkpoint.csv", index=False)
            print(f"  [{idx+1}/{len(notes_df)}] "
                  f"overrides={rag_overrides} "
                  f"corrections={self_corrections}")

    exp_df = pd.DataFrame(rows)
    exp_df.to_csv("exp6_outputs_raw.csv", index=False)
    total    = len(exp_df)
    n_failed = exp_df["exp_output_dict"].apply(
        lambda x: not isinstance(x,dict)).sum()

    print(f"\nexp6_outputs_raw.csv ({total} rows)")
    print(f"Parse failures:   {n_failed}/{total}")
    print(f"RAG overrides:    {rag_overrides}")
    print(f"Self-corrections: {self_corrections}")

    print("\nTop 30 GROUP B concepts across all notes:")
    for c, n in sorted(group_b_freq.items(),
                        key=lambda x: x[1], reverse=True)[:30]:
        print(f"  {c:<40} {n:>4} notes")

    valid_df = exp_df[
        exp_df["exp_output_dict"].apply(lambda x: isinstance(x,dict))
    ].copy().reset_index(drop=True)
    print(f"\nValid rows: {len(valid_df)}/{total}")
    return valid_df

def run_metrics(valid_df):
    print("\n" + "="*65)
    print("METRICS — Experiment 6: BM25 RAG (Full UMLS)")
    print("="*65)

    metric_df = valid_df.copy()
    for symptom, conf_key, inf_key in SYMPTOM_SPECS:
        metric_df[symptom]  = metric_df["exp_output_dict"].apply(
            lambda d, s=symptom:  d.get(s,"")     if isinstance(d,dict) else "")
        metric_df[conf_key] = metric_df["exp_output_dict"].apply(
            lambda d, k=conf_key: d.get(k,np.nan) if isinstance(d,dict) else np.nan)
        metric_df[inf_key]  = metric_df["exp_output_dict"].apply(
            lambda d, k=inf_key:  d.get(k,"")     if isinstance(d,dict) else "")
        metric_df[f"{symptom} Conf_num"] = metric_df[conf_key].apply(to_num)

    print("Computing BLEU...")
    for symptom, _, inf_key in SYMPTOM_SPECS:
        bleu_col = f"{symptom} BLEU_noBP"
        vals = []
        for _, row in metric_df.iterrows():
            hyp = row[inf_key]
            if isinstance(hyp,str) and hyp.startswith("[RAG-OVERRIDE]"):
                hyp = hyp.replace("[RAG-OVERRIDE] ","")
            ref = row["Clean_note_text"]
            if not isinstance(hyp,str) or hyp.strip().lower() in BAD_INFERENCE_VALS:
                vals.append(0.0)
            else:
                vals.append(compute_bleu_no_bp(ref,hyp))
        metric_df[bleu_col] = vals
    print("BLEU done.")

    for symptom,_,_ in SYMPTOM_SPECS:
        metric_df[f"{symptom} BERT_P"] = np.nan
    if HAVE_BERTSCORE:
        print("Computing BERTScore...")
        for symptom,_,inf_key in SYMPTOM_SPECS:
            bert_col   = f"{symptom} BERT_P"
            idxs,refs,hyps = [],[],[]
            for i,row in metric_df.iterrows():
                hyp = row[inf_key]
                if isinstance(hyp,str) and hyp.startswith("[RAG-OVERRIDE]"):
                    hyp = hyp.replace("[RAG-OVERRIDE] ","")
                ref = row["Clean_note_text"]
                if not isinstance(hyp,str) or hyp.strip().lower() in BAD_INFERENCE_VALS:
                    continue
                idxs.append(i); refs.append(ref); hyps.append(hyp)
            if idxs:
                vals = compute_bertscore_batch(refs,hyps)
                for i,val in zip(idxs,vals):
                    metric_df.at[i,bert_col] = val
        print("BERTScore done.")

    metric_df.to_csv("exp6_note_level_metrics.csv", index=False)

    summary_rows = []
    for symptom,conf_key,inf_key in SYMPTOM_SPECS:
        conf_col = f"{symptom} Conf_num"
        bleu_col = f"{symptom} BLEU_noBP"
        bert_col = f"{symptom} BERT_P"
        labels   = metric_df[symptom].astype(str).str.strip().str.lower()
        yes_mask = labels == "yes"
        no_mask  = labels == "no"
        summary_rows.append({
            "Symptom":       symptom,
            "Yes_Count":     int(yes_mask.sum()),
            "No_Count":      int(no_mask.sum()),
            "Conf_Mean":     metric_df[conf_col].mean(),
            "Conf_SD":       metric_df[conf_col].std(),
            "Conf_Mean_Yes": metric_df.loc[yes_mask,conf_col].mean(),
            "Conf_Mean_No":  metric_df.loc[no_mask, conf_col].mean(),
            "BLEU_Mean":     metric_df[bleu_col].mean(),
            "BLEU_SD":       metric_df[bleu_col].std(),
            "BERTP_Mean":    metric_df[bert_col].mean(),
            "BERTP_SD":      metric_df[bert_col].std(),
        })

    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv("exp6_summary_table.csv", index=False)

    print("\n" + "="*65)
    print("TABLE I — POSITIVE DETECTION COUNTS (Exp 6: BM25 RAG)")
    print("="*65)
    print(f"{'Symptom':<46} {'Yes':>6} {'No':>6}")
    print("-"*60)
    for _,r in summary_table.iterrows():
        print(f"  {r['Symptom']:<44} {int(r['Yes_Count']):>6} {int(r['No_Count']):>6}")
    print(f"\n  TOTAL YES: {int(summary_table['Yes_Count'].sum())}")

    print("\n" + "="*65)
    print("TABLE II — CONFIDENCE, BLEU, BERTSCORE (Exp 6)")
    print("="*65)
    print(f"{'Symptom':<46} {'Conf':>12} {'BLEU':>12} {'BERT-P':>12}")
    print("-"*84)
    for _,r in summary_table.iterrows():
        print(f"  {r['Symptom']:<44} "
              f"{r['Conf_Mean']:>4.2f}+-{r['Conf_SD']:>4.2f}  "
              f"{r['BLEU_Mean']:>4.2f}+-{r['BLEU_SD']:>4.2f}  "
              f"{r['BERTP_Mean']:>4.2f}+-{r['BERTP_SD']:>4.2f}")

    return summary_table


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Exp6: BM25 RAG with Full UMLS Concept Scanning")
    parser.add_argument("--phase", default="all",
                        choices=["inference","metrics","all"])
    parser.add_argument("--model", default=HF_MODEL_ID)
    parser.add_argument("--max_notes", type=int, default=None)
    args = parser.parse_args()

    if not HAVE_BM25:
        print("ERROR: rank_bm25 not installed")
        exit(1)

    print("\n" + "="*65)
    print("EXPERIMENT 6 — BM25 RAG with Full UMLS Concept Scanning")
    print("  Step 1: Scan each note for ALL UMLS concepts (via cache)")
    print("  Step 2: GROUP A = 7 symptoms | GROUP B = all others")
    print("  Step 3: GROUP B drives RAG query construction")
    print("  Step 4: PubMed retrieval about GROUP B concepts + EOCRC")
    print("  Step 5: LLM sees note + GROUP B context + evidence")
    print("="*65)

    umls_synonyms        = load_umls_synonyms()
    cache, group_b_lookup = load_concept_cache()
    chunks               = build_chunks()
    bm25                 = build_bm25_index(chunks)
    notes_df             = load_notes()
    if args.max_notes:
        notes_df = notes_df.head(args.max_notes)

    valid_df = None

    if args.phase in ("inference","all"):
        tokenizer, model = load_llama(args.model)
        valid_df = run_agentic_inference(
            notes_df, bm25, chunks, umls_synonyms,
            cache, group_b_lookup, tokenizer, model)
        valid_df.to_csv("exp6_valid_outputs.csv", index=False)

    if args.phase == "metrics":
        raw_df = pd.read_csv("exp6_outputs_raw.csv")
        raw_df["exp_output_dict"] = raw_df["exp_output_raw"].apply(
            lambda x: safe_json_loads(str(x))[0] if pd.notna(x) else None)
        valid_df = raw_df[
            raw_df["exp_output_dict"].apply(lambda x: isinstance(x,dict))
        ].copy().reset_index(drop=True)
        print(f"Valid rows loaded: {len(valid_df)}")

    if args.phase in ("metrics","all") and valid_df is not None:
        run_metrics(valid_df)
