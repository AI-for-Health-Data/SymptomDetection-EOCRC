import os, re, json, math, time, argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from rag_prompt import RAG_PROMPT, RAG_RETRY_PROMPT

os.environ["TRANSFORMERS_NO_TF"]   = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"]               = "0"
os.environ["USE_FLAX"]             = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from rank_bm25 import BM25Okapi
    HAVE_BM25 = True
except ImportError:
    HAVE_BM25 = False
    print("WARNING: rank_bm25 not installed.")

try:
    from bert_score import score as bert_score_fn
    HAVE_BERTSCORE = True
except ImportError:
    HAVE_BERTSCORE = False
    print("WARNING: bert_score not installed.")

try:
    from sentence_transformers import CrossEncoder
    HAVE_CROSSENCODER = True
except ImportError:
    HAVE_CROSSENCODER = False
    print("INFO: sentence_transformers not installed — cross-encoder disabled.")


PATH              = "rebuilt_notes_by_noteid.csv"
HF_MODEL_ID       = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_CACHE_DIR      = "/lustre/smuexa01/client/users/nikkieh/hf_cache"
UMLS_JSON         = "umls_synonyms.json"
UMLS_CACHE        = "umls_concept_cache.json"
CHUNKS_PATH       = "pubmed_chunks_sentences.json"

MEDCPT_QUERY_MODEL    = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE_MODEL  = "ncbi/MedCPT-Article-Encoder"
CROSSENCODER_MODEL    = "/lustre/smuexa01/client/users/nikkieh/hf_cache/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots/c5ee24cb16019beea0893ab7796b1df96625c6b8"
os.environ["HF_HOME"] = "/lustre/smuexa01/client/users/nikkieh/hf_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

DENSE_TOP_K        = 32
BM25_TOP_K         = 32
RRF_CANDIDATE_K    = 20
KEEP_K             = 5
RRF_K              = 60
HYBRID_ALPHA       = 0.5

BLEU_CORRECT       = 0.15
BLEU_AMBIGUOUS     = 0.05

MAX_CRITIQUE_ROUNDS = 2
MAX_SYMPTOMS_RETRY  = 4
NOTE_CHAR_LIMIT     = 12000
DENIAL_CONTEXT      = 25
RAG_SCORE_MIN       = 0.30

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

FH_EXCLUSION_KEYWORDS = [
    # Genetic/molecular markers (population genetics papers)
    "germline",
    "penetrance",
    "msh2",
    "mlh1",
    "msh6",
    "pms2",
    "microsatellite",
    "mismatch repair",
    "amsterdam criteria",
    "amsterdam ii",
    "bethesda criteria",
    "bethesda guideline",
    # Population epidemiology framing
    "prevalence of lynch",
    "incidence of lynch",
    "lynch syndrome risk",
    "lynch syndrome mutation",
    "lynch syndrome carrier",
    "hereditary nonpolyposis",
    "hnpcc diagnosis",
    "familial adenomatous polyposis gene",
    "apc gene mutation",
    # Universal screening programs (population, not clinical)
    "universal tumor testing",
    "universal screening",
    "reflex testing",
    "population-based screening",
]

FH_SYM = "Family history of colorectal cancer"

def build_fh_filtered_corpus(chunks):
    """
    FIX A: Called ONCE at startup. Returns (filtered_chunks, filtered_bm25).
    Filters using single keywords that match sentence-level PubMed text.
    """
    kept = []
    excluded_count = 0
    for chunk in chunks:
        chunk_lower = chunk.lower()
        if any(term in chunk_lower for term in FH_EXCLUSION_KEYWORDS):
            excluded_count += 1
        else:
            kept.append(chunk)

    pct = 100 * excluded_count / len(chunks) if chunks else 0
    print(f"\nFH corpus filter (built once at startup):")
    print(f"  Full corpus:    {len(chunks)} chunks")
    print(f"  Excluded:       {excluded_count} ({pct:.1f}%) Lynch/MMR epidemiology")
    print(f"  FH corpus:      {len(kept)} chunks")

    if not kept:
        print("  WARNING: All chunks excluded — using full corpus for FH")
        kept = chunks

    if not HAVE_BM25:
        return kept, None
    fh_bm25 = BM25Okapi([c.lower().split() for c in kept])
    print(f"  FH BM25 index:  ready")
    return kept, fh_bm25


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
        "gas pain","gassy","dyspepsia","indigestion","heartburn",
        "acid reflux","gerd symptoms","gastritis","peptic ulcer","ulcer pain",
        "c/o abdominal pain","complains of abdominal pain",
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
    ],
    "Rectal pain": [
        "rectal pain","pain in rectum","painful rectum","anal pain",
        "pain in anus","anorectal pain","proctalgia","proctalgia fugax",
        "rectal discomfort","anal discomfort","rectal soreness","anal soreness",
        "pain with bowel movement","painful bowel movement",
        "painful defecation","dyschezia","tenesmus",
        "rectal pressure","anal fissure pain","fissure pain",
        "hemorrhoid pain","thrombosed hemorrhoid",
        "perianal pain","perirectal pain",
    ],
    "Diarrhea": [
        "diarrhea","diarrhoea","loose stools","loose stool",
        "watery stools","watery stool","liquid stool","runny stool",
        "the runs","frequent stools","frequent bowel movements",
        "increased bowel movements","multiple loose bms","loose bm",
        "watery bm","urgent bowel movements","fecal urgency","bowel urgency",
        "explosive diarrhea","bristol 6","bristol 7","soft stools",
        "gastroenteritis","enteritis","colitis with diarrhea",
        "c diff","c. diff","clostridioides difficile",
    ],
    "Constipation": [
        "constipation","constipated","hard stools","hard stool",
        "infrequent stools","infrequent bowel movements",
        "decreased stool frequency","decreased bowel movements",
        "no bm","no bowel movement","difficulty passing stool",
        "difficulty stooling","straining","strains with bm",
        "incomplete evacuation","obstipation","fecal impaction",
        "stool impaction","retained stool","bristol 1","bristol 2",
        "pellet stools","scybalous stools",
    ],
    "Weight loss": [
        "weight loss","wt loss","lost weight","losing weight",
        "weight down","weight decreased","decreased weight",
        "unintentional weight loss","unexplained weight loss",
        "involuntary weight loss","poor weight gain","failure to thrive",
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
    print(f"UMLS synonyms: {total} terms")
    return lowered

def load_concept_cache():
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
    unique = len(set(t for v in merged.values() for t in v))
    print(f"Merged vocabulary: {unique} unique surface forms")
    return merged

def load_chunks():
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    print(f"Sentence chunks: {len(chunks)}")
    return chunks

def load_notes():
    notes_df = pd.read_csv(PATH)
    for col in ["DATE_OF_SERVIC_DTTM", "SPEC_NOTE_TIME_DTTM"]:
        if col in notes_df.columns:
            notes_df[col] = pd.to_datetime(notes_df[col], errors="coerce")
    sort_cols = [c for c in [
        "PAT_ID","PAT_ENC_CSN_ID","DATE_OF_SERVIC_DTTM",
        "SPEC_NOTE_TIME_DTTM","CONTACT_NUM","NOTE_ID"
    ] if c in notes_df.columns]
    if sort_cols:
        notes_df = notes_df.sort_values(sort_cols).reset_index(drop=True)
    notes_df = notes_df[
        notes_df["Clean_note_text"].str.strip().ne("")
    ].reset_index(drop=True)
    print(f"Notes loaded: {len(notes_df)}")
    return notes_df


class MedCPTEncoder:
    def __init__(self, cache_dir):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch  = torch
        print("Loading MedCPT Article Encoder...")
        self.art_tok = AutoTokenizer.from_pretrained(
            MEDCPT_ARTICLE_MODEL, cache_dir=cache_dir)
        self.art_enc = AutoModel.from_pretrained(
            MEDCPT_ARTICLE_MODEL, cache_dir=cache_dir).to(self.device)
        self.art_enc.eval()
        print("Loading MedCPT Query Encoder...")
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
                vecs = vecs / (vecs.norm(dim=1, keepdim=True) + 1e-8)
                all_vecs.append(vecs.cpu().float().numpy())
        return np.vstack(all_vecs)

    def encode_articles(self, texts, batch_size=128):
        return self._encode(self.art_enc, self.art_tok, texts, batch_size)

    def encode_queries(self, texts):
        return self._encode(self.qry_enc, self.qry_tok, texts, batch_size=32)

def build_dense_index(encoder, chunks):
    print(f"Encoding {len(chunks)} chunks with MedCPT...")
    t0 = time.time()
    vecs = encoder.encode_articles(chunks, batch_size=128)
    print(f"Dense index: {vecs.shape} in {(time.time()-t0)/60:.1f} min")
    return vecs

def build_bm25_index(chunks):
    if not HAVE_BM25:
        return None
    bm25 = BM25Okapi([c.lower().split() for c in chunks])
    print(f"BM25 index: {len(chunks)} chunks")
    return bm25


class CrossEncoderReranker:
    def __init__(self):
        if not HAVE_CROSSENCODER:
            self.model = None
            print("Cross-encoder: disabled (sentence_transformers not installed)")
            print("  To enable: pip install sentence-transformers --break-system-packages")
            return
        print(f"Loading cross-encoder: {CROSSENCODER_MODEL} (CPU)...")
        self.model = CrossEncoder(CROSSENCODER_MODEL, device="cuda", max_length=512)
        print("Cross-encoder ready")

    def rerank(self, query, passages, top_k=KEEP_K):
        if self.model is None or not passages:
            return passages[:top_k]
        pairs = [(query, p["text"][:400]) for p in passages]
        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
            ranked = sorted(zip(scores, passages),
                            key=lambda x: x[0], reverse=True)
            result = [p for _, p in ranked[:top_k]]
            for i, (s, _) in enumerate(
                    sorted(zip(scores, passages),
                           key=lambda x: x[0], reverse=True)[:top_k]):
                result[i] = dict(result[i])
                result[i]["ce_score"] = float(s)
            return result
        except Exception as e:
            print(f"  Cross-encoder error: {e}")
            return passages[:top_k]


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


def build_initial_query(note_text, symptom, merged_vocab, group_b):
    note_lower = note_text.lower()
    note_found = [t for t in merged_vocab[symptom] if t in note_lower][:3]
    umls_terms = merged_vocab[symptom][:3]
    gb_names   = list(group_b.keys())[:4]
    parts = list(note_found[:2]) if note_found else [symptom.lower()]
    for t in umls_terms:
        if t not in parts and len(parts) < 5:
            parts.append(t)
    parts.extend(gb_names[:3])
    parts.append("colorectal cancer")
    return " ".join(parts)

def build_refined_query(symptom, critique_label, merged_vocab, group_b,
                        note_text, round_num):
    note_lower    = note_text.lower()
    note_found    = [t for t in merged_vocab[symptom] if t in note_lower]
    gb_names      = list(group_b.keys())[:6]
    all_sym_terms = merged_vocab[symptom]

    if critique_label == "Incorrect":
        core = note_found[:4] if note_found else all_sym_terms[:3]
        clinical_anchors = [
            "clinical presentation patient reported",
            "early onset colorectal cancer EOCRC diagnosis",
        ]
        parts = core + gb_names[:3] + clinical_anchors
        if symptom == FH_SYM:
            # Patient-centric query — will search filtered corpus
            parts = [
                "patient family member first degree relative",
                "colon cancer family documented clinical note",
                "colorectal cancer family history personal",
                "mother father sibling colon cancer history",
            ] + gb_names[:2]
    elif critique_label == "Ambiguous":
        core     = note_found[:2] if note_found else all_sym_terms[:2]
        synonyms = [t for t in all_sym_terms if t not in core][:4]
        parts    = core + synonyms + gb_names[:3] + ["colorectal cancer EOCRC"]
    else:
        parts = all_sym_terms[:3] + gb_names[:2] + ["colorectal cancer"]

    if round_num >= 2:
        parts += ["individual patient clinical note symptom documented"]
    return " ".join(parts[:15])


def retrieve_hybrid(encoder, chunk_vecs, chunks, bm25,
                    fh_chunks, fh_bm25,
                    queries, merged_vocab, reranker=None,
                    dense_k=DENSE_TOP_K, bm25_k=BM25_TOP_K,
                    candidate_k=RRF_CANDIDATE_K, keep_k=KEEP_K,
                    alpha=HYBRID_ALPHA):
    """
    Hybrid retrieval with:
    - Pre-built filtered corpus for FH (FIX A — no rebuild per note)
    - Cross-encoder reranking of top candidates (when available)
    - Global RRF across all 7 symptom queries for evidence block
    """
    query_texts  = list(queries.values())
    query_vecs   = encoder.encode_queries(query_texts)
    dense_scores = chunk_vecs @ query_vecs.T          # (N_full, 7)

    per_symptom = {}
    global_rrf  = defaultdict(float)

    for i, (symptom, query) in enumerate(queries.items()):
        sym_terms = merged_vocab.get(symptom, [])[:40]

        # FH uses pre-built filtered BM25 
        if symptom == FH_SYM and fh_bm25 is not None:
            b_scores = fh_bm25.get_scores(query.lower().split())
            b_ranked = np.argsort(b_scores)[::-1][:bm25_k]
            sym_rrf  = defaultdict(float)
            for rank, idx in enumerate(b_ranked):
                sym_rrf[int(idx)] += (1 - alpha) / (RRF_K + rank + 1)
            top_fh = sorted(sym_rrf.items(),
                            key=lambda x: x[1], reverse=True)[:candidate_k]
            candidates = [
                {"text": fh_chunks[idx], "score": score, "dense": 0.0}
                for idx, score in top_fh
            ]
            relevant = [c for c in candidates
                        if any(t in c["text"].lower() for t in sym_terms)]
            to_rank  = relevant if relevant else candidates
            if reranker is not None:
                per_symptom[symptom] = reranker.rerank(query, to_rank, keep_k)
            else:
                per_symptom[symptom] = to_rank[:keep_k]
            continue

        # Standard path for all other symptoms 
        d_scores = dense_scores[:, i]
        d_ranked = np.argsort(d_scores)[::-1][:dense_k]

        if HAVE_BM25 and bm25 is not None:
            b_scores = bm25.get_scores(query.lower().split())
            b_ranked = np.argsort(b_scores)[::-1][:bm25_k]
        else:
            b_ranked = d_ranked

        sym_rrf = defaultdict(float)
        for rank, idx in enumerate(d_ranked):
            sym_rrf[int(idx)] += alpha / (RRF_K + rank + 1)
        for rank, idx in enumerate(b_ranked):
            sym_rrf[int(idx)] += (1 - alpha) / (RRF_K + rank + 1)

        top_sym    = sorted(sym_rrf.items(),
                            key=lambda x: x[1], reverse=True)[:candidate_k]
        candidates = [
            {"text": chunks[idx], "score": score, "dense": float(d_scores[idx])}
            for idx, score in top_sym
        ]
        relevant   = [c for c in candidates
                      if any(t in c["text"].lower() for t in sym_terms)]
        to_rank    = relevant if relevant else candidates

        if reranker is not None:
            per_symptom[symptom] = reranker.rerank(query, to_rank, keep_k)
        else:
            per_symptom[symptom] = to_rank[:keep_k]

        for idx, score in sym_rrf.items():
            global_rrf[idx] += score

    # Global top passages for the evidence block
    top_global     = sorted(global_rrf.items(),
                            key=lambda x: x[1], reverse=True)[:candidate_k]
    global_cands   = [{"text": chunks[idx], "score": score}
                      for idx, score in top_global]
    if reranker is not None:
        top_passages = reranker.rerank(
            "colorectal cancer EOCRC symptoms clinical note",
            global_cands, keep_k)
    else:
        top_passages = global_cands[:keep_k]

    return per_symptom, top_passages

def format_passages(passages):
    if not passages:
        return "  No relevant passages retrieved."
    return "\n".join(
        f"  [Evidence {i+1}] {p['text'][:250]}"
        for i, p in enumerate(passages))

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
    return note_text[:ros_start] + note_text[ros_end:], note_text[ros_start:ros_end]

def symptom_present_outside_ros(note_text, symptom, merged_vocab):
    non_ros, _ = split_note_ros(note_text)
    note_lower  = non_ros.lower()
    for term in merged_vocab.get(symptom, []):
        idx = 0
        while True:
            idx = note_lower.find(term, idx)
            if idx == -1: break
            ctx = note_lower[max(0, idx - DENIAL_CONTEXT):idx]
            if not any(p in ctx for p in DENIAL_PREFIXES):
                return True
            idx += 1
    return False

_token_pat = re.compile(r"\w+|\S")

def simple_tokenize(text):
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    return _token_pat.findall(text.lower())

def modified_precision(ref_tokens, hyp_tokens, n):
    if len(hyp_tokens) < n: return 0.0
    hyp_ng = Counter(zip(*[hyp_tokens[i:] for i in range(n)]))
    ref_ng  = Counter(zip(*[ref_tokens[i:]  for i in range(n)]))
    match   = sum(min(c, ref_ng[ng]) for ng, c in hyp_ng.items())
    total   = sum(hyp_ng.values())
    return 0.0 if total == 0 else match / total

def compute_bleu_no_bp(ref, hyp, max_n=4):
    r = simple_tokenize(ref); h = simple_tokenize(hyp)
    if not r or not h: return 0.0
    max_n = min(max_n, len(h))
    logs  = []
    for n in range(1, max_n + 1):
        p = modified_precision(r, h, n)
        if p == 0.0: return 0.0
        logs.append(math.log(p))
    return math.exp(sum(logs) / len(logs))

def compute_bertscore_batch(refs, hyps):
    if not refs or not HAVE_BERTSCORE: return []
    P, R, F1 = bert_score_fn(
        hyps, refs, lang="en",
        rescale_with_baseline=False,
        batch_size=16, verbose=False)
    return [float(x) for x in P]

def classify_grounding(bleu_score):
    if bleu_score >= BLEU_CORRECT:   return "Correct"
    if bleu_score >= BLEU_AMBIGUOUS: return "Ambiguous"
    return "Incorrect"

def critique_output(parsed, note_text):
    critique = {}
    for symptom, conf_key, inf_key in SYMPTOM_SPECS:
        if not isinstance(parsed, dict):
            critique[symptom] = {"bleu": 0.0, "label": "Incorrect"}
            continue
        inf = parsed.get(inf_key, "")
        for tag in ["[RAG-OVERRIDE] ","[HYBRID-OVERRIDE] ","[SELF-CORRECTED] "]:
            if isinstance(inf, str): inf = inf.replace(tag, "")
        if (not isinstance(inf, str)
                or inf.strip().lower() in BAD_INFERENCE_VALS
                or len(inf.strip()) < 5):
            bleu = 0.0
        else:
            bleu = compute_bleu_no_bp(note_text, inf)
        critique[symptom] = {"bleu": bleu, "label": classify_grounding(bleu)}
    return critique

def symptoms_needing_retry(critique_results, parsed,
                            max_retry=MAX_SYMPTOMS_RETRY):
    """Only retry Yes predictions."""
    to_retry = [
        sym for sym, c in critique_results.items()
        if c["label"] in ("Incorrect", "Ambiguous")
        and str(parsed.get(sym, "")).strip().lower() == "yes"
    ]
    return to_retry[:max_retry]

INITIAL_PROMPT = RAG_PROMPT

FOCUSED_RETRY_PROMPT = RAG_RETRY_PROMPT

def build_alias_section(merged_vocab, symptoms=None):
    if symptoms is None: symptoms = SYMPTOMS
    return "\n".join(
        f"{sym}: {', '.join(merged_vocab[sym][:40])}" for sym in symptoms)

def build_json_keys_for_symptoms(symptoms):
    keys = []
    for sym in symptoms:
        keys += [f'"{sym}"', f'"{sym} confidence"', f'"{sym} inference"',
                 f'"Duration of {sym.lower()}"',
                 f'"Duration of {sym.lower()} confidence"',
                 f'"Duration of {sym.lower()} inference"']
    return ", ".join(keys)

def strip_code_fences(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
        if s.endswith("```"): s = s[:-3].strip()
    return s

def extract_first_json(s):
    s = strip_code_fences(s)
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
    if not isinstance(d, dict): return d
    result = dict(d)
    for sym in SYMPTOMS:
        val = d.get(sym)
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ("yes","1","true","present","y"):  result[sym] = "Yes"
            elif v in ("no","0","false","absent","n"): result[sym] = "No"
            continue
        if not isinstance(val, dict): continue
        inner = val; sym_lower = sym.lower()
        conf_key = sym + " confidence"; inf_key = sym + " inference"
        answer = None
        for tk in [sym, sym_lower, "answer", "presence"]:
            if tk in inner:
                v = str(inner[tk]).strip().lower()
                if v in ("yes","1","true","present","y"): answer="Yes"; break
                if v in ("no","0","false","absent","n"):  answer="No";  break
        if not answer:
            ic = None
            for k, v in inner.items():
                if "confidence" in k.lower() and sym_lower in k.lower():
                    try: ic = float(v)
                    except: pass
                    break
            if ic is not None: answer = "Yes" if ic >= 4 else "No"
        if not answer:
            for k, v in inner.items():
                if "inference" in k.lower() and sym_lower in k.lower():
                    vs = str(v).strip().lower()
                    if vs == "yes": answer="Yes"; break
                    if vs == "no":  answer="No";  break
        result[sym] = answer if answer else "No"
        if result.get(conf_key) is None:
            for k, v in inner.items():
                if "confidence" in k.lower() and sym_lower in k.lower():
                    result[conf_key] = v; break
        outer_inf = result.get(inf_key, "")
        if (not isinstance(outer_inf, str)
                or outer_inf.strip().lower() in BAD_INFERENCE_VALS
                or len(outer_inf.strip()) < 10):
            for k, v in inner.items():
                if "inference" in k.lower() and sym_lower in k.lower():
                    vs = str(v).strip()
                    if vs.lower() not in BAD_INFERENCE_VALS and len(vs) > 8:
                        result[inf_key] = vs; break
    return result

def normalize_answer(x):
    if pd.isna(x): return ""
    return str(x).strip().lower()

def to_num(x):
    if pd.isna(x): return np.nan
    m = re.search(r"-?\d+(?:\.\d+)?", str(x))
    return float(m.group()) if m else np.nan

def load_llama(model_id=HF_MODEL_ID):
    import torch as _torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"\nLoading: {model_id}")
    if _torch.cuda.is_available():
        print(f"GPU: {_torch.cuda.get_device_name(0)}")
        print(f"VRAM: {round(_torch.cuda.get_device_properties(0).total_memory/1e9,1)} GB")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=HF_CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=_torch.float16, device_map="auto",
        trust_remote_code=True, cache_dir=HF_CACHE_DIR)
    model.eval()
    print("LLaMA loaded")
    return tokenizer, model

def generate(prompt, tokenizer, model, max_new_tokens=1024):
    import torch
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = tokenizer(
        formatted, return_tensors="pt",
        truncation=True, max_length=8192).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def maybe_truncate(text, max_chars=NOTE_CHAR_LIMIT):
    if text is None: return ""
    t = str(text)
    if len(t) <= max_chars: return t
    return t[:max_chars//2] + "\n...\n" + t[-(max_chars//2):]


def detect_contradiction(parsed, passages, symptom, merged_vocab, note_text):
    if str(parsed.get(symptom, "")).strip().lower() != "no":
        return False, None
    note_lower = note_text.lower()
    note_found = [t for t in merged_vocab.get(symptom, []) if t in note_lower]
    if len(note_found) < 2:
        return False, None
    if not symptom_present_outside_ros(note_text, symptom, merged_vocab):
        return False, None
    for p in passages.get(symptom, []):
        if p.get("dense", 0) >= RAG_SCORE_MIN:
            return True, p["text"][:200]
    return False, None

def run_inference(notes_df, encoder, chunk_vecs, chunks, bm25,
                  fh_chunks, fh_bm25, reranker,
                  merged_vocab, group_b_lookup, tokenizer, model):

    print("\n" + "="*70)
    print("EXPERIMENT 8 v2 — AGENTIC SELF-RAG + CORPUS FILTER + RERANKER")
    print(f"  FH corpus filter: {len(chunks)} → {len(fh_chunks)} chunks")
    print(f"  Cross-encoder:    {'enabled' if reranker.model else 'disabled'}")
    print("="*70)

    alias_section   = build_alias_section(merged_vocab)
    patient_memory  = {}
    rows            = []
    nested_recovered      = 0
    total_initial_correct = 0
    total_ambiguous_fixed = 0
    total_incorrect_fixed = 0
    total_still_wrong     = 0
    total_rag_overrides   = 0
    bleu_trajectory_log   = []

    for idx, row in tqdm(notes_df.iterrows(), total=len(notes_df),
                         desc="Exp8v2 Self-RAG+Filter+Rerank"):

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

        gb_section = (
            prior_context + "\n".join(
                f"  - {n} (found as: '{f}')"
                for n, f in list(group_b.items())[:15])
            if group_b else prior_context + "  None detected.")

        # Round 0: initial retrieval
        init_queries = {
            sym: build_initial_query(note_text, sym, merged_vocab, group_b)
            for sym in SYMPTOMS
        }
        per_sym_pass, top_pass = retrieve_hybrid(
            encoder, chunk_vecs, chunks, bm25,
            fh_chunks, fh_bm25,
            init_queries, merged_vocab, reranker=reranker)

        prompt = (INITIAL_PROMPT
                  .replace("{GROUP_B_SECTION}",  gb_section)
                  .replace("{EVIDENCE_SECTION}", format_passages(top_pass))
                  .replace("{ALIAS_SECTION}",    alias_section)
                  .replace("<<NOTE_TEXT>>",      note_text))

        try:
            content = generate(prompt, tokenizer, model)
        except Exception as e:
            content = ""; print(f"  Error note {idx}: {e}")

        parsed, raw_json = safe_json_loads(content)

        if parsed and isinstance(parsed, dict):
            if any(isinstance(parsed.get(s), dict) for s in SYMPTOMS):
                parsed = flatten_parsed(parsed)
                nested_recovered += 1

        audit = {sym: {"bleu_rounds": [], "label_rounds": [],
                       "final_round": 0} for sym in SYMPTOMS}
        per_sym_retry = per_sym_pass  # FIX 4

        if parsed:
            critique0 = critique_output(parsed, note_text)
            for sym in SYMPTOMS:
                b0 = critique0[sym]["bleu"]
                l0 = critique0[sym]["label"]
                audit[sym]["bleu_rounds"].append(b0)
                audit[sym]["label_rounds"].append(l0)
                bleu_trajectory_log.append((idx, sym, 0, b0, l0))
                if l0 == "Correct": total_initial_correct += 1

            for round_num in range(1, MAX_CRITIQUE_ROUNDS + 1):
                retry_syms = symptoms_needing_retry(critique0, parsed)
                if not retry_syms: break

                refined_queries = {
                    sym: (build_refined_query(
                              sym, critique0[sym]["label"],
                              merged_vocab, group_b, note_text, round_num)
                          if sym in retry_syms else init_queries[sym])
                    for sym in SYMPTOMS
                }

                per_sym_retry, top_retry = retrieve_hybrid(
                    encoder, chunk_vecs, chunks, bm25,
                    fh_chunks, fh_bm25,
                    refined_queries, merged_vocab, reranker=reranker)

                prev_answers = {
                    sym: {
                        "answer":     parsed.get(sym, "unknown"),
                        "confidence": parsed.get(f"{sym} confidence", "?"),
                        "inference":  parsed.get(f"{sym} inference", "N/A"),
                        "bleu":       round(critique0[sym]["bleu"], 3),
                        "label":      critique0[sym]["label"],
                    }
                    for sym in retry_syms
                }

                retry_prompt = (
                    FOCUSED_RETRY_PROMPT
                    .replace("{RETRY_SYMPTOMS}",
                             ", ".join(f'"{s}"' for s in retry_syms))
                    .replace("{PREVIOUS_ANSWERS}",
                             json.dumps(prev_answers, indent=2))
                    .replace("{GROUP_B_SECTION}",    gb_section)
                    .replace("{EVIDENCE_SECTION}",   format_passages(top_retry))
                    .replace("{ALIAS_SECTION_FOCUSED}",
                             build_alias_section(merged_vocab, retry_syms))
                    .replace("<<NOTE_TEXT>>",         note_text)
                    .replace("{JSON_KEYS}",
                             build_json_keys_for_symptoms(retry_syms)))

                try:
                    rc = generate(retry_prompt, tokenizer, model)
                    rp, _ = safe_json_loads(rc)
                    if rp and isinstance(rp, dict):
                        if any(isinstance(rp.get(s), dict) for s in SYMPTOMS):
                            rp = flatten_parsed(rp)
                except Exception as e:
                    print(f"  Retry error {idx} r{round_num}: {e}")
                    rp = None

                if rp:
                    for sym in retry_syms:
                        inf_key = f"{sym} inference"
                        new_inf = rp.get(inf_key, "")
                        if isinstance(new_inf, str):
                            new_inf = new_inf.replace("[SELF-CORRECTED] ", "")
                        new_bleu = (
                            compute_bleu_no_bp(note_text, new_inf)
                            if isinstance(new_inf, str)
                            and new_inf.strip().lower() not in BAD_INFERENCE_VALS
                            else 0.0)
                        new_lbl  = classify_grounding(new_bleu)
                        bleu_trajectory_log.append(
                            (idx, sym, round_num, new_bleu, new_lbl))
                        audit[sym]["bleu_rounds"].append(new_bleu)
                        audit[sym]["label_rounds"].append(new_lbl)

                        old_bleu = critique0[sym]["bleu"]
                        old_lbl  = critique0[sym]["label"]
                        if new_bleu > old_bleu:
                            for key in [sym, f"{sym} confidence",
                                        f"{sym} inference",
                                        f"Duration of {sym.lower()}",
                                        f"Duration of {sym.lower()} confidence",
                                        f"Duration of {sym.lower()} inference"]:
                                if key in rp: parsed[key] = rp[key]
                            audit[sym]["final_round"] = round_num
                            if old_lbl == "Ambiguous" and new_lbl == "Correct":
                                total_ambiguous_fixed += 1
                            elif old_lbl == "Incorrect" and new_lbl in (
                                    "Correct","Ambiguous"):
                                total_incorrect_fixed += 1
                        else:
                            if old_lbl in ("Ambiguous","Incorrect"):
                                total_still_wrong += 1
                        critique0[sym] = {"bleu": new_bleu, "label": new_lbl}

            for sym in SYMPTOMS:
                contr, evidence = detect_contradiction(
                    parsed, per_sym_retry, sym, merged_vocab, note_text)
                if contr:
                    total_rag_overrides += 1
                    parsed[sym]                  = "Yes"
                    parsed[f"{sym} confidence"]  = 3
                    parsed[f"{sym} inference"]   = f"[HYBRID-OVERRIDE] {evidence[:150]}"

        if parsed and pat_id:
            if pat_id not in patient_memory:
                patient_memory[pat_id] = {}
            for sym in SYMPTOMS:
                ans = normalize_answer(parsed.get(sym, ""))
                if ans in ("yes","no"):
                    patient_memory[pat_id][sym] = ans

        out = row.to_dict()
        out["exp_output_raw"]  = raw_json
        out["exp_output_dict"] = parsed
        out["group_a_found"]   = json.dumps(group_a)
        out["group_b_found"]   = json.dumps(group_b)
        out["critique_audit"]  = json.dumps(audit)
        if parsed:
            fc = critique_output(parsed, note_text)
            for sym in SYMPTOMS:
                out[f"{sym} final_bleu"]  = fc[sym]["bleu"]
                out[f"{sym} final_label"] = fc[sym]["label"]
                out[f"{sym} final_round"] = audit[sym]["final_round"]
        rows.append(out)

        if (idx + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv("exp9_ce_checkpoint.csv", index=False)
            print(f"  [{idx+1}/{len(notes_df)}] "
                  f"init_correct={total_initial_correct} "
                  f"ambig_fixed={total_ambiguous_fixed} "
                  f"incorr_fixed={total_incorrect_fixed} "
                  f"still_wrong={total_still_wrong} "
                  f"overrides={total_rag_overrides} "
                  f"nested_recovered={nested_recovered}")

    exp_df = pd.DataFrame(rows)
    exp_df.to_csv("exp9_ce_outputs_raw.csv", index=False)
    n_failed = exp_df["exp_output_dict"].apply(
        lambda x: not isinstance(x, dict)).sum()

    print("\n" + "="*70)
    print("EXPERIMENT 8 v2 COMPLETE")
    print(f"  Total rows:              {len(exp_df)}")
    print(f"  Parse failures:          {n_failed}/{len(exp_df)}")
    print(f"  Nested recovered:        {nested_recovered}")
    print(f"  Initial Correct (Yes):   {total_initial_correct}")
    print(f"  Ambiguous → Correct:     {total_ambiguous_fixed}")
    print(f"  Incorrect → Fixed:       {total_incorrect_fixed}")
    print(f"  Still wrong after retry: {total_still_wrong}")
    print(f"  Hybrid overrides:        {total_rag_overrides}")
    print("="*70)

    pd.DataFrame(bleu_trajectory_log,
                 columns=["note_idx","symptom","round","bleu","label"]
                 ).to_csv("exp9_ce_bleu_trajectory.csv", index=False)

    valid_df = exp_df[
        exp_df["exp_output_dict"].apply(lambda x: isinstance(x, dict))
    ].copy().reset_index(drop=True)
    return valid_df

def run_metrics(valid_df):
    print("\n" + "="*70)
    print("METRICS — Experiment 8 v2: Self-RAG + Filter + Reranker")
    print("="*70)

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
            for tag in ["[HYBRID-OVERRIDE] ","[RAG-OVERRIDE] ","[SELF-CORRECTED] "]:
                if isinstance(hyp, str): hyp = hyp.replace(tag,"")
            ref = row["Clean_note_text"]
            vals.append(
                compute_bleu_no_bp(ref, hyp)
                if isinstance(hyp, str)
                and hyp.strip().lower() not in BAD_INFERENCE_VALS
                and len(hyp.strip()) > 5
                else 0.0)
        metric_df[bleu_col] = vals
    print("BLEU done.")

    for sym, _, _ in SYMPTOM_SPECS:
        metric_df[f"{sym} BERT_P"] = np.nan
    if HAVE_BERTSCORE:
        print("Computing BERTScore...")
        for symptom, _, inf_key in SYMPTOM_SPECS:
            bert_col = f"{symptom} BERT_P"
            idxs, refs, hyps = [], [], []
            for i, row in metric_df.iterrows():
                hyp = row[inf_key]
                for tag in ["[HYBRID-OVERRIDE] ","[RAG-OVERRIDE] ","[SELF-CORRECTED] "]:
                    if isinstance(hyp, str): hyp = hyp.replace(tag,"")
                ref = row["Clean_note_text"]
                if (not isinstance(hyp, str)
                        or hyp.strip().lower() in BAD_INFERENCE_VALS
                        or len(hyp.strip()) < 5):
                    continue
                idxs.append(i); refs.append(ref); hyps.append(hyp)
            if idxs:
                for i, val in zip(idxs, compute_bertscore_batch(refs, hyps)):
                    metric_df.at[i, bert_col] = val
        print("BERTScore done.")

    metric_df.to_csv("exp9_ce_note_level_metrics.csv", index=False)

    print("\n--- Final Critique Label Distribution ---")
    for sym in SYMPTOMS:
        col = f"{sym} final_label"
        if col in metric_df.columns:
            dist = metric_df[col].value_counts().to_dict()
            print(f"  {sym:<46} {dist}")

    summary_rows = []
    for symptom, conf_key, inf_key in SYMPTOM_SPECS:
        conf_col = f"{symptom} Conf_num"
        bleu_col = f"{symptom} BLEU_noBP"
        bert_col = f"{symptom} BERT_P"
        labels   = metric_df[symptom].astype(str).str.strip().str.lower()
        yes_mask = labels == "yes"
        no_mask  = labels == "no"
        summary_rows.append({
            "Symptom":      symptom,
            "Yes_Count":    int(yes_mask.sum()),
            "No_Count":     int(no_mask.sum()),
            "Yes_Conf":     metric_df.loc[yes_mask, conf_col].mean(),
            "Yes_Conf_SD":  metric_df.loc[yes_mask, conf_col].std(),
            "No_Conf":      metric_df.loc[no_mask,  conf_col].mean(),
            "No_Conf_SD":   metric_df.loc[no_mask,  conf_col].std(),
            "Yes_BLEU":     metric_df.loc[yes_mask, bleu_col].mean(),
            "Yes_BLEU_SD":  metric_df.loc[yes_mask, bleu_col].std(),
            "No_BLEU":      metric_df.loc[no_mask,  bleu_col].mean(),
            "No_BLEU_SD":   metric_df.loc[no_mask,  bleu_col].std(),
            "BLEU_Gap":     (metric_df.loc[yes_mask, bleu_col].mean()
                             - metric_df.loc[no_mask, bleu_col].mean()),
            "Yes_BERTP":    metric_df.loc[yes_mask, bert_col].mean(),
            "Yes_BERTP_SD": metric_df.loc[yes_mask, bert_col].std(),
            "No_BERTP":     metric_df.loc[no_mask,  bert_col].mean(),
            "No_BERTP_SD":  metric_df.loc[no_mask,  bert_col].std(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("exp9_ce_summary_table.csv", index=False)

    print("\n" + "="*70)
    print("DETECTION COUNTS (Exp 8 v2)")
    print("="*70)
    for _, r in summary_df.iterrows():
        print(f"  {r['Symptom']:<46} Yes={int(r['Yes_Count']):>4} "
              f"No={int(r['No_Count']):>4}")
    print(f"\n  TOTAL YES: {int(summary_df['Yes_Count'].sum())}")

    print("\n" + "="*70)
    print("STRATIFIED METRICS Yes vs No (Exp 8 v2)")
    print("="*70)
    hdr = (f"{'Symptom':<44} {'Lbl':>3} {'N':>5}  "
           f"{'Conf':>5} {'SD':>4}  {'BLEU':>5} {'SD':>5}  "
           f"{'BERT-P':>6} {'SD':>6}  {'Gap':>6}")
    print(hdr); print("-"*len(hdr))
    for _, r in summary_df.iterrows():
        sym = r["Symptom"]
        print(f"  {sym:<42} Yes {int(r['Yes_Count']):>5}  "
              f"{r['Yes_Conf']:>5.2f} {r['Yes_Conf_SD']:>4.2f}  "
              f"{r['Yes_BLEU']:>5.3f} {r['Yes_BLEU_SD']:>5.3f}  "
              f"{r['Yes_BERTP']:>6.4f} {r['Yes_BERTP_SD']:>6.4f}  "
              f"{r['BLEU_Gap']:>6.3f}")
        print(f"  {'':42}  No {int(r['No_Count']):>5}  "
              f"{r['No_Conf']:>5.2f} {r['No_Conf_SD']:>4.2f}  "
              f"{r['No_BLEU']:>5.3f} {r['No_BLEU_SD']:>5.3f}  "
              f"{r['No_BERTP']:>6.4f} {r['No_BERTP_SD']:>6.4f}")
        print()

    fh = summary_df[summary_df["Symptom"] == FH_SYM]
    if not fh.empty:
        fh = fh.iloc[0]
        print("--- Family History Contamination (KEY FINDING) ---")
        print(f"  Yes BLEU: {fh['Yes_BLEU']:.3f}")
        print(f"  No  BLEU: {fh['No_BLEU']:.3f}")
        print(f"  Gap:      {fh['BLEU_Gap']:.3f}  (E8 base: 0.078, E6: 0.039)")
        if   fh["BLEU_Gap"] >= 0.15: print("  STATUS: CONTAMINATION RESOLVED ✓")
        elif fh["BLEU_Gap"] >= 0.10: print("  STATUS: SUBSTANTIALLY REDUCED")
        elif fh["BLEU_Gap"] >= 0.05: print("  STATUS: PARTIAL IMPROVEMENT")
        else:                         print("  STATUS: CONTAMINATION PERSISTS")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp8 v2 (FIXED): Self-RAG + Corpus Filter + Reranker")
    parser.add_argument("--phase", default="all",
                        choices=["inference","metrics","all"])
    parser.add_argument("--model", default=HF_MODEL_ID)
    parser.add_argument("--max_notes", type=int, default=None)
    parser.add_argument("--max_rounds", type=int, default=MAX_CRITIQUE_ROUNDS)
    parser.add_argument("--no_reranker", action="store_true",
                        help="Disable cross-encoder (filter only)")
    args = parser.parse_args()
    MAX_CRITIQUE_ROUNDS = args.max_rounds

    print("\n" + "="*70)
    print("EXPERIMENT 8 v2 (FIXED) — SELF-RAG + CORPUS FILTER + RERANKER")
    print("  FIX A: FH corpus filter built ONCE at startup (not per note)")
    print("  FIX B: Single-keyword filter terms (match sentence-level text)")
    print("  FIX C: No log spam inside inference loop")
    print("="*70)

    umls_synonyms         = load_umls_synonyms()
    merged_vocab          = build_merged_vocabulary(umls_synonyms)
    cache, group_b_lookup = load_concept_cache()
    chunks                = load_chunks()
    notes_df              = load_notes()

    fh_chunks, fh_bm25 = build_fh_filtered_corpus(chunks)

    if args.max_notes:
        notes_df = notes_df.head(args.max_notes)
        print(f"[TEST MODE] {args.max_notes} notes")

    valid_df = None

    if args.phase in ("inference","all"):
        encoder    = MedCPTEncoder(HF_CACHE_DIR)
        chunk_vecs = build_dense_index(encoder, chunks)
        bm25       = build_bm25_index(chunks)

        reranker = CrossEncoderReranker() if not args.no_reranker else type(
            "DR", (), {"model": None, "rerank": lambda s,q,p,top_k=5: p[:top_k]})()

        tokenizer, model = load_llama(args.model)
        valid_df = run_inference(
            notes_df, encoder, chunk_vecs, chunks, bm25,
            fh_chunks, fh_bm25, reranker,
            merged_vocab, group_b_lookup, tokenizer, model)
        valid_df.to_csv("exp9_ce_valid_outputs.csv", index=False)

    if args.phase == "metrics":
        raw_df = pd.read_csv("exp9_ce_outputs_raw.csv")
        raw_df["exp_output_dict"] = raw_df["exp_output_raw"].apply(
            lambda x: flatten_parsed(safe_json_loads(str(x))[0])
            if pd.notna(x) else None)
        raw_df["exp_output_dict"] = raw_df["exp_output_dict"].apply(
            lambda x: x if isinstance(x, dict) else None)
        valid_df = raw_df[
            raw_df["exp_output_dict"].notna()
        ].copy().reset_index(drop=True)
        print(f"Valid rows: {len(valid_df)}")

    if args.phase in ("metrics","all") and valid_df is not None:
        run_metrics(valid_df)
