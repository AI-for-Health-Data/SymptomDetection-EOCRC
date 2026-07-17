import os, re, json, math, html, time, argparse
from collections import Counter, defaultdict
from pathlib import Path

import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ["TRANSFORMERS_NO_TF"]   = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"]               = "0"
os.environ["USE_FLAX"]             = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from bert_score import score as bert_score
    HAVE_BERTSCORE = True
    print("bert_score OK")
except ModuleNotFoundError:
    HAVE_BERTSCORE = False
    print("bert_score not installed — BERT-P will be NaN")


UMLS_API_KEY    = "60b16a44-704e-45a1-9fed-6b1c3a107f9f"
BASE_URL        = "https://uts-ws.nlm.nih.gov/rest"
PATH            = "rebuilt_notes_by_noteid.csv"
SYNONYMS_PATH   = "umls_synonyms.json"
HF_MODEL_ID     = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TEMPERATURE     = 0
MAX_NOTE_CHARS  = None
MAX_PER_SYMPTOM = 40

SYMPTOMS = [
    "Abdominal pain",
    "Rectal bleeding",
    "Rectal pain",
    "Diarrhea",
    "Constipation",
    "Weight loss",
    "Family history of colorectal cancer",
]

SYMPTOM_SPECS = [(s, f"{s} confidence", f"{s} inference") for s in SYMPTOMS]

# CUIs that map directly to our 7 symptoms
SYMPTOM_CUIS = {
    "C0000737": "Abdominal pain",
    "C0232492": "Abdominal pain",
    "C0232495": "Abdominal pain",
    "C0267596": "Rectal bleeding",
    "C0018932": "Rectal bleeding",
    "C0034886": "Rectal pain",
    "C0238637": "Rectal pain",
    "C0011991": "Diarrhea",
    "C2129214": "Diarrhea",
    "C0009806": "Constipation",
    "C0401149": "Constipation",
    "C1262477": "Weight loss",
    "C0043096": "Weight loss",
    "C0241889": "Family history of colorectal cancer",
}

BAD_INFERENCE_VALS = {
    "", "n/a", "na", "not mentioned", "none", "no inference",
    "not reported", "not applicable", "not stated",
    "not mentioned outside ros",
}

COLORS = {
    "Abdominal pain":                      "#FFD700",
    "Rectal bleeding":                     "#FF6B6B",
    "Rectal pain":                         "#FFA07A",
    "Diarrhea":                            "#90EE90",
    "Constipation":                        "#87CEEB",
    "Weight loss":                         "#DDA0DD",
    "Family history of colorectal cancer": "#F0E68C",
    "extra":                               "#D3D3D3",
}

STOP_WORDS = {
    "this", "that", "with", "from", "have", "been", "will", "were",
    "they", "them", "their", "said", "each", "which", "when", "your",
    "more", "also", "into", "than", "then", "some", "what", "there",
    "about", "would", "other", "these", "those", "after", "before",
    "patient", "report", "noted", "denies", "states", "history",
    "assessment", "plan", "visit", "follow", "reviewed", "discussed",
    "normal", "negative", "positive", "stable", "unchanged",
}


MANUAL_ALIASES = {
    "Abdominal pain": [
        "abdominal pain", "abd pain", "abd pn", "abdo pain", "stomach pain",
        "stomach ache", "stomachache", "tummy pain", "tummy ache", "belly pain",
        "bellyache", "gut pain", "abdominal discomfort", "abdominal soreness",
        "abdominal tenderness", "tender abdomen", "tenderness in abdomen",
        "epigastric pain", "epigastric discomfort", "epigastric tenderness",
        "ruq pain", "right upper quadrant pain", "luq pain", "left upper quadrant pain",
        "rlq pain", "right lower quadrant pain", "llq pain", "left lower quadrant pain",
        "periumbilical pain", "peri-umbilical pain", "umbilical pain",
        "suprapubic pain", "pelvic pain", "lower abdominal pain", "upper abdominal pain",
        "cramping", "abdominal cramping", "colicky pain", "colicky abdominal pain",
        "sharp abdominal pain", "dull abdominal pain", "burning abdominal pain",
        "gnawing pain", "pressure in abdomen", "fullness", "abdominal pressure",
        "bloating", "abdominal bloating", "distension", "abdominal distension",
        "gas pain", "gassy", "flatulence with pain",
        "dyspepsia", "indigestion", "heartburn", "acid reflux", "gerd symptoms",
        "gastritis", "peptic ulcer", "ulcer pain",
        "c/o abdominal pain", "complains of abdominal pain", "reports abdominal pain",
        "pain in abdomen",
    ],
    "Rectal bleeding": [
        "rectal bleeding", "bleeding per rectum", "blood per rectum",
        "blood from rectum", "rectal hemorrhage", "rectal haemorrhage", "rectorrhagia",
        "blood in stool", "bloody stool", "blood on stool", "stool with blood",
        "streaks of blood", "blood streaked stool", "blood on toilet paper",
        "blood when wiping", "hematochezia", "haematochezia", "brbpr",
        "bright red blood per rectum", "maroon stools", "melena", "black tarry stools",
        "positive fobt", "positive fit", "occult blood", "heme positive stool",
        "hemorrhoids with bleeding", "haemorrhoids with bleeding",
        "anal fissure bleeding", "fissure with bleeding",
    ],
    "Rectal pain": [
        "rectal pain", "pain in rectum", "painful rectum",
        "anal pain", "pain in anus", "anorectal pain",
        "proctalgia", "proctalgia fugax", "rectal discomfort",
        "anal discomfort", "rectal soreness", "anal soreness",
        "pain with bowel movement", "painful bowel movement", "painful defecation",
        "dyschezia", "odynochezia", "pain during defecation", "pain after bowel movement",
        "tenesmus", "rectal pressure", "feels pressure in rectum",
        "anal fissure pain", "fissure pain", "hemorrhoid pain",
        "thrombosed hemorrhoid", "perianal pain", "perirectal pain",
    ],
    "Diarrhea": [
        "diarrhea", "diarrhoea", "d+", "loose stools", "loose stool",
        "watery stools", "watery stool", "liquid stool", "runny stool",
        "the runs", "frequent stools", "frequent bowel movements",
        "increased bowel movements", "increased stool frequency",
        "multiple loose bms", "loose bm", "watery bm", "urgent bowel movements",
        "fecal urgency", "bowel urgency", "explosive diarrhea",
        "bristol 6", "bristol 7", "soft stools", "mushy stools",
        "gastroenteritis", "enteritis", "colitis with diarrhea",
        "travelers diarrhea", "c diff", "c. diff", "clostridioides difficile",
    ],
    "Constipation": [
        "constipation", "constipated", "hard stools", "hard stool",
        "infrequent stools", "infrequent bowel movements",
        "decreased stool frequency", "decreased bowel movements",
        "no bm", "no bowel movement", "no stool for",
        "difficulty passing stool", "difficulty stooling",
        "straining", "strains with bm", "incomplete evacuation",
        "obstipation", "fecal impaction", "stool impaction",
        "retained stool", "stool burden", "slow transit constipation",
        "bristol 1", "bristol 2", "pellet stools", "scybalous stools",
    ],
    "Weight loss": [
        "weight loss", "wt loss", "lost weight", "losing weight",
        "weight down", "weight decreased", "decreased weight",
        "unintentional weight loss", "unexplained weight loss",
        "involuntary weight loss", "clothes fitting looser",
        "poor weight gain", "failure to thrive",
        "cachexia", "wasting", "cachectic", "malnutrition",
        "anorexia", "loss of appetite", "decreased appetite",
    ],
    "Family history of colorectal cancer": [
        "family history of colorectal cancer", "family history colorectal cancer",
        "family history of colon cancer", "family history colon cancer",
        "fh colon cancer", "fhx colon cancer", "fhx crc", "fh crc",
        "crc in family", "colon ca in family", "colon cancer runs in family",
        "mother had colon cancer", "father had colon cancer",
        "sister had colon cancer", "brother had colon cancer",
        "first degree relative with colon cancer", "fdr with colon cancer",
        "lynch syndrome", "hnpcc", "familial adenomatous polyposis", "fap",
        "hereditary colorectal cancer",
        "family history of bowel cancer", "fh bowel cancer",
    ],
}


def load_notes(path=PATH):
    notes_df = pd.read_csv(path)
    if "Clean_note_text" not in notes_df.columns:
        raise ValueError("Clean_note_text not found.")
    for col in ["DATE_OF_SERVIC_DTTM", "SPEC_NOTE_TIME_DTTM", "CONTACT_DATE"]:
        if col in notes_df.columns:
            notes_df[col] = pd.to_datetime(notes_df[col], errors="coerce")
    sort_cols = [c for c in [
        "PAT_ID", "PAT_ENC_CSN_ID",
        "DATE_OF_SERVIC_DTTM", "SPEC_NOTE_TIME_DTTM", "CONTACT_NUM", "NOTE_ID"
    ] if c in notes_df.columns]
    if sort_cols:
        notes_df = notes_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    notes_df = notes_df[notes_df["Clean_note_text"].str.strip().ne("")].reset_index(drop=True)
    print(f"Notes loaded: {len(notes_df)}")
    return notes_df


def load_umls_synonyms(path=SYNONYMS_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"'{path}' not found.")
    with open(path) as f:
        return json.load(f)


def build_symptom_lookup(umls_synonyms):
    """
    {term_lower → symptom} from MANUAL_ALIASES + umls_synonyms.json
    MANUAL takes priority. Sorted longest-first.
    """
    lookup = {}
    for symptom, aliases in MANUAL_ALIASES.items():
        for alias in aliases:
            k = alias.lower().strip()
            if k and k not in lookup:
                lookup[k] = symptom
    for symptom, aliases in umls_synonyms.items():
        for alias in aliases:
            k = alias.lower().strip()
            if k and k not in lookup:
                lookup[k] = symptom
    return dict(sorted(lookup.items(), key=lambda x: -len(x[0])))


def classify_concept(term, cui, symptom_lookup):
    """
    Classify a UMLS concept into one of 7 symptoms or 'extra'.
    Priority: CUI match → term match → partial match → extra
    """
    if cui and cui in SYMPTOM_CUIS:
        return SYMPTOM_CUIS[cui]
    term_lower = term.lower().strip()
    if term_lower in symptom_lookup:
        return symptom_lookup[term_lower]
    for alias, symptom in symptom_lookup.items():
        if len(alias) > 5 and alias in term_lower:
            return symptom
    return "extra"


def normalize_answer(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def to_num(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"-?\d+(?:\.\d+)?", str(x))
    return float(m.group()) if m else np.nan


_token_pat = re.compile(r"\w+|\S")

def simple_tokenize(text):
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    return _token_pat.findall(text.lower())


def modified_precision(ref_tokens, hyp_tokens, n):
    if len(hyp_tokens) < n:
        return 0.0
    hyp_ngrams = Counter(zip(*[hyp_tokens[i:] for i in range(n)]))
    ref_ngrams  = Counter(zip(*[ref_tokens[i:]  for i in range(n)]))
    match = sum(min(c, ref_ngrams[ng]) for ng, c in hyp_ngrams.items())
    total = sum(hyp_ngrams.values())
    return 0.0 if total == 0 else match / total


def compute_bleu_no_bp(ref, hyp, max_n=4):
    ref_toks = simple_tokenize(ref)
    hyp_toks = simple_tokenize(hyp)
    if not ref_toks or not hyp_toks:
        return 0.0
    max_n = min(max_n, len(hyp_toks))
    log_precs = []
    for n in range(1, max_n + 1):
        p = modified_precision(ref_toks, hyp_toks, n)
        if p == 0.0:
            return 0.0
        log_precs.append(math.log(p))
    return math.exp(sum(log_precs) / len(log_precs))


def compute_bertscore_batch(refs, hyps):
    if not refs:
        return []
    P, R, F1 = bert_score(hyps, refs, lang="en",
                           rescale_with_baseline=False,
                           batch_size=16, verbose=False)
    return [float(x) for x in P]


def maybe_truncate(text, max_chars=None):
    if text is None:
        return ""
    t = str(text)
    if max_chars is None or len(t) <= max_chars:
        return t
    return t[: max_chars // 2] + "\n...\n" + t[-(max_chars // 2):]


def strip_code_fences(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    return s


def extract_first_json_object(s):
    s = strip_code_fences(s)
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return s


def safe_json_loads(s):
    s0 = extract_first_json_object(s)
    try:
        return json.loads(s0), s0
    except Exception:
        pass
    s1 = s0
    s1 = re.sub(r'("|\d|true|false|null)\s*\n(\s*")', r'\1,\n\2', s1)
    s1 = re.sub(r",\s*([}\]])", r"\1", s1)
    s1 = (s1.replace("\u201c", '"').replace("\u201d", '"')
             .replace("\u2018", "'").replace("\u2019", "'"))
    s1 = re.sub(r':\s*N/A(\s*[,\n}\]])',  r': "N/A"\1',  s1)
    s1 = re.sub(r':\s*None(\s*[,\n}\]])', r': "None"\1', s1)
    s1 = re.sub(r':\s*n/a(\s*[,\n}\]])',  r': "N/A"\1',  s1)
    try:
        return json.loads(s1), s1
    except Exception:
        pass
    s2 = re.sub(r'("|\d|true|false|null)(\s*")', r'\1,\2', s1)
    try:
        return json.loads(s2), s2
    except Exception:
        return None, s0


def extract_candidate_phrases(note_text):
    """Extract 2-5 word candidate phrases for UMLS API lookup."""
    patterns = [
        r'\b[A-Za-z](?:[A-Za-z\-]+\s+){3,4}[A-Za-z\-]+\b',
        r'\b[A-Za-z](?:[A-Za-z\-]+\s+){1,2}[A-Za-z\-]+\b',
    ]
    candidates = []
    seen = set()
    for pattern in patterns:
        for m in re.finditer(pattern, note_text, re.IGNORECASE):
            phrase = m.group().strip()
            pl = phrase.lower()
            if pl in seen or pl in STOP_WORDS or len(phrase) < 4:
                continue
            seen.add(pl)
            candidates.append((phrase, m.start(), m.end()))
    return candidates


def search_umls_api(phrase, api_key):
    """
    Call UMLS /search for a phrase → return ALL matching concepts.
    This finds ANY clinical concept, not just the 7 symptoms.
    """
    for search_type in ("exact", "words"):
        try:
            r = requests.get(
                f"{BASE_URL}/search/current",
                params={"apiKey": api_key, "string": phrase,
                        "searchType": search_type, "pageSize": 3},
                timeout=20,
            )
            if r.status_code != 200:
                continue
            results = r.json().get("result", {}).get("results", [])
            found = [{"cui": res["ui"], "name": res["name"]}
                     for res in results
                     if res.get("ui") and res["ui"] != "NONE" and res.get("name")]
            if found:
                return found
            time.sleep(0.05)
        except Exception:
            continue
    return []


def run_phase1(notes_df, umls_synonyms, api_key=UMLS_API_KEY,
               use_api=True, show_notes=3):
    """
    PHASE 1: Scan every note for ALL UMLS clinical concepts.
    PHASE 2: Group into 7 symptoms + save extras.
    Saves highlighted HTML showing all found concepts color-coded.
    """
    print("\n" + "="*65)
    print("PHASE 1 — Find ALL UMLS Clinical Concepts")
    print("PHASE 2 — Group into 7 Symptoms + Save Extras")
    print("="*65)

    # Build symptom lookup for classification
    symptom_lookup = build_symptom_lookup(umls_synonyms)
    print(f"MANUAL aliases:         {sum(len(v) for v in MANUAL_ALIASES.values())} terms")
    print(f"UMLS synonyms:          {sum(len(v) for v in umls_synonyms.values())} terms")
    print(f"Combined lookup:        {len(symptom_lookup)} unique terms")
    print(f"API mode:               {'ON — finds ALL clinical concepts' if use_api else 'OFF — dictionary only'}")

    # Also verify API
    if use_api:
        r = requests.get(f"{BASE_URL}/content/current/CUI/C0000737",
                         params={"apiKey": api_key}, timeout=10)
        if r.status_code == 200:
            print(f"UMLS API connected:     OK")
        else:
            print(f"UMLS API error {r.status_code} — switching to dictionary")
            use_api = False

    all_concept_rows = []
    symptom_rows     = []
    extra_rows       = []
    html_blocks      = []
    notes_shown      = 0

    for _, row in tqdm(notes_df.iterrows(), total=len(notes_df),
                       desc="PHASE 1+2"):

        note_text = str(row.get("Clean_note_text", ""))
        pat_id    = row.get("PAT_ID", "")
        note_id   = row.get("NOTE_ID", "")
        enc_id    = row.get("PAT_ENC_CSN_ID", "")

        # Find ALL UMLS concepts
        found   = []   # (start, end, phrase, cui, umls_name)
        covered = []

        if use_api:
            for phrase, ph_s, ph_e in extract_candidate_phrases(note_text):
                if any(cs <= ph_s < ce or cs < ph_e <= ce for cs, ce in covered):
                    continue
                api_results = search_umls_api(phrase, api_key)
                if not api_results:
                    continue
                best = api_results[0]
                pattern = r'\b' + re.escape(phrase) + r'\b'
                m = re.search(pattern, note_text, re.IGNORECASE)
                if not m:
                    continue
                s, e = m.start(), m.end()
                if any(cs <= s < ce or cs < e <= ce for cs, ce in covered):
                    continue
                covered.append((s, e))
                found.append((s, e, phrase, best["cui"], best["name"]))
            time.sleep(0.02)
        else:
            # Dictionary fallback — scan with combined lookup
            text_lower = note_text.lower()
            for term, symptom in symptom_lookup.items():
                pattern = r'\b' + re.escape(term) + r'\b'
                for m in re.finditer(pattern, text_lower):
                    s, e = m.start(), m.end()
                    if any(cs <= s < ce or cs < e <= ce for cs, ce in covered):
                        continue
                    covered.append((s, e))
                    found.append((s, e, term, "", term))

        # PHASE 2: Classify each concept
        symptom_groups = defaultdict(list)
        extra_concepts = []
        classified     = []   # (s, e, phrase, cui, name, symptom)

        for s, e, phrase, cui, name in found:
            symptom = classify_concept(name, cui, symptom_lookup)
            if symptom == "extra":
                symptom = classify_concept(phrase, "", symptom_lookup)
            label = symptom if symptom in SYMPTOMS else "extra"
            if label in SYMPTOMS:
                symptom_groups[label].append(name)
            else:
                extra_concepts.append(name)
            classified.append((s, e, phrase, cui, name, label))

        # Console sample c
        if notes_shown < show_notes:
            print(f"\n{'='*65}")
            print(f"PAT_ID: {pat_id}  |  NOTE_ID: {note_id}")
            print(f"{'='*65}")
            # Show note with inline markers
            if classified:
                parts, prev = [], 0
                abbrev = {"Abdominal pain":"ABD","Rectal bleeding":"RBLEED",
                          "Rectal pain":"RPAIN","Diarrhea":"DIARR",
                          "Constipation":"CONST","Weight loss":"WTLOSS",
                          "Family history of colorectal cancer":"FHCRC"}
                for s, e, phrase, cui, name, lbl in sorted(classified, key=lambda x: x[0]):
                    parts.append(note_text[prev:s])
                    tag = abbrev.get(lbl, "EXTRA")
                    parts.append(f"[[{note_text[s:e]}|{tag}:{name[:20]}]]")
                    prev = e
                parts.append(note_text[prev:])
                print("".join(parts)[:2000])
            print(f"\n--- GROUPED CONCEPTS ---")
            for sym in SYMPTOMS:
                terms = symptom_groups.get(sym, [])
                if terms:
                    print(f"  [{sym}]")
                    for t in sorted(set(terms))[:5]:
                        print(f"      -> '{t}'")
            if extra_concepts:
                print(f"  [EXTRA] {sorted(set(extra_concepts))[:5]}")
            print(f"  Total: {len(found)} concepts found, "
                  f"{len(extra_concepts)} extras")
            notes_shown += 1

        # Build rows
        for s, e, phrase, cui, name, lbl in classified:
            all_concept_rows.append({
                "PAT_ID": pat_id, "PAT_ENC_CSN_ID": enc_id, "NOTE_ID": note_id,
                "phrase": phrase, "cui": cui, "umls_name": name,
                "symptom_group": lbl,
                "char_start": s, "char_end": e,
                "context": note_text[max(0,s-25):e+25].replace("\n"," "),
            })

        sym_row = {"PAT_ID": pat_id, "PAT_ENC_CSN_ID": enc_id, "NOTE_ID": note_id}
        for symptom in SYMPTOMS:
            terms = symptom_groups.get(symptom, [])
            sym_row[f"{symptom}_hit"]   = 1 if terms else 0
            sym_row[f"{symptom}_terms"] = " | ".join(sorted(set(terms)))
            sym_row[f"{symptom}_count"] = len(set(terms))
        symptom_rows.append(sym_row)

        if extra_concepts:
            extra_rows.append({
                "PAT_ID": pat_id, "PAT_ENC_CSN_ID": enc_id, "NOTE_ID": note_id,
                "extra_terms": " | ".join(sorted(set(extra_concepts))),
                "n_extra": len(set(extra_concepts)),
            })

        html_blocks.append(
            _make_note_html(note_text, classified, pat_id, note_id,
                            dict(symptom_groups))
        )

    concept_df = pd.DataFrame(all_concept_rows)
    concept_df.to_csv("all_umls_concepts.csv", index=False)

    symptom_df = pd.DataFrame(symptom_rows)
    symptom_df.to_csv("note_symptom_groups.csv", index=False)

    extra_df = pd.DataFrame(
        extra_rows if extra_rows else [],
        columns=["PAT_ID","PAT_ENC_CSN_ID","NOTE_ID","extra_terms","n_extra"]
    )
    extra_df.to_csv("extra_note_umls.csv", index=False)

    summary_rows = []
    for symptom in SYMPTOMS:
        col   = f"{symptom}_hit"
        n_hit = int(symptom_df[col].sum())
        summary_rows.append({
            "Symptom": symptom, "Notes_with_hit": n_hit,
            "Total_notes": len(symptom_df),
            "Hit_rate_pct": round(100*n_hit/max(len(symptom_df),1),1),
        })
    pd.DataFrame(summary_rows).to_csv("symptom_hit_summary.csv", index=False)

    # HTML
    legend_items = "".join(
        f'<span style="background:{c};padding:2px 8px;margin:2px;'
        f'border-radius:3px;font-size:11px;">'
        f'{"Extra" if s=="extra" else s}</span>'
        for s, c in COLORS.items()
    )
    legend = (f"<div style='padding:10px;background:#f8f8f8;"
              f"border-radius:6px;margin:10px 0;'>{legend_items}</div>")
    full_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>EOCRC UMLS Concept Extraction</title>
<style>
  body{{font-family:monospace;background:#f5f5f5;margin:20px;max-width:1200px;}}
  h1{{font-family:Georgia;color:#1a1a2e;border-bottom:3px solid #1a1a2e;padding-bottom:8px;}}
  .stats{{background:#1a1a2e;color:#fff;padding:10px 16px;border-radius:6px;font-size:13px;margin:10px 0;}}
  .stats strong{{color:#FFD700;}}
  .note{{background:#fff;border:1px solid #ccc;border-radius:8px;margin:14px 0;overflow:hidden;}}
  .note-header{{background:#1a1a2e;color:#fff;padding:8px 14px;font-family:Georgia;font-size:12px;}}
  .note-groups{{background:#fafafa;border-bottom:1px solid #eee;padding:8px 14px;}}
  .note-body{{padding:14px;font-size:13px;line-height:1.7;white-space:pre-wrap;}}
</style></head><body>
<h1>EOCRC UMLS Clinical Concept Extraction — Experiment 3</h1>
<div class="stats">
  Notes: <strong>{len(notes_df)}</strong> |
  UMLS concepts found: <strong>{len(concept_df)}</strong> |
  Notes with extras: <strong>{len(extra_df)}</strong>
</div>
{legend}
{"".join(html_blocks)}
</body></html>"""
    with open("highlighted_notes.html", "w", encoding="utf-8") as f:
        f.write(full_html)

    # Print summary
    print(f"\nFiles saved:")
    print(f"  all_umls_concepts.csv    {len(concept_df)} concepts")
    print(f"  note_symptom_groups.csv  {len(symptom_df)} notes")
    print(f"  extra_note_umls.csv      {len(extra_df)} notes with extras")
    print(f"  symptom_hit_summary.csv")
    print(f"  highlighted_notes.html")

    print("\n" + "="*65)
    print("EXTRACTION SUMMARY (PHASE 1+2)")
    print("="*65)
    print(f"{'Symptom':<46} {'Hits':>8} {'Rate':>7}")
    print("-"*65)
    for r in summary_rows:
        print(f"  {r['Symptom']:<44} {r['Notes_with_hit']:>8}  {r['Hit_rate_pct']:>5.1f}%")

    if len(extra_df) > 0:
        all_extras = []
        for terms in extra_df["extra_terms"].dropna():
            all_extras.extend([t.strip() for t in terms.split("|")])
        print(f"\n  Top 10 extra clinical concepts:")
        for term, cnt in Counter(all_extras).most_common(10):
            print(f"    {cnt:>4}x  '{term}'")

    return concept_df, symptom_df, extra_df


def _make_note_html(note_text, classified, pat_id, note_id, symptom_groups):
    if not classified:
        body = html.escape(note_text).replace("\n", "<br>")
        return (f"<div class='note'><div class='note-header'>"
                f"PAT {html.escape(str(pat_id))} | NOTE {html.escape(str(note_id))}"
                f" — no concepts</div><div class='note-body'>{body}</div></div>")
    spans = sorted(classified, key=lambda x: x[0])
    parts, prev = [], 0
    for s, e, phrase, cui, name, symptom in spans:
        parts.append(html.escape(note_text[prev:s]))
        color = COLORS.get(symptom, COLORS["extra"])
        tip   = html.escape(f"{name} [{cui}] → {symptom}")
        parts.append(
            f'<mark style="background:{color};padding:1px 4px;border-radius:3px;'
            f'cursor:help;" title="{tip}">{html.escape(note_text[s:e])}</mark>'
        )
        prev = e
    parts.append(html.escape(note_text[prev:]))
    body = "".join(parts).replace("\n", "<br>")
    group_html = ""
    for sym in SYMPTOMS:
        terms = symptom_groups.get(sym, [])
        if terms:
            color = COLORS[sym]
            tlist = ", ".join(
                f'<span style="background:{color};padding:0 3px;'
                f'border-radius:2px;font-size:10px;">{html.escape(t)}</span>'
                for t in sorted(set(terms))
            )
            group_html += (f"<div style='margin:2px;font-size:11px;'>"
                           f"<b>{html.escape(sym)}:</b> {tlist}</div>")
    return f"""<div class="note">
  <div class="note-header">PAT_ID: <b>{html.escape(str(pat_id))}</b> | NOTE_ID: <b>{html.escape(str(note_id))}</b></div>
  <div class="note-groups">{group_html or '<em>No symptom groups</em>'}</div>
  <div class="note-body">{body}</div>
</div>"""


PROMPT_TEMPLATE = """
You are an experienced gastroenterology clinician.
You will analyze a patient's ORIGINAL clinical note text and extract information
ONLY from the note text (no assumptions, no external knowledge).

IMPORTANT RULE ABOUT ROS (Review of Systems):
- ROS often contains templated negatives (e.g., "no abdominal pain", "negative for diarrhea")
  that can conflict with the rest of the note.
- You MUST NOT use ROS-negative templated statements as evidence to answer "No".
- If ROS says "no X" but other parts (Chief Complaint, HPI, Assessment/Plan, Diagnosis)
  indicate X, then answer "Yes" using NON-ROS evidence for inference.
- If the note (outside ROS) explicitly denies a symptom (e.g., in HPI: "patient denies rectal bleeding"),
  then answer "No" with high confidence and use that NON-ROS denial as inference.
- If the symptom is not mentioned outside ROS at all, answer "No" with low confidence (2),
  and inference can be "Not mentioned outside ROS".

SYNONYM GUIDANCE — treat these terms as matches for each symptom:
{ALIAS_SECTION}

For each symptom provide:
- Answer: Yes or No
- Confidence: integer 1-5 (1=Very Low, 5=Very High)
- Inference: short quote copied from note text supporting the answer
  (must NOT come from ROS-negative text)

Return ONLY a valid JSON object with these keys:
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
"Family history of colorectal cancer", "Family history of colorectal cancer confidence", "Family history of colorectal cancer inference",
"Other comments", "Other comments confidence", "Other comments inference"

Rules:
- Clearly present outside ROS -> "Yes" (confidence 4-5)
- Explicitly denied outside ROS -> "No" (confidence 4-5)
- Only in ROS-negative, nowhere else -> "No" (confidence 2)
- Use "N/A" for duration if not reported
- Output ONLY JSON — no prose, no markdown

Patient NOTE TEXT:
<<NOTE_TEXT>>
""".strip()


def build_alias_section(umls_synonyms, max_per=MAX_PER_SYMPTOM):
    """Merge MANUAL + UMLS into {ALIAS_SECTION} prompt block."""
    merged = {s: list(aliases) for s, aliases in MANUAL_ALIASES.items()}
    for symptom, aliases in umls_synonyms.items():
        manual_lower = {a.lower() for a in merged.get(symptom, [])}
        for a in aliases:
            if a.lower() not in manual_lower:
                merged.setdefault(symptom, []).append(a)
    lines = []
    for symptom in SYMPTOMS:
        aliases = merged.get(symptom, [])
        seen, deduped = set(), []
        for a in aliases:
            if a.lower() not in seen:
                seen.add(a.lower())
                deduped.append(a)
        deduped = deduped[:max_per]
        lines.append(f"{symptom}: {', '.join(deduped)}" if deduped
                     else f"{symptom}: (none)")
    return "\n".join(lines)


def load_llama(model_id=HF_MODEL_ID):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"\nLoading: {model_id}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {round(torch.cuda.get_device_properties(0).total_memory/1e9,1)} GB")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True,
        cache_dir="/lustre/smuexa01/client/users/nikkieh/hf_cache")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
        cache_dir="/lustre/smuexa01/client/users/nikkieh/hf_cache")
    model.eval()
    print("Model loaded")
    return tokenizer, model


def generate(prompt, tokenizer, model, max_new_tokens=1024):
    import torch
    messages = [{"role": "user", "content": prompt}]
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


def run_phase3(notes_df, umls_synonyms, model_id=HF_MODEL_ID):
    """PHASE 3: LLaMA inference with MANUAL+UMLS synonyms in prompt."""
    print("\n" + "="*65)
    print("PHASE 3 — LLaMA Inference (Experiment 3)")
    print("="*65)

    alias_section = build_alias_section(umls_synonyms)
    prompt_filled = PROMPT_TEMPLATE.replace("{ALIAS_SECTION}", alias_section)
    print(f"Prompt length (without note): {len(prompt_filled):,} chars")
    print(f"Alias section: MANUAL + UMLS merged, capped at {MAX_PER_SYMPTOM}/symptom")

    tokenizer, model = load_llama(model_id)
    rows = []

    for idx, row in tqdm(notes_df.iterrows(), total=len(notes_df),
                         desc="Exp 3 inference"):
        note_text = maybe_truncate(row["Clean_note_text"], MAX_NOTE_CHARS)
        prompt    = prompt_filled.replace("<<NOTE_TEXT>>", note_text)
        try:
            content = generate(prompt, tokenizer, model)
        except Exception as e:
            content = ""
            print(f"  Error {idx}: {e}")

        parsed, raw_json = safe_json_loads(content)
        out = row.to_dict()
        out["exp3_output_raw"]  = raw_json
        out["exp3_output_dict"] = parsed
        rows.append(out)

        if (idx + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv("experiment3_checkpoint.csv", index=False)
            print(f"  Checkpoint: {idx+1}/{len(notes_df)}")

    exp3_df = pd.DataFrame(rows)
    exp3_df.to_csv("experiment3_outputs_raw.csv", index=False)
    total    = len(exp3_df)
    n_failed = exp3_df["exp3_output_dict"].apply(lambda x: not isinstance(x, dict)).sum()
    print(f"\nexperiment3_outputs_raw.csv  ({total} rows)")
    print(f"Parse failures: {n_failed}/{total}  ({100*n_failed/total:.1f}%)")

    valid_df = exp3_df[
        exp3_df["exp3_output_dict"].apply(lambda x: isinstance(x, dict))
    ].copy().reset_index(drop=True)
    print(f"Valid rows: {len(valid_df)}/{total}")
    return valid_df


def run_phase4(valid_df):
    """PHASE 4: Compute and print Table I and Table II."""
    print("\n" + "="*65)
    print("PHASE 4 — Metrics (Table I + Table II)")
    print("="*65)

    # Yes counts
    count_rows = []
    for symptom, conf_key, inf_key in SYMPTOM_SPECS:
        yes_count = valid_df["exp3_output_dict"].apply(
            lambda d, s=symptom: normalize_answer(d.get(s,"")) == "yes"
        ).sum()
        count_rows.append({"Symptom": symptom, "Exp3_Yes_Count": int(yes_count)})
    count_df = pd.DataFrame(count_rows)
    count_df.to_csv("experiment3_yes_counts.csv", index=False)

    # Unpack JSON
    metric_df = valid_df.copy()
    for symptom, conf_key, inf_key in SYMPTOM_SPECS:
        metric_df[symptom]  = metric_df["exp3_output_dict"].apply(
            lambda d, s=symptom:  d.get(s,"")     if isinstance(d,dict) else "")
        metric_df[conf_key] = metric_df["exp3_output_dict"].apply(
            lambda d, k=conf_key: d.get(k,np.nan) if isinstance(d,dict) else np.nan)
        metric_df[inf_key]  = metric_df["exp3_output_dict"].apply(
            lambda d, k=inf_key:  d.get(k,"")     if isinstance(d,dict) else "")
        metric_df[f"{symptom} Conf_num"] = metric_df[conf_key].apply(to_num)

    # BLEU
    print("Computing BLEU...")
    for symptom, _, inf_key in SYMPTOM_SPECS:
        bleu_col = f"{symptom} BLEU_noBP"
        vals = []
        for _, row in metric_df.iterrows():
            hyp = row[inf_key]
            ref = row["Clean_note_text"]
            if not isinstance(hyp,str) or hyp.strip().lower() in BAD_INFERENCE_VALS:
                vals.append(0.0)
            else:
                vals.append(compute_bleu_no_bp(ref, hyp))
        metric_df[bleu_col] = vals
    print("BLEU done.")

    # BERTScore
    for symptom, _, _ in SYMPTOM_SPECS:
        metric_df[f"{symptom} BERT_P"] = np.nan
    if HAVE_BERTSCORE:
        print("Computing BERTScore...")
        for symptom, _, inf_key in SYMPTOM_SPECS:
            bert_col = f"{symptom} BERT_P"
            idxs, refs, hyps = [], [], []
            for i, row in metric_df.iterrows():
                hyp = row[inf_key]
                ref = row["Clean_note_text"]
                if not isinstance(hyp,str) or hyp.strip().lower() in BAD_INFERENCE_VALS:
                    continue
                idxs.append(i); refs.append(ref); hyps.append(hyp)
            print(f"  {symptom}: {len(idxs)} rows")
            if idxs:
                vals = compute_bertscore_batch(refs, hyps)
                for i, val in zip(idxs, vals):
                    metric_df.at[i, bert_col] = val
        print("BERTScore done.")

    metric_df.to_csv("experiment3_note_level_metrics.csv", index=False)

    # Build summary table
    summary_rows = []
    for symptom, conf_key, inf_key in SYMPTOM_SPECS:
        conf_col = f"{symptom} Conf_num"
        bleu_col = f"{symptom} BLEU_noBP"
        bert_col = f"{symptom} BERT_P"
        labels   = metric_df[symptom].astype(str).str.strip().str.lower()
        yes_mask = labels == "yes"
        no_mask  = labels == "no"
        summary_rows.append({
            "Symptom":        symptom,
            "Yes_Count":      int(yes_mask.sum()),
            "No_Count":       int(no_mask.sum()),
            "Conf_Mean":      metric_df[conf_col].mean(),
            "Conf_SD":        metric_df[conf_col].std(),
            "Conf_Mean_Yes":  metric_df.loc[yes_mask, conf_col].mean(),
            "Conf_Mean_No":   metric_df.loc[no_mask,  conf_col].mean(),
            "BLEU_Mean":      metric_df[bleu_col].mean(),
            "BLEU_SD":        metric_df[bleu_col].std(),
            "BLEU_Mean_Yes":  metric_df.loc[yes_mask, bleu_col].mean(),
            "BLEU_Mean_No":   metric_df.loc[no_mask,  bleu_col].mean(),
            "BERTP_Mean":     metric_df[bert_col].mean(),
            "BERTP_SD":       metric_df[bert_col].std(),
            "BERTP_Mean_Yes": metric_df.loc[yes_mask, bert_col].mean(),
            "BERTP_Mean_No":  metric_df.loc[no_mask,  bert_col].mean(),
        })

    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv("experiment3_summary_table.csv", index=False)

    # TABLE I
    print("\n" + "="*65)
    print("TABLE I — POSITIVE SYMPTOM DETECTION COUNTS (EXPERIMENT 3)")
    print("="*65)
    print(f"{'Symptom':<46} {'Yes':>6} {'No':>6}")
    print("-"*60)
    for _, r in summary_table.iterrows():
        print(f"  {r['Symptom']:<44} {int(r['Yes_Count']):>6} {int(r['No_Count']):>6}")
    print(f"\n  TOTAL YES: {int(summary_table['Yes_Count'].sum())}")

    # TABLE II
    print("\n" + "="*65)
    print("TABLE II — CONFIDENCE, BLEU, BERTSCORE PRECISION (EXPERIMENT 3)")
    print("Mean +- SD across ALL predictions")
    print("="*65)
    print(f"{'Symptom':<46} {'Conf':>12} {'BLEU':>12} {'BERT-P':>12}")
    print("-"*84)
    for _, r in summary_table.iterrows():
        print(f"  {r['Symptom']:<44} "
              f"{r['Conf_Mean']:>4.2f}+-{r['Conf_SD']:>4.2f}  "
              f"{r['BLEU_Mean']:>4.2f}+-{r['BLEU_SD']:>4.2f}  "
              f"{r['BERTP_Mean']:>4.2f}+-{r['BERTP_SD']:>4.2f}")

    print("\n  Stratified Yes vs No:")
    print(f"  {'Symptom':<44} {'CY':>5} {'CN':>5} {'BY':>6} {'BN':>6} {'PY':>7} {'PN':>7}")
    print("  " + "-"*80)
    for _, r in summary_table.iterrows():
        print(f"  {r['Symptom']:<44} "
              f"{r['Conf_Mean_Yes']:>5.2f} {r['Conf_Mean_No']:>5.2f} "
              f"{r['BLEU_Mean_Yes']:>6.3f} {r['BLEU_Mean_No']:>6.3f} "
              f"{r['BERTP_Mean_Yes']:>7.3f} {r['BERTP_Mean_No']:>7.3f}")
    print("  CY=Conf Yes  CN=Conf No  BY=BLEU Yes  BN=BLEU No  PY=BERT-P Yes  PN=BERT-P No")
    print("  Expected: CN approx 2.3")

    print("\nFiles saved:")
    print("  experiment3_yes_counts.csv")
    print("  experiment3_note_level_metrics.csv")
    print("  experiment3_summary_table.csv")

    return summary_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EOCRC Full Pipeline — UMLS Extraction + Experiment 3"
    )
    parser.add_argument("--phase",      default="all",
                        choices=["extract","inference","metrics","all"])
    parser.add_argument("--model",      default=HF_MODEL_ID)
    parser.add_argument("--max_notes",  type=int, default=None)
    parser.add_argument("--show_notes", type=int, default=3)
    parser.add_argument("--no_api",     action="store_true",
                        help="Skip UMLS API in PHASE 1 (dictionary only)")
    args = parser.parse_args()

    notes_df      = load_notes()
    umls_synonyms = load_umls_synonyms()
    if args.max_notes:
        notes_df = notes_df.head(args.max_notes)

    print(f"UMLS synonyms: {sum(len(v) for v in umls_synonyms.values())} terms")
    print(f"MANUAL aliases: {sum(len(v) for v in MANUAL_ALIASES.values())} terms")

    if args.phase in ("extract", "all"):
        run_phase1(notes_df, umls_synonyms,
                   use_api=not args.no_api,
                   show_notes=args.show_notes)

    if args.phase in ("inference", "all"):
        valid_df = run_phase3(notes_df, umls_synonyms, model_id=args.model)
        valid_df.to_csv("experiment3_valid_outputs.csv", index=False)

    if args.phase == "metrics":
        raw_df = pd.read_csv("experiment3_outputs_raw.csv")
        raw_df["exp3_output_dict"] = raw_df["exp3_output_raw"].apply(
            lambda x: safe_json_loads(str(x))[0] if pd.notna(x) else None)
        valid_df = raw_df[
            raw_df["exp3_output_dict"].apply(lambda x: isinstance(x, dict))
        ].copy().reset_index(drop=True)
        print(f"Valid rows loaded: {len(valid_df)}")

    if args.phase in ("metrics", "all"):
        run_phase4(valid_df)
