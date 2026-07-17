import re, json, time, sys, requests, pandas as pd
from collections import Counter

PATH       = "rebuilt_notes_by_noteid.csv"
CACHE_PATH = "umls_concept_cache.json"
UMLS_API_KEY = "60b16a44-704e-45a1-9fed-6b1c3a107f9f"
BASE         = "https://uts-ws.nlm.nih.gov/rest"

SYMPTOM_CUI_FAMILIES = {
    "C0000737","C0152171","C0232503","C0694868","C1963065","C0085584",
    "C0267596","C0018932","C0018937","C1321898","C0025209","C0267615","C0267614",
    "C0034886","C0085606","C0085644","C0232607","C0232608",
    "C0011991","C0152164","C0860904","C0232726","C0232727",
    "C0009806","C0687720","C0232720","C0232721",
    "C1262477","C0043096","C0085295","C0003123","C0162429",
    "C0241889","C0332265","C0728708","C1553497",
}

CLINICAL_STYS = {
    "T047",  # Disease or Syndrome
    "T191",  # Neoplastic Process
    "T033",  # Finding
    "T034",  # Laboratory or Test Result
    "T060",  # Diagnostic Procedure
    "T061",  # Therapeutic or Preventive Procedure
    "T046",  # Pathologic Function
    "T184",  # Sign or Symptom
    "T032",  # Organism Attribute
    "T201",  # Clinical Attribute
    "T023",  # Body Part, Organ, or Organ Component
    "T121",  # Pharmacologic Substance
    "T028",  # Gene or Genome
    "T116",  # Amino Acid, Peptide, or Protein
    "T058",  # Health Care Activity
    "T059",  # Laboratory Procedure
}

STOPWORDS = {
    "the","and","or","but","in","on","at","to","for","of","a","an",
    "is","was","were","are","has","have","had","been","be","not","no",
    "with","without","from","by","as","this","that","these","those",
    "patient","patients","pt","she","he","they","we","it","his","her",
    "their","our","which","who","when","where","how","what","than","then",
    "also","only","well","will","would","could","should","may","might",
    "does","did","do","per","via","vs","about","any","all","both","each",
    "after","before","since","until","during","within","between","among",
    "left","right","upper","lower","mild","moderate","severe","stable",
    "normal","abnormal","positive","negative","elevated","decreased",
    "history","medical","clinical","surgical","family","social",
    "review","systems","plan","assessment","note","date","time",
    "pain","complaint","chief","denies","deny","weight",
    "respiratory","presents","presenting","bowel","medications",
    "oriented","fever","ros","subjective","objective",
    "hpi","results","heart","general","abdomen",
    "extremities","neuro","allergies","follow","return",
    "alert","exam","physical","vital","vitals","temp",
    "pulse","blood","pressure","oxygen","saturation",
    "impression","interval","compared","unchanged",
    "acute","chronic","known","found","noted","seen",
    "year","years","month","months","week","weeks",
    "daily","twice","three","times","morning","evening",
    "right","left","bilateral","unilateral","prior","current",
    "new","old","recent","previous","reported","states","denies",
}

# Medical indicators that boost bigram/trigram candidacy
MEDICAL_INDICATORS = [
    "cancer","carcinoma","syndrome","disease","disorder","deficiency",
    "colitis","adenoma","polyp","bleeding","anemia","mutation","gene",
    "therapy","surgery","resection","colectomy","colonoscopy",
    "obstruction","perforation","fistula","hemorrhoid","dysplasia",
    "neoplasm","tumor","tumour","mass","lesion","metastasis",
    "ectomy","oscopy","itis","osis","emia","algia","rrhea","pathy",
]


def extract_candidates(notes_df):
    """
    Extract candidate medical phrases from all notes.
    Returns Counter of {term: frequency_in_notes}.
    """
    print("Extracting candidate terms from all notes...")
    candidates = Counter()

    for _, row in notes_df.iterrows():
        note  = str(row["Clean_note_text"])
        lower = note.lower()


        words = re.findall(r'\b[a-zA-Z]{3,}\b', lower)
        for w in words:
            if w not in STOPWORDS and len(w) >= 6:
                candidates[w] += 1

        for i in range(len(words)-1):
            w1, w2 = words[i], words[i+1]
            if (w1 not in STOPWORDS and w2 not in STOPWORDS
                    and len(w1) >= 3 and len(w2) >= 3):
                candidates[f"{w1} {w2}"] += 1

        for i in range(len(words)-2):
            phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
            if (any(ind in phrase for ind in MEDICAL_INDICATORS)
                    and words[i] not in STOPWORDS
                    and words[i+2] not in STOPWORDS):
                candidates[phrase] += 1

        for abbr in re.findall(r'\b[A-Z]{2,6}\b', note):
            if abbr.lower() not in STOPWORDS:
                candidates[abbr.lower()] += 1

        for hyph in re.findall(r'\b[a-z]{3,}-[a-z0-9]{2,}\b', lower):
            candidates[hyph] += 1

    candidates = Counter({t: f for t, f in candidates.items() if f >= 2})
    print(f"  Unique candidates (freq >= 2): {len(candidates)}")
    return candidates


def umls_search(term, api_key):
    """Search UMLS for a term. Returns best result dict or None."""
    url    = f"{BASE}/search/current"
    params = {
        "string":       term,
        "apiKey":       api_key,
        "returnIdType": "concept",
        "searchType":   "words" if len(term) > 8 else "exact",
        "pageSize":     3,
    }
    for attempt in range(2):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                results = r.json().get("result", {}).get("results", [])
                valid   = [x for x in results
                           if x.get("ui","") not in ("","NONE","none")]
                return valid[0] if valid else None
            elif r.status_code == 429:
                time.sleep(5)
            else:
                return None
        except Exception:
            time.sleep(2)
    return None


def get_semantic_types(cui, api_key):
    """Get semantic type codes for a UMLS CUI."""
    url    = f"{BASE}/content/current/CUI/{cui}"
    params = {"apiKey": api_key}
    try:
        r = requests.get(url, params=params, timeout=12)
        if r.status_code == 200:
            stys = r.json().get("result", {}).get("semanticTypes", [])
            return [s.get("uri","").split("/")[-1] for s in stys]
    except Exception:
        pass
    return []


def build_cache(candidates, api_key):
    """
    Query UMLS REST API for each candidate term.
    Classify into GROUP A (symptom CUI) or GROUP B (other clinical concept).
    Returns cache dict.
    """
    sorted_terms = sorted(candidates.items(),
                          key=lambda x: x[1], reverse=True)
    total  = len(sorted_terms)
    print(f"\nQuerying UMLS REST API for {total} candidates...")
    print(f"  Estimated time: ~{total*0.28/60:.0f} min")
    print(f"  (0.25s search + 0.10s type lookup per term)\n")

    cache = {}
    found = ga = gb = skipped = 0
    t0    = time.time()

    for i, (term, freq) in enumerate(sorted_terms):

        # Progress every 100 terms
        if i % 100 == 0 and i > 0:
            elapsed = (time.time() - t0) / 60
            eta     = (total - i) / max(i / max(elapsed, 0.01), 1)
            print(f"  [{i:>5}/{total}] found={found} A={ga} B={gb} "
                  f"skip={skipped} elapsed={elapsed:.1f}min ETA={eta:.1f}min")

        # Query UMLS
        result = umls_search(term, api_key)
        time.sleep(0.25)

        if result is None:
            skipped += 1
            continue

        cui  = result.get("ui", "")
        name = result.get("name", "")
        if not cui:
            skipped += 1
            continue

        # Get semantic types
        stys = get_semantic_types(cui, api_key)
        time.sleep(0.10)

        # Keep only clinical semantic types
        clinical = [s for s in stys if s in CLINICAL_STYS]
        if not clinical:
            skipped += 1
            continue

        # Classify
        is_group_a = cui in SYMPTOM_CUI_FAMILIES

        cache[term] = {
            "cui":          cui,
            "name":         name,
            "stys":         clinical,
            "is_group_a":   is_group_a,
            "freq_in_notes": freq,
        }
        found += 1
        if is_group_a:
            ga += 1
        else:
            gb += 1

        # Checkpoint every 500 found
        if found % 500 == 0:
            ckpt = CACHE_PATH + ".checkpoint"
            with open(ckpt, "w") as f:
                json.dump(cache, f, indent=2)
            print(f"  Checkpoint saved: {found} concepts → {ckpt}")

    # Final save
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"CACHE COMPLETE in {elapsed:.1f} min")
    print(f"  Total concepts: {found}")
    print(f"  GROUP A (symptoms):      {ga}")
    print(f"  GROUP B (other clinical): {gb}")
    print(f"  Skipped (not clinical):  {skipped}")
    print(f"  Saved: {CACHE_PATH}")

    # Print top GROUP B
    group_b = {t: d for t, d in cache.items()
               if not d.get("is_group_a", False)}
    print(f"\nTop 40 GROUP B concepts (most frequent in notes):")
    for t, d in sorted(group_b.items(),
                        key=lambda x: x[1]["freq_in_notes"], reverse=True)[:40]:
        print(f"  {t:<35} {d['name']:<35} freq={d['freq_in_notes']}")

    return cache


def main():
    print("="*60)
    print("BUILD UMLS CONCEPT CACHE FOR EXP6 RAG")
    print("="*60)

    # Test API first
    print("\nTesting UMLS API...")
    test = umls_search("colorectal cancer", UMLS_API_KEY)
    if test is None:
        print("ERROR: UMLS API failed. Check API key and internet connection.")
        sys.exit(1)
    print(f"  API OK: found '{test.get('name','')}' (CUI: {test.get('ui','')})")

    # Load notes
    print(f"\nLoading notes from {PATH}...")
    notes_df = pd.read_csv(PATH)
    notes_df = notes_df[
        notes_df["Clean_note_text"].str.strip().ne("")
    ].reset_index(drop=True)
    print(f"  Notes: {len(notes_df)}")

    # Extract candidates
    candidates = extract_candidates(notes_df)

    # Build cache via UMLS API
    cache = build_cache(candidates, UMLS_API_KEY)

    print("\nDone. Run Exp6.py to use this cache.")
    print(f"  $PYTHON Exp6.py --phase all")


if __name__ == "__main__":
    main()
