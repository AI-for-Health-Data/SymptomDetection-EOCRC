from __future__ import annotations
import argparse, os, re
import pandas as pd

SYMPTOMS = [
    "Abdominal pain", "Rectal bleeding", "Rectal pain", "Diarrhea",
    "Constipation", "Weight loss", "Family history of colorectal cancer"
]

ADJ_SYS = (
    "You are a careful clinical adjudicator. A prior strict verification step "
    "rejected a symptom prediction for this patient. Your job is to catch REAL "
    "but tersely-worded clinical findings that the strict check missed.\n\n"
    "You will be given a CLINICAL NOTE and a SYMPTOM CLAIM. Accept common "
    "clinical shorthand and indirect phrasing.\n\n"
    "Rules:\n"
    " - REINSTATE only if the NOTE contains concrete, specific evidence that "
    "the symptom is present, even if abbreviated or indirect.\n"
    " - A denial, negation, or absence of the symptom means REJECT.\n"
    " - If the note gives no evidence for the symptom, REJECT.\n"
    " - For 'family history of colorectal cancer': REINSTATE only if the note "
    "names a FAMILY MEMBER (or 'family history') linked to colorectal/colon/"
    "rectal/bowel cancer. A personal history, a different cancer, or "
    "unspecified cancer type is NOT family history of CRC.\n"
    " - Never use external knowledge; judge only from the note.\n\n"
    "Answer on one line with exactly one word: REINSTATE or REJECT."
)

# One-shot examples per symptom category
ONESHOT_EXAMPLES = {
    "Family history of colorectal cancer": (
        "NOTE:\nPMH notable. FH sig for CA colon in mother.\n\n"
        "SYMPTOM: Family history of colorectal cancer.\n"
        "One word:",
        "REINSTATE"
    ),
    "_somatic": (  # used for all 6 somatic symptoms
        "NOTE:\nPt c/o abd pn x 3 days, intermittent cramping.\n\n"
        "SYMPTOM: Abdominal pain.\n"
        "One word:",
        "REINSTATE"
    ),
}


def build_adj_prompt(note_text: str, symptom: str) -> str:
    return (
        f"NOTE:\n{note_text[:6000]}\n\n"
        f"SYMPTOM: {symptom}.\n"
        f"One word:"
    )


def get_oneshot(symptom: str):
    if symptom in ONESHOT_EXAMPLES:
        return ONESHOT_EXAMPLES[symptom]
    return ONESHOT_EXAMPLES["_somatic"]

_pipe = None

def get_pipe(model_id: str, cache_dir: str):
    global _pipe
    if _pipe is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=cache_dir,
            torch_dtype=torch.bfloat16, device_map="auto")
        _pipe = pipeline("text-generation", model=mdl, tokenizer=tok,
                         max_new_tokens=8, do_sample=False)
    return _pipe


def parse_adj(text: str):
    """Return (reinstate: bool, parsed_ok: bool)."""
    t = text.strip().upper()
    for line in t.splitlines():
        line = line.strip()
        if not line: continue
        if line.startswith("REINSTATE"): return True, True
        if line.startswith("REJECT"):    return False, True
        break
    if re.search(r'\bREINSTATE\b', t): return True, True
    if re.search(r'\bREJECT\b', t):    return False, True
    return False, False  # unparseable -> conservative: do NOT reinstate


DRY_RUN_PATTERNS = {
    "Family history of colorectal cancer": re.compile(
        r'(family|fh|fhx|father|mother|dad|mom|sister|brother|sibling|parent|'
        r'aunt|uncle|grandmother|grandfather|maternal|paternal)\b'
        r'[^.\n]{0,50}\b(colorectal|colon|rectal|bowel)\b'
        r'[^.\n]{0,20}\b(ca\b|cancer|carcinoma|malignancy)', re.I),
    "Abdominal pain": re.compile(
        r'(abd|abdominal|stomach|belly|epigastric).{0,15}(pain|tender|cramp|discomfort|ache)', re.I),
    "Rectal bleeding": re.compile(
        r'(rectal|rectum).{0,10}(bleed|blood|hemorrh|hematochezia|brbpr)', re.I),
    "Rectal pain": re.compile(
        r'(rectal|anal|anorectal|perianal).{0,10}(pain|discomfort|tender|proctalgia)', re.I),
    "Diarrhea": re.compile(
        r'(diarr|loose stool|watery stool|frequent bm|frequent bowel)', re.I),
    "Constipation": re.compile(
        r'(constipat|hard stool|no bm|obstipat|straining)', re.I),
    "Weight loss": re.compile(
        r'(weight loss|wt loss|lost weight|losing weight|cachex)', re.I),
}


def adjudicate_one(note_text: str, symptom: str,
                   model_id: str, cache_dir: str, dry_run: bool):
    """Adjudicate one (note, symptom) rejected prediction."""
    if dry_run:
        pat = DRY_RUN_PATTERNS.get(symptom)
        if pat:
            return bool(pat.search(note_text or "")), True
        return False, True

    pipe = get_pipe(model_id, cache_dir)
    oneshot_u, oneshot_a = get_oneshot(symptom)
    msgs = [
        {"role": "system",    "content": ADJ_SYS},
        {"role": "user",      "content": oneshot_u},
        {"role": "assistant", "content": oneshot_a},
        {"role": "user",      "content": build_adj_prompt(note_text, symptom)},
    ]
    out = pipe(msgs)[0]["generated_text"]
    text = out[-1]["content"] if isinstance(out, list) else str(out)
    reinstate, ok = parse_adj(text)
    return reinstate, ok


def main():
    ap = argparse.ArgumentParser(
        description="Agent 3: recall-oriented adjudicator (all 7 symptoms)")
    ap.add_argument("--verified", required=True,
                    help="Output from verify_agent.py")
    ap.add_argument("--pred",     required=True,
                    help="Raw extractor output CSV (for note text)")
    ap.add_argument("--note_col", default="Clean_note_text")
    ap.add_argument("--id_col",   default="NOTE_ID")
    ap.add_argument("--pred_id_col", default=None,
                    help="ID column in --pred (default: --id_col)")
    ap.add_argument("--out",      required=True)
    ap.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--cache_dir",default="/lustre/smuexa01/client/users/nikkieh/hf_cache")
    ap.add_argument("--dry_run",  action="store_true")
    a = ap.parse_args()

    v    = pd.read_csv(a.verified)
    pred = pd.read_csv(a.pred)
    pid  = a.pred_id_col or a.id_col
    notes = dict(zip(pred[pid].astype(str), pred[a.note_col].astype(str)))

    if "verifier_keep" not in v.columns:
        raise SystemExit("verified file lacks 'verifier_keep' column")
    v["verifier_keep"] = v["verifier_keep"].astype(str).str.lower().isin(
        {"true", "1", "yes"})

    print("="*60)
    print("AGENT 3: RECALL-ORIENTED ADJUDICATOR")
    print(f"  Input       : {a.verified}")
    print(f"  Total rows  : {len(v)}")
    print(f"  Kept by verifier: {int(v['verifier_keep'].sum())}")
    print(f"  Rejected (to adjudicate): {int((~v['verifier_keep']).sum())}")
    print(f"  Mode        : {'DRY RUN' if a.dry_run else a.model_id}")
    print(f"  Rounds      : 1 (single pass, no iteration)")
    print("="*60)

    reinstated_counts = {s: 0 for s in SYMPTOMS}
    examined_counts   = {s: 0 for s in SYMPTOMS}
    unparsed = 0
    adj_keep = []

    for _, r in v.iterrows():
        if r["verifier_keep"]:
            adj_keep.append(False)  # already kept; adjudicator not invoked
            continue

        symptom = r["symptom"]
        examined_counts[symptom] = examined_counts.get(symptom, 0) + 1
        note = notes.get(str(r["note_id"]), "")

        reinstate, ok = adjudicate_one(
            note, symptom, a.model_id, a.cache_dir, a.dry_run)
        if not ok:
            unparsed += 1
        if reinstate:
            reinstated_counts[symptom] = reinstated_counts.get(symptom, 0) + 1
        adj_keep.append(bool(reinstate))

    v["adjudicator_keep"] = adj_keep
    v["final_keep"]       = v["verifier_keep"] | v["adjudicator_keep"]
    v["label_after"]      = v["final_keep"].map(lambda k: "yes" if k else "no")
    v.to_csv(a.out, index=False)

    total_reinstated = sum(reinstated_counts.values())
    total_examined   = sum(examined_counts.values())

    print("\n" + "="*60)
    print("ADJUDICATOR SUMMARY")
    print(f"  {'Symptom':<42} {'Examined':>8}  {'Reinstated':>10}")
    for s in SYMPTOMS:
        e = examined_counts.get(s, 0)
        r = reinstated_counts.get(s, 0)
        print(f"  {s:<42} {e:>8}  {r:>10}")
    print(f"  {'TOTAL':<42} {total_examined:>8}  {total_reinstated:>10}")
    if unparsed:
        print(f"  Unparseable (kept rejected): {unparsed}")
    print(f"\n  Final kept: {int(v['final_keep'].sum())} "
          f"(verifier {int(v['verifier_keep'].sum())} + "
          f"adjudicator {total_reinstated})")
    print("="*60)
    print(f"Wrote {a.out}")
    print(f"\nNext: python apply_verifier.py "
          f"--metrics <exp7_note_level_metrics.csv> "
          f"--verified {a.out} --keep_col final_keep --out final_metrics.csv")


if __name__ == "__main__":
    main()
