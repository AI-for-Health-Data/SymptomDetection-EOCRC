from __future__ import annotations
import argparse, ast, json, os, re, math
from collections import Counter
import pandas as pd

def bleu_no_bp(hypothesis: str, reference: str, max_n: int = 4) -> float:
    _tok = re.compile(r"\w+|\S")
    hyp_tokens = _tok.findall(hypothesis.lower())
    ref_tokens = _tok.findall(reference.lower())
    if not hyp_tokens:
        return 0.0
    precisions = []
    for n in range(1, min(max_n, len(hyp_tokens)) + 1):
        hyp_ng = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1))
        ref_ng = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1))
        clipped = sum(min(c, ref_ng[ng]) for ng, c in hyp_ng.items())
        total = sum(hyp_ng.values())
        if total == 0 or clipped == 0:
            return 0.0
        precisions.append(clipped / total)
    return math.exp(sum(math.log(p) for p in precisions) / len(precisions))

GRADE_HIGH     = 0.66
GRADE_MODERATE = 0.33

def grade_tier(bleu: float) -> str:
    if bleu >= GRADE_HIGH:     return "HIGH"
    if bleu >= GRADE_MODERATE: return "MODERATE"
    return "LOW"

SYMPTOMS = [
    "Abdominal pain", "Rectal bleeding", "Rectal pain", "Diarrhea",
    "Constipation", "Weight loss", "Family history of colorectal cancer"
]

VERIFY_SYS = (
    "You are verifying whether a clinical note contains evidence that a "
    "patient has a specific symptom. Clinical notes use abbreviations, "
    "shorthand, and indirect language. Accept ALL of the following as "
    "valid positive evidence:\n"
    "- Direct statements: \"patient has abdominal pain\"\n"
    "- Abbreviations: \"abd pn\", \"BRBPR\", \"c/o wt loss\", \"FH CA colon\"\n"
    "- Indirect findings: \"guaiac positive\" = rectal bleeding, "
    "\"tenderness in RLQ\" = abdominal pain\n"
    "- Terse mentions: \"FH sig for CA colon\" = family history of CRC\n"
    "- Exam or lab results that confirm the symptom\n\n"
    "Respond on a SINGLE LINE with EXACTLY ONE WORD:\n"
    "  SUPPORTED     - the note contains ANY evidence (direct, abbreviated, "
    "or indirect) that this symptom is present\n"
    "  CONTRADICTED  - the note explicitly denies or negates this symptom\n"
    "  ABSENT        - the note contains NO mention of this symptom at all, "
    "not even indirectly\n\n"
    "For 'family history of colorectal cancer':\n"
    "  SUPPORTED requires a FAMILY MEMBER (parent, sibling, relative) linked "
    "to colorectal/colon/rectal/bowel cancer.\n"
    "  A personal history, a different cancer type, or unspecified cancer "
    "is ABSENT for this claim.\n\n"
    "Use ONLY the note. No outside knowledge. Output only the one word."
)

ONESHOT_U = (
    "CLINICAL NOTE:\nPt's father was diagnosed with colon cancer at age 55. "
    "No abdominal complaints today.\n\n"
    "CLAIM: The patient has a family history of colorectal cancer.\n\n"
    "One word:"
)
ONESHOT_A = "SUPPORTED"


def build_prompt(note_text: str, symptom: str) -> str:
    claim = (
        "The patient has a family history of colorectal cancer."
        if symptom.startswith("Family history")
        else f"The patient has/experiences: {symptom}."
    )
    return f"CLINICAL NOTE:\n{note_text[:6000]}\n\nCLAIM: {claim}\n\nOne word:"


_PIPE = None

def get_pipe(model_id: str, cache_dir: str):
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir,
        torch_dtype=torch.bfloat16, device_map="auto")
    _PIPE = pipeline("text-generation", model=mdl, tokenizer=tok,
                     max_new_tokens=120, do_sample=False)
    return _PIPE


def parse_verdict(text: str):
    """Return (verdict, parsed_ok). Maps all label variants to standard forms."""
    t = text.strip().upper()
    LABEL_MAP = {
        # New clinical labels
        "SUPPORTED": "supported",
        "CONTRADICTED": "contradicted",
        "ABSENT": "absent",
        # Old NLI labels (backward compatibility)
        "ENTAILMENT": "supported",
        "ENTAIL": "supported",
        "CONTRADICTION": "contradicted",
        "CONTRADICT": "contradicted",
        "NEUTRAL": "absent",
    }
    for line in t.splitlines():
        line = line.strip()
        if not line: continue
        for keyword, label in LABEL_MAP.items():
            if line.startswith(keyword):
                return label, True
        break
    for keyword, label in [("CONTRADICTED","contradicted"),
                            ("CONTRADICTION","contradicted"),
                            ("SUPPORTED","supported"),
                            ("ENTAILMENT","supported"),
                            ("CONTRADICT","contradicted"),
                            ("ENTAIL","supported"),
                            ("ABSENT","absent"),
                            ("NEUTRAL","absent")]:
        if re.search(rf'\b{keyword}\b', t):
            return label, True
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            v = str(d.get("verdict","")).strip().upper()
            for keyword, label in LABEL_MAP.items():
                if v == keyword:
                    return label, True
        except Exception: pass
    return "absent", False  # unparseable -> flagged


def verify_one(note_text: str, symptom: str, model_id: str,
               cache_dir: str, dry_run: bool):
    if dry_run:
        nt = note_text.lower()
        if symptom.startswith("Family history"):
            ok = bool(re.search(
                r'(family|father|mother|sibling|brother|sister|parent|fh|fhx)'
                r'.{0,40}(colorectal|colon|rectal|bowel|ca colon|crc)'
                r'.{0,20}(cancer|ca\b|carcinoma|)', nt))
        else:
            ok = any(w in nt for w in symptom.lower().split())
        return ("supported" if ok else "absent"), True, "DRY_RUN"

    pipe = get_pipe(model_id, cache_dir)
    msgs = [
        {"role": "system",    "content": VERIFY_SYS},
        {"role": "user",      "content": ONESHOT_U},
        {"role": "assistant", "content": ONESHOT_A},
        {"role": "user",      "content": build_prompt(note_text, symptom)},
    ]
    out = pipe(msgs)[0]["generated_text"]
    text = out[-1]["content"] if isinstance(out, list) else str(out)
    verdict, ok = parse_verdict(text)
    if not ok:
        msgs2 = msgs + [
            {"role": "assistant", "content": text},
            {"role": "user", "content":
             "Answer with ONE WORD only: SUPPORTED, CONTRADICTED, or ABSENT."},
        ]
        out2 = pipe(msgs2)[0]["generated_text"]
        text2 = out2[-1]["content"] if isinstance(out2, list) else str(out2)
        v2, ok2 = parse_verdict(text2)
        if ok2:
            return v2, True, (text + " || " + text2)[:300]
    return verdict, ok, text[:300]


def get_pred_dict(cell):
    if isinstance(cell, dict): return cell
    if isinstance(cell, str):
        try: return ast.literal_eval(cell)
        except Exception:
            try: return json.loads(cell)
            except Exception: return {}
    return {}

def norm(ans: str) -> str:
    a = str(ans).strip().lower()
    if a.startswith("y"): return "yes"
    if a.startswith("n"): return "no"
    return a

def get_inference_span(pred_dict: dict, symptom: str) -> str:
    for key in (f"{symptom} inference", f"{symptom}_inference",
                f"{symptom} Inference", "inference"):
        val = pred_dict.get(key, "")
        if val and str(val).strip().lower() not in ("", "n/a", "none", "null"):
            return str(val).strip()
    return ""


def main():
    ap = argparse.ArgumentParser(
        description="Agent 2: per-(note, symptom) verifier with GRADE gate")
    ap.add_argument("--pred",     required=True)
    ap.add_argument("--note_col", default="Clean_note_text")
    ap.add_argument("--pred_col", default="exp_output_dict")
    ap.add_argument("--id_col",   default="NOTE_ID")
    ap.add_argument("--out",      default="verify_output.csv")
    ap.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--cache_dir",default="/lustre/smuexa01/client/users/nikkieh/hf_cache")
    ap.add_argument("--symptoms_only", default=None)
    ap.add_argument("--reject_on", default="absent,contradicted",
                    help="Verdicts that trigger rejection (default: absent,contradicted)")
    ap.add_argument("--grade_reject_tier", default="LOW",
                    choices=["LOW","MODERATE","HIGH","NONE"])
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.pred)
    if a.note_col not in df.columns:
        for c in ("Clean_note_text","note_text","NOTE_TEXT","text","note"):
            if c in df.columns: a.note_col = c; break
    if a.note_col not in df.columns:
        raise SystemExit(f"[ERROR] note column not found; have {list(df.columns)[:12]}")
    if a.pred_col not in df.columns:
        raise SystemExit(f"[ERROR] pred column '{a.pred_col}' not found")

    targets = ([s.strip() for s in a.symptoms_only.split(",")]
               if a.symptoms_only else SYMPTOMS)
    nli_reject = set(v.strip().lower() for v in a.reject_on.split(","))

    grade_tiers_list = ["LOW", "MODERATE", "HIGH"]
    if a.grade_reject_tier == "NONE":
        grade_reject_set = set()
    else:
        cutoff = grade_tiers_list.index(a.grade_reject_tier)
        grade_reject_set = set(grade_tiers_list[:cutoff+1])

    print("="*60)
    print("AGENT 2: NOTE-ONLY VERIFIER (clinically calibrated)")
    print(f"  Labels         : SUPPORTED / CONTRADICTED / ABSENT")
    print(f"  Symptoms       : {targets}")
    print(f"  Reject on      : {nli_reject}")
    print(f"  GRADE reject   : {grade_reject_set if grade_reject_set else 'disabled'}")
    print(f"  Mode           : {'DRY RUN' if a.dry_run else a.model_id}")
    print(f"  Operation      : per-note per-symptom")
    print(f"  Rounds         : 1 (single pass)")
    print("="*60)

    ckpt = a.out + ".ckpt"
    done, rows = set(), []
    if os.path.exists(ckpt):
        prev = pd.read_csv(ckpt)
        rows = prev.to_dict("records")
        done = set((str(r["note_id"]), r["symptom"]) for r in rows)
        print(f"[resume] {len(rows)} prior verdicts loaded")

    flipped        = {s: 0 for s in targets}
    checked        = {s: 0 for s in targets}
    grade_rejected = {s: 0 for s in targets}

    for r in rows:
        s = r.get("symptom","")
        if s in checked:
            checked[s] += 1
            if not r.get("verifier_keep", True): flipped[s] += 1
            if r.get("grade_rejected", False):   grade_rejected[s] += 1

    for idx, row in df.iterrows():
        note      = str(row[a.note_col])
        pred_dict = get_pred_dict(row[a.pred_col])
        nid       = str(row.get(a.id_col, idx))

        for symptom in targets:
            if norm(pred_dict.get(symptom, "")) != "yes":
                continue
            if (nid, symptom) in done:
                continue

            checked[symptom] += 1

            # Gate 1: Clinical evidence check
            verdict, parsed_ok, raw = verify_one(
                note, symptom, a.model_id, a.cache_dir, a.dry_run)

            # Gate 2: GRADE grounding tier
            inference_span = get_inference_span(pred_dict, symptom)
            bleu  = bleu_no_bp(inference_span, note) if inference_span else 0.0
            tier  = grade_tier(bleu)

            # Combined decision
            nli_rejects   = parsed_ok and (verdict in nli_reject)
            grade_rejects = (parsed_ok and verdict == "supported"
                             and tier in grade_reject_set)
            keep = not (nli_rejects or grade_rejects)
            if not parsed_ok:
                keep = True

            if not keep:      flipped[symptom] += 1
            if grade_rejects: grade_rejected[symptom] += 1

            rows.append(dict(
                note_id=nid, symptom=symptom, verdict=verdict,
                parsed_ok=parsed_ok, grade_tier=tier,
                grade_bleu=round(bleu, 4),
                nli_rejected=nli_rejects, grade_rejected=grade_rejects,
                verifier_keep=keep, label_before="yes",
                label_after="yes" if keep else "no",
                raw_output=raw,
            ))

        if (idx+1) % 50 == 0:
            pd.DataFrame(rows).to_csv(ckpt, index=False)
            flip_str = ", ".join(f"{s.split()[0]}={flipped[s]}" for s in targets)
            print(f"  [{idx+1}/{len(df)}] flips: {flip_str}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)
    if os.path.exists(ckpt): os.remove(ckpt)

    print("\n" + "="*60)
    print("VERIFIER SUMMARY")
    print(f"  {'Symptom':<42} {'Total':>8}  {'NLI':>6}  {'GRADE':>6}")
    for s in targets:
        c = checked[s]; f = flipped[s]; g = grade_rejected[s]
        rate = f"{f/c:.1%}" if c else "n/a"
        print(f"  {s:<42} {f:>4}/{c:<4} ({rate})  nli={f-g}  grade={g}")
    if "verdict" in res.columns:
        print("\nVerdict distribution:")
        print(res["verdict"].value_counts(dropna=False).to_string())
    if "grade_tier" in res.columns:
        print("\nGRADE tier (all verified):")
        print(res["grade_tier"].value_counts(dropna=False).to_string())
        kept = res[res["verifier_keep"]==True]
        print(f"\nGRADE tier (kept only, n={len(kept)}):")
        print(kept["grade_tier"].value_counts(dropna=False).to_string())
    n_fail = int((~res["parsed_ok"].astype(bool)).sum()) if "parsed_ok" in res.columns else 0
    if n_fail: print(f"\nParse failures (kept, flagged): {n_fail}/{len(res)}")
    print("="*60)
    print(f"Wrote {a.out}")
    print(f"\nNext: python adjudicate_agent.py --verified {a.out} "
          f"--pred <exp7_outputs_raw.csv> --out adjudicated.csv")


if __name__ == "__main__":
    main()
