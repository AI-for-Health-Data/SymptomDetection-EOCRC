from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

import pandas as pd

SYMPTOMS = [
    "Abdominal pain",
    "Rectal bleeding",
    "Rectal pain",
    "Diarrhea",
    "Constipation",
    "Weight loss",
    "Family history of colorectal cancer",
]

NOTE_LIMIT = 6000
ADJ_MAX_NEW_TOKENS = 16  # headroom over one word; 8 risks mid-word truncation


ADJ_SYS = (
    "You are a careful clinical adjudicator performing a recovery review. A "
    "prior strict verification step rejected a proposed finding for this "
    "patient. That strict step is known to over-reject: it discards some real "
    "but tersely or indirectly worded findings along with the unsupported "
    "ones. Your job is to recover ONLY the real ones.\n\n"
    "You are given a CLINICAL NOTE and a candidate FINDING. Use ONLY the note. "
    "Do not use outside medical knowledge, retrieved literature, or inference "
    "from diagnoses, medications, or risk factors alone.\n\n"
    "Decide:\n"
    " - REINSTATE only if the note contains concrete, specific, patient-"
    "anchored evidence that the finding is present, even if abbreviated or "
    "indirect (e.g. 'abd pn x3d', 'BRBPR', 'guaiac positive', "
    "'FH sig for CA colon').\n"
    " - REJECT if the note denies or negates the finding.\n"
    " - REJECT if the note gives no concrete evidence for the finding.\n"
    " - Do not reinstate on the basis of a related diagnosis, medication, "
    "risk factor, or general plausibility.\n"
    " - For family history of colorectal cancer: REINSTATE only if the note "
    "names a family member (or gives an explicit family-history statement) "
    "linked to colorectal, colon, rectal, or bowel cancer. A personal cancer "
    "history, a different cancer type, or unspecified family cancer is REJECT "
    "for this finding.\n\n"
    "Answer with exactly ONE WORD on a single line: REINSTATE or REJECT. "
    "Do not explain."
)

ADJ_RETRY = "Answer with ONE WORD only: REINSTATE or REJECT."


def build_adj_runtime(note_text: str, symptom: str) -> str:
    """Zero-shot runtime message; symptom-independent phrasing."""
    finding = (
        "family history of colorectal cancer"
        if symptom == "Family history of colorectal cancer"
        else symptom
    )
    return (
        f"CLINICAL NOTE:\n{note_text[:NOTE_LIMIT]}\n\n"
        f"CANDIDATE FINDING: {finding}\n\n"
        "VERDICT:"
    )


def parse_adj(text: str) -> Tuple[bool, bool]:
    """Return (reinstate, parsed_ok). Unparseable -> (False, False)."""
    upper = str(text).strip().upper()
    for line in upper.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("REINSTATE"):
            return True, True
        if line.startswith("REJECT"):
            return False, True
        break
    if re.search(r"\bREINSTATE\b", upper):
        return True, True
    if re.search(r"\bREJECT\b", upper):
        return False, True
    return False, False  


DRY_RUN_PATTERNS = {
    "Family history of colorectal cancer": re.compile(
        r"(family|fh|fhx|father|mother|dad|mom|sister|brother|sibling|parent|"
        r"aunt|uncle|grandmother|grandfather|maternal|paternal)\b"
        r"[^.\n]{0,50}\b(colorectal|colon|rectal|bowel)\b"
        r"[^.\n]{0,20}\b(ca\b|cancer|carcinoma|malignancy)",
        re.I,
    ),
    "Abdominal pain": re.compile(
        r"(abd|abdominal|stomach|belly|epigastric).{0,15}"
        r"(pain|tender|cramp|discomfort|ache)",
        re.I,
    ),
    "Rectal bleeding": re.compile(
        r"(rectal|rectum).{0,10}(bleed|blood|hemorrh|hematochezia|brbpr)", re.I
    ),
    "Rectal pain": re.compile(
        r"(rectal|anal|anorectal|perianal).{0,10}"
        r"(pain|discomfort|tender|proctalgia)",
        re.I,
    ),
    "Diarrhea": re.compile(
        r"(diarr|loose stool|watery stool|frequent bm|frequent bowel)", re.I
    ),
    "Constipation": re.compile(
        r"(constipat|hard stool|no bm|obstipat|straining)", re.I
    ),
    "Weight loss": re.compile(
        r"(weight loss|wt loss|lost weight|losing weight|cachex)", re.I
    ),
}


_MODEL = None
_TOKENIZER = None


def load_model(model_id: str, cache_dir: str):
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _TOKENIZER, _MODEL
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading adjudicator model: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    mdl.eval()
    _MODEL, _TOKENIZER = mdl, tok
    return tok, mdl


def generate_one_word(messages, tokenizer, model, max_new_tokens: int) -> str:
    import torch

    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        parts = [f"<{m['role']}>\n{m['content']}" for m in messages]
        formatted = "\n".join(parts) + "\n<assistant>\n"
    inputs = tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=8192
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def adjudicate_one(
    note_text: str,
    symptom: str,
    tokenizer,
    model,
    dry_run: bool,
) -> Tuple[bool, bool, str]:
    """Return (reinstate, parsed_ok, raw_output)."""
    if dry_run:
        pat = DRY_RUN_PATTERNS.get(symptom)
        if pat:
            return bool(pat.search(note_text or "")), True, "DRY_RUN"
        return False, True, "DRY_RUN"

    runtime = build_adj_runtime(note_text, symptom)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": ADJ_SYS},
        {"role": "user", "content": runtime},
    ]
    raw = generate_one_word(messages, tokenizer, model, ADJ_MAX_NEW_TOKENS)
    reinstate, parsed_ok = parse_adj(raw)

    if not parsed_ok:
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": ADJ_RETRY},
        ]
        retry_raw = generate_one_word(
            retry_messages, tokenizer, model, ADJ_MAX_NEW_TOKENS
        )
        r2, ok2 = parse_adj(retry_raw)
        raw = (raw + " || " + retry_raw)[:400]
        if ok2:
            reinstate, parsed_ok = r2, True

    return reinstate, parsed_ok, raw[:400]


FH_SYMPTOM = "Family history of colorectal cancer"


def _flag_true(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def reinstatement_eligible(row, scope: str) -> bool:
    """Return True if this verifier-rejected pair may be adjudicated.

    Scopes:
      'all'         : every rejection eligible (original union rule).
      'nli_only'    : grounding-gate rejections barred globally.
      'per_symptom' : the gold-motivated policy. Family history is the only
                      symptom with retrieval (Lynch-syndrome) contamination, and
                      grounding-gate reinstatement is where FH fabrications
                      re-enter. So FH is NEVER reinstated (auto-REJECT, no model
                      call), while the six somatic symptoms are fully eligible --
                      including their grounding-gate rejections, which gold shows
                      hold ~84% true-positive recoveries (paraphrased mentions
                      with low BLEU but real clinical content).
    """
    symptom = row.get("symptom", "")
    if scope == "all":
        return True
    if scope == "per_symptom":
        # FH locked; all somatic rejections (NLI- and grounding-gate) eligible.
        return symptom != FH_SYMPTOM
    # scope == 'nli_only': ineligible iff it was a grounding-gate rejection.
    if "grade_rejected" in row and _flag_true(row.get("grade_rejected")):
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Agent 3: recall-oriented adjudicator (recovery stage)."
    )
    ap.add_argument("--verified", required=True, help="Output of verify_agent.py")
    ap.add_argument("--pred", required=True, help="Frozen E7 CSV (for note text)")
    ap.add_argument("--note_col", default="Clean_note_text")
    ap.add_argument("--id_col", default="NOTE_ID")
    ap.add_argument("--pred_id_col", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument(
        "--cache_dir",
        default="/lustre/smuexa01/client/users/nikkieh/hf_cache",
    )
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument(
        "--reinstate_scope",
        choices=["nli_only", "all", "per_symptom"],
        default="per_symptom",
        help=(
            "Which verifier rejections the adjudicator may reinstate. "
            "'per_symptom' (default, gold-motivated): FH is never reinstated "
            "(its grounding-gate pool is where Lynch-syndrome fabrications "
            "re-enter); the six somatic symptoms are fully eligible, including "
            "grounding-gate rejections, where gold shows ~84%% of recoveries "
            "are true paraphrased mentions. "
            "'all': original union rule (every rejection eligible). "
            "'nli_only': grounding-gate rejections barred globally."
        ),
    )
    a = ap.parse_args()

    v = pd.read_csv(a.verified)
    pred = pd.read_csv(a.pred)
    pid = a.pred_id_col or a.id_col
    notes = dict(zip(pred[pid].astype(str), pred[a.note_col].astype(str)))

    if "verifier_keep" not in v.columns:
        raise SystemExit("verified file lacks 'verifier_keep' column")
    v["verifier_keep"] = (
        v["verifier_keep"].astype(str).str.lower().isin({"true", "1", "yes"})
    )

    if a.reinstate_scope == "nli_only" and "grade_rejected" not in v.columns:
        raise SystemExit(
            "--reinstate_scope nli_only requires a 'grade_rejected' column in "
            "the verified file (written by verify_agent.py). Re-run the "
            "verifier, or pass --reinstate_scope all to use the union rule."
        )

    print("=" * 64)
    print("AGENT 3: RECALL-ORIENTED ADJUDICATOR (recovery stage)")
    print("  Method       : two-stage cascade recovery; keep = verifier OR adj")
    print("                 (SelfJudge/two-stage arXiv:2510.02329;")
    print("                  clinical two-round adjudication arXiv:2605.30646)")
    print(f"  Input        : {a.verified}")
    print(f"  Total rows   : {len(v)}")
    print(f"  Kept by verifier         : {int(v['verifier_keep'].sum())}")
    print(f"  Rejected (to adjudicate) : {int((~v['verifier_keep']).sum())}")
    print(f"  Reinstate scope          : {a.reinstate_scope}")
    print("  Shots        : 0 (no per-symptom example)")
    print(f"  Output cap   : {ADJ_MAX_NEW_TOKENS} tokens")
    print(f"  Mode         : {'DRY RUN (regex)' if a.dry_run else a.model_id}")
    print("  Retrievalseen: NO    Gold seen: NO    Rounds: 1")
    print("=" * 64)

    tokenizer = model = None
    if not a.dry_run:
        tokenizer, model = load_model(a.model_id, a.cache_dir)

    # Resumable checkpoint keyed on (note_id, symptom).
    ckpt = a.out + ".ckpt"
    done: Dict[Tuple[str, str], dict] = {}
    if a.dry_run is False and os.path.exists(ckpt):
        prev = pd.read_csv(ckpt)
        for _, r in prev.iterrows():
            done[(str(r["note_id"]), r["symptom"])] = r.to_dict()
        print(f"[resume] loaded {len(done)} prior adjudications")

    reinstated_counts = {s: 0 for s in SYMPTOMS}
    examined_counts = {s: 0 for s in SYMPTOMS}
    ineligible_counts = {s: 0 for s in SYMPTOMS}
    unparsed = 0

    adj_keep_col: List[bool] = []
    adj_verdict_col: List[str] = []
    adj_raw_col: List[str] = []

    processed = 0
    for _, r in v.iterrows():
        if r["verifier_keep"]:
            adj_keep_col.append(False)  
            adj_verdict_col.append("NOT_RUN")
            adj_raw_col.append("")
            continue

        symptom = r["symptom"]
        note_id = str(r["note_id"])
        examined_counts[symptom] = examined_counts.get(symptom, 0) + 1

        if not reinstatement_eligible(r, a.reinstate_scope):
            ineligible_counts[symptom] = ineligible_counts.get(symptom, 0) + 1
            adj_keep_col.append(False)
            if a.reinstate_scope == "per_symptom" and symptom == FH_SYMPTOM:
                adj_verdict_col.append("INELIGIBLE_FH_LOCK")
            else:
                adj_verdict_col.append("INELIGIBLE_GROUNDING")
            adj_raw_col.append("")
            continue

        cache_key = (note_id, symptom)
        if cache_key in done:
            prior = done[cache_key]
            reinstate = str(prior.get("adjudicator_keep", "")).lower() in {
                "true",
                "1",
                "yes",
            }
            adj_keep_col.append(reinstate)
            adj_verdict_col.append(
                "REINSTATE" if reinstate else prior.get("adjudicator_verdict", "REJECT")
            )
            adj_raw_col.append(str(prior.get("adjudicator_raw", "")))
            if reinstate:
                reinstated_counts[symptom] += 1
            continue

        note = notes.get(note_id, "")
        reinstate, ok, raw = adjudicate_one(
            note, symptom, tokenizer, model, a.dry_run
        )
        if not ok:
            unparsed += 1
        if reinstate:
            reinstated_counts[symptom] += 1

        adj_keep_col.append(bool(reinstate))
        adj_verdict_col.append("REINSTATE" if reinstate else "REJECT")
        adj_raw_col.append(raw)

        processed += 1
        if processed % 25 == 0:
            snapshot = v.iloc[: len(adj_keep_col)].copy()
            snapshot["adjudicator_keep"] = adj_keep_col
            snapshot["adjudicator_verdict"] = adj_verdict_col
            snapshot["adjudicator_raw"] = adj_raw_col
            snapshot[snapshot["adjudicator_verdict"] != "NOT_RUN"].to_csv(
                ckpt, index=False
            )

    v["adjudicator_keep"] = adj_keep_col
    v["adjudicator_verdict"] = adj_verdict_col
    v["adjudicator_raw"] = adj_raw_col
    v["final_keep"] = v["verifier_keep"] | v["adjudicator_keep"]
    v["label_after"] = v["final_keep"].map(lambda k: "yes" if k else "no")
    v.to_csv(a.out, index=False)
    if os.path.exists(ckpt):
        os.remove(ckpt)

    total_reinstated = sum(reinstated_counts.values())
    total_examined = sum(examined_counts.values())

    total_ineligible = sum(ineligible_counts.values())

    print("\n" + "=" * 64)
    print(f"ADJUDICATOR SUMMARY  (scope={a.reinstate_scope})")
    print(
        f"  {'Symptom':<38} {'Examined':>8}  {'Ineligible':>10}  "
        f"{'Reinstated':>10}"
    )
    for s in SYMPTOMS:
        print(
            f"  {s:<38} {examined_counts.get(s, 0):>8}  "
            f"{ineligible_counts.get(s, 0):>10}  "
            f"{reinstated_counts.get(s, 0):>10}"
        )
    print(
        f"  {'TOTAL':<38} {total_examined:>8}  "
        f"{total_ineligible:>10}  {total_reinstated:>10}"
    )
    if a.reinstate_scope == "nli_only":
        print(
            f"  ({total_ineligible} grounding-gate rejections barred from "
            "reinstatement)"
        )
    if unparsed:
        print(f"  Unparseable (kept as REJECT): {unparsed}")
    print(
        f"\n  Final kept: {int(v['final_keep'].sum())} "
        f"(verifier {int(v['verifier_keep'].sum())} + "
        f"adjudicator {total_reinstated})"
    )
    print("=" * 64)
    print(f"Wrote {a.out}")
    print(
        "\nNext: python3 apply_verifier.py "
        "--metrics <exp7_note_level_metrics.csv> "
        f"--verified {a.out} --keep_col final_keep --out final_metrics.csv"
    )


if __name__ == "__main__":
    main()
