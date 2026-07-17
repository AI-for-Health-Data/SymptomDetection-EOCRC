from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

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

GRADE_HIGH = 0.66
GRADE_MODERATE = 0.33

_TOKEN_PATTERN = re.compile(r"\w+|\S")


def bleu_no_bp(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """BLEU precision (geometric mean of 1..4-gram) without brevity penalty.

    hypothesis = the extractor's cited inference span.
    reference  = the full clinical note.
    High score => the cited span appears (near) verbatim in the note.
    Score ~0   => the cited span is not in the note (ungrounded citation).
    """
    hyp_tokens = _TOKEN_PATTERN.findall(str(hypothesis).lower())
    ref_tokens = _TOKEN_PATTERN.findall(str(reference).lower())
    if not hyp_tokens or not ref_tokens:
        return 0.0
    precisions: List[float] = []
    effective_max_n = min(max_n, len(hyp_tokens))
    for n in range(1, effective_max_n + 1):
        hyp_ngrams = Counter(
            tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
        )
        clipped = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())
        if total == 0 or clipped == 0:
            return 0.0
        precisions.append(clipped / total)
    return math.exp(sum(math.log(p) for p in precisions) / len(precisions))


def grade_tier(bleu: float) -> str:
    if bleu >= GRADE_HIGH:
        return "HIGH"
    if bleu >= GRADE_MODERATE:
        return "MODERATE"
    return "LOW"


VERIFIER_SYSTEM = """You are a clinical fact-checking judge. You decide whether a patient's clinical note supports the claim that the patient has a specific finding.

Use ONLY the clinical note. Do not use outside medical knowledge, retrieved literature, or inference from diagnoses, medications, or risk factors alone.

Decide in two steps.

STEP 1 - Is the finding ADDRESSED anywhere in the note?
A finding is addressed if the note refers to it at all, whether affirming or denying it. Addressing includes:
- direct statements ("patient has abdominal pain");
- clinical shorthand and abbreviations ("abd pn", "BRBPR", "c/o wt loss", "FH CA colon");
- unambiguous indirect evidence ("guaiac positive" for rectal bleeding, "RLQ tenderness" for abdominal pain);
- an exam, laboratory, or procedure result that demonstrates the finding.
If the note does NOT refer to the finding at all, answer ABSENT.

STEP 2 - If addressed, does the note AFFIRM or DENY the finding for THIS patient?
- Answer SUPPORTED if the note affirms the finding for this patient, including terse, abbreviated, or indirect affirmative evidence.
- Answer CONTRADICTED if the note explicitly denies the finding, or only refers to it without affirming it for this patient, and provides no affirmative patient-specific evidence elsewhere.
- If the note contains both a templated denial and clear affirmative evidence outside that denial, answer SUPPORTED.

Rules:
- Absence of a denial is not evidence of presence.
- Evidence about another person does not establish a finding in the patient.
- A related diagnosis, medication, procedure, or risk factor is not by itself affirmative evidence of the finding.
- For family history of colorectal cancer: SUPPORTED requires a family member (or an explicit family-history statement) linked to colorectal, colon, rectal, or bowel cancer. The patient's own cancer history, a different cancer type, unspecified family cancer, or hereditary-risk discussion alone is ABSENT for this claim.

Respond with exactly ONE WORD on a single line: SUPPORTED, CONTRADICTED, or ABSENT. Do not explain."""

VERIFIER_RETRY = "Answer with ONE WORD only: SUPPORTED, CONTRADICTED, or ABSENT."


def build_verifier_runtime(note_text: str, symptom: str, note_limit: int) -> str:
    """Zero-shot runtime message. Symptom-independent phrasing for all seven."""
    claim = (
        "family history of colorectal cancer"
        if symptom == "Family history of colorectal cancer"
        else symptom
    )
    return (
        f"CLINICAL NOTE:\n{note_text[:note_limit]}\n\n"
        f"CANDIDATE FINDING: {claim}\n\n"
        "VERDICT:"
    )


_LABEL_MAP = {
    "SUPPORTED": "supported",
    "CONTRADICTED": "contradicted",
    "ABSENT": "absent",
    # backward-compatibility with a prior NLI-labelled verifier
    "ENTAILMENT": "supported",
    "ENTAIL": "supported",
    "CONTRADICTION": "contradicted",
    "CONTRADICT": "contradicted",
    "NEUTRAL": "absent",
}


_SCAN_ORDER = [
    ("CONTRADICTED", "contradicted"),
    ("CONTRADICTION", "contradicted"),
    ("SUPPORTED", "supported"),
    ("ENTAILMENT", "supported"),
    ("CONTRADICT", "contradicted"),
    ("ENTAIL", "supported"),
    ("ABSENT", "absent"),
    ("NEUTRAL", "absent"),
]


def parse_verdict(text: str) -> Tuple[str, bool]:
    """Return (verdict, parsed_ok). Unparseable -> ('absent', False) so the
    caller can flag it; the keep-rule then retains flagged predictions."""
    upper = str(text).strip().upper()

    # First non-empty line that starts with a known keyword.
    for line in upper.splitlines():
        line = line.strip()
        if not line:
            continue
        for keyword, label in _LABEL_MAP.items():
            if line.startswith(keyword):
                return label, True
        break

    # Keyword anywhere (CONTRADICTED-first to avoid substring collisions).
    for keyword, label in _SCAN_ORDER:
        if re.search(rf"\b{keyword}\b", upper):
            return label, True

    match = re.search(r"\{.*\}", str(text), re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            value = str(data.get("verdict", "")).strip().upper()
            if value in _LABEL_MAP:
                return _LABEL_MAP[value], True
        except Exception:
            pass

    return "absent", False


def get_pred_dict(cell) -> dict:
    if isinstance(cell, dict):
        return cell
    if isinstance(cell, str):
        try:
            return ast.literal_eval(cell)
        except Exception:
            try:
                return json.loads(cell)
            except Exception:
                return {}
    return {}


def norm_label(answer) -> str:
    a = str(answer).strip().lower()
    if a.startswith("y"):
        return "yes"
    if a.startswith("n"):
        return "no"
    return a


def get_inference_span(pred_dict: dict, symptom: str) -> str:
    for key in (
        f"{symptom} inference",
        f"{symptom}_inference",
        f"{symptom} Inference",
        "inference",
    ):
        value = pred_dict.get(key, "")
        if value and str(value).strip().lower() not in ("", "n/a", "none", "null"):
            return str(value).strip()
    return ""


_MODEL = None
_TOKENIZER = None


def load_model(model_id: str, cache_dir: str, max_new_tokens: int):
    """Load LLaMA once. Uses text-generation with chat messages."""
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _TOKENIZER, _MODEL
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading verifier model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    _MODEL, _TOKENIZER = model, tokenizer
    return tokenizer, model


def generate_one_word(
    messages: List[Dict[str, str]],
    tokenizer,
    model,
    max_new_tokens: int,
) -> str:
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


def verify_clinical_evidence(
    note_text: str,
    symptom: str,
    tokenizer,
    model,
    note_limit: int,
    max_new_tokens: int,
    dry_run: bool,
) -> Tuple[str, bool, str, bool]:
    """Return (verdict, parsed_ok, raw_output, retry_used).

    verdict is one of 'supported' / 'contradicted' / 'absent'.
    Zero-shot: system prompt + one runtime message. No demonstration.
    """
    if dry_run:
        # Regex stub for pipeline plumbing tests only — NOT used for results.
        text = note_text.lower()
        if symptom == "Family history of colorectal cancer":
            ok = bool(
                re.search(
                    r"(family|father|mother|sibling|brother|sister|parent|fh|fhx)"
                    r".{0,40}(colorectal|colon|rectal|bowel|ca colon|crc)",
                    text,
                )
            )
        else:
            ok = any(word in text for word in symptom.lower().split())
        return ("supported" if ok else "absent"), True, "DRY_RUN", False

    runtime = build_verifier_runtime(note_text, symptom, note_limit)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": VERIFIER_SYSTEM},
        {"role": "user", "content": runtime},
    ]
    raw = generate_one_word(messages, tokenizer, model, max_new_tokens)
    verdict, parsed_ok = parse_verdict(raw)
    retry_used = False

    if not parsed_ok:
        retry_used = True
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": VERIFIER_RETRY},
        ]
        retry_raw = generate_one_word(
            retry_messages, tokenizer, model, max_new_tokens
        )
        retry_verdict, retry_ok = parse_verdict(retry_raw)
        raw = (raw + " || " + retry_raw)[:400]
        if retry_ok:
            verdict, parsed_ok = retry_verdict, True

    return verdict, parsed_ok, raw[:400], retry_used


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Agent 2: note-only verifier with VeriFact-grounded verdicts "
            "and a programmatic BLEU grounding gate."
        )
    )
    ap.add_argument("--pred", required=True, help="Frozen E7 extractor CSV.")
    ap.add_argument("--note_col", default="Clean_note_text")
    ap.add_argument("--pred_col", default="exp_output_dict")
    ap.add_argument("--id_col", default="NOTE_ID")
    ap.add_argument("--out", default="verify_output.csv")
    ap.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument(
        "--cache_dir",
        default="/lustre/smuexa01/client/users/nikkieh/hf_cache",
    )
    ap.add_argument("--symptoms_only", default=None)
    ap.add_argument(
        "--reject_on",
        default="absent,contradicted",
        help="Verdicts that trigger rejection (Gate 1).",
    )
    ap.add_argument(
        "--grade_reject_tier",
        default="LOW",
        choices=["LOW", "MODERATE", "HIGH", "NONE"],
        help="Grounding tiers at or below which to reject (Gate 2).",
    )
    ap.add_argument("--note_limit", type=int, default=6000)
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=16,
        help="Verifier output cap. 16 leaves headroom over a one-word answer; "
        "8 risks truncating a prepended token into an unparseable output.",
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.pred)

    # Resolve note column.
    if args.note_col not in df.columns:
        for candidate in ("Clean_note_text", "note_text", "NOTE_TEXT", "text", "note"):
            if candidate in df.columns:
                args.note_col = candidate
                break
    if args.note_col not in df.columns:
        raise SystemExit(
            f"[ERROR] note column not found; have {list(df.columns)[:12]}"
        )
    if args.pred_col not in df.columns:
        raise SystemExit(f"[ERROR] pred column '{args.pred_col}' not found")

    targets = (
        [s.strip() for s in args.symptoms_only.split(",")]
        if args.symptoms_only
        else SYMPTOMS
    )

    nli_reject = {v.strip().lower() for v in args.reject_on.split(",")}
    grade_tiers_list = ["LOW", "MODERATE", "HIGH"]
    if args.grade_reject_tier == "NONE":
        grade_reject_set: set = set()
    else:
        cutoff = grade_tiers_list.index(args.grade_reject_tier)
        grade_reject_set = set(grade_tiers_list[: cutoff + 1])

    print("=" * 64)
    print("AGENT 2: NOTE-ONLY VERIFIER (VeriFact-grounded, zero-shot)")
    print("  Verdicts       : SUPPORTED / CONTRADICTED / ABSENT")
    print("                   (adapted from VeriFact Supported/Not Supported/")
    print("                    Not Addressed; Chung et al., NEJM AI 2025)")
    print(f"  Symptoms       : {targets}")
    print(f"  Gate 1 reject  : {nli_reject}")
    print(
        f"  Gate 2 reject  : "
        f"{grade_reject_set if grade_reject_set else 'disabled'}"
    )
    print("  Shots          : 0 (no per-symptom example)")
    print(f"  Output cap     : {args.max_new_tokens} tokens")
    print(f"  Note limit     : {args.note_limit} chars")
    print(f"  Mode           : {'DRY RUN (regex)' if args.dry_run else args.model_id}")
    print("  Retrieval seen : NO    Gold seen: NO")
    print("=" * 64)

    tokenizer = model = None
    if not args.dry_run:
        tokenizer, model = load_model(
            args.model_id, args.cache_dir, args.max_new_tokens
        )

    # Resumable checkpoint.
    ckpt = args.out + ".ckpt"
    rows: List[dict] = []
    done: set = set()
    if os.path.exists(ckpt):
        prev = pd.read_csv(ckpt)
        rows = prev.to_dict("records")
        done = {(str(r["note_id"]), r["symptom"]) for r in rows}
        print(f"[resume] loaded {len(rows)} prior verdicts")

    checked = {s: 0 for s in targets}
    flipped = {s: 0 for s in targets}
    grade_rej = {s: 0 for s in targets}
    for r in rows:
        s = r.get("symptom", "")
        if s in checked:
            checked[s] += 1
            if not r.get("verifier_keep", True):
                flipped[s] += 1
            if r.get("grade_rejected", False):
                grade_rej[s] += 1

    for idx, row in df.iterrows():
        note = str(row[args.note_col])
        pred_dict = get_pred_dict(row[args.pred_col])
        note_id = str(row.get(args.id_col, idx))

        for symptom in targets:
            if norm_label(pred_dict.get(symptom, "")) != "yes":
                continue
            if (note_id, symptom) in done:
                continue
            checked[symptom] += 1

            # Gate 1: clinical evidence check (LLM).
            verdict, parsed_ok, raw, retry_used = verify_clinical_evidence(
                note,
                symptom,
                tokenizer,
                model,
                args.note_limit,
                args.max_new_tokens,
                args.dry_run,
            )

            # Gate 2: grounding of the extractor's cited inference (BLEU).
            inference_span = get_inference_span(pred_dict, symptom)
            bleu = bleu_no_bp(inference_span, note) if inference_span else 0.0
            tier = grade_tier(bleu)

            # Combined keep rule.
            nli_rejects = parsed_ok and (verdict in nli_reject)
            grade_rejects = (
                parsed_ok and verdict == "supported" and tier in grade_reject_set
            )
            keep = not (nli_rejects or grade_rejects)
            if not parsed_ok:
                keep = True  # unparseable -> retain and flag

            if not keep:
                flipped[symptom] += 1
            if grade_rejects:
                grade_rej[symptom] += 1

            rows.append(
                dict(
                    note_id=note_id,
                    symptom=symptom,
                    verdict=verdict,
                    parsed_ok=parsed_ok,
                    retry_used=retry_used,
                    grade_tier=tier,
                    grade_bleu=round(bleu, 4),
                    nli_rejected=nli_rejects,
                    grade_rejected=grade_rejects,
                    verifier_keep=keep,
                    label_before="yes",
                    label_after="yes" if keep else "no",
                    raw_output=raw,
                )
            )

        if (idx + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(ckpt, index=False)
            flip_str = ", ".join(f"{s.split()[0]}={flipped[s]}" for s in targets)
            print(f"  [{idx + 1}/{len(df)}] flips: {flip_str}", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(args.out, index=False)
    if os.path.exists(ckpt):
        os.remove(ckpt)

    # ---- Summary ----
    print("\n" + "=" * 64)
    print("VERIFIER SUMMARY")
    print(f"  {'Symptom':<42} {'Rejected/Checked':>18}  {'Gate2':>6}")
    for s in targets:
        c, f, g = checked[s], flipped[s], grade_rej[s]
        rate = f"{f / c:.1%}" if c else "n/a"
        print(f"  {s:<42} {f:>7}/{c:<7} ({rate})  g={g}")

    if "verdict" in result.columns:
        print("\nVerdict distribution (VeriFact taxonomy):")
        print(result["verdict"].value_counts(dropna=False).to_string())
        if (result["verdict"] == "contradicted").sum() == 0:
            print(
                "\n[NOTE] CONTRADICTED not emitted. Consistent with VeriFact's "
                "finding that the two negative verdicts are hard to separate; "
                "report the verifier as SUPPORTED/ABSENT in practice."
            )
    if "grade_tier" in result.columns:
        print("\nGrounding tier (all checked):")
        print(result["grade_tier"].value_counts(dropna=False).to_string())
        kept = result[result["verifier_keep"] == True]  # noqa: E712
        print(f"\nGrounding tier (kept only, n={len(kept)}):")
        print(kept["grade_tier"].value_counts(dropna=False).to_string())

    if "parsed_ok" in result.columns:
        n_fail = int((~result["parsed_ok"].astype(bool)).sum())
        if n_fail:
            print(f"\nUnparseable (kept and flagged): {n_fail}/{len(result)}")

    print("=" * 64)
    print(f"Wrote {args.out}")
    print(
        f"\nNext: python3 adjudicate_agent.py --verified {args.out} "
        f"--pred {args.pred} --out adjudicated.csv"
    )


if __name__ == "__main__":
    main()
