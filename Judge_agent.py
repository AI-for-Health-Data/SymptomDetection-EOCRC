from __future__ import annotations
import argparse
import json
import os
import re
import time

import pandas as pd

try:
    from tqdm import tqdm
except ModuleNotFoundError:           
    def tqdm(it, **kw):
        return it

import eval_common as ec


# Prompt — adapted from RAGAS faithfulness + G-Eval

JUDGE_SYSTEM = (
    "You are a meticulous clinical evidence auditor. You evaluate whether a "
    "symptom extraction made by another system is faithfully grounded in the "
    "source clinical note. You never use outside medical knowledge to fill "
    "gaps: a statement counts as supported ONLY if the note itself supports it."
)

JUDGE_TEMPLATE = """\
TASK
You are auditing one symptom extraction. You are given (a) the SOURCE NOTE,
(b) the target SYMPTOM, (c) the system's LABEL (Yes/No), and (d) the EVIDENCE
span the system cited from the note. Judge how faithfully the evidence is
grounded in the note and how well the label fits.

EVALUATION CRITERIA
- Faithfulness (grounding): The EVIDENCE must be present in the SOURCE NOTE
  verbatim, or be a direct paraphrase/inference clearly entailed by the note.
  Text that appears nowhere in the note, or that is pulled from outside
  knowledge or from retrieved literature rather than the note, is NOT faithful.
- Clinical plausibility (label fit): Does the LABEL (Yes/No) follow from the
  cited EVIDENCE and the overall note? A "No" supported by an explicit denial in
  the note is plausible; a "Yes" whose evidence does not actually describe the
  symptom is not.
- Judge faithfulness INDEPENDENTLY of the label. A correctly grounded "No"
  (e.g. a real denial quoted from the note) should still receive a HIGH
  faithfulness score.

EVALUATION STEPS (think through these before answering)
1. Locate the EVIDENCE in the SOURCE NOTE. Is it there verbatim, paraphrased,
   or absent?
2. Decide evidence_in_note: "Yes" only if the note supports the evidence
   statement; otherwise "No".
3. Score grounding_faithfulness 1-5:
   5 = evidence is verbatim or an unambiguous paraphrase from the note;
   4 = clearly inferable from the note with minor wording change;
   3 = partially supported / loosely related to note content;
   2 = weakly related, mostly not in the note;
   1 = not in the note at all (fabricated or from outside the note).
4. Score clinical_plausibility 1-5 for whether the LABEL fits the evidence and
   the note (5 = label clearly correct given the note; 1 = label contradicted).
5. Write a one-sentence rationale (<=40 words) BEFORE giving the scores.

SOURCE NOTE:
<<NOTE>>

SYMPTOM: <<SYMPTOM>>
SYSTEM LABEL: <<LABEL>>
CITED EVIDENCE: "<<EVIDENCE>>"

Return ONLY valid JSON, no prose, no markdown:
{
  "rationale": "<one sentence, <=40 words>",
  "evidence_in_note": "Yes" or "No",
  "grounding_faithfulness": <integer 1-5>,
  "clinical_plausibility": <integer 1-5>
}
"""


def build_prompt(note_text, symptom, label, evidence, max_note_chars=12000):
    note = note_text if isinstance(note_text, str) else ""
    if len(note) > max_note_chars:                      
        half = max_note_chars // 2
        note = note[:half] + "\n...\n" + note[-half:]
    return (JUDGE_TEMPLATE
            .replace("<<NOTE>>", note)
            .replace("<<SYMPTOM>>", str(symptom))
            .replace("<<LABEL>>", str(label))
            .replace("<<EVIDENCE>>", str(evidence).replace('"', "'")))


def parse_judge_json(text):
    if not isinstance(text, str):
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    if s.endswith("```"):
        s = s[:-3].strip()
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                blob = s[start:i + 1]
                try:
                    return json.loads(blob)
                except Exception:
                    blob2 = re.sub(r",\s*([}\]])", r"\1", blob)
                    try:
                        return json.loads(blob2)
                    except Exception:
                        return None
    return None


def _coerce(d):
    """Normalise judge output into a clean record."""
    def to_int(x, lo=1, hi=5):
        try:
            v = int(round(float(re.search(r"-?\d+(\.\d+)?", str(x)).group())))
            return max(lo, min(hi, v))
        except Exception:
            return None
    if not isinstance(d, dict):
        return {"evidence_in_note": None, "grounding_faithfulness": None,
                "clinical_plausibility": None, "rationale": "",
                "judge_parse_ok": False}
    ein = str(d.get("evidence_in_note", "")).strip().lower()
    ein = "Yes" if ein.startswith("y") else ("No" if ein.startswith("n") else None)
    return {
        "evidence_in_note": ein,
        "grounding_faithfulness": to_int(d.get("grounding_faithfulness")),
        "clinical_plausibility": to_int(d.get("clinical_plausibility")),
        "rationale": str(d.get("rationale", ""))[:300],
        "judge_parse_ok": True,
    }


class OllamaJudge:
    def __init__(self, model="llama3:latest", temperature=0):
        import ollama
        self.ollama = ollama
        self.model = model
        self.temperature = temperature

    def __call__(self, prompt):
        r = self.ollama.chat(
            model=self.model,
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
        )
        return r["message"]["content"]


class HFJudge:
    def __init__(self, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                 max_new_tokens=200):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt):
        msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt}]
        inputs = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
                pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)


def get_backend(name, model):
    if name == "ollama":
        return OllamaJudge(model or "llama3:latest")
    if name == "hf":
        return HFJudge(model or "meta-llama/Meta-Llama-3.1-8B-Instruct")
    raise ValueError(name)

def judge_experiment(exp_key, base_dir=".", backend="ollama", model=None,
                     only_yes=False, limit=None, out_dir="judge_results"):
    long_df = ec.to_long(ec.load_experiment(exp_key, base_dir), exp_key)
    if long_df.empty:
        raise SystemExit(f"No data for {exp_key}. Check EXPERIMENTS / base_dir.")

    long_df = long_df[~long_df["evidence"].map(ec.is_trivial_inference)].copy()
    if only_yes:
        long_df = long_df[long_df["answer"] == "yes"].copy()
    if limit:
        long_df = long_df.head(limit).copy()

    judge = get_backend(backend, model)
    os.makedirs(out_dir, exist_ok=True)

    recs = []
    for _, r in tqdm(long_df.iterrows(), total=len(long_df),
                     desc=f"Judging {exp_key}"):
        prompt = build_prompt(r["note_text"], r["symptom"],
                              r["answer"], r["evidence"])
        try:
            raw = judge(prompt)
        except Exception as e:
            raw = f"__ERROR__ {e}"
            time.sleep(1)
        rec = _coerce(parse_judge_json(raw))
        # deterministic grounding cross-check (no LLM)
        g = ec.grounding_check(r["note_text"], r["evidence"])
        rec.update({
            "exp": exp_key, "note_id": r["note_id"], "symptom": r["symptom"],
            "label": r["answer"], "evidence": r["evidence"],
            "bleu": r["bleu"],
            "det_ground_verdict": g["verdict"],
            "det_fuzzy_ratio": g["fuzzy_ratio"],
            "judge_raw": raw,
        })
        recs.append(rec)

    res = pd.DataFrame(recs)
    out_csv = os.path.join(out_dir, f"judge_{exp_key}.csv")
    res.to_csv(out_csv, index=False)

    res["det_in_note"] = res["det_ground_verdict"].isin(
        ["exact", "normalized", "fuzzy"])
    res["judge_in_note"] = res["evidence_in_note"] == "Yes"
    both = res.dropna(subset=["evidence_in_note"])
    agree = (both["det_in_note"] == both["judge_in_note"]).mean() if len(both) else float("nan")

    summary = (res.groupby("symptom")
               .agg(n=("note_id", "size"),
                    judge_in_note_rate=("judge_in_note", "mean"),
                    det_in_note_rate=("det_in_note", "mean"),
                    mean_faithfulness=("grounding_faithfulness", "mean"),
                    mean_plausibility=("clinical_plausibility", "mean"),
                    mean_bleu=("bleu", "mean"))
               .reset_index())
    summary.to_csv(os.path.join(out_dir, f"judge_{exp_key}_summary.csv"),
                   index=False)

    print(f"\nSaved {out_csv}")
    print(f"Judge vs. deterministic grounding agreement: {agree:.1%}"
          if agree == agree else "Judge vs. deterministic agreement: n/a")
    print(summary.to_string(index=False))
    return res, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="E1..E8")
    ap.add_argument("--base_dir", default=".")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "hf"])
    ap.add_argument("--judge_model", default=None)
    ap.add_argument("--only_yes", action="store_true",
                    help="audit only positive predictions (matches paper focus)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out_dir", default="judge_results")
    args = ap.parse_args()
    judge_experiment(args.exp, args.base_dir, args.backend, args.judge_model,
                     args.only_yes, args.limit, args.out_dir)


if __name__ == "__main__":
    main()
