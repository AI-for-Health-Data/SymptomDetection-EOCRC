import os
import re
import json
import math
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from bert_score import score as bert_score
    HAVE_BERTSCORE = True
    print("bert_score OK")
except ModuleNotFoundError:
    HAVE_BERTSCORE = False
    print("bert_score not installed — BERT-P will be NaN")

os.environ["TRANSFORMERS_NO_TF"]   = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"]               = "0"
os.environ["USE_FLAX"]             = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


PATH           = "rebuilt_notes_by_noteid.csv"
HF_MODEL_ID    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_CACHE_DIR   = "/lustre/smuexa01/client/users/nikkieh/hf_cache"
TEMPERATURE    = 0
MAX_NOTE_CHARS = None

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

BAD_INFERENCE_VALS = {
    "", "n/a", "na", "not mentioned", "none", "no inference",
    "not reported", "not applicable", "not stated",
    "not mentioned outside ros",
}

PROMPT_TEMPLATE = """
You are an experienced gastroenterology clinician.
You will analyze a patient's ORIGINAL clinical note text and extract information
ONLY from the note text (no assumptions, no external knowledge).

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

For each item, provide:
- Answer (Yes/No)
- Confidence: integer 1-5 (1=Very Low, 2=Low, 3=Moderate, 4=High, 5=Very High)
- Inference: a short quote or phrase copied from the note text supporting the answer
  (must NOT come from ROS-negative text)

Return a SINGLE valid JSON object with these keys exactly:
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
- If clearly present outside ROS -> "Yes" (confidence 4-5).
- If explicitly denied outside ROS -> "No" (confidence 4-5).
- If only appears as ROS-negative and nowhere else -> "No" (confidence 2).
- Use "N/A" for duration if not applicable or not reported.
- Inference MUST be copied from the NOTE TEXT (must NOT come from ROS-negative text).
- Output ONLY JSON — no prose, no markdown fences, no extra text.

Patient NOTE TEXT:
<<NOTE_TEXT>>
""".strip()

def load_notes():
    notes_df = pd.read_csv(PATH)
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
    notes_df = notes_df[
        notes_df["Clean_note_text"].str.strip().ne("")
    ].reset_index(drop=True)
    print(f"Notes loaded: {len(notes_df)}")
    return notes_df

def maybe_truncate(text, max_chars=None):
    if text is None:
        return ""
    t = str(text)
    if max_chars is None or len(t) <= max_chars:
        return t
    return t[:max_chars // 2] + "\n...\n" + t[-(max_chars // 2):]

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
                return s[start:i+1]
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
        model_id, torch_dtype="float16",
        device_map="auto", trust_remote_code=True,
        cache_dir=HF_CACHE_DIR)
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


def run_inference(notes_df, model_id=HF_MODEL_ID):
    print("\n" + "="*65)
    print("INFERENCE — Experiment: ROS Rules Only")
    print("  ROS rules:      ON")
    print("  Manual aliases: OFF  <- key difference from Exp 4")
    print("  UMLS synonyms:  OFF")
    print("="*65)
    print(f"Prompt length (without note): {len(PROMPT_TEMPLATE):,} chars")

    tokenizer, model = load_llama(model_id)
    rows = []

    for idx, row in tqdm(notes_df.iterrows(), total=len(notes_df),
                         desc="ROS only inference"):
        note_text = maybe_truncate(row["Clean_note_text"], MAX_NOTE_CHARS)
        prompt    = PROMPT_TEMPLATE.replace("<<NOTE_TEXT>>", note_text)
        try:
            content = generate(prompt, tokenizer, model)
        except Exception as e:
            content = ""
            print(f"  Error {idx}: {e}")

        parsed, raw_json = safe_json_loads(content)
        out = row.to_dict()
        out["exp_output_raw"]  = raw_json
        out["exp_output_dict"] = parsed
        rows.append(out)

        if (idx + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(
                "exp_ros_only_checkpoint.csv", index=False)
            print(f"  Checkpoint: {idx+1}/{len(notes_df)}")

    exp_df = pd.DataFrame(rows)
    exp_df.to_csv("exp_ros_only_outputs_raw.csv", index=False)
    total    = len(exp_df)
    n_failed = exp_df["exp_output_dict"].apply(
        lambda x: not isinstance(x, dict)).sum()
    print(f"\nexp_ros_only_outputs_raw.csv  ({total} rows)")
    print(f"Parse failures: {n_failed}/{total}  ({100*n_failed/total:.1f}%)")

    valid_df = exp_df[
        exp_df["exp_output_dict"].apply(lambda x: isinstance(x, dict))
    ].copy().reset_index(drop=True)
    print(f"Valid rows: {len(valid_df)}/{total}")
    return valid_df


def run_metrics(valid_df):
    print("\n" + "="*65)
    print("METRICS — ROS Only")
    print("="*65)

    # Yes counts
    count_rows = []
    for symptom, conf_key, inf_key in SYMPTOM_SPECS:
        yes_count = valid_df["exp_output_dict"].apply(
            lambda d, s=symptom: normalize_answer(d.get(s, "")) == "yes"
        ).sum()
        count_rows.append({"Symptom": symptom, "Yes_Count": int(yes_count)})
    count_df = pd.DataFrame(count_rows)
    count_df.to_csv("exp_ros_only_yes_counts.csv", index=False)

    metric_df = valid_df.copy()
    for symptom, conf_key, inf_key in SYMPTOM_SPECS:
        metric_df[symptom]  = metric_df["exp_output_dict"].apply(
            lambda d, s=symptom:  d.get(s, "")     if isinstance(d, dict) else "")
        metric_df[conf_key] = metric_df["exp_output_dict"].apply(
            lambda d, k=conf_key: d.get(k, np.nan) if isinstance(d, dict) else np.nan)
        metric_df[inf_key]  = metric_df["exp_output_dict"].apply(
            lambda d, k=inf_key:  d.get(k, "")     if isinstance(d, dict) else "")
        metric_df[f"{symptom} Conf_num"] = metric_df[conf_key].apply(to_num)

    # BLEU
    print("Computing BLEU...")
    for symptom, _, inf_key in SYMPTOM_SPECS:
        bleu_col = f"{symptom} BLEU_noBP"
        vals = []
        for _, row in metric_df.iterrows():
            hyp = row[inf_key]
            ref = row["Clean_note_text"]
            if not isinstance(hyp, str) or hyp.strip().lower() in BAD_INFERENCE_VALS:
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
                if not isinstance(hyp, str) or hyp.strip().lower() in BAD_INFERENCE_VALS:
                    continue
                idxs.append(i); refs.append(ref); hyps.append(hyp)
            print(f"  {symptom}: {len(idxs)} rows")
            if idxs:
                vals = compute_bertscore_batch(refs, hyps)
                for i, val in zip(idxs, vals):
                    metric_df.at[i, bert_col] = val
        print("BERTScore done.")

    metric_df.to_csv("exp_ros_only_note_level_metrics.csv", index=False)

    # Summary
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
    summary_table.to_csv("exp_ros_only_summary_table.csv", index=False)

    print("\n" + "="*65)
    print("TABLE I — POSITIVE DETECTION COUNTS (ROS Only)")
    print("ROS rules: ON | Manual aliases: OFF | UMLS: OFF")
    print("="*65)
    print(f"{'Symptom':<46} {'Yes':>6} {'No':>6}")
    print("-"*60)
    for _, r in summary_table.iterrows():
        print(f"  {r['Symptom']:<44} {int(r['Yes_Count']):>6} {int(r['No_Count']):>6}")
    print(f"\n  TOTAL YES: {int(summary_table['Yes_Count'].sum())}")

    print("\n" + "="*65)
    print("TABLE II — CONFIDENCE, BLEU, BERTSCORE (ROS Only)")
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
    print("  CY=Conf Yes  CN=Conf No  BY=BLEU Yes  BN=BLEU No")
    print("\n  NOTE: YES count should be LOWER than Exp 4 (ROS+Aliases)")
    print("  because informal terms (brbpr, wt loss etc) are not recognised")
    print("  CN should drop to ~2.3 — same as Exp 4 — because ROS rules are ON")

    print("\nFiles saved:")
    print("  exp_ros_only_outputs_raw.csv")
    print("  exp_ros_only_yes_counts.csv")
    print("  exp_ros_only_note_level_metrics.csv")
    print("  exp_ros_only_summary_table.csv")

    return summary_table


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="ROS Only — ROS rules in prompt, no aliases, no UMLS"
    )
    parser.add_argument("--phase", default="all",
                        choices=["inference", "metrics", "all"])
    parser.add_argument("--model", default=HF_MODEL_ID)
    parser.add_argument("--max_notes", type=int, default=None)
    args = parser.parse_args()

    notes_df = load_notes()
    if args.max_notes:
        notes_df = notes_df.head(args.max_notes)

    print(f"\nExperiment: ROS Only")
    print(f"  ROS rules:      ON  <- LLaMA ignores ROS negatives")
    print(f"  Manual aliases: OFF <- key difference from Exp 4 (ROS+Manual)")
    print(f"  UMLS synonyms:  OFF")
    print(f"  Notes:          {len(notes_df)}")
    print(f"\n  Expected findings:")
    print(f"  - CN drops to ~2.3 (ROS rules calibrate uncertainty)")
    print(f"  - YES counts LOWER than Exp 4 (no informal term recognition)")
    print(f"  - Proves: aliases drive recall, ROS rules drive calibration")

    valid_df = None

    if args.phase in ("inference", "all"):
        valid_df = run_inference(notes_df, model_id=args.model)
        valid_df.to_csv("exp_ros_only_valid_outputs.csv", index=False)

    if args.phase == "metrics":
        raw_df = pd.read_csv("exp_ros_only_outputs_raw.csv")
        raw_df["exp_output_dict"] = raw_df["exp_output_raw"].apply(
            lambda x: safe_json_loads(str(x))[0] if pd.notna(x) else None)
        valid_df = raw_df[
            raw_df["exp_output_dict"].apply(lambda x: isinstance(x, dict))
        ].copy().reset_index(drop=True)
        print(f"Valid rows loaded: {len(valid_df)}")

    if args.phase in ("metrics", "all") and valid_df is not None:
        run_metrics(valid_df)
