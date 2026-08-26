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
except ModuleNotFoundError:
    HAVE_BERTSCORE = False
    print("bert_score is not installed. Running Exp. 1 with counts + Conf + BLEU only.")


PATH = "rebuilt_notes_by_noteid.csv"   
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TEMPERATURE = 0
MAX_NOTE_CHARS = None   

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

notes_df = pd.read_csv(PATH)

if "Clean_note_text" not in notes_df.columns:
    raise ValueError("Clean_note_text not found in preprocessed file.")

for col in ["DATE_OF_SERVIC_DTTM", "SPEC_NOTE_TIME_DTTM", "CONTACT_DATE"]:
    if col in notes_df.columns:
        notes_df[col] = pd.to_datetime(notes_df[col], errors="coerce")

sort_cols = [c for c in [
    "PAT_ID", "PAT_ENC_CSN_ID", "DATE_OF_SERVIC_DTTM",
    "SPEC_NOTE_TIME_DTTM", "CONTACT_NUM", "NOTE_ID"
] if c in notes_df.columns]
if sort_cols:
    notes_df = notes_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

notes_df = notes_df[
    notes_df["Clean_note_text"].str.strip().ne("")
].reset_index(drop=True)

print(f"Notes loaded and ready for Experiment 1: {len(notes_df)}")
#display(notes_df.head())

PROMPT_TEMPLATE_EXP1 = """
You are an experienced gastroenterology clinician.
You will analyze a patient's ORIGINAL clinical note text and extract information
ONLY from the note text (no assumptions, no external knowledge).

For each item, provide:
- Answer (Yes/No or short text)
- Confidence: integer 1–5
    (1 = Very Low, 2 = Low, 3 = Moderate, 4 = High, 5 = Very High)
- Inference: a short quote or phrase copied from the note text supporting the answer

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
- If a symptom is clearly present, answer "Yes" and give duration if stated (e.g. "3 weeks", "2 months").
- If clearly denied (e.g. "denies abdominal pain"), answer "No" with high confidence.
- If not mentioned at all, answer "No" with lower confidence (e.g. 2).
- Use "N/A" for duration if not applicable or not reported.
- Inference MUST be copied from the NOTE TEXT (keep all evidence inside the JSON string).
- Output ONLY JSON — no prose, no markdown fences, no extra text.

Patient NOTE TEXT:
<<NOTE_TEXT>>
""".strip()


def maybe_truncate(text, max_chars=None):
    if text is None:
        return ""
    t = str(text)
    if max_chars is None or len(t) <= max_chars:
        return t
    head = t[: max_chars // 2]
    tail = t[-(max_chars // 2):]
    return head + "\n...\n" + tail

def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    return s

def extract_first_json_object(s: str) -> str:
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

def safe_json_loads(s: str):
    s0 = extract_first_json_object(s)

    try:
        return json.loads(s0), s0
    except Exception:
        pass

    s1 = s0
    # missing comma between fields (newline-separated)
    s1 = re.sub(r'("|\d|true|false|null)\s*\n(\s*")', r'\1,\n\2', s1)
    # trailing commas
    s1 = re.sub(r",\s*([}\]])", r"\1", s1)
    # smart quotes
    s1 = (s1.replace("\u201c", '"').replace("\u201d", '"')
             .replace("\u2018", "'").replace("\u2019", "'"))
    # unquoted N/A values — e.g.  : N/A,  or  : N/A\n
    s1 = re.sub(r':\s*N/A(\s*[,\n}\]])', r': "N/A"\1', s1)
    s1 = re.sub(r':\s*None(\s*[,\n}\]])', r': "None"\1', s1)
    s1 = re.sub(r':\s*n/a(\s*[,\n}\]])', r': "N/A"\1', s1)

    try:
        return json.loads(s1), s1
    except Exception:
        pass

    s2 = re.sub(r'("|\d|true|false|null)(\s*")', r'\1,\2', s1)

    try:
        return json.loads(s2), s2
    except Exception:
        return None, s0

def to_num(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"-?\d+(?:\.\d+)?", str(x))
    return float(m.group()) if m else np.nan

def normalize_answer(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
_CACHE = "/lustre/smuexa01/client/users/nikkieh/hf_cache"
print(f"Loading {MODEL}...")
_tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, cache_dir=_CACHE)
_tokenizer.pad_token = _tokenizer.eos_token
_model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16,
    device_map="auto", trust_remote_code=True, cache_dir=_CACHE)
_model.eval()
print(f"Model loaded on {next(_model.parameters()).device}")

rows = []

for _, row in tqdm(notes_df.iterrows(), total=len(notes_df), desc="Running Experiment 1"):
    note_text = maybe_truncate(row["Clean_note_text"], MAX_NOTE_CHARS)
    prompt = PROMPT_TEMPLATE_EXP1.replace("<<NOTE_TEXT>>", note_text)
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = _tokenizer(formatted, return_tensors="pt",
                        truncation=True, max_length=8192).to(_model.device)
    with torch.no_grad():
        outputs = _model.generate(
            **inputs, max_new_tokens=1024,
            do_sample=False, pad_token_id=_tokenizer.eos_token_id)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    content = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    parsed, raw_json = safe_json_loads(content)

    out = row.to_dict()
    out["exp1_output_raw"] = raw_json
    out["exp1_output_dict"] = parsed
    rows.append(out)

exp1_df = pd.DataFrame(rows)
exp1_df.to_csv("experiment1_outputs_raw.csv", index=False)
print("Saved raw Exp1 outputs: experiment1_outputs_raw.csv")
#display(exp1_df.head())

# keep only valid parsed outputs
valid_df = exp1_df[exp1_df["exp1_output_dict"].apply(lambda x: isinstance(x, dict))].copy()
print(f"Valid parsed outputs: {len(valid_df)} / {len(exp1_df)}")


symptom_specs = [
    ("Abdominal pain", "Abdominal pain confidence", "Abdominal pain inference"),
    ("Rectal bleeding", "Rectal bleeding confidence", "Rectal bleeding inference"),
    ("Rectal pain", "Rectal pain confidence", "Rectal pain inference"),
    ("Diarrhea", "Diarrhea confidence", "Diarrhea inference"),
    ("Constipation", "Constipation confidence", "Constipation inference"),
    ("Weight loss", "Weight loss confidence", "Weight loss inference"),
    ("Family history of colorectal cancer", "Family history of colorectal cancer confidence", "Family history of colorectal cancer inference"),
]

count_rows = []
for symptom, conf_key, inf_key in symptom_specs:
    yes_count = valid_df["exp1_output_dict"].apply(
        lambda d: normalize_answer(d.get(symptom, "")) == "yes"
    ).sum()
    count_rows.append({
        "Symptom": symptom,
        "Positive_Yes_Count": int(yes_count)
    })

count_df = pd.DataFrame(count_rows)
count_df.to_csv("experiment1_yes_counts.csv", index=False)

print("\nPositive counts:")
#display(count_df)


token_pattern = re.compile(r"\w+|\S")

def simple_tokenize(text: str):
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    return token_pattern.findall(text.lower())

def modified_precision(ref_tokens, hyp_tokens, n: int) -> float:
    if len(hyp_tokens) < n:
        return 0.0
    hyp_ngrams = Counter(zip(*[hyp_tokens[i:] for i in range(n)]))
    ref_ngrams = Counter(zip(*[ref_tokens[i:] for i in range(n)]))
    match_count = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())
    total_count = sum(hyp_ngrams.values())
    return 0.0 if total_count == 0 else match_count / total_count

def compute_bleu_no_bp(ref: str, hyp: str, max_n: int = 4) -> float:
    ref_toks = simple_tokenize(ref)
    hyp_toks = simple_tokenize(hyp)
    if not ref_toks or not hyp_toks:
        return 0.0

    max_n = min(max_n, len(hyp_toks))
    precisions = []
    for n in range(1, max_n + 1):
        p_n = modified_precision(ref_toks, hyp_toks, n)
        if p_n == 0.0:
            return 0.0   # any zero precision → BLEU = 0, no smoothing
        precisions.append(p_n)

    avg_log_p = sum(math.log(p) for p in precisions) / len(precisions)
    return math.exp(avg_log_p)


metric_df = valid_df.copy()

for symptom, conf_key, inf_key in symptom_specs:
    metric_df[symptom] = metric_df["exp1_output_dict"].apply(lambda d, s=symptom: d.get(s, "") if isinstance(d, dict) else "")
    metric_df[conf_key] = metric_df["exp1_output_dict"].apply(lambda d, k=conf_key: d.get(k, np.nan) if isinstance(d, dict) else np.nan)
    metric_df[inf_key] = metric_df["exp1_output_dict"].apply(lambda d, k=inf_key: d.get(k, "") if isinstance(d, dict) else "")

for symptom, conf_key, inf_key in symptom_specs:
    metric_df[f"{symptom} Conf_num"] = metric_df[conf_key].apply(to_num)

# BLEU-noBP
for symptom, conf_key, inf_key in symptom_specs:
    bleu_col = f"{symptom} BLEU_noBP"
    vals = []

    for _, row in metric_df.iterrows():
        ref = row["Clean_note_text"]
        hyp = row[inf_key]

        if not isinstance(hyp, str) or hyp.strip().lower() in ["", "n/a", "na", "not mentioned", "none", "no inference"]:
            vals.append(0.0)
        else:
            vals.append(compute_bleu_no_bp(ref, hyp))

    metric_df[bleu_col] = vals

# BERTScore precision
def compute_bertscore_precision_batch(refs, hyps):
    if len(refs) == 0:
        return []
    P, R, F1 = bert_score(hyps, refs, lang="en", rescale_with_baseline=False,
                          batch_size=16, verbose=False)
    return [float(x) for x in P]

for symptom, conf_key, inf_key in symptom_specs:
    bert_col = f"{symptom} BERT_P"
    metric_df[bert_col] = np.nan

if HAVE_BERTSCORE:
    for symptom, conf_key, inf_key in symptom_specs:
        bert_col = f"{symptom} BERT_P"

        idxs, refs, hyps = [], [], []
        for idx, row in metric_df.iterrows():
            hyp = row[inf_key]
            ref = row["Clean_note_text"]

            if not isinstance(hyp, str) or hyp.strip().lower() in ["", "n/a", "na", "not mentioned", "none", "no inference"]:
                continue

            idxs.append(idx)
            refs.append(ref if isinstance(ref, str) else "")
            hyps.append(hyp)

        if len(idxs) > 0:
            bert_vals = compute_bertscore_precision_batch(refs, hyps)
            for idx, val in zip(idxs, bert_vals):
                metric_df.at[idx, bert_col] = val

metric_df.to_csv("experiment1_note_level_metrics.csv", index=False)

summary_rows = []

for symptom, conf_key, inf_key in symptom_specs:
    conf_col = f"{symptom} Conf_num"
    bleu_col = f"{symptom} BLEU_noBP"
    bert_col = f"{symptom} BERT_P"

    yes_count = (metric_df[symptom].astype(str).str.strip().str.lower() == "yes").sum()

    summary_rows.append({
        "Symptom": symptom,
        "Yes_Count": int(yes_count),
        "Conf_Mean": metric_df[conf_col].mean(),
        "Conf_SD": metric_df[conf_col].std(),
        "BLEU_Mean": metric_df[bleu_col].mean(),
        "BLEU_SD": metric_df[bleu_col].std(),
        "BERTP_Mean": metric_df[bert_col].mean(),
        "BERTP_SD": metric_df[bert_col].std(),
    })

summary_table = pd.DataFrame(summary_rows)
summary_table.to_csv("experiment1_summary_table.csv", index=False)

print("\nExperiment 1 summary table:")
#display(summary_table)

print("\nPretty summary:")
for _, r in summary_table.iterrows():
    print(
        f"{r['Symptom']}: "
        f"Yes={r['Yes_Count']}, "
        f"Conf={r['Conf_Mean']:.2f}±{r['Conf_SD']:.2f}, "
        f"BLEU={r['BLEU_Mean']:.2f}±{r['BLEU_SD']:.2f}, "
        f"BERT-P={r['BERTP_Mean']:.2f}±{r['BERTP_SD']:.2f}"
    )
