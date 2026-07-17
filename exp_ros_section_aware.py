import os, re, json, math
from collections import Counter
import numpy as np, pandas as pd
from tqdm import tqdm

os.environ["TRANSFORMERS_NO_TF"]="1"; os.environ["TRANSFORMERS_NO_FLAX"]="1"
os.environ["USE_TF"]="0"; os.environ["USE_FLAX"]="0"; os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

PATH="rebuilt_notes_by_noteid.csv"
HF_MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_CACHE_DIR="/lustre/smuexa01/client/users/nikkieh/hf_cache"
MAX_NOTE_CHARS=None

SYMPTOMS=["Abdominal pain","Rectal bleeding","Rectal pain","Diarrhea",
          "Constipation","Weight loss","Family history of colorectal cancer"]
SYMPTOM_SPECS=[(s,f"{s} confidence",f"{s} inference") for s in SYMPTOMS]
BAD_INFERENCE_VALS={"", "n/a","na","not mentioned","none","no inference","not reported",
    "not applicable","not stated","not mentioned outside ros"}

PROMPT_TEMPLATE = """
You are an experienced gastroenterology clinician.
Analyze the patient's ORIGINAL clinical note and extract information ONLY from the note text
(no assumptions, no external knowledge).

SECTION-AWARE RULE ABOUT ROS (Review of Systems):
- A "Review of Systems" (ROS) section is a specifically LABELED block (headed "ROS" or
  "Review of Systems") that lists templated positives/negatives.
- ONLY inside that labeled ROS block should you discount a templated negative (e.g. "no
  abdominal pain") when OTHER sections (Chief Complaint, HPI, Assessment/Plan, Diagnosis)
  clearly indicate the symptom is present.
- OUTSIDE the labeled ROS block, TRUST the note. If any section states or denies a symptom,
  use it directly.
- CRITICAL: Do NOT treat "not explicitly denied" as evidence of presence. Absence of a denial
  is NOT a Yes. Answer "Yes" ONLY when the note affirmatively states the symptom is present.
- If a symptom is not clearly stated as present anywhere, answer "No".
  * If it is explicitly denied outside ROS -> "No", confidence 4-5.
  * If it is simply not mentioned as present -> "No", confidence 2.

For each item provide: Answer (Yes/No); Confidence 1-5; Inference (a short quote copied from
the NOTE supporting the answer; must NOT be an ROS-negative template).

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
- Affirmatively present (outside ROS, or present-in-ROS confirmed elsewhere) -> "Yes" (conf 4-5).
- Explicitly denied outside ROS -> "No" (conf 4-5).
- Not clearly present anywhere -> "No" (conf 2).  [do NOT infer presence from missing denial]
- Use "N/A" for duration if not applicable.
- Output ONLY JSON — no prose, no markdown fences.

Patient NOTE TEXT:
<<NOTE_TEXT>>
""".strip()

def load_notes():
    df=pd.read_csv(PATH)
    if "Clean_note_text" not in df.columns: raise ValueError("Clean_note_text not found.")
    df=df[df["Clean_note_text"].str.strip().ne("")].reset_index(drop=True)
    print(f"Notes loaded: {len(df)}"); return df
def maybe_truncate(t,m=None):
    if t is None: return ""
    t=str(t); return t if (m is None or len(t)<=m) else t[:m//2]+"\n...\n"+t[-(m//2):]
def strip_fences(s):
    s=s.strip()
    if s.startswith("```"):
        s=re.sub(r"^```(?:json)?","",s,flags=re.IGNORECASE).strip()
        if s.endswith("```"): s=s[:-3].strip()
    return s
def first_json(s):
    s=strip_fences(s); st=s.find("{")
    if st==-1: return s
    d=0
    for i in range(st,len(s)):
        if s[i]=="{": d+=1
        elif s[i]=="}":
            d-=1
            if d==0: return s[st:i+1]
    return s
def safe_json(s):
    s0=first_json(s)
    try: return json.loads(s0),s0
    except: pass
    s1=re.sub(r'("|\d|true|false|null)\s*\n(\s*")',r'\1,\n\2',s0)
    s1=re.sub(r",\s*([}\]])",r"\1",s1)
    s1=(s1.replace("\u201c",'"').replace("\u201d",'"').replace("\u2018","'").replace("\u2019","'"))
    for pat in [r':\s*N/A(\s*[,\n}\]])',r':\s*None(\s*[,\n}\]])',r':\s*n/a(\s*[,\n}\]])']:
        s1=re.sub(pat,lambda m:': "N/A"'+m.group(1),s1)
    try: return json.loads(s1),s1
    except: return None,s0
def norm(x): return "" if pd.isna(x) else str(x).strip().lower()
def to_num(x):
    if pd.isna(x): return np.nan
    m=re.search(r"-?\d+(?:\.\d+)?",str(x)); return float(m.group()) if m else np.nan

def load_llama(mid=HF_MODEL_ID):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Loading {mid}")
    tok=AutoTokenizer.from_pretrained(mid,trust_remote_code=True,cache_dir=HF_CACHE_DIR)
    tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(mid,torch_dtype="float16",device_map="auto",
        trust_remote_code=True,cache_dir=HF_CACHE_DIR); model.eval()
    print("Model loaded"); return tok,model
def generate(prompt,tok,model,max_new=1024):
    import torch
    msgs=[{"role":"user","content":prompt}]
    try: f=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    except: f=f"<|user|>\n{prompt}\n<|assistant|>\n"
    inp=tok(f,return_tensors="pt",truncation=True,max_length=8192).to(model.device)
    with torch.no_grad():
        out=model.generate(**inp,max_new_tokens=max_new,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True).strip()

def run_inference(df,mid=HF_MODEL_ID):
    print("="*60); print("INFERENCE — E3b: SECTION-AWARE ROS rule"); print("="*60)
    tok,model=load_llama(mid); rows=[]
    for idx,row in tqdm(df.iterrows(),total=len(df),desc="section-aware ROS"):
        note=maybe_truncate(row["Clean_note_text"],MAX_NOTE_CHARS)
        prompt=PROMPT_TEMPLATE.replace("<<NOTE_TEXT>>",note)
        try: content=generate(prompt,tok,model)
        except Exception as e: content=""; print(f"  err {idx}: {e}")
        parsed,raw=safe_json(content)
        out=row.to_dict(); out["exp_output_raw"]=raw; out["exp_output_dict"]=parsed
        rows.append(out)
        if (idx+1)%50==0:
            pd.DataFrame(rows).to_csv("exp_ros_section_checkpoint.csv",index=False)
            print(f"  ckpt {idx+1}/{len(df)}")
    exp=pd.DataFrame(rows); exp.to_csv("exp_ros_section_outputs_raw.csv",index=False)
    nf=exp["exp_output_dict"].apply(lambda x:not isinstance(x,dict)).sum()
    print(f"\nexp_ros_section_outputs_raw.csv ({len(exp)} rows) parse-fail {nf}")
    return exp[exp["exp_output_dict"].apply(lambda x:isinstance(x,dict))].reset_index(drop=True)

def run_metrics(valid):
    md=valid.copy()
    for s,ck,ik in SYMPTOM_SPECS:
        md[s]=md["exp_output_dict"].apply(lambda d,s=s: d.get(s,"") if isinstance(d,dict) else "")
        md[ck]=md["exp_output_dict"].apply(lambda d,k=ck: d.get(k,np.nan) if isinstance(d,dict) else np.nan)
        md[ik]=md["exp_output_dict"].apply(lambda d,k=ik: d.get(k,"") if isinstance(d,dict) else "")
    md.to_csv("exp_ros_section_note_level_metrics.csv",index=False)
    # yes counts
    print("\nYES counts (section-aware ROS):")
    for s,_,_ in SYMPTOM_SPECS:
        n=(md[s].astype(str).str.strip().str.lower()=="yes").sum()
        print(f"  {s:40s} {int(n)}")
    print("\nWrote exp_ros_section_note_level_metrics.csv")
    print("Next: score with gold_eval.py (put it in a folder as exp_ros_section) and compare to E3.")
    return md

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",default="all",choices=["inference","metrics","all"])
    ap.add_argument("--model",default=HF_MODEL_ID)
    ap.add_argument("--max_notes",type=int,default=None)
    a=ap.parse_args()
    df=load_notes()
    if a.max_notes: df=df.head(a.max_notes)
    valid=None
    if a.phase in ("inference","all"):
        valid=run_inference(df,a.model)
    if a.phase=="metrics":
        raw=pd.read_csv("exp_ros_section_outputs_raw.csv")
        raw["exp_output_dict"]=raw["exp_output_raw"].apply(lambda x: safe_json(str(x))[0] if pd.notna(x) else None)
        valid=raw[raw["exp_output_dict"].apply(lambda x:isinstance(x,dict))].reset_index(drop=True)
    if a.phase in ("metrics","all") and valid is not None:
        run_metrics(valid)
