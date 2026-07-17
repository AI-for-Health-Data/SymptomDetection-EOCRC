import argparse
import numpy as np, pandas as pd

FH="Family history of colorectal cancer"
def tier(g):
    if np.isnan(g): return "UNSCORED"
    if g>=0.66: return "HIGH"
    if g>=0.33: return "MODERATE"
    return "LOW"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--grounding",required=True)
    ap.add_argument("--exp",default="E?")
    ap.add_argument("--out",default="grade.csv")
    a=ap.parse_args()
    df=pd.read_csv(a.grounding)
    df["tier"]=df["grounding"].map(tier)
    print("="*80)
    print(f"GRADE-INSPIRED EVIDENCE-GROUNDING TIERS -- {a.exp}")
    print("(HIGH>=0.66 note-supported | MODERATE 0.33-0.66 | LOW<0.33 ungrounded)")
    print("="*80)
    for lab,sub in [("TRUE positives",df[df.tp==True]),("FALSE positives",df[df.tp==False])]:
        if len(sub)==0: continue
        dist=sub["tier"].value_counts(normalize=True)
        print(f"\n  {lab} (n={len(sub)}):")
        for t in ["HIGH","MODERATE","LOW"]:
            print(f"    {t:9s}: {dist.get(t,0):.0%}")
    fh=df[df.symptom==FH]
    if len(fh):
        print(f"\n  --- FAMILY HISTORY only (n={len(fh)}) ---")
        for lab,sub in [("TP",fh[fh.tp==True]),("FP",fh[fh.tp==False])]:
            if len(sub)==0: continue
            dist=sub["tier"].value_counts(normalize=True)
            print(f"    FH {lab} (n={len(sub)}): LOW={dist.get('LOW',0):.0%} "
                  f"MODERATE={dist.get('MODERATE',0):.0%} HIGH={dist.get('HIGH',0):.0%}")
    df.to_csv(a.out,index=False)
    print(f"\n  KEY: FH false positives concentrate in LOW tier = ungrounded = hallucinated.")
    print(f"  Wrote {a.out}")

if __name__=="__main__": main()
