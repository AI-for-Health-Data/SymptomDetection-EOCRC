from __future__ import annotations
import argparse
import os
import pandas as pd

import eval_common as ec


def build_trajectory(base_dir=".", exp_keys=None):
    """Wide table: (note_id, symptom) x experiments -> answer/evidence/bleu."""
    long_df = ec.load_all_long(base_dir, exp_keys)
    if long_df.empty:
        raise SystemExit(
            "No experiment metric CSVs found. Check EXPERIMENTS paths in "
            "eval_common.py and --base_dir."
        )

    verdicts, ratios = [], []
    for _, r in long_df.iterrows():
        g = ec.grounding_check(r["note_text"], r["evidence"])
        verdicts.append(g["verdict"])
        ratios.append(g["fuzzy_ratio"])
    long_df["ground_verdict"] = verdicts
    long_df["ground_ratio"] = ratios

    keys = ["note_id", "symptom"]
    pieces = []
    for field in ["answer", "evidence", "confidence", "bleu",
                  "ground_verdict", "ground_ratio"]:
        p = long_df.pivot_table(index=keys, columns="exp", values=field,
                                aggfunc="first")
        p.columns = [f"{c}_{field}" for c in p.columns]
        pieces.append(p)
    wide = pd.concat(pieces, axis=1).reset_index()

    # attach note text once
    note_text = (long_df[["note_id", "note_text"]]
                 .drop_duplicates("note_id").set_index("note_id")["note_text"])
    wide["note_text"] = wide["note_id"].map(note_text)
    return long_df, wide


def _norm_ev(s):
    import re
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.lower()).strip()


def divergence_report(long_df, exp_keys):
    rows = []
    grp = long_df.groupby(["note_id", "symptom"])
    for (nid, sym), g in grp:
        g = g.set_index("exp")
        present = [e for e in exp_keys if e in g.index]
        answers = {e: g.loc[e, "answer"] for e in present}
        yes_exps = [e for e in present if answers[e] == "yes"]

        answer_flipped = len(set(answers.values())) > 1
        # distinct evidence spans among the 'yes' experiments
        yes_ev = {e: _norm_ev(g.loc[e, "evidence"]) for e in yes_exps}
        distinct_yes_spans = len(set(v for v in yes_ev.values() if v))
        evidence_changed = distinct_yes_spans > 1

        if not (answer_flipped or evidence_changed):
            continue

        row = {
            "note_id": nid,
            "symptom": sym,
            "answer_flipped": answer_flipped,
            "evidence_changed_among_yes": evidence_changed,
            "n_yes": len(yes_exps),
            "yes_experiments": ",".join(yes_exps),
            "answer_sequence": " | ".join(f"{e}:{answers[e]}" for e in present),
        }
        for e in present:
            row[f"{e}_answer"] = answers[e]
            row[f"{e}_evidence"] = g.loc[e, "evidence"]
            row[f"{e}_ground"] = g.loc[e, "ground_verdict"]
        rows.append(row)
    return pd.DataFrame(rows)


def _group_by_value(items):
    buckets = {}
    order = []
    for exp, val in items:
        if val is None or val == "":
            continue
        if val not in buckets:
            buckets[val] = []
            order.append(val)
        buckets[val].append(exp)
    parts = []
    for val in order:
        exps = ",".join(buckets[val])
        v = val if len(str(val)) <= 80 else (str(val)[:77] + "…")
        parts.append(f"[{exps}]={v}")
    return " ; ".join(parts)


def divergence_report_long(long_df, exp_keys):
    rows = []
    grp = long_df.groupby(["note_id", "symptom"])
    for (nid, sym), g in grp:
        g = g.set_index("exp")
        present = [e for e in exp_keys if e in g.index]
        if not present:
            continue
        answers = {e: g.loc[e, "answer"] for e in present}
        yes_exps = [e for e in present if answers[e] == "yes"]

        answer_flipped = len(set(answers.values())) > 1
        yes_ev = {e: _norm_ev(g.loc[e, "evidence"]) for e in yes_exps}
        distinct_yes_spans = len(set(v for v in yes_ev.values() if v))
        evidence_changed = distinct_yes_spans > 1

        if not (answer_flipped or evidence_changed):
            continue

        # Row 1: ANSWER dimension (all present experiments)
        ans_items = [(e, answers[e]) for e in present]
        rows.append({
            "note_id": nid,
            "symptom": sym,
            "dimension": "answer",
            "changed": answer_flipped,
            "n_groups": len(set(answers.values())),
            "n_yes": len(yes_exps),
            "groups": _group_by_value(ans_items),
            "sequence": " | ".join(f"{e}:{answers[e]}" for e in present),
        })

        # Row 2: EVIDENCE dimension (Yes experiments only; raw span shown,
        ev_items_norm = [(e, yes_ev[e]) for e in yes_exps]
        ev_seq = " | ".join(
            f"{e}:{str(g.loc[e, 'evidence'])[:60]}" for e in yes_exps
        )
        rows.append({
            "note_id": nid,
            "symptom": sym,
            "dimension": "evidence",
            "changed": evidence_changed,
            "n_groups": distinct_yes_spans,
            "n_yes": len(yes_exps),
            "groups": _group_by_value(ev_items_norm),
            "sequence": ev_seq,
        })
    return pd.DataFrame(rows)


def write_exhibit(long_df, symptom, answer_filter="yes",
                  max_examples=15, out_path=None, exp_keys=None):
    if exp_keys is None:
        exp_keys = [e for e in ec.EXPERIMENTS if e in set(long_df["exp"])]

    sub = long_df[long_df["symptom"] == symptom].copy()
    def disagreement(nid):
        g = sub[sub["note_id"] == nid]
        return g["answer"].nunique() + g["evidence"].map(_norm_ev).nunique()

    note_ids = sub["note_id"].unique()
    note_ids = sorted(note_ids, key=disagreement, reverse=True)

    selected = []
    for nid in note_ids:
        g = sub[sub["note_id"] == nid]
        if answer_filter is None or (g["answer"] == answer_filter).any():
            selected.append(nid)
        if len(selected) >= max_examples:
            break

    lines = [f"# Evidence exhibit — {symptom}",
             f"_Filter: answer == {answer_filter}; "
             f"{len(selected)} notes, ranked by cross-experiment disagreement._\n"]
    for nid in selected:
        g = sub[sub["note_id"] == nid].set_index("exp")
        note_text = g["note_text"].iloc[0]
        snippet = (note_text[:600] + " …") if isinstance(note_text, str) and len(note_text) > 600 else note_text
        lines.append(f"\n## Note `{nid}`\n")
        lines.append(f"> {snippet}\n")
        lines.append("| Exp | Ans | Conf | BLEU | Grounded | Cited evidence |")
        lines.append("|-----|-----|------|------|----------|----------------|")
        for e in exp_keys:
            if e not in g.index:
                continue
            r = g.loc[e]
            ev = str(r["evidence"]).replace("\n", " ")
            ev = (ev[:160] + "…") if len(ev) > 160 else ev
            bleu = r["bleu"]
            bleu_s = f"{bleu:.2f}" if pd.notna(bleu) else "—"
            conf = r["confidence"]
            conf_s = f"{conf}" if pd.notna(conf) else "—"
            lines.append(
                f"| {e} {ec.EXPERIMENT_LABELS.get(e,'')} | {r['answer']} | "
                f"{conf_s} | {bleu_s} | {r['ground_verdict']} | {ev} |"
            )
        lines.append("")

    md = "\n".join(lines)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=".")
    ap.add_argument("--symptom", default="Abdominal pain")
    ap.add_argument("--answer", default="yes",
                    help="exhibit filter: yes / no / none(=all)")
    ap.add_argument("--max_examples", type=int, default=15)
    ap.add_argument("--out_dir", default="evidence_comparison")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    present, missing, collisions = ec.discover_experiments(args.base_dir)
    print("Experiments found :", ", ".join(present) or "NONE")
    if missing:
        print("Missing CSVs      :", ", ".join(missing))
    if collisions:
        print("!! FILENAME COLLISIONS (will overwrite each other):", collisions)
        print("   -> rename one on disk and update EXPERIMENTS in eval_common.py")
    exp_keys = list(present.keys())

    long_df, wide = build_trajectory(args.base_dir, exp_keys)
    wide.to_csv(os.path.join(args.out_dir, "evidence_trajectory.csv"), index=False)
    print(f"[1] evidence_trajectory.csv  ({len(wide)} note-symptom rows)")

    div = divergence_report(long_df, exp_keys)
    div.to_csv(os.path.join(args.out_dir, "divergence_report.csv"), index=False)
    print(f"[2] divergence_report.csv    ({len(div)} diverging note-symptoms, wide)")

    div_long = divergence_report_long(long_df, exp_keys)
    div_long.to_csv(os.path.join(args.out_dir, "divergence_report_long.csv"), index=False)
    n_items = len(div_long) // 2 if len(div_long) else 0
    print(f"[2b] divergence_report_long.csv ({len(div_long)} rows = {n_items} items x 2 dims)")

    ans = None if args.answer.lower() in ("none", "all") else args.answer.lower()
    sym_short = ec.SYMPTOM_SHORT.get(args.symptom, "symptom")
    exhibit_path = os.path.join(args.out_dir, f"evidence_exhibit_{sym_short}.md")
    write_exhibit(long_df, args.symptom, ans, args.max_examples,
                  exhibit_path, exp_keys)
    print(f"[3] evidence_exhibit_{sym_short}.md")


if __name__ == "__main__":
    main()
