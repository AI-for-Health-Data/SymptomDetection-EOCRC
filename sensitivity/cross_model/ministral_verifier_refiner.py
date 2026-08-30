import argparse
import ast
import json
import math
import re
import time
from pathlib import Path
import pandas as pd


ROOT = Path(
    "/lustre/smuexa01/client/users/nikkieh/utsw"
)

MANIFEST = (
    ROOT
    / "runs/verge_m2_ministral3_14b_20260824/"
      "subset/m2_ministral_subset_manifest.csv"
)

A1 = (
    ROOT
    / "runs/verge_agent1_grade_ready_v2/"
      "agent1_outputs_structural_repaired.csv"
)

A2 = (
    ROOT
    / "runs/verge_agent2_20260816/full/"
      "agent2_claims_pair_level.csv"
)

import m2_ministral_backend as backend
import sage_common

sage_common.HF_MODEL_ID = backend.MODEL_PATH
sage_common._LOADED_MODEL = None
sage_common._LOADED_TOKENIZER = None
sage_common.load_llm = backend.load_llm
sage_common.generate_text = backend.generate_text

import agent3_unified_verifier as agent3
import agent4_refiner as agent4
import run_verge_continuation_FINAL as controller

agent3.generate_text = backend.generate_text

agent4.generate_text = backend.generate_text
agent4.load_llm = backend.load_llm

controller.verify_claim = agent3.verify_claim
controller.refine_claim = agent4.refine_claim
controller.load_llm = backend.load_llm


def norm_id(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    s = str(value).strip()

    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]

    return s


def norm_feature(value):
    return re.sub(
        r"\s+",
        " ",
        str(value or "").strip(),
    ).casefold()


def clean(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return value


def as_bool(value):
    if isinstance(value, bool):
        return value

    return (
        str(value)
        .strip()
        .lower()
        in {"1", "true", "yes"}
    )


def yesno01(value):
    s = str(value).strip().lower()

    if s == "yes":
        return 1

    if s == "no":
        return 0

    raise ValueError(
        f"Expected Yes/No, received {value!r}"
    )


def json_safe(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )


def load_cases():
    manifest = pd.read_csv(
        MANIFEST,
        low_memory=False,
    )

    a1 = pd.read_csv(
        A1,
        low_memory=False,
    )

    a2 = pd.read_csv(
        A2,
        low_memory=False,
    )

    for df in (manifest, a1, a2):
        df["_pat_key"] = (
            df["PAT_ID"].map(norm_id)
        )

        df["_note_key"] = (
            df["NOTE_ID"].map(norm_id)
        )

    manifest["_feature_key"] = (
        manifest["Symptom"]
        .map(norm_feature)
    )

    a2["_feature_key"] = (
        a2["feature"]
        .map(norm_feature)
    )

    if "Clean_note_text" not in a1.columns:
        raise RuntimeError(
            "Clean_note_text missing from Agent 1 file."
        )

    note_counts = (
        a1.groupby(
            ["_pat_key", "_note_key"]
        )["Clean_note_text"]
        .nunique(dropna=False)
    )

    if (note_counts > 1).any():
        raise RuntimeError(
            "Conflicting source-note text detected."
        )

    notes = (
        a1[
            [
                "_pat_key",
                "_note_key",
                "Clean_note_text",
            ]
        ]
        .drop_duplicates(
            ["_pat_key", "_note_key"]
        )
    )

    pair_keys = [
        "_pat_key",
        "_note_key",
        "_feature_key",
    ]

    if a2.duplicated(pair_keys).any():
        raise RuntimeError(
            "Duplicate Agent 2 pair keys."
        )

    keep_a2 = [
        "_pat_key",
        "_note_key",
        "_feature_key",
        "feature",
        "prediction",
        "confidence",
        "evidence_quote",
        "note_section",
        "experiencer",
        "assertion_status",
        "relation_found",
        "cancer_type_found",
        "structurer_source",
        "structurer_operational_error",
    ]

    merged = manifest.merge(
        a2[keep_a2],
        on=pair_keys,
        how="left",
        validate="one_to_one",
    )

    merged = merged.merge(
        notes,
        on=[
            "_pat_key",
            "_note_key",
        ],
        how="left",
        validate="many_to_one",
    )

    if len(merged) != 561:
        raise RuntimeError(
            f"Expected 561 cases; found {len(merged)}"
        )

    if merged["feature"].isna().any():
        raise RuntimeError(
            "Missing Agent 2 matches."
        )

    if merged["Clean_note_text"].isna().any():
        raise RuntimeError(
            "Missing source clinical notes."
        )

    a2_labels = (
        merged["prediction"]
        .map(yesno01)
        .astype(int)
    )

    if not (
        a2_labels.values
        == merged["Extractor"].astype(int).values
    ).all():
        raise RuntimeError(
            "Agent 2 labels differ from frozen "
            "Extractor labels."
        )

    return merged.reset_index(drop=True)


def make_claim(row):
    return {
        "feature":
            str(clean(row["feature"])),

        "prediction":
            str(
                clean(row["prediction"])
            ).strip(),

        "confidence":
            clean(
                row.get(
                    "confidence",
                    "",
                )
            ),

        "evidence_quote":
            str(
                clean(
                    row.get(
                        "evidence_quote",
                        "",
                    )
                )
            ),

        "note_section":
            str(
                clean(
                    row.get(
                        "note_section",
                        "",
                    )
                )
            ),

        "experiencer":
            str(
                clean(
                    row.get(
                        "experiencer",
                        "",
                    )
                )
            ),

        "assertion_status":
            str(
                clean(
                    row.get(
                        "assertion_status",
                        "",
                    )
                )
            ),

        "relation_found":
            clean(
                row.get(
                    "relation_found",
                    "",
                )
            ),

        "cancer_type_found":
            clean(
                row.get(
                    "cancer_type_found",
                    "",
                )
            ),
    }


def existing_keys(path):
    if not path.exists():
        return set()

    old = pd.read_csv(
        path,
        low_memory=False,
    )

    return {
        (
            norm_id(r.PAT_ID),
            norm_id(r.NOTE_ID),
            norm_feature(r.Symptom),
        )
        for r in old.itertuples(
            index=False
        )
    }


def append_record(path, record):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = path.exists()

    pd.DataFrame(
        [record]
    ).to_csv(
        path,
        mode="a",
        header=not exists,
        index=False,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--filter",
        choices=[
            "all",
            "changed",
            "shared",
            "representative",
        ],
        default="all",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    output = Path(args.output)

    data = load_cases()

    if args.filter == "changed":
        data = data[
            data[
                "in_all_changed_133"
            ].map(as_bool)
        ].copy()

    elif args.filter == "shared":
        data = data[
            data[
                "in_shared_error_100"
            ].map(as_bool)
        ].copy()

    elif args.filter == "representative":
        data = data[
            data[
                "in_representative_350"
            ].map(as_bool)
        ].copy()

    if args.limit is not None:
        data = data.iloc[
            :args.limit
        ].copy()

    if (
        output.exists()
        and not args.resume
    ):
        raise RuntimeError(
            f"Output already exists: {output}"
        )

    completed = (
        existing_keys(output)
        if args.resume
        else set()
    )

    print("=" * 92)
    print("VERGE M2 — MINISTRAL-3-14B")
    print("=" * 92)
    print("Model:", backend.MODEL_PATH)
    print("Selected cases:", len(data))
    print("Already completed:", len(completed))
    print("Full-note policy: YES")
    print("BERTScore enabled: YES")
    print("Max refinement rounds: 5")
    print("Primary VERGE source modified: NO")
    print("=" * 92)

    processed = 0

    for _, row in data.iterrows():
        key = (
            norm_id(row["PAT_ID"]),
            norm_id(row["NOTE_ID"]),
            norm_feature(
                row["Symptom"]
            ),
        )

        if key in completed:
            continue

        start = time.time()

        claim = make_claim(row)

        note_text = str(
            row["Clean_note_text"]
        )

        initial_error = ""
        loop_error = ""

        try:
            initial_verification = (
                agent3.verify_claim(
                    structured_claim=claim,
                    note_text=note_text,
                    skip_bertscore=False,
                    label_probs=None,
                )
            )

        except Exception as error:
            initial_error = str(error)

            initial_verification = (
                controller.make_verification_error(
                    claim,
                    (
                        "M2 Ministral initial "
                        f"verification failure: {error}"
                    ),
                    label_probs=None,
                )
            )

        try:
            loop = (
                controller
                .continue_verify_refine_from_frozen_agent3(
                    original_claim=claim,
                    initial_verification=initial_verification,
                    note_text=note_text,
                    label_probs=None,
                    skip_bertscore=False,
                    max_rounds=5,
                )
            )

        except Exception as error:
            loop_error = str(error)
            loop = {}

        final_claim = (
            loop.get(
                "final_claim",
                {},
            )
            if isinstance(loop, dict)
            else {}
        )

        final_label = str(
            final_claim.get(
                "prediction",
                "",
            )
        ).strip()

        if final_label == "Yes":
            ministral_pred = 1

        elif final_label == "No":
            ministral_pred = 0

        else:
            ministral_pred = math.nan

        runtime = (
            time.time() - start
        )

        final_verification = (
            loop.get(
                "final_verification",
                {},
            )
            if isinstance(loop, dict)
            else {}
        )

        record = {
            "PAT_ID":
                row["PAT_ID"],

            "NOTE_ID":
                row["NOTE_ID"],

            "Symptom":
                row["Symptom"],

            "Gold":
                int(row["Gold"]),

            "Extractor":
                int(row["Extractor"]),

            "Llama_VERGE":
                int(row["Llama_VERGE"]),

            "Ministral_VERGE":
                ministral_pred,

            "in_representative_350":
                as_bool(
                    row[
                        "in_representative_350"
                    ]
                ),

            "in_all_changed_133":
                as_bool(
                    row[
                        "in_all_changed_133"
                    ]
                ),

            "in_shared_error_100":
                as_bool(
                    row[
                        "in_shared_error_100"
                    ]
                ),

            "change_category":
                row["change_category"],

            "ministral_initial_verdict":
                str(
                    initial_verification.get(
                        "verdict",
                        "",
                    )
                ),

            "ministral_final_prediction":
                final_label,

            "ministral_changed_from_extractor":
                (
                    ministral_pred
                    != int(row["Extractor"])
                    if not pd.isna(
                        ministral_pred
                    )
                    else ""
                ),

            "ministral_exit_reason":
                loop.get(
                    "exit_reason",
                    "",
                )
                if isinstance(loop, dict)
                else "",

            "ministral_refinement_rounds":
                loop.get(
                    "total_refinement_rounds",
                    "",
                )
                if isinstance(loop, dict)
                else "",

            "ministral_flip_count":
                loop.get(
                    "flip_count",
                    "",
                )
                if isinstance(loop, dict)
                else "",

            "ministral_human_review_oscillation":
                loop.get(
                    "human_review_oscillation",
                    "",
                )
                if isinstance(loop, dict)
                else "",

            "ministral_operational_failure":
                loop.get(
                    "operational_failure",
                    True,
                )
                if isinstance(loop, dict)
                else True,

            "ministral_final_verdict":
                str(
                    final_verification.get(
                        "verdict",
                        "",
                    )
                ),

            "initial_error":
                initial_error,

            "loop_error":
                loop_error,

            "runtime_seconds":
                runtime,

            "model_snapshot":
                backend.MODEL_PATH,

            "initial_verification_json":
                json_safe(
                    initial_verification
                ),

            "loop_history_json":
                json_safe(
                    loop.get(
                        "history",
                        [],
                    )
                    if isinstance(loop, dict)
                    else []
                ),
        }

        append_record(
            output,
            record,
        )

        processed += 1

        print(
            f"[{processed:4d}] "
            f"{row['Symptom']} "
            f"E={int(row['Extractor'])} "
            f"L={int(row['Llama_VERGE'])} "
            f"M={ministral_pred} "
            f"init={record['ministral_initial_verdict']} "
            f"exit={record['ministral_exit_reason']} "
            f"rounds={record['ministral_refinement_rounds']} "
            f"{runtime:.1f}s",
            flush=True,
        )

    print("=" * 92)
    print("M2 RUN COMPLETE")
    print("Output:", output)
    print("=" * 92)


if __name__ == "__main__":
    main()
