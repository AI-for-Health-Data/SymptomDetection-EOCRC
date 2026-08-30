import argparse
import json
import math
import re
import time
from pathlib import Path
import pandas as pd
import agent3_unified_verifier as agent3
import agent4_refiner as agent4
import run_verge_continuation_FINAL as controller

ROOT = Path(
    "/lustre/smuexa01/client/users/nikkieh/utsw"
)

AUDIT = (
    ROOT
    / "runs/verge_recall_v2_20260821/full/"
      "continuation_checkpoint_audit.json"
)

A1 = (
    ROOT
    / "runs/verge_agent1_grade_ready_v2/"
      "agent1_outputs_structural_repaired.csv"
)


def norm_id(x):
    if x is None:
        return ""

    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass

    s = str(x).strip()

    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]

    return s


def norm_feature(x):
    return re.sub(
        r"\s+",
        " ",
        str(x or "").strip(),
    ).casefold()


def key_of(pat, note, feature):
    return (
        norm_id(pat),
        norm_id(note),
        norm_feature(feature),
    )


def json_safe(x):
    return json.loads(
        json.dumps(
            x,
            ensure_ascii=False,
            default=str,
        )
    )


def load_notes():
    d = pd.read_csv(
        A1,
        usecols=[
            "PAT_ID",
            "NOTE_ID",
            "Clean_note_text",
        ],
        low_memory=False,
    )

    d["_pat"] = d["PAT_ID"].map(norm_id)
    d["_note"] = d["NOTE_ID"].map(norm_id)

    conflicts = (
        d.groupby(
            ["_pat", "_note"]
        )["Clean_note_text"]
        .nunique(dropna=False)
    )

    if (conflicts > 1).any():
        raise RuntimeError(
            "Conflicting note text for a note key."
        )

    return {
        (r["_pat"], r["_note"]):
            str(r["Clean_note_text"])
        for _, r in d.drop_duplicates(
            ["_pat", "_note"]
        ).iterrows()
    }


def load_bound_cases():
    with open(
        AUDIT,
        encoding="utf-8",
    ) as f:
        records = json.load(f)

    bound = []

    for r in records:
        loop = r.get(
            "loop_info",
            {},
        )

        if (
            str(
                loop.get(
                    "exit_reason",
                    "",
                )
            ).strip()
            != "MAX_ROUNDS_LABEL_OSCILLATION"
        ):
            continue

        claim = loop.get(
            "final_claim",
            {},
        )

        verification = loop.get(
            "final_verification",
            {},
        )

        if not isinstance(claim, dict) or not claim:
            raise RuntimeError(
                "Missing frozen final claim."
            )

        if (
            str(
                verification.get(
                    "verdict",
                    "",
                )
            ).strip().upper()
            != "REFINE"
        ):
            raise RuntimeError(
                "Frozen bound case does not end in REFINE."
            )

        if not bool(
            loop.get(
                "final_verification_applies_to_final_claim",
                False,
            )
        ):
            raise RuntimeError(
                "Final verifier state does not apply "
                "to final claim."
            )

        if not bool(
            loop.get(
                "reached_max_refinement_bound",
                False,
            )
        ):
            raise RuntimeError(
                "Case did not actually reach frozen bound."
            )

        bound.append(r)

    if len(bound) != 59:
        raise RuntimeError(
            f"Expected 59 bound cases; found {len(bound)}"
        )

    return bound


def load_completed(path):
    if not path.exists():
        return set()

    keys = set()

    with open(
        path,
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            r = json.loads(line)

            keys.add(
                key_of(
                    r["PAT_ID"],
                    r["NOTE_ID"],
                    r["feature"],
                )
            )

    return keys


def append_jsonl(path, record):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


def extend_case(record, note_text):
    loop = record["loop_info"]

    current_claim = dict(
        loop["final_claim"]
    )

    current_verification = dict(
        loop["final_verification"]
    )

    initial_bound5_label = str(
        current_claim.get(
            "prediction",
            "",
        )
    ).strip()

    if initial_bound5_label not in {
        "Yes",
        "No",
    }:
        raise RuntimeError(
            "Invalid frozen bound-5 label."
        )

    label_probs = record.get(
        "extractor_label_probs",
        {},
    )

    extension_history = []
    added_rounds = 0
    flip_count = 0

    operational_failure = False
    label_stable = False
    unresolved_at_bound7 = False
    final_verification_applies = True

    exit_reason = ""
    final_refinement = {}

    for round_num in (6, 7):
        added_rounds += 1

        entering_claim = dict(
            current_claim
        )

        entering_verification = dict(
            current_verification
        )

        entering_label = str(
            current_claim.get(
                "prediction",
                "",
            )
        ).strip()

        try:
            refinement = (
                agent4.refine_claim(
                    structured_claim=current_claim,
                    verification=current_verification,
                    note_text=note_text,
                    label_probs=label_probs,
                )
            )

        except Exception as error:
            refinement = (
                controller.make_refinement_error(
                    current_claim,
                    current_verification,
                    (
                        "Bound-7 sensitivity "
                        f"Agent 4 exception: {error}"
                    ),
                    label_probs=label_probs,
                )
            )

        final_refinement = dict(
            refinement
        )

        ref_error = (
            controller
            .refinement_operational_error(
                refinement
            )
        )

        operational_failure |= ref_error

        round_record = {
            "round": round_num,
            "entering_label":
                entering_label,
            "entering_claim":
                json_safe(
                    entering_claim
                ),
            "pre_refinement_verification":
                json_safe(
                    entering_verification
                ),
            "refinement":
                json_safe(
                    refinement
                ),
            "refinement_operational_error":
                bool(ref_error),
            "post_refinement_verification":
                None,
        }

        if ref_error:
            round_record[
                "exiting_label"
            ] = entering_label

            round_record[
                "label_changed"
            ] = False

            extension_history.append(
                round_record
            )

            exit_reason = (
                "BOUND7_REFINEMENT_ERROR"
            )

            final_verification_applies = True
            break

        corrected_label = str(
            refinement.get(
                "corrected_prediction",
                "",
            )
        ).strip()

        if corrected_label not in {
            "Yes",
            "No",
        }:
            operational_failure = True

            round_record[
                "exiting_label"
            ] = entering_label

            round_record[
                "label_changed"
            ] = False

            round_record[
                "error"
            ] = (
                "Invalid corrected_prediction"
            )

            extension_history.append(
                round_record
            )

            exit_reason = (
                "BOUND7_REFINEMENT_ERROR"
            )

            final_verification_applies = True
            break

        promoted_claim = (
            controller.build_refined_claim(
                current_claim,
                refinement,
            )
        )

        label_changed = (
            corrected_label
            != entering_label
        )

        current_claim = dict(
            promoted_claim
        )

        if label_changed:
            flip_count += 1

        round_record[
            "exiting_label"
        ] = corrected_label

        round_record[
            "label_changed"
        ] = bool(label_changed)

        if not label_changed:
            label_stable = True

            extension_history.append(
                round_record
            )

            exit_reason = (
                f"BOUND7_LABEL_STABLE_R{round_num}"
            )

            final_verification_applies = False
            break

        try:
            post_verification = (
                agent3.verify_claim(
                    structured_claim=current_claim,
                    note_text=note_text,
                    skip_bertscore=False,
                    label_probs=label_probs,
                )
            )

        except Exception as error:
            post_verification = (
                controller.make_verification_error(
                    current_claim,
                    (
                        "Bound-7 sensitivity "
                        f"Agent 3 exception: {error}"
                    ),
                    label_probs=label_probs,
                )
            )

        post_error = (
            controller
            .verification_operational_error(
                post_verification
            )
        )

        operational_failure |= post_error

        current_verification = dict(
            post_verification
        )

        round_record[
            "post_refinement_verification"
        ] = json_safe(
            post_verification
        )

        round_record[
            "verification_operational_error"
        ] = bool(post_error)

        extension_history.append(
            round_record
        )

        verdict = str(
            post_verification.get(
                "verdict",
                "",
            )
        ).strip().upper()

        if verdict == "VERIFIED":
            exit_reason = (
                f"BOUND7_VERIFIED_R{round_num}"
            )

            final_verification_applies = True
            break

        if verdict != "REFINE":
            operational_failure = True

            exit_reason = (
                "BOUND7_INVALID_VERIFIER_STATE"
            )

            final_verification_applies = True
            break

        if round_num == 7:
            unresolved_at_bound7 = True

            if post_error:
                exit_reason = (
                    "BOUND7_FINAL_VERIFICATION_ERROR"
                )
            else:
                exit_reason = (
                    "BOUND7_LABEL_OSCILLATION"
                )

            final_verification_applies = True
            break

    final_label = str(
        current_claim.get(
            "prediction",
            "",
        )
    ).strip()

    if final_label not in {
        "Yes",
        "No",
    }:
        raise RuntimeError(
            "Invalid final bound-7 label."
        )

    return {
        "bound5_prediction":
            initial_bound5_label,

        "bound7_prediction":
            final_label,

        "label_changed_vs_bound5":
            final_label
            != initial_bound5_label,

        "added_refinement_rounds":
            added_rounds,

        "total_refinement_rounds":
            5 + added_rounds,

        "additional_flip_count":
            flip_count,

        "label_stable":
            label_stable,

        "unresolved_at_bound7":
            unresolved_at_bound7,

        "operational_failure":
            operational_failure,

        "exit_reason":
            exit_reason,

        "final_verification_applies_to_final_claim":
            final_verification_applies,

        "final_claim":
            json_safe(
                current_claim
            ),

        "final_verification":
            json_safe(
                current_verification
            ),

        "final_refinement":
            json_safe(
                final_refinement
            ),

        "extension_history":
            extension_history,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-jsonl",
        required=True,
    )

    parser.add_argument(
        "--output-csv",
        required=True,
    )

    args = parser.parse_args()

    output_jsonl = Path(
        args.output_jsonl
    )

    output_csv = Path(
        args.output_csv
    )

    notes = load_notes()
    bound = load_bound_cases()

    completed = load_completed(
        output_jsonl
    )

    print("=" * 84)
    print("VERGE ROUND-7 SENSITIVITY")
    print("=" * 84)
    print("Frozen bound-5 cases:", len(bound))
    print("Already completed:", len(completed))
    print("Rounds replayed:", "NONE")
    print("New rounds allowed:", "6 and 7 only")
    print("Full-note policy:", "YES")
    print("BERTScore:", "ENABLED")
    print("Primary pipeline modified:", "NO")
    print("=" * 84)

    for i, record in enumerate(
        bound,
        start=1,
    ):
        key = key_of(
            record["PAT_ID"],
            record["NOTE_ID"],
            record["feature"],
        )

        if key in completed:
            continue

        note_key = (
            norm_id(
                record["PAT_ID"]
            ),
            norm_id(
                record["NOTE_ID"]
            ),
        )

        if note_key not in notes:
            raise RuntimeError(
                "Source note unavailable."
            )

        start = time.time()

        extension = extend_case(
            record,
            notes[note_key],
        )

        runtime = time.time() - start

        out = {
            "_run_row_index":
                record["_run_row_index"],

            "PAT_ID":
                record["PAT_ID"],

            "NOTE_ID":
                record["NOTE_ID"],

            "feature":
                record["feature"],

            **extension,

            "runtime_seconds":
                runtime,
        }

        append_jsonl(
            output_jsonl,
            out,
        )

        print(
            f"[{i:02d}/59] "
            f"{record['feature']} "
            f"B5={extension['bound5_prediction']} "
            f"B7={extension['bound7_prediction']} "
            f"added={extension['added_refinement_rounds']} "
            f"exit={extension['exit_reason']} "
            f"{runtime:.1f}s",
            flush=True,
        )

    # Build compact summary CSV from checkpoint.
    rows = []

    with open(
        output_jsonl,
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)

            rows.append(
                {
                    "_run_row_index":
                        r["_run_row_index"],

                    "PAT_ID":
                        r["PAT_ID"],

                    "NOTE_ID":
                        r["NOTE_ID"],

                    "feature":
                        r["feature"],

                    "bound5_prediction":
                        r["bound5_prediction"],

                    "bound7_prediction":
                        r["bound7_prediction"],

                    "label_changed_vs_bound5":
                        r[
                            "label_changed_vs_bound5"
                        ],

                    "added_refinement_rounds":
                        r[
                            "added_refinement_rounds"
                        ],

                    "total_refinement_rounds":
                        r[
                            "total_refinement_rounds"
                        ],

                    "additional_flip_count":
                        r[
                            "additional_flip_count"
                        ],

                    "label_stable":
                        r["label_stable"],

                    "unresolved_at_bound7":
                        r[
                            "unresolved_at_bound7"
                        ],

                    "operational_failure":
                        r[
                            "operational_failure"
                        ],

                    "exit_reason":
                        r["exit_reason"],

                    "runtime_seconds":
                        r["runtime_seconds"],
                }
            )

    summary = pd.DataFrame(
        rows
    )

    if len(summary) != 59:
        raise RuntimeError(
            f"Expected 59 completed cases; "
            f"found {len(summary)}"
        )

    if summary.duplicated(
        [
            "PAT_ID",
            "NOTE_ID",
            "feature",
        ]
    ).any():
        raise RuntimeError(
            "Duplicate bound-7 cases detected."
        )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        output_csv,
        index=False,
    )

    print("=" * 84)
    print("BOUND-7 EXTENSION COMPLETE")
    print("Cases:", len(summary))
    print(
        "Changed vs bound 5:",
        int(
            summary[
                "label_changed_vs_bound5"
            ].sum()
        ),
    )
    print("\nExit reasons:")
    print(
        summary[
            "exit_reason"
        ].value_counts().to_string()
    )
    print("=" * 84)


if __name__ == "__main__":
    main()
