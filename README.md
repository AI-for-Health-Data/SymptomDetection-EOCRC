**VERGE: Verification-Enhanced Refinement for Grounded Extraction of EOCRC Symptoms via Constrained Agentic Verification**

VERGE is a four-agent framework for evidence-grounded extraction of
early-onset colorectal cancer (EOCRC) symptoms from clinical notes.

The system extracts seven predefined targets:

1. Abdominal pain
2. Rectal bleeding
3. Rectal pain
4. Diarrhea
5. Constipation
6. Weight loss
7. Family history of colorectal cancer

---

## Architecture

VERGE contains four role-separated agents:

**Agent 1 — Extractor**  
Jointly predicts all seven targets from the full clinical note. The Extractor
uses manually defined terminology, UMLS-derived terminology, MedCPT retrieval,
within-symptom Reciprocal Rank Fusion, and GRADE-based evidence prioritization.
Retrieved biomedical literature provides terminology and background context
only; patient-specific labels must be supported by the clinical note.

→ **Code:** [`verge/agent1_extractor.py`](verge/agent1_extractor.py)

**Agent 2 — Claim Composer**  
Converts the joint Extractor output into one standardized claim record per
target. It does not search the clinical note for new evidence and does not
change the Extractor label.

→ **Code:** [`verge/agent2_claim_composer.py`](verge/agent2_claim_composer.py)

**Agent 3 — Verifier**  
Checks each standardized claim against the full clinical note using
hierarchical textual grounding and LLM-based clinical validation. Claims are
assigned either `VERIFIED` or `REFINE`.

→ **Code:** [`verge/agent3_verifier.py`](verge/agent3_verifier.py)

**Agent 4 — Refiner**  
Corrects claims assigned `REFINE`. Label-changing corrections are returned to
the Verifier for re-verification. The Verifier–Refiner loop is bounded to five
refinement rounds. Unresolved or operationally failed claims are flagged for
human review.

→ **Code:** [`verge/agent4_refiner.py`](verge/agent4_refiner.py)

The deterministic evidence rules shared by the Verifier and Refiner are
implemented here:

→ **Evidence policy:** [`verge/clinical_evidence_policy.py`](verge/clinical_evidence_policy.py)

Shared model loading, generation, schema, terminology, and utility functions
are implemented here:

→ **Shared utilities:** [`verge/common.py`](verge/common.py)

The primary VERGE prediction reported in the paper is the output of the
bounded four-agent loop, stored as `final_loop_prediction`.

---

# Paper-to-Code Guide

This section maps each experiment, system component, and analysis reported in
the paper to the corresponding source code.

## 1. Baseline and comparison systems

| Paper configuration | Description | Code |
| --- | --- | --- |
| **Note-Only** | LLM extraction from the clinical note without terminology or retrieval augmentation | [`baselines/note_only.py`](baselines/note_only.py) |
| **Review-of-Systems Rules** | Adds rules for conflicts between templated ROS negatives and narrative documentation | [`baselines/ros_rules.py`](baselines/ros_rules.py) |
| **ROS + Manual + UMLS** | Adds manually curated and UMLS-derived terminology | [`baselines/ros_manual_umls.py`](baselines/ros_manual_umls.py) |
| **RAG** | MedCPT dense retrieval with within-symptom Reciprocal Rank Fusion | [`baselines/medcpt_rag.py`](baselines/medcpt_rag.py) |
| **medspaCy + ConText** | Rule-based clinical NLP comparison using medspaCy/ConText | [`baselines/medspacy_context.py`](baselines/medspacy_context.py) |
| **Extractor** | Final Agent 1 with MedCPT, RRF, and GRADE-based evidence prioritization | [`verge/agent1_extractor.py`](verge/agent1_extractor.py) |
| **VERGE** | Final four-agent Extractor → Composer → Verifier ↔ Refiner system | [`verge/`](verge/) |

### Original experiment correspondence

The paper-facing filenames above are descriptive names. Their relationship to
the original experiment files is:

| Paper configuration | Original experiment | Paper-facing file |
| --- | --- | --- |
| Note-Only | Experiment 1 | [`baselines/note_only.py`](baselines/note_only.py) |
| Review-of-Systems Rules | Experiment 3 | [`baselines/ros_rules.py`](baselines/ros_rules.py) |
| ROS + Manual + UMLS | Experiment 5 | [`baselines/ros_manual_umls.py`](baselines/ros_manual_umls.py) |
| MedCPT-RRF RAG | Experiment 7 RAG configuration | [`baselines/medcpt_rag.py`](baselines/medcpt_rag.py) |
| Final Extractor | Formal-GRADE Experiment 7 | [`verge/agent1_extractor.py`](verge/agent1_extractor.py) |

---

## 2. Agent 1 — Extractor

The final Extractor jointly predicts all seven targets in a single
note-level generation.

### Main Extractor

→ [`verge/agent1_extractor.py`](verge/agent1_extractor.py)

This file implements:

- manual target aliases;
- UMLS-derived terminology;
- Review-of-Systems conflict handling;
- MedCPT dense retrieval;
- up to three retrieval queries per target;
- within-target Reciprocal Rank Fusion;
- symptom-specific passage screening;
- GRADE-based evidence prioritization;
- final prompt construction;
- structured seven-target generation;
- generation-time label-probability auditing;
- output parsing and repair.

### Formal GRADE support

GRADE loading and validation:

→ [`grade/formal_grade.py`](grade/formal_grade.py)

Build the literature/GRADE corpus:

→ [`grade/build_formal_grade_corpus.py`](grade/build_formal_grade_corpus.py)

Validate that GRADE resources are ready:

→ [`grade/check_grade_ready.py`](grade/check_grade_ready.py)

Repair structurally incomplete Extractor outputs:

→ [`grade/repair_exp7_schema_outputs.py`](grade/repair_exp7_schema_outputs.py)

The final GRADE mappings used by the Extractor are:

→ [`grade/grade_evidence_profiles.csv`](grade/grade_evidence_profiles.csv)  
→ [`grade/grade_study_map.csv`](grade/grade_study_map.csv)  
→ [`grade/grade_chunk_map.csv`](grade/grade_chunk_map.csv)

---

## 3. Agent 2 — Claim Composer

The Claim Composer converts the joint seven-target Extractor output into one
standardized record for each target.

→ [`verge/agent2_claim_composer.py`](verge/agent2_claim_composer.py)

It standardizes:

- target name;
- prediction;
- confidence;
- evidence quote;
- note section;
- experiencer;
- assertion status;
- family relation when applicable;
- cancer type when applicable.

The Claim Composer does not independently inspect the clinical note and does
not introduce new patient evidence.

---

## 4. Agent 3 — Verifier

The Verifier evaluates every standardized claim against the full clinical note.

→ [`verge/agent3_verifier.py`](verge/agent3_verifier.py)

### Textual grounding

The Verifier uses:

- exact substring matching;
- normalized exact matching;
- contiguous token-subsequence matching;
- ROUGE-L;
- modified BLEU without brevity penalty;
- BERTScore precision.

Grounding is categorized as:

`STRONG`, `MODERATE`, `WEAK`, `NONE`, or `NOT_APPLICABLE`.

### Clinical validation

For positive claims, the Verifier evaluates:

- source faithfulness;
- contextual validity;
- direct target support;
- biological family-relation specificity when applicable;
- colorectal-cancer specificity when applicable.

For negative claims, it searches the full note for missed grounded affirmative
evidence.

The final routing decision is deterministic:

`VERIFIED` or `REFINE`.

Shared deterministic evidence checks are defined in:

→ [`verge/clinical_evidence_policy.py`](verge/clinical_evidence_policy.py)

---

## 5. Agent 4 — Refiner

Claims assigned `REFINE` are passed to:

→ [`verge/agent4_refiner.py`](verge/agent4_refiner.py)

The Refiner receives:

- the current standardized claim;
- the specific Verifier issues;
- the full clinical note.

It can correct:

- the Yes/No prediction;
- evidence;
- note section;
- experiencer;
- assertion status;
- family relation;
- cancer type.

A corrected positive claim must contain grounded note evidence. Family-history
positives must additionally satisfy relation and colorectal-cancer specificity
requirements.

The same frozen evidence policy is shared with the Verifier:

→ [`verge/clinical_evidence_policy.py`](verge/clinical_evidence_policy.py)

---

## 6. Bounded Verifier–Refiner loop

The primary VERGE output is produced by the bounded Agent 3 ↔ Agent 4 loop.

The loop follows these rules:

1. Agent 3 verifies the claim.
2. `VERIFIED` claims exit.
3. `REFINE` claims are sent to Agent 4.
4. Label-stable valid corrections exit.
5. Label-changing corrections return to Agent 3 for re-verification.
6. The process continues for at most five refinement rounds.
7. Unresolved claims or operational failures are flagged for human review.

The exact controller executed for the reported experiments is preserved here:

→ [`reproducibility/executed_pipeline/run_verge_continuation_FINAL_executed.py`](reproducibility/executed_pipeline/run_verge_continuation_FINAL_executed.py)

**Important:** this historical controller also executed a developmental Agent 5
after the bounded loop. The primary VERGE prediction used in the paper is
`final_loop_prediction`, before the Agent-5 output.

The script that explicitly constructs the paper's primary four-agent result is:

→ [`analysis/finalize_primary_results.py`](analysis/finalize_primary_results.py)

---

## 7. Primary performance analysis

Primary Extractor and VERGE metrics, label changes, and the final
paper-facing dataset are generated by:

→ [`analysis/finalize_primary_results.py`](analysis/finalize_primary_results.py)

This analysis uses the bounded-loop output as the VERGE prediction.

Reported performance on 4,033 labeled note–target pairs:

| System | Precision | Recall | F1 | Balanced accuracy | MCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Extractor | 0.764 | 0.705 | 0.733 | 0.830 | 0.681 |
| VERGE | **0.849** | 0.702 | **0.769** | **0.838** | **0.730** |

VERGE reduced false positives from 153 to 88 while retaining 493/495
Extractor true positives.

---

## 8. Verifier–Refiner process analysis

Routing counts, refinement behavior, loop exits, and human-review flags are
analyzed with:

→ [`analysis/process_analysis.py`](analysis/process_analysis.py)

This includes analyses of:

- claims entering refinement;
- `VERIFIED` exits;
- label-stable exits;
- five-round unresolved cases;
- operational refinement errors;
- human-review flags.

---

## 9. Evidence-faithfulness analysis

Changes in note-evidence correspondence before and after VERGE are analyzed
with:

→ [`analysis/evidence_faithfulness.py`](analysis/evidence_faithfulness.py)

This includes:

- usable evidence coverage;
- exact/normalized note correspondence;
- modified BLEU;
- BERTScore precision;
- paired before-versus-after comparisons.

---

## 10. Additional statistical analyses

Additional paper analyses are implemented in:

→ [`analysis/remaining_analyses.py`](analysis/remaining_analyses.py)

Final consistency and manuscript-number checks are implemented in:

→ [`analysis/final_checks.py`](analysis/final_checks.py)

These analyses cover the additional reported process, label-change,
human-review, uncertainty, and sensitivity results.

---

## 11. Cross-model sensitivity analysis

The Verifier–Refiner subsystem was also evaluated using
`mistralai/Ministral-3-14B-Instruct-2512-BF16` while upstream Extractor and
Claim Composer outputs were held fixed.

Build the evaluation subset:

→ [`analysis/cross_model/build_ministral_subset.py`](analysis/cross_model/build_ministral_subset.py)

Run the Ministral Verifier–Refiner experiment:

→ [`analysis/cross_model/run_ministral_verifier_refiner.py`](analysis/cross_model/run_ministral_verifier_refiner.py)

Score and compare the results:

→ [`analysis/cross_model/score_ministral.py`](analysis/cross_model/score_ministral.py)

Optional model-download helper:

→ [`analysis/cross_model/download_ministral3_14b.py`](analysis/cross_model/download_ministral3_14b.py)

This experiment evaluates portability of the Verifier–Refiner subsystem rather
than a new end-to-end VERGE model.

---

## 12. Developmental Agent-5 ablations

Agent 5 is **not part of the primary VERGE architecture**.

Developmental Agent-5 experiments are retained for transparency under:

→ [`analysis/ablations/agent5/`](analysis/ablations/agent5/)

Agent-5 implementation:

→ [`analysis/ablations/agent5/agent5_verifier_judge.py`](analysis/ablations/agent5/agent5_verifier_judge.py)

Ontology/entailment variant:

→ [`analysis/ablations/agent5/evaluate_agent5_ontology_entailment.py`](analysis/ablations/agent5/evaluate_agent5_ontology_entailment.py)

Other VERGE ablation code:

→ [`analysis/ablations/run_verge_ablation.py`](analysis/ablations/run_verge_ablation.py)

These experiments were developmental sensitivity analyses and were not used to
define the primary VERGE prediction.

---

## Repository structure

```text
VERGE/
├── README.md
├── requirements.txt
├── LICENSE
│
├── verge/
│   ├── agent1_extractor.py
│   ├── agent2_claim_composer.py
│   ├── agent3_verifier.py
│   ├── agent4_refiner.py
│   ├── clinical_evidence_policy.py
│   └── common.py
│
├── grade/
│   ├── formal_grade.py
│   ├── build_formal_grade_corpus.py
│   ├── check_grade_ready.py
│   ├── repair_exp7_schema_outputs.py
│   ├── grade_evidence_profiles.csv
│   ├── grade_study_map.csv
│   └── grade_chunk_map.csv
│
├── baselines/
│   ├── note_only.py
│   ├── ros_rules.py
│   ├── ros_manual_umls.py
│   ├── medcpt_rag.py
│   └── medspacy_context.py
│
├── analysis/
│   ├── finalize_primary_results.py
│   ├── process_analysis.py
│   ├── evidence_faithfulness.py
│   ├── remaining_analyses.py
│   ├── final_checks.py
│   │
│   ├── cross_model/
│   │   ├── build_ministral_subset.py
│   │   ├── run_ministral_verifier_refiner.py
│   │   ├── score_ministral.py
│   │   └── download_ministral3_14b.py
│   │
│   └── ablations/
│       ├── run_verge_ablation.py
│       └── agent5/
│           ├── agent5_verifier_judge.py
│           └── evaluate_agent5_ontology_entailment.py
│
└── reproducibility/
    ├── executed_pipeline/
    │   └── run_verge_continuation_FINAL_executed.py
    ├── FROZEN_PRIMARY_FILES.txt
    └── SHA256SUMS_PYTHON.txt
