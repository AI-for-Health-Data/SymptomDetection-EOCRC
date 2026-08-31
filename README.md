
# VERGE: Verification-Enhanced Refinement for Grounded Extraction of Early-Onset Colorectal Cancer Symptoms in Clinical Notes

VERGE is a four-agent clinical NLP framework for extracting evidence-grounded early-onset colorectal cancer (EOCRC) symptoms from unstructured clinical notes.

The framework targets seven predefined symptoms:

1. Abdominal pain
2. Rectal bleeding
3. Rectal pain
4. Diarrhea
5. Constipation
6. Weight loss
7. Family history of colorectal cancer

VERGE separates initial extraction from claim standardization, evidence verification, and targeted refinement. A deterministic controller manages a bounded Verifier–Refiner loop and flags unresolved claims for human review.

---

## Architecture

<p align="center">
  <img src="figures/workflow.png"
       alt="VERGE four-agent workflow"
       width="100%">
</p>

**VERGE workflow.** The Extractor produces evidence-linked predictions for all seven symptoms, the Claim Composer converts them into standardized claim records, and the Verifier evaluates each claim against the clinical note. Claims requiring correction are sent to the Refiner. Label-changing refinements return to the Verifier for re-verification. The loop is bounded at five refinement rounds, after which unresolved claims are flagged for human review.

The primary VERGE prediction reported in the paper is the output of this bounded four-agent workflow.

---

## Method

### 1. Extractor

**Code:** [`verge/extractor.py`](verge/extractor.py)

The Extractor jointly evaluates all seven target symptoms in one language-model generation.

Before generation, symptom-specific biomedical context is constructed using:

- manually defined clinical aliases;
- UMLS-derived terminology;
- MedCPT retrieval;
- up to three query variants per symptom: **base**, **note-anchored**, and **ontology-expanded**;
- top-32 retrieval per query;
- within-symptom Reciprocal Rank Fusion (RRF; \(k=60\));
- symptom-specific phrase screening;
- evidence-quality/source-design prioritization;
- at most two retrieved passages per symptom in the final prompt.

The evidence-quality layer uses GRADE-informed evidence prioritization. When a valid provenance mapping is available, a retrieved passage inherits the GRADE certainty rating of the corresponding evidence body. Eligible mapped passages are prioritized by certainty, with RRF score used to break ties.

Retrieved literature is used only as terminology and background context. It cannot establish, negate, or override a patient-specific label. Patient-specific positive evidence must be supported by the clinical note.

For each symptom, the Extractor returns a structured prediction containing the binary label, confidence, evidence, note section, experiencer, and assertion status.

---

### 2. Claim Composer

**Code:** [`verge/claim_composer.py`](verge/claim_composer.py)

The Claim Composer converts the Extractor's joint output into one standardized claim record per symptom.

It does not independently review the clinical note, introduce new evidence, or change the Extractor label. Its purpose is to provide a common structured representation for downstream verification.

Each claim contains fields such as:

- symptom;
- prediction;
- confidence;
- evidence quote;
- note section;
- experiencer;
- assertion status;
- family relation and cancer type when applicable.

---

### 3. Verifier

**Code:** [`verge/verifier.py`](verge/verifier.py)

**Shared evidence rules:** [`verge/clinical_evidence_policy.py`](verge/clinical_evidence_policy.py)

The Verifier checks each standardized claim against the clinical note using two complementary layers.

**Textual grounding** evaluates whether the cited evidence can be recovered from the note using:

- exact matching;
- normalized exact matching;
- contiguous token-subsequence matching;
- ROUGE-L;
- modified BLEU without the standard brevity penalty;
- BERTScore precision.

Grounding is represented as `STRONG`, `MODERATE`, `WEAK`, `NONE`, or `NOT_APPLICABLE`.

**Clinical validation** evaluates whether grounded text actually supports the requested symptom. This includes source faithfulness, negation, experiencer, temporality, direct clinical support, and family-history specificity.

For negative claims, the Verifier also searches for missed grounded affirmative evidence before accepting the negative prediction.

The Verifier deterministically routes each claim to:

- `VERIFIED`, or
- `REFINE`.

---

### 4. Refiner

**Code:** [`verge/refiner.py`](verge/refiner.py)

The Refiner receives claims routed to `REFINE` and re-examines the clinical note to correct the claim.

Refinement may include:

- replacing unsupported evidence;
- correcting assertion or contextual metadata;
- changing a positive claim to negative when valid positive support cannot be established;
- recovering a missed positive claim when acceptable affirmative evidence is present;
- enforcing biological-relation and colorectal-cancer specificity for family history.

Corrected positive evidence must satisfy the configured grounding requirements.

---

## Bounded Verifier–Refiner Loop

The Verifier and Refiner operate under a deterministic bounded controller.

```text
Extractor
    ↓
Claim Composer
    ↓
Verifier
    ├── VERIFIED ───────────────→ VERGE label
    │
    └── REFINE
          ↓
       Refiner
          │
          ├── label unchanged ──→ VERGE label
          │
          └── label changed
                  ↓
              Verifier
                  ↺
```

The controller:

- allows at most five refinement rounds;
- returns label-changing refinements to the Verifier;
- permits valid label-stable corrections to exit the loop;
- records loop state and exit status;
- flags unresolved oscillations and operational failures for human review.

The exact historical execution controller is preserved at:

[`verge/bounded_loop_controller.py`](verge/bounded_loop_controller.py)

---

## Main Code Map

| Paper component | Implementation |
| --- | --- |
| Agent 1 — Extractor | [`verge/extractor.py`](verge/extractor.py) |
| Agent 2 — Claim Composer | [`verge/claim_composer.py`](verge/claim_composer.py) |
| Agent 3 — Verifier | [`verge/verifier.py`](verge/verifier.py) |
| Agent 4 — Refiner | [`verge/refiner.py`](verge/refiner.py) |
| Shared evidence policy | [`verge/clinical_evidence_policy.py`](verge/clinical_evidence_policy.py) |
| Shared utilities | [`verge/utilities.py`](verge/utilities.py) |
| Executed bounded-loop controller | [`verge/bounded_loop_controller.py`](verge/bounded_loop_controller.py) |

---

## Baselines

The repository includes the comparison systems reported in the paper.

| Baseline | Code |
| --- | --- |
| Note-Only | [`baselines/note_only_baseline.py`](baselines/note_only_baseline.py) |
| Review-of-Systems Rules | [`baselines/ros_baseline.py`](baselines/ros_baseline.py) |
| ROS + UMLS | [`baselines/ros_umls_baseline.py`](baselines/ros_umls_baseline.py) |
| MedCPT + RRF RAG | [`baselines/medcpt_rag_baseline.py`](baselines/medcpt_rag_baseline.py) |
| medspaCy + ConText | [`baselines/medspacy_context_baseline.py`](baselines/medspacy_context_baseline.py) |

The final VERGE Extractor is provided separately under [`verge/`](verge/) and should not be confused with the MedCPT-RAG baseline.

---

## Sensitivity Analyses

### Loop-Bound Sensitivity

**Code:** [`sensitivity/loop_bound_sensitivity.py`](sensitivity/loop_bound_sensitivity.py)

This developmental analysis evaluates the stability of the bounded Verifier–Refiner workflow under alternative refinement limits, including 3-, 5-, and 7-round configurations. The primary VERGE workflow reported in the paper uses a maximum of five refinement rounds.

### Cross-Model Sensitivity

The Verifier–Refiner subsystem was additionally evaluated using Ministral to assess whether the verification and refinement framework transfers across language-model backbones.

Code:

- [`sensitivity/cross_model/ministral_backend.py`](sensitivity/cross_model/ministral_backend.py)
- [`sensitivity/cross_model/ministral_verifier_refiner.py`](sensitivity/cross_model/ministral_verifier_refiner.py)
- [`sensitivity/cross_model/score_ministral_results.py`](sensitivity/cross_model/score_ministral_results.py)

This analysis replaces the Verifier–Refiner language-model backbone with `mistralai/Ministral-3-14B-Instruct-2512-BF16` while retaining the frozen Extractor output. It evaluates the portability of the verification and refinement subsystem rather than an end-to-end replacement of the primary VERGE pipeline.
---

## Models

The primary implementation uses:

- **Extractor / language-model roles:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Biomedical retrieval:** `ncbi/MedCPT-Query-Encoder` and `ncbi/MedCPT-Article-Encoder`
- **BERTScore:** `roberta-large`

The cross-model sensitivity analysis uses:

- `mistralai/Ministral-3-14B-Instruct-2512-BF16`

Primary language-model inference uses greedy decoding.

---

## Repository Structure

```text
VERGE/
├── figures/
│   └── verge_workflow.png
├── verge/
│   ├── agent1_extractor.py
│   ├── agent2_claim_structurer.py
│   ├── agent3_unified_verifier.py
│   ├── agent4_refiner.py
│   ├── clinical_evidence_policy.py
│   └── sage_common.py
├── baselines/
├── ablations/
├── sensitivity/
```

This release focuses on the **method implementation, comparison baselines, ablations, and sensitivity analyses**. Internal result-assembly scripts and development history are intentionally excluded to keep the repository concise.

---

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

Model weights are not included in this repository and must be obtained from their respective providers.

UMLS resources are also not redistributed and must be obtained separately under the applicable UMLS license.

---

## Data

The clinical notes and patient-level study data are not distributed with this repository.

The repository does not contain:

- clinical note text;
- patient identifiers;
- clinician reference-label files;
- patient-level model outputs.

Users wishing to reproduce the complete clinical evaluation must have authorized access to the corresponding source data and required terminology resources.

---

## Citation

If you use VERGE, please cite:

> **VERGE: Verification-Enhanced Refinement for Grounded Extraction of EOCRC Symptoms via Constrained Agentic Verification**

Full citation information will be added after publication.

---
