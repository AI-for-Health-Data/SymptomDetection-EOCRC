from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

CERTAINTY_ORDER: Dict[str, int] = {
    "VERY_LOW": 1,
    "LOW": 2,
    "MODERATE": 3,
    "HIGH": 4,
}
ORDER_TO_CERTAINTY = {value: key for key, value in CERTAINTY_ORDER.items()}

GRADE_POLICIES = {"annotate", "prioritize", "filter"}
GRADE_MODES = {"off", "optional", "required"}
GRADE_PROFILE_ORIGINS = {"external_published", "internal_formal"}
GRADE_FINAL_STATUSES = {"final", "finalized", "consensus_final", "published"}

GRADE_PROTOCOL_START = {
    "diagnostic_test_accuracy": "HIGH",
    "prognostic_factor": "HIGH",
}

DOWNGRADE_JUDGMENT_TO_LEVELS = {
    "not_serious": 0,
    "serious": 1,
    "very_serious": 2,
}
PUBLICATION_BIAS_TO_LEVELS = {
    "undetected": 0,
    "strongly_suspected": 1,
}
LARGE_EFFECT_TO_LEVELS = {
    "none": 0,
    "large": 1,
    "very_large": 2,
}
BINARY_UPGRADE_TO_LEVELS = {
    "absent": 0,
    "present": 1,
}

PROFILE_REQUIRED_COLUMNS = [
    "body_id",
    "finding",
    "grade_protocol",
    "profile_origin",
    "source_citation",
    "population",
    "index_factor_or_test",
    "reference_standard_or_comparator",
    "outcome",
    "final_certainty",
    "applicability_status",
    "applicability_rationale",
    "status",
]

INTERNAL_FORMAL_REQUIRED_COLUMNS = [
    "systematic_search_completed",
    "search_end_date",
    "certainty_target",
    "starting_certainty",
    "risk_of_bias_judgment",
    "risk_of_bias_downgrade",
    "risk_of_bias_rationale",
    "inconsistency_judgment",
    "inconsistency_downgrade",
    "inconsistency_rationale",
    "indirectness_judgment",
    "indirectness_downgrade",
    "indirectness_rationale",
    "imprecision_judgment",
    "imprecision_downgrade",
    "imprecision_rationale",
    "publication_bias_judgment",
    "publication_bias_downgrade",
    "publication_bias_rationale",
    "large_effect_judgment",
    "large_effect_upgrade",
    "large_effect_rationale",
    "dose_response_judgment",
    "dose_response_upgrade",
    "dose_response_rationale",
    "residual_confounding_judgment",
    "residual_confounding_upgrade",
    "residual_confounding_rationale",
]

STUDY_MAP_REQUIRED_COLUMNS = ["body_id", "study_id", "pmid", "include"]
CHUNK_MAP_REQUIRED_COLUMNS = ["chunk_idx", "study_id", "pmid"]


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _clean_upper(value: Any) -> str:
    return _clean_text(value).upper().replace(" ", "_").replace("-", "_")


def _clean_lower(value: Any) -> str:
    return _clean_text(value).lower().replace(" ", "_").replace("-", "_")


def _clean_bool(value: Any) -> bool:
    return _clean_lower(value) in {"1", "true", "yes", "y", "include", "included"}


def _require_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _require_nonblank(row: Mapping[str, Any], fields: Iterable[str], context: str) -> None:
    missing = [field for field in fields if not _clean_text(row.get(field))]
    if missing:
        raise ValueError(f"{context} has blank required fields: {missing}")


def _parse_nonnegative_int(value: Any, field_name: str, context: str) -> int:
    try:
        numeric = int(float(_clean_text(value)))
    except ValueError as exc:
        raise ValueError(f"{context}: {field_name} must be an integer") from exc
    if numeric < 0:
        raise ValueError(f"{context}: {field_name} must be >= 0")
    return numeric


def _clamp_certainty_score(score: int) -> int:
    return max(CERTAINTY_ORDER["VERY_LOW"], min(CERTAINTY_ORDER["HIGH"], score))


@dataclass(frozen=True)
class GradeBody:
    body_id: str
    finding: str
    grade_protocol: str
    profile_origin: str
    source_citation: str
    source_identifier: str
    source_url_or_doi: str
    final_certainty: str
    starting_certainty: str
    outcome: str
    population: str
    index_factor_or_test: str
    reference_standard_or_comparator: str
    applicability_status: str
    applicability_rationale: str
    evidence_review_id: str
    total_downgrade: Optional[int]
    total_upgrade: Optional[int]
    profile_row: Dict[str, Any]


@dataclass(frozen=True)
class GradeCandidate:
    mapped: bool
    eligible: bool
    body_id: str = ""
    body_ids: Tuple[str, ...] = ()
    body_certainties: Tuple[str, ...] = ()
    profile_origins: Tuple[str, ...] = ()
    source_citations: Tuple[str, ...] = ()
    study_id: str = ""
    pmid: str = ""
    final_certainty: str = "UNMAPPED"
    grade_protocol: str = ""
    outcome: str = ""
    reason: str = ""


def _validate_internal_formal_profile(
    row: Mapping[str, Any],
    protocol: str,
    context: str,
) -> Tuple[str, int, int]:
    _require_nonblank(row, INTERNAL_FORMAL_REQUIRED_COLUMNS, context)
    if not _clean_bool(row["systematic_search_completed"]):
        raise ValueError(
            f"{context}: internal_formal GRADE requires a completed prespecified "
            "evidence review; dynamic MedCPT retrieval cannot define the evidence body."
        )
    try:
        pd.to_datetime(_clean_text(row["search_end_date"]), errors="raise")
    except Exception as exc:
        raise ValueError(f"{context}: invalid search_end_date") from exc

    if protocol == "diagnostic_test_accuracy":
        if "decision_thresholds" in row and not _clean_text(row.get("decision_thresholds")):
            raise ValueError(
                f"{context}: diagnostic_test_accuracy profile must record decision_thresholds."
            )
    if protocol == "prognostic_factor" and "certainty_context" in row:
        certainty_context = _clean_lower(row.get("certainty_context"))
        if certainty_context and certainty_context not in {"contextualized", "noncontextualized"}:
            raise ValueError(
                f"{context}: certainty_context must be contextualized or noncontextualized."
            )

    start_certainty = _clean_upper(row["starting_certainty"])
    if start_certainty not in CERTAINTY_ORDER:
        raise ValueError(f"{context}: invalid starting_certainty {start_certainty!r}")
    expected_start = GRADE_PROTOCOL_START[protocol]
    if start_certainty != expected_start:
        raise ValueError(
            f"{context}: {protocol} requires starting certainty {expected_start}; "
            f"profile states {start_certainty}."
        )

    total_downgrade = 0
    for domain in ["risk_of_bias", "inconsistency", "indirectness", "imprecision"]:
        judgment = _clean_lower(row[f"{domain}_judgment"])
        if judgment not in DOWNGRADE_JUDGMENT_TO_LEVELS:
            raise ValueError(
                f"{context}: invalid {domain}_judgment {judgment!r}; expected one of "
                f"{sorted(DOWNGRADE_JUDGMENT_TO_LEVELS)}"
            )
        levels = _parse_nonnegative_int(
            row[f"{domain}_downgrade"], f"{domain}_downgrade", context
        )
        expected_levels = DOWNGRADE_JUDGMENT_TO_LEVELS[judgment]
        if levels != expected_levels:
            raise ValueError(
                f"{context}: {domain}_judgment={judgment!r} implies downgrade="
                f"{expected_levels}, but {levels} was supplied."
            )
        if not _clean_text(row[f"{domain}_rationale"]):
            raise ValueError(f"{context}: missing {domain}_rationale")
        total_downgrade += levels

    pub_judgment = _clean_lower(row["publication_bias_judgment"])
    if pub_judgment not in PUBLICATION_BIAS_TO_LEVELS:
        raise ValueError(
            f"{context}: invalid publication_bias_judgment {pub_judgment!r}; expected one of "
            f"{sorted(PUBLICATION_BIAS_TO_LEVELS)}"
        )
    pub_levels = _parse_nonnegative_int(
        row["publication_bias_downgrade"], "publication_bias_downgrade", context
    )
    if pub_levels != PUBLICATION_BIAS_TO_LEVELS[pub_judgment]:
        raise ValueError(
            f"{context}: publication_bias_judgment={pub_judgment!r} implies downgrade="
            f"{PUBLICATION_BIAS_TO_LEVELS[pub_judgment]}, but {pub_levels} was supplied."
        )
    if not _clean_text(row["publication_bias_rationale"]):
        raise ValueError(f"{context}: missing publication_bias_rationale")
    total_downgrade += pub_levels

    large_judgment = _clean_lower(row["large_effect_judgment"])
    if large_judgment not in LARGE_EFFECT_TO_LEVELS:
        raise ValueError(
            f"{context}: invalid large_effect_judgment {large_judgment!r}; expected one of "
            f"{sorted(LARGE_EFFECT_TO_LEVELS)}"
        )
    large_levels = _parse_nonnegative_int(
        row["large_effect_upgrade"], "large_effect_upgrade", context
    )
    if large_levels != LARGE_EFFECT_TO_LEVELS[large_judgment]:
        raise ValueError(
            f"{context}: large_effect_judgment={large_judgment!r} implies upgrade="
            f"{LARGE_EFFECT_TO_LEVELS[large_judgment]}, but {large_levels} was supplied."
        )
    if not _clean_text(row["large_effect_rationale"]):
        raise ValueError(f"{context}: missing large_effect_rationale")

    total_upgrade = large_levels
    for domain in ["dose_response", "residual_confounding"]:
        judgment = _clean_lower(row[f"{domain}_judgment"])
        if judgment not in BINARY_UPGRADE_TO_LEVELS:
            raise ValueError(
                f"{context}: invalid {domain}_judgment {judgment!r}; expected one of "
                f"{sorted(BINARY_UPGRADE_TO_LEVELS)}"
            )
        levels = _parse_nonnegative_int(
            row[f"{domain}_upgrade"], f"{domain}_upgrade", context
        )
        if judgment == "absent" and levels != 0:
            raise ValueError(f"{context}: {domain}_judgment='absent' requires upgrade=0.")
        if judgment == "present" and levels not in {1, 2}:
            raise ValueError(
                f"{context}: {domain}_judgment='present' requires an explicit "
                "reviewer-assigned upgrade of 1 or 2 levels."
            )
        if not _clean_text(row[f"{domain}_rationale"]):
            raise ValueError(f"{context}: missing {domain}_rationale")
        total_upgrade += levels

    return start_certainty, total_downgrade, total_upgrade


class FormalGradeIndex:

    def __init__(
        self,
        bodies: Mapping[str, GradeBody],
        study_to_bodies: Mapping[str, Sequence[str]],
        pmid_to_studies: Mapping[str, Sequence[str]],
        chunk_to_study: Mapping[int, str],
        chunk_to_pmid: Mapping[int, str],
        policy: str,
        minimum_certainty: str,
        mode: str = "optional",
        source_paths: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.bodies = dict(bodies)
        self.study_to_bodies = {key: tuple(value) for key, value in study_to_bodies.items()}
        self.pmid_to_studies = {key: tuple(value) for key, value in pmid_to_studies.items()}
        self.chunk_to_study = dict(chunk_to_study)
        self.chunk_to_pmid = dict(chunk_to_pmid)
        self.policy = _clean_lower(policy)
        self.minimum_certainty = _clean_upper(minimum_certainty)
        self.mode = _clean_lower(mode)
        self.source_paths = dict(source_paths or {})

    @property
    def active(self) -> bool:
        return bool(self.bodies)

    @classmethod
    def empty(
        cls,
        policy: str = "prioritize",
        minimum_certainty: str = "VERY_LOW",
        mode: str = "optional",
    ) -> "FormalGradeIndex":
        return cls(
            bodies={},
            study_to_bodies={},
            pmid_to_studies={},
            chunk_to_study={},
            chunk_to_pmid={},
            policy=policy,
            minimum_certainty=minimum_certainty,
            mode=mode,
        )

    @classmethod
    def load(
        cls,
        profile_path: str,
        study_map_path: str,
        chunk_map_path: str,
        policy: str = "prioritize",
        minimum_certainty: str = "VERY_LOW",
        mode: str = "optional",
        valid_symptoms: Optional[Sequence[str]] = None,
    ) -> "FormalGradeIndex":
        mode_normalized = _clean_lower(mode)
        policy_normalized = _clean_lower(policy)
        minimum_normalized = _clean_upper(minimum_certainty)
        if mode_normalized not in GRADE_MODES:
            raise ValueError(f"Unknown grade_mode {mode!r}; expected one of {sorted(GRADE_MODES)}")
        if policy_normalized not in GRADE_POLICIES:
            raise ValueError(
                f"Unknown GRADE retrieval policy {policy!r}; expected one of {sorted(GRADE_POLICIES)}"
            )
        if minimum_normalized not in CERTAINTY_ORDER:
            raise ValueError(
                f"Unknown minimum certainty {minimum_certainty!r}; expected one of {list(CERTAINTY_ORDER)}"
            )
        if mode_normalized == "off":
            if policy_normalized == "filter":
                raise ValueError("--grade_policy=filter cannot be used with --grade_mode=off")
            return cls.empty(policy_normalized, minimum_normalized, mode_normalized)

        paths = {
            "profile": str(profile_path),
            "study_map": str(study_map_path),
            "chunk_map": str(chunk_map_path),
        }
        exists = {name: Path(path).exists() for name, path in paths.items()}
        if not any(exists.values()):
            if mode_normalized == "required":
                raise FileNotFoundError(
                    "GRADE mode is required, but no GRADE profile/study/chunk mapping files exist."
                )
            if policy_normalized == "filter":
                raise ValueError(
                    "--grade_policy=filter requires an active GRADE index; no GRADE files were found."
                )
            print("[GRADE] No GRADE files found; continuing with all passages UNMAPPED.")
            return cls.empty(policy_normalized, minimum_normalized, mode_normalized)
        if not all(exists.values()):
            missing = [paths[name] for name, ok in exists.items() if not ok]
            raise FileNotFoundError(
                "GRADE inputs are partially present. Provide all three files or none. Missing: "
                + ", ".join(missing)
            )

        profiles = pd.read_csv(profile_path, dtype=str).fillna("")
        if profiles.empty:
            if mode_normalized == "required":
                raise ValueError(
                    "GRADE mode is required, but grade_evidence_profiles.csv has no finalized body."
                )
            if policy_normalized == "filter":
                raise ValueError(
                    "--grade_policy=filter requires at least one finalized GRADE body."
                )
            print("[GRADE] GRADE profile is empty; continuing with all passages UNMAPPED.")
            return cls.empty(policy_normalized, minimum_normalized, mode_normalized)

        return cls.from_csv(
            profile_path=profile_path,
            study_map_path=study_map_path,
            chunk_map_path=chunk_map_path,
            policy=policy_normalized,
            minimum_certainty=minimum_normalized,
            mode=mode_normalized,
            valid_symptoms=valid_symptoms,
        )

    @classmethod
    def from_csv(
        cls,
        profile_path: str,
        study_map_path: str,
        chunk_map_path: str,
        policy: str = "prioritize",
        minimum_certainty: str = "VERY_LOW",
        mode: str = "required",
        valid_symptoms: Optional[Sequence[str]] = None,
    ) -> "FormalGradeIndex":
        policy_normalized = _clean_lower(policy)
        mode_normalized = _clean_lower(mode)
        minimum_normalized = _clean_upper(minimum_certainty)
        if policy_normalized not in GRADE_POLICIES:
            raise ValueError(
                f"Unknown GRADE retrieval policy {policy!r}; expected one of {sorted(GRADE_POLICIES)}"
            )
        if mode_normalized not in GRADE_MODES:
            raise ValueError(f"Unknown grade_mode {mode!r}; expected one of {sorted(GRADE_MODES)}")
        if minimum_normalized not in CERTAINTY_ORDER:
            raise ValueError(
                f"Unknown minimum certainty {minimum_certainty!r}; expected one of {list(CERTAINTY_ORDER)}"
            )

        for path, label in [
            (profile_path, "GRADE evidence profile"),
            (study_map_path, "GRADE study map"),
            (chunk_map_path, "GRADE chunk map"),
        ]:
            if not Path(path).exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        profiles = pd.read_csv(profile_path, dtype=str).fillna("")
        study_map = pd.read_csv(study_map_path, dtype=str).fillna("")
        chunk_map = pd.read_csv(chunk_map_path, dtype=str).fillna("")
        _require_columns(profiles, PROFILE_REQUIRED_COLUMNS, "GRADE evidence profile")
        _require_columns(study_map, STUDY_MAP_REQUIRED_COLUMNS, "GRADE study map")
        _require_columns(chunk_map, CHUNK_MAP_REQUIRED_COLUMNS, "GRADE chunk map")
        if profiles.empty:
            raise ValueError("GRADE evidence profile contains no rows.")
        if study_map.empty:
            raise ValueError("GRADE study map is empty.")
        if chunk_map.empty:
            raise ValueError(
                "GRADE chunk map is empty; a retrieved passage cannot inherit parent-body "
                "certainty without explicit source provenance."
            )

        valid_finding_set = set(valid_symptoms or [])
        bodies: Dict[str, GradeBody] = {}
        for row_index, raw_row in profiles.iterrows():
            row = raw_row.to_dict()
            context = f"GRADE profile row {row_index + 2}"
            _require_nonblank(row, PROFILE_REQUIRED_COLUMNS, context)

            status = _clean_lower(row["status"])
            if status not in GRADE_FINAL_STATUSES:
                raise ValueError(
                    f"{context} is not finalized/published (status={row['status']!r}). "
                    "Agent 1 must not consume draft GRADE judgments."
                )

            body_id = _clean_text(row["body_id"])
            if body_id in bodies:
                raise ValueError(f"Duplicate body_id in GRADE profile: {body_id}")

            finding = _clean_text(row["finding"])
            if valid_finding_set and finding not in valid_finding_set:
                raise ValueError(
                    f"{context} uses unknown finding {finding!r}; expected one of {sorted(valid_finding_set)}"
                )

            protocol = _clean_lower(row["grade_protocol"])
            if protocol not in GRADE_PROTOCOL_START:
                raise ValueError(
                    f"{context} has unsupported grade_protocol {protocol!r}; expected one of "
                    f"{sorted(GRADE_PROTOCOL_START)}"
                )

            origin = _clean_lower(row["profile_origin"])
            if origin not in GRADE_PROFILE_ORIGINS:
                raise ValueError(
                    f"{context} has unsupported profile_origin {origin!r}; expected one of "
                    f"{sorted(GRADE_PROFILE_ORIGINS)}"
                )

            applicability = _clean_lower(row["applicability_status"])
            if applicability != "applicable":
                raise ValueError(
                    f"{context}: only profiles explicitly marked applicability_status=APPLICABLE "
                    "may affect VERGE retrieval. Partially/not-applicable bodies should remain outside "
                    "the production mapping."
                )

            final_certainty = _clean_upper(row["final_certainty"])
            if final_certainty not in CERTAINTY_ORDER:
                raise ValueError(
                    f"{context}: invalid final_certainty {row['final_certainty']!r}; expected one of "
                    f"{list(CERTAINTY_ORDER)}"
                )

            start_certainty = _clean_upper(row.get("starting_certainty"))
            total_downgrade: Optional[int] = None
            total_upgrade: Optional[int] = None
            if origin == "internal_formal":
                _require_columns(profiles, INTERNAL_FORMAL_REQUIRED_COLUMNS, "internal formal GRADE profile")
                start_certainty, total_downgrade, total_upgrade = _validate_internal_formal_profile(
                    row, protocol, context
                )
                expected_score = _clamp_certainty_score(
                    CERTAINTY_ORDER[start_certainty] - int(total_downgrade) + int(total_upgrade)
                )
                expected_final = ORDER_TO_CERTAINTY[expected_score]
                if final_certainty != expected_final:
                    raise ValueError(
                        f"{context}: final_certainty={final_certainty}, but recorded internal "
                        f"downgrade/upgrade decisions yield {expected_final}."
                    )
            else:
                if start_certainty:
                    if start_certainty not in CERTAINTY_ORDER:
                        raise ValueError(f"{context}: invalid starting_certainty {start_certainty!r}")
                    expected_start = GRADE_PROTOCOL_START[protocol]
                    if start_certainty != expected_start:
                        raise ValueError(
                            f"{context}: starting_certainty={start_certainty} conflicts with the "
                            f"configured {protocol} framework start {expected_start}."
                        )
                else:
                    start_certainty = GRADE_PROTOCOL_START[protocol]

            bodies[body_id] = GradeBody(
                body_id=body_id,
                finding=finding,
                grade_protocol=protocol,
                profile_origin=origin,
                source_citation=_clean_text(row["source_citation"]),
                source_identifier=_clean_text(row.get("source_identifier")),
                source_url_or_doi=_clean_text(row.get("source_url_or_doi")),
                final_certainty=final_certainty,
                starting_certainty=start_certainty,
                outcome=_clean_text(row["outcome"]),
                population=_clean_text(row["population"]),
                index_factor_or_test=_clean_text(row["index_factor_or_test"]),
                reference_standard_or_comparator=_clean_text(
                    row["reference_standard_or_comparator"]
                ),
                applicability_status="APPLICABLE",
                applicability_rationale=_clean_text(row["applicability_rationale"]),
                evidence_review_id=_clean_text(row.get("evidence_review_id")),
                total_downgrade=total_downgrade,
                total_upgrade=total_upgrade,
                profile_row=row,
            )

        if not bodies:
            raise ValueError("No finalized applicable GRADE bodies were loaded.")

        study_to_bodies: Dict[str, List[str]] = {}
        pmid_to_studies: Dict[str, List[str]] = {}
        included_counts: Counter = Counter()
        for row_index, raw_row in study_map.iterrows():
            row = raw_row.to_dict()
            context = f"GRADE study-map row {row_index + 2}"
            _require_nonblank(row, ["body_id", "include"], context)
            body_id = _clean_text(row["body_id"])
            if body_id not in bodies:
                raise ValueError(f"{context} references unknown body_id {body_id!r}")
            if not _clean_bool(row["include"]):
                continue
            study_id = _clean_text(row.get("study_id"))
            pmid = _clean_text(row.get("pmid"))
            if not study_id and not pmid:
                raise ValueError(f"{context} must contain at least one of study_id or pmid")
            if not study_id:
                study_id = f"PMID_{pmid}"
            study_to_bodies.setdefault(study_id, []).append(body_id)
            if pmid:
                pmid_to_studies.setdefault(pmid, []).append(study_id)
            included_counts[body_id] += 1

        for body_id in bodies:
            if included_counts.get(body_id, 0) == 0:
                raise ValueError(
                    f"GRADE body {body_id!r} has no included study/PMID mapping. A body cannot "
                    "be propagated to passages without source-study provenance."
                )

        chunk_to_study: Dict[int, str] = {}
        chunk_to_pmid: Dict[int, str] = {}
        for row_index, raw_row in chunk_map.iterrows():
            row = raw_row.to_dict()
            context = f"GRADE chunk-map row {row_index + 2}"
            try:
                chunk_idx = int(float(_clean_text(row["chunk_idx"])))
            except ValueError as exc:
                raise ValueError(f"{context} has invalid chunk_idx") from exc
            study_id = _clean_text(row.get("study_id"))
            pmid = _clean_text(row.get("pmid"))
            if not study_id and not pmid:
                raise ValueError(f"{context} must contain at least one of study_id or pmid")
            if chunk_idx in chunk_to_study or chunk_idx in chunk_to_pmid:
                raise ValueError(f"Duplicate chunk_idx in GRADE chunk map: {chunk_idx}")
            if study_id:
                chunk_to_study[chunk_idx] = study_id
            if pmid:
                chunk_to_pmid[chunk_idx] = pmid

        resolved_chunk_count = 0
        for chunk_idx in set(chunk_to_study) | set(chunk_to_pmid):
            study_id = chunk_to_study.get(chunk_idx, "")
            pmid = chunk_to_pmid.get(chunk_idx, "")
            candidate_studies: List[str] = []
            if study_id:
                candidate_studies.append(study_id)
            if pmid:
                candidate_studies.extend(pmid_to_studies.get(pmid, ()))
                candidate_studies.append(f"PMID_{pmid}")
            if any(study_to_bodies.get(study, ()) for study in candidate_studies):
                resolved_chunk_count += 1
        if resolved_chunk_count == 0:
            raise ValueError(
                "GRADE chunk map contains no chunk that resolves to an included study in a "
                "finalized applicable evidence body. Check study_id/PMID mappings."
            )

        return cls(
            bodies=bodies,
            study_to_bodies=study_to_bodies,
            pmid_to_studies=pmid_to_studies,
            chunk_to_study=chunk_to_study,
            chunk_to_pmid=chunk_to_pmid,
            policy=policy_normalized,
            minimum_certainty=minimum_normalized,
            mode=mode_normalized,
            source_paths={
                "profile": str(profile_path),
                "study_map": str(study_map_path),
                "chunk_map": str(chunk_map_path),
            },
        )

    def _candidate_bodies(
        self, finding: str, chunk_idx: int
    ) -> Tuple[List[GradeBody], str, str]:
        study_id = self.chunk_to_study.get(chunk_idx, "")
        pmid = self.chunk_to_pmid.get(chunk_idx, "")
        study_ids: List[str] = []
        if study_id:
            study_ids.append(study_id)
        if pmid:
            study_ids.extend(self.pmid_to_studies.get(pmid, ()))
            study_ids.append(f"PMID_{pmid}")
        seen_studies: set[str] = set()
        dedup_studies = [s for s in study_ids if not (s in seen_studies or seen_studies.add(s))]

        seen_bodies: set[str] = set()
        body_ids: List[str] = []
        for candidate_study in dedup_studies:
            for body_id in self.study_to_bodies.get(candidate_study, ()):
                if body_id not in seen_bodies:
                    body_ids.append(body_id)
                    seen_bodies.add(body_id)
        bodies = [
            self.bodies[body_id]
            for body_id in body_ids
            if self.bodies[body_id].finding == finding
        ]
        return bodies, study_id, pmid

    def annotate(self, finding: str, chunk_idx: int) -> GradeCandidate:
        bodies, study_id, pmid = self._candidate_bodies(finding, chunk_idx)
        if not bodies:
            eligible = self.policy in {"annotate", "prioritize"}
            return GradeCandidate(
                mapped=False,
                eligible=eligible,
                study_id=study_id,
                pmid=pmid,
                reason="No finalized applicable parent GRADE body maps to this chunk for the target finding.",
            )

        bodies = sorted(bodies, key=lambda body: body.body_id)
        body_ids = tuple(body.body_id for body in bodies)
        body_certainties = tuple(body.final_certainty for body in bodies)
        min_score = min(CERTAINTY_ORDER[certainty] for certainty in body_certainties)
        conservative_certainty = ORDER_TO_CERTAINTY[min_score]
        protocols = sorted({body.grade_protocol for body in bodies})
        outcomes = [body.outcome for body in bodies]
        origins = tuple(body.profile_origin for body in bodies)
        citations = tuple(body.source_citation for body in bodies)

        meets_threshold = (
            CERTAINTY_ORDER[conservative_certainty]
            >= CERTAINTY_ORDER[self.minimum_certainty]
        )
        eligible = not (self.policy == "filter" and not meets_threshold)
        if self.policy == "filter" and not meets_threshold:
            reason = (
                f"Mapped to finalized GRADE body/bodies, but conservative parent certainty "
                f"{conservative_certainty} is below filter threshold {self.minimum_certainty}."
            )
        else:
            reason = (
                "Mapped by explicit chunk->study/PMID->finalized evidence-body provenance; "
                "conservative minimum parent certainty is used when multiple bodies apply."
            )
        return GradeCandidate(
            mapped=True,
            eligible=eligible,
            body_id="|".join(body_ids),
            body_ids=body_ids,
            body_certainties=body_certainties,
            profile_origins=origins,
            source_citations=citations,
            study_id=study_id,
            pmid=pmid,
            final_certainty=conservative_certainty,
            grade_protocol="|".join(protocols),
            outcome=" | ".join(outcomes),
            reason=reason,
        )

    def sort_key(
        self, candidate: GradeCandidate, rrf_score: float, cosine: float
    ) -> Tuple[int, int, float, float]:
        mapped_priority = 1 if candidate.mapped else 0
        certainty_priority = CERTAINTY_ORDER.get(candidate.final_certainty, 0)
        if self.policy == "annotate":
            mapped_priority = 0
            certainty_priority = 0
        return mapped_priority, certainty_priority, float(rrf_score), float(cosine)

    def metadata(self) -> Dict[str, Any]:
        protocol_counts = Counter(body.grade_protocol for body in self.bodies.values())
        origin_counts = Counter(body.profile_origin for body in self.bodies.values())
        return {
            "formal_grade_available": self.active,
            "formal_grade_assessment_consumed": self.active,
            "grade_mode": self.mode,
            "rating_unit": "body of evidence for a specified question/outcome",
            "grade_judgments_generated_by_extractor": False,
            "passage_level_grade_generated": False,
            "retrieval_policy": self.policy,
            "minimum_certainty_for_filter_only": self.minimum_certainty,
            "multi_body_retrieval_aggregation": "conservative minimum parent-body certainty",
            "n_finalized_bodies": len(self.bodies),
            "n_included_studies_or_ids": len(self.study_to_bodies),
            "n_mapped_chunks": len(set(self.chunk_to_study) | set(self.chunk_to_pmid)),
            "grade_protocol_counts": dict(protocol_counts),
            "profile_origin_counts": dict(origin_counts),
            "findings_with_finalized_grade_bodies": sorted(
                {body.finding for body in self.bodies.values()}
            ),
            "source_citations": sorted({body.source_citation for body in self.bodies.values()}),
            "source_paths": self.source_paths,
            "note": (
                "GRADE is optional external evidence-quality metadata. Agent 1 never infers GRADE "
                "from sentence text and never uses GRADE to determine a patient label. Passages "
                "without a valid parent-body mapping remain UNMAPPED under annotate/prioritize."
            ),
        }

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from bert_score import score as bert_score_fn

    HAVE_BERTSCORE = True
except ModuleNotFoundError:
    HAVE_BERTSCORE = False

PATH = os.environ.get("EOCRC_NOTES", "rebuilt_notes_by_noteid.csv")
HF_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_CACHE_DIR = os.environ.get(
    "HF_CACHE_DIR", "/lustre/smuexa01/client/users/nikkieh/hf_cache"
)
UMLS_JSON = os.environ.get("UMLS_JSON", "umls_synonyms.json")
UMLS_CACHE = os.environ.get("UMLS_CACHE", "umls_concept_cache.json")
CHUNKS_PATH = os.environ.get(
    "PUBMED_CHUNKS", "pubmed_chunks_sentences.json"
)

MEDCPT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"

OUT_RAW = os.environ.get("EXP7_RAW", "agent1_grade_aware_outputs_raw.csv")
OUT_METRICS = os.environ.get(
    "EXP7_METRICS", "agent1_grade_aware_note_level_metrics.csv"
)
OUT_CHECKPOINT = os.environ.get("EXP7_CHECKPOINT", "agent1_grade_aware_checkpoint.csv")
OUT_SUMMARY = os.environ.get("EXP7_SUMMARY", "agent1_grade_aware_summary_table.csv")
OUT_VALID = os.environ.get("EXP7_VALID", "agent1_grade_aware_valid_outputs.csv")
OUT_SCHEMA_FAIL = os.environ.get(
    "EXP7_SCHEMA_FAIL", "agent1_grade_aware_schema_failures.csv"
)
OUT_PARSE_FAIL = os.environ.get(
    "EXP7_PARSE_FAIL", "agent1_grade_aware_parse_failures.json"
)

DENSE_TOP_K = 32
RRF_K = 60
SOURCE_CANDIDATE_K = 10
DENSE_KEEP_K = 2
BLEU_THRESHOLD = 0.10

GRADE_PROFILE_PATH = os.environ.get("GRADE_PROFILE", "grade_evidence_profiles.csv")
GRADE_STUDY_MAP_PATH = os.environ.get("GRADE_STUDY_MAP", "grade_study_map.csv")
GRADE_CHUNK_MAP_PATH = os.environ.get("GRADE_CHUNK_MAP", "grade_chunk_map.csv")
GRADE_MODE = os.environ.get("GRADE_MODE", "optional")
GRADE_POLICY = os.environ.get("GRADE_POLICY", "prioritize")
GRADE_MIN_CERTAINTY = os.environ.get("GRADE_MIN_CERTAINTY", "VERY_LOW")
MODEL_CONTEXT_SAFETY_TOKENS = int(
    os.environ.get("MODEL_CONTEXT_SAFETY_TOKENS", "256")
)
MAIN_MAX_NEW_TOKENS = 1_536
RETRY_MAX_NEW_TOKENS = 512
DENIAL_CONTEXT = 40
CHECKPOINT_EVERY = 25

MAX_GROUP_B_IN_QUERY = 3
MAX_GROUP_B_IN_PROMPT = 10
CONTEXT_BUDGET_MODES = [
    ("FULL", 15, 10, 2),
    ("COMPACT", 8, 5, 1),
    ("LEAN", 4, 0, 1),
    ("NOTE_PLUS_MIN_ALIASES", 3, 0, 0),
    ("NOTE_ONLY", 0, 0, 0),
]

GRADE_AUDIT_LEVELS = ["HIGH", "MODERATE", "LOW", "VERY_LOW", "UNMAPPED"]

SYMPTOMS = [
    "Abdominal pain",
    "Rectal bleeding",
    "Rectal pain",
    "Diarrhea",
    "Constipation",
    "Weight loss",
    "Family history of colorectal cancer",
]

SYMPTOM_SPECS = [
    (symptom, f"{symptom} confidence", f"{symptom} inference")
    for symptom in SYMPTOMS
]

EXPECTED_OUTPUT_KEYS: List[str] = []
for _symptom in SYMPTOMS[:-1]:
    _duration = f"Duration of {_symptom.lower()}"
    EXPECTED_OUTPUT_KEYS.extend(
        [
            _symptom,
            f"{_symptom} confidence",
            f"{_symptom} inference",
            f"{_symptom} note_section",
            f"{_symptom} experiencer",
            f"{_symptom} assertion_status",
            _duration,
            f"{_duration} confidence",
            f"{_duration} inference",
        ]
    )
EXPECTED_OUTPUT_KEYS.extend(
    [
        "Family history of colorectal cancer",
        "Family history of colorectal cancer confidence",
        "Family history of colorectal cancer inference",
        "Family history of colorectal cancer note_section",
        "Family history of colorectal cancer experiencer",
        "Family history of colorectal cancer assertion_status",
        "Other comments",
        "Other comments confidence",
        "Other comments inference",
    ]
)

METADATA_KEYS = ["note_section", "experiencer", "assertion_status"]
VALID_NOTE_SECTIONS = {
    "hpi", "cc", "chief complaint", "history of present illness",
    "ros", "review of systems", "assessment", "plan", "a/p",
    "assessment and plan", "pmh", "past medical history",
    "fh", "family history", "pe", "physical exam",
    "physical examination", "vitals", "medications",
    "surgical history", "social history", "diagnosis",
    "impression", "hospital course", "other", "unknown", "none",
}
VALID_EXPERIENCERS = {"patient", "family_member", "other", "unknown"}
VALID_ASSERTION_STATUSES = {
    "affirmed", "negated", "hypothetical", "planned",
    "historical", "absent", "unknown",
}

ASSERTION_STATUS_ALIASES = {
    "asserted": "affirmed",
    "present": "affirmed",
    "positive": "affirmed",
    "denied": "negated",
    "negative": "negated",
    "not present": "absent",
    "not_present": "absent",
    "not documented": "absent",
}
NOTE_SECTION_ALIASES = {
    "review of symptoms": "review of systems",
    "review of symptom": "review of systems",
    "history present illness": "history of present illness",
}

BAD_INFERENCE_VALS = {
    "",
    "n/a",
    "na",
    "not mentioned",
    "none",
    "no inference",
    "not reported",
    "not applicable",
    "not stated",
    "not mentioned outside ros",
    "yes",
    "no",
    "present",
    "absent",
    "only in ros-negative context",
    "only ros-negative, nowhere else",
}

DIRECT_ALIASES: Dict[str, List[str]] = {
    "Abdominal pain": [
        "abdominal pain", "abd pain", "abd pn", "abdo pain",
        "stomach pain", "stomach ache", "belly pain",
        "epigastric pain", "epigastric discomfort", "epigastric tenderness",
        "abdominal tenderness", "tender abdomen",
        "ruq pain", "right upper quadrant pain",
        "luq pain", "left upper quadrant pain",
        "rlq pain", "right lower quadrant pain",
        "llq pain", "left lower quadrant pain",
        "periumbilical pain", "umbilical pain",
        "suprapubic pain", "pelvic pain",
        "lower abdominal pain", "upper abdominal pain",
        "abdominal cramping", "colicky pain", "colicky abdominal pain",
        "sharp abdominal pain", "dull abdominal pain",
        "abdominal discomfort", "abdominal soreness",
        "c/o abdominal pain", "complains of abdominal pain",
        "reports abdominal pain", "pain in abdomen",
    ],
    "Rectal bleeding": [
        "rectal bleeding", "bleeding per rectum", "blood per rectum",
        "blood from rectum", "blood in stool", "bloody stool",
        "blood on stool", "stool with blood", "streaks of blood",
        "blood streaked stool", "blood on toilet paper",
        "blood when wiping", "hematochezia", "haematochezia",
        "brbpr", "bright red blood per rectum",
        "maroon stools", "melena", "black tarry stools",
        "positive fobt", "positive fit", "occult blood",
        "heme positive stool",
    ],
    "Rectal pain": [
        "rectal pain", "pain in rectum", "painful rectum",
        "anal pain", "pain in anus", "anorectal pain",
        "proctalgia", "proctalgia fugax",
        "rectal discomfort", "anal discomfort",
        "rectal soreness", "anal soreness",
        "pain with bowel movement", "painful bowel movement",
        "painful defecation", "dyschezia",
        "pain during defecation", "pain after bowel movement",
        "tenesmus", "rectal pressure",
        "perianal pain", "perirectal pain",
    ],
    "Diarrhea": [
        "diarrhea", "diarrhoea", "loose stools", "loose stool",
        "watery stools", "watery stool", "liquid stool", "runny stool",
        "frequent stools", "frequent bowel movements",
        "increased bowel movements", "increased stool frequency",
        "multiple loose bms", "loose bm", "watery bm",
        "explosive diarrhea", "bristol 6", "bristol 7",
    ],
    "Constipation": [
        "constipation", "constipated", "hard stools", "hard stool",
        "infrequent stools", "infrequent bowel movements",
        "decreased stool frequency", "decreased bowel movements",
        "no bm", "no bowel movement", "no stool for",
        "difficulty passing stool", "straining",
        "incomplete evacuation", "obstipation",
        "fecal impaction", "stool impaction",
        "bristol 1", "bristol 2", "pellet stools",
    ],
    "Weight loss": [
        "weight loss", "wt loss", "lost weight", "losing weight",
        "weight down", "weight decreased", "decreased weight",
        "unintentional weight loss", "unexplained weight loss",
        "involuntary weight loss", "clothes fitting looser",
        "cachexia", "wasting", "cachectic",
    ],
    "Family history of colorectal cancer": [
        "family history of colorectal cancer",
        "family history colorectal cancer",
        "family history of colon cancer",
        "family history colon cancer",
        "family history of rectal cancer",
        "family history of bowel cancer",
        "fh colon cancer", "fhx colon cancer",
        "fhx crc", "fh crc",
        "crc in family", "colon cancer in family",
        "colon cancer runs in family",
        "mother had colon cancer", "father had colon cancer",
        "sister had colon cancer", "brother had colon cancer",
        "parent had colon cancer", "relative with colon cancer",
        "first degree relative with colon cancer",
        "fh bowel cancer",
    ],
}

RETRIEVAL_EXTRA_TERMS: Dict[str, List[str]] = {
    "Abdominal pain": [
        "cramping", "bloating", "abdominal bloating",
        "distension", "abdominal distension",
        "dyspepsia", "indigestion", "fullness",
        "abdominal pressure", "gas pain",
        "heartburn", "acid reflux", "gerd symptoms", "gastritis",
    ],
    "Rectal bleeding": [
        "rectal hemorrhage", "rectorrhagia",
        "hemorrhoids with bleeding", "anal fissure bleeding",
    ],
    "Rectal pain": [
        "hemorrhoid pain", "thrombosed hemorrhoid",
        "anal fissure pain", "fissure pain",
    ],
    "Diarrhea": [
        "the runs", "fecal urgency", "bowel urgency",
        "soft stools", "mushy stools",
        "gastroenteritis", "colitis with diarrhea",
    ],
    "Constipation": [
        "retained stool", "stool burden",
        "slow transit constipation", "scybalous stools",
        "strains with bm",
    ],
    "Weight loss": [
        "malnutrition", "anorexia", "loss of appetite",
        "decreased appetite", "poor weight gain",
        "failure to thrive",
    ],
    "Family history of colorectal cancer": [
        "lynch syndrome", "hnpcc",
        "familial adenomatous polyposis", "fap",
        "hereditary colorectal cancer",
    ],
}

def require_file(path: str, description: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} not found: {path}")


def phrase_in_text(term: str, text: str) -> bool:
    term_value = str(term).strip().lower()
    text_value = str(text).lower()
    if not term_value or len(term_value) < 3:
        return False
    pattern = r"(?<!\w)" + re.escape(term_value) + r"(?!\w)"
    return bool(re.search(pattern, text_value))


def find_phrase_spans(term: str, text: str) -> List[Tuple[int, int]]:
    term_value = str(term).strip().lower()
    text_value = str(text).lower()
    if not term_value or len(term_value) < 3:
        return []
    pattern = re.compile(r"(?<!\w)" + re.escape(term_value) + r"(?!\w)")
    return [(match.start(), match.end()) for match in pattern.finditer(text_value)]


PRE_NEGATION = [
    r"\bno\b",
    r"\bdenies?\b",
    r"\bnegative for\b",
    r"\bwithout\b",
    r"\babsence of\b",
    r"\bno history of\b",
    r"\bnot\b",
]

POST_NEGATION = [
    r"\babsent\b",
    r"\bdenied\b",
    r"\bnegative\b",
    r"\bnot present\b",
    r"\bresolved\b",
]


def mention_is_negated(
    text_lower: str,
    start: int,
    end: int,
    window: int = DENIAL_CONTEXT,
) -> bool:
    before = text_lower[max(0, start - window) : start]
    after = text_lower[end : min(len(text_lower), end + window)]
    pre = any(re.search(pattern, before) for pattern in PRE_NEGATION)
    post = any(re.search(pattern, after) for pattern in POST_NEGATION)
    return pre or post

def load_umls_synonyms() -> Dict[str, List[str]]:
    require_file(UMLS_JSON, "UMLS synonym file")
    with open(UMLS_JSON, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    lowered = {
        symptom: [str(term).lower() for term in terms]
        for symptom, terms in data.items()
    }
    total = sum(len(values) for values in lowered.values())
    print(f"UMLS synonyms: {total} terms across {len(lowered)} symptoms")
    return lowered


def load_concept_cache() -> Tuple[Dict[str, Any], Dict[str, str]]:
    require_file(UMLS_CACHE, "UMLS concept cache")
    with open(UMLS_CACHE, "r", encoding="utf-8") as handle:
        cache = json.load(handle)
    n_group_a = sum(
        1 for value in cache.values() if value.get("is_group_a", False)
    )
    n_group_b = len(cache) - n_group_a
    print(
        f"UMLS cache: {len(cache)} concepts "
        f"(GROUP A={n_group_a}, GROUP B={n_group_b})"
    )
    group_b_lookup = {
        str(term).lower(): str(value.get("name", term))
        for term, value in cache.items()
        if not value.get("is_group_a", False)
    }
    return cache, group_b_lookup


def build_vocabularies(
    umls_synonyms: Mapping[str, Sequence[str]],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build separate direct-note and broad-retrieval vocabularies."""
    direct_vocab: Dict[str, List[str]] = {}
    retrieval_vocab: Dict[str, List[str]] = {}
    for symptom in SYMPTOMS:
        direct = list(
            dict.fromkeys(
                str(term).lower()
                for term in DIRECT_ALIASES.get(symptom, [])
                if str(term).strip()
            )
        )
        extras = [
            str(term).lower()
            for term in RETRIEVAL_EXTRA_TERMS.get(symptom, [])
            if str(term).strip()
        ]
        umls = [
            str(term).lower()
            for term in umls_synonyms.get(symptom, [])
            if str(term).strip()
        ]
        direct_vocab[symptom] = direct
        retrieval_vocab[symptom] = list(dict.fromkeys(direct + extras + umls))
    direct_total = sum(len(values) for values in direct_vocab.values())
    retrieval_total = sum(len(values) for values in retrieval_vocab.values())
    direct_unique = len(
        {term for values in direct_vocab.values() for term in values}
    )
    retrieval_unique = len(
        {term for values in retrieval_vocab.values() for term in values}
    )
    print(
        f"Direct vocabulary:    {direct_total} entries "
        f"({direct_unique} unique) — note matching + prompt"
    )
    print(
        f"Retrieval vocabulary: {retrieval_total} entries "
        f"({retrieval_unique} unique) — MedCPT queries + relevance"
    )
    return direct_vocab, retrieval_vocab


def _chunk_to_text(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, Mapping):
        for key in ["text", "chunk", "sentence", "content", "abstract"]:
            if key in chunk and chunk[key] is not None:
                return str(chunk[key])
    return str(chunk)


def load_chunks() -> List[str]:
    require_file(CHUNKS_PATH, "PubMed sentence-chunk corpus")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as handle:
        raw_chunks = json.load(handle)
    if not isinstance(raw_chunks, list):
        raise ValueError("PubMed chunk file must contain a JSON list.")
    chunks = [_chunk_to_text(chunk).strip() for chunk in raw_chunks]
    if not chunks:
        raise ValueError("PubMed chunk corpus is empty after normalization.")

    empty_indices = [i for i, chunk in enumerate(chunks) if not chunk]
    if empty_indices:
        preview = empty_indices[:20]
        raise ValueError(
            "PubMed chunk corpus contains empty entries at indices "
            f"{preview}{'...' if len(empty_indices) > 20 else ''}. "
            "Refusing to filter them because chunk_idx provenance must remain stable."
        )
    print(f"Sentence chunks loaded: {len(chunks)}")
    if len(chunks) < 100:
        print(
            f"[CORPUS WARNING] Only {len(chunks)} PubMed chunks were loaded. "
            "Verify PUBMED_CHUNKS before a production run; a tiny corpus can "
            "make the retrieval/GRADE-aware audit misleading."
        )
    return chunks


def load_notes() -> pd.DataFrame:
    require_file(PATH, "Clinical notes CSV")
    notes_df = pd.read_csv(PATH)
    if "Clean_note_text" not in notes_df.columns:
        raise ValueError("Clean_note_text column is missing from the notes CSV.")
    notes_df["Clean_note_text"] = (
        notes_df["Clean_note_text"].fillna("").astype(str)
    )
    for column in [
        "DATE_OF_SERVIC_DTTM",
        "SPEC_NOTE_TIME_DTTM",
        "CONTACT_DATE",
    ]:
        if column in notes_df.columns:
            notes_df[column] = pd.to_datetime(notes_df[column], errors="coerce")
    sort_columns = [
        column
        for column in [
            "PAT_ID", "PAT_ENC_CSN_ID", "DATE_OF_SERVIC_DTTM",
            "SPEC_NOTE_TIME_DTTM", "CONTACT_NUM", "NOTE_ID",
        ]
        if column in notes_df.columns
    ]
    if sort_columns:
        notes_df = notes_df.sort_values(
            sort_columns, kind="stable"
        ).reset_index(drop=True)
    notes_df = notes_df[
        notes_df["Clean_note_text"].str.strip().ne("")
    ].reset_index(drop=True)
    notes_df["_run_row_index"] = np.arange(len(notes_df))
    print(f"Notes loaded: {len(notes_df)}")
    return notes_df


class MedCPTEncoder:
    def __init__(self, cache_dir: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading MedCPT Article Encoder: {MEDCPT_ARTICLE_MODEL}")
        self.article_tokenizer = AutoTokenizer.from_pretrained(
            MEDCPT_ARTICLE_MODEL, cache_dir=cache_dir
        )
        self.article_encoder = AutoModel.from_pretrained(
            MEDCPT_ARTICLE_MODEL, cache_dir=cache_dir
        ).to(self.device)
        self.article_encoder.eval()

        print(f"Loading MedCPT Query Encoder: {MEDCPT_QUERY_MODEL}")
        self.query_tokenizer = AutoTokenizer.from_pretrained(
            MEDCPT_QUERY_MODEL, cache_dir=cache_dir
        )
        self.query_encoder = AutoModel.from_pretrained(
            MEDCPT_QUERY_MODEL, cache_dir=cache_dir
        ).to(self.device)
        self.query_encoder.eval()

        print(f"MedCPT encoders ready on {self.device}")

    def _encode(
        self,
        model: Any,
        tokenizer: Any,
        texts: Sequence[str],
        batch_size: int,
        max_length: int = 512,
    ) -> np.ndarray:
        all_vectors: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            with self.torch.no_grad():
                tokens = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt",
                ).to(self.device)
                output = model(**tokens)
                vectors = output.last_hidden_state[:, 0, :]
                vectors = vectors / (
                    vectors.norm(dim=1, keepdim=True) + 1e-8
                )
                all_vectors.append(vectors.cpu().float().numpy())
        if not all_vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(all_vectors)

    def encode_articles(
        self, texts: Sequence[str], batch_size: int = 64
    ) -> np.ndarray:
        return self._encode(
            self.article_encoder, self.article_tokenizer,
            texts, batch_size=batch_size,
        )

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(
            self.query_encoder, self.query_tokenizer,
            texts, batch_size=32,
        )


def build_dense_index(
    encoder: MedCPTEncoder,
    chunks: Sequence[str],
    batch_size: int = 128,
) -> np.ndarray:
    print(f"Encoding {len(chunks)} chunks with MedCPT Article Encoder...")
    start = time.time()
    vectors = encoder.encode_articles(chunks, batch_size=batch_size)
    elapsed_minutes = (time.time() - start) / 60
    print(
        f"Dense index built: {vectors.shape} in "
        f"{elapsed_minutes:.1f} minutes"
    )
    return vectors

def scan_note_for_umls_concepts(
    note_text: str,
    direct_vocab: Mapping[str, Sequence[str]],
    group_b_lookup: Mapping[str, str],
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    note_lower = note_text.lower()
    group_a: Dict[str, List[str]] = {}
    for symptom in SYMPTOMS:
        found = [
            term
            for term in direct_vocab.get(symptom, [])
            if phrase_in_text(term, note_lower)
        ]
        if found:
            group_a[symptom] = found[:5]
    group_b: Dict[str, str] = {}
    for term, concept_name in group_b_lookup.items():
        if phrase_in_text(term, note_lower) and concept_name not in group_b:
            group_b[concept_name] = term
    return group_a, group_b


def build_query_variants(
    note_text: str,
    symptom: str,
    direct_vocab: Mapping[str, Sequence[str]],
    retrieval_vocab: Mapping[str, Sequence[str]],
    group_b: Mapping[str, str],
) -> Dict[str, str]:
    """Construct distinct query formulations for within-symptom RRF."""
    note_lower = note_text.lower()
    note_matches = [
        term
        for term in direct_vocab.get(symptom, [])
        if phrase_in_text(term, note_lower)
    ][:2]
    direct_set = set(direct_vocab.get(symptom, []))
    broader_terms = [
        term
        for term in retrieval_vocab.get(symptom, [])
        if term not in direct_set
    ][:3]
    group_b_names = list(group_b.keys())[:MAX_GROUP_B_IN_QUERY]
    variants: Dict[str, str] = {
        "base": f"{symptom} colorectal cancer clinical presentation",
        "note_anchored": " ".join(
            list(dict.fromkeys([symptom] + note_matches + group_b_names))
        ),
        "ontology_expanded": " ".join(
            list(
                dict.fromkeys(
                    [symptom] + broader_terms + ["colorectal cancer"]
                )
            )
        ),
    }
    unique: Dict[str, str] = {}
    seen: set[str] = set()
    for name, query in variants.items():
        normalized = " ".join(query.split()).strip()
        if normalized and normalized.lower() not in seen:
            unique[name] = normalized
            seen.add(normalized.lower())
    return unique

def retrieve_with_rrf(
    encoder: MedCPTEncoder,
    chunk_vectors: np.ndarray,
    chunks: Sequence[str],
    query_variants: Mapping[str, Mapping[str, str]],
    retrieval_vocab: Mapping[str, Sequence[str]],
    top_k: int = DENSE_TOP_K,
    candidate_k: int = SOURCE_CANDIDATE_K,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    flat_queries: List[str] = []
    flat_keys: List[Tuple[str, str]] = []
    for symptom in SYMPTOMS:
        variants = query_variants.get(symptom, {})
        for variant_name, query_text in variants.items():
            flat_keys.append((symptom, variant_name))
            flat_queries.append(query_text)
    if not flat_queries:
        return {symptom: [] for symptom in SYMPTOMS}, {}

    query_vectors = encoder.encode_queries(flat_queries)
    all_scores = chunk_vectors @ query_vectors.T

    ranked_lists: Dict[str, Dict[str, List[int]]] = defaultdict(dict)
    score_maps: Dict[str, Dict[str, Dict[int, float]]] = defaultdict(dict)

    for column_index, (symptom, variant_name) in enumerate(flat_keys):
        scores = all_scores[:, column_index]
        top_indices = np.argsort(scores)[::-1][:top_k]
        ranked_lists[symptom][variant_name] = [
            int(index) for index in top_indices
        ]
        score_maps[symptom][variant_name] = {
            int(index): float(scores[index]) for index in top_indices
        }

    per_symptom_candidates: Dict[str, List[Dict[str, Any]]] = {}
    retrieval_audit: Dict[str, Any] = {}

    for symptom in SYMPTOMS:
        fused_scores: Dict[int, float] = defaultdict(float)
        ranks_by_chunk: Dict[int, Dict[str, int]] = defaultdict(dict)
        cosines_by_chunk: Dict[int, Dict[str, float]] = defaultdict(dict)

        for variant_name, ranked_indices in ranked_lists.get(symptom, {}).items():
            for zero_based_rank, chunk_index in enumerate(ranked_indices):
                # RRF formula: 1/(k + rank) where rank is one-based.
                one_based_rank = zero_based_rank + 1
                fused_scores[chunk_index] += 1.0 / (RRF_K + one_based_rank)
                ranks_by_chunk[chunk_index][variant_name] = one_based_rank
                cosines_by_chunk[chunk_index][variant_name] = score_maps[
                    symptom
                ][variant_name][chunk_index]

        fused_indices = sorted(
            fused_scores,
            key=lambda index: (
                fused_scores[index],
                max(cosines_by_chunk[index].values()),
            ),
            reverse=True,
        )

        fused_passages: List[Dict[str, Any]] = []
        for fused_rank, chunk_index in enumerate(fused_indices, start=1):
            cosine_values = list(cosines_by_chunk[chunk_index].values())
            fused_passages.append(
                {
                    "text": chunks[chunk_index],
                    "chunk_idx": int(chunk_index),
                    "rrf_score": float(fused_scores[chunk_index]),
                    "rrf_rank": fused_rank,
                    "max_cosine": float(max(cosine_values)),
                    "mean_cosine": float(np.mean(cosine_values)),
                    "query_ranks": ranks_by_chunk[chunk_index],
                    "query_cosines": cosines_by_chunk[chunk_index],
                }
            )

        symptom_terms = retrieval_vocab.get(symptom, [])
        relevant = [
            passage
            for passage in fused_passages
            if any(
                phrase_in_text(term, passage["text"])
                for term in symptom_terms
            )
        ]
        relevance_fallback = len(relevant) == 0
        selected_pool = relevant if relevant else fused_passages
        candidates = selected_pool[:candidate_k]

        per_symptom_candidates[symptom] = candidates
        retrieval_audit[symptom] = {
            "query_variants": dict(query_variants.get(symptom, {})),
            "n_query_variants": len(query_variants.get(symptom, {})),
            "n_fused_unique_chunks": len(fused_passages),
            "n_phrase_relevant": len(relevant),
            "phrase_relevance_fallback": relevance_fallback,
            "n_candidates_for_source_filter": len(candidates),
        }

    return per_symptom_candidates, retrieval_audit


def render_evidence_section_from_audit(
    formal_grade_audit: Mapping[str, Sequence[Mapping[str, Any]]],
    per_finding_limit: int = DENSE_KEEP_K,
) -> str:
    blocks: List[str] = []
    keep = max(0, int(per_finding_limit))
    for finding in SYMPTOMS:
        rows = [dict(row) for row in formal_grade_audit.get(finding, []) if row.get("included")]
        rows.sort(key=lambda row: int(row.get("selection_rank", 10**9)))
        rows = rows[:keep]
        if not rows:
            blocks.append(f"[{finding}]\n  No external PubMed passage supplied in this prompt.")
            continue
        lines = [f"[{finding}]"]
        for evidence_index, row in enumerate(rows, start=1):
            body_ids = row.get("grade_body_ids") or []
            if isinstance(body_ids, str):
                body_ids_text = body_ids
            else:
                body_ids_text = "|".join(str(v) for v in body_ids) or "UNMAPPED"
            lines.append(
                "  [Evidence {index}; PARENT_BODY_GRADE_CERTAINTY={certainty}; "
                "BODY_IDS={bodies}; RRF={rrf:.6f}; max_cosine={cosine:.4f}] {text}".format(
                    index=evidence_index,
                    certainty=row.get("grade_final_certainty", "UNMAPPED"),
                    bodies=body_ids_text,
                    rrf=float(row.get("rrf_score", 0.0)),
                    cosine=float(row.get("max_cosine", 0.0)),
                    text=str(row.get("text", "")).strip(),
                )
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def prepare_source_filtered_evidence(
    per_symptom_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    grade_index: FormalGradeIndex,
) -> Tuple[str, Dict[str, Any], Counter, Counter, int]:
    formal_grade_audit: Dict[str, Any] = {}
    candidate_counter: Counter = Counter()
    included_counter: Counter = Counter()
    total_included = 0

    for finding in SYMPTOMS:
        tagged: List[Dict[str, Any]] = []
        for candidate_position, raw_passage in enumerate(
            per_symptom_candidates.get(finding, []), start=1
        ):
            passage = dict(raw_passage)
            chunk_idx = int(passage.get("chunk_idx", -1))
            grade = grade_index.annotate(finding, chunk_idx)
            passage.update(
                {
                    "candidate_position": candidate_position,
                    "grade_mapped": grade.mapped,
                    "grade_eligible": grade.eligible,
                    "grade_body_id": grade.body_id,
                    "grade_body_ids": list(grade.body_ids),
                    "grade_body_certainties": list(grade.body_certainties),
                    "grade_profile_origins": list(grade.profile_origins),
                    "grade_source_citations": list(grade.source_citations),
                    "grade_study_id": grade.study_id,
                    "grade_pmid": grade.pmid,
                    "grade_final_certainty": grade.final_certainty,
                    "grade_protocol": grade.grade_protocol,
                    "grade_outcome": grade.outcome,
                    "grade_reason": grade.reason,
                    "included": False,
                    "selection_rank": None,
                }
            )
            candidate_counter[grade.final_certainty] += 1
            tagged.append(passage)

        eligible = [passage for passage in tagged if passage["grade_eligible"]]
        eligible.sort(
            key=lambda passage: grade_index.sort_key(
                grade_index.annotate(finding, int(passage["chunk_idx"])),
                float(passage.get("rrf_score", 0.0)),
                float(passage.get("max_cosine", 0.0)),
            ),
            reverse=True,
        )
        included = eligible[:DENSE_KEEP_K]
        rank_by_chunk = {
            int(passage["chunk_idx"]): rank
            for rank, passage in enumerate(included, start=1)
        }

        audit_rows: List[Dict[str, Any]] = []
        for passage in tagged:
            chunk_idx = int(passage.get("chunk_idx", -1))
            if chunk_idx in rank_by_chunk:
                passage["included"] = True
                passage["selection_rank"] = rank_by_chunk[chunk_idx]
                included_counter[passage["grade_final_certainty"]] += 1
                total_included += 1
            audit_rows.append(
                {
                    "candidate_position": passage.get("candidate_position"),
                    "selection_rank": passage.get("selection_rank"),
                    "chunk_idx": passage.get("chunk_idx"),
                    "rrf_rank": passage.get("rrf_rank"),
                    "rrf_score": passage.get("rrf_score"),
                    "max_cosine": passage.get("max_cosine"),
                    "mean_cosine": passage.get("mean_cosine"),
                    "query_ranks": passage.get("query_ranks", {}),
                    "grade_mapped": passage.get("grade_mapped", False),
                    "grade_eligible": passage.get("grade_eligible", False),
                    "grade_body_id": passage.get("grade_body_id", ""),
                    "grade_body_ids": passage.get("grade_body_ids", []),
                    "grade_body_certainties": passage.get("grade_body_certainties", []),
                    "grade_profile_origins": passage.get("grade_profile_origins", []),
                    "grade_source_citations": passage.get("grade_source_citations", []),
                    "grade_study_id": passage.get("grade_study_id", ""),
                    "grade_pmid": passage.get("grade_pmid", ""),
                    "grade_final_certainty": passage.get("grade_final_certainty", "UNMAPPED"),
                    "grade_protocol": passage.get("grade_protocol", ""),
                    "grade_outcome": passage.get("grade_outcome", ""),
                    "grade_reason": passage.get("grade_reason", ""),
                    "included": passage.get("included", False),
                    "text": passage.get("text", ""),
                }
            )
        formal_grade_audit[finding] = audit_rows

    evidence_section = render_evidence_section_from_audit(
        formal_grade_audit, per_finding_limit=DENSE_KEEP_K
    )
    return (
        evidence_section,
        formal_grade_audit,
        candidate_counter,
        included_counter,
        total_included,
    )

def split_note_ros(note_text: str) -> Tuple[str, str]:
    note_lower = note_text.lower()
    ros_header_patterns = [
        r"(?im)^\s*review of systems\s*:??\s*$",
        r"(?im)^\s*ros\s*:??\s*$",
        r"(?im)^\s*r\.o\.s\.\s*:??\s*$",
        r"(?im)^\s*systems review\s*:??\s*$",
        r"(?im)^\s*review of system\s*:??\s*$",
    ]
    starts: List[int] = []
    for pattern in ros_header_patterns:
        match = re.search(pattern, note_text)
        if match:
            starts.append(match.start())
    if not starts:
        inline = [
            note_lower.find(header)
            for header in ["review of systems", "ros:", "r.o.s."]
            if note_lower.find(header) >= 0
        ]
        if not inline:
            return note_text, ""
        ros_start = min(inline)
    else:
        ros_start = min(starts)

    next_header_pattern = re.compile(
        r"(?im)^\s*(assessment(?: and plan)?|plan|physical exam(?:ination)?|"
        r"medications|allergies|vital signs|past medical history|"
        r"social history|family history|objective|subjective|hpi|"
        r"history of present illness|diagnosis|impression)\s*:??\s*$"
    )
    following = next_header_pattern.search(note_text, pos=ros_start + 5)
    ros_end = following.start() if following else len(note_text)
    non_ros = note_text[:ros_start] + note_text[ros_end:]
    ros_section = note_text[ros_start:ros_end]
    return non_ros, ros_section


def symptom_present_outside_ros(
    note_text: str,
    symptom: str,
    direct_vocab: Mapping[str, Sequence[str]],
) -> bool:
    non_ros, _ = split_note_ros(note_text)
    non_ros_lower = non_ros.lower()
    for term in direct_vocab.get(symptom, []):
        for start, end in find_phrase_spans(term, non_ros_lower):
            if not mention_is_negated(non_ros_lower, start, end):
                return True
    return False


FH_MEMBER_PATTERN = (
    r"\b(?:mother|father|parent|sister|brother|sibling|relative|"
    r"family member|first[- ]degree relative|dad|mom|grandmother|"
    r"grandfather|aunt|uncle|son|daughter)\b"
)
FH_CRC_PATTERN = (
    r"(?:\bcrc\b|\b(?:colorectal|colon|rectal|bowel)\b.{0,20}"
    r"\b(?:cancer|ca|carcinoma)\b)"
)
FH_PATTERNS = [
    rf"{FH_MEMBER_PATTERN}[^.;]{{0,100}}{FH_CRC_PATTERN}",
    rf"{FH_CRC_PATTERN}[^.;]{{0,100}}{FH_MEMBER_PATTERN}",
]


def explicit_family_history_crc(note_text: str) -> bool:
    non_ros, _ = split_note_ros(note_text)
    text = non_ros.lower()
    return any(re.search(pattern, text) for pattern in FH_PATTERNS)


def build_alias_section(
    direct_vocab: Mapping[str, Sequence[str]],
    max_terms_per_finding: int = 15,
) -> str:
    limit = max(0, int(max_terms_per_finding))
    if limit == 0:
        return "  Omitted by context-budget policy."
    lines = []
    for symptom in SYMPTOMS:
        terms = list(direct_vocab.get(symptom, []))[:limit]
        lines.append(f"{symptom}: {', '.join(terms)}")
    return "\n".join(lines)


def build_group_b_section(
    group_b: Mapping[str, str],
    max_terms: int = MAX_GROUP_B_IN_PROMPT,
) -> str:
    limit = max(0, int(max_terms))
    if not group_b or limit == 0:
        return "  None supplied in this prompt."
    return "\n".join(
        f"  - {name} (matched note phrase: '{surface_form}')"
        for name, surface_form in list(group_b.items())[:limit]
    )


def build_main_prompt(
    note_text: str,
    alias_section: str,
    group_b_section: str,
    evidence_section: str,
) -> str:
    key_listing = "\n".join(f'"{key}"' for key in EXPECTED_OUTPUT_KEYS)
    return f"""You are an experienced gastroenterology clinician.
Extract seven target findings from the patient's clinical note.

PATIENT-EVIDENCE REQUIREMENT:
- Patient-specific labels must be based ONLY on the Patient NOTE TEXT.
- Retrieved biomedical passages and terminology lists may help recognize
  equivalent wording, but they CANNOT establish or negate a patient symptom.
- Every positive inference must quote or closely paraphrase evidence from the
  Patient NOTE TEXT. Never quote or paraphrase retrieved literature as the
  patient inference.
- Do not make external assumptions from diagnoses, medications, risk factors,
  or general medical knowledge.

REVIEW OF SYSTEMS (ROS) RULE:
- ROS may contain templated negatives that conflict with narrative sections.
- If ROS denies a symptom but Chief Complaint, HPI, Assessment/Plan, or
  Diagnosis affirms it, answer "Yes" and cite the NON-ROS evidence.
- If the note outside ROS explicitly denies the symptom, answer "No" and cite
  the NON-ROS denial.
- If the symptom appears only as a negative in ROS and nowhere else, answer
  "No".
- Absence of a denial is not evidence that a symptom is present.

FAMILY-HISTORY RULE:
- Answer "Yes" only when the Patient NOTE TEXT explicitly links a biological
  family member to colorectal, colon, rectal, or bowel cancer.
- The patient's personal cancer history is not family history.
- Lynch syndrome, FAP, genetic testing, hereditary-risk discussion, or a
  generic family history of unspecified cancer is insufficient by itself.

OUTPUT RULES:
- Every target-finding label must be exactly "Yes" or "No".

STRUCTURED METADATA (required for each symptom):
For each target finding, in addition to label, confidence, and inference, also report:
- note_section: which note section contains the primary evidence. Use one of:
  HPI, CC, ROS, Assessment, Plan, A/P, PMH, FH, PE, Medications,
  Surgical History, Social History, Diagnosis, Impression, Hospital Course,
  Other, Unknown, None.
  Use "None" when no relevant text exists for a "No" answer.
- experiencer: who has the finding. Use one of:
  patient, family_member, other, unknown.
  For the six patient symptoms, this should almost always be "patient".
  For family history, a "Yes" answer should have "family_member".
- assertion_status: the assertion status of the finding. Use one of:
  affirmed, negated, hypothetical, planned, historical, absent, unknown.
  A "Yes" answer should normally have "affirmed" or "historical".
  A "No" answer should have "negated" or "absent".

CONFIDENCE RUBRIC:
- Confidence reflects ONLY how clearly the note supports your answer.
  It must NOT reflect which rule above you applied.
  5 = the note states the finding explicitly and unambiguously
  4 = clearly stated but with minor wording ambiguity or abbreviation
  3 = indirect evidence requiring clinical inference
  2 = weak, conflicting, or single-token evidence
  1 = essentially no usable evidence either way
- Use the full 1-5 range. Do not default to 5.
- Confidence must be an integer from 1 to 5.
- Every inference, duration, and Other comments field must be a string.
- Use "N/A" when duration or Other comments are not reported.
- Output ONE flat JSON object. Do not use nested dictionaries.
- Include every key listed below exactly once.
- Output only JSON, without prose or markdown fences.

REQUIRED KEYS:
{key_listing}

PATIENT NOTE TEXT:
<NOTE>
{note_text}
</NOTE>

TERMINOLOGY GUIDANCE:
These direct aliases help recognize equivalent note wording. They are not
patient evidence by themselves.
{alias_section}

RELATED UMLS CONCEPTS DETECTED IN THE NOTE:
These are terminology hints only. A related concept does not establish a
specific target finding.
{group_b_section}

EXTERNAL BIOMEDICAL CONTEXT:
The passages below were selected using MedCPT retrieval and within-finding RRF.
When a finalized applicable formal GRADE evidence body is available, a passage
may also carry PARENT_BODY_GRADE_CERTAINTY through explicit
chunk-to-study/PMID-to-evidence-body provenance. UNMAPPED means that no valid
parent GRADE body was linked; it does NOT mean low-quality evidence. Agent 1
never infers GRADE from sentence wording. The configured annotate/prioritize/
filter behavior is a VERGE retrieval policy, not a GRADE rule.
These passages are terminology/background context ONLY. GRADE, RRF, cosine
similarity, and literature content cannot determine or override a patient
label and cannot be used as the supporting quotation for a patient claim.
{evidence_section}
""".strip()



def _format_prompt_for_counting(prompt: str, tokenizer: Any) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"<|user|>\n{prompt}\n<|assistant|>\n"


def count_prompt_tokens(prompt: str, tokenizer: Any) -> int:
    formatted = _format_prompt_for_counting(prompt, tokenizer)
    encoded = tokenizer(formatted, add_special_tokens=True, truncation=False)
    return len(encoded["input_ids"])


def fit_main_prompt_to_context(
    note_text: str,
    direct_vocab: Mapping[str, Sequence[str]],
    group_b: Mapping[str, str],
    formal_grade_audit: Mapping[str, Sequence[Mapping[str, Any]]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int = MAIN_MAX_NEW_TOKENS,
) -> Tuple[str, Dict[str, Any]]:
    context_limit = _model_context_limit(tokenizer, model)
    attempts: List[Dict[str, Any]] = []
    for mode, alias_limit, group_b_limit, passage_limit in CONTEXT_BUDGET_MODES:
        alias_section = build_alias_section(
            direct_vocab, max_terms_per_finding=alias_limit
        )
        group_b_section = build_group_b_section(group_b, max_terms=group_b_limit)
        evidence_section = render_evidence_section_from_audit(
            formal_grade_audit, per_finding_limit=passage_limit
        )
        prompt = build_main_prompt(
            note_text, alias_section, group_b_section, evidence_section
        )
        prompt_tokens = count_prompt_tokens(prompt, tokenizer)
        required_tokens = (
            prompt_tokens + int(max_new_tokens) + MODEL_CONTEXT_SAFETY_TOKENS
        )
        attempt = {
            "mode": mode,
            "alias_terms_per_finding": alias_limit,
            "group_b_terms": group_b_limit,
            "passages_per_finding": passage_limit,
            "prompt_tokens": prompt_tokens,
            "requested_output_tokens": int(max_new_tokens),
            "safety_tokens": MODEL_CONTEXT_SAFETY_TOKENS,
            "context_limit": context_limit,
            "fits": required_tokens <= context_limit,
        }
        attempts.append(attempt)
        if attempt["fits"]:
            return prompt, {
                "selected_mode": mode,
                "attempts": attempts,
                "note_truncated": False,
                "tokenizer_truncation": False,
            }

    raise ValueError(
        "FULL_NOTE_PLUS_MINIMAL_PROMPT_EXCEEDS_MODEL_CONTEXT: even NOTE_ONLY "
        "mode cannot fit the complete clinical note plus output reserve. The note "
        "was NOT truncated. Use a model/context configuration with more capacity. "
        f"Attempts={json.dumps(attempts, ensure_ascii=False)}"
    )


def build_grounding_retry_prompt(
    note_text: str,
    parsed: Mapping[str, Any],
    weak_symptoms: Sequence[str],
) -> str:
    current: Dict[str, Any] = {}
    required_keys: List[str] = []
    for symptom in weak_symptoms:
        current[symptom] = {
            "answer": parsed.get(symptom),
            "confidence": parsed.get(f"{symptom} confidence"),
            "inference": parsed.get(f"{symptom} inference"),
        }
        required_keys.extend([
            symptom,
            f"{symptom} confidence",
            f"{symptom} inference",
            f"{symptom} note_section",
            f"{symptom} experiencer",
            f"{symptom} assertion_status",
        ])
    return f"""Review only the target findings listed below.
Use ONLY the patient note. Do not use retrieved literature, terminology lists,
medical assumptions, related diagnoses, medications, or general knowledge.
For each target finding:
- Answer "Yes" only if the note explicitly supports the finding for this
  patient.
- The inference must quote or closely paraphrase an exact note statement.
- If the preserved label is "No" and no explicit denial exists, use confidence 2 and
  inference "Not documented in the note".
- For family history of colorectal cancer, "Yes" requires a family member
  linked to colorectal, colon, rectal, or bowel cancer.
- Include note_section, experiencer, and assertion_status for each symptom.
- Output one flat JSON object containing exactly these keys:
{json.dumps(required_keys, ensure_ascii=False)}

CURRENT PREDICTIONS REQUIRING REVIEW:
{json.dumps(current, ensure_ascii=False)}

PATIENT NOTE:
<NOTE>
{note_text}
</NOTE>
""".strip()


def build_missing_label_prompt(
    note_text: str,
    missing_findings: Sequence[str],
) -> str:
    """Recover only explicitly missing Yes/No labels from the full note.

    This deliberately asks for no confidence, inference, or metadata fields.
    It prevents a model from hiding the label inside an inference field and
    prevents the parser from ever inferring a label from confidence/evidence.
    """
    required = {finding: "Yes or No" for finding in missing_findings}
    return f"""Some binary finding labels were omitted from a prior extraction.
Using ONLY the FULL patient note, return the missing labels.

RULES:
- Return exactly one flat JSON object.
- Include exactly the requested finding names as keys.
- Each value must be exactly "Yes" or "No".
- Do not return confidence, evidence, metadata, duration, explanation, markdown,
  or nested dictionaries.
- "Yes" requires explicit patient-specific support in the note.
- For family history of colorectal cancer, "Yes" requires a biological
  family member explicitly linked to colorectal, colon, rectal, or bowel cancer.

REQUIRED OUTPUT SHAPE:
{json.dumps(required, ensure_ascii=False)}

FULL PATIENT NOTE:
<NOTE>
{note_text}
</NOTE>
""".strip()


def build_fixed_label_metadata_prompt(
    note_text: str,
    symptom: str,
    fixed_label: str,
    strict_retry: bool = False,
) -> str:
    """Repair one finding's metadata while freezing its accepted label.

    The output deliberately omits the binary label, so this repair cannot alter
    label provenance or create a new label from metadata text.
    """
    if fixed_label not in {"Yes", "No"}:
        raise ValueError("fixed_label must be Yes or No")

    keys = [
        f"{symptom} confidence",
        f"{symptom} inference",
        f"{symptom} note_section",
        f"{symptom} experiencer",
        f"{symptom} assertion_status",
    ]
    if symptom != "Family history of colorectal cancer":
        duration = f"Duration of {symptom.lower()}"
        keys.extend([duration, f"{duration} confidence", f"{duration} inference"])

    if fixed_label == "Yes":
        label_rules = f"""
- The fixed label is Yes. Return a direct note-supported evidence statement for
  the target finding itself. Do not infer it solely from a diagnosis, medication,
  procedure, laboratory result, test, risk factor, or related finding.
- assertion_status must be affirmed or historical.
- experiencer must be {'family_member' if symptom == 'Family history of colorectal cancer' else 'patient'}.
- For family history of colorectal cancer, the evidence must explicitly identify
  a biological family relation AND colorectal/colon/rectal/bowel cancer.
"""
    else:
        label_rules = """
- The fixed label is No. Do NOT convert it to Yes.
- If the note explicitly denies the target finding, quote or closely paraphrase
  that denial and use assertion_status=negated.
- Otherwise use inference="Not documented in the note", note_section=none,
  experiencer=unknown, and assertion_status=absent.
- Do not use an affirmative statement about a different symptom, cancer type,
  diagnosis, medication, procedure, laboratory result, or risk factor as the
  inference for this negative target finding.
"""

    retry_text = (
        "STRICT RETRY: The previous metadata response was invalid. Follow the exact flat schema and fixed-label rules.\n"
        if strict_retry else ""
    )

    return f"""{retry_text}You are repairing metadata for ONE fixed clinical extraction label.
Use ONLY the FULL patient note.

TARGET FINDING: {symptom}
FIXED BINARY LABEL: {fixed_label}
The binary label is already accepted and MUST NOT be reconsidered, repeated, or changed.
{label_rules}
GENERAL RULES:
- Output one flat JSON object containing exactly the keys listed below.
- Do not output the target-finding label itself.
- confidence must be an integer 1-5 reflecting only note support for the fixed label.
- note_section must be one of: HPI, CC, ROS, Assessment, Plan, A/P, PMH, FH, PE,
  Medications, Surgical History, Social History, Diagnosis, Impression, Hospital
  Course, Other, Unknown, None.
- experiencer must be patient, family_member, other, or unknown.
- assertion_status must be affirmed, negated, hypothetical, planned, historical,
  absent, or unknown.
- For patient symptoms, use N/A for duration when it is not documented.
- Evidence/inference must come only from this note.
- No markdown, explanation, nested dictionaries, or extra keys.

REQUIRED KEYS:
{json.dumps(keys, ensure_ascii=False)}

FULL PATIENT NOTE:
<NOTE>
{note_text}
</NOTE>
""".strip()


def build_missing_block_prompt(
    note_text: str,
    missing_symptoms: Sequence[str],
    current_labels: Mapping[str, str],
) -> str:
    """Regenerate omitted finding blocks using only the patient note.

    For the six patient symptoms, duration fields are included so a repair does
    not silently lose documented duration information.
    """
    required_keys: List[str] = []
    for symptom in missing_symptoms:
        required_keys.extend([
            symptom,
            f"{symptom} confidence",
            f"{symptom} inference",
            f"{symptom} note_section",
            f"{symptom} experiencer",
            f"{symptom} assertion_status",
        ])
        if symptom != "Family history of colorectal cancer":
            duration = f"Duration of {symptom.lower()}"
            required_keys.extend([
                duration,
                f"{duration} confidence",
                f"{duration} inference",
            ])

    return f"""The following target-finding blocks were omitted from a prior JSON
extraction. Regenerate only the missing auxiliary fields using ONLY the patient note.
The binary labels below are already explicit and MUST NOT be changed.
CURRENT LABELS TO PRESERVE:
{json.dumps(dict(current_labels), ensure_ascii=False)}

For each finding:
- Repeat exactly the current "Yes" or "No" label shown above.
- Do not reinterpret or flip the label in this metadata-completion repair.
- For a preserved "Yes" label, return note-supported evidence/metadata.
- Copy or closely paraphrase note evidence in the inference field.
- If the preserved label is "No" and no explicit denial exists, use confidence 2 and
  inference "Not documented in the note".
- Confidence must be an integer from 1 to 5.
- Include note_section, experiencer, and assertion_status.
- For the six patient symptoms, also return duration, duration confidence, and
  duration inference; use "N/A" when duration is not documented.
- Family history of colorectal cancer has no duration fields.
- Output one flat JSON object containing exactly these keys and nothing else:
{json.dumps(required_keys, ensure_ascii=False)}

PATIENT NOTE:
<NOTE>
{note_text}
</NOTE>
""".strip()


def build_full_note_only_repair_prompt(note_text: str) -> str:
    """Last-resort parse repair; explicitly audited when used."""
    key_listing = "\n".join(f'"{key}"' for key in EXPECTED_OUTPUT_KEYS)
    return f"""Extract all seven target findings using ONLY the patient note.
Return one compact, flat, valid JSON object.
Rules:
- Each target-finding label must be exactly "Yes" or "No".
- "Yes" requires explicit patient-specific note evidence.
- Inference must quote or closely paraphrase the note.
- Confidence must be an integer from 1 to 5.
- Use "N/A" for unreported durations and Other comments.
- Include note_section, experiencer, and assertion_status for each symptom.
- Do not use nested dictionaries.
- Output only JSON.
Required keys:
{key_listing}

PATIENT NOTE:
<NOTE>
{note_text}
</NOTE>
""".strip()


def strip_code_fences(text: str) -> str:
    value = str(text).strip()
    if value.startswith("```"):
        value = re.sub(
            r"^```(?:json)?", "", value, flags=re.IGNORECASE
        ).strip()
        if value.endswith("```"):
            value = value[:-3].strip()
    return value


def extract_first_json(text: str) -> str:
    """Extract the first balanced JSON object, respecting quoted strings."""
    value = strip_code_fences(text)
    start = value.find("{")
    if start < 0:
        return value
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(value)):
        character = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return value[start : index + 1]
    return value[start:]


def safe_json_loads(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    raw = extract_first_json(str(text))
    attempts = [raw]
    normalized = (
        raw.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    normalized = re.sub(
        r':\s*N/A(\s*[,\n}\]])', r': "N/A"\1', normalized
    )
    normalized = re.sub(
        r':\s*None(\s*[,\n}\]])', r': "None"\1', normalized
    )
    normalized = re.sub(r",\s*([}\]])", r"\1", normalized)
    normalized = re.sub(
        r'("|\d|true|false|null)\s*\n(\s*")', r"\1,\n\2", normalized
    )
    attempts.append(normalized)
    missing_comma_fix = re.sub(
        r'("|\d|true|false|null)(\s*")', r"\1,\2", normalized
    )
    attempts.append(missing_comma_fix)
    for candidate in attempts:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, candidate
        except (json.JSONDecodeError, TypeError):
            continue
    return None, raw


def parse_saved_dict(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed, _ = safe_json_loads(text)
    if isinstance(parsed, dict):
        return parsed
    try:
        literal = ast.literal_eval(text)
        return literal if isinstance(literal, dict) else None
    except (ValueError, SyntaxError):
        return None


def normalize_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"yes", "1", "true", "present", "y"}:
        return "Yes"
    if normalized in {
        "no", "0", "false", "absent", "n",
        "not mentioned", "not mentioned outside ros",
    }:
        return "No"
    return None


def flatten_parsed(parsed: Any) -> Any:
    """Recover nested model outputs without inventing labels."""
    if not isinstance(parsed, dict):
        return parsed
    result = dict(parsed)
    for symptom in SYMPTOMS:
        value = parsed.get(symptom)
        confidence_key = f"{symptom} confidence"
        inference_key = f"{symptom} inference"
        if isinstance(value, str):
            normalized = normalize_label(value)
            if normalized is not None:
                result[symptom] = normalized
            continue
        if not isinstance(value, dict):
            continue
        explicit_answer: Optional[str] = None
        candidate_keys = [
            symptom, symptom.lower(),
            "answer", "presence", "result", "label",
        ]
        for key in candidate_keys:
            if key in value:
                explicit_answer = normalize_label(value[key])
                if explicit_answer is not None:
                    break
        if explicit_answer is not None:
            result[symptom] = explicit_answer
        else:
            result.pop(symptom, None)

        if confidence_key not in result or pd.isna(result.get(confidence_key)):
            for key, nested_value in value.items():
                if "confidence" in str(key).lower():
                    result[confidence_key] = nested_value
                    break
        current_inference = result.get(inference_key)
        current_bad = (
            not isinstance(current_inference, str)
            or current_inference.strip().lower() in BAD_INFERENCE_VALS
            or len(current_inference.strip()) < 5
        )
        if current_bad:
            for key, nested_value in value.items():
                if "inference" in str(key).lower() and isinstance(
                    nested_value, str
                ):
                    result[inference_key] = nested_value
                    break
        # Also extract metadata from nested dicts.
        for meta_key in METADATA_KEYS:
            full_key = f"{symptom} {meta_key}"
            if full_key not in result or not result.get(full_key):
                for key, nested_value in value.items():
                    if meta_key in str(key).lower() and isinstance(
                        nested_value, str
                    ):
                        result[full_key] = nested_value
                        break
    return result


def to_num(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else np.nan


def missing_symptom_blocks(parsed: Any) -> List[str]:
    if not isinstance(parsed, dict):
        return list(SYMPTOMS)
    missing = []
    for symptom in SYMPTOMS:
        if normalize_label(parsed.get(symptom)) is None:
            missing.append(symptom)
    return missing


def complete_auxiliary_fields(parsed: Any) -> Any:
    if not isinstance(parsed, dict):
        return parsed
    result = dict(parsed)

    def normalize_aux_confidence(value: Any) -> int | float:
        confidence = to_num(value)
        if np.isnan(confidence) or confidence < 1 or confidence > 5:
            return 1
        if float(confidence).is_integer():
            return int(confidence)
        return confidence

    for symptom in SYMPTOMS[:-1]:
        duration = f"Duration of {symptom.lower()}"
        duration_confidence = f"{duration} confidence"
        duration_inference = f"{duration} inference"
        duration_value = result.get(duration)
        if not isinstance(duration_value, str) or not duration_value.strip():
            result[duration] = "N/A"
        result[duration_confidence] = normalize_aux_confidence(
            result.get(duration_confidence)
        )
        inference_value = result.get(duration_inference)
        if not isinstance(inference_value, str) or not inference_value.strip():
            result[duration_inference] = "N/A"

    comments_value = result.get("Other comments")
    if not isinstance(comments_value, str) or not comments_value.strip():
        result["Other comments"] = "N/A"
    result["Other comments confidence"] = normalize_aux_confidence(
        result.get("Other comments confidence")
    )
    comments_inference = result.get("Other comments inference")
    if not isinstance(comments_inference, str) or not comments_inference.strip():
        result["Other comments inference"] = "N/A"

    for symptom in SYMPTOMS:
        section_key = f"{symptom} note_section"
        experiencer_key = f"{symptom} experiencer"
        assertion_key = f"{symptom} assertion_status"

        section_val = str(result.get(section_key, "")).strip().lower()
        section_val = NOTE_SECTION_ALIASES.get(section_val, section_val)
        if section_val not in VALID_NOTE_SECTIONS:
            result[section_key] = "unknown"
        else:
            result[section_key] = section_val

        exp_val = str(result.get(experiencer_key, "")).strip().lower()
        if exp_val not in VALID_EXPERIENCERS:
            result[experiencer_key] = "unknown"
        else:
            result[experiencer_key] = exp_val

        assert_val = str(result.get(assertion_key, "")).strip().lower()
        assert_val = ASSERTION_STATUS_ALIASES.get(assert_val, assert_val)
        if assert_val not in VALID_ASSERTION_STATUSES:
            result[assertion_key] = "unknown"
        else:
            result[assertion_key] = assert_val

    return result


def _normalize_grounding_text(text: Any) -> str:
    value = str(text or "").lower().strip()
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    value = value.replace("\u2018", "'").replace("\u2019", "'")
    return re.sub(r"\s+", " ", value)


def _inference_is_grounded(note_text: str, inference: str) -> bool:
    inf = str(inference or "").strip()
    if len(inf) <= 5:
        return False
    n = _normalize_grounding_text(note_text)
    q = _normalize_grounding_text(inf)
    if q and q in n:
        return True
    try:
        return compute_bleu_no_bp(note_text, inf) >= BLEU_THRESHOLD
    except Exception:
        return False


def metadata_consistency_issues(
    parsed: Mapping[str, Any],
    symptom: str,
    note_text: str,
) -> List[str]:
    issues: List[str] = []
    label = normalize_label(parsed.get(symptom))
    if label not in {"Yes", "No"}:
        return ["invalid_or_missing_label"]

    confidence = to_num(parsed.get(f"{symptom} confidence"))
    if np.isnan(confidence) or confidence < 1 or confidence > 5:
        issues.append("invalid_confidence")

    inference = str(parsed.get(f"{symptom} inference", "") or "").strip()
    section = str(parsed.get(f"{symptom} note_section", "") or "").strip().lower()
    section = NOTE_SECTION_ALIASES.get(section, section)
    experiencer = str(parsed.get(f"{symptom} experiencer", "") or "").strip().lower()
    assertion = str(parsed.get(f"{symptom} assertion_status", "") or "").strip().lower()
    assertion = ASSERTION_STATUS_ALIASES.get(assertion, assertion)

    if section not in VALID_NOTE_SECTIONS:
        issues.append("invalid_note_section")
    if experiencer not in VALID_EXPERIENCERS:
        issues.append("invalid_experiencer")
    if assertion not in VALID_ASSERTION_STATUSES:
        issues.append("invalid_assertion_status")

    is_fh = symptom == "Family history of colorectal cancer"
    if label == "Yes":
        if assertion not in {"affirmed", "historical"}:
            issues.append("positive_label_requires_affirmed_or_historical_assertion")
        expected_experiencer = "family_member" if is_fh else "patient"
        if experiencer != expected_experiencer:
            issues.append(f"positive_label_requires_{expected_experiencer}_experiencer")
        if (
            not inference
            or inference.lower() in BAD_INFERENCE_VALS
            or not _inference_is_grounded(note_text, inference)
        ):
            issues.append("positive_inference_not_note_grounded")
        if is_fh and inference:
            text = inference.lower()
            relation_ok = bool(re.search(FH_MEMBER_PATTERN, text))
            cancer_ok = bool(re.search(FH_CRC_PATTERN, text))
            if not relation_ok:
                issues.append("family_history_positive_missing_biological_relation")
            if not cancer_ok:
                issues.append("family_history_positive_missing_crc_specificity")
    else:
        if assertion not in {"negated", "absent", "unknown"}:
            issues.append("negative_label_cannot_have_affirmed_hypothetical_planned_or_historical_assertion")
        if assertion == "negated":
            if (
                not inference
                or inference.lower() in BAD_INFERENCE_VALS
                or not _inference_is_grounded(note_text, inference)
            ):
                issues.append("negated_label_requires_grounded_denial_evidence")
        else:
            generic_absence = (
                not inference
                or inference.lower() in BAD_INFERENCE_VALS
                or inference.lower().startswith("not documented")
                or inference.lower().startswith("no documentation")
                or inference.lower().startswith("not mentioned")
                or inference.lower().startswith("no evidence")
            )
            if not generic_absence:
                issues.append("negative_absence_should_not_retain_affirmative_or_unrelated_inference")

    if symptom != "Family history of colorectal cancer":
        duration = f"Duration of {symptom.lower()}"
        if not isinstance(parsed.get(duration), str):
            issues.append("duration_not_string")
        dconf = to_num(parsed.get(f"{duration} confidence"))
        if np.isnan(dconf) or dconf < 1 or dconf > 5:
            issues.append("invalid_duration_confidence")
        if not isinstance(parsed.get(f"{duration} inference"), str):
            issues.append("duration_inference_not_string")

    return issues


def normalize_negative_metadata_deterministically(
    parsed: Mapping[str, Any],
    symptom: str,
    note_text: str,
) -> Tuple[Dict[str, Any], List[str], bool]:
    result = dict(parsed)
    if normalize_label(result.get(symptom)) != "No":
        return result, [], False

    before_issues = metadata_consistency_issues(result, symptom, note_text)
    if not before_issues:
        return result, [], False

    changes: List[str] = []
    confidence_key = f"{symptom} confidence"
    inference_key = f"{symptom} inference"
    section_key = f"{symptom} note_section"
    experiencer_key = f"{symptom} experiencer"
    assertion_key = f"{symptom} assertion_status"

    confidence = to_num(result.get(confidence_key))
    if np.isnan(confidence) or confidence < 1 or confidence > 5:
        result[confidence_key] = 1
        changes.append("invalid_confidence->1")

    inference = str(result.get(inference_key, "") or "").strip()
    section = str(result.get(section_key, "") or "").strip().lower()
    section = NOTE_SECTION_ALIASES.get(section, section)
    experiencer = str(result.get(experiencer_key, "") or "").strip().lower()
    assertion = str(result.get(assertion_key, "") or "").strip().lower()
    assertion = ASSERTION_STATUS_ALIASES.get(assertion, assertion)
    denial_grounded = bool(
        assertion == "negated"
        and inference
        and inference.lower() not in BAD_INFERENCE_VALS
        and _inference_is_grounded(note_text, inference)
    )

    if denial_grounded:
        if assertion != str(result.get(assertion_key, "") or "").strip().lower():
            changes.append("assertion_alias_normalized")
        result[assertion_key] = "negated"
        if section not in VALID_NOTE_SECTIONS:
            result[section_key] = "unknown"
            changes.append("invalid_section->unknown")
        else:
            result[section_key] = section
        if experiencer not in VALID_EXPERIENCERS:
            result[experiencer_key] = "unknown"
            changes.append("invalid_experiencer->unknown")
        else:
            result[experiencer_key] = experiencer
    else:
        generic_inference = "Not documented in the note"
        if inference != generic_inference:
            result[inference_key] = generic_inference
            changes.append("unsupported_negative_inference->not_documented")
        if section != "none":
            result[section_key] = "none"
            changes.append("negative_section->none")
        else:
            result[section_key] = "none"
        if experiencer != "unknown":
            result[experiencer_key] = "unknown"
            changes.append("negative_experiencer->unknown")
        else:
            result[experiencer_key] = "unknown"
        if assertion != "unknown":
            result[assertion_key] = "unknown"
            changes.append("negative_assertion->unknown")
        else:
            result[assertion_key] = "unknown"

    if symptom != "Family history of colorectal cancer":
        duration = f"Duration of {symptom.lower()}"
        duration_conf = f"{duration} confidence"
        duration_inf = f"{duration} inference"
        if result.get(duration) != "N/A":
            result[duration] = "N/A"
            changes.append("negative_duration->N/A")
        else:
            result[duration] = "N/A"
        dconf = to_num(result.get(duration_conf))
        if np.isnan(dconf) or dconf < 1 or dconf > 5 or dconf != 1:
            result[duration_conf] = 1
            changes.append("negative_duration_confidence->1")
        else:
            result[duration_conf] = 1
        if result.get(duration_inf) != "N/A":
            result[duration_inf] = "N/A"
            changes.append("negative_duration_inference->N/A")
        else:
            result[duration_inf] = "N/A"

    result = complete_auxiliary_fields(result)
    remaining = metadata_consistency_issues(result, symptom, note_text)
    normalized = len(remaining) == 0
    return result, changes, normalized


def missing_expected_keys(parsed: Any) -> List[str]:
    if not isinstance(parsed, dict):
        return list(EXPECTED_OUTPUT_KEYS)
    return [key for key in EXPECTED_OUTPUT_KEYS if key not in parsed]


def invalid_output_values(parsed: Any) -> List[str]:
    if not isinstance(parsed, dict):
        return ["Output is not a JSON object."]
    issues: List[str] = []
    for symptom, confidence_key, inference_key in SYMPTOM_SPECS:
        label = normalize_label(parsed.get(symptom))
        if label is None:
            issues.append(f"{symptom}: invalid label {parsed.get(symptom)!r}")
        confidence = to_num(parsed.get(confidence_key))
        if np.isnan(confidence) or confidence < 1 or confidence > 5:
            issues.append(
                f"{confidence_key}: invalid confidence "
                f"{parsed.get(confidence_key)!r}"
            )
        if not isinstance(parsed.get(inference_key), str):
            issues.append(f"{inference_key}: inference is not a string")
    for key in EXPECTED_OUTPUT_KEYS:
        if key.endswith(" confidence"):
            confidence = to_num(parsed.get(key))
            if np.isnan(confidence) or confidence < 1 or confidence > 5:
                issue = f"{key}: invalid confidence {parsed.get(key)!r}"
                if issue not in issues:
                    issues.append(issue)
        elif key not in SYMPTOMS and key not in [
            f"{symptom} inference" for symptom in SYMPTOMS
        ]:
            if not isinstance(parsed.get(key), str):
                issues.append(f"{key}: value is not a string")
    return issues

_TOKEN_PATTERN = re.compile(r"\w+|\S")


def simple_tokenize(text: Any) -> List[str]:
    if not isinstance(text, str):
        text = "" if text is None or pd.isna(text) else str(text)
    return _TOKEN_PATTERN.findall(text.lower())


def modified_precision(
    reference_tokens: Sequence[str],
    hypothesis_tokens: Sequence[str],
    n: int,
) -> float:
    if len(hypothesis_tokens) < n:
        return 0.0
    hypothesis_ngrams = Counter(
        zip(*[hypothesis_tokens[offset:] for offset in range(n)])
    )
    reference_ngrams = Counter(
        zip(*[reference_tokens[offset:] for offset in range(n)])
    )
    matches = sum(
        min(count, reference_ngrams[ngram])
        for ngram, count in hypothesis_ngrams.items()
    )
    total = sum(hypothesis_ngrams.values())
    return 0.0 if total == 0 else matches / total


def compute_bleu_no_bp(
    reference: Any, hypothesis: Any, max_n: int = 4
) -> float:
    reference_tokens = simple_tokenize(reference)
    hypothesis_tokens = simple_tokenize(hypothesis)
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    effective_max_n = min(max_n, len(hypothesis_tokens))
    log_precisions: List[float] = []
    for n in range(1, effective_max_n + 1):
        precision = modified_precision(
            reference_tokens, hypothesis_tokens, n
        )
        if precision == 0.0:
            return 0.0
        log_precisions.append(math.log(precision))
    return math.exp(sum(log_precisions) / len(log_precisions))


def compute_bertscore_batch(
    references: Sequence[str], hypotheses: Sequence[str]
) -> List[float]:
    if not references:
        return []
    if not HAVE_BERTSCORE:
        return [np.nan] * len(references)
    precision, _, _ = bert_score_fn(
        list(hypotheses), list(references),
        lang="en", model_type="roberta-large",
        rescale_with_baseline=False, batch_size=16, verbose=False,
    )
    return [float(value) for value in precision]

def load_llama(model_id: str = HF_MODEL_ID) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading extractor model: {model_id}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {total_memory:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=HF_CACHE_DIR
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
    )
    model.eval()
    print("LLaMA extractor loaded")
    return tokenizer, model


def _binary_entropy(p_yes_norm: float) -> float:
    p = min(max(float(p_yes_norm), 1e-12), 1.0 - 1e-12)
    q = 1.0 - p
    return -(p * math.log(p) + q * math.log(q))


def _invalid_label_probability(source: str, reason: str) -> Dict[str, Any]:
    return {
        "p_yes": -1.0,
        "p_no": -1.0,
        "p_yes_norm": -1.0,
        "p_no_norm": -1.0,
        "label_entropy": -1.0,
        "yes_minus_no_logit_margin": 0.0,
        "entropy_unit": "nats",
        "probability_scope": "generation-time first divergent Yes/No literal token",
        "full_label_probability_exact": False,
        "generated_label": "",
        "binary_argmax_label": "",
        "generated_label_matches_binary_argmax": False,
        "chosen_label_binary_preference": -1.0,
        "probability_source": source,
        "probability_valid": False,
        "probability_reason": reason,
    }


def _first_divergent_token(
    tokenizer: Any,
    prefix: str,
    generated_token_ids: Sequence[int],
) -> Tuple[
    Optional[int], Optional[int], Optional[int], Tuple[int, ...], Tuple[int, ...]
]:
    yes_ids = [int(v) for v in tokenizer.encode(prefix + "Yes", add_special_tokens=False)]
    no_ids = [int(v) for v in tokenizer.encode(prefix + "No", add_special_tokens=False)]
    common = 0
    limit = min(len(yes_ids), len(no_ids))
    while common < limit and yes_ids[common] == no_ids[common]:
        common += 1
    if common >= len(yes_ids) or common >= len(no_ids):
        return None, None, None, (), ()

    actual = [int(v) for v in generated_token_ids]
    if common > len(actual) or actual[:common] != yes_ids[:common]:
        return None, None, None, (), ()

    yes_suffix = tuple(yes_ids[common:])
    no_suffix = tuple(no_ids[common:])
    return common, yes_suffix[0], no_suffix[0], yes_suffix, no_suffix


def compute_generation_label_probs(
    generated_text: str,
    generated_token_ids: Sequence[int],
    generation_scores: Sequence[Any],
    tokenizer: Any,
    features: Sequence[str],
    source: str,
) -> Dict[str, Dict[str, Any]]:
    import torch

    results: Dict[str, Dict[str, Any]] = {}
    text = str(generated_text or "")
    token_ids = [int(v) for v in generated_token_ids]

    for feature in features:
        pattern = re.compile(
            r'"' + re.escape(feature) + r'"\s*:\s*"(Yes|No)"',
            flags=re.IGNORECASE,
        )
        match = pattern.search(text)
        if not match:
            results[feature] = _invalid_label_probability(
                source, "label field not found as an explicit quoted Yes/No literal in generated text"
            )
            continue

        generated_label = "Yes" if match.group(1).lower() == "yes" else "No"
        label_char_start = match.start(1)
        prefix = text[:label_char_start]
        step, yes_id, no_id, yes_suffix, no_suffix = _first_divergent_token(
            tokenizer, prefix, token_ids
        )
        if step is None or yes_id is None or no_id is None:
            results[feature] = _invalid_label_probability(
                source,
                "could not exactly align counterfactual Yes/No tokenization with the actual generated prefix",
            )
            results[feature]["generated_label"] = generated_label
            continue
        if step >= len(generation_scores) or step >= len(token_ids):
            results[feature] = _invalid_label_probability(
                source, "divergent label-token step is outside captured generation scores"
            )
            results[feature]["generated_label"] = generated_label
            continue

        actual_id = int(token_ids[step])
        expected_generated_id = int(yes_id if generated_label == "Yes" else no_id)
        if actual_id != expected_generated_id:
            results[feature] = _invalid_label_probability(
                source,
                "decoded label literal does not match the actual token at the aligned divergence step",
            )
            results[feature].update(
                {
                    "generated_label": generated_label,
                    "token_step": int(step),
                    "actual_token_id": actual_id,
                    "expected_generated_token_id": expected_generated_id,
                    "yes_token_id": int(yes_id),
                    "no_token_id": int(no_id),
                    "yes_suffix_token_count": len(yes_suffix),
                    "no_suffix_token_count": len(no_suffix),
                }
            )
            continue

        logits = generation_scores[step][0].detach().float()
        yes_logit = logits[int(yes_id)]
        no_logit = logits[int(no_id)]
        log_denom = torch.logsumexp(logits, dim=-1)
        p_yes = float(torch.exp(yes_logit - log_denom).item())
        p_no = float(torch.exp(no_logit - log_denom).item())
        binary = torch.softmax(torch.stack([yes_logit, no_logit]), dim=0)
        p_yes_norm = float(binary[0].item())
        p_no_norm = float(binary[1].item())
        logit_margin = float((yes_logit - no_logit).item())
        binary_argmax = "Yes" if p_yes_norm >= p_no_norm else "No"
        chosen_preference = p_yes_norm if generated_label == "Yes" else p_no_norm
        exact_full_literal = len(yes_suffix) == 1 and len(no_suffix) == 1

        results[feature] = {
            "p_yes": round(p_yes, 10),
            "p_no": round(p_no, 10),
            "p_yes_norm": round(p_yes_norm, 10),
            "p_no_norm": round(p_no_norm, 10),
            "yes_minus_no_logit_margin": round(logit_margin, 10),
            "label_entropy": round(_binary_entropy(p_yes_norm), 10),
            "entropy_unit": "nats",
            "probability_scope": "generation-time first divergent Yes/No literal token",
            "full_label_probability_exact": bool(exact_full_literal),
            "generated_label": generated_label,
            "binary_argmax_label": binary_argmax,
            "generated_label_matches_binary_argmax": generated_label == binary_argmax,
            "chosen_label_binary_preference": round(chosen_preference, 10),
            "probability_source": source,
            "probability_valid": True,
            "probability_reason": (
                "exact generated-prefix alignment; raw next-token and binary-normalized "
                "preference measured at the first Yes/No divergent generation step"
            ),
            "token_step": int(step),
            "actual_token_id": actual_id,
            "yes_token_id": int(yes_id),
            "no_token_id": int(no_id),
            "yes_suffix_token_count": len(yes_suffix),
            "no_suffix_token_count": len(no_suffix),
        }

    return results


def _model_context_limit(tokenizer: Any, model: Any) -> int:
    """Return a conservative usable model context length."""
    candidates: List[int] = []
    config = getattr(model, "config", None)
    for attr in ["max_position_embeddings", "n_positions", "max_sequence_length"]:
        value = getattr(config, attr, None) if config is not None else None
        if isinstance(value, int) and 0 < value < 10_000_000:
            candidates.append(value)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10_000_000:
        candidates.append(tokenizer_limit)
    if not candidates:
        raise RuntimeError(
            "Could not determine the model context limit; refusing to truncate "
            "the clinical note silently."
        )
    return min(candidates)


def generate(
    prompt: str,
    tokenizer: Any,
    model: Any,
    max_new_tokens: int = MAIN_MAX_NEW_TOKENS,
    probability_features: Optional[Sequence[str]] = None,
    probability_source: str = "generation",
) -> Tuple[str, bool, bool, int, int, int, Dict[str, Dict[str, Any]]]:
    import torch

    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"

    encoded = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=True,
    )
    full_token_count = int(encoded["input_ids"].shape[1])
    context_limit = _model_context_limit(tokenizer, model)
    required = full_token_count + int(max_new_tokens) + MODEL_CONTEXT_SAFETY_TOKENS
    if required > context_limit:
        raise ValueError(
            "FULL_PROMPT_EXCEEDS_MODEL_CONTEXT: "
            f"prompt_tokens={full_token_count}, requested_output={max_new_tokens}, "
            f"margin={MODEL_CONTEXT_SAFETY_TOKENS}, context_limit={context_limit}. "
            "The clinical note was NOT truncated. Reduce external retrieval/"
            "terminology context or use a model with a larger context window."
        )

    inputs = encoded.to(model.device)
    used_token_count = full_token_count
    input_truncated = False

    capture_probs = bool(probability_features)
    with torch.no_grad():
        if capture_probs:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            sequences = outputs.sequences
            score_trace = outputs.scores
        else:
            sequences = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            score_trace = ()

    new_tokens = sequences[0][used_token_count:]
    output_token_count = int(len(new_tokens))
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    text = raw_text.strip()

    label_probability_audit: Dict[str, Dict[str, Any]] = {}
    if capture_probs:
        label_probability_audit = compute_generation_label_probs(
            generated_text=raw_text,
            generated_token_ids=[int(v) for v in new_tokens.tolist()],
            generation_scores=score_trace,
            tokenizer=tokenizer,
            features=list(probability_features or []),
            source=probability_source,
        )
        del score_trace
        del outputs

    hit_cap = output_token_count >= max_new_tokens
    unbalanced = text.count("{") != text.count("}")
    output_truncated = bool(hit_cap and unbalanced)

    return (
        text,
        output_truncated,
        input_truncated,
        full_token_count,
        used_token_count,
        output_token_count,
        label_probability_audit,
    )

def _row_identifier(row: pd.Series, fallback_index: int) -> Any:
    if "NOTE_ID" in row and pd.notna(row.get("NOTE_ID")):
        return row.get("NOTE_ID")
    return int(row.get("_run_row_index", fallback_index))


def run_inference(
    notes_df: pd.DataFrame,
    encoder: MedCPTEncoder,
    chunk_vectors: np.ndarray,
    chunks: Sequence[str],
    direct_vocab: Mapping[str, Sequence[str]],
    retrieval_vocab: Mapping[str, Sequence[str]],
    group_b_lookup: Mapping[str, str],
    tokenizer: Any,
    model: Any,
    grade_index: FormalGradeIndex,
    on_schema_failure: str = "warn",
    resume: bool = False,
) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("VERGE AGENT 1 — MedCPT + RRF + OPTIONAL PROVENANCE-LINKED GRADE")
    print("  (VERGE Agent 1)")
    print(
        f"  Retrieval: top-{DENSE_TOP_K} per query variant; "
        f"RRF k={RRF_K}; classify up to {SOURCE_CANDIDATE_K}; "
        f"retain up to {DENSE_KEEP_K}/symptom"
    )
    print(f"  GRADE mode: {grade_index.mode}")
    print(f"  GRADE policy: {grade_index.policy}")
    print(f"  GRADE minimum certainty (filter only): {grade_index.minimum_certainty}")
    print(f"  Finalized applicable GRADE bodies: {len(grade_index.bodies)}")
    print("  Patient note: before aliases/UMLS/retrieved context")
    print("  Clinical note: FULL NOTE; no character/tokenizer truncation")
    print("  Retrieval label override: NONE")
    print("  Prior-visit memory: NONE")
    print("  Deterministic family-history label gate: NONE (audit only)")
    print("  Decoding: greedy (do_sample=False)")
    print("  Metadata output: note_section, experiencer, assertion_status")
    print("=" * 78)

    rows: List[Dict[str, Any]] = []
    processed_ids: set[str] = set()

    if resume and os.path.exists(OUT_CHECKPOINT):
        existing = pd.read_csv(OUT_CHECKPOINT)
        if "exp_output_dict" in existing.columns:
            existing["exp_output_dict"] = existing["exp_output_dict"].apply(
                parse_saved_dict
            )
        rows = existing.to_dict("records")
        if "_run_row_index" in existing.columns:
            processed_ids = {
                str(int(value))
                for value in existing["_run_row_index"].dropna().tolist()
            }
        elif "NOTE_ID" in existing.columns:
            processed_ids = {
                str(value)
                for value in existing["NOTE_ID"].dropna().tolist()
            }
        print(
            f"[RESUME] Loaded {len(rows)} checkpoint rows from "
            f"{OUT_CHECKPOINT}"
        )

    self_corrections = 0
    label_repairs = 0
    metadata_repairs = 0
    negative_metadata_normalizations = 0
    compact_retries = 0
    full_note_only_repairs = 0
    nested_recovered = 0
    schema_failures = 0
    token_truncations = 0
    note_char_truncations = 0
    notes_with_zero_evidence = 0
    output_token_counts: List[int] = []
    tier_counter: Counter = Counter()
    included_counter: Counter = Counter()
    parse_failure_log: List[Dict[str, Any]] = []
    label_prob_valid_claims = 0
    label_prob_invalid_claims = 0
    label_prob_final_provenance_mismatch_claims = 0
    context_budget_counter: Counter = Counter()

    iterator = tqdm(
        notes_df.iterrows(),
        total=len(notes_df),
        desc="Exp7 MedCPT+RRF RAG",
    )

    for dataframe_index, row in iterator:
        note_identifier = _row_identifier(row, dataframe_index)
        resume_identifier = str(
            int(row.get("_run_row_index", dataframe_index))
        )
        if resume and resume_identifier in processed_ids:
            continue

        original_note_text = str(row.get("Clean_note_text", ""))
        note_text = original_note_text
        note_char_truncated = False

        group_a, group_b = scan_note_for_umls_concepts(
            note_text, direct_vocab, group_b_lookup
        )
        query_variants = {
            symptom: build_query_variants(
                note_text, symptom,
                direct_vocab, retrieval_vocab, group_b,
            )
            for symptom in SYMPTOMS
        }
        per_symptom_candidates, retrieval_audit = retrieve_with_rrf(
            encoder, chunk_vectors, chunks,
            query_variants, retrieval_vocab,
        )
        (
            evidence_section, source_tier_audit,
            note_tier_counts, note_included_counts,
            n_included_this_note,
        ) = prepare_source_filtered_evidence(
            per_symptom_candidates,
            grade_index=grade_index,
        )
        tier_counter.update(note_tier_counts)
        included_counter.update(note_included_counts)
        if n_included_this_note == 0:
            notes_with_zero_evidence += 1

        prompt, context_budget_audit = fit_main_prompt_to_context(
            note_text=note_text,
            direct_vocab=direct_vocab,
            group_b=group_b,
            formal_grade_audit=source_tier_audit,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=MAIN_MAX_NEW_TOKENS,
        )
        context_budget_counter[context_budget_audit["selected_mode"]] += 1

        content = ""
        initial_generation_text = ""
        initial_raw_json = ""
        raw_json = ""
        output_truncated = False
        input_truncated = False
        full_tokens = 0
        used_tokens = 0
        output_tokens = 0
        compact_retry_used = False
        full_note_only_repair_used = False
        label_probs: Dict[str, Dict[str, Any]] = {}
        repair_history: List[Dict[str, Any]] = []
        label_repaired_features: set[str] = set()
        negative_metadata_normalized_features: set[str] = set()
        metadata_consistency_audit: Dict[str, List[str]] = {}

        try:
            (
                content, output_truncated, input_truncated,
                full_tokens, used_tokens, output_tokens, label_probs,
            ) = generate(
                prompt,
                tokenizer,
                model,
                probability_features=SYMPTOMS,
                probability_source="main_generation",
            )
        except Exception as error:
            print(f"  Initial generation error for {note_identifier}: {error}")

        initial_generation_text = content
        parsed, raw_json = safe_json_loads(content)
        initial_raw_json = raw_json
        output_token_counts.append(output_tokens)

        if input_truncated:
            token_truncations += 1

        if output_truncated or not isinstance(parsed, dict):
            compact_retry_used = True
            compact_retries += 1
            compact_prompt = prompt + (
                "\n\nIMPORTANT FORMAT RETRY: Keep every inference under "
                "15 words. Return one compact flat JSON object only."
            )
            try:
                (
                    retry_content, _, retry_input_truncated,
                    retry_full_tokens, retry_used_tokens, retry_output_tokens,
                    retry_label_probs,
                ) = generate(
                    compact_prompt,
                    tokenizer,
                    model,
                    max_new_tokens=MAIN_MAX_NEW_TOKENS,
                    probability_features=SYMPTOMS,
                    probability_source="compact_retry",
                )
                retry_parsed, retry_raw = safe_json_loads(retry_content)
                compact_accepted = isinstance(retry_parsed, dict)
                repair_history.append({
                    "type": "compact_retry",
                    "requested_findings": list(SYMPTOMS),
                    "parsed": compact_accepted,
                    "accepted": compact_accepted,
                    "raw_response": retry_content,
                })
                if compact_accepted:
                    content = retry_content
                    parsed = retry_parsed
                    raw_json = retry_raw
                    input_truncated = retry_input_truncated
                    full_tokens = retry_full_tokens
                    used_tokens = retry_used_tokens
                    output_tokens = retry_output_tokens
                    label_probs = retry_label_probs
            except Exception as error:
                repair_history.append({
                    "type": "compact_retry",
                    "requested_findings": list(SYMPTOMS),
                    "parsed": False,
                    "accepted": False,
                    "error": str(error),
                })
                print(
                    f"  Compact retry error for {note_identifier}: {error}"
                )

        if not isinstance(parsed, dict):
            full_note_only_repair_used = True
            full_note_only_repairs += 1
            try:
                repair_content, _, _, _, _, _, repair_label_probs = generate(
                    build_full_note_only_repair_prompt(note_text),
                    tokenizer,
                    model,
                    max_new_tokens=MAIN_MAX_NEW_TOKENS,
                    probability_features=SYMPTOMS,
                    probability_source="full_note_only_repair",
                )
                repair_parsed, repair_raw = safe_json_loads(repair_content)
                full_repair_accepted = isinstance(repair_parsed, dict)
                repair_history.append({
                    "type": "full_note_only_repair",
                    "requested_findings": list(SYMPTOMS),
                    "parsed": full_repair_accepted,
                    "accepted": full_repair_accepted,
                    "raw_response": repair_content,
                })
                if full_repair_accepted:
                    parsed = repair_parsed
                    raw_json = repair_raw
                    label_probs = repair_label_probs
            except Exception as error:
                repair_history.append({
                    "type": "full_note_only_repair",
                    "requested_findings": list(SYMPTOMS),
                    "parsed": False,
                    "accepted": False,
                    "error": str(error),
                })
                print(
                    f"  Full note-only repair error for "
                    f"{note_identifier}: {error}"
                )

        if not isinstance(parsed, dict):
            parse_failure_log.append(
                {
                    "note_id": note_identifier,
                    "len_chars": len(content),
                    "brace_open": content.count("{"),
                    "brace_close": content.count("}"),
                    "prompt_tokens_full": full_tokens,
                    "prompt_tokens_used": used_tokens,
                    "output_tokens": output_tokens,
                    "input_truncated": input_truncated,
                    "compact_retry_used": compact_retry_used,
                    "full_note_only_repair_used": full_note_only_repair_used,
                    "output_tail": content[-500:],
                }
            )

        if isinstance(parsed, dict):
            had_nested = any(
                isinstance(parsed.get(symptom), dict)
                for symptom in SYMPTOMS
            )
            parsed = flatten_parsed(parsed)
            if had_nested:
                nested_recovered += 1

        if isinstance(parsed, dict):
            missing_labels = missing_symptom_blocks(parsed)
            if missing_labels:
                label_repairs += 1
                for start in range(0, len(missing_labels), 4):
                    batch = missing_labels[start : start + 4]
                    try:
                        (
                            repair_content, _, _, _, _, _, repair_label_probs,
                        ) = generate(
                            build_missing_label_prompt(note_text, batch),
                            tokenizer,
                            model,
                            max_new_tokens=192,
                            probability_features=batch,
                            probability_source="missing_label_repair",
                        )
                        repair_parsed, _ = safe_json_loads(repair_content)
                        parsed_ok = isinstance(repair_parsed, dict)
                        accepted_features: List[str] = []
                        if parsed_ok:
                            for symptom in batch:
                                raw_value = repair_parsed.get(symptom)
                                repaired_label = (
                                    normalize_label(raw_value)
                                    if isinstance(raw_value, str)
                                    else None
                                )
                                if repaired_label is None:
                                    continue
                                parsed[symptom] = repaired_label
                                accepted_features.append(symptom)
                                label_repaired_features.add(symptom)
                                if symptom in repair_label_probs:
                                    label_probs[symptom] = repair_label_probs[symptom]
                                else:
                                    label_probs[symptom] = _invalid_label_probability(
                                        "missing_label_repair",
                                        "accepted explicit label repair but no valid generation-time probability was captured",
                                    )
                        repair_history.append({
                            "type": "missing_label_repair",
                            "requested_findings": list(batch),
                            "parsed": parsed_ok,
                            "accepted": bool(accepted_features),
                            "accepted_findings": accepted_features,
                            "raw_response": repair_content,
                        })
                    except Exception as error:
                        repair_history.append({
                            "type": "missing_label_repair",
                            "requested_findings": list(batch),
                            "parsed": False,
                            "accepted": False,
                            "error": str(error),
                        })
                        print(
                            f"  Missing-label repair error for "
                            f"{note_identifier}: {error}"
                        )

        if isinstance(parsed, dict):
            parsed = complete_auxiliary_fields(parsed)
            repair_targets: List[str] = []
            pre_repair_issues: Dict[str, List[str]] = {}
            for symptom in SYMPTOMS:
                issues = metadata_consistency_issues(parsed, symptom, note_text)

    
                if normalize_label(parsed.get(symptom)) == "No" and issues:
                    normalized_candidate, normalization_changes, normalized_ok = (
                        normalize_negative_metadata_deterministically(
                            parsed=parsed,
                            symptom=symptom,
                            note_text=note_text,
                        )
                    )
                    if normalized_ok:
                        parsed = normalized_candidate
                        negative_metadata_normalizations += 1
                        negative_metadata_normalized_features.add(symptom)
                        repair_history.append({
                            "type": "deterministic_negative_metadata_normalization",
                            "requested_findings": [symptom],
                            "fixed_label": "No",
                            "initial_issues": list(issues),
                            "changes": normalization_changes,
                            "remaining_issues": [],
                            "label_probability_provenance_preserved": True,
                        })
                        issues = []

                pre_repair_issues[symptom] = issues
                repaired_positive = (
                    symptom in label_repaired_features
                    and normalize_label(parsed.get(symptom)) == "Yes"
                )
                if issues or repaired_positive:
                    repair_targets.append(symptom)

            for symptom in repair_targets:
                fixed_label = normalize_label(parsed.get(symptom))
                if fixed_label not in {"Yes", "No"}:
                    continue
                metadata_repairs += 1
                accepted = False
                initial_issues = list(pre_repair_issues.get(symptom, []))
                for attempt in (1, 2):
                    try:
                        repair_content, _, _, _, _, _, _ = generate(
                            build_fixed_label_metadata_prompt(
                                note_text=note_text,
                                symptom=symptom,
                                fixed_label=fixed_label,
                                strict_retry=(attempt == 2),
                            ),
                            tokenizer,
                            model,
                            max_new_tokens=384 if attempt == 1 else 256,
                            probability_features=None,
                            probability_source="fixed_label_metadata_repair",
                        )
                        repair_parsed, _ = safe_json_loads(repair_content)
                        parsed_ok = isinstance(repair_parsed, dict)
                        accepted_fields: List[str] = []
                        remaining_issues: List[str] = ["unparsed_metadata_repair"]
                        if parsed_ok:
                            if symptom in repair_parsed:
                                repeated = normalize_label(repair_parsed.get(symptom))
                                if repeated is not None and repeated != fixed_label:
                                    parsed_ok = False
                            if parsed_ok:
                                candidate = dict(parsed)
                                keys_to_copy = [
                                    f"{symptom} confidence",
                                    f"{symptom} inference",
                                    f"{symptom} note_section",
                                    f"{symptom} experiencer",
                                    f"{symptom} assertion_status",
                                ]
                                if symptom != "Family history of colorectal cancer":
                                    duration = f"Duration of {symptom.lower()}"
                                    keys_to_copy.extend([
                                        duration,
                                        f"{duration} confidence",
                                        f"{duration} inference",
                                    ])
                                for key in keys_to_copy:
                                    if key in repair_parsed:
                                        candidate[key] = repair_parsed[key]
                                        accepted_fields.append(key)
                                candidate = complete_auxiliary_fields(candidate)
                                remaining_issues = metadata_consistency_issues(
                                    candidate, symptom, note_text
                                )
                                if not remaining_issues:
                                    parsed = candidate
                                    accepted = True

                        repair_history.append({
                            "type": "fixed_label_metadata_repair",
                            "requested_findings": [symptom],
                            "fixed_label": fixed_label,
                            "attempt": attempt,
                            "initial_issues": initial_issues,
                            "parsed": parsed_ok,
                            "accepted": accepted,
                            "accepted_fields": accepted_fields if parsed_ok else [],
                            "remaining_issues": remaining_issues,
                            "label_probability_provenance_preserved": True,
                            "raw_response": repair_content,
                        })
                        if accepted:
                            break
                    except Exception as error:
                        repair_history.append({
                            "type": "fixed_label_metadata_repair",
                            "requested_findings": [symptom],
                            "fixed_label": fixed_label,
                            "attempt": attempt,
                            "initial_issues": initial_issues,
                            "parsed": False,
                            "accepted": False,
                            "remaining_issues": ["metadata_repair_operational_error"],
                            "label_probability_provenance_preserved": True,
                            "error": str(error),
                        })
                        print(
                            f"  Fixed-label metadata repair error for "
                            f"{note_identifier}/{symptom}: {error}"
                        )

            parsed = complete_auxiliary_fields(parsed)
            metadata_consistency_audit = {
                symptom: metadata_consistency_issues(parsed, symptom, note_text)
                for symptom in SYMPTOMS
            }

        if isinstance(parsed, dict):
            weak_symptoms: List[str] = []
            weak_bleu_scores: Dict[str, float] = {}
            for symptom, _, inference_key in SYMPTOM_SPECS:
                if normalize_label(parsed.get(symptom)) != "Yes":
                    continue
                inference = parsed.get(inference_key)
                inference_missing = (
                    not isinstance(inference, str)
                    or inference.strip().lower() in BAD_INFERENCE_VALS
                    or len(inference.strip()) <= 5
                )
                if inference_missing:
                    weak_symptoms.append(symptom)
                    weak_bleu_scores[symptom] = 0.0
                    continue
                original_bleu = compute_bleu_no_bp(note_text, inference)
                if original_bleu < BLEU_THRESHOLD:
                    weak_symptoms.append(symptom)
                    weak_bleu_scores[symptom] = original_bleu

            if weak_symptoms:
                self_corrections += 1
                for start in range(0, len(weak_symptoms), 4):
                    batch = weak_symptoms[start : start + 4]
                    try:
                        retry_content, _, _, _, _, _, retry_label_probs = generate(
                            build_grounding_retry_prompt(
                                note_text, parsed, batch
                            ),
                            tokenizer,
                            model,
                            max_new_tokens=RETRY_MAX_NEW_TOKENS,
                            probability_features=batch,
                            probability_source="grounding_retry",
                        )
                        retry_parsed, _ = safe_json_loads(retry_content)
                        if not isinstance(retry_parsed, dict):
                            continue
                        retry_parsed = flatten_parsed(retry_parsed)
                        accepted_features: List[str] = []
                        for symptom in batch:
                            retry_inference = retry_parsed.get(
                                f"{symptom} inference", ""
                            )
                            if (
                                isinstance(retry_inference, str)
                                and len(retry_inference.strip()) > 5
                                and retry_inference.strip().lower()
                                not in BAD_INFERENCE_VALS
                            ):
                                retry_bleu = compute_bleu_no_bp(
                                    note_text, retry_inference
                                )
                                original_bleu = weak_bleu_scores.get(
                                    symptom, 0.0
                                )
                                if (
                                    retry_bleu >= BLEU_THRESHOLD
                                    or retry_bleu > original_bleu
                                ):
                                    retry_label = normalize_label(retry_parsed.get(symptom))
                                    current_label = normalize_label(parsed.get(symptom))
                                    if (
                                        retry_label not in {"Yes", "No"}
                                        or current_label not in {"Yes", "No"}
                                        or retry_label != current_label
                                    ):
                                        continue
                                    candidate = dict(parsed)
                                    for key in [
                                        symptom,
                                        f"{symptom} confidence",
                                        f"{symptom} inference",
                                        f"{symptom} note_section",
                                        f"{symptom} experiencer",
                                        f"{symptom} assertion_status",
                                    ]:
                                        if key in retry_parsed:
                                            candidate[key] = retry_parsed[key]
                                    candidate = complete_auxiliary_fields(candidate)
                                    if metadata_consistency_issues(
                                        candidate, symptom, note_text
                                    ):
                                        continue
                                    parsed = candidate
                                    accepted_features.append(symptom)
                        repair_history.append({
                            "type": "grounding_retry",
                            "requested_findings": list(batch),
                            "parsed": True,
                            "accepted": bool(accepted_features),
                            "accepted_findings": accepted_features,
                            "label_probability_provenance_preserved": True,
                            "raw_response": retry_content,
                        })
                    except Exception as error:
                        repair_history.append({
                            "type": "grounding_retry",
                            "requested_findings": list(batch),
                            "parsed": False,
                            "accepted": False,
                            "error": str(error),
                        })
                        print(
                            f"  Grounding retry error for "
                            f"{note_identifier}: {error}"
                        )

        parsed = complete_auxiliary_fields(parsed)
        if isinstance(parsed, dict):
            metadata_consistency_audit = {
                symptom: metadata_consistency_issues(parsed, symptom, note_text)
                for symptom in SYMPTOMS
            }
        metadata_consistency_valid = bool(
            isinstance(parsed, dict)
            and all(not issues for issues in metadata_consistency_audit.values())
        )


        if isinstance(parsed, dict):
            for symptom in SYMPTOMS:
                final_label = normalize_label(parsed.get(symptom))
                prob = label_probs.get(symptom)
                if not isinstance(prob, dict):
                    prob = _invalid_label_probability(
                        "unavailable",
                        "no generation-time probability record available for final finding",
                    )
                    label_probs[symptom] = prob
                prob["final_extractor_label"] = final_label or ""
                generated_label = str(prob.get("generated_label", "") or "")
                prob["matches_final_extractor_label"] = bool(
                    prob.get("probability_valid", False)
                    and final_label in {"Yes", "No"}
                    and generated_label == final_label
                )
                prob["probability_valid_for_final_label"] = bool(
                    prob.get("probability_valid", False)
                    and prob["matches_final_extractor_label"]
                )
                if prob.get("probability_valid", False):
                    label_prob_valid_claims += 1
                    if not prob["matches_final_extractor_label"]:
                        label_prob_final_provenance_mismatch_claims += 1
                else:
                    label_prob_invalid_claims += 1

        missing_keys = missing_expected_keys(parsed)
        invalid_values = invalid_output_values(parsed)
        structural_schema_valid = bool(
            isinstance(parsed, dict)
            and len(missing_keys) == 0
            and len(invalid_values) == 0
        )
        schema_valid = structural_schema_valid
        requires_downstream_verification = bool(
            structural_schema_valid and not metadata_consistency_valid
        )
        if not schema_valid:
            schema_failures += 1
        fh_audit = {
            "predicted_label": (
                normalize_label(
                    parsed.get("Family history of colorectal cancer")
                )
                if isinstance(parsed, dict)
                else None
            ),
            "explicit_family_member_crc_pattern": explicit_family_history_crc(
                note_text
            ),
            "label_modified_by_rule": False,
        }

        grounding_flags: Dict[str, Any] = {}
        if isinstance(parsed, dict):
            for symptom in SYMPTOMS:
                if normalize_label(parsed.get(symptom)) == "Yes":
                    grounding_flags[symptom] = {
                        "predicted_yes": True,
                        "note_match_outside_ros": symptom_present_outside_ros(
                            note_text, symptom, direct_vocab
                        ),
                    }

        output_row = row.to_dict()
        output_row["exp_output_text_initial"] = initial_generation_text
        output_row["exp_output_raw_initial"] = initial_raw_json
        output_row["exp_output_raw"] = (
            json.dumps(parsed, ensure_ascii=False)
            if isinstance(parsed, dict)
            else raw_json
        )
        output_row["exp_output_dict"] = parsed
        output_row["group_a_found"] = json.dumps(
            group_a, ensure_ascii=False
        )
        output_row["group_b_found"] = json.dumps(
            group_b, ensure_ascii=False
        )
        output_row["query_variants"] = json.dumps(
            query_variants, ensure_ascii=False
        )
        output_row["retrieval_audit"] = json.dumps(
            retrieval_audit, ensure_ascii=False
        )
        output_row["source_tier_audit"] = json.dumps(
            source_tier_audit, ensure_ascii=False
        )
        output_row["formal_grade_audit"] = json.dumps(
            source_tier_audit, ensure_ascii=False
        )
        output_row["formal_grade_metadata"] = json.dumps(
            grade_index.metadata(), ensure_ascii=False
        )
        output_row["context_budget_audit"] = json.dumps(
            context_budget_audit, ensure_ascii=False
        )
        output_row["context_budget_mode"] = context_budget_audit.get(
            "selected_mode", ""
        )
        output_row["n_passages_selected_before_budget"] = n_included_this_note
        passage_limit_used = 0
        for attempt in context_budget_audit.get("attempts", []):
            if attempt.get("mode") == context_budget_audit.get("selected_mode"):
                passage_limit_used = int(attempt.get("passages_per_finding", 0))
                break
        output_row["n_passages_prompted"] = sum(
            min(
                passage_limit_used,
                sum(1 for item in source_tier_audit.get(finding, []) if item.get("included")),
            )
            for finding in SYMPTOMS
        )

        output_row["n_passages_included"] = n_included_this_note
        output_row["fh_audit"] = json.dumps(fh_audit, ensure_ascii=False)
        output_row["grounding_flags"] = json.dumps(
            grounding_flags, ensure_ascii=False
        )
        output_row["label_probs"] = json.dumps(
            label_probs, ensure_ascii=False
        )
        output_row["repair_history"] = json.dumps(
            repair_history, ensure_ascii=False
        )
        output_row["metadata_consistency_audit"] = json.dumps(
            metadata_consistency_audit, ensure_ascii=False
        )
        output_row["metadata_consistency_valid"] = metadata_consistency_valid
        output_row["structural_schema_valid"] = structural_schema_valid
        output_row["requires_downstream_verification"] = requires_downstream_verification
        output_row["label_repaired_features"] = json.dumps(
            sorted(label_repaired_features), ensure_ascii=False
        )
        output_row["negative_metadata_normalized_features"] = json.dumps(
            sorted(negative_metadata_normalized_features), ensure_ascii=False
        )
        output_row["compact_retry_used"] = compact_retry_used
        output_row["full_note_only_repair_used"] = full_note_only_repair_used
        output_row["missing_output_keys"] = json.dumps(
            missing_keys, ensure_ascii=False
        )
        output_row["invalid_output_values"] = json.dumps(
            invalid_values, ensure_ascii=False
        )
        output_row["schema_valid"] = schema_valid
        output_row["note_chars_original"] = len(original_note_text)
        output_row["note_chars_used"] = len(note_text)
        output_row["note_char_truncated"] = note_char_truncated
        output_row["prompt_tokens_full"] = full_tokens
        output_row["prompt_tokens_used"] = used_tokens
        output_row["output_tokens"] = output_tokens
        output_row["input_truncated"] = input_truncated

        rows.append(output_row)

        if len(rows) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_csv(OUT_CHECKPOINT, index=False)
            print(
                f"  [{len(rows)}/{len(notes_df)}] "
                f"compact_retries={compact_retries} "
                f"full_repairs={full_note_only_repairs} "
                f"label_repairs={label_repairs} metadata_repairs={metadata_repairs} "
                f"negative_norm={negative_metadata_normalizations} "
                f"grounding_retries={self_corrections} "
                f"schema_failures={schema_failures} "
                f"evidence_included={sum(included_counter.values())}"
            )

    experiment_df = pd.DataFrame(rows)
    experiment_df.to_csv(OUT_RAW, index=False)

    if parse_failure_log:
        with open(OUT_PARSE_FAIL, "w", encoding="utf-8") as handle:
            json.dump(
                parse_failure_log, handle, indent=2, ensure_ascii=False
            )
    elif os.path.exists(OUT_PARSE_FAIL):
        os.remove(OUT_PARSE_FAIL)

    schema_mask = experiment_df["schema_valid"].fillna(False).astype(bool)
    valid_df = experiment_df.loc[schema_mask].copy()
    valid_df.to_csv(OUT_VALID, index=False)

    if not schema_mask.all():
        failure_columns = [
            column
            for column in [
                "NOTE_ID", "_run_row_index",
                "missing_output_keys", "invalid_output_values",
                "exp_output_raw_initial", "exp_output_raw",
            ]
            if column in experiment_df.columns
        ]
        experiment_df.loc[~schema_mask, failure_columns].to_csv(
            OUT_SCHEMA_FAIL, index=False
        )

    # --- Summary with output token statistics ---
    output_arr = np.array(output_token_counts) if output_token_counts else np.array([0])

    print("\n" + "=" * 78)
    print("RUN SUMMARY")
    print("=" * 78)
    print(f"Rows written:             {len(experiment_df)} -> {OUT_RAW}")
    print(f"Schema-valid rows:        {int(schema_mask.sum())} -> {OUT_VALID}")
    print(
        f"Parse failures remaining: "
        f"{sum(not isinstance(v, dict) for v in experiment_df['exp_output_dict'])}"
    )
    print(f"Schema failures:          {int((~schema_mask).sum())}")
    if "metadata_consistency_valid" in experiment_df.columns:
        print(
            "Metadata-consistency failures: "
            f"{int((~experiment_df['metadata_consistency_valid'].fillna(False).astype(bool)).sum())}"
        )
    print(f"Compact full-RAG retries: {compact_retries}")
    print(f"Full note-only repairs:   {full_note_only_repairs}")
    print(f"Missing-label repair batches: {label_repairs}")
    print(f"Deterministic negative metadata normalizations: {negative_metadata_normalizations}")
    print(f"Fixed-label metadata LLM repairs: {metadata_repairs}")
    print(f"Grounding retries:        {self_corrections}")
    print(f"Nested outputs recovered: {nested_recovered}")
    print(f"Valid generation-time probability audits:   {label_prob_valid_claims}")
    print(f"Invalid generation-time probability audits: {label_prob_invalid_claims}")
    print(
        "Valid audits whose source label differs from final label: "
        f"{label_prob_final_provenance_mismatch_claims}"
    )
    print(f"Input token truncations:  {token_truncations}/{len(experiment_df)}")
    print(
        f"Note-char truncations:    "
        f"{note_char_truncations}/{len(experiment_df)}"
    )
    print(
        f"Output tokens: median={np.median(output_arr):.0f}, "
        f"p95={np.percentile(output_arr, 95):.0f}, "
        f"max={np.max(output_arr)}, "
        f"at-cap={int(np.sum(output_arr >= MAIN_MAX_NEW_TOKENS))}"
    )

    print("\n" + "-" * 78)
    print("FULL-NOTE CONTEXT-BUDGET AUDIT")
    print("-" * 78)
    for mode, _, _, _ in CONTEXT_BUDGET_MODES:
        print(f"{mode:<28}{context_budget_counter.get(mode, 0):>8}")

    print("\n" + "-" * 78)
    print("FORMAL GRADE RETRIEVAL AUDIT")
    print("-" * 78)
    print(f"{'GRADE certainty':<20}{'Candidates':>14}{'Included':>12}")
    for level in GRADE_AUDIT_LEVELS:
        print(
            f"{level:<20}{tier_counter.get(level, 0):>14}"
            f"{included_counter.get(level, 0):>12}"
        )
    total_classified = sum(tier_counter.values())
    total_included = sum(included_counter.values())
    print(f"{'TOTAL':<20}{total_classified:>14}{total_included:>12}")
    print(
        f"Notes with zero included evidence: "
        f"{notes_with_zero_evidence}/{len(experiment_df)}"
    )

    if total_included == 0:
        raise RuntimeError(
            "No passages entered any prompt. Inspect retrieval relevance screening, "
            "formal GRADE mappings, and the configured retrieval policy."
        )

    if not schema_mask.all():
        message = (
            f"{int((~schema_mask).sum())} notes have incomplete schemas. "
            f"See {OUT_SCHEMA_FAIL}."
        )
        if on_schema_failure == "raise":
            raise RuntimeError(message)
        print(f"[SCHEMA WARNING] {message}")

    return experiment_df


def run_metrics(
    experiment_df: pd.DataFrame, skip_bertscore: bool = False
) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("EVIDENCE-GROUNDING METRICS — EXPERIMENT 7")
    print("=" * 78)

    metric_df = experiment_df.copy()

    if "exp_output_dict" not in metric_df.columns:
        source_column = (
            "exp_output_raw"
            if "exp_output_raw" in metric_df.columns
            else None
        )
        if source_column is None:
            raise ValueError(
                "No parsed or raw output column is available."
            )
        metric_df["exp_output_dict"] = metric_df[source_column].apply(
            parse_saved_dict
        )
    else:
        metric_df["exp_output_dict"] = metric_df["exp_output_dict"].apply(
            parse_saved_dict
        )

    metric_df["exp_output_dict"] = metric_df["exp_output_dict"].apply(
        lambda value: complete_auxiliary_fields(flatten_parsed(value))
        if isinstance(value, dict)
        else None
    )

    for symptom, confidence_key, inference_key in SYMPTOM_SPECS:
        metric_df[symptom] = metric_df["exp_output_dict"].apply(
            lambda value, key=symptom: (
                normalize_label(value.get(key))
                if isinstance(value, dict)
                else None
            )
        )
        metric_df[confidence_key] = metric_df["exp_output_dict"].apply(
            lambda value, key=confidence_key: (
                value.get(key, np.nan)
                if isinstance(value, dict)
                else np.nan
            )
        )
        metric_df[inference_key] = metric_df["exp_output_dict"].apply(
            lambda value, key=inference_key: (
                value.get(key, "")
                if isinstance(value, dict)
                else ""
            )
        )
        metric_df[f"{symptom} Conf_num"] = metric_df[
            confidence_key
        ].apply(to_num)

    print("Computing BLEU without brevity penalty...")
    for symptom, _, inference_key in SYMPTOM_SPECS:
        bleu_values = []
        for _, row in metric_df.iterrows():
            inference = row[inference_key]
            if (
                not isinstance(inference, str)
                or inference.strip().lower() in BAD_INFERENCE_VALS
                or len(inference.strip()) < 5
            ):
                bleu_values.append(0.0)
            else:
                bleu_values.append(
                    compute_bleu_no_bp(row["Clean_note_text"], inference)
                )
        metric_df[f"{symptom} BLEU_noBP"] = bleu_values
    print("BLEU complete.")

    for symptom, _, _ in SYMPTOM_SPECS:
        metric_df[f"{symptom} BERT_P"] = np.nan

    if HAVE_BERTSCORE and not skip_bertscore:
        print("Computing BERTScore precision...")
        for symptom, _, inference_key in SYMPTOM_SPECS:
            indices: List[int] = []
            references: List[str] = []
            hypotheses: List[str] = []
            for row_index, row in metric_df.iterrows():
                inference = row[inference_key]
                if (
                    not isinstance(inference, str)
                    or inference.strip().lower() in BAD_INFERENCE_VALS
                    or len(inference.strip()) < 5
                ):
                    continue
                indices.append(row_index)
                references.append(str(row["Clean_note_text"]))
                hypotheses.append(inference)
            scores = compute_bertscore_batch(references, hypotheses)
            for row_index, score in zip(indices, scores):
                metric_df.at[row_index, f"{symptom} BERT_P"] = score
        print("BERTScore complete.")
    elif skip_bertscore:
        print("BERTScore skipped by command-line option.")
    else:
        print("bert_score package not installed; BERTScore skipped.")

    metric_df.to_csv(OUT_METRICS, index=False)

    summary_rows: List[Dict[str, Any]] = []
    for symptom, _, _ in SYMPTOM_SPECS:
        confidence_column = f"{symptom} Conf_num"
        bleu_column = f"{symptom} BLEU_noBP"
        bert_column = f"{symptom} BERT_P"
        yes_mask = metric_df[symptom] == "Yes"
        no_mask = metric_df[symptom] == "No"
        labelled_mask = yes_mask | no_mask
        yes_bleu = metric_df.loc[yes_mask, bleu_column].mean()
        no_bleu = metric_df.loc[no_mask, bleu_column].mean()
        summary_rows.append(
            {
                "Symptom": symptom,
                "Labelled_Count": int(labelled_mask.sum()),
                "Yes_Count": int(yes_mask.sum()),
                "No_Count": int(no_mask.sum()),
                "Yes_Conf_Mean": metric_df.loc[
                    yes_mask, confidence_column
                ].mean(),
                "Yes_Conf_SD": metric_df.loc[
                    yes_mask, confidence_column
                ].std(),
                "No_Conf_Mean": metric_df.loc[
                    no_mask, confidence_column
                ].mean(),
                "No_Conf_SD": metric_df.loc[
                    no_mask, confidence_column
                ].std(),
                "Yes_BLEU_Mean": yes_bleu,
                "Yes_BLEU_SD": metric_df.loc[
                    yes_mask, bleu_column
                ].std(),
                "No_BLEU_Mean": no_bleu,
                "No_BLEU_SD": metric_df.loc[no_mask, bleu_column].std(),
                "BLEU_Gap_Yes_Minus_No": yes_bleu - no_bleu,
                "Yes_BERTP_Mean": metric_df.loc[
                    yes_mask, bert_column
                ].mean(),
                "Yes_BERTP_SD": metric_df.loc[
                    yes_mask, bert_column
                ].std(),
                "No_BERTP_Mean": metric_df.loc[
                    no_mask, bert_column
                ].mean(),
                "No_BERTP_SD": metric_df.loc[
                    no_mask, bert_column
                ].std(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    print("\n" + "-" * 78)
    print("DETECTION COUNTS AND GROUNDING GAPS")
    print("-" * 78)
    print(
        f"{'Symptom':<44}{'Labelled':>10}{'Yes':>7}{'No':>7}{'Gap':>10}"
    )
    for _, summary in summary_df.iterrows():
        gap = summary["BLEU_Gap_Yes_Minus_No"]
        flag = (
            "  <-- LOW GAP WARNING"
            if pd.notna(gap) and gap < 0.10
            else ""
        )
        print(
            f"{summary['Symptom']:<44}"
            f"{int(summary['Labelled_Count']):>10}"
            f"{int(summary['Yes_Count']):>7}"
            f"{int(summary['No_Count']):>7}"
            f"{gap:>10.3f}{flag}"
        )

    print(f"\nMetrics written to: {OUT_METRICS}")
    print(f"Summary written to: {OUT_SUMMARY}")
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "VERGE Agent 1: full-note joint seven-finding extraction with MedCPT, "
            "within-finding RRF, optional provenance-linked formal GRADE metadata, "
            "and generation-time Yes/No uncertainty auditing"
        )
    )
    parser.add_argument(
        "--phase", choices=["inference", "metrics", "all"], default="all"
    )
    parser.add_argument("--model", default=HF_MODEL_ID)
    parser.add_argument("--max_notes", type=int, default=None)
    parser.add_argument(
        "--on_schema_failure", choices=["warn", "raise"], default="warn"
    )
    parser.add_argument(
        "--skip_bertscore", action="store_true",
        help="Skip BERTScore to reduce runtime/memory."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=f"Resume from {OUT_CHECKPOINT} when it exists."
    )
    parser.add_argument("--article_batch_size", type=int, default=128)

    parser.add_argument(
        "--grade_mode",
        choices=["off", "optional", "required"],
        default=GRADE_MODE.lower(),
        help=(
            "off: ignore GRADE; optional: consume finalized GRADE if present, otherwise "
            "all passages remain UNMAPPED; required: fail unless >=1 finalized applicable "
            "GRADE body with study/chunk provenance is available."
        ),
    )
    parser.add_argument("--grade_profile", default=GRADE_PROFILE_PATH)
    parser.add_argument("--grade_study_map", default=GRADE_STUDY_MAP_PATH)
    parser.add_argument("--grade_chunk_map", default=GRADE_CHUNK_MAP_PATH)
    parser.add_argument(
        "--grade_policy",
        choices=["annotate", "prioritize", "filter"],
        default=GRADE_POLICY,
        help=(
            "annotate: record parent-body GRADE but preserve RRF/cosine order; "
            "prioritize: mapped > unmapped, then certainty > RRF > cosine; "
            "filter: exclude unmapped/below-threshold passages. This is a VERGE RAG "
            "policy, not a GRADE rule."
        ),
    )
    parser.add_argument(
        "--grade_min_certainty",
        choices=["HIGH", "MODERATE", "LOW", "VERY_LOW"],
        default=GRADE_MIN_CERTAINTY.upper(),
        help="Minimum final certainty used only with --grade_policy=filter.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("VERGE AGENT 1 — FULL-NOTE GRADE-AWARE EXTRACTOR")
    print("=" * 78)
    print("  Model:            LLaMA-3.1-8B-Instruct")
    print("  Decoding:         greedy (do_sample=False)")
    print("  Components:       aliases + UMLS + ROS + MedCPT + within-finding RRF")
    print(
        f"  Retrieval:        top-{DENSE_TOP_K} per query variant; "
        f"RRF k={RRF_K}; up to {SOURCE_CANDIDATE_K} candidates/finding"
    )
    print(f"  Prompt evidence:  up to {DENSE_KEEP_K} passages/finding")
    print(f"  GRADE mode:       {args.grade_mode}")
    print(f"  GRADE policy:     {args.grade_policy}")
    print(f"  GRADE minimum:    {args.grade_min_certainty} (filter only)")
    print(f"  GRADE profile:    {args.grade_profile}")
    print(f"  GRADE study map:  {args.grade_study_map}")
    print(f"  GRADE chunk map:  {args.grade_chunk_map}")
    print("  GRADE semantics:  parent body only; never passage-generated; never a patient-label rule")
    print("  Label probs:      first-divergent Yes/No generation logits; audit only")
    print("  Patient labels:   FULL clinical-note evidence only")
    print("  Note truncation:  NONE")
    print("  Input truncation: DISABLED; explicit failure on context overflow")
    print("  FH handling:      biological relative + CRC specificity in note; no programmatic label gate")
    print("  Metadata output:  note_section, experiencer, assertion_status")
    print(f"  Raw output:       {OUT_RAW}")
    print(f"  Metrics output:   {OUT_METRICS}")
    print("=" * 78)

    grade_index = FormalGradeIndex.load(
        profile_path=args.grade_profile,
        study_map_path=args.grade_study_map,
        chunk_map_path=args.grade_chunk_map,
        policy=args.grade_policy,
        minimum_certainty=args.grade_min_certainty,
        mode=args.grade_mode,
        valid_symptoms=SYMPTOMS,
    )
    print(f"GRADE metadata: {grade_index.metadata()}")
    if args.grade_mode == "required" and not grade_index.active:
        raise RuntimeError("GRADE mode is required but no finalized applicable body was loaded.")

    umls_synonyms = load_umls_synonyms()
    direct_vocab, retrieval_vocab = build_vocabularies(umls_synonyms)
    _, group_b_lookup = load_concept_cache()
    chunks = load_chunks()
    notes_df = load_notes()

    if args.max_notes is not None:
        if args.max_notes <= 0:
            raise ValueError("--max_notes must be a positive integer.")
        notes_df = notes_df.head(args.max_notes).copy()
        print(f"[TEST MODE] Running first {len(notes_df)} notes")

    experiment_df: Optional[pd.DataFrame] = None

    if args.phase in {"inference", "all"}:
        encoder = MedCPTEncoder(HF_CACHE_DIR)
        chunk_vectors = build_dense_index(
            encoder, chunks, batch_size=args.article_batch_size
        )
        tokenizer, model = load_llama(args.model)
        experiment_df = run_inference(
            notes_df=notes_df,
            encoder=encoder,
            chunk_vectors=chunk_vectors,
            chunks=chunks,
            direct_vocab=direct_vocab,
            retrieval_vocab=retrieval_vocab,
            group_b_lookup=group_b_lookup,
            tokenizer=tokenizer,
            model=model,
            grade_index=grade_index,
            on_schema_failure=args.on_schema_failure,
            resume=args.resume,
        )

    if args.phase == "metrics":
        require_file(OUT_RAW, "Agent 1 raw output")
        experiment_df = pd.read_csv(OUT_RAW)

    if args.phase in {"metrics", "all"}:
        if experiment_df is None:
            raise RuntimeError("No Agent 1 output is available for metrics.")
        run_metrics(experiment_df, skip_bertscore=args.skip_bertscore)


if __name__ == "__main__":
    main()
