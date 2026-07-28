"""D-02 新规格 LG/LC/MD/GG 基线合同和反向破坏 T0。"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_language_baseline_catalog import (
    build_capability_ledger,
    build_course_coverage_ledger,
    build_gg00_audit,
    build_language_baseline_manifest,
    build_md00_preregistration,
    build_verifier_registry,
)
from pure_integer_ai.experiments.ph2_language_baseline_manifest import (
    GG_COMBINATION_AXIS_KEYS,
    GG_COURSE_FAMILY_KEYS,
    GG_STAGE_KEYS,
    LanguageBaselineManifestError,
    PublicGateBaseline,
    inventory_public_files,
    read_language_baseline_manifest,
    scan_public_patterns,
    verify_language_baseline_files,
    write_language_baseline_manifest,
)
from pure_integer_ai.experiments.ph2_language_coverage_contract import (
    CAPABILITY_KEYS,
    FACT_DIMENSIONS,
    SAMPLE_FAMILIES,
    CapabilityCourseCoverageLedger,
    LanguageCapabilityCoverageLedger,
    LanguageCoverageContractError,
)


MANIFEST_PATH = Path(
    "data/ph2/manifests/language_capability_baseline_v39.json")
REPOSITORY = Path(__file__).resolve().parents[1]
PUBLICATION_RECEIPT_PATH = (
    REPOSITORY
    / "data/ph2/manifests/j_lg_d03_gate_v4_git_publication_v1.json"
)


def _identity_matches(payload: bytes, item) -> bool:
    """按冻结尺寸和摘要比较一份基线候选字节。"""
    return (
        len(payload) == item.size_bytes
        and hashlib.sha256(payload).hexdigest() == item.sha256
    )


def _verify_historical_repository_baseline(manifest) -> None:
    """以当前发布 override 或历史基线 blob 复验 v39 文件。"""
    receipt = json.loads(PUBLICATION_RECEIPT_PATH.read_bytes())
    overrides = {
        str(item["relative_path"]): item
        for item in receipt["current_path_overrides"]
    }
    for item in manifest.file_inventory:
        current_path = REPOSITORY / item.relative_path
        if current_path.is_file():
            current = current_path.read_bytes()
            if _identity_matches(current, item):
                continue
            override = overrides.get(item.relative_path)
            if override is not None:
                assert len(current) == override["size_bytes"]
                assert hashlib.sha256(current).hexdigest() == override["sha256"]
                continue
        historical = subprocess.run(
            ("git", "show", f"{manifest.head_sha1}:{item.relative_path}"),
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
        ).stdout
        assert _identity_matches(historical, item)
    for item in manifest.paper_files:
        assert _identity_matches(
            (REPOSITORY / item.relative_path).read_bytes(), item)


def _synthetic_manifest(tmp_path: Path):
    repo = tmp_path / "repo"
    source = repo / "src" / "one.py"
    test = repo / "tests" / "test_one.py"
    paper = repo / "paper" / "main.tex"
    source.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    paper.parent.mkdir(parents=True)
    source.write_text("value = 1\n# LEGACY_MARK\n", encoding="utf-8")
    test.write_text("def test_one():\n    assert 1 == 1\n", encoding="utf-8")
    paper.write_text("paper bytes\n", encoding="utf-8")
    inventory = inventory_public_files(
        repo, ("src/one.py", "tests/test_one.py"))
    legacy, binary, unreadable = scan_public_patterns(
        repo, inventory, (("LEGACY_NAME_V1", re.compile("LEGACY_MARK")),))
    secret, binary_2, unreadable_2 = scan_public_patterns(
        repo, inventory, (("API_KEY_V1", re.compile("SECRET_[A-Z]+")),))
    assert binary == binary_2
    assert unreadable == unreadable_2
    gate = PublicGateBaseline(
        2, 2, binary, unreadable,
        ("LEGACY_NAME_V1",), legacy, "BLOCKED",
        ("API_KEY_V1",), secret, "CLEAR", 1, 0,
    )
    manifest = build_language_baseline_manifest(
        artifact_version="LG-LC-MD-GG-baseline-test-v1",
        head_sha1="1" * 40,
        origin_master_sha1="1" * 40,
        untracked_file_count=3,
        inventory_exclusions=(
            "data/ph2/manifests/language_capability_baseline_v1.json",),
        file_inventory=inventory,
        paper_files=inventory_public_files(repo, ("paper/main.tex",)),
        public_gate=gate,
    )
    return repo, manifest


def test_lc00_lists_every_family_without_runtime_or_retention_claims():
    """20 个能力族必须显式列出，脚手架不能冒充 runtime learned。"""
    ledger = build_capability_ledger()
    assert tuple(item.capability_key for item in ledger.entries) == CAPABILITY_KEYS
    assert {item.implementation_state for item in ledger.entries} == {
        "ABSENT", "DESIGN_ONLY", "SCAFFOLD_ONLY",
    }
    forbidden = {
        "RETENTION_EVIDENCED", "RUNTIME_CONNECTED", "RUNTIME_EVIDENCED",
    }
    for item in ledger.entries:
        states = item.fact_states.to_value()
        assert set(states) == set(FACT_DIMENSIONS)
        assert not (set(states.values()) & forbidden)
        assert set(item.scope_axes.to_value()) == {
            "code_switch", "dialect", "domain", "era", "genre", "language",
            "length", "medium", "noise", "register", "script_orthography",
        }
        if item.capability_key != "NON_TEXT_MEDIA":
            assert states["DIRECTIONAL_CONSUMPTION"] == "COURSE_FROZEN"
        assert "LC-13" in item.task_keys
        assert "LC13_DIRECTIONAL_CONSUMER_MAP_V1" in (
            item.representation_contracts)
        assert "LC13_DIRECTIONAL_CONSUMER_VERIFIER_V1" in item.verifier_keys
    transfer = next(
        item for item in ledger.entries
        if item.capability_key == "TRANSFER_AXES")
    assert transfer.implementation_state == "SCAFFOLD_ONLY"
    assert {
        "AXIS_SCOPED_APPLICABILITY_REQUIRED",
        "LC09_TRANSFER_AXIS_MANIFEST_V1",
        "MANIFEST_COMBINATION_SPLIT_REQUIRED",
        "SCOPE_CONTRACTION_PROTOCOL_V1",
        "SINGLE_DOUBLE_FULL_COMBINATION_SPLIT_V1",
    } <= set(transfer.representation_contracts)
    transfer_states = transfer.fact_states.to_value()
    assert {transfer_states[key] for key in (
        "DATA_ISOLATION", "REPRESENTATION", "RESOURCE", "SCOPE",
        "VERIFIER_CAPABILITY")
    } == {"COURSE_FROZEN"}
    assert transfer_states["RETENTION"] == "ABSENT"
    assert set(transfer.resource_contracts) == {
        "MAX_BLOCKED_SOURCES=1", "MAX_FORMAL_PACKS=16",
        "MAX_LOGIC_STEPS=ABSENT", "MAX_OUTPUT_UNITS=ABSENT",
        "MAX_PAGE_SEGMENTS=ABSENT", "MAX_QUERIES=ABSENT",
        "MAX_RECOMPUTE_OBJECTS=ABSENT", "MAX_RECURSION_DEPTH=ABSENT",
        "MAX_SPLIT_PROBES=3", "MAX_TRANSFER_AXES=10",
    }
    retention = next(
        item for item in ledger.entries
        if item.capability_key == "EVALUATOR_RETENTION_RESOURCE")
    retention_states = retention.fact_states.to_value()
    assert {retention_states[key] for key in (
        "DATA_ISOLATION", "REPRESENTATION", "RESOURCE", "RETENTION",
        "SCOPE", "VERIFIER_CAPABILITY",
    )} == {"COURSE_FROZEN"}
    assert "RETENTION_EVIDENCED" not in retention_states.values()
    assert "LC10_RETENTION_ROLLBACK_MANIFEST_V1" in (
        retention.representation_contracts)
    assert set(retention.resource_contracts) == {
        "MAX_EVIDENCE_FILES=8", "MAX_FIXTURES=3",
        "MAX_LOGIC_STEPS=ABSENT", "MAX_OUTCOME_CLASSES=5",
        "MAX_OUTPUT_UNITS=ABSENT", "MAX_PAGE_SEGMENTS=ABSENT",
        "MAX_PHASES=10", "MAX_QUERIES=ABSENT",
        "MAX_RECOMPUTE_OBJECTS=ABSENT", "MAX_RECURSION_DEPTH=ABSENT",
        "MAX_RUNTIME_BINDINGS=7",
    }
    nonliteral = next(
        item for item in ledger.entries
        if item.capability_key == "NONLITERAL_CULTURAL")
    assert nonliteral.implementation_state == "DESIGN_ONLY"
    assert nonliteral.fact_states.to_value()["SCOPE"] == "COURSE_FROZEN"
    assert nonliteral.fact_states.to_value()["REPRESENTATION"] == "DESIGNED"
    assert nonliteral.fact_states.to_value()["VERIFIER_CAPABILITY"] == "ABSENT"
    assert "NL00_NONLITERAL_SCOPE_DECISION_V1" in (
        nonliteral.representation_contracts)
    assert "NL00_NONLITERAL_SCOPE_PROBE_VERIFIER_V1" in (
        nonliteral.verifier_keys)


def test_lc11_registers_blind_spots_ne_and_zero_runtime_pass_authority():
    """静态合同 verifier 只能判声明维度，缺失语言 verifier 必须为 ABSENT。"""
    registry = build_verifier_registry()
    assert len(registry.records) == 31
    assert all(item.blind_spots for item in registry.records)
    assert all(item.ne_conditions for item in registry.records)
    assert all(item.owner_key != "TEACHER" for item in registry.records)
    assert all(item.can_issue_runtime_pass == 0 for item in registry.records)
    by_key = {item.verifier_key: item for item in registry.records}
    assert by_key["LANGUAGE_RUNTIME_HELD_OUT_VERIFIER_V1"].registry_state == (
        "ABSENT")
    lc01 = by_key["LC01_TEXT_FIDELITY_VERIFIER_V1"]
    assert lc01.registry_state == "RUNTIME_EVIDENCED"
    assert lc01.can_issue_runtime_pass == 0
    md01 = by_key["MD01_MEMORY_DYNAMICS_CONTRACT_VERIFIER_V1"]
    assert md01.registry_state == "RUNTIME_EVIDENCED"
    assert md01.can_issue_runtime_pass == 0
    assert "MD04_PROBE_NOT_EXECUTED" not in md01.ne_conditions
    md02 = by_key["MD02_SITUATION_STATE_ADAPTER_VERIFIER_V1"]
    assert md02.registry_state == "RUNTIME_EVIDENCED"
    assert md02.can_issue_runtime_pass == 0
    assert set(md02.decidable_dimensions) == {
        "BACKING_EVENT_IDENTITY", "DEPENDENCY_INDEX_REBUILD",
        "LOCAL_INVALIDATION_SCOPE", "ORIGINAL_EVENT_PRESERVATION",
        "OWNER_SCOPE_VERSION_ISOLATION",
        "UNAFFECTED_PROJECTION_BIT_IDENTITY", "ZERO_HOST_LEARNING_WRITE",
    }
    assert "MD03_DIRECTIONAL_ADAPTER_NOT_EXECUTED" not in md02.ne_conditions
    md03 = by_key["MD03_DIRECTIONAL_CENTER_ADAPTER_VERIFIER_V1"]
    assert md03.registry_state == "RUNTIME_EVIDENCED"
    assert md03.can_issue_runtime_pass == 0
    assert set(md03.decidable_dimensions) == {
        "ACTIVATION_ADOPTION_SEPARATION", "DIRECTIONAL_PAYLOAD_DISTINCTION",
        "EXACT_DEDUP_AND_PROVENANCE_MERGE", "OWNER_SCOPE_VERSION_ISOLATION",
        "STRENGTH_PRESERVATION", "WRITE_PERMISSION_ORTHOGONALITY",
        "ZERO_HOST_LEARNING_WRITE",
    }
    assert "MD04_PROBE_NOT_EXECUTED" not in md03.ne_conditions
    md05 = by_key["MD05_CENTER_DIFFUSION_PROBE_VERIFIER_V1"]
    assert md05.registry_state == "RUNTIME_EVIDENCED"
    assert md05.can_issue_runtime_pass == 0
    assert set(md05.decidable_dimensions) == {
        "ABLATION_CAUSAL_DEGRADATION",
        "ACCESS_BUDGET_GROUNDING_CLASSIFICATION",
        "EXACT_STRUCTURE_DISTANCE_HELD_OUT",
        "FOUR_BASELINE_SAME_FIXTURE", "K04_TYPED_RANGE_RECEIPT",
        "MULTICENTER_PHYSICAL_READ_SHARING",
        "SOURCE_EVIDENCE_CHAIN_RECOVERY",
        "UNRELATED_1X_10X_100X_RESOURCE_SCALING",
        "ZERO_HOST_LEARNING_WRITE",
    }
    gg01 = by_key["GG01_GENERATION_CHOICE_CONTRACT_VERIFIER_V1"]
    assert gg01.registry_state == "COURSE_FROZEN"
    assert gg01.can_issue_runtime_pass == 0
    assert "GG02_USE_OUTCOME_BRIDGE_NOT_CONNECTED" in gg01.blind_spots
    gg01_v2 = by_key["GG01_GENERATION_CHOICE_CONTRACT_VERIFIER_V2"]
    assert gg01_v2.registry_state == "COURSE_FROZEN"
    assert gg01_v2.can_issue_runtime_pass == 0
    assert "EXACT_KEYS_PRESERVE_ZERO_BEARING_CORE_MEMORY_IDENTITIES" in (
        gg01_v2.decidable_dimensions)
    gg02 = by_key["GG02_GENERATION_CHOICE_OUTCOME_BRIDGE_VERIFIER_V1"]
    assert gg02.registry_state == "COURSE_FROZEN"
    assert gg02.can_issue_runtime_pass == 0
    assert set(gg02.decidable_dimensions) == {
        "ASSESSMENT_LAYER_ISOLATION", "CLAIM_LAYER_AUTHORIZATION",
        "EXACT_USE_OUTCOME_LINK", "FIVE_LAYER_COMPLETENESS",
        "NO_SENTENCE_WIDE_BROADCAST", "OWNER_SCOPE_QUERY_BINDING",
        "READ_ONLY_VERIFIER_INPUT", "ZERO_HOST_LEARNING_WRITE",
    }
    assert "ASSESSMENT_CONSUMER_NOT_CONNECTED" in gg02.blind_spots
    gg03 = by_key["GG03_GENERATION_GENERALIZATION_COURSE_VERIFIER_V1"]
    assert gg03.registry_state == "COURSE_FROZEN"
    assert gg03.can_issue_runtime_pass == 0
    assert set(gg03.decidable_dimensions) == {
        "ADDRESSEE_RECOVERABILITY", "COMBINATION_HELD_OUT",
        "COMMUNICATIVE_TASK", "EXACT_MEMORY_BASELINE_REJECT",
        "FAILURE_LAYER_LOCALIZATION", "LEGAL_OBJECT_COMPOSITION",
        "MULTIPLE_LEGAL_SURFACE_SET", "RETENTION_REVERIFY",
        "REVISION_SUPERSEDE", "SEMANTIC_ROLE_SCOPE_POLARITY",
        "SOURCE_UNCERTAINTY_CITATION", "STANCE_CONTENT_WORDING_SEPARATION",
        "STRUCTURE_SLOT_ORDER", "USE_OUTCOME_TEMPLATE_PROMOTION_REJECT",
    }
    assert "RUNTIME_GENERALIZATION_NOT_EVIDENCED" in gg03.blind_spots
    assert set(lc01.decidable_dimensions) == {
        "CANDIDATE_LATTICE", "GENERATION_SURFACE_FIDELITY",
        "IRREVERSIBLE_LOSS_DISCLOSURE", "LEARNING_OBJECTIVE_BINDING",
        "NORMALIZATION_RECEIPT", "RAW_OBSERVATION_PRESERVATION",
        "RETENTION_REVERIFY",
    }
    lc02 = by_key["LC02_MORPHOLOGY_COURSE_VERIFIER_V1"]
    assert lc02.registry_state == "RUNTIME_EVIDENCED"
    assert lc02.can_issue_runtime_pass == 0
    assert set(lc02.decidable_dimensions) == {
        "AMBIGUOUS_SEGMENTATION", "DICTIONARY_REPLAY_REJECT",
        "EXCEPTION_SCOPE", "HELD_OUT_STEM_CONSTRUCTION", "LANGUAGE_SCOPE",
        "MORPHOLOGY_RELATION_INTEGRITY", "RETENTION_REVERIFY",
        "REVERSE_GENERATION",
    }
    lc03 = by_key["LC03_CONSTRUCTION_COURSE_VERIFIER_V1"]
    assert lc03.registry_state == "RUNTIME_EVIDENCED"
    assert lc03.can_issue_runtime_pass == 0
    assert set(lc03.decidable_dimensions) == {
        "ANTI_LITERAL_BASELINE", "CONSTRUCTION_OBJECT_ABLATION",
        "DISCONTINUOUS_SPAN", "EVENT_CORE_MAPPING", "FIXED_VARIABLE_SLOT",
        "HELD_OUT_FILLER_CONSTRUCTION", "LEXICALIZATION_IDENTITY",
        "REGISTER_SCOPE", "RETENTION_REVERIFY", "REVERSE_GENERATION",
        "SAME_PROPOSITION_DIFFERENT_CONSTRUCTION",
        "SAME_SURFACE_DIFFERENT_CONSTRUCTION",
    }
    lc04 = by_key["LC04_RECURSIVE_PARSE_COURSE_VERIFIER_V1"]
    assert lc04.registry_state == "RUNTIME_EVIDENCED"
    assert lc04.can_issue_runtime_pass == 0
    assert set(lc04.decidable_dimensions) == {
        "AMBIGUOUS_PARSE_COMPETITION", "COORDINATION_STRUCTURE",
        "DISCONTINUOUS_DEPENDENCY", "HELD_OUT_FILLER_PARSE_DEPTH",
        "LOCAL_REPARSE_SUPERSEDE", "NESTED_DEPTH",
        "NULL_OPTIONAL_CONSTITUENT", "PRESELECTED_TREE_REJECT",
        "REPEATED_TOKEN_IDENTITY", "RETENTION_REVERIFY",
        "REVERSE_LINEARIZATION", "ROLE_SCOPE_PRESERVATION",
    }
    lc05 = by_key["LC05_EVENT_TIME_ASPECT_COURSE_VERIFIER_V1"]
    assert lc05.registry_state == "RUNTIME_EVIDENCED"
    assert lc05.can_issue_runtime_pass == 0
    assert set(lc05.decidable_dimensions) == {
        "COMPLETED_ASPECT", "DURATIVE_INTERVAL", "EVENT_IDENTITY",
        "HABITUAL_ITERATIVE_ASPECT", "IMPLICIT_NOW_REJECT",
        "LOCAL_TIME_REVISION", "NARRATIVE_ORDER", "RETENTION_REVERIFY",
        "REVERSE_GENERATION_ANCHOR_ASPECT",
        "SAME_SURFACE_DIFFERENT_ANCHOR", "STATE_IDENTITY",
        "SURFACE_ORDER_REJECT", "TIME_ANCHOR_SCOPE", "TIME_UNKNOWN",
    }
    lc06 = by_key["LC06_COMPARISON_QUANTITY_COURSE_VERIFIER_V1"]
    assert lc06.registry_state == "RUNTIME_EVIDENCED"
    assert lc06.can_issue_runtime_pass == 0
    assert set(lc06.decidable_dimensions) == {
        "AMBIGUOUS_STANDARD_COMPETITION", "APPROXIMATE_EXACT_DISTINCTION",
        "BARE_PROPERTY_REJECT", "COMPARISON_DIRECTION_STANDARD",
        "DEGREE_SCALE_THRESHOLD", "LOCAL_QUANTITY_REVISION",
        "MEASURE_UNIT_DIMENSION", "QUANTIFIER_SCOPE", "QUANTITY_COUNT",
        "QUANTITY_UNKNOWN", "RANGE_BOUNDARY", "RETENTION_REVERIFY",
        "REVERSE_GENERATION_COMPARISON_MEASURE", "UNIT_ERASURE_REJECT",
    }
    lc07 = by_key["LC07_DISCOURSE_INFORMATION_COURSE_VERIFIER_V1"]
    assert lc07.registry_state == "RUNTIME_EVIDENCED"
    assert lc07.can_issue_runtime_pass == 0
    assert set(lc07.capability_keys) == {
        "DISCOURSE_INFORMATION_STRUCTURE", "REFERENCE_DISCOURSE_REVISION",
        "TYPED_LEARNING_OBJECTIVES",
    }
    assert set(lc07.decidable_dimensions) == {
        "AMBIGUOUS_RELATION_COMPETITION", "CAUSE_RELATION",
        "CONCESSION_RELATION", "CONTRAST_RELATION", "DISCOURSE_UNKNOWN",
        "ELABORATION_RELATION", "GIVEN_NEW_STATUS",
        "LOCAL_RELATION_REVISION", "NO_CONNECTIVE_REJECT",
        "PRESUPPOSITION_CANCELLATION", "PRESUPPOSITION_PROJECTION",
        "QUD_STATE", "RETENTION_REVERIFY",
        "REVERSE_GENERATION_ORDER_EXPLICITNESS",
        "TOPIC_FOCUS_MINIMAL_CONTRAST", "WRONG_CONNECTIVE_REJECT",
    }
    assert "MD02_MD03_RUNTIME_PREREQUISITE_MISSING" not in lc07.ne_conditions
    lc08 = by_key["LC08_OPEN_SET_CLARIFICATION_COURSE_VERIFIER_V1"]
    assert lc08.registry_state == "RUNTIME_EVIDENCED"
    assert lc08.can_issue_runtime_pass == 0
    assert set(lc08.capability_keys) == {
        "OPEN_SET_CONTINUAL_LEARNING", "PRAGMATIC_CLARIFICATION_REPAIR",
        "TYPED_LEARNING_OBJECTIVES",
    }
    assert set(lc08.decidable_dimensions) == {
        "ACCESS_BLOCKED_DISTINCT", "ACTIVE_EVIDENCE_REQUEST",
        "AMBIGUOUS_BRANCH_PRESERVATION", "BUDGET_BLOCKED_DISTINCT",
        "CLARIFICATION_ANSWER_LOCAL_UPDATE",
        "CLARIFICATION_REVISION_SUPERSEDE", "INSUFFICIENT_GUESS_REJECT",
        "KNOWN_SUFFICIENT_NO_QUESTION", "MINIMAL_BRANCH_CLARIFICATION",
        "NEW_CONSTRUCTION_DETECTION", "NEW_SENSE_DETECTION",
        "NEW_USAGE_DETECTION", "NEW_WORD_DETECTION", "OPEN_SET_UNKNOWN",
        "OVERQUESTION_REJECT", "RETENTION_REVERIFY",
        "REVERSE_GENERATION_MINIMAL_QUESTION",
    }
    assert "MD03_RUNTIME_PREREQUISITE_MISSING" not in lc08.ne_conditions
    assert "NOVELTY_RUNTIME_NOT_EXECUTED" in lc08.ne_conditions
    lc14 = by_key["LC14_ATTRIBUTION_QUOTATION_COURSE_VERIFIER_V1"]
    assert lc14.registry_state == "RUNTIME_EVIDENCED"
    assert lc14.can_issue_runtime_pass == 0
    assert set(lc14.capability_keys) == {
        "ATTRIBUTION_QUOTATION_PERSPECTIVE", "SOURCE_UNCERTAINTY_REALITY",
        "TYPED_LEARNING_OBJECTIVES",
    }
    assert set(lc14.decidable_dimensions) == {
        "AMBIGUOUS_SCOPE_COMPETITION", "ATTRIBUTION_LOCAL_REVISION",
        "BELIEF_HOLDER_SCOPE", "CLAIM_HOLDER_SCOPE",
        "DIRECT_QUOTE_BOUNDARY_FIDELITY",
        "HYPOTHESIS_UNCERTAINTY_SCOPE",
        "LATER_DENIAL_NO_CURRENT_PROJECTION", "NESTED_HOLDER_SCOPE",
        "PARAPHRASE_VERSION_LINK",
        "PRONOUN_TRANSFER_HOLDER_PRESERVATION",
        "QUOTE_BOUNDARY_SHORTCUT_REJECT", "REPORTED_AS_FACT_REJECT",
        "RETENTION_REVERIFY", "REVERSE_GENERATION_ATTRIBUTION_UNCERTAINTY",
        "SOURCE_CONFLICT_NO_CURRENT_PROJECTION",
        "TENSE_TRANSFER_ANCHOR_PRESERVATION", "UNKNOWN_SCOPE_NO_GUESS",
    }
    assert "RUNTIME_SCOPE_CONSUMER_NOT_EXECUTED" in lc14.ne_conditions
    lc15 = by_key["LC15_FINAL_LEARNING_OBJECTIVE_VERIFIER_V1"]
    assert lc15.registry_state == "RUNTIME_EVIDENCED"
    assert lc15.can_issue_runtime_pass == 0
    assert "TYPED_LEARNING_OBJECTIVES" in lc15.capability_keys
    assert set(lc15.decidable_dimensions) == {
        "BASELINE_ABLATION_PRE_REGISTRATION",
        "CANDIDATE_LIFECYCLE_BINDING", "CAPABILITY_OBJECTIVE_BINDING",
        "COURSE_MANIFEST_HASH_BINDING", "EVIDENCE_OWNER_ISOLATION",
        "SAMPLE_FAMILY_COVERAGE", "ZERO_RUNTIME_PASS_AUTHORITY",
    }
    assert "CANDIDATE_ELIMINATION_NOT_EXECUTED" in lc15.ne_conditions
    assert "RUNTIME_ABLATION_NOT_EXECUTED" in lc15.ne_conditions
    multi_legal = by_key["MULTI_LEGAL_SURFACE_VERIFIER_V1"]
    assert multi_legal.registry_state == "COURSE_FROZEN"
    assert multi_legal.owner_key == "GG03_EVALUATOR_OWNER"
    assert multi_legal.can_issue_runtime_pass == 0
    lc09 = by_key["LC09_TRANSFER_AXIS_VERIFIER_V1"]
    assert lc09.registry_state == "COURSE_FROZEN"
    assert lc09.owner_key == "LC09_EVALUATOR_OWNER"
    assert lc09.can_issue_runtime_pass == 0
    assert set(lc09.decidable_dimensions) == {
        "ALL_FORMAL_PACKS_INVENTORIED", "ALL_TRANSFER_AXES_EXPLICIT_OR_NE",
        "DOUBLE_AXIS_HELD_OUT", "FULL_COMBINATION_HELD_OUT",
        "NO_DOMAIN_TO_GLOBAL_EXTRAPOLATION", "SCOPE_CONTRACTION_REPLAYABLE",
        "SINGLE_AXIS_HELD_OUT", "ZERO_HOST_LEARNING_WRITE",
    }
    assert "TRANSFER_RESULT_NOT_OBSERVED" in lc09.ne_conditions
    lc10 = by_key["RETENTION_ROLLBACK_VERIFIER_V1"]
    assert lc10.registry_state == "COURSE_FROZEN"
    assert lc10.owner_key == "LC10_EVALUATOR_OWNER"
    assert lc10.can_issue_runtime_pass == 0
    assert set(lc10.decidable_dimensions) == {
        "A_B_REVERIFY_A", "CORE_BIT_IDENTITY",
        "DUMP_RESUME_BIT_IDENTITY", "NO_MEAN_MASKING",
        "ROLLBACK_LOCALITY", "SCOPE_CONTRACTION_REPLAYABLE",
        "SOURCE_WITHDRAWAL_LOCALITY",
        "UNAFFECTED_CAPABILITY_BIT_IDENTITY", "ZERO_HOST_LEARNING_WRITE",
    }
    assert "RETENTION_EPISODE_NOT_EXECUTED" in lc10.ne_conditions
    assert "V06_CLONE_NOT_EXECUTED" in lc10.ne_conditions
    lc13 = by_key["LC13_DIRECTIONAL_CONSUMER_VERIFIER_V1"]
    assert lc13.registry_state == "COURSE_FROZEN"
    assert lc13.owner_key == "LC13_DIRECTIONAL_EVALUATOR_OWNER"
    assert lc13.can_issue_runtime_pass == 0
    assert set(lc13.capability_keys) == set(CAPABILITY_KEYS)
    assert set(lc13.decidable_dimensions) == {
        "ALL_CAPABILITY_DIRECTIONS_INVENTORIED", "APPLICABILITY_EXPLICIT",
        "CONSUMER_FILE_IDENTITY", "CONSUMER_STATE_HONEST",
        "DIRECTION_OWNER_EXPLICIT", "EXACT_USE_OUTCOME_STATE_EXPLICIT",
        "LAYERED_POSTCHECK_ROUTE", "NE_FOR_MISSING_CONSUMER",
        "NO_CROSS_DIRECTION_BROADCAST",
        "WRITE_PERMISSION_LEAST_PRIVILEGE", "ZERO_HOST_LEARNING_WRITE",
    }
    assert "DIRECTIONAL_CONSUMER_NOT_CONNECTED" in lc13.ne_conditions
    nl00 = by_key["NL00_NONLITERAL_SCOPE_PROBE_VERIFIER_V1"]
    assert nl00.registry_state == "COURSE_FROZEN"
    assert nl00.owner_key == "NL00_EVALUATOR_OWNER"
    assert nl00.can_issue_runtime_pass == 0
    assert set(nl00.capability_keys) == {"NONLITERAL_CULTURAL"}
    assert set(nl00.decidable_dimensions) == {
        "CANDIDATE_SOURCE_EXPLICIT", "COUNTEREXAMPLE_FAMILY_PRESENT",
        "GROUNDING_BOUNDARY_EXPLICIT", "LAYER_SPECIFIC_REPRESENTABILITY",
        "NO_DEFINITIVE_MIND_READING", "NO_SCOPE_AGGREGATION",
        "SOURCE_SCOPE_PRESERVATION", "VERIFIER_AUTHORITY_BOUNDED",
        "ZERO_HOST_LEARNING_WRITE",
    }
    assert "CULTURAL_GROUNDING_EVALUATOR_UNAUTHORIZED" in nl00.ne_conditions
    assert "DISC08_DEPTH_UNDECIDED" in nl00.ne_conditions
    assert "DISC12_EVALUATOR_SIGNAL_UNDECIDED" in nl00.ne_conditions
    ri00 = by_key["RI00_REASONING_MODE_PROBE_VERIFIER_V1"]
    assert ri00.registry_state == "COURSE_FROZEN"
    assert ri00.owner_key == "RI00_EVALUATOR_OWNER"
    assert ri00.can_issue_runtime_pass == 0
    assert set(ri00.decidable_dimensions) == {
        "ABDUCTION_NO_NEW_CAUSES",
        "COUNTERFACTUAL_CURRENT_PROJECTION_ISOLATION",
        "DEFEASIBLE_EXCEPTION_REVERSAL",
        "DEONTIC_NORMATIVE_FACT_SEPARATION", "MODE_SPECIFIC_SCOPE",
        "NO_AGGREGATE_MASKING", "SOURCE_SCOPE_PRESERVATION",
        "TEMPORAL_FOUR_STATE", "ZERO_HOST_LEARNING_WRITE",
    }
    assert "COUNTERFACTUAL_BRANCH_RUNTIME_ABSENT" in ri00.ne_conditions


def test_lc12_lists_every_gap_and_has_acyclic_prerequisites():
    """未列出不等于完成；所有能力必须有最早失效、后缀和七类课程状态。"""
    ledger = build_course_coverage_ledger()
    assert tuple(item.capability_key for item in ledger.records) == CAPABILITY_KEYS
    assert all(item.earliest_failure_stage for item in ledger.records)
    assert all(item.failure_suffix for item in ledger.records)
    assert all(set(item.sample_family_states.to_value()) == set(SAMPLE_FAMILIES)
               for item in ledger.records)
    records = {item.capability_key: item for item in ledger.records}
    assert all("LC13_DIRECTIONAL_CONSUMER_MANIFEST_V1"
               in item.external_prerequisites for item in ledger.records)
    assert all(
        "data/ph2/manifests/lc13_directional_consumer_manifest_v1.json"
        in item.evidence_refs for item in ledger.records)
    assert {item.capability_key for item in ledger.records
            if item.exit_state == "COURSE_FROZEN"} == {
                "ATTRIBUTION_QUOTATION_PERSPECTIVE",
                "COMPARISON_QUANTITY_MEASURE",
                "DISCOURSE_INFORMATION_STRUCTURE",
                "EVALUATOR_RETENTION_RESOURCE", "EVENT_TIME_ASPECT",
                "LAYERED_GENERATION",
                "MORPHOLOGY_WORD_FORM",
                "MULTIWORD_CONSTRUCTION", "OPEN_SET_CONTINUAL_LEARNING",
                 "PRAGMATIC_CLARIFICATION_REPAIR", "RAW_TEXT_NOISE",
                 "RECURSIVE_PARSE",
                 "REFERENCE_DISCOURSE_REVISION",
                 "TRANSFER_AXES",
                 "TYPED_LEARNING_OBJECTIVES"}
    assert set(records["RAW_TEXT_NOISE"].sample_family_states.to_value().values()) == {
        "FROZEN"}
    assert set(records[
        "MORPHOLOGY_WORD_FORM"].sample_family_states.to_value().values()) == {
            "FROZEN"}
    assert set(records[
        "MULTIWORD_CONSTRUCTION"].sample_family_states.to_value().values()) == {
            "FROZEN"}
    assert set(records[
        "RECURSIVE_PARSE"].sample_family_states.to_value().values()) == {
            "FROZEN"}
    assert set(records[
        "EVENT_TIME_ASPECT"].sample_family_states.to_value().values()) == {
            "FROZEN"}
    assert set(records[
        "COMPARISON_QUANTITY_MEASURE"].sample_family_states.to_value().values()) == {
            "FROZEN"}
    assert set(records[
        "DISCOURSE_INFORMATION_STRUCTURE"].sample_family_states.to_value().values()) == {
            "FROZEN"}
    assert set(records[
        "REFERENCE_DISCOURSE_REVISION"].sample_family_states.to_value().values()) == {
            "FROZEN"}
    assert set(records[
        "ATTRIBUTION_QUOTATION_PERSPECTIVE"].sample_family_states.to_value().values()) == {
            "FROZEN"}
    assert set(records["TRANSFER_AXES"].sample_family_states.to_value().values()) == {
        "NE"}
    assert set(records[
        "EVALUATOR_RETENTION_RESOURCE"].sample_family_states.to_value().values()) == {
            "NE"}
    assert "LC10_RETENTION_ROLLBACK_MANIFEST_V1" in records[
        "EVALUATOR_RETENTION_RESOURCE"].external_prerequisites
    assert "data/ph2/manifests/lc10_retention_rollback_manifest_v1.json" in (
        records["EVALUATOR_RETENTION_RESOURCE"].evidence_refs)
    relation_logic = records["RELATION_LOGIC_FOUR_STATE"]
    assert relation_logic.exit_state == "PARTIAL_COURSE"
    assert "RI00_REASONING_MODE_PROBE_MANIFEST_V1" in (
        relation_logic.external_prerequisites)
    assert "data/ph2/manifests/ri00_reasoning_mode_probe_manifest_v2.json" in (
        relation_logic.evidence_refs)
    nonliteral = records["NONLITERAL_CULTURAL"]
    assert nonliteral.exit_state == "BASELINE_ONLY"
    assert set(nonliteral.sample_family_states.to_value().values()) == {
        "MISSING"}
    assert "NL00_NONLITERAL_SCOPE_PROBE_MANIFEST_V1" in (
        nonliteral.external_prerequisites)
    assert "data/ph2/manifests/nl00_nonliteral_scope_probe_manifest_v1.json" in (
        nonliteral.evidence_refs)
    assert "LC09_TRANSFER_AXIS_MANIFEST_V1" in (
        records["TRANSFER_AXES"].external_prerequisites)
    assert "data/ph2/manifests/lc09_transfer_axis_manifest_v1.json" in (
        records["TRANSFER_AXES"].evidence_refs)
    assert "LC14_ATTRIBUTION_QUOTATION_COURSE_V1" in records[
        "ATTRIBUTION_QUOTATION_PERSPECTIVE"].external_prerequisites
    attribution_capability = next(
        item for item in build_capability_ledger().entries
        if item.capability_key == "ATTRIBUTION_QUOTATION_PERSPECTIVE")
    assert set(attribution_capability.resource_contracts) == {
        "MAX_ATTRIBUTIONS=3", "MAX_DEPENDENCY_EDGES=3",
        "MAX_LOGIC_STEPS=ABSENT", "MAX_NESTING_DEPTH=2",
        "MAX_OUTPUT_UNITS=160", "MAX_PAGE_SEGMENTS=ABSENT",
        "MAX_PROPOSITIONS=2", "MAX_QUERIES=ABSENT", "MAX_QUOTE_SPANS=2",
        "MAX_RECOMPUTE_OBJECTS=ABSENT", "MAX_RECURSION_DEPTH=ABSENT",
    }
    for capability_key in (
            "OPEN_SET_CONTINUAL_LEARNING",
            "PRAGMATIC_CLARIFICATION_REPAIR"):
        assert set(records[
            capability_key].sample_family_states.to_value().values()) == {
                "FROZEN"}
        assert "LC08_OPEN_SET_CLARIFICATION_COURSE_V1" in (
            records[capability_key].external_prerequisites)
        assert "MD03_DIRECTIONAL_CENTER_ADAPTER_V1" in (
            records[capability_key].external_prerequisites)
        assert "MD03_RUNTIME_PREREQUISITE_MISSING" not in (
            records[capability_key].external_prerequisites)
    assert "LC02_MORPHOLOGY_COURSE_V1" in (
        records["MORPHOLOGY_WORD_FORM"].external_prerequisites)
    assert "LC03_CONSTRUCTION_COURSE_V1" in (
        records["MULTIWORD_CONSTRUCTION"].external_prerequisites)
    assert "LC04_RECURSIVE_PARSE_COURSE_V1" in (
        records["RECURSIVE_PARSE"].external_prerequisites)
    assert "LC05_EVENT_TIME_ASPECT_COURSE_V1" in (
        records["EVENT_TIME_ASPECT"].external_prerequisites)
    assert "LC06_COMPARISON_QUANTITY_COURSE_V1" in (
        records["COMPARISON_QUANTITY_MEASURE"].external_prerequisites)
    for capability_key in (
            "DISCOURSE_INFORMATION_STRUCTURE", "REFERENCE_DISCOURSE_REVISION"):
        assert "LC07_DISCOURSE_INFORMATION_COURSE_V1" in (
            records[capability_key].external_prerequisites)
        assert "MD02_SITUATION_STATE_ADAPTER_V1" in (
            records[capability_key].external_prerequisites)
        assert "MD02_RUNTIME_PREREQUISITE_MISSING" not in (
            records[capability_key].external_prerequisites)
        assert "MD03_DIRECTIONAL_CENTER_ADAPTER_V1" in (
            records[capability_key].external_prerequisites)
        assert "MD03_RUNTIME_PREREQUISITE_MISSING" not in (
            records[capability_key].external_prerequisites)
    recursive_capability = next(
        item for item in build_capability_ledger().entries
        if item.capability_key == "RECURSIVE_PARSE")
    assert set(recursive_capability.resource_contracts) == {
        "MAX_CANDIDATES=2", "MAX_LOGIC_STEPS=ABSENT",
        "MAX_OUTPUT_UNITS=64", "MAX_PAGE_SEGMENTS=ABSENT",
        "MAX_QUERIES=ABSENT", "MAX_RECOMPUTE_OBJECTS=12",
        "MAX_RECURSION_DEPTH=4",
    }
    event_time_capability = next(
        item for item in build_capability_ledger().entries
        if item.capability_key == "EVENT_TIME_ASPECT")
    assert set(event_time_capability.resource_contracts) == {
        "MAX_CANDIDATES=2", "MAX_LOGIC_STEPS=ABSENT",
        "MAX_OUTPUT_UNITS=64", "MAX_PAGE_SEGMENTS=ABSENT",
        "MAX_QUERIES=ABSENT", "MAX_RECOMPUTE_OBJECTS=8",
        "MAX_RECURSION_DEPTH=ABSENT",
    }
    comparison_capability = next(
        item for item in build_capability_ledger().entries
        if item.capability_key == "COMPARISON_QUANTITY_MEASURE")
    assert set(comparison_capability.resource_contracts) == {
        "MAX_CANDIDATES=2", "MAX_COMPARISONS=2",
        "MAX_LOGIC_STEPS=ABSENT", "MAX_OBJECTS=2",
        "MAX_OUTPUT_UNITS=80", "MAX_PAGE_SEGMENTS=ABSENT",
        "MAX_QUANTIFIERS=2", "MAX_QUANTITIES=2", "MAX_QUERIES=ABSENT",
        "MAX_RECURSION_DEPTH=ABSENT", "MAX_SCALES=1", "MAX_STANDARDS=2",
        "MAX_UNITS=1",
    }
    for capability_key in (
            "DISCOURSE_INFORMATION_STRUCTURE", "REFERENCE_DISCOURSE_REVISION"):
        discourse_capability = next(
            item for item in build_capability_ledger().entries
            if item.capability_key == capability_key)
        assert set(discourse_capability.resource_contracts) == {
            "MAX_CANDIDATES=2", "MAX_DEPENDENCY_EDGES=4",
            "MAX_INFORMATION_STATES=1", "MAX_LOGIC_STEPS=ABSENT",
            "MAX_OUTPUT_UNITS=160", "MAX_PAGE_SEGMENTS=ABSENT",
            "MAX_PRESUPPOSITIONS=1", "MAX_PROPOSITIONS=2", "MAX_QUD=1",
            "MAX_QUERIES=ABSENT", "MAX_RECOMPUTE_OBJECTS=ABSENT",
            "MAX_RECURSION_DEPTH=ABSENT", "MAX_RELATIONS=2",
        }
    for capability_key in (
            "OPEN_SET_CONTINUAL_LEARNING",
            "PRAGMATIC_CLARIFICATION_REPAIR"):
        open_set_capability = next(
            item for item in build_capability_ledger().entries
            if item.capability_key == capability_key)
        assert set(open_set_capability.resource_contracts) == {
            "MAX_CANDIDATE_BRANCHES=3", "MAX_DEPENDENCY_EDGES=6",
            "MAX_EVIDENCE_REQUESTS=1", "MAX_LOGIC_STEPS=ABSENT",
            "MAX_OBLIGATIONS=1", "MAX_OUTPUT_UNITS=160",
            "MAX_PAGE_SEGMENTS=ABSENT", "MAX_QUERIES=ABSENT",
            "MAX_QUESTION_COST=1", "MAX_RECOMPUTE_OBJECTS=ABSENT",
            "MAX_RECURSION_DEPTH=ABSENT",
        }
    assert records["TYPED_LEARNING_OBJECTIVES"].exit_state == "COURSE_FROZEN"
    assert "LC15_FINAL_LEARNING_OBJECTIVES_V1" in (
        records["TYPED_LEARNING_OBJECTIVES"].external_prerequisites)
    assert "LC07_DISCOURSE_INFORMATION_COURSE_V1" in (
        records["TYPED_LEARNING_OBJECTIVES"].external_prerequisites)
    assert "LC08_OPEN_SET_CLARIFICATION_COURSE_V1" in (
        records["TYPED_LEARNING_OBJECTIVES"].external_prerequisites)
    assert "LC14_ATTRIBUTION_QUOTATION_COURSE_V1" in (
        records["TYPED_LEARNING_OBJECTIVES"].external_prerequisites)
    objective_capability = next(
        item for item in build_capability_ledger().entries
        if item.capability_key == "TYPED_LEARNING_OBJECTIVES")
    assert set(objective_capability.resource_contracts) == {
        "BASELINE_ABLATIONS=4", "CAPABILITY_BINDINGS=12",
        "COURSE_SOURCES=9", "LEARNING_OBJECTIVES=11",
        "MAX_LOGIC_STEPS=ABSENT", "MAX_OUTPUT_UNITS=ABSENT",
        "MAX_PAGE_SEGMENTS=ABSENT", "MAX_QUERIES=ABSENT",
        "MAX_RECOMPUTE_OBJECTS=ABSENT", "MAX_RECURSION_DEPTH=ABSENT",
    }
    assert "W02_CC_CEDICT_BLOCKED_ALTERNATIVES_PARTIAL" in (
        records["MORPHOLOGY_WORD_FORM"].external_prerequisites)
    assert "W03_CC_CEDICT_BLOCKED_ALTERNATIVES_PARTIAL" in (
        records["MULTIWORD_CONSTRUCTION"].external_prerequisites)
    assert "W03_CC_CEDICT_BLOCKED_ALTERNATIVES_PARTIAL" in (
        records["SOURCE_UNCERTAINTY_REALITY"].external_prerequisites)
    assert {item.exit_state for item in ledger.records} == {
        "BASELINE_ONLY", "COURSE_FROZEN", "OUT_OF_SCOPE", "PARTIAL_COURSE",
    }


def test_missing_family_false_runtime_and_course_cycle_fail_closed():
    """漏能力、假 runtime state 和前置环均不得恢复为正式基线。"""
    coverage = build_capability_ledger()
    with pytest.raises(LanguageCoverageContractError, match="列全"):
        LanguageCapabilityCoverageLedger(
            1, coverage.ledger_version, coverage.scope_statement,
            coverage.entries[:-1])
    first = coverage.entries[0]
    states = first.fact_states.to_value()
    states["REPRESENTATION"] = "RUNTIME_EVIDENCED"
    with pytest.raises(LanguageCoverageContractError, match="ACTIVE_RUNTIME"):
        replace(first, fact_states=CanonicalJsonObject.from_value(states))

    courses = build_course_coverage_ledger()
    by_key = {item.capability_key: item for item in courses.records}
    raw = by_key["RAW_TEXT_NOISE"]
    by_key["RAW_TEXT_NOISE"] = replace(
        raw, prerequisite_capability_keys=("MORPHOLOGY_WORD_FORM",))
    with pytest.raises(LanguageCoverageContractError, match="有环"):
        CapabilityCourseCoverageLedger(
            1, courses.ledger_version, tuple(by_key.values()))


def test_course_frozen_cannot_hide_missing_retention():
    """只冻结部分课程时不能把退出事实改成 COURSE_FROZEN。"""
    courses = build_course_coverage_ledger()
    relation = next(item for item in courses.records
                    if item.capability_key == "RELATION_LOGIC_FOUR_STATE")
    assert relation.sample_family_states.to_value()["RETENTION"] == "MISSING"
    with pytest.raises(LanguageCoverageContractError, match="隐藏缺失"):
        replace(relation, exit_state="COURSE_FROZEN")


def test_teacher_owner_and_unproven_runtime_pass_fail_closed():
    """teacher owner 或无 runtime evidence 的 PASS 权限均属越权。"""
    record = build_verifier_registry().records[0]
    with pytest.raises(LanguageCoverageContractError, match="teacher"):
        replace(record, owner_key="TEACHER")
    designed = replace(record, registry_state="DESIGNED")
    with pytest.raises(LanguageCoverageContractError, match="不得发 PASS"):
        replace(designed, can_issue_runtime_pass=1)


def test_md00_is_result_blind_and_gg03_course_freeze_is_complete():
    """MD 阈值先冻结，GG-03 课程、组合轴和阶段配对必须完整列出。"""
    md = build_md00_preregistration()
    assert md.decision_state == "PRE_REGISTERED"
    assert md.results_observed == 0
    assert md.threshold_policy.to_value()["freeze_before_run"] == 1
    with pytest.raises(LanguageBaselineManifestError, match="不得先看结果"):
        replace(md, results_observed=1)
    gg = build_gg00_audit()
    assert gg.gg03_exit_state == "COURSE_FROZEN"
    assert tuple(item.row_key for item in gg.course_family_rows) == (
        GG_COURSE_FAMILY_KEYS)
    assert tuple(item.row_key for item in gg.combination_axis_rows) == (
        GG_COMBINATION_AXIS_KEYS)
    assert tuple(item.row_key for item in gg.stage_rows) == GG_STAGE_KEYS
    assert all(item.state == "PRESENT" for item in gg.course_family_rows)
    assert all(item.state == "PRESENT" for item in gg.combination_axis_rows)
    assert all(item.state == "PRESENT" for item in gg.stage_rows)
    w08 = next(item for item in gg.stage_rows if item.row_key == "W08")
    assert w08.state == "PRESENT"
    assert w08.gap_code == "NONE"
    assert "data/ph2/manifests/gg03_generation_generalization_course_v1.json" in (
        w08.evidence_refs)
    assert "data/ph2/manifests/lc07_discourse_information_course_v1.json" in (
        w08.evidence_refs)
    assert "data/ph2/manifests/lc08_open_set_clarification_course_v1.json" in (
        w08.evidence_refs)
    assert "data/ph2/manifests/lc14_attribution_quotation_course_v1.json" in (
        w08.evidence_refs)
    missing_row = replace(
        gg.course_family_rows[0], state="PARTIAL", gap_code="BROKEN")
    with pytest.raises(LanguageBaselineManifestError, match="不得隐藏"):
        replace(gg, course_family_rows=(missing_row, *gg.course_family_rows[1:]))
    with pytest.raises(LanguageBaselineManifestError, match="证据冲突"):
        replace(gg, gg03_exit_state="MISSING")


def test_file_inventory_and_pattern_scan_do_not_copy_matched_text(tmp_path):
    """LG-00 保存文件/hash/行 hash，不把命中内容复制进 artifact。"""
    _, manifest = _synthetic_manifest(tmp_path)
    assert len(manifest.file_inventory) == 2
    assert len(manifest.public_gate.legacy_findings) == 1
    finding = manifest.public_gate.legacy_findings[0]
    assert set(finding.to_dict()) == {
        "line_number", "line_sha256", "relative_path", "rule_key",
    }
    assert manifest.public_gate.legacy_status == "BLOCKED"
    assert manifest.public_gate.secret_status == "CLEAR"
    assert manifest.public_gate.public_release_allowed == 0


def test_baseline_round_trip_nonoverwrite_and_strict_fields(tmp_path):
    """合并 manifest 可规范恢复、幂等核对并拒绝覆盖或额外字段。"""
    _, manifest = _synthetic_manifest(tmp_path)
    output = tmp_path / "baseline.json"
    write_language_baseline_manifest(manifest, output)
    restored = read_language_baseline_manifest(output)
    assert restored == manifest
    write_language_baseline_manifest(manifest, output)
    output.write_bytes(canonical_json_line({"damaged": 1}))
    with pytest.raises(LanguageBaselineManifestError, match="内容不同"):
        write_language_baseline_manifest(manifest, output)
    value = manifest.to_dict()
    value["readiness"] = 1
    output.write_bytes(canonical_json_line(value))
    with pytest.raises(LanguageBaselineManifestError, match="字段不精确"):
        read_language_baseline_manifest(output)


def test_bad_paths_scan_scope_and_public_verdict_fail_closed(tmp_path):
    """路径逃逸、漏扫文件和伪造 CLEAR 都必须停线。"""
    repo, manifest = _synthetic_manifest(tmp_path)
    with pytest.raises(LanguageBaselineManifestError, match="安全 POSIX"):
        inventory_public_files(repo, ("../outside",))
    gate = manifest.public_gate
    with pytest.raises(LanguageBaselineManifestError, match="范围未闭合"):
        replace(gate, scanned_text_file_count=1)
    with pytest.raises(LanguageBaselineManifestError, match="不诚实"):
        replace(gate, legacy_status="CLEAR")


def test_file_level_verify_detects_candidate_or_paper_drift(tmp_path):
    """LG-00 清单必须能逐字节回验候选和论文，不只保存计数。"""
    repo, manifest = _synthetic_manifest(tmp_path)
    exclusion = repo / manifest.inventory_exclusions[0]
    exclusion.parent.mkdir(parents=True)
    exclusion.write_bytes(manifest.canonical_bytes())
    verify_language_baseline_files(manifest, repo_root=repo)
    (repo / "src" / "one.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(LanguageBaselineManifestError, match="public file"):
        verify_language_baseline_files(manifest, repo_root=repo)


def test_repository_baseline_is_explicitly_blocked_not_learned():
    """正式基线记录当前阻断和缺口，不声明 D-03/readiness/mastered。"""
    manifest = read_language_baseline_manifest(MANIFEST_PATH)
    assert manifest.artifact_status == "BASELINE_FROZEN"
    assert manifest.artifact_version == (
        "LG-LC-MD-GG-baseline-v39-supersedes-v1-v2-v3-v4-v5-v6-v7-v8-v9-v10-v11-v12-v13-v14-v15-v16-v17-v18-v19-v20-v21-v22-v23-v24-v25-v26-v27-v28-v29-v30-v31-v32-v33-v34-v35-v36-v37-v38")
    assert manifest.head_sha1 == (
        "bf7a3a723d33cdafefcf365a9f3dec74ac5cc194")
    assert manifest.public_gate.legacy_status == "CLEAR"
    assert manifest.untracked_file_count == 275
    assert len(manifest.file_inventory) == 274
    assert manifest.inventory_exclusions == (
        "data/ph2/manifests/language_capability_baseline_v39.json",)
    assert manifest.public_gate.legacy_findings == ()
    assert manifest.public_gate.secret_status == "CLEAR"
    assert manifest.public_gate.secret_findings == ()
    assert manifest.public_gate.final_rescan_required == 1
    assert manifest.public_gate.public_release_allowed == 0
    assert all(value == 0 for value in manifest.execution_state.to_value().values())
    assert manifest.gg00_audit.gg03_exit_state == "COURSE_FROZEN"
    assert manifest.md00_preregistration.results_observed == 0
    layered_generation = next(
        item for item in manifest.capability_ledger.entries
        if item.capability_key == "LAYERED_GENERATION")
    assert layered_generation.fact_states.to_value()["REPRESENTATION"] == (
        "COURSE_FROZEN")
    assert layered_generation.directional_consumption.to_value()[
        "GENERATION"]["fact_state"] == "COURSE_FROZEN"
    assert layered_generation.directional_consumption.to_value()[
        "REASONING"]["fact_state"] == "ABSENT"
    assert "GG01_GENERATION_CHOICE_CONTRACT_VERIFIER_V1" in (
        layered_generation.verifier_keys)
    assert "GG01_GENERATION_CHOICE_CONTRACT_VERIFIER_V2" in (
        layered_generation.verifier_keys)
    assert "GG02_GENERATION_CHOICE_OUTCOME_BRIDGE_VERIFIER_V1" in (
        layered_generation.verifier_keys)
    assert "GG03_GENERATION_GENERALIZATION_COURSE_VERIFIER_V1" in (
        layered_generation.verifier_keys)
    assert {
        "GENERATION_CHOICE_HYPOTHESIS_V2_ZERO_BEARING_EXACT_KEYS",
        "GENERATION_CHOICE_EPISODE_ATTRIBUTION_V1",
        "GENERATION_CHOICE_LAYER_OUTCOME_V1",
        "GENERATION_COMBINATION_SPLIT_V1",
        "GENERATION_GENERALIZATION_CANDIDATE_V1",
        "GENERATION_LAYERED_OUTCOME_REPORT_V1",
        "LOSSLESS_INTEGER_KEY_V1",
        "MULTI_LEGAL_SURFACE_SET_CONSTRAINT_V1",
    } <= set(layered_generation.representation_contracts)
    transfer = next(
        item for item in manifest.capability_ledger.entries
        if item.capability_key == "TRANSFER_AXES")
    assert transfer.implementation_state == "SCAFFOLD_ONLY"
    assert "LC09_TRANSFER_AXIS_MANIFEST_V1" in (
        transfer.representation_contracts)
    assert "LC09_TRANSFER_AXIS_VERIFIER_V1" in transfer.verifier_keys
    assert transfer.fact_states.to_value()["REPRESENTATION"] == (
        "COURSE_FROZEN")
    assert transfer.fact_states.to_value()["RETENTION"] == "ABSENT"
    retention = next(
        item for item in manifest.capability_ledger.entries
        if item.capability_key == "EVALUATOR_RETENTION_RESOURCE")
    assert retention.fact_states.to_value()["RETENTION"] == "COURSE_FROZEN"
    assert "RETENTION_EVIDENCED" not in (
        retention.fact_states.to_value().values())
    assert "LC10_RETENTION_ROLLBACK_MANIFEST_V1" in (
        retention.representation_contracts)
    assert "RETENTION_ROLLBACK_VERIFIER_V1" in retention.verifier_keys
    lc10_verifier = next(
        item for item in manifest.verifier_registry.records
        if item.verifier_key == "RETENTION_ROLLBACK_VERIFIER_V1")
    assert lc10_verifier.registry_state == "COURSE_FROZEN"
    assert lc10_verifier.can_issue_runtime_pass == 0
    assert all(
        item.fact_states.to_value()["DIRECTIONAL_CONSUMPTION"]
        == ("WALL_BLOCKED" if item.capability_key == "NON_TEXT_MEDIA"
            else "COURSE_FROZEN")
        for item in manifest.capability_ledger.entries)
    assert all("LC13_DIRECTIONAL_CONSUMER_MAP_V1"
               in item.representation_contracts
               for item in manifest.capability_ledger.entries)
    lc13_verifier = next(
        item for item in manifest.verifier_registry.records
        if item.verifier_key == "LC13_DIRECTIONAL_CONSUMER_VERIFIER_V1")
    assert lc13_verifier.registry_state == "COURSE_FROZEN"
    assert lc13_verifier.can_issue_runtime_pass == 0
    memory_dynamics = next(
        item for item in manifest.capability_ledger.entries
        if item.capability_key == "MEMORY_DYNAMICS")
    assert memory_dynamics.fact_states.to_value()["REPRESENTATION"] == (
        "COURSE_FROZEN")
    assert "MD01_MEMORY_DYNAMICS_CONTRACT_VERIFIER_V1" in (
        memory_dynamics.verifier_keys)
    assert "MD02_SITUATION_STATE_ADAPTER_VERIFIER_V1" in (
        memory_dynamics.verifier_keys)
    assert "MD03_DIRECTIONAL_CENTER_ADAPTER_VERIFIER_V1" in (
        memory_dynamics.verifier_keys)
    assert "MD05_CENTER_DIFFUSION_PROBE_VERIFIER_V1" in (
        memory_dynamics.verifier_keys)
    assert {
        "CURRENT_SITUATION_PROJECTION_V1",
        "DIRECTIONAL_CENTER_PROFILE_V1",
        "DIRECTIONAL_MEMORY_CENTER_V1",
        "DIRECTIONAL_WRITE_BOUNDARY_V1",
        "MEMORY_CENTER_FORMATION_REPORT_V1",
        "MD04_PROBE_PLAN_V1", "MD04_PROBE_RUN_ARTIFACT_V1",
        "MD05_PROBE_DECISION_V1",
        "SITUATION_DEPENDENCY_INDEX_V1",
        "SITUATION_EVENT_LOG_FACADE_V1",
        "SITUATION_REBUILD_RECEIPT_V1",
    } <= set(memory_dynamics.representation_contracts)
    relation_logic = next(
        item for item in manifest.capability_ledger.entries
        if item.capability_key == "RELATION_LOGIC_FOUR_STATE")
    assert relation_logic.fact_states.to_value()["REPRESENTATION"] == (
        "COURSE_FROZEN")
    assert "RI00_ADDITIONAL_REASONING_MODE_DECISION_V1" in (
        relation_logic.representation_contracts)
    assert "RI00_REASONING_MODE_PROBE_VERIFIER_V1" in (
        relation_logic.verifier_keys)
    ri00_verifier = next(
        item for item in manifest.verifier_registry.records
        if item.verifier_key == "RI00_REASONING_MODE_PROBE_VERIFIER_V1")
    assert ri00_verifier.registry_state == "COURSE_FROZEN"
    assert ri00_verifier.can_issue_runtime_pass == 0
    relation_course = next(
        item for item in manifest.course_coverage_ledger.records
        if item.capability_key == "RELATION_LOGIC_FOUR_STATE")
    assert relation_course.exit_state == "PARTIAL_COURSE"
    assert "RI00_REASONING_MODE_PROBE_MANIFEST_V1" in (
        relation_course.external_prerequisites)
    nonliteral = next(
        item for item in manifest.capability_ledger.entries
        if item.capability_key == "NONLITERAL_CULTURAL")
    assert nonliteral.fact_states.to_value()["SCOPE"] == "COURSE_FROZEN"
    assert nonliteral.fact_states.to_value()["REPRESENTATION"] == "DESIGNED"
    assert nonliteral.fact_states.to_value()["VERIFIER_CAPABILITY"] == "ABSENT"
    assert "NL00_NONLITERAL_SCOPE_DECISION_V1" in (
        nonliteral.representation_contracts)
    assert "NL00_NONLITERAL_SCOPE_PROBE_VERIFIER_V1" in (
        nonliteral.verifier_keys)
    nl00_verifier = next(
        item for item in manifest.verifier_registry.records
        if item.verifier_key == "NL00_NONLITERAL_SCOPE_PROBE_VERIFIER_V1")
    assert nl00_verifier.registry_state == "COURSE_FROZEN"
    assert nl00_verifier.can_issue_runtime_pass == 0
    nonliteral_course = next(
        item for item in manifest.course_coverage_ledger.records
        if item.capability_key == "NONLITERAL_CULTURAL")
    assert nonliteral_course.exit_state == "BASELINE_ONLY"
    assert "NL00_NONLITERAL_SCOPE_PROBE_MANIFEST_V1" in (
        nonliteral_course.external_prerequisites)
    assert {item.capability_key
            for item in manifest.course_coverage_ledger.records
            if item.exit_state == "COURSE_FROZEN"} == {
                "ATTRIBUTION_QUOTATION_PERSPECTIVE",
                "COMPARISON_QUANTITY_MEASURE",
                "DISCOURSE_INFORMATION_STRUCTURE",
                "EVALUATOR_RETENTION_RESOURCE", "EVENT_TIME_ASPECT",
                "LAYERED_GENERATION",
                "MORPHOLOGY_WORD_FORM",
                "MULTIWORD_CONSTRUCTION", "OPEN_SET_CONTINUAL_LEARNING",
                 "PRAGMATIC_CLARIFICATION_REPAIR", "RAW_TEXT_NOISE",
                 "RECURSIVE_PARSE",
                 "REFERENCE_DISCOURSE_REVISION",
                 "TRANSFER_AXES",
                 "TYPED_LEARNING_OBJECTIVES"}
    transfer_course = next(
        item for item in manifest.course_coverage_ledger.records
        if item.capability_key == "TRANSFER_AXES")
    assert set(transfer_course.sample_family_states.to_value().values()) == {
        "NE"}
    assert {item.relative_path: item.sha256 for item in manifest.paper_files} == {
        "paper/main.pdf": (
            "04cfb5d7741117d5888ef8a6018de5de0979f759915b4f863f4df0d77ea04898"),
        "paper/main.tex": (
            "fedde37d06790b919373c23e1bc507275c8fecdcb1150a23d5b20590ef7a15c1"),
    }
    _verify_historical_repository_baseline(manifest)
