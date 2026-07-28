"""切片 2 的能力、verifier、课程、MD 和 GG 基线目录。"""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_language_baseline_manifest import (
    GG_COMBINATION_AXIS_KEYS,
    GG_COURSE_FAMILY_KEYS,
    GG_STAGE_KEYS,
    MD_BASELINE_KEYS,
    MD_HARD_INVARIANT_KEYS,
    MD_SAMPLE_GROUP_KEYS,
    GenerationCoverageAudit,
    GenerationCoverageAuditRow,
    LanguageBaselineManifest,
    MDProbePreRegistration,
    PublicFileIdentity,
    PublicGateBaseline,
)
from pure_integer_ai.experiments.ph2_language_coverage_contract import (
    CAPABILITY_KEYS,
    FACT_DIMENSIONS,
    SAMPLE_FAMILIES,
    CapabilityCourseCoverage,
    CapabilityCourseCoverageLedger,
    LanguageCapabilityCoverageEntry,
    LanguageCapabilityCoverageLedger,
    VerifierCapabilityRecord,
    VerifierCapabilityRegistry,
)
from pure_integer_ai.experiments.ph2_transfer_axis_contract import (
    VERIFIER_DIMENSIONS as LC09_VERIFIER_DIMENSIONS,
    VERIFIER_NE_CONDITIONS as LC09_VERIFIER_NE_CONDITIONS,
)
from pure_integer_ai.experiments.ph2_retention_rollback_contract import (
    VERIFIER_DIMENSIONS as LC10_VERIFIER_DIMENSIONS,
    VERIFIER_NE_CONDITIONS as LC10_VERIFIER_NE_CONDITIONS,
)
from pure_integer_ai.experiments.ph2_directional_consumer_contract import (
    VERIFIER_DIMENSIONS as LC13_VERIFIER_DIMENSIONS,
    VERIFIER_NE_CONDITIONS as LC13_VERIFIER_NE_CONDITIONS,
)
from pure_integer_ai.experiments.ph2_reasoning_mode_probe_contract import (
    VERIFIER_DIMENSIONS as RI00_VERIFIER_DIMENSIONS,
    VERIFIER_NE_CONDITIONS as RI00_VERIFIER_NE_CONDITIONS,
)
from pure_integer_ai.experiments.ph2_nonliteral_scope_probe_contract import (
    VERIFIER_DIMENSIONS as NL00_VERIFIER_DIMENSIONS,
    VERIFIER_NE_CONDITIONS as NL00_VERIFIER_NE_CONDITIONS,
)


_TASKS = {
    "ATTRIBUTION_QUOTATION_PERSPECTIVE": ("LC-14",),
    "COMPARISON_QUANTITY_MEASURE": ("LC-06",),
    "DISCOURSE_INFORMATION_STRUCTURE": ("LC-07",),
    "EVALUATOR_RETENTION_RESOURCE": ("LC-10", "LC-11", "LC-12"),
    "EVENT_TIME_ASPECT": ("LC-05",),
    "LAYERED_GENERATION": ("GG-00", "GG-01", "GG-02", "GG-03", "LC-13"),
    "MEMORY_DYNAMICS": ("MD-00", "MD-01", "MD-02", "MD-03", "MD-04", "MD-05"),
    "MORPHOLOGY_WORD_FORM": ("LC-02",),
    "MULTIWORD_CONSTRUCTION": ("LC-03",),
    "NONLITERAL_CULTURAL": ("NL-00",),
    "NON_TEXT_MEDIA": ("DISC-07",),
    "OPEN_SET_CONTINUAL_LEARNING": ("LC-08", "LC-10"),
    "PRAGMATIC_CLARIFICATION_REPAIR": ("LC-08",),
    "RAW_TEXT_NOISE": ("LC-01",),
    "RECURSIVE_PARSE": ("LC-04",),
    "REFERENCE_DISCOURSE_REVISION": ("LC-07",),
    "RELATION_LOGIC_FOUR_STATE": ("RI-00",),
    "SOURCE_UNCERTAINTY_REALITY": ("LC-05", "LC-07", "LC-11", "LC-14"),
    "TRANSFER_AXES": ("LC-09",),
    "TYPED_LEARNING_OBJECTIVES": ("LC-15",),
}

_SCOPES = {
    "ATTRIBUTION_QUOTATION_PERSPECTIVE": "转述、引语、holder、认识立场和嵌套 scope",
    "COMPARISON_QUANTITY_MEASURE": "比较标准、尺度、程度、数量、单位和近似",
    "DISCOURSE_INFORMATION_STRUCTURE": "篇章关系、预设、topic/focus、given/new 和 QUD",
    "EVALUATOR_RETENTION_RESOURCE": "verifier 能力、后续重验、范围收缩和资源停止",
    "EVENT_TIME_ASPECT": "Event/State、时间锚、区间、时态和体貌",
    "LAYERED_GENERATION": "内容、结构、词形、篇章选择与逐层 Use/outcome",
    "MEMORY_DYNAMICS": "typed obligation、多中心、扩域 ring 和停止决断",
    "MORPHOLOGY_WORD_FORM": "生产性构词、词形、复合、重叠和例外",
    "MULTIWORD_CONSTRUCTION": "连续/非连续多词单位和构式身份",
    "NONLITERAL_CULTURAL": "习语、隐喻、转喻、反讽、幽默和文化典故",
    "NON_TEXT_MEDIA": "代码、数学、表格、标记、语音和多模态指称",
    "OPEN_SET_CONTINUAL_LEARNING": "新词/义/构式发现、主动补证和持续修正",
    "PRAGMATIC_CLARIFICATION_REPAIR": "unknown、ambiguous、最小澄清和交互修复",
    "RAW_TEXT_NOISE": "raw/normalized 双轨、边界候选和不可逆损失",
    "RECURSIVE_PARSE": "递归 span、联合 parse、歧义竞争和 reparse",
    "REFERENCE_DISCOURSE_REVISION": "指代、篇章状态、supersede 和局部重算",
    "RELATION_LOGIC_FOUR_STATE": "关系、组合逻辑、四态和额外推理模式边界",
    "SOURCE_UNCERTAINTY_REALITY": "来源、冲突、不确定性和现实事实边界",
    "TRANSFER_AXES": "来源/领域/文体/长度/时代/方言/语言/code-switch 分轴迁移",
    "TYPED_LEARNING_OBJECTIVES": "重构、预测、次序、扰动、描述长度和生成失败",
}

_IMPLEMENTATION = {
    "ATTRIBUTION_QUOTATION_PERSPECTIVE": "SCAFFOLD_ONLY",
    "COMPARISON_QUANTITY_MEASURE": "SCAFFOLD_ONLY",
    "DISCOURSE_INFORMATION_STRUCTURE": "SCAFFOLD_ONLY",
    "EVALUATOR_RETENTION_RESOURCE": "SCAFFOLD_ONLY",
    "EVENT_TIME_ASPECT": "SCAFFOLD_ONLY",
    "LAYERED_GENERATION": "SCAFFOLD_ONLY",
    "MEMORY_DYNAMICS": "SCAFFOLD_ONLY",
    "MORPHOLOGY_WORD_FORM": "SCAFFOLD_ONLY",
    "MULTIWORD_CONSTRUCTION": "SCAFFOLD_ONLY",
    "NONLITERAL_CULTURAL": "DESIGN_ONLY",
    "NON_TEXT_MEDIA": "ABSENT",
    "OPEN_SET_CONTINUAL_LEARNING": "SCAFFOLD_ONLY",
    "PRAGMATIC_CLARIFICATION_REPAIR": "SCAFFOLD_ONLY",
    "RAW_TEXT_NOISE": "SCAFFOLD_ONLY",
    "RECURSIVE_PARSE": "SCAFFOLD_ONLY",
    "REFERENCE_DISCOURSE_REVISION": "SCAFFOLD_ONLY",
    "RELATION_LOGIC_FOUR_STATE": "SCAFFOLD_ONLY",
    "SOURCE_UNCERTAINTY_REALITY": "SCAFFOLD_ONLY",
    "TRANSFER_AXES": "SCAFFOLD_ONLY",
    "TYPED_LEARNING_OBJECTIVES": "SCAFFOLD_ONLY",
}

_REPRESENTATIONS = {
    "ATTRIBUTION_QUOTATION_PERSPECTIVE": (
        "ATTRIBUTION_QUOTATION_CANDIDATE_V1",
        "CLAIM_BELIEF_HYPOTHESIS_QUOTATION_SCOPE_REQUIRED",
        "SPEAKER_HOLDER_VERSION_REQUIRED",
    ),
    "COMPARISON_QUANTITY_MEASURE": (
        "COMPARISON_QUANTITY_CANDIDATE_V1",
        "COMPARISON_STANDARD_SCALE_UNIT_REQUIRED",
        "PROPOSITION_SET_EXPR_REUSE",
        "RATIONAL_RANGE_QUANTIFIER_SCOPE_REQUIRED",
    ),
    "DISCOURSE_INFORMATION_STRUCTURE": (
        "DISCOURSE_INFORMATION_CANDIDATE_V1",
        "DISCOURSE_RELATION_PRESUPPOSITION_INFORMATION_QUD_REQUIRED",
        "PROPOSITION_CONTEXT_SCOPE_REUSE",
    ),
    "EVALUATOR_RETENTION_RESOURCE": (
        "A_B_REVERIFY_A_PROTOCOL_V1",
        "CAPABILITY_COVERAGE_LEDGER",
        "COURSE_COVERAGE_LEDGER",
        "LC10_RETENTION_ROLLBACK_MANIFEST_V1",
        "RETENTION_CHECKPOINT_DIGEST_V1",
        "RETENTION_DIMENSION_RESULT_V1",
        "RUNTIME_BINDING_FILE_IDENTITY_V1",
        "SOURCE_WITHDRAWAL_SCOPE_CONTRACTION_PROTOCOL_V1",
        "VERIFIER_CAPABILITY_REGISTRY",
    ),
    "EVENT_TIME_ASPECT": (
        "EVENT_STATE_PROPOSITION_REUSE",
        "TIME_ANCHOR_INTERVAL_ASPECT_REQUIRED",
    ),
    "LAYERED_GENERATION": (
        "GENERATION_CHOICE_HYPOTHESIS_V1",
        "GENERATION_CHOICE_HYPOTHESIS_V2_ZERO_BEARING_EXACT_KEYS",
        "GENERATION_CONTEXT_CONTRACT_V1",
        "GENERATION_CHOICE_ASSESSMENT_INPUT_V1",
        "GENERATION_CHOICE_EPISODE_ATTRIBUTION_V1",
        "GENERATION_CHOICE_LAYER_OUTCOME_V1",
        "GENERATION_CHOICE_USE_ATTRIBUTION_V1",
        "GENERATION_LAYERED_OUTCOME_REPORT_V1",
        "GENERATION_VERIFIER_LAYER_ROUTE_V1",
        "GENERATION_GENERALIZATION_CANDIDATE_V1",
        "GENERATION_GENERALIZATION_PAYLOAD_AUDIT_V1",
        "GENERATION_COMBINATION_SPLIT_V1",
        "LOSSLESS_INTEGER_KEY_V1",
        "MULTI_LEGAL_SURFACE_SET_CONSTRAINT_V1",
        "USE_OUTCOME_LAYER_ATTRIBUTION_FROZEN",
    ),
    "MEMORY_DYNAMICS": (
        "CURRENT_SITUATION_PROJECTION_V1",
        "DIRECTIONAL_CENTER_PROFILE_V1",
        "DIRECTIONAL_MEMORY_CENTER_V1",
        "DIRECTIONAL_WRITE_BOUNDARY_V1",
        "MEMORY_ATTENTION_CENTER_V1",
        "MEMORY_CENTER_FORMATION_REPORT_V1",
        "MEMORY_DYNAMICS_RUN_REPORT_V1",
        "MEMORY_DYNAMICS_STOP_DECISION_V1",
        "MEMORY_EXPANSION_PROFILE_V1",
        "MEMORY_CURRENT_QUERY_REASONING_OBLIGATION_REUSE",
        "MEMORY_RING_RECEIPT_V1",
        "MD04_PROBE_PLAN_V1",
        "MD04_PROBE_RUN_ARTIFACT_V1",
        "MD05_PROBE_DECISION_V1",
        "SITUATION_DEPENDENCY_INDEX_V1",
        "SITUATION_EVENT_LOG_FACADE_V1",
        "SITUATION_REBUILD_RECEIPT_V1",
    ),
    "MORPHOLOGY_WORD_FORM": (
        "LANGUAGE_SCOPED_MORPHOLOGY_CANDIDATE_REQUIRED",
        "WORD_FORM_INDEX_REUSE",
    ),
    "MULTIWORD_CONSTRUCTION": (
        "STRUCTURE_CONCEPT_ROLE_BINDING_REUSE",
        "WHOLE_PARTIAL_COMPOSITION_IDENTITY_REQUIRED",
    ),
    "NONLITERAL_CULTURAL": (
        "LAYERED_NONLITERAL_HYPOTHESIS_REQUIRED",
        "NL00_NONLITERAL_SCOPE_DECISION_V1",
        "SOURCE_AND_GROUNDING_SCOPE_REQUIRED",
    ),
    "NON_TEXT_MEDIA": ("WALL_SCOPE_ONLY",),
    "OPEN_SET_CONTINUAL_LEARNING": (
        "CANDIDATE_LEARNING_RUNTIME_REUSE",
        "MISSING_INFORMATION_OBLIGATION_REQUIRED",
        "OPEN_SET_CLARIFICATION_CANDIDATE_V1",
    ),
    "PRAGMATIC_CLARIFICATION_REPAIR": (
        "COMMUNICATIVE_GOAL_HYPOTHESIS_REQUIRED",
        "OPEN_SET_CLARIFICATION_CANDIDATE_V1",
        "QUESTION_ANSWER_COURSE_SCAFFOLD",
    ),
    "RAW_TEXT_NOISE": (
        "OBSERVATION_APPEND_ONLY_REUSE",
        "RAW_NORMALIZED_SEGMENT_TOKEN_RECEIPT_REQUIRED",
    ),
    "RECURSIVE_PARSE": (
        "OCCURRENCE_SPAN_PROPOSITION_REUSE",
        "PARSE_CANDIDATE_COMPETITION_REQUIRED",
    ),
    "REFERENCE_DISCOURSE_REVISION": (
        "CONTEXT_SCOPE_OCCURRENCE_REUSE",
        "DISCOURSE_INFORMATION_CANDIDATE_V1",
        "EVENT_LOG_CURRENT_PROJECTION_DEPENDENCY_INDEX_REQUIRED",
    ),
    "RELATION_LOGIC_FOUR_STATE": (
        "PROPOSITION_RELATION_LOGIC_TYPED_OBJECTS",
        "RI00_ADDITIONAL_REASONING_MODE_DECISION_V1",
        "REASONING_OBLIGATION_REUSE",
    ),
    "SOURCE_UNCERTAINTY_REALITY": (
        "EVIDENCE_SOURCE_REF_REUSE",
        "UNKNOWN_CONFLICT_WALL_STATE_REUSE",
    ),
    "TRANSFER_AXES": (
        "AXIS_SCOPED_APPLICABILITY_REQUIRED",
        "LC09_TRANSFER_AXIS_MANIFEST_V1",
        "MANIFEST_COMBINATION_SPLIT_REQUIRED",
        "SCOPE_CONTRACTION_PROTOCOL_V1",
        "SINGLE_DOUBLE_FULL_COMBINATION_SPLIT_V1",
    ),
    "TYPED_LEARNING_OBJECTIVES": (
        "CANDIDATE_OBJECTIVE_SIGNAL_OWNER_REQUIRED",
        "LC15_FINAL_LEARNING_OBJECTIVE_MANIFEST_V1",
        "MODEL_COMPETITION_TARGET_REQUIRED",
    ),
}

_EVIDENCE = {
    "ATTRIBUTION_QUOTATION_PERSPECTIVE": (
        "data/ph2/manifests/lc14_attribution_quotation_course_v1.json",
        "src/pure_integer_ai/cognition/shared/reasoning_planner.py",
        "src/pure_integer_ai/experiments/ph2_authored_attribution_quotation_course.py",
        "src/pure_integer_ai/storage/source_record.py",
        "tests/test_d02_lc14_attribution_quotation_course.py",
    ),
    "COMPARISON_QUANTITY_MEASURE": (
        "data/ph2/manifests/lc06_comparison_quantity_course_v1.json",
        "src/pure_integer_ai/cognition/shared/semantic_object.py",
        "src/pure_integer_ai/experiments/ph2_authored_comparison_quantity_course.py",
        "src/pure_integer_ai/experiments/ph2_authored_property_course.py",
        "tests/test_d02_lc06_comparison_quantity_course.py",
    ),
    "DISCOURSE_INFORMATION_STRUCTURE": (
        "data/ph2/manifests/lc07_discourse_information_course_v1.json",
        "src/pure_integer_ai/experiments/ph2_authored_discourse_course.py",
        "src/pure_integer_ai/experiments/ph2_authored_discourse_information_course.py",
        "src/pure_integer_ai/cognition/shared/work_memory.py",
        "tests/test_d02_lc07_discourse_information_course.py",
    ),
    "EVALUATOR_RETENTION_RESOURCE": (
        "data/ph2/manifests/lc10_retention_rollback_manifest_v1.json",
        "src/pure_integer_ai/cognition/shared/candidate_runtime.py",
        "src/pure_integer_ai/cognition/shared/memory_batch.py",
        "src/pure_integer_ai/cognition/shared/memory_event_log.py",
        "src/pure_integer_ai/cognition/shared/scoped_persistence.py",
        "src/pure_integer_ai/cognition/shared/situation_state.py",
        "src/pure_integer_ai/experiments/evaluation_isolation.py",
        "src/pure_integer_ai/experiments/ph2_dataset_validation.py",
        "src/pure_integer_ai/experiments/ph2_retention_rollback_catalog.py",
        "src/pure_integer_ai/experiments/ph2_retention_rollback_contract.py",
        "src/pure_integer_ai/storage/memory_recovery.py",
        "src/pure_integer_ai/training/cursor.py",
        "tests/test_d02_lc10_retention_rollback_manifest.py",
        "tests/test_v06_evaluation_isolation.py",
    ),
    "EVENT_TIME_ASPECT": (
        "data/ph2/manifests/lc05_event_time_aspect_course_v1.json",
        "src/pure_integer_ai/cognition/shared/event_time.py",
        "src/pure_integer_ai/experiments/event_time_runtime.py",
        "src/pure_integer_ai/experiments/ph2_authored_event_time_aspect_course.py",
        "src/pure_integer_ai/experiments/ph2_authored_precedes_course.py",
        "tests/test_d02_lc05_event_time_aspect_course.py",
    ),
    "LAYERED_GENERATION": (
        "data/ph2/authored_generation_generalization_seed_v1.jsonl.sample",
        "data/ph2/manifests/gg01_generation_choice_contract_v1.json",
        "data/ph2/manifests/gg01_generation_choice_contract_v2.json",
        "data/ph2/manifests/gg02_generation_choice_outcome_bridge_v1.json",
        "data/ph2/manifests/gg03_generation_generalization_course_v1.json",
        "src/pure_integer_ai/experiments/language_generation_connector.py",
        "src/pure_integer_ai/experiments/ph2_authored_generation_generalization_course.py",
        "src/pure_integer_ai/experiments/ph2_generation_choice_contract.py",
        "src/pure_integer_ai/experiments/ph2_generation_choice_outcome_bridge.py",
        "src/pure_integer_ai/experiments/ph2_generation_generalization_contract.py",
        "src/pure_integer_ai/experiments/ph2_authored_generation_course.py",
        "tests/test_d02_gg01_generation_choice_contract.py",
        "tests/test_d02_gg02_generation_choice_outcome_bridge.py",
        "tests/test_d02_gg03_generation_generalization_course.py",
    ),
    "MEMORY_DYNAMICS": (
        "data/ph2/manifests/md01_memory_dynamics_contract_v1.json",
        "data/ph2/manifests/md02_situation_state_adapter_v1.json",
        "data/ph2/manifests/md03_directional_center_adapter_v1.json",
        "data/ph2/manifests/md04_center_diffusion_probe_plan_v1.json",
        "data/ph2/manifests/md04_center_diffusion_probe_runs_v1.json",
        "data/ph2/manifests/md05_center_diffusion_decision_v1.json",
        "src/pure_integer_ai/cognition/shared/situation_state.py",
        "src/pure_integer_ai/cognition/shared/memory_query.py",
        "src/pure_integer_ai/cognition/shared/reasoning_planner.py",
        "src/pure_integer_ai/experiments/ph2_md02_manifest.py",
        "src/pure_integer_ai/experiments/ph2_md03_center_adapter.py",
        "src/pure_integer_ai/experiments/ph2_md03_manifest.py",
        "src/pure_integer_ai/experiments/ph2_md04_probe_contract.py",
        "src/pure_integer_ai/experiments/ph2_md04_probe_fixture.py",
        "src/pure_integer_ai/experiments/ph2_md04_probe_runtime.py",
        "src/pure_integer_ai/experiments/ph2_md05_probe_evaluator.py",
        "src/pure_integer_ai/experiments/ph2_memory_dynamics_contract.py",
        "tests/test_d02_md01_memory_dynamics_contract.py",
        "tests/test_d02_md02_situation_state_adapter.py",
        "tests/test_d02_md03_directional_center_adapter.py",
        "tests/test_d02_md04_md05_center_diffusion_probe.py",
    ),
    "MORPHOLOGY_WORD_FORM": (
        "data/ph2/manifests/lc02_morphology_course_v1.json",
        "src/pure_integer_ai/experiments/ph2_authored_morphology_course.py",
        "src/pure_integer_ai/experiments/ph2_capability_course_contract.py",
        "src/pure_integer_ai/storage/word_form_index.py",
        "src/pure_integer_ai/experiments/ph2_authored_sense_course.py",
        "tests/test_d02_lc02_morphology_course.py",
    ),
    "MULTIWORD_CONSTRUCTION": (
        "data/ph2/manifests/lc03_construction_course_v1.json",
        "src/pure_integer_ai/cognition/shared/semantic_object.py",
        "src/pure_integer_ai/experiments/ph2_authored_construction_course.py",
        "src/pure_integer_ai/experiments/ph2_authored_primitive_course.py",
        "tests/test_d02_lc03_construction_course.py",
    ),
    "NONLITERAL_CULTURAL": (
        "data/ph2/manifests/lc03_construction_course_v1.json",
        "data/ph2/manifests/lc07_discourse_information_course_v1.json",
        "data/ph2/manifests/lc08_open_set_clarification_course_v1.json",
        "data/ph2/manifests/nl00_nonliteral_scope_probe_manifest_v1.json",
        "src/pure_integer_ai/experiments/ph2_nonliteral_scope_probe_catalog.py",
        "src/pure_integer_ai/experiments/ph2_nonliteral_scope_probe_contract.py",
        "tests/test_d02_nl00_nonliteral_scope_probe.py",
    ),
    "NON_TEXT_MEDIA": ("wall-boundary:W1-W2",),
    "OPEN_SET_CONTINUAL_LEARNING": (
        "data/ph2/manifests/lc08_open_set_clarification_course_v1.json",
        "src/pure_integer_ai/cognition/shared/candidate_runtime.py",
        "src/pure_integer_ai/experiments/ph2_authored_open_set_clarification_course.py",
        "src/pure_integer_ai/experiments/ph2_authored_qa_course.py",
        "tests/test_d02_lc08_open_set_clarification_course.py",
    ),
    "PRAGMATIC_CLARIFICATION_REPAIR": (
        "data/ph2/manifests/lc08_open_set_clarification_course_v1.json",
        "src/pure_integer_ai/experiments/ph2_authored_open_set_clarification_course.py",
        "src/pure_integer_ai/experiments/ph2_authored_qa_course.py",
        "src/pure_integer_ai/experiments/question_answer_runtime.py",
        "tests/test_d02_lc08_open_set_clarification_course.py",
    ),
    "RAW_TEXT_NOISE": (
        "data/ph2/manifests/lc01_lc15_initial_course_v1.json",
        "src/pure_integer_ai/experiments/ph2_authored_text_fidelity_course.py",
        "src/pure_integer_ai/experiments/ph2_dataset_records.py",
        "src/pure_integer_ai/experiments/ph2_mediawiki_multistream_adapter.py",
        "tests/test_d02_lc01_text_fidelity_course.py",
    ),
    "RECURSIVE_PARSE": (
        "data/ph2/manifests/lc04_recursive_parse_course_v1.json",
        "src/pure_integer_ai/cognition/shared/semantic_object.py",
        "src/pure_integer_ai/experiments/ph2_authored_recursive_parse_course.py",
        "src/pure_integer_ai/storage/span.py",
        "tests/test_d02_lc04_recursive_parse_course.py",
    ),
    "REFERENCE_DISCOURSE_REVISION": (
        "data/ph2/manifests/lc07_discourse_information_course_v1.json",
        "src/pure_integer_ai/experiments/ph2_authored_discourse_course.py",
        "src/pure_integer_ai/experiments/ph2_authored_discourse_information_course.py",
        "src/pure_integer_ai/cognition/shared/work_memory.py",
        "tests/test_d02_lc07_discourse_information_course.py",
    ),
    "RELATION_LOGIC_FOUR_STATE": (
        "data/ph2/manifests/ri00_reasoning_mode_probe_manifest_v2.json",
        "src/pure_integer_ai/experiments/ph2_authored_logic_schema.py",
        "src/pure_integer_ai/experiments/ph2_authored_relation_schema.py",
        "src/pure_integer_ai/experiments/ph2_reasoning_mode_probe_catalog.py",
        "src/pure_integer_ai/experiments/ph2_reasoning_mode_probe_contract.py",
        "tests/test_d02_ri00_reasoning_mode_probe.py",
    ),
    "SOURCE_UNCERTAINTY_REALITY": (
        "data/ph2/manifests/lc14_attribution_quotation_course_v1.json",
        "src/pure_integer_ai/experiments/ph2_authored_attribution_quotation_course.py",
        "src/pure_integer_ai/storage/source_record.py",
        "src/pure_integer_ai/storage/source_trust.py",
        "tests/test_d02_lc14_attribution_quotation_course.py",
    ),
    "TRANSFER_AXES": (
        "data/ph2/manifests/lc09_transfer_axis_manifest_v1.json",
        "src/pure_integer_ai/experiments/ph2_dataset_contract.py",
        "src/pure_integer_ai/experiments/ph2_dataset_manifest.py",
        "src/pure_integer_ai/experiments/ph2_transfer_axis_catalog.py",
        "src/pure_integer_ai/experiments/ph2_transfer_axis_contract.py",
        "tests/test_d02_lc09_transfer_axis_manifest.py",
    ),
    "TYPED_LEARNING_OBJECTIVES": (
        "data/ph2/manifests/lc01_lc15_initial_course_v1.json",
        "data/ph2/manifests/lc02_morphology_course_v1.json",
        "data/ph2/manifests/lc03_construction_course_v1.json",
        "data/ph2/manifests/lc04_recursive_parse_course_v1.json",
        "data/ph2/manifests/lc05_event_time_aspect_course_v1.json",
        "data/ph2/manifests/lc06_comparison_quantity_course_v1.json",
        "data/ph2/manifests/lc07_discourse_information_course_v1.json",
        "data/ph2/manifests/lc08_open_set_clarification_course_v1.json",
        "data/ph2/manifests/lc14_attribution_quotation_course_v1.json",
        "data/ph2/manifests/lc15_final_learning_objectives_v1.json",
        "src/pure_integer_ai/cognition/shared/candidate_runtime.py",
        "src/pure_integer_ai/experiments/ph2_dataset_pilot.py",
        "src/pure_integer_ai/experiments/ph2_language_course_contract.py",
        "src/pure_integer_ai/experiments/ph2_learning_objective_coverage.py",
        "tests/test_d02_lc01_text_fidelity_course.py",
        "tests/test_d02_lc03_construction_course.py",
        "tests/test_d02_lc04_recursive_parse_course.py",
        "tests/test_d02_lc05_event_time_aspect_course.py",
        "tests/test_d02_lc06_comparison_quantity_course.py",
        "tests/test_d02_lc07_discourse_information_course.py",
        "tests/test_d02_lc08_open_set_clarification_course.py",
        "tests/test_d02_lc14_attribution_quotation_course.py",
        "tests/test_d02_lc15_final_learning_objectives.py",
    ),
}


def _scope_axes(capability_key: str) -> CanonicalJsonObject:
    if capability_key == "NON_TEXT_MEDIA":
        return CanonicalJsonObject.from_value({
            "code_switch": ["OUT_OF_SCOPE"],
            "dialect": ["OUT_OF_SCOPE"],
            "domain": ["OUT_OF_SCOPE"],
            "era": ["OUT_OF_SCOPE"],
            "genre": ["OUT_OF_SCOPE"],
            "language": ["OUT_OF_SCOPE"],
            "length": ["OUT_OF_SCOPE"],
            "medium": ["NON_TEXT_WALL_BLOCKED"],
            "noise": ["OUT_OF_SCOPE"],
            "register": ["OUT_OF_SCOPE"],
            "script_orthography": ["OUT_OF_SCOPE"],
        })
    return CanonicalJsonObject.from_value({
        "code_switch": ["ABSENT_FIRST_PHASE"],
        "dialect": ["STANDARD_MANDARIN_BASELINE"],
        "domain": ["OPEN_DOMAIN_NOT_EVIDENCED"],
        "era": ["CONTEMPORARY_BASELINE"],
        "genre": ["DECLARATIVE", "DIALOGUE", "EXPLANATORY", "INSTRUCTIONAL"],
        "language": ["zh"],
        "length": ["BOUNDED_DOCUMENT"],
        "medium": ["TEXT"],
        "noise": ["CLEAN", "CONTROLLED_NOISE_REQUIRED"],
        "register": ["NEUTRAL_BASELINE"],
        "script_orthography": ["HAN_SIMPLIFIED", "HAN_TRADITIONAL"],
    })


def _fact_states(capability_key: str) -> CanonicalJsonObject:
    if capability_key == "NON_TEXT_MEDIA":
        return CanonicalJsonObject.from_value({
            dimension: ("WALL_BLOCKED" if dimension != "SCOPE"
                        else "OUT_OF_SCOPE")
            for dimension in FACT_DIMENSIONS
        })
    strong_course = capability_key in {
        "ATTRIBUTION_QUOTATION_PERSPECTIVE",
        "COMPARISON_QUANTITY_MEASURE",
        "DISCOURSE_INFORMATION_STRUCTURE",
        "EVALUATOR_RETENTION_RESOURCE",
        "EVENT_TIME_ASPECT",
        "LAYERED_GENERATION",
        "MORPHOLOGY_WORD_FORM",
        "MULTIWORD_CONSTRUCTION",
        "OPEN_SET_CONTINUAL_LEARNING",
        "PRAGMATIC_CLARIFICATION_REPAIR",
        "RAW_TEXT_NOISE",
        "RECURSIVE_PARSE",
        "REFERENCE_DISCOURSE_REVISION",
        "RELATION_LOGIC_FOUR_STATE",
        "SOURCE_UNCERTAINTY_REALITY",
        "TRANSFER_AXES",
        "TYPED_LEARNING_OBJECTIVES",
    }
    initial_objective_contract = capability_key == "TYPED_LEARNING_OBJECTIVES"
    implementation = _IMPLEMENTATION[capability_key]
    return CanonicalJsonObject.from_value({
        "DATA_ISOLATION": (
            "COURSE_FROZEN"
            if strong_course or initial_objective_contract else "DESIGNED"),
        "DIRECTIONAL_CONSUMPTION": (
            "COURSE_FROZEN"),
        "LEARNING_LOOP": (
            "DESIGNED" if implementation == "SCAFFOLD_ONLY" else "ABSENT"),
        "OBSERVATION_FIDELITY": (
            "COURSE_FROZEN"
            if capability_key == "RAW_TEXT_NOISE" else "ABSENT"),
        "REPRESENTATION": (
            "COURSE_FROZEN"
            if (strong_course or initial_objective_contract
                or capability_key in {
                    "LAYERED_GENERATION", "MEMORY_DYNAMICS"}) else "DESIGNED"),
        "RESOURCE": (
            "COURSE_FROZEN"
            if capability_key in {
                "ATTRIBUTION_QUOTATION_PERSPECTIVE",
                "COMPARISON_QUANTITY_MEASURE", "EVENT_TIME_ASPECT",
                "DISCOURSE_INFORMATION_STRUCTURE",
                "EVALUATOR_RETENTION_RESOURCE", "RECURSIVE_PARSE",
                "OPEN_SET_CONTINUAL_LEARNING",
                "PRAGMATIC_CLARIFICATION_REPAIR",
                "REFERENCE_DISCOURSE_REVISION",
                "TYPED_LEARNING_OBJECTIVES", "MEMORY_DYNAMICS",
                "LAYERED_GENERATION", "TRANSFER_AXES"}
            else "DESIGNED"),
        "RETENTION": (
            "COURSE_FROZEN"
            if capability_key == "EVALUATOR_RETENTION_RESOURCE"
            else "ABSENT"),
        "SCOPE": (
            "COURSE_FROZEN"
            if capability_key in {
                "EVALUATOR_RETENTION_RESOURCE", "LAYERED_GENERATION",
                "NONLITERAL_CULTURAL", "TRANSFER_AXES"} else "DESIGNED"),
        "VERIFIER_CAPABILITY": (
            "COURSE_FROZEN"
            if capability_key in {
                "ATTRIBUTION_QUOTATION_PERSPECTIVE",
                "COMPARISON_QUANTITY_MEASURE", "EVENT_TIME_ASPECT",
                "EVALUATOR_RETENTION_RESOURCE",
                "MORPHOLOGY_WORD_FORM", "MULTIWORD_CONSTRUCTION",
                "RAW_TEXT_NOISE", "RECURSIVE_PARSE",
                "DISCOURSE_INFORMATION_STRUCTURE",
                "OPEN_SET_CONTINUAL_LEARNING",
                "PRAGMATIC_CLARIFICATION_REPAIR",
                "REFERENCE_DISCOURSE_REVISION",
                "TYPED_LEARNING_OBJECTIVES", "MEMORY_DYNAMICS",
                "LAYERED_GENERATION", "TRANSFER_AXES"}
            else "ABSENT"),
    })


def _directional_consumption(capability_key: str) -> CanonicalJsonObject:
    if capability_key == "NON_TEXT_MEDIA":
        return CanonicalJsonObject.from_value({
            direction: {
                "applicability": "N_A",
                "consumer_refs": [],
                "fact_state": "OUT_OF_SCOPE",
                "write_permissions": [],
            }
            for direction in ("GENERATION", "REASONING", "UNDERSTANDING")
        })
    designed = _IMPLEMENTATION[capability_key] == "SCAFFOLD_ONLY"
    consumers = {
        "LAYERED_GENERATION": {
            "GENERATION": [
                "src/pure_integer_ai/experiments/language_generation_connector.py",
                "src/pure_integer_ai/experiments/ph2_generation_choice_contract.py",
                "src/pure_integer_ai/experiments/ph2_generation_choice_outcome_bridge.py"],
            "REASONING": [],
            "UNDERSTANDING": [
                "src/pure_integer_ai/experiments/language_semantic_query.py"],
        },
        "MEMORY_DYNAMICS": {
            "GENERATION": [
                "src/pure_integer_ai/experiments/memory_generation_runtime.py"],
            "REASONING": [
                "src/pure_integer_ai/experiments/attractor_runtime.py"],
            "UNDERSTANDING": [
                "src/pure_integer_ai/experiments/memory_query_runtime.py"],
        },
        "RELATION_LOGIC_FOUR_STATE": {
            "GENERATION": [
                "src/pure_integer_ai/experiments/language_generation_connector.py"],
            "REASONING": [
                "src/pure_integer_ai/experiments/logic_closure_runtime.py"],
            "UNDERSTANDING": [
                "src/pure_integer_ai/experiments/relation_closure_runtime.py"],
        },
        "SOURCE_UNCERTAINTY_REALITY": {
            "GENERATION": [
                "src/pure_integer_ai/experiments/generation_postcheck_course.py"],
            "REASONING": [
                "src/pure_integer_ai/experiments/source_trust_runtime.py"],
            "UNDERSTANDING": [
                "src/pure_integer_ai/experiments/language_semantic_query.py"],
        },
    }.get(capability_key, {})
    result = {}
    for direction in ("GENERATION", "REASONING", "UNDERSTANDING"):
        refs = sorted(consumers.get(direction, []))
        state = (
            "COURSE_FROZEN"
            if (capability_key == "LAYERED_GENERATION"
                and direction == "GENERATION")
            else "DESIGNED" if designed and refs else "ABSENT")
        result[direction] = {
            "applicability": "REQUIRED",
            "consumer_refs": refs,
            "fact_state": state,
            "write_permissions": (
                (["CANDIDATE_ONLY", "NO_HOST_LEARNING_WRITE"]
                 if refs and direction == "GENERATION"
                 else ["NO_HOST_LEARNING_WRITE"] if refs else [])),
        }
    return CanonicalJsonObject.from_value(result)


def _verifier_keys(capability_key: str) -> tuple[str, ...]:
    keys = {
        "D02_RECORD_CONTRACT_V1",
        "LC13_DIRECTIONAL_CONSUMER_VERIFIER_V1",
        "LANGUAGE_RUNTIME_HELD_OUT_VERIFIER_V1",
    }
    if capability_key in {
            "LAYERED_GENERATION", "PRAGMATIC_CLARIFICATION_REPAIR",
            "REFERENCE_DISCOURSE_REVISION"}:
        keys.add("MULTI_LEGAL_SURFACE_VERIFIER_V1")
    if capability_key == "LAYERED_GENERATION":
        keys.add("GG01_GENERATION_CHOICE_CONTRACT_VERIFIER_V1")
        keys.add("GG01_GENERATION_CHOICE_CONTRACT_VERIFIER_V2")
        keys.add("GG02_GENERATION_CHOICE_OUTCOME_BRIDGE_VERIFIER_V1")
    if capability_key in {
            "LAYERED_GENERATION", "PRAGMATIC_CLARIFICATION_REPAIR",
            "REFERENCE_DISCOURSE_REVISION", "SOURCE_UNCERTAINTY_REALITY",
            "TYPED_LEARNING_OBJECTIVES"}:
        keys.add("GG03_GENERATION_GENERALIZATION_COURSE_VERIFIER_V1")
    if capability_key in {
            "EVALUATOR_RETENTION_RESOURCE", "OPEN_SET_CONTINUAL_LEARNING",
            "TRANSFER_AXES"}:
        keys.add("RETENTION_ROLLBACK_VERIFIER_V1")
    if capability_key == "TRANSFER_AXES":
        keys.add("LC09_TRANSFER_AXIS_VERIFIER_V1")
    if capability_key in {
            "RAW_TEXT_NOISE", "SOURCE_UNCERTAINTY_REALITY"}:
        keys.add("SOURCE_SNAPSHOT_INTEGRITY_V1")
    if capability_key == "MORPHOLOGY_WORD_FORM":
        keys.add("LC02_MORPHOLOGY_COURSE_VERIFIER_V1")
    if capability_key == "MULTIWORD_CONSTRUCTION":
        keys.add("LC03_CONSTRUCTION_COURSE_VERIFIER_V1")
    if capability_key == "RECURSIVE_PARSE":
        keys.add("LC04_RECURSIVE_PARSE_COURSE_VERIFIER_V1")
    if capability_key == "EVENT_TIME_ASPECT":
        keys.add("LC05_EVENT_TIME_ASPECT_COURSE_VERIFIER_V1")
    if capability_key == "COMPARISON_QUANTITY_MEASURE":
        keys.add("LC06_COMPARISON_QUANTITY_COURSE_VERIFIER_V1")
    if capability_key in {
            "DISCOURSE_INFORMATION_STRUCTURE",
            "REFERENCE_DISCOURSE_REVISION"}:
        keys.add("LC07_DISCOURSE_INFORMATION_COURSE_VERIFIER_V1")
    if capability_key in {
            "OPEN_SET_CONTINUAL_LEARNING",
            "PRAGMATIC_CLARIFICATION_REPAIR"}:
        keys.add("LC08_OPEN_SET_CLARIFICATION_COURSE_VERIFIER_V1")
    if capability_key in {
            "ATTRIBUTION_QUOTATION_PERSPECTIVE",
            "SOURCE_UNCERTAINTY_REALITY"}:
        keys.add("LC14_ATTRIBUTION_QUOTATION_COURSE_VERIFIER_V1")
    if capability_key == "TYPED_LEARNING_OBJECTIVES":
        keys.add("LC15_FINAL_LEARNING_OBJECTIVE_VERIFIER_V1")
    if capability_key == "MEMORY_DYNAMICS":
        keys.add("MD01_MEMORY_DYNAMICS_CONTRACT_VERIFIER_V1")
        keys.add("MD02_SITUATION_STATE_ADAPTER_VERIFIER_V1")
        keys.add("MD03_DIRECTIONAL_CENTER_ADAPTER_VERIFIER_V1")
        keys.add("MD05_CENTER_DIFFUSION_PROBE_VERIFIER_V1")
    if capability_key == "RELATION_LOGIC_FOUR_STATE":
        keys.add("RI00_REASONING_MODE_PROBE_VERIFIER_V1")
    if capability_key == "NONLITERAL_CULTURAL":
        keys.add("NL00_NONLITERAL_SCOPE_PROBE_VERIFIER_V1")
    return tuple(sorted(keys))


def _resource_contracts(capability_key: str) -> tuple[str, ...]:
    """返回能力当前课程已冻结的资源 ceiling；未冻结维继续显式 ABSENT。"""
    if capability_key == "EVALUATOR_RETENTION_RESOURCE":
        return (
            "MAX_EVIDENCE_FILES=8",
            "MAX_FIXTURES=3",
            "MAX_LOGIC_STEPS=ABSENT",
            "MAX_OUTCOME_CLASSES=5",
            "MAX_OUTPUT_UNITS=ABSENT",
            "MAX_PAGE_SEGMENTS=ABSENT",
            "MAX_PHASES=10",
            "MAX_QUERIES=ABSENT",
            "MAX_RECOMPUTE_OBJECTS=ABSENT",
            "MAX_RECURSION_DEPTH=ABSENT",
            "MAX_RUNTIME_BINDINGS=7",
        )
    if capability_key == "TYPED_LEARNING_OBJECTIVES":
        return (
            "BASELINE_ABLATIONS=4",
            "CAPABILITY_BINDINGS=12",
            "COURSE_SOURCES=9",
            "LEARNING_OBJECTIVES=11",
            "MAX_LOGIC_STEPS=ABSENT",
            "MAX_OUTPUT_UNITS=ABSENT",
            "MAX_PAGE_SEGMENTS=ABSENT",
            "MAX_QUERIES=ABSENT",
            "MAX_RECOMPUTE_OBJECTS=ABSENT",
            "MAX_RECURSION_DEPTH=ABSENT",
        )
    if capability_key == "ATTRIBUTION_QUOTATION_PERSPECTIVE":
        return (
            "MAX_ATTRIBUTIONS=3",
            "MAX_DEPENDENCY_EDGES=3",
            "MAX_LOGIC_STEPS=ABSENT",
            "MAX_NESTING_DEPTH=2",
            "MAX_OUTPUT_UNITS=160",
            "MAX_PAGE_SEGMENTS=ABSENT",
            "MAX_PROPOSITIONS=2",
            "MAX_QUERIES=ABSENT",
            "MAX_QUOTE_SPANS=2",
            "MAX_RECOMPUTE_OBJECTS=ABSENT",
            "MAX_RECURSION_DEPTH=ABSENT",
        )
    if capability_key == "RECURSIVE_PARSE":
        return (
            "MAX_CANDIDATES=2",
            "MAX_LOGIC_STEPS=ABSENT",
            "MAX_OUTPUT_UNITS=64",
            "MAX_PAGE_SEGMENTS=ABSENT",
            "MAX_QUERIES=ABSENT",
            "MAX_RECOMPUTE_OBJECTS=12",
            "MAX_RECURSION_DEPTH=4",
        )
    if capability_key == "EVENT_TIME_ASPECT":
        return (
            "MAX_CANDIDATES=2",
            "MAX_LOGIC_STEPS=ABSENT",
            "MAX_OUTPUT_UNITS=64",
            "MAX_PAGE_SEGMENTS=ABSENT",
            "MAX_QUERIES=ABSENT",
            "MAX_RECOMPUTE_OBJECTS=8",
            "MAX_RECURSION_DEPTH=ABSENT",
        )
    if capability_key == "COMPARISON_QUANTITY_MEASURE":
        return (
            "MAX_CANDIDATES=2",
            "MAX_COMPARISONS=2",
            "MAX_LOGIC_STEPS=ABSENT",
            "MAX_OBJECTS=2",
            "MAX_OUTPUT_UNITS=80",
            "MAX_PAGE_SEGMENTS=ABSENT",
            "MAX_QUANTIFIERS=2",
            "MAX_QUANTITIES=2",
            "MAX_QUERIES=ABSENT",
            "MAX_RECURSION_DEPTH=ABSENT",
            "MAX_SCALES=1",
            "MAX_STANDARDS=2",
            "MAX_UNITS=1",
        )
    if capability_key in {
            "DISCOURSE_INFORMATION_STRUCTURE",
            "REFERENCE_DISCOURSE_REVISION"}:
        return (
            "MAX_CANDIDATES=2",
            "MAX_DEPENDENCY_EDGES=4",
            "MAX_INFORMATION_STATES=1",
            "MAX_LOGIC_STEPS=ABSENT",
            "MAX_OUTPUT_UNITS=160",
            "MAX_PAGE_SEGMENTS=ABSENT",
            "MAX_PRESUPPOSITIONS=1",
            "MAX_PROPOSITIONS=2",
            "MAX_QUD=1",
            "MAX_QUERIES=ABSENT",
            "MAX_RECOMPUTE_OBJECTS=ABSENT",
            "MAX_RECURSION_DEPTH=ABSENT",
            "MAX_RELATIONS=2",
        )
    if capability_key in {
            "OPEN_SET_CONTINUAL_LEARNING",
            "PRAGMATIC_CLARIFICATION_REPAIR"}:
        return (
            "MAX_CANDIDATE_BRANCHES=3",
            "MAX_DEPENDENCY_EDGES=6",
            "MAX_EVIDENCE_REQUESTS=1",
            "MAX_LOGIC_STEPS=ABSENT",
            "MAX_OBLIGATIONS=1",
            "MAX_OUTPUT_UNITS=160",
            "MAX_PAGE_SEGMENTS=ABSENT",
            "MAX_QUERIES=ABSENT",
            "MAX_QUESTION_COST=1",
            "MAX_RECOMPUTE_OBJECTS=ABSENT",
            "MAX_RECURSION_DEPTH=ABSENT",
        )
    if capability_key == "MEMORY_DYNAMICS":
        return (
            "MAX_AGENDA_ENTRIES=4",
            "MAX_CANDIDATES=8",
            "MAX_COLD_BYTES=100000",
            "MAX_CONSUMPTIONS=4",
            "MAX_HOT_OBJECTS=4",
            "MAX_LOGIC_STEPS=64",
            "MAX_OUTPUT_UNITS=ABSENT",
            "MAX_PAGE_SEGMENTS=2",
            "MAX_QUERIES=1",
            "MAX_RECOMPUTE_OBJECTS=1",
            "MAX_RECURSION_DEPTH=ABSENT",
            "MAX_SCANNED_OBJECTS=8",
        )
    if capability_key == "TRANSFER_AXES":
        return (
            "MAX_BLOCKED_SOURCES=1",
            "MAX_FORMAL_PACKS=16",
            "MAX_LOGIC_STEPS=ABSENT",
            "MAX_OUTPUT_UNITS=ABSENT",
            "MAX_PAGE_SEGMENTS=ABSENT",
            "MAX_QUERIES=ABSENT",
            "MAX_RECOMPUTE_OBJECTS=ABSENT",
            "MAX_RECURSION_DEPTH=ABSENT",
            "MAX_SPLIT_PROBES=3",
            "MAX_TRANSFER_AXES=10",
        )
    return (
        "MAX_CANDIDATES=ABSENT",
        "MAX_LOGIC_STEPS=ABSENT",
        "MAX_OUTPUT_UNITS=ABSENT",
        "MAX_PAGE_SEGMENTS=ABSENT",
        "MAX_QUERIES=ABSENT",
        "MAX_RECOMPUTE_OBJECTS=ABSENT",
        "MAX_RECURSION_DEPTH=ABSENT",
    )


def build_capability_ledger() -> LanguageCapabilityCoverageLedger:
    """从当前 D-02 事实构建 LC-00，不补代码掩盖 ABSENT。"""
    entries = []
    for capability_key in CAPABILITY_KEYS:
        task_keys = {*_TASKS[capability_key], "LC-13"}
        representations = {
            *_REPRESENTATIONS[capability_key],
            "LC13_DIRECTIONAL_CONSUMER_MAP_V1",
        }
        evidence_refs = {
            *_EVIDENCE[capability_key],
            "data/ph2/manifests/lc13_directional_consumer_manifest_v1.json",
            "src/pure_integer_ai/experiments/ph2_directional_consumer_catalog.py",
            "src/pure_integer_ai/experiments/ph2_directional_consumer_contract.py",
            "tests/test_d02_lc13_directional_consumer_manifest.py",
        }
        entries.append(LanguageCapabilityCoverageEntry(
            capability_key,
            tuple(sorted(task_keys)),
            _SCOPES[capability_key],
            _IMPLEMENTATION[capability_key],
            _scope_axes(capability_key),
            (
                "NORMALIZATION_SEGMENTATION_TOKENIZATION_RECEIPT_REQUIRED",
                "RAW_OBSERVATION_APPEND_ONLY",
            ),
            tuple(sorted(representations)),
            (
                "CANDIDATE_PROPOSE",
                "EVIDENCE_DEDUP_REFUTE",
                "SPLIT_SUPERSEDE_ARCHIVE_REPARSE",
            ),
            _directional_consumption(capability_key),
            (
                "ADVERSARIAL_WALL_PHYSICAL_SPLIT",
                "EVALUATOR_OWNER_ISOLATION",
                "SOURCE_CONTENT_TEMPLATE_SHAPE_COMBINATION_CLUSTER",
                "TRAIN_DEV_HELD_OUT_SPLIT",
            ),
            _verifier_keys(capability_key),
            (
                "A_TO_B_REVERIFY_A",
                "DUMP_RESUME",
                "ROLLBACK_SCOPE_CONTRACTION",
                "SOURCE_WITHDRAWAL_LOCAL_INVALIDATION",
            ),
            _resource_contracts(capability_key),
            _fact_states(capability_key),
            tuple(sorted(evidence_refs)),
        ))
    return LanguageCapabilityCoverageLedger(
        1,
        "LC-00-language-capability-frontier-public-clean-ri00-v2-v19",
        "第一阶段中文文本 D-03 前能力前沿；课程或设施存在不等于 runtime 已学会",
        tuple(entries),
    )


def _verifier(
        key: str,
        *,
        state: str,
        capabilities: tuple[str, ...],
        dimensions: tuple[str, ...],
        prerequisites: tuple[str, ...],
        blind_spots: tuple[str, ...],
        owner: str,
        sources: tuple[str, ...],
        evidence: tuple[str, ...],
        ne: tuple[str, ...],
        ) -> VerifierCapabilityRecord:
    return VerifierCapabilityRecord(
        key,
        "v1",
        state,
        tuple(sorted(capabilities)),
        tuple(sorted(dimensions)),
        tuple(sorted(prerequisites)),
        tuple(sorted(blind_spots)),
        owner,
        tuple(sorted(sources)),
        tuple(sorted(evidence)),
        tuple(sorted(ne)),
        (),
        0,
    )


def build_verifier_registry() -> VerifierCapabilityRegistry:
    """登记当前 verifier 能力和盲区，全部禁止发能力 runtime PASS。"""
    all_caps = CAPABILITY_KEYS
    records = (
        _verifier(
            "D02_COURSE_COMPILER_V1", state="RUNTIME_EVIDENCED",
            capabilities=all_caps,
            dimensions=("DETERMINISTIC_COMPILE", "RECORD_COUNT", "SCHEMA"),
            prerequisites=("COURSE_SEED", "D02_RECORD_CONTRACT"),
            blind_spots=("LANGUAGE_GENERALIZATION", "LEARNED_STATE", "SEMANTIC_TRUTH"),
            owner="COURSE_COMPILER_OWNER", sources=("AUTHORED_COURSE",),
            evidence=("tests/test_d02f_dataset_pilot.py",),
            ne=("CAPABILITY_PASS_REQUESTED", "STUDENT_BEHAVIOR_REQUIRED")),
        _verifier(
            "D02_EVALUATOR_LABEL_INTEGRITY_V1", state="RUNTIME_EVIDENCED",
            capabilities=all_caps,
            dimensions=("EVALUATOR_OWNER", "LABEL_IDENTITY", "SPLIT_ISOLATION"),
            prerequisites=("EVALUATOR_LABEL_RECORD",),
            blind_spots=("LABEL_CORRECTNESS", "RUNTIME_LEARNING", "STUDENT_OUTPUT"),
            owner="EVALUATOR_OWNER", sources=("EVALUATOR_LABEL",),
            evidence=("tests/test_d02_dataset_contract.py",),
            ne=("EXPECTED_ONLY_PASS", "SEMANTIC_ABILITY_REQUESTED")),
        _verifier(
            "D02_PILOT_ISOLATION_V1", state="RUNTIME_EVIDENCED",
            capabilities=all_caps,
            dimensions=("FRESH_RESUME", "HOST_ZERO_WRITE", "WORKER_EQUIVALENCE"),
            prerequisites=("D02_PILOT_ARTIFACT",),
            blind_spots=("FORMAL_TRAINING", "GENERALIZATION", "MASTERED"),
            owner="PILOT_EVALUATOR_OWNER", sources=("PILOT_EVENT_LOG",),
            evidence=("tests/test_d02f_dataset_pilot.py",),
            ne=("LANGUAGE_CAPABILITY_PASS_REQUESTED",)),
        _verifier(
            "D02_RECORD_CONTRACT_V1", state="RUNTIME_EVIDENCED",
            capabilities=all_caps,
            dimensions=(
                "FUTURE_STAGE_LEAK", "LICENSE", "OWNER", "SPLIT",
                "STABLE_KEY", "SUPERSEDE_DAG"),
            prerequisites=("D02_RECORD_OR_MANIFEST",),
            blind_spots=("LANGUAGE_MEANING", "LEARNED_STATE", "WORLD_TRUTH"),
            owner="DATASET_CONTRACT_OWNER", sources=("RECORD_BYTES",),
            evidence=("tests/test_d02_dataset_contract.py",),
            ne=("NONEMPTY_OUTPUT_PASS", "SEMANTIC_ABILITY_REQUESTED")),
        _verifier(
            "LANGUAGE_RUNTIME_HELD_OUT_VERIFIER_V1", state="ABSENT",
            capabilities=all_caps,
            dimensions=("UNDERSTANDING_REASONING_GENERATION_HELD_OUT",),
            prerequisites=("INDEPENDENT_EXPECTATION_OWNER", "REAL_RUNTIME_CONSUMER"),
            blind_spots=("UNIMPLEMENTED",), owner="UNASSIGNED",
            sources=("ABSENT",), evidence=("design-task:LC-11",),
            ne=("CURRENT_BASELINE", "NO_RUNTIME_EVIDENCE")),
        _verifier(
            "LC01_TEXT_FIDELITY_VERIFIER_V1", state="RUNTIME_EVIDENCED",
            capabilities=("RAW_TEXT_NOISE", "TYPED_LEARNING_OBJECTIVES"),
            dimensions=(
                "CANDIDATE_LATTICE", "GENERATION_SURFACE_FIDELITY",
                "IRREVERSIBLE_LOSS_DISCLOSURE", "LEARNING_OBJECTIVE_BINDING",
                "NORMALIZATION_RECEIPT", "RAW_OBSERVATION_PRESERVATION",
                "RETENTION_REVERIFY"),
            prerequisites=(
                "LC01_COURSE_MANIFEST", "RAW_OBSERVATION",
                "READ_ONLY_EVALUATOR_LABEL"),
            blind_spots=(
                "LANGUAGE_GENERALIZATION", "LEARNED_STATE",
                "SEMANTIC_CORRECTNESS"),
            owner="LC01_EVALUATOR_OWNER",
            sources=("COURSE_MANIFEST", "EVALUATOR_LABEL", "OBSERVATION"),
            evidence=(
                "data/ph2/manifests/lc01_lc15_initial_course_v1.json",
                "tests/test_d02_lc01_text_fidelity_course.py"),
            ne=(
                "CAPABILITY_LEARNED_REQUESTED", "NO_EVALUATOR_LABEL",
                "RUNTIME_GENERALIZATION_REQUESTED",
                "SEMANTIC_CORRECTNESS_REQUESTED")),
        _verifier(
            "LC02_MORPHOLOGY_COURSE_VERIFIER_V1", state="RUNTIME_EVIDENCED",
            capabilities=("MORPHOLOGY_WORD_FORM", "TYPED_LEARNING_OBJECTIVES"),
            dimensions=(
                "AMBIGUOUS_SEGMENTATION", "DICTIONARY_REPLAY_REJECT",
                "EXCEPTION_SCOPE", "HELD_OUT_STEM_CONSTRUCTION",
                "LANGUAGE_SCOPE", "MORPHOLOGY_RELATION_INTEGRITY",
                "RETENTION_REVERIFY", "REVERSE_GENERATION"),
            prerequisites=(
                "LC02_COURSE_MANIFEST", "MORPHOLOGY_CANDIDATE",
                "READ_ONLY_EVALUATOR_LABEL"),
            blind_spots=(
                "LEARNED_STATE", "PRODUCTIVE_RUNTIME_GENERALIZATION",
                "SEMANTIC_TRUTH"),
            owner="LC02_EVALUATOR_OWNER",
            sources=("COURSE_MANIFEST", "EVALUATOR_LABEL", "OBSERVATION"),
            evidence=(
                "data/ph2/manifests/lc02_morphology_course_v1.json",
                "tests/test_d02_lc02_morphology_course.py"),
            ne=(
                "CAPABILITY_LEARNED_REQUESTED",
                "GENERATION_RESULT_NOT_EXECUTED", "NO_EVALUATOR_LABEL",
                "OUT_OF_LANGUAGE_SCOPE", "SEMANTIC_TRUTH_REQUESTED")),
        _verifier(
            "LC03_CONSTRUCTION_COURSE_VERIFIER_V1", state="RUNTIME_EVIDENCED",
            capabilities=("MULTIWORD_CONSTRUCTION", "TYPED_LEARNING_OBJECTIVES"),
            dimensions=(
                "ANTI_LITERAL_BASELINE", "CONSTRUCTION_OBJECT_ABLATION",
                "DISCONTINUOUS_SPAN", "EVENT_CORE_MAPPING",
                "FIXED_VARIABLE_SLOT", "HELD_OUT_FILLER_CONSTRUCTION",
                "LEXICALIZATION_IDENTITY", "REGISTER_SCOPE",
                "RETENTION_REVERIFY", "REVERSE_GENERATION",
                "SAME_PROPOSITION_DIFFERENT_CONSTRUCTION",
                "SAME_SURFACE_DIFFERENT_CONSTRUCTION"),
            prerequisites=(
                "CONSTRUCTION_CANDIDATE", "LC03_COURSE_MANIFEST",
                "READ_ONLY_EVALUATOR_LABEL"),
            blind_spots=(
                "LEARNED_STATE", "PRODUCTIVE_RUNTIME_GENERALIZATION",
                "SEMANTIC_TRUTH"),
            owner="LC03_EVALUATOR_OWNER",
            sources=("COURSE_MANIFEST", "EVALUATOR_LABEL", "OBSERVATION"),
            evidence=(
                "data/ph2/manifests/lc03_construction_course_v1.json",
                "tests/test_d02_lc03_construction_course.py"),
            ne=(
                "CAPABILITY_LEARNED_REQUESTED",
                "CONSTRUCTION_OBJECT_MISSING",
                "GENERATION_RESULT_NOT_EXECUTED", "NO_EVALUATOR_LABEL",
                "OUT_OF_REGISTER_SCOPE", "SEMANTIC_TRUTH_REQUESTED")),
        _verifier(
            "LC04_RECURSIVE_PARSE_COURSE_VERIFIER_V1",
            state="RUNTIME_EVIDENCED",
            capabilities=("RECURSIVE_PARSE", "TYPED_LEARNING_OBJECTIVES"),
            dimensions=(
                "AMBIGUOUS_PARSE_COMPETITION", "COORDINATION_STRUCTURE",
                "DISCONTINUOUS_DEPENDENCY", "HELD_OUT_FILLER_PARSE_DEPTH",
                "LOCAL_REPARSE_SUPERSEDE", "NESTED_DEPTH",
                "NULL_OPTIONAL_CONSTITUENT", "PRESELECTED_TREE_REJECT",
                "REPEATED_TOKEN_IDENTITY", "RETENTION_REVERIFY",
                "REVERSE_LINEARIZATION", "ROLE_SCOPE_PRESERVATION"),
            prerequisites=(
                "LC04_COURSE_MANIFEST", "READ_ONLY_EVALUATOR_LABEL",
                "RECURSIVE_PARSE_CANDIDATE"),
            blind_spots=(
                "LEARNED_STATE", "PRODUCTIVE_RUNTIME_GENERALIZATION",
                "SEMANTIC_TRUTH"),
            owner="LC04_EVALUATOR_OWNER",
            sources=("COURSE_MANIFEST", "EVALUATOR_LABEL", "OBSERVATION"),
            evidence=(
                "data/ph2/manifests/lc04_recursive_parse_course_v1.json",
                "tests/test_d02_lc04_recursive_parse_course.py"),
            ne=(
                "CAPABILITY_LEARNED_REQUESTED",
                "GENERATION_RESULT_NOT_EXECUTED", "NO_EVALUATOR_LABEL",
                "PARSE_CANDIDATE_MISSING", "RECURSION_BUDGET_EXCEEDED",
                "SEMANTIC_TRUTH_REQUESTED")),
        _verifier(
            "LC05_EVENT_TIME_ASPECT_COURSE_VERIFIER_V1",
            state="RUNTIME_EVIDENCED",
            capabilities=("EVENT_TIME_ASPECT", "TYPED_LEARNING_OBJECTIVES"),
            dimensions=(
                "COMPLETED_ASPECT", "DURATIVE_INTERVAL", "EVENT_IDENTITY",
                "HABITUAL_ITERATIVE_ASPECT", "IMPLICIT_NOW_REJECT",
                "LOCAL_TIME_REVISION", "NARRATIVE_ORDER", "RETENTION_REVERIFY",
                "REVERSE_GENERATION_ANCHOR_ASPECT",
                "SAME_SURFACE_DIFFERENT_ANCHOR", "STATE_IDENTITY",
                "SURFACE_ORDER_REJECT", "TIME_ANCHOR_SCOPE", "TIME_UNKNOWN"),
            prerequisites=(
                "EVENT_TIME_ASPECT_CANDIDATE", "LC05_COURSE_MANIFEST",
                "READ_ONLY_EVALUATOR_LABEL"),
            blind_spots=(
                "LEARNED_STATE", "PRODUCTIVE_RUNTIME_GENERALIZATION",
                "REALITY_TIME_TRUTH"),
            owner="LC05_EVALUATOR_OWNER",
            sources=("COURSE_MANIFEST", "EVALUATOR_LABEL", "OBSERVATION"),
            evidence=(
                "data/ph2/manifests/lc05_event_time_aspect_course_v1.json",
                "tests/test_d02_lc05_event_time_aspect_course.py"),
            ne=(
                "ANCHOR_OR_ASPECT_CANDIDATE_MISSING",
                "CAPABILITY_LEARNED_REQUESTED",
                "GENERATION_RESULT_NOT_EXECUTED", "NO_EVALUATOR_LABEL",
                "OUT_OF_CONTEXT_SCOPE", "SEMANTIC_TRUTH_REQUESTED")),
        _verifier(
            "LC06_COMPARISON_QUANTITY_COURSE_VERIFIER_V1",
            state="RUNTIME_EVIDENCED",
            capabilities=(
                "COMPARISON_QUANTITY_MEASURE", "TYPED_LEARNING_OBJECTIVES"),
            dimensions=(
                "AMBIGUOUS_STANDARD_COMPETITION",
                "APPROXIMATE_EXACT_DISTINCTION", "BARE_PROPERTY_REJECT",
                "COMPARISON_DIRECTION_STANDARD", "DEGREE_SCALE_THRESHOLD",
                "LOCAL_QUANTITY_REVISION", "MEASURE_UNIT_DIMENSION",
                "QUANTIFIER_SCOPE", "QUANTITY_COUNT", "QUANTITY_UNKNOWN",
                "RANGE_BOUNDARY", "RETENTION_REVERIFY",
                "REVERSE_GENERATION_COMPARISON_MEASURE",
                "UNIT_ERASURE_REJECT"),
            prerequisites=(
                "COMPARISON_QUANTITY_CANDIDATE", "LC06_COURSE_MANIFEST",
                "READ_ONLY_EVALUATOR_LABEL"),
            blind_spots=(
                "LEARNED_STATE", "PRODUCTIVE_RUNTIME_GENERALIZATION",
                "SEMANTIC_TRUTH"),
            owner="LC06_EVALUATOR_OWNER",
            sources=("COURSE_MANIFEST", "EVALUATOR_LABEL", "OBSERVATION"),
            evidence=(
                "data/ph2/manifests/lc06_comparison_quantity_course_v1.json",
                "tests/test_d02_lc06_comparison_quantity_course.py"),
            ne=(
                "CAPABILITY_LEARNED_REQUESTED",
                "GENERATION_RESULT_NOT_EXECUTED", "NO_EVALUATOR_LABEL",
                "OUT_OF_CONTEXT_SCOPE",
                "SCALE_STANDARD_UNIT_OR_QUANTITY_MISSING",
                "SEMANTIC_TRUTH_REQUESTED")),
        _verifier(
            "LC07_DISCOURSE_INFORMATION_COURSE_VERIFIER_V1",
            state="RUNTIME_EVIDENCED",
            capabilities=(
                "DISCOURSE_INFORMATION_STRUCTURE",
                "REFERENCE_DISCOURSE_REVISION",
                "TYPED_LEARNING_OBJECTIVES"),
            dimensions=(
                "AMBIGUOUS_RELATION_COMPETITION", "CAUSE_RELATION",
                "CONCESSION_RELATION", "CONTRAST_RELATION",
                "DISCOURSE_UNKNOWN", "ELABORATION_RELATION",
                "GIVEN_NEW_STATUS", "LOCAL_RELATION_REVISION",
                "NO_CONNECTIVE_REJECT", "PRESUPPOSITION_CANCELLATION",
                "PRESUPPOSITION_PROJECTION", "QUD_STATE",
                "RETENTION_REVERIFY",
                "REVERSE_GENERATION_ORDER_EXPLICITNESS",
                "TOPIC_FOCUS_MINIMAL_CONTRAST",
                "WRONG_CONNECTIVE_REJECT"),
            prerequisites=(
                "DISCOURSE_INFORMATION_CANDIDATE",
                "LC07_COURSE_MANIFEST", "READ_ONLY_EVALUATOR_LABEL"),
            blind_spots=(
                "LEARNED_STATE", "PRODUCTIVE_RUNTIME_GENERALIZATION",
                "SEMANTIC_TRUTH"),
            owner="LC07_EVALUATOR_OWNER",
            sources=("COURSE_MANIFEST", "EVALUATOR_LABEL", "OBSERVATION"),
            evidence=(
                "data/ph2/manifests/lc07_discourse_information_course_v1.json",
                "tests/test_d02_lc07_discourse_information_course.py"),
            ne=(
                "CAPABILITY_LEARNED_REQUESTED",
                "GENERATION_RESULT_NOT_EXECUTED", "NO_EVALUATOR_LABEL",
                "OUT_OF_DOCUMENT_SCOPE",
                "SEMANTIC_TRUTH_REQUESTED")),
        _verifier(
            "LC08_OPEN_SET_CLARIFICATION_COURSE_VERIFIER_V1",
            state="RUNTIME_EVIDENCED",
            capabilities=(
                "OPEN_SET_CONTINUAL_LEARNING",
                "PRAGMATIC_CLARIFICATION_REPAIR",
                "TYPED_LEARNING_OBJECTIVES"),
            dimensions=(
                "ACCESS_BLOCKED_DISTINCT", "ACTIVE_EVIDENCE_REQUEST",
                "AMBIGUOUS_BRANCH_PRESERVATION",
                "BUDGET_BLOCKED_DISTINCT",
                "CLARIFICATION_ANSWER_LOCAL_UPDATE",
                "CLARIFICATION_REVISION_SUPERSEDE",
                "INSUFFICIENT_GUESS_REJECT",
                "KNOWN_SUFFICIENT_NO_QUESTION",
                "MINIMAL_BRANCH_CLARIFICATION",
                "NEW_CONSTRUCTION_DETECTION", "NEW_SENSE_DETECTION",
                "NEW_USAGE_DETECTION", "NEW_WORD_DETECTION",
                "OPEN_SET_UNKNOWN", "OVERQUESTION_REJECT",
                "RETENTION_REVERIFY",
                "REVERSE_GENERATION_MINIMAL_QUESTION"),
            prerequisites=(
                "LC08_COURSE_MANIFEST",
                "OPEN_SET_CLARIFICATION_CANDIDATE",
                "READ_ONLY_EVALUATOR_LABEL"),
            blind_spots=(
                "LEARNED_STATE", "NOVELTY_RUNTIME_NOT_IMPLEMENTED",
                "PRODUCTIVE_RUNTIME_GENERALIZATION", "SEMANTIC_TRUTH"),
            owner="LC08_EVALUATOR_OWNER",
            sources=("COURSE_MANIFEST", "EVALUATOR_LABEL", "OBSERVATION"),
            evidence=(
                "data/ph2/manifests/lc08_open_set_clarification_course_v1.json",
                "tests/test_d02_lc08_open_set_clarification_course.py"),
            ne=(
                "CAPABILITY_LEARNED_REQUESTED",
                "GENERATION_RESULT_NOT_EXECUTED", "NO_EVALUATOR_LABEL",
                "NOVELTY_RUNTIME_NOT_EXECUTED",
                "OUT_OF_DOCUMENT_SCOPE", "SEMANTIC_TRUTH_REQUESTED")),
        _verifier(
            "LC14_ATTRIBUTION_QUOTATION_COURSE_VERIFIER_V1",
            state="RUNTIME_EVIDENCED",
            capabilities=(
                "ATTRIBUTION_QUOTATION_PERSPECTIVE",
                "SOURCE_UNCERTAINTY_REALITY",
                "TYPED_LEARNING_OBJECTIVES"),
            dimensions=(
                "AMBIGUOUS_SCOPE_COMPETITION",
                "ATTRIBUTION_LOCAL_REVISION", "BELIEF_HOLDER_SCOPE",
                "CLAIM_HOLDER_SCOPE", "DIRECT_QUOTE_BOUNDARY_FIDELITY",
                "HYPOTHESIS_UNCERTAINTY_SCOPE",
                "LATER_DENIAL_NO_CURRENT_PROJECTION",
                "NESTED_HOLDER_SCOPE", "PARAPHRASE_VERSION_LINK",
                "PRONOUN_TRANSFER_HOLDER_PRESERVATION",
                "QUOTE_BOUNDARY_SHORTCUT_REJECT",
                "REPORTED_AS_FACT_REJECT", "RETENTION_REVERIFY",
                "REVERSE_GENERATION_ATTRIBUTION_UNCERTAINTY",
                "SOURCE_CONFLICT_NO_CURRENT_PROJECTION",
                "TENSE_TRANSFER_ANCHOR_PRESERVATION",
                "UNKNOWN_SCOPE_NO_GUESS"),
            prerequisites=(
                "ATTRIBUTION_QUOTATION_CANDIDATE",
                "LC14_COURSE_MANIFEST", "READ_ONLY_EVALUATOR_LABEL"),
            blind_spots=(
                "CURRENT_PROJECTION_TRUTH", "LEARNED_STATE",
                "PRODUCTIVE_RUNTIME_GENERALIZATION",
                "RUNTIME_SCOPE_CONSUMER_NOT_EXECUTED", "SEMANTIC_TRUTH"),
            owner="LC14_EVALUATOR_OWNER",
            sources=("COURSE_MANIFEST", "EVALUATOR_LABEL", "OBSERVATION"),
            evidence=(
                "data/ph2/manifests/lc14_attribution_quotation_course_v1.json",
                "tests/test_d02_lc14_attribution_quotation_course.py"),
            ne=(
                "CAPABILITY_LEARNED_REQUESTED",
                "CURRENT_PROJECTION_TRUTH_REQUESTED",
                "GENERATION_RESULT_NOT_EXECUTED", "NO_EVALUATOR_LABEL",
                "OUT_OF_DOCUMENT_SCOPE",
                "RUNTIME_SCOPE_CONSUMER_NOT_EXECUTED")),
        _verifier(
            "LC15_FINAL_LEARNING_OBJECTIVE_VERIFIER_V1",
            state="RUNTIME_EVIDENCED",
            capabilities=(
                "ATTRIBUTION_QUOTATION_PERSPECTIVE",
                "COMPARISON_QUANTITY_MEASURE",
                "DISCOURSE_INFORMATION_STRUCTURE", "EVENT_TIME_ASPECT",
                "MORPHOLOGY_WORD_FORM", "MULTIWORD_CONSTRUCTION",
                "OPEN_SET_CONTINUAL_LEARNING",
                "PRAGMATIC_CLARIFICATION_REPAIR", "RAW_TEXT_NOISE",
                "RECURSIVE_PARSE", "REFERENCE_DISCOURSE_REVISION",
                "SOURCE_UNCERTAINTY_REALITY",
                "TYPED_LEARNING_OBJECTIVES"),
            dimensions=(
                "BASELINE_ABLATION_PRE_REGISTRATION",
                "CANDIDATE_LIFECYCLE_BINDING",
                "CAPABILITY_OBJECTIVE_BINDING",
                "COURSE_MANIFEST_HASH_BINDING",
                "EVIDENCE_OWNER_ISOLATION", "SAMPLE_FAMILY_COVERAGE",
                "ZERO_RUNTIME_PASS_AUTHORITY"),
            prerequisites=(
                "LC15_FINAL_LEARNING_OBJECTIVE_MANIFEST",
                "READ_ONLY_EVALUATOR_LABEL", "UPSTREAM_COURSE_MANIFESTS"),
            blind_spots=(
                "CANDIDATE_ELIMINATION_NOT_EXECUTED", "LEARNED_STATE",
                "RUNTIME_ABLATION_NOT_EXECUTED",
                "RUNTIME_GENERALIZATION_NOT_EXECUTED"),
            owner="LC15_EVALUATOR_OWNER",
            sources=(
                "COURSE_MANIFEST", "EVALUATOR_LABEL", "OBSERVATION",
                "TEACHER_EVIDENCE"),
            evidence=(
                "data/ph2/manifests/lc15_final_learning_objectives_v1.json",
                "tests/test_d02_lc15_final_learning_objectives.py"),
            ne=(
                "CANDIDATE_ELIMINATION_NOT_EXECUTED",
                "CAPABILITY_LEARNED_REQUESTED", "NO_EVALUATOR_LABEL",
                "RUNTIME_ABLATION_NOT_EXECUTED",
                "RUNTIME_GENERALIZATION_REQUESTED")),
        _verifier(
            "MD01_MEMORY_DYNAMICS_CONTRACT_VERIFIER_V1",
            state="RUNTIME_EVIDENCED",
            capabilities=("MEMORY_DYNAMICS",),
            dimensions=(
                "CANONICAL_ROUND_TRIP", "CENTER_IDENTITY_AND_BOUNDARY",
                "CHANNEL_BUDGET_AND_ADMISSION",
                "MULTI_CENTER_IDENTITY_PRESERVATION",
                "OWNER_SCOPE_VERSION_ISOLATION",
                "RECEIPT_COUNT_AND_REASON_COMPLETENESS",
                "STOP_STATE_SUFFICIENCY", "ZERO_LEARNING_WRITE"),
            prerequisites=(
                "MD00_PREREGISTRATION", "MD01_CONTRACT_MANIFEST"),
            blind_spots=(
                "CENTER_DIFFUSION_QUALITY_OUTSIDE_STATIC_CONTRACT",
                "RETENTION_NOT_EVIDENCED"),
            owner="MD01_CONTRACT_EVALUATOR_OWNER",
            sources=("CONTRACT_MANIFEST", "SYNTHETIC_T0"),
            evidence=(
                "data/ph2/manifests/md01_memory_dynamics_contract_v1.json",
                "tests/test_d02_md01_memory_dynamics_contract.py"),
            ne=(
                "CENTER_DIFFUSION_QUALITY_REQUESTED",
                "RUNTIME_GENERALIZATION_REQUESTED")),
        _verifier(
            "MD02_SITUATION_STATE_ADAPTER_VERIFIER_V1",
            state="RUNTIME_EVIDENCED",
            capabilities=(
                "DISCOURSE_INFORMATION_STRUCTURE", "MEMORY_DYNAMICS",
                "REFERENCE_DISCOURSE_REVISION"),
            dimensions=(
                "BACKING_EVENT_IDENTITY", "DEPENDENCY_INDEX_REBUILD",
                "LOCAL_INVALIDATION_SCOPE", "ORIGINAL_EVENT_PRESERVATION",
                "OWNER_SCOPE_VERSION_ISOLATION",
                "UNAFFECTED_PROJECTION_BIT_IDENTITY",
                "ZERO_HOST_LEARNING_WRITE"),
            prerequisites=(
                "MD01_CONTRACT_MANIFEST", "REAL_MEMORY_EVENT_LOG",
                "REAL_WORK_MEMORY_CONTENT", "REAL_WORK_MEMORY_DISCOURSE"),
            blind_spots=(
                "CENTER_DIFFUSION_QUALITY_OUTSIDE_ADAPTER_SCOPE",
                "RETENTION_NOT_EVIDENCED"),
            owner="MD02_ADAPTER_EVALUATOR_OWNER",
            sources=(
                "APPEND_ONLY_MEMORY_EVENT", "SYNTHETIC_T0",
                "TRANSIENT_WORK_MEMORY"),
            evidence=(
                "data/ph2/manifests/md02_situation_state_adapter_v1.json",
                "src/pure_integer_ai/cognition/shared/situation_state.py",
                "tests/test_d02_md02_situation_state_adapter.py"),
            ne=(
                "CENTER_DIFFUSION_QUALITY_REQUESTED",
                "RUNTIME_GENERALIZATION_REQUESTED")),
        _verifier(
            "MD03_DIRECTIONAL_CENTER_ADAPTER_VERIFIER_V1",
            state="RUNTIME_EVIDENCED",
            capabilities=("MEMORY_DYNAMICS",),
            dimensions=(
                "ACTIVATION_ADOPTION_SEPARATION",
                "DIRECTIONAL_PAYLOAD_DISTINCTION",
                "EXACT_DEDUP_AND_PROVENANCE_MERGE",
                "OWNER_SCOPE_VERSION_ISOLATION",
                "STRENGTH_PRESERVATION",
                "WRITE_PERMISSION_ORTHOGONALITY",
                "ZERO_HOST_LEARNING_WRITE"),
            prerequisites=(
                "MD01_CONTRACT_MANIFEST", "MD02_SITUATION_STATE_ADAPTER",
                "REAL_ANSWER_GENERATION_GOAL", "REAL_MEMORY_CURRENT_QUERY",
                "REAL_REASONING_OBLIGATION"),
            blind_spots=(
                "CENTER_DIFFUSION_QUALITY_OUTSIDE_ADAPTER_SCOPE",
                "RETENTION_NOT_EVIDENCED",
                "RUNTIME_CONSUMPTION_NOT_EXECUTED"),
            owner="MD03_ADAPTER_EVALUATOR_OWNER",
            sources=("REAL_TYPED_INPUT_CONTRACTS", "SYNTHETIC_T0"),
            evidence=(
                "data/ph2/manifests/md03_directional_center_adapter_v1.json",
                "src/pure_integer_ai/experiments/ph2_md03_center_adapter.py",
                "tests/test_d02_md03_directional_center_adapter.py"),
            ne=(
                "CENTER_DIFFUSION_QUALITY_REQUESTED",
                "RETENTION_GENERALIZATION_REQUESTED",
                "RUNTIME_CONSUMPTION_REQUESTED")),
        _verifier(
            "MD05_CENTER_DIFFUSION_PROBE_VERIFIER_V1",
            state="RUNTIME_EVIDENCED",
            capabilities=("MEMORY_DYNAMICS",),
            dimensions=(
                "ABLATION_CAUSAL_DEGRADATION",
                "ACCESS_BUDGET_GROUNDING_CLASSIFICATION",
                "EXACT_STRUCTURE_DISTANCE_HELD_OUT",
                "FOUR_BASELINE_SAME_FIXTURE",
                "K04_TYPED_RANGE_RECEIPT",
                "MULTICENTER_PHYSICAL_READ_SHARING",
                "SOURCE_EVIDENCE_CHAIN_RECOVERY",
                "UNRELATED_1X_10X_100X_RESOURCE_SCALING",
                "ZERO_HOST_LEARNING_WRITE"),
            prerequisites=(
                "MD04_FROZEN_PROBE_PLAN", "MD04_RAW_RUN_ARTIFACT",
                "READ_ONLY_EVALUATOR_LABEL"),
            blind_spots=(
                "FORMAL_W_RUNTIME_NOT_EXECUTED",
                "PRODUCTIVE_LANGUAGE_GENERALIZATION",
                "RETENTION_NOT_EVIDENCED"),
            owner="MD05_PROBE_EVALUATOR_OWNER",
            sources=(
                "EVALUATOR_LABEL", "K04_PAGE_IN_RECEIPT", "RAW_PROBE_RUN"),
            evidence=(
                "data/ph2/manifests/md04_center_diffusion_probe_plan_v1.json",
                "data/ph2/manifests/md04_center_diffusion_probe_runs_v1.json",
                "data/ph2/manifests/md05_center_diffusion_decision_v1.json",
                "tests/test_d02_md04_md05_center_diffusion_probe.py"),
            ne=(
                "CAPABILITY_LEARNED_REQUESTED",
                "FORMAL_RUNTIME_INTEGRATION_REQUESTED",
                "RETENTION_GENERALIZATION_REQUESTED")),
        _verifier(
            "GG01_GENERATION_CHOICE_CONTRACT_VERIFIER_V1",
            state="COURSE_FROZEN",
            capabilities=("LAYERED_GENERATION",),
            dimensions=(
                "CANDIDATE_PREFLIGHT_ZERO_WRITE",
                "CONDITION_ACTION_SEPARATION",
                "CONTEXT_FIELD_COMPLETENESS",
                "FIVE_LAYER_CHOICE_COVERAGE",
                "GOAL_CONTEXT_BINDING",
                "OWNER_SCOPE_VERSION_ISOLATION"),
            prerequisites=(
                "CANDIDATE_LEARNING_RUNTIME",
                "GG00_GENERATION_COVERAGE_AUDIT"),
            blind_spots=(
                "GG02_USE_OUTCOME_BRIDGE_NOT_CONNECTED",
                "RUNTIME_GENERALIZATION_NOT_EVIDENCED"),
            owner="GG01_CONTRACT_VERIFIER_OWNER",
            sources=("CONTRACT_MANIFEST", "TYPED_FIXTURE"),
            evidence=(
                "data/ph2/manifests/gg01_generation_choice_contract_v1.json",
                "src/pure_integer_ai/experiments/ph2_generation_choice_contract.py",
                "tests/test_d02_gg01_generation_choice_contract.py"),
            ne=(
                "GENERATION_QUALITY_REQUESTED",
                "OUTCOME_ASSESSMENT_REQUESTED",
                "RUNTIME_LEARNING_REQUESTED")),
        _verifier(
            "GG01_GENERATION_CHOICE_CONTRACT_VERIFIER_V2",
            state="COURSE_FROZEN",
            capabilities=("LAYERED_GENERATION",),
            dimensions=(
                "CANDIDATE_PREFLIGHT_ZERO_WRITE",
                "CONDITION_ACTION_SEPARATION",
                "CONTEXT_FIELD_COMPLETENESS",
                "EXACT_KEYS_PRESERVE_ZERO_BEARING_CORE_MEMORY_IDENTITIES",
                "FIVE_LAYER_CHOICE_COVERAGE",
                "GOAL_CONTEXT_BINDING",
                "OWNER_SCOPE_VERSION_ISOLATION"),
            prerequisites=(
                "CANDIDATE_LEARNING_RUNTIME",
                "GG00_GENERATION_COVERAGE_AUDIT",
                "GG01_V1_IMMUTABLE_ARTIFACT"),
            blind_spots=(
                "ASSESSMENT_CONSUMER_NOT_CONNECTED",
                "RUNTIME_GENERALIZATION_NOT_EVIDENCED"),
            owner="GG01_CONTRACT_VERIFIER_OWNER",
            sources=("CONTRACT_MANIFEST", "TYPED_FIXTURE"),
            evidence=(
                "data/ph2/manifests/gg01_generation_choice_contract_v1.json",
                "data/ph2/manifests/gg01_generation_choice_contract_v2.json",
                "src/pure_integer_ai/experiments/ph2_generation_choice_contract.py",
                "tests/test_d02_gg01_generation_choice_contract.py"),
            ne=(
                "GENERATION_QUALITY_REQUESTED",
                "OUTCOME_ASSESSMENT_REQUESTED",
                "RUNTIME_LEARNING_REQUESTED")),
        _verifier(
            "GG02_GENERATION_CHOICE_OUTCOME_BRIDGE_VERIFIER_V1",
            state="COURSE_FROZEN",
            capabilities=("LAYERED_GENERATION",),
            dimensions=(
                "ASSESSMENT_LAYER_ISOLATION",
                "CLAIM_LAYER_AUTHORIZATION",
                "EXACT_USE_OUTCOME_LINK",
                "FIVE_LAYER_COMPLETENESS",
                "NO_SENTENCE_WIDE_BROADCAST",
                "OWNER_SCOPE_QUERY_BINDING",
                "READ_ONLY_VERIFIER_INPUT",
                "ZERO_HOST_LEARNING_WRITE"),
            prerequisites=(
                "EXACT_CORE_MEMORY_USE_KEYS",
                "GG01_V2_GENERATION_CHOICE_CONTRACT",
                "READ_ONLY_VERIFICATION_REPORT"),
            blind_spots=(
                "ASSESSMENT_CONSUMER_NOT_CONNECTED",
                "RUNTIME_CHOICE_ADOPTION_NOT_EXECUTED",
                "W_TRAINING_NOT_EXECUTED"),
            owner="GG02_BRIDGE_VERIFIER_OWNER",
            sources=(
                "CONTRACT_MANIFEST", "READ_ONLY_VERIFIER_REPORT",
                "TYPED_FIXTURE"),
            evidence=(
                "data/ph2/manifests/gg02_generation_choice_outcome_bridge_v1.json",
                "src/pure_integer_ai/experiments/ph2_generation_choice_outcome_bridge.py",
                "tests/test_d02_gg02_generation_choice_outcome_bridge.py"),
            ne=(
                "ASSESSMENT_UPDATE_REQUESTED",
                "GENERATION_QUALITY_REQUESTED",
                "RUNTIME_LEARNING_REQUESTED")),
        _verifier(
            "GG03_GENERATION_GENERALIZATION_COURSE_VERIFIER_V1",
            state="COURSE_FROZEN",
            capabilities=(
                "LAYERED_GENERATION", "PRAGMATIC_CLARIFICATION_REPAIR",
                "REFERENCE_DISCOURSE_REVISION", "SOURCE_UNCERTAINTY_REALITY",
                "TYPED_LEARNING_OBJECTIVES"),
            dimensions=(
                "ADDRESSEE_RECOVERABILITY", "COMBINATION_HELD_OUT",
                "COMMUNICATIVE_TASK", "EXACT_MEMORY_BASELINE_REJECT",
                "FAILURE_LAYER_LOCALIZATION", "LEGAL_OBJECT_COMPOSITION",
                "MULTIPLE_LEGAL_SURFACE_SET", "RETENTION_REVERIFY",
                "REVISION_SUPERSEDE", "SEMANTIC_ROLE_SCOPE_POLARITY",
                "SOURCE_UNCERTAINTY_CITATION",
                "STANCE_CONTENT_WORDING_SEPARATION", "STRUCTURE_SLOT_ORDER",
                "USE_OUTCOME_TEMPLATE_PROMOTION_REJECT"),
            prerequisites=(
                "D02_FOUR_OWNER_PACK", "GG01_V2_GENERATION_CHOICE_CONTRACT",
                "GG02_LAYERED_USE_OUTCOME_BRIDGE"),
            blind_spots=(
                "ASSESSMENT_CONSUMER_NOT_CONNECTED",
                "FORMAL_W_RUNTIME_NOT_EXECUTED",
                "RUNTIME_GENERALIZATION_NOT_EVIDENCED"),
            owner="GG03_EVALUATOR_OWNER",
            sources=(
                "COURSE_MANIFEST", "EVALUATOR_LABEL", "OBSERVATION_RECORD"),
            evidence=(
                "data/ph2/authored_generation_generalization_seed_v1.jsonl.sample",
                "data/ph2/manifests/gg03_generation_generalization_course_v1.json",
                "src/pure_integer_ai/experiments/ph2_authored_generation_generalization_course.py",
                "src/pure_integer_ai/experiments/ph2_generation_generalization_contract.py",
                "tests/test_d02_gg03_generation_generalization_course.py"),
            ne=(
                "INDEPENDENT_LAYER_INPUT_MISSING", "NO_EVALUATOR_LABEL",
                "OUT_OF_REGISTER_OR_CONTEXT_SCOPE",
                "RETENTION_EPISODE_NOT_EXECUTED")),
        _verifier(
            "MULTI_LEGAL_SURFACE_VERIFIER_V1", state="COURSE_FROZEN",
            capabilities=(
                "LAYERED_GENERATION", "PRAGMATIC_CLARIFICATION_REPAIR",
                "REFERENCE_DISCOURSE_REVISION"),
            dimensions=(
                "ADDRESSEE_RECOVERABILITY", "MULTIPLE_LEGAL_SURFACE",
                "STRUCTURE_SEMANTIC_SOURCE_TASK_CONJUNCTION"),
            prerequisites=("GENERATION_CONTEXT", "INDEPENDENT_LAYER_VERIFIERS"),
            blind_spots=(
                "ASSESSMENT_CONSUMER_NOT_CONNECTED",
                "FORMAL_W_RUNTIME_NOT_EXECUTED"),
            owner="GG03_EVALUATOR_OWNER",
            sources=("EVALUATOR_LABEL", "OBSERVATION_RECORD"),
            evidence=(
                "data/ph2/manifests/gg03_generation_generalization_course_v1.json",
                "src/pure_integer_ai/experiments/ph2_generation_generalization_contract.py",
                "tests/test_d02_gg03_generation_generalization_course.py"),
            ne=(
                "INDEPENDENT_LAYER_INPUT_MISSING", "NO_EVALUATOR_LABEL",
                "OUT_OF_REGISTER_OR_CONTEXT_SCOPE")),
        _verifier(
            "LC09_TRANSFER_AXIS_VERIFIER_V1", state="COURSE_FROZEN",
            capabilities=("TRANSFER_AXES",),
            dimensions=LC09_VERIFIER_DIMENSIONS,
            prerequisites=(
                "D02_FORMAL_PACK_INVENTORY",
                "LC09_TRANSFER_AXIS_MANIFEST_V1"),
            blind_spots=(
                "FORMAL_RUNTIME_NOT_EXECUTED",
                "TRANSFER_RESULT_NOT_OBSERVED"),
            owner="LC09_EVALUATOR_OWNER",
            sources=(
                "ARTIFACT_MANIFEST", "OBSERVATION_RECORD",
                "SOURCE_REF_RECORD", "TYPED_SPLIT_FIXTURE"),
            evidence=(
                "data/ph2/manifests/lc09_transfer_axis_manifest_v1.json",
                "src/pure_integer_ai/experiments/ph2_transfer_axis_catalog.py",
                "src/pure_integer_ai/experiments/ph2_transfer_axis_contract.py",
                "tests/test_d02_lc09_transfer_axis_manifest.py"),
            ne=LC09_VERIFIER_NE_CONDITIONS),
        _verifier(
            "RETENTION_ROLLBACK_VERIFIER_V1", state="COURSE_FROZEN",
            capabilities=(
                "EVALUATOR_RETENTION_RESOURCE", "OPEN_SET_CONTINUAL_LEARNING",
                "TRANSFER_AXES"),
            dimensions=LC10_VERIFIER_DIMENSIONS,
            prerequisites=(
                "D02_APPEND_ONLY_EVENT_FACILITIES",
                "LC10_RETENTION_ROLLBACK_MANIFEST_V1"),
            blind_spots=(
                "GENERAL_SOURCE_WITHDRAWAL_PROTOCOL_ONLY",
                "LC13_DIRECTIONAL_CONSUMER_MAP_NOT_FROZEN",
                "RETENTION_EPISODE_NOT_EXECUTED",
                "V06_RETENTION_CLONE_NOT_EXECUTED",
                "W01_STATE_PROTOCOL_NOT_STARTED",
                "W09_NOT_STARTED"),
            owner="LC10_EVALUATOR_OWNER",
            sources=(
                "RETENTION_CHECKPOINT_DIGEST", "RETENTION_DIMENSION_RESULT",
                "RUNTIME_BINDING_FILE_IDENTITY", "TYPED_PROTOCOL_FIXTURE"),
            evidence=(
                "data/ph2/manifests/lc10_retention_rollback_manifest_v1.json",
                "src/pure_integer_ai/experiments/ph2_retention_rollback_catalog.py",
                "src/pure_integer_ai/experiments/ph2_retention_rollback_contract.py",
                "tests/test_d02_lc10_retention_rollback_manifest.py"),
            ne=LC10_VERIFIER_NE_CONDITIONS),
        _verifier(
            "LC13_DIRECTIONAL_CONSUMER_VERIFIER_V1",
            state="COURSE_FROZEN",
            capabilities=all_caps,
            dimensions=LC13_VERIFIER_DIMENSIONS,
            prerequisites=(
                "GG01_GENERATION_CHOICE_CONTRACT_V2",
                "GG02_LAYERED_USE_OUTCOME_BRIDGE_V1",
                "LANGUAGE_CAPABILITY_BASELINE_V32"),
            blind_spots=(
                "ASSESSMENT_CONSUMER_NOT_CONNECTED",
                "DIRECTIONAL_RUNTIME_NOT_EXECUTED",
                "FORMAL_W_RUNTIME_NOT_STARTED",
                "MISSING_CONSUMER_REMAINS_NE"),
            owner="LC13_DIRECTIONAL_EVALUATOR_OWNER",
            sources=(
                "CONSUMER_FILE_IDENTITY", "DIRECTIONAL_CONSUMER_ROUTE",
                "LAYERED_POSTCHECK_DECLARATION"),
            evidence=(
                "data/ph2/manifests/lc13_directional_consumer_manifest_v1.json",
                "src/pure_integer_ai/experiments/ph2_directional_consumer_catalog.py",
                "src/pure_integer_ai/experiments/ph2_directional_consumer_contract.py",
                "tests/test_d02_lc13_directional_consumer_manifest.py"),
            ne=LC13_VERIFIER_NE_CONDITIONS),
        _verifier(
            "NL00_NONLITERAL_SCOPE_PROBE_VERIFIER_V1",
            state="COURSE_FROZEN",
            capabilities=("NONLITERAL_CULTURAL",),
            dimensions=NL00_VERIFIER_DIMENSIONS,
            prerequisites=(
                "LC03_CONSTRUCTION_COURSE_V1",
                "LC07_DISCOURSE_INFORMATION_COURSE_V1",
                "LC08_OPEN_SET_CLARIFICATION_COURSE_V1",
                "NL00_NONLITERAL_SCOPE_PROBE_MANIFEST_V1"),
            blind_spots=(
                "CULTURAL_GROUNDING_NOT_AUTHORIZED",
                "DEEP_NONLITERAL_RUNTIME_NOT_IMPLEMENTED",
                "DISC08_FIRST_PHASE_DEPTH_UNDECIDED",
                "DISC12_INDEPENDENT_EVALUATOR_SIGNAL_UNDECIDED",
                "LEXICALIZED_IDIOM_PASS_IS_REPRESENTABILITY_ONLY"),
            owner="NL00_EVALUATOR_OWNER",
            sources=(
                "CURRENT_FILE_IDENTITY", "LAYER_INVARIANT_RESULT",
                "SCOPE_AND_WALL_DECISION"),
            evidence=(
                "data/ph2/manifests/nl00_nonliteral_scope_probe_manifest_v1.json",
                "src/pure_integer_ai/experiments/ph2_nonliteral_scope_probe_catalog.py",
                "src/pure_integer_ai/experiments/ph2_nonliteral_scope_probe_contract.py",
                "tests/test_d02_nl00_nonliteral_scope_probe.py"),
            ne=NL00_VERIFIER_NE_CONDITIONS),
        _verifier(
            "RI00_REASONING_MODE_PROBE_VERIFIER_V1",
            state="COURSE_FROZEN",
            capabilities=("RELATION_LOGIC_FOUR_STATE",),
            dimensions=RI00_VERIFIER_DIMENSIONS,
            prerequisites=(
                "LC05_EVENT_TIME_ASPECT_COURSE_V1",
                "LC06_COMPARISON_QUANTITY_COURSE_V1",
                "LC07_DISCOURSE_INFORMATION_COURSE_V1",
                "RI00_REASONING_MODE_PROBE_MANIFEST_V1"),
            blind_spots=(
                "FOUR_REJECTED_MODES_NOT_IMPLEMENTED",
                "FORMAL_W07_RUNTIME_NOT_STARTED",
                "TEMPORAL_PASS_IS_BOUNDED_ONLY"),
            owner="RI00_EVALUATOR_OWNER",
            sources=(
                "CURRENT_FILE_IDENTITY", "MODE_INVARIANT_RESULT",
                "SCOPE_DECISION"),
            evidence=(
                "data/ph2/manifests/ri00_reasoning_mode_probe_manifest_v2.json",
                "src/pure_integer_ai/experiments/ph2_reasoning_mode_probe_catalog.py",
                "src/pure_integer_ai/experiments/ph2_reasoning_mode_probe_contract.py",
                "tests/test_d02_ri00_reasoning_mode_probe.py"),
            ne=RI00_VERIFIER_NE_CONDITIONS),
        _verifier(
            "SOURCE_SNAPSHOT_INTEGRITY_V1", state="RUNTIME_EVIDENCED",
            capabilities=("RAW_TEXT_NOISE", "SOURCE_UNCERTAINTY_REALITY"),
            dimensions=("HASH", "LICENSE", "PARSER_REPORT", "PROVENANCE"),
            prerequisites=("RAW_SNAPSHOT_MANIFEST",),
            blind_spots=("LANGUAGE_TRUTH", "STUDENT_LEARNING"),
            owner="SOURCE_AUDIT_OWNER", sources=("RAW_BYTES", "UPSTREAM_METADATA"),
            evidence=("tests/test_d02_mediawiki_snapshot_adapter.py",),
            ne=("CAPABILITY_PASS_REQUESTED",)),
        _verifier(
            "V06_HOST_WRITE_ISOLATION_V1", state="RUNTIME_EVIDENCED",
            capabilities=all_caps,
            dimensions=("CLONE_WRITE", "HOST_BIT_IDENTITY", "OWNER_ISOLATION"),
            prerequisites=("V06_CLONE",),
            blind_spots=("CAPABILITY_CORRECTNESS", "SEMANTIC_TRUTH"),
            owner="EVALUATION_ISOLATION_OWNER", sources=("CLONE_EVENT_LOG",),
            evidence=("tests/test_v06_evaluation_isolation.py",),
            ne=("NO_CLONE_EXECUTION", "SEMANTIC_ABILITY_REQUESTED")),
    )
    return VerifierCapabilityRegistry(
        1, "LC-11-verifier-capability-registry-public-clean-ri00-v2-v24",
        records)


_PREREQUISITES = {
    "ATTRIBUTION_QUOTATION_PERSPECTIVE": (
        "DISCOURSE_INFORMATION_STRUCTURE", "EVENT_TIME_ASPECT"),
    "COMPARISON_QUANTITY_MEASURE": ("MORPHOLOGY_WORD_FORM", "RECURSIVE_PARSE"),
    "DISCOURSE_INFORMATION_STRUCTURE": (
        "EVENT_TIME_ASPECT", "REFERENCE_DISCOURSE_REVISION"),
    "EVALUATOR_RETENTION_RESOURCE": (),
    "EVENT_TIME_ASPECT": ("RECURSIVE_PARSE",),
    "LAYERED_GENERATION": (
        "RAW_TEXT_NOISE", "REFERENCE_DISCOURSE_REVISION",
        "RELATION_LOGIC_FOUR_STATE"),
    "MEMORY_DYNAMICS": ("REFERENCE_DISCOURSE_REVISION",),
    "MORPHOLOGY_WORD_FORM": ("RAW_TEXT_NOISE",),
    "MULTIWORD_CONSTRUCTION": ("MORPHOLOGY_WORD_FORM", "RAW_TEXT_NOISE"),
    "NONLITERAL_CULTURAL": (
        "DISCOURSE_INFORMATION_STRUCTURE", "MULTIWORD_CONSTRUCTION",
        "OPEN_SET_CONTINUAL_LEARNING"),
    "NON_TEXT_MEDIA": (),
    "OPEN_SET_CONTINUAL_LEARNING": ("DISCOURSE_INFORMATION_STRUCTURE",),
    "PRAGMATIC_CLARIFICATION_REPAIR": ("DISCOURSE_INFORMATION_STRUCTURE",),
    "RAW_TEXT_NOISE": (),
    "RECURSIVE_PARSE": ("MULTIWORD_CONSTRUCTION", "RAW_TEXT_NOISE"),
    "REFERENCE_DISCOURSE_REVISION": ("RECURSIVE_PARSE",),
    "RELATION_LOGIC_FOUR_STATE": ("RECURSIVE_PARSE",),
    "SOURCE_UNCERTAINTY_REALITY": (),
    "TRANSFER_AXES": ("RAW_TEXT_NOISE",),
    "TYPED_LEARNING_OBJECTIVES": ("RAW_TEXT_NOISE",),
}

_FROZEN_SAMPLE_FAMILIES = {
    "ATTRIBUTION_QUOTATION_PERSPECTIVE": set(SAMPLE_FAMILIES),
    "COMPARISON_QUANTITY_MEASURE": set(SAMPLE_FAMILIES),
    "DISCOURSE_INFORMATION_STRUCTURE": set(SAMPLE_FAMILIES),
    "LAYERED_GENERATION": set(SAMPLE_FAMILIES),
    "MORPHOLOGY_WORD_FORM": set(SAMPLE_FAMILIES),
    "MULTIWORD_CONSTRUCTION": set(SAMPLE_FAMILIES),
    "OPEN_SET_CONTINUAL_LEARNING": set(SAMPLE_FAMILIES),
    "PRAGMATIC_CLARIFICATION_REPAIR": set(SAMPLE_FAMILIES),
    "RAW_TEXT_NOISE": set(SAMPLE_FAMILIES),
    "RECURSIVE_PARSE": set(SAMPLE_FAMILIES),
    "EVENT_TIME_ASPECT": set(SAMPLE_FAMILIES),
    "REFERENCE_DISCOURSE_REVISION": set(SAMPLE_FAMILIES),
    "RELATION_LOGIC_FOUR_STATE": {
        "GENERATION", "NEGATIVE", "POSITIVE", "REVISION",
    },
    "SOURCE_UNCERTAINTY_REALITY": {
        "AMBIGUOUS", "GENERATION", "NEGATIVE", "POSITIVE", "REVISION",
        "UNKNOWN",
    },
    "TYPED_LEARNING_OBJECTIVES": set(SAMPLE_FAMILIES),
}

_SOURCE_LICENSE_GAPS = {
    "MORPHOLOGY_WORD_FORM": "W02_CC_CEDICT_BLOCKED_ALTERNATIVES_PARTIAL",
    "MULTIWORD_CONSTRUCTION": "W03_CC_CEDICT_BLOCKED_ALTERNATIVES_PARTIAL",
    "SOURCE_UNCERTAINTY_REALITY": (
        "W03_CC_CEDICT_BLOCKED_ALTERNATIVES_PARTIAL"),
}
_CC_CEDICT_RECONCILIATION_REF = (
    "data/ph2/manifests/"
    "cc_cedict_20260725.license_reconciliation_v1.json")


def _sample_states(capability_key: str) -> CanonicalJsonObject:
    if capability_key == "NON_TEXT_MEDIA":
        return CanonicalJsonObject.from_value({
            key: "OUT_OF_SCOPE" for key in SAMPLE_FAMILIES
        })
    if capability_key in {
            "EVALUATOR_RETENTION_RESOURCE", "TRANSFER_AXES"}:
        return CanonicalJsonObject.from_value({
            key: "NE" for key in SAMPLE_FAMILIES
        })
    values = {key: "MISSING" for key in SAMPLE_FAMILIES}
    for key in _FROZEN_SAMPLE_FAMILIES.get(capability_key, ()):
        values[key] = "FROZEN"
    return CanonicalJsonObject.from_value(values)


def _failure_suffix(capability_key: str) -> tuple[str, ...]:
    mapping = {
        "ATTRIBUTION_QUOTATION_PERSPECTIVE": ("D-03", "W-05", "W-08", "W-09"),
        "COMPARISON_QUANTITY_MEASURE": ("D-03", "W-05", "W-07", "W-09"),
        "DISCOURSE_INFORMATION_STRUCTURE": ("D-03", "W-08", "W-09"),
        "EVALUATOR_RETENTION_RESOURCE": ("D-03", "W-09"),
        "EVENT_TIME_ASPECT": ("D-03", "W-05", "W-07", "W-08", "W-09"),
        "LAYERED_GENERATION": ("D-03", "W-02", "W-03", "W-04", "W-05", "W-06", "W-07", "W-08", "W-09"),
        "MEMORY_DYNAMICS": ("D-03", "W-08", "W-09"),
        "MORPHOLOGY_WORD_FORM": ("D-03", "W-02", "W-03", "W-09"),
        "MULTIWORD_CONSTRUCTION": ("D-03", "W-03", "W-05", "W-09"),
        "NONLITERAL_CULTURAL": ("D-03", "W-09"),
        "NON_TEXT_MEDIA": ("WALL",),
        "OPEN_SET_CONTINUAL_LEARNING": ("D-03", "W-08", "W-09"),
        "PRAGMATIC_CLARIFICATION_REPAIR": ("D-03", "W-08", "W-09"),
        "RAW_TEXT_NOISE": ("D-03", "W-02", "W-03", "W-05", "W-09"),
        "RECURSIVE_PARSE": ("D-03", "W-05", "W-08", "W-09"),
        "REFERENCE_DISCOURSE_REVISION": ("D-03", "W-08", "W-09"),
        "RELATION_LOGIC_FOUR_STATE": ("D-03", "W-06", "W-07", "W-09"),
        "SOURCE_UNCERTAINTY_REALITY": ("D-03", "W-06", "W-07", "W-08", "W-09"),
        "TRANSFER_AXES": ("D-03", "W-02", "W-03", "W-04", "W-05", "W-06", "W-07", "W-08", "W-09"),
        "TYPED_LEARNING_OBJECTIVES": ("D-03", "W-02", "W-03", "W-04", "W-05", "W-06", "W-07", "W-08", "W-09"),
    }
    return mapping[capability_key]


def build_course_coverage_ledger() -> CapabilityCourseCoverageLedger:
    """建立 LC-12 当前账，显示最早失效、后缀和来源许可缺口。"""
    records = []
    for capability_key in CAPABILITY_KEYS:
        if capability_key == "NON_TEXT_MEDIA":
            exit_state = "OUT_OF_SCOPE"
            earliest = "WALL"
        elif capability_key in {
                "ATTRIBUTION_QUOTATION_PERSPECTIVE",
                "COMPARISON_QUANTITY_MEASURE", "EVENT_TIME_ASPECT",
                "DISCOURSE_INFORMATION_STRUCTURE",
                "EVALUATOR_RETENTION_RESOURCE", "MORPHOLOGY_WORD_FORM",
                "MULTIWORD_CONSTRUCTION", "OPEN_SET_CONTINUAL_LEARNING",
                "PRAGMATIC_CLARIFICATION_REPAIR", "RAW_TEXT_NOISE",
                "RECURSIVE_PARSE",
                "REFERENCE_DISCOURSE_REVISION",
                "LAYERED_GENERATION",
                "TRANSFER_AXES",
                "TYPED_LEARNING_OBJECTIVES"}:
            exit_state = "COURSE_FROZEN"
            earliest = "D-03_RUNTIME_NOT_STARTED"
        elif capability_key in _FROZEN_SAMPLE_FAMILIES:
            exit_state = "PARTIAL_COURSE"
            earliest = "D-03_PREP"
        else:
            exit_state = "BASELINE_ONLY"
            earliest = "D-02"
        external_prerequisites = {
            "D02_DATASET_CONTRACT_V1", "LC-13",
            "LC13_DIRECTIONAL_CONSUMER_MANIFEST_V1",
            *_TASKS[capability_key]
        }
        if capability_key in {
                "RAW_TEXT_NOISE", "TYPED_LEARNING_OBJECTIVES"}:
            external_prerequisites.add("LC01_LC15_INITIAL_COURSE_V1")
        if capability_key in {
                "MORPHOLOGY_WORD_FORM", "TYPED_LEARNING_OBJECTIVES"}:
            external_prerequisites.add("LC02_MORPHOLOGY_COURSE_V1")
        if capability_key in {
                "MULTIWORD_CONSTRUCTION", "TYPED_LEARNING_OBJECTIVES"}:
            external_prerequisites.add("LC03_CONSTRUCTION_COURSE_V1")
        if capability_key in {
                "RECURSIVE_PARSE", "TYPED_LEARNING_OBJECTIVES"}:
            external_prerequisites.add("LC04_RECURSIVE_PARSE_COURSE_V1")
        if capability_key in {
                "EVENT_TIME_ASPECT", "TYPED_LEARNING_OBJECTIVES"}:
            external_prerequisites.add("LC05_EVENT_TIME_ASPECT_COURSE_V1")
        if capability_key in {
                "COMPARISON_QUANTITY_MEASURE", "TYPED_LEARNING_OBJECTIVES"}:
            external_prerequisites.add("LC06_COMPARISON_QUANTITY_COURSE_V1")
        if capability_key in {
                "DISCOURSE_INFORMATION_STRUCTURE",
                "REFERENCE_DISCOURSE_REVISION",
                "TYPED_LEARNING_OBJECTIVES"}:
            external_prerequisites.add(
                "LC07_DISCOURSE_INFORMATION_COURSE_V1")
        if capability_key in {
                "OPEN_SET_CONTINUAL_LEARNING",
                "PRAGMATIC_CLARIFICATION_REPAIR",
                "TYPED_LEARNING_OBJECTIVES"}:
            external_prerequisites.add(
                "LC08_OPEN_SET_CLARIFICATION_COURSE_V1")
        if capability_key in {
                "ATTRIBUTION_QUOTATION_PERSPECTIVE",
                "SOURCE_UNCERTAINTY_REALITY",
                "TYPED_LEARNING_OBJECTIVES"}:
            external_prerequisites.add(
                "LC14_ATTRIBUTION_QUOTATION_COURSE_V1")
        if capability_key == "ATTRIBUTION_QUOTATION_PERSPECTIVE":
            external_prerequisites.add(
                "RUNTIME_SCOPE_CONSUMER_NOT_EXECUTED")
        if capability_key in {
                "DISCOURSE_INFORMATION_STRUCTURE",
                "REFERENCE_DISCOURSE_REVISION"}:
            external_prerequisites.update({
                "MD02_SITUATION_STATE_ADAPTER_V1",
                "MD03_DIRECTIONAL_CENTER_ADAPTER_V1",
            })
        if capability_key in {
                "OPEN_SET_CONTINUAL_LEARNING",
                "PRAGMATIC_CLARIFICATION_REPAIR"}:
            external_prerequisites.add(
                "MD03_DIRECTIONAL_CENTER_ADAPTER_V1")
        if capability_key == "TYPED_LEARNING_OBJECTIVES":
            external_prerequisites.add("LC15_FINAL_LEARNING_OBJECTIVES_V1")
        if capability_key == "TRANSFER_AXES":
            external_prerequisites.add("LC09_TRANSFER_AXIS_MANIFEST_V1")
        if capability_key in {
                "EVALUATOR_RETENTION_RESOURCE",
                "OPEN_SET_CONTINUAL_LEARNING", "TRANSFER_AXES"}:
            external_prerequisites.add(
                "LC10_RETENTION_ROLLBACK_MANIFEST_V1")
        if capability_key == "RELATION_LOGIC_FOUR_STATE":
            external_prerequisites.add(
                "RI00_REASONING_MODE_PROBE_MANIFEST_V1")
        if capability_key == "NONLITERAL_CULTURAL":
            external_prerequisites.add(
                "NL00_NONLITERAL_SCOPE_PROBE_MANIFEST_V1")
        if capability_key in {
                "LAYERED_GENERATION", "PRAGMATIC_CLARIFICATION_REPAIR",
                "REFERENCE_DISCOURSE_REVISION", "SOURCE_UNCERTAINTY_REALITY",
                "TYPED_LEARNING_OBJECTIVES"}:
            external_prerequisites.add(
                "GG03_GENERATION_GENERALIZATION_COURSE_V1")
        evidence_refs = set(_EVIDENCE[capability_key])
        evidence_refs.update({
            "data/ph2/manifests/lc13_directional_consumer_manifest_v1.json",
            "src/pure_integer_ai/experiments/ph2_directional_consumer_catalog.py",
            "src/pure_integer_ai/experiments/ph2_directional_consumer_contract.py",
            "tests/test_d02_lc13_directional_consumer_manifest.py",
        })
        if capability_key in {
                "OPEN_SET_CONTINUAL_LEARNING", "TRANSFER_AXES"}:
            evidence_refs.add(
                "data/ph2/manifests/"
                "lc10_retention_rollback_manifest_v1.json")
        if capability_key in {
                "DISCOURSE_INFORMATION_STRUCTURE", "OPEN_SET_CONTINUAL_LEARNING",
                "PRAGMATIC_CLARIFICATION_REPAIR",
                "REFERENCE_DISCOURSE_REVISION"}:
            evidence_refs.update({
                "data/ph2/manifests/md02_situation_state_adapter_v1.json",
                "data/ph2/manifests/md03_directional_center_adapter_v1.json",
                "src/pure_integer_ai/cognition/shared/situation_state.py",
                "src/pure_integer_ai/experiments/ph2_md03_center_adapter.py",
                "tests/test_d02_md02_situation_state_adapter.py",
                "tests/test_d02_md03_directional_center_adapter.py",
            })
        source_gap = _SOURCE_LICENSE_GAPS.get(capability_key)
        if source_gap is not None:
            external_prerequisites.add(source_gap)
            evidence_refs.add(_CC_CEDICT_RECONCILIATION_REF)
        records.append(CapabilityCourseCoverage(
            capability_key,
            tuple(sorted(_PREREQUISITES[capability_key])),
            tuple(sorted(external_prerequisites)),
            _sample_states(capability_key),
            earliest,
            _failure_suffix(capability_key),
            exit_state,
            tuple(sorted(evidence_refs)),
        ))
    return CapabilityCourseCoverageLedger(
        1, "LC-12-course-coverage-stop-ledger-public-clean-ri00-v2-v20",
        tuple(records))


def build_md00_preregistration() -> MDProbePreRegistration:
    """冻结四基线、十样本组、指标、硬零项和 theater 判据。"""
    return MDProbePreRegistration(
        "MD-00-center-expansion-preregistration-v1",
        "PRE_REGISTERED",
        0,
        MD_BASELINE_KEYS,
        MD_SAMPLE_GROUP_KEYS,
        CanonicalJsonObject.from_value({
            "audit": [
                "OLD_OBSERVATION_EVIDENCE_PRESERVED",
                "OWNER_SCOPE_VERSION_VIOLATION",
                "RECEIPT_COMPLETENESS",
                "TEACHER_HELD_OUT_LEAKAGE",
                "UNAFFECTED_PROJECTION_BIT_IDENTITY",
            ],
            "quality": [
                "ADOPTED_CORRECT",
                "CLARIFY_UNKNOWN_BLOCKED_CLASSIFICATION",
                "GENERATION_ADDRESSEE_RECOVERABILITY",
                "GENERATION_SEMANTIC_PRESERVATION",
                "MISSED_REFUTE",
                "WRONG_ADOPTION",
            ],
            "resource": [
                "AGENDA_ENTRIES",
                "CONSUMED_OBJECTS",
                "LOGIC_STEPS",
                "OPENED_PAGE_SEGMENT",
                "RECOMPUTED_OBJECTS",
                "SCANNED_OBJECTS",
            ],
        }),
        MD_HARD_INVARIANT_KEYS,
        (
            "DEPENDENCY_INVALIDATION",
            "LAYERED_ATTRIBUTION",
            "STOP_DECISION",
            "TYPED_CENTER",
            "TYPED_CHANNEL_SELECTION",
        ),
        CanonicalJsonObject.from_value({
            "comparison_order": [
                "FIXED_TOP_K", "RECENCY_HOT_ONLY", "TYPED_FIXED_RING",
                "OBLIGATION_CONDITIONED_MULTICHANNEL_STOP",
            ],
            "decision_rule": (
                "PASS_IFF_HARD_ZERO_AND_NO_QUALITY_REGRESSION_AND_AT_LEAST_"
                "ONE_CHALLENGE_IMPROVEMENT_AND_ALL_ABLATIONS_DEGRADE"),
            "freeze_before_run": 1,
            "hard_zero_policy": 1,
            "primary_candidate": "OBLIGATION_CONDITIONED_MULTICHANNEL_STOP",
            "resource_ceiling_rule": (
                "EACH_SAMPLE_GROUP_INTEGER_CEILING_FROZEN_BEFORE_FIRST_RUN"),
            "theater_rule": (
                "EACH_ABLATION_MUST_DEGRADE_AT_LEAST_ONE_PREREGISTERED_DIMENSION"),
        }),
        (
            "src/pure_integer_ai/cognition/shared/memory_query.py",
            "src/pure_integer_ai/cognition/shared/reasoning_planner.py",
            "src/pure_integer_ai/experiments/attractor_runtime.py",
            "src/pure_integer_ai/experiments/memory_generation_runtime.py",
        ),
    )


def _gg_row(
        key: str,
        *,
        state: str,
        gap: str,
        evidence: tuple[str, ...],
        ) -> GenerationCoverageAuditRow:
    return GenerationCoverageAuditRow(
        key, state, gap, tuple(sorted(evidence)))


def build_gg00_audit() -> GenerationCoverageAudit:
    """冻结 GG-03 课程覆盖；runtime、训练和 assessment 消费仍保持未开始。"""
    gg03_evidence = (
        "data/ph2/authored_generation_generalization_seed_v1.jsonl.sample",
        "data/ph2/manifests/gg03_generation_generalization_course_v1.json",
        "src/pure_integer_ai/experiments/ph2_authored_generation_generalization_course.py",
        "src/pure_integer_ai/experiments/ph2_generation_generalization_contract.py",
        "tests/test_d02_gg03_generation_generalization_course.py",
    )
    course_state = {
        key: ("PRESENT", "NONE") for key in GG_COURSE_FAMILY_KEYS
    }
    course_rows = tuple(_gg_row(
        key,
        state=course_state[key][0],
        gap=course_state[key][1],
        evidence=gg03_evidence,
    ) for key in GG_COURSE_FAMILY_KEYS)
    axis_state = {
        key: ("PRESENT", "NONE") for key in GG_COMBINATION_AXIS_KEYS
    }
    axis_rows = tuple(_gg_row(
        key,
        state=axis_state[key][0],
        gap=axis_state[key][1],
        evidence=(
            "src/pure_integer_ai/experiments/ph2_dataset_contract.py",
            "src/pure_integer_ai/experiments/ph2_dataset_manifest.py",
            *gg03_evidence,
        ),
    ) for key in GG_COMBINATION_AXIS_KEYS)
    stage_state = {key: ("PRESENT", "NONE") for key in GG_STAGE_KEYS}
    stage_base_evidence = (
        "data/ph2/manifests/gg01_generation_choice_contract_v2.json",
        "data/ph2/manifests/gg02_generation_choice_outcome_bridge_v1.json",
        "src/pure_integer_ai/experiments/language_generation_course.py",
        "src/pure_integer_ai/experiments/ph2_authored_generation_course.py",
        "tests/test_d02e_authored_generation_postcheck_course.py",
        *gg03_evidence,
    )
    stage_rows = tuple(_gg_row(
        key,
        state=stage_state[key][0],
        gap=stage_state[key][1],
        evidence=(
            "data/ph2/manifests/lc06_comparison_quantity_course_v1.json",
            "tests/test_d02_lc06_comparison_quantity_course.py",
            *stage_base_evidence,
        ) if key in {"W05", "W07"} else (
            "data/ph2/manifests/lc07_discourse_information_course_v1.json",
            "data/ph2/manifests/lc08_open_set_clarification_course_v1.json",
            "data/ph2/manifests/lc14_attribution_quotation_course_v1.json",
            "tests/test_d02_lc07_discourse_information_course.py",
            "tests/test_d02_lc08_open_set_clarification_course.py",
            "tests/test_d02_lc14_attribution_quotation_course.py",
            *stage_base_evidence,
        ) if key == "W08" else stage_base_evidence,
    ) for key in GG_STAGE_KEYS)
    return GenerationCoverageAudit(
        "GG-00-generation-coverage-audit-gg03-v2",
        "COURSE_FROZEN",
        course_rows,
        axis_rows,
        stage_rows,
    )


def build_language_baseline_manifest(
        *,
        artifact_version: str,
        head_sha1: str,
        origin_master_sha1: str,
        untracked_file_count: int,
        inventory_exclusions: tuple[str, ...],
        file_inventory: tuple[PublicFileIdentity, ...],
        paper_files: tuple[PublicFileIdentity, ...],
        public_gate: PublicGateBaseline,
        ) -> LanguageBaselineManifest:
    """汇合切片 2 的纯账目与调用方文件级证据。"""
    return LanguageBaselineManifest(
        1,
        artifact_version,
        "BASELINE_FROZEN",
        head_sha1,
        origin_master_sha1,
        0,
        0,
        untracked_file_count,
        inventory_exclusions,
        file_inventory,
        paper_files,
        public_gate,
        build_capability_ledger(),
        build_verifier_registry(),
        build_course_coverage_ledger(),
        build_md00_preregistration(),
        build_gg00_audit(),
        CanonicalJsonObject.from_value({
            "companion_writes": 0,
            "core_learning_writes": 0,
            "d03_published": 0,
            "formal_training_runs": 0,
            "mastered_claims": 0,
            "memory_learning_writes": 0,
            "readiness_claims": 0,
            "teacher_calls": 0,
            "use_learning_writes": 0,
            "w01_started": 0,
        }),
    )


__all__ = [
    "build_capability_ledger",
    "build_course_coverage_ledger",
    "build_gg00_audit",
    "build_language_baseline_manifest",
    "build_md00_preregistration",
    "build_verifier_registry",
]
