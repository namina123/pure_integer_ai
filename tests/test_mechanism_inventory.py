"""生产四态和消费者台账回归。"""
from __future__ import annotations

from pure_integer_ai.experiments.mechanism_inventory import (
    STATUS_DEAD,
    STATUS_OPT_IN,
    STATUS_PRODUCTION,
    STATUS_TEST_ONLY,
    SCOPE_SHARED,
    inventory_by_id,
    inventory_json,
    readiness_candidates,
    validate_inventory,
)


def test_inventory_is_structurally_valid_and_json_ready():
    assert validate_inventory() == ()
    payload = inventory_json()
    assert payload
    assert len(payload) == len(inventory_by_id())
    assert len(payload) == len({row["mechanism_id"] for row in payload})


def test_test_only_and_dead_facilities_never_enter_readiness():
    candidates = {record.mechanism_id for record in readiness_candidates()}
    inventory = inventory_by_id()
    assert inventory["sense.typed_index"].status == STATUS_OPT_IN
    assert inventory["curriculum.stage_mastered"].status == STATUS_OPT_IN
    assert "sense.typed_index" not in candidates
    assert "curriculum.stage_mastered" not in candidates


def test_v05_post_weaning_stop_runner_remains_opt_in_and_not_ready():
    """V-05 只能给出分轨停止建议，不得形成正式断奶 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    record = inventory["evaluation.post_weaning_memory_ablation_stop"]

    assert record.status == STATUS_OPT_IN
    assert record.readiness_eligible is False
    assert record.writers and record.readers and record.recovery
    assert record.mechanism_id not in candidates


def test_k00_capability_negotiation_is_consumed_by_k02_repository():
    """K-00 契约被 K-02 正式仓库消费后可进入设施 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    capability = inventory["storage.backend_capability_negotiation"]

    assert capability.status == STATUS_PRODUCTION
    assert capability.readiness_eligible is True
    assert capability.writers and capability.readers
    assert "K-03" in capability.limitation
    assert capability.mechanism_id in candidates


def test_k01_placement_manifest_is_persisted_and_consumed_by_k02():
    """K-01 manifest 被真实发布、读取和恢复后可进入设施 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    placement = inventory["storage.placement_manifest_protocol"]

    assert placement.status == STATUS_PRODUCTION
    assert placement.readiness_eligible is True
    assert placement.writers and placement.readers
    assert placement.recovery
    assert "K-04" in placement.limitation
    assert placement.mechanism_id in candidates


def test_k02_tiered_segment_protocol_has_production_read_write_and_recovery():
    """K-02 台账必须列出正式构造、分页、迁移、淘汰和恢复边界。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    tiered = inventory["storage.tiered_segment_protocol"]

    assert tiered.status == STATUS_PRODUCTION
    assert tiered.readiness_eligible is True
    assert "storage:build_tiered_segment_store" in tiered.writers
    assert "BoundedSegmentReader.page" in " ".join(tiered.readers)
    assert tiered.recovery
    assert "K-03" in tiered.limitation and "K-04" in tiered.limitation
    assert tiered.mechanism_id in candidates


def test_k03_sharded_barrier_is_opt_in_until_formal_training_consumes_it():
    """K-03 runtime 已可恢复执行，但 W-01 正式 caller 前不得进入 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    barrier = inventory["training.sharded_barrier_protocol"]

    assert barrier.status == STATUS_OPT_IN
    assert barrier.readiness_eligible is False
    assert barrier.writers and barrier.readers and barrier.recovery
    assert "formal_train" in barrier.limitation
    assert "W-01" in barrier.limitation
    assert barrier.mechanism_id not in candidates


def test_k04_hot_set_mechanisms_are_registered_but_not_default_ready():
    """K-04 四项机制须列出真实消费和恢复，但默认 caller/profile 前不计 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    mechanism_ids = (
        "storage.rebuildable_segment_release",
        "storage.edge_budget_profile",
        "memory.candidate_projection_generation",
        "memory.query_hot_set_runtime",
    )
    for mechanism_id in mechanism_ids:
        record = inventory[mechanism_id]
        assert record.status == STATUS_OPT_IN
        assert record.readiness_eligible is False
        assert record.writers and record.readers
        assert mechanism_id not in candidates
    assert inventory["storage.rebuildable_segment_release"].recovery
    assert inventory["memory.candidate_projection_generation"].recovery
    assert inventory["memory.query_hot_set_runtime"].recovery
    assert "Pareto" in inventory["storage.edge_budget_profile"].limitation
    assert "A-10" in inventory["memory.query_hot_set_runtime"].limitation
    assert "M-09" in inventory["memory.query_hot_set_runtime"].limitation


def test_v03_recovery_is_production_but_not_storage_readiness():
    """V-03 已接入正式 dump/resume，但不得冒充 K-02 物理冷热完成。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    recovery = inventory["storage.recovery_package_protocol"]

    assert recovery.status == STATUS_PRODUCTION
    assert recovery.readiness_eligible is False
    assert "experiments.formal_train:formal_train" in recovery.writers
    assert "training.cursor:load_run_package" in recovery.readers
    assert "K-02" in recovery.limitation
    assert recovery.mechanism_id not in candidates


def test_m09_maintenance_is_opt_in_and_never_claims_physical_migration():
    """M-09 三轴维护已接线，但默认训练和 K 线迁移未完成时不计 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    maintenance = inventory["memory.maintenance_protocol"]

    assert maintenance.status == STATUS_OPT_IN
    assert maintenance.readiness_eligible is False
    assert maintenance.writers and maintenance.readers
    assert "install_memory_maintenance_runtime" in maintenance.gates
    assert "hint 不执行迁移" in maintenance.limitation
    assert maintenance.mechanism_id not in candidates


def test_v00_ledger_is_production_but_probe_runner_is_opt_in():
    """V-00 split ledger 可解锁隔离，领域 probe runner 未注入时不得计 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    ledger = inventory["evaluation.split_ledger"]
    runner = inventory["evaluation.unified_probe_runner"]
    assert ledger.status == STATUS_PRODUCTION
    assert ledger.readiness_eligible is True
    assert "evaluation-plan.json" in ledger.recovery
    assert runner.status == STATUS_OPT_IN
    assert runner.readiness_eligible is False
    assert "evaluation.split_ledger" in candidates
    assert "evaluation.unified_probe_runner" not in candidates


def test_word_form_production_chain_records_l02_candidates_and_h00_state():
    inventory = inventory_by_id()
    assert inventory["word_form.schema"].status == STATUS_PRODUCTION
    assert inventory["word_form.schema"].readiness_eligible is True
    assert inventory["word_form.collection"].status == STATUS_PRODUCTION
    assert inventory["word_form.collection"].writers
    segmentation = inventory["word_form.segmentation_fmm"]
    assert segmentation.status == STATUS_PRODUCTION
    assert segmentation.readiness_eligible is True
    candidates = inventory["word_form.segmentation_candidates"]
    assert candidates.status == STATUS_PRODUCTION
    assert candidates.writers
    hypothesis = inventory["hypothesis.evidence_protocol"]
    assert hypothesis.status == STATUS_PRODUCTION
    assert "Memory adapter" in hypothesis.limitation


def test_h01_prediction_is_formally_wired_but_not_default_readiness():
    """H-01 默认未注入协议时不得借通用预测设施计入课程 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    prediction = inventory["hypothesis.conditional_prediction"]
    assert prediction.status == STATUS_OPT_IN
    assert prediction.readiness_eligible is False
    assert "FormalTrainConfig.language_prediction_protocol" in prediction.gates
    assert "hypothesis.conditional_prediction" not in candidates


def test_h02_perturbation_is_test_only_until_real_ledger_consumer_exists():
    """H-02A/B 不得因通用 engine 和 typed adapter 存在就冒充正式生产链。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    perturbation = inventory["hypothesis.perturbation_evidence"]

    assert perturbation.status == STATUS_TEST_ONLY
    assert perturbation.readiness_eligible is False
    assert "hypothesis.perturbation_evidence" not in candidates
    assert any("RoleSwapPerturbationAdapter" in item
               for item in perturbation.readers)
    assert any("ScopeFlipPerturbationAdapter" in item
               for item in perturbation.readers)


def test_h03_description_length_waits_for_s02_real_candidates():
    """H-03 已有 H-04 adapter，但缺 S-02 真实候选时仍不得计 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    description_length = inventory["hypothesis.description_length"]

    assert description_length.status == STATUS_TEST_ONLY
    assert description_length.readiness_eligible is False
    assert "hypothesis.description_length" not in candidates


def test_h04_resolver_is_live_on_real_owners_and_enters_readiness():
    """H-04 必须有生产 writer/reader，且诚实记录跨 run 恢复边界。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    resolver = inventory["hypothesis.candidate_resolver"]

    assert resolver.status == STATUS_PRODUCTION
    assert resolver.readiness_eligible is True
    assert resolver.writers
    assert resolver.readers
    assert "hypothesis.candidate_resolver" in candidates
    assert "M-03" in resolver.limitation


def test_h05_typed_language_candidates_are_wired_but_protocol_gated():
    """H-05 正式 caller 已接线，但默认无课程协议时不得进入 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    structure = inventory["hypothesis.language_structure_candidates"]
    sense = inventory["sense.typed_index"]
    boundary = inventory["language.structure_boundary_evidence"]

    assert structure.status == STATUS_OPT_IN
    assert structure.writers and structure.readers
    assert sense.status == STATUS_OPT_IN
    assert sense.writers and sense.readers
    assert boundary.status == STATUS_OPT_IN
    assert boundary.writers and boundary.readers
    assert structure.mechanism_id not in candidates
    assert sense.mechanism_id not in candidates
    assert boundary.mechanism_id not in candidates
    assert "M-03" in structure.limitation
    assert "不再回退" in structure.limitation
    assert "M-03" in sense.limitation
    assert "M-03" in boundary.limitation


def test_h06_order_accumulation_is_opt_in_until_s02_typed_mapper():
    """H-06 已有正式配置 caller，但缺少 S-02 默认 mapper 时不得进入 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    order = inventory["hypothesis.order_accumulation"]

    assert order.status == STATUS_OPT_IN
    assert order.readiness_eligible is False
    assert order.writers
    assert order.readers
    assert "hypothesis.order_accumulation" not in candidates
    assert "S-02" in order.limitation


def test_s00_semantic_protocol_waits_for_s02_production_builder():
    """统一语义对象完成不等于生产语义学习，缺 S-02 时不得进入 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    semantic = inventory["semantic.atomic_proposition"]

    assert semantic.status == STATUS_TEST_ONLY
    assert semantic.readiness_eligible is False
    assert semantic.writers
    assert semantic.readers
    assert "semantic.atomic_proposition" not in candidates
    assert "S-02" in semantic.limitation


def test_s01_relation_algebra_waits_for_typed_production_adapters():
    """typed 关系代数有专项实现但无正式 caller 时不得进入 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    relation = inventory["semantic.typed_relation_algebra"]

    assert relation.status == STATUS_TEST_ONLY
    assert relation.readiness_eligible is False
    assert relation.writers
    assert relation.readers
    assert relation.mechanism_id not in candidates
    assert "旧边迁移" in relation.limitation


def test_r00_relation_closure_is_complete_infrastructure_not_learned_relation():
    """R-00 有全链 writer/reader，但具体关系未接时不得进入 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    closure = inventory["relation.closure_framework"]

    assert closure.status == STATUS_TEST_ONLY
    assert closure.readiness_eligible is False
    assert closure.writers and closure.readers
    assert "relation.closure_framework" not in candidates
    assert "R-09 多维编排已可用" in closure.limitation
    assert "M-03" in closure.limitation


def test_r02_typed_set_closure_is_opt_in_until_formal_course_data_exists():
    """R-02 有 formal/V-06/Use 闭环，但正式集合课程前不得进入 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    relation = inventory["relation.set_typed_closure"]

    assert relation.status == STATUS_OPT_IN
    assert relation.readiness_eligible is False
    assert relation.writers and relation.readers
    assert "language_set_relation_builder" in relation.gates[0]
    assert "显式 mapper" in relation.limitation
    assert "D-01" in relation.limitation
    assert relation.mechanism_id not in candidates


def test_r09_multi_verifier_orchestration_is_production_and_open_typed():
    """R-09 正式 caller 已多维执行，但未冒充事件时间或因果 verifier。"""
    inventory = inventory_by_id()
    record = inventory["verification.multi_dimension_orchestration"]
    candidates = {
        item.mechanism_id for item in readiness_candidates()
    }
    assert record.status == STATUS_PRODUCTION
    assert record.readiness_eligible is True
    assert record.writers and record.readers
    assert "verification.multi_dimension_orchestration" in candidates
    assert "occurrence-order" in record.limitation
    assert "causal" in record.limitation


def test_s02_semantic_builder_is_opt_in_but_waits_for_read_only_recovery():
    """S-02 已有正式课程 caller，但只读恢复和自动 mapper 前不得进入 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    builder = inventory["semantic.span_candidate_builder"]

    assert builder.status == STATUS_OPT_IN
    assert builder.readiness_eligible is False
    assert builder.writers
    assert builder.readers
    assert builder.mechanism_id not in candidates
    assert "formal semantic course" in builder.limitation
    assert "只读 query/recovery" in builder.limitation


def test_s03_typed_binding_is_opt_in_but_waits_for_logic_and_memory():
    """S-03 已有正式课程 caller，但逻辑和 Memory 未接时不得进入 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    binding = inventory["semantic.typed_binding_substitution"]

    assert binding.status == STATUS_OPT_IN
    assert binding.readiness_eligible is False
    assert binding.writers
    assert binding.readers
    assert binding.mechanism_id not in candidates
    assert "S-04" in binding.limitation
    assert "WorkMemory" in binding.limitation


def test_l05b2b_semantic_course_request_is_opt_in_and_not_ready():
    """首个课程切片已接 formal 请求，但只读和剩余消费者迁移前不计 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    course = inventory["semantic.language_course_generation_request"]

    assert course.status == STATUS_OPT_IN
    assert course.readiness_eligible is False
    assert len(course.writers) == 3
    assert len(course.readers) == 3
    assert "GenerationPlanningRequest" in course.sources
    assert "只读 query/recovery" in course.limitation
    assert "H2" in course.limitation
    assert "connector" in course.limitation
    assert course.mechanism_id not in candidates


def test_l05b2b_read_only_semantic_recovery_is_ground_only_and_not_ready():
    """只读恢复已支持 scoped Variable，但不得掩盖默认 mapper 与 M-03 缺口。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    recovery = inventory["semantic.language_query_recovery"]

    assert recovery.status == STATUS_OPT_IN
    assert recovery.readiness_eligible is False
    assert recovery.writers == ()
    assert recovery.readers
    assert "ActiveSenseCourseView" in recovery.sources
    assert "BindingEnvironment" in recovery.limitation
    assert "M-03" in recovery.limitation
    assert recovery.mechanism_id not in candidates

    template_scope = inventory["semantic.template_scope_graph"]
    assert template_scope.status == STATUS_OPT_IN
    assert template_scope.readiness_eligible is False
    assert template_scope.writers and template_scope.readers
    assert "ContextScope" in template_scope.sources
    assert "空 scope" in template_scope.limitation
    assert template_scope.mechanism_id not in candidates


def test_s07_structure_order_is_opt_in_until_mapper_and_generation_caller():
    """S-07 已有正式配置 caller，但不得替代默认语言 mapper 与 G-03 接线。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    order = inventory["semantic.structure_order_constraints"]

    assert order.status == STATUS_OPT_IN
    assert order.readiness_eligible is False
    assert order.writers
    assert order.readers
    assert order.mechanism_id not in candidates
    assert "formal_train" in order.limitation
    assert "G-03" in order.limitation


def test_r06_structure_closure_and_event_time_keep_honest_states():
    """结构与事件时间都有 opt-in caller，但真实课程缺失时不进 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    structure = inventory["relation.precedence_structure_closure"]
    event_time = inventory["relation.event_time_typed"]

    assert structure.status == STATUS_OPT_IN
    assert structure.readiness_eligible is False
    assert structure.writers and structure.readers
    assert "formal_train" in structure.readers[-1]
    assert "M-03" in structure.limitation
    assert event_time.status == STATUS_OPT_IN
    assert event_time.readiness_eligible is False
    assert event_time.writers and event_time.readers
    assert "same_round_S02_Proposition" in event_time.sources
    assert "training_hypothesis_event" in event_time.recovery
    assert "FormalTrainConfig.language_event_time_course" in event_time.gates
    assert "filtered verifier" in event_time.limitation
    assert "M-03" in event_time.limitation
    assert structure.mechanism_id not in candidates
    assert event_time.mechanism_id not in candidates


def test_r07_causal_closure_is_opt_in_but_not_ready_without_real_course_mapper():
    """typed causal 正式 caller 已接线，但默认无课程 mapper 时不得进入 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    causal = inventory["relation.causal_typed_closure"]

    assert causal.status == STATUS_OPT_IN
    assert causal.readiness_eligible is False
    assert causal.writers and causal.readers
    assert "FormalTrainConfig.language_causal_protocol" in causal.gates
    assert "FormalTrainConfig.language_causal_course" in causal.gates
    assert "formal_train" in causal.limitation
    assert "M-03" in causal.limitation
    assert "EDGE_CAUSES" in causal.limitation
    assert causal.mechanism_id not in candidates


def test_s04_logic_executor_waits_for_production_mapper_and_work_memory():
    """S-04 纯执行协议无真实课程 caller 时不得进入 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    executor = inventory["semantic.logic_executor"]

    assert executor.status == STATUS_TEST_ONLY
    assert executor.readiness_eligible is False
    assert executor.writers
    assert executor.readers
    assert executor.mechanism_id not in candidates
    assert "formal_train" in executor.limitation
    assert "WorkMemory" in executor.limitation
    assert "definitive truth" in executor.limitation


def test_s05_reasoning_planner_is_opt_in_through_a10_consumer_but_not_ready():
    """S-05 已有 A-10 typed consumer，但仍不能冒充逻辑学习或 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    planner = inventory["semantic.reasoning_planner"]

    assert planner.status == STATUS_OPT_IN
    assert planner.readiness_eligible is False
    assert planner.writers
    assert planner.readers
    assert planner.mechanism_id not in candidates
    assert "formal_train" in planner.limitation
    assert "A-10" in planner.limitation
    assert "sink" in planner.limitation
    assert "salience" in planner.limitation


def test_a10_attractor_agenda_is_registered_but_not_ready_by_itself():
    """A-10 有真实 consumer 和 M-08 caller，但仍无默认语言闭环。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    attractor = inventory["memory.query_attractor_agenda"]

    assert attractor.status == STATUS_OPT_IN
    assert attractor.readiness_eligible is False
    assert attractor.writers
    assert attractor.readers
    assert "AttractorProcessingTrace" in attractor.sources
    assert "M-08" in attractor.limitation
    assert "formal_train" in attractor.limitation
    assert attractor.mechanism_id not in candidates


def test_s06_formal_artifact_has_a06_and_test_only_capability_caller():
    """S-06 已接 A-06/C-02 专项，仍不得冒充生产能力或语言真值链。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    artifact = inventory["semantic.formal_artifact_bridge"]

    assert artifact.status == STATUS_OPT_IN
    assert artifact.readiness_eligible is False
    assert artifact.writers
    assert artifact.readers
    assert artifact.mechanism_id not in candidates
    assert "A-06" in artifact.limitation
    assert "formal_train" in artifact.limitation
    assert "C-02" in artifact.limitation
    assert "test-only" in artifact.limitation
    assert "definitive truth" in artifact.limitation


def test_c00_candidate_is_test_only_and_cannot_claim_readiness():
    """C-00 即使已有后续专项纵切，也不得越级计入 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    candidate = inventory["capability.provisional_candidate"]

    assert candidate.status == STATUS_TEST_ONLY
    assert candidate.readiness_eligible is False
    assert candidate.writers
    assert candidate.readers
    assert "CapabilityFormationInput" in candidate.sources
    assert "C-01" in candidate.limitation
    assert "C-02" in candidate.limitation
    assert "没有生产 caller" in candidate.limitation
    assert candidate.mechanism_id not in candidates


def test_c01_held_out_remains_test_only_after_c02_reuse():
    """C-02 可消费 C-01 报告，但不能把 C-01 提升为生产机制。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    held_out = inventory["capability.independent_held_out_verification"]

    assert held_out.status == STATUS_TEST_ONLY
    assert held_out.readiness_eligible is False
    assert held_out.writers
    assert held_out.readers
    assert "independent_expected_Artifact" in held_out.sources
    assert "V-06" in held_out.limitation
    assert "C-02" in held_out.limitation
    assert "无生产 caller" in held_out.limitation
    assert held_out.mechanism_id not in candidates


def test_c02_verified_memory_reuse_is_opt_in_and_not_ready():
    """F-01 实际 caller 允许 C-02 提升 opt-in，但仍排除 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    reuse = inventory["capability.verified_memory_reuse"]

    assert reuse.status == STATUS_OPT_IN
    assert reuse.readiness_eligible is False
    assert reuse.writers
    assert reuse.readers
    assert "CapabilityVerificationReport" in reuse.sources
    assert "SQLiteBackend_restart" in reuse.recovery
    assert "F-01" in reuse.limitation
    assert "opt-in" in reuse.limitation
    assert "readiness=false" in reuse.limitation
    assert reuse.mechanism_id not in candidates


def test_f01_facility_assembly_is_opt_in_and_preserves_phase_boundaries():
    """F-01 台账必须列出真实总装、恢复证据和禁止越界状态。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    facility = inventory["runtime.facility_readiness_assembly"]

    assert facility.status == STATUS_OPT_IN
    assert facility.readiness_eligible is False
    assert facility.writers
    assert facility.readers
    assert "FacilityExerciseMeasurement" in facility.sources
    assert "V-06 isolated Core fixture" in facility.recovery
    assert "V-03 cross-backend package" in facility.recovery
    assert "mastered/readiness" in facility.limitation
    assert "D-02/D-03/W-01" in facility.limitation
    assert "PH2" in facility.limitation
    assert facility.mechanism_id not in candidates


def test_occurrence_chain_records_source_scope_candidates_and_recovery():
    """L-03 台账必须同时列出来源原文、occurrence 和恢复链。"""
    inventory = inventory_by_id()
    source = inventory["language.source_record"]
    occurrence = inventory["language.occurrence"]
    assert source.status == STATUS_PRODUCTION
    assert source.scope == SCOPE_SHARED
    assert source.readiness_eligible is True
    assert "source_record" in source.recovery
    assert "text_assoc" in source.recovery
    assert "CompanionAssocIdentity" in source.sources
    assert any("SourceIntake.ensure" in writer for writer in source.writers)
    assert any("SourceIntake.read_slice" in reader for reader in source.readers)
    assert occurrence.status == STATUS_PRODUCTION
    assert occurrence.readiness_eligible is True
    assert "source_document_scope" in occurrence.sources
    assert "occurrence_candidate" in occurrence.recovery


def test_a05_source_trust_admission_is_opt_in_and_not_ready():
    """A-05 必须登记真实 caller、来源簇恢复和 readiness 边界。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    admission = inventory["memory.source_trust_admission"]

    assert admission.status == STATUS_OPT_IN
    assert admission.scope == "post_weaning"
    assert admission.readiness_eligible is False
    assert len(admission.writers) == 3
    assert len(admission.readers) == 5
    assert "zero-write batch preflight" in admission.gates
    assert "source_cluster_key" in admission.sources
    assert "source_trust_assessment" in admission.recovery
    assert "PW-00" in admission.limitation
    assert "opt-in/readiness=false" in admission.limitation
    assert admission.mechanism_id not in candidates


def test_typed_order_chain_replaces_legacy_precedes_readiness():
    """L-06 台账只允许分型顺序事实进入 readiness，旧宽边必须显式降级。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    order_fact = inventory["language.order_fact"]
    occurrence_order = inventory["language.occurrence_order"]
    legacy = inventory["relation.precedes"]
    assert order_fact.status == STATUS_PRODUCTION
    assert occurrence_order.status == STATUS_PRODUCTION
    assert order_fact.readiness_eligible is True
    assert occurrence_order.readiness_eligible is True
    assert "graph_statement" in order_fact.recovery
    assert "source_document_scope" in occurrence_order.sources
    assert legacy.status == STATUS_PRODUCTION
    assert legacy.scope == "legacy"
    assert legacy.readiness_eligible is False
    assert "typed owner" in legacy.limitation
    assert "默认 factory" in legacy.limitation
    assert "relation.precedes" not in candidates


def test_recursive_span_chain_records_full_members_and_recovery():
    """L-04 台账必须列出完整成员、候选生命周期和 dump 恢复链。"""
    record = inventory_by_id()["language.recursive_span"]
    assert record.status == STATUS_PRODUCTION
    assert record.readiness_eligible is True
    assert "segmentation_lattice" in record.sources
    assert "span" in record.recovery
    assert "span_member" in record.recovery
    assert "assertion_supersede" in record.recovery


def test_sentence_boundary_chain_uses_span_selection_and_recovery():
    """U-03 台账必须列出 active 选择消费者和无白名单边界。"""
    record = inventory_by_id()["language.sentence_boundary"]
    assert record.status == STATUS_PRODUCTION
    assert record.readiness_eligible is True
    assert "winner_token_span" in record.sources
    assert "committed_structure_support" in record.sources
    assert any("structure_boundary_runtime" in writer
               for writer in record.writers)
    assert "assertion_supersede" in record.recovery
    assert any("_item_sentence_bounds" in reader for reader in record.readers)


def test_memory_and_struct_bind_are_split_by_real_consumers():
    inventory = inventory_by_id()
    assert inventory["memory.reward_sink"].status == STATUS_PRODUCTION
    assert inventory["memory.reward_sink"].readiness_eligible is False
    assert inventory["memory.replay_seed"].status == STATUS_TEST_ONLY
    overlay = inventory["memory.overlay_relation_owner"]
    assert overlay.status == STATUS_PRODUCTION
    assert overlay.readiness_eligible is False
    assert overlay.writers and overlay.readers
    assert "M-05" in overlay.limitation
    assert "memory_overlay_relation" in overlay.recovery
    events = inventory["memory.event_schema"]
    assert events.status == STATUS_PRODUCTION
    assert events.readiness_eligible is False
    assert "memory_event" in events.recovery
    assert "ResolverDecision" in events.sources
    assert any("load_decisions" in item for item in events.readers)
    assert "H-04" in events.limitation
    assert "M-06" in events.limitation
    intake = inventory["memory.source_intake_protocol"]
    assert intake.status == STATUS_PRODUCTION
    assert intake.readiness_eligible is False
    assert "ParseFailureDraft" in intake.sources
    assert "memory_event" in intake.recovery
    batch = inventory["memory.batch_recovery_protocol"]
    assert batch.status == STATUS_PRODUCTION
    assert batch.readiness_eligible is False
    assert "group commit" in batch.sources
    assert "memory_event_batch_link" in batch.recovery
    assert "batch_id 索引" in batch.limitation
    assert "K-03" in batch.limitation
    isolation = inventory["memory.object_isolation_export_forget"]
    assert isolation.status == STATUS_PRODUCTION
    assert isolation.readiness_eligible is False
    assert "MemoryManagementContext" in isolation.sources
    assert "memory_forget_set_segment" in isolation.recovery
    assert "SQLite 重启" in isolation.limitation
    assert "J-M/J-F1" in isolation.limitation
    query = inventory["memory.current_input_query_compiler"]
    assert query.status == STATUS_OPT_IN
    assert query.readiness_eligible is False
    assert query.writers and query.readers
    assert "M-07 已" in query.limitation
    assert "上一 reward" in query.limitation
    resolver = inventory["memory.core_memory_overlay_resolver"]
    assert resolver.status == STATUS_OPT_IN
    assert resolver.readiness_eligible is False
    assert resolver.writers and resolver.readers
    assert "dirty aggregate" in resolver.limitation
    assert "M-08" in resolver.limitation
    source_to_use = inventory["memory.source_to_use"]
    assert source_to_use.status == STATUS_OPT_IN
    assert source_to_use.readiness_eligible is False
    assert source_to_use.writers and source_to_use.readers
    assert "AttractorProcessingTrace" in source_to_use.sources
    assert "formal_train" in source_to_use.limitation
    assert "事实正确" in source_to_use.limitation
    outcome_bridge = inventory["memory.generation_use_outcome_bridge"]
    assert outcome_bridge.status == STATUS_OPT_IN
    assert outcome_bridge.readiness_eligible is False
    assert outcome_bridge.writers and outcome_bridge.readers
    assert "TypedLanguageRewardSignal" in outcome_bridge.sources
    assert "scalar reward" in outcome_bridge.limitation
    assert "definitive truth" in outcome_bridge.limitation
    assert inventory["struct_bind.boot"].status == STATUS_OPT_IN
    assert inventory["struct_bind.generate_consumer"].status == STATUS_DEAD
    artifact_consumer = inventory["struct_bind.typed_artifact_consumer"]
    assert artifact_consumer.status == STATUS_OPT_IN
    assert artifact_consumer.readiness_eligible is False
    assert artifact_consumer.writers and artifact_consumer.readers
    assert "ArtifactBindingChoice" in artifact_consumer.sources
    assert "C-02" in artifact_consumer.limitation
    assert "M-08 Use" in artifact_consumer.limitation


def test_g00_generation_plan_is_opt_in_until_real_structure_mappers_exist():
    """G-00 已接语义请求和 G-04，但真实结构 mapper 前不得计 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    plan = inventory["generation.typed_plan"]
    surface = inventory["generation.proposition_to_surface"]

    assert plan.status == STATUS_OPT_IN
    assert plan.readiness_eligible is False
    assert plan.writers and not plan.readers
    assert "ReasoningPlanResult" in plan.sources
    assert "L-05B2B" in plan.limitation
    assert "G-04" in plan.limitation
    assert "只读恢复" in plan.limitation
    assert plan.mechanism_id not in candidates
    assert surface.status == STATUS_DEAD
    assert "G-00" in surface.limitation


def test_g05_question_runtime_is_opt_in_but_not_ready():
    """G-05 对话 caller 已接线，但正式课程与断奶产物前不得进入 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    runtime = inventory["question.typed_answer_generation_runtime"]

    assert runtime.status == STATUS_OPT_IN
    assert runtime.readiness_eligible is False
    assert len(runtime.writers) == 4
    assert len(runtime.readers) == 8
    assert "MemoryGenerationEvidence" in runtime.sources
    assert "MemoryGenerationCommitReport" in runtime.sources
    assert "MemoryGenerationOutcomeReport" in runtime.sources
    assert "G-05" in runtime.limitation
    assert "opt-in" in runtime.limitation
    assert "分维 outcome" in runtime.limitation
    assert "J-F2" in runtime.limitation
    assert runtime.mechanism_id not in candidates


def test_pw00_dry_run_runtime_is_opt_in_but_not_ready():
    """PW-00 有真实 dry-run caller 和恢复链，但 PW-00A 前不得进入 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    runtime = inventory["runtime.post_weaning_dry_run"]

    assert runtime.status == STATUS_OPT_IN
    assert runtime.scope == "post_weaning"
    assert runtime.readiness_eligible is False
    assert len(runtime.writers) == 2
    assert len(runtime.readers) == 3
    assert "PostWeaningDryRunManifest" in runtime.gates
    assert "V-03 recovery package" in runtime.recovery
    assert "PW-00A" in runtime.limitation
    assert "readiness=true" in runtime.limitation
    assert runtime.mechanism_id not in candidates


def test_g01_answer_content_is_opt_in_until_real_policy_exists():
    """G-01 可经 formal opt-in caller 执行，但真实内容策略前仍不进入 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    content = inventory["generation.answer_content_selection"]

    assert content.status == STATUS_OPT_IN
    assert content.readiness_eligible is False
    assert len(content.writers) == 3
    assert not content.readers
    assert "ArtifactInvocationResult" in content.sources
    assert "L-05B2B" in content.limitation
    assert "G-04" in content.limitation
    assert "真实内容策略" in content.limitation
    assert content.mechanism_id not in candidates


def test_g02_structure_plan_is_opt_in_until_default_and_multi_mapper_exist():
    """G-02 已接单命题 connector，但默认模板和多命题 mapper 前不计 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    structure = inventory["generation.discourse_proposition_syntax"]

    assert structure.status == STATUS_OPT_IN
    assert structure.readiness_eligible is False
    assert len(structure.writers) == 6
    assert not structure.readers
    assert "StructureSlotDefinition" in structure.sources
    assert "单命题 connector" in structure.limitation
    assert "多命题 discourse" in structure.limitation
    assert "旧结构链全局退役" in structure.limitation
    assert structure.mechanism_id not in candidates


def test_l05b1_structure_execution_bridge_is_opt_in_but_not_ready():
    """active S-07 已被 connector 正式消费，但全局退役前不计 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    bridge = inventory["generation.structure_order_execution_bridge"]

    assert bridge.status == STATUS_OPT_IN
    assert bridge.readiness_eligible is False
    assert bridge.writers
    assert len(bridge.readers) == 3
    assert "StructureOrderProjection" in bridge.sources
    assert "单命题 connector" in bridge.limitation
    assert "全局退役" in bridge.limitation
    assert bridge.mechanism_id not in candidates


def test_g03_typed_surface_is_opt_in_but_waits_for_real_mapper_and_b2b():
    """G-03 已接 R-01 fixture 课程，但正式资料和全局退役前不计 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    surface = inventory["generation.typed_surface_linearization"]

    assert surface.status == STATUS_OPT_IN
    assert surface.readiness_eligible is False
    assert len(surface.writers) == 7
    assert len(surface.readers) == 4
    assert "Representation" in surface.sources
    assert "L-05B2" in surface.limitation
    assert "单命题 connector" in surface.limitation
    assert "正式 D-01" in surface.limitation
    assert surface.mechanism_id not in candidates


def test_l05b2b_connector_is_single_proposition_opt_in_and_not_ready():
    """connector 已有默认 loader，但多命题和全局退役前不计 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    connector = inventory["generation.semantic_structure_alias_connector"]

    assert connector.status == STATUS_OPT_IN
    assert connector.readiness_eligible is False
    assert len(connector.writers) == 5
    assert len(connector.readers) == 5
    assert "StructureOrderProjection" in connector.sources
    assert "MinimalInstruction" in connector.sources
    assert "单命题" in connector.limitation
    assert "Unicode" in connector.limitation
    assert "版本化 connector/R-01/G-04 课程 loader" in connector.limitation
    assert "PH2 Core Use 恢复" in connector.limitation
    assert "production builder" in connector.limitation
    assert connector.mechanism_id not in candidates


def test_l05b2b_connector_candidate_lifecycle_is_opt_in_and_not_ready():
    """connector 候选和默认 loader 已恢复，但独立来源和全局退役仍缺。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    lifecycle = inventory["generation.connector_candidate_lifecycle"]

    assert lifecycle.status == STATUS_OPT_IN
    assert lifecycle.readiness_eligible is False
    assert len(lifecycle.writers) == 5
    assert len(lifecycle.readers) == 10
    assert "TypedLanguageRewardSignal" in lifecycle.sources
    assert "candidate_projection_event" in lifecycle.recovery
    assert "memory_event" in lifecycle.recovery
    assert "M03_Hypothesis_Evidence_Resolution" in lifecycle.recovery
    assert "refute 优先" in lifecycle.limitation
    assert "PH2 Core" in lifecycle.limitation
    assert "派生恢复" in lifecycle.limitation
    assert "内容锁课程 loader" in lifecycle.limitation
    assert "唯一 exact forming 才 trial" in lifecycle.limitation
    assert "round 内不扫描历史" in lifecycle.limitation
    assert lifecycle.mechanism_id not in candidates


def test_l05b2a_typed_production_bridge_is_opt_in_and_not_ready():
    """B2A 已接默认 builder，但缺具体独立来源与消费者全局迁移。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    bridge = inventory["generation.typed_production_bridge"]

    assert bridge.status == STATUS_OPT_IN
    assert bridge.readiness_eligible is False
    assert len(bridge.writers) == 4
    assert len(bridge.readers) == 4
    assert "GenerationSurfacePreview" in bridge.sources
    assert "GenerationSurfacePlan" in bridge.sources
    assert "G-04" in bridge.limitation
    assert "真实默认 builder" in bridge.limitation
    assert "真实语言 parser/verifier" in bridge.limitation
    assert bridge.mechanism_id not in candidates


def test_g04_post_surface_verification_is_opt_in_and_read_only():
    """G-04 已有版本化设施课程，但无真实语言组件且不计 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    postcheck = inventory["generation.post_surface_verification"]

    assert postcheck.status == STATUS_OPT_IN
    assert postcheck.readiness_eligible is False
    assert len(postcheck.writers) == 3
    assert len(postcheck.readers) == 8
    assert "GenerationSurfaceParseRequest" in postcheck.sources
    assert "RelationSchema" in postcheck.sources
    assert "VerificationReport" in postcheck.sources
    assert "graph_statement" in postcheck.recovery
    assert "V06_cloned_postcheck_owner" in postcheck.recovery
    assert "不写 effect" in postcheck.limitation
    assert "不再暴露" in postcheck.limitation
    assert "设施 fixture" in postcheck.limitation
    assert "G-05" in postcheck.limitation
    assert postcheck.mechanism_id not in candidates


def test_l05b2b_typed_episode_keeps_dimensions_and_legacy_metrics_separate():
    """typed episode 已有生产消费者，但 H2/floor/connector 前不计 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    episode = inventory["generation.typed_language_episode_reward"]

    assert episode.status == STATUS_OPT_IN
    assert episode.readiness_eligible is False
    assert len(episode.writers) == 2
    assert len(episode.readers) == 6
    assert "VerificationResult" in episode.sources
    assert "supplemental_VerificationReport" in episode.sources
    assert "绝不合成 reward:int" in episode.limitation
    assert "H2" in episode.limitation
    assert "floor" in episode.limitation
    assert "connector" in episode.limitation
    assert "adapter artifact" in episode.limitation
    assert episode.mechanism_id not in candidates


def test_l05b2b_typed_h2_is_development_only_and_not_scalarized():
    """typed H2 已阻断正式阶段3，但默认期望缺失前不计 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    h2 = inventory["evaluation.typed_language_h2"]

    assert h2.status == STATUS_OPT_IN
    assert h2.readiness_eligible is False
    assert len(h2.writers) == 1
    assert len(h2.readers) == 1
    assert "V00_development_split" in h2.sources
    assert "development split" in h2.limitation
    assert "held-out" in h2.limitation
    assert "JudgeWeights" in h2.limitation
    assert h2.mechanism_id not in candidates


def test_l05b2b_typed_floor_is_held_out_and_dimension_gated():
    """typed floor 已替换旧生成 floor，但默认 case/阈值前不计 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    floor = inventory["evaluation.typed_language_floor"]

    assert floor.status == STATUS_OPT_IN
    assert floor.readiness_eligible is False
    assert "V00_held_out_split" in floor.sources
    assert "held-out split" in floor.limitation
    assert "legacy OutputResult" in floor.limitation
    assert "综合均值" in floor.limitation
    assert floor.mechanism_id not in candidates


def test_r01_typed_alias_replaces_legacy_boot_as_readiness_owner():
    """旧宽边继续兼容，typed R-01 课程仍因正式资料缺失而不计 readiness。"""
    inventory = inventory_by_id()
    candidates = {record.mechanism_id for record in readiness_candidates()}
    legacy = inventory["relation.alias_legacy_boot"]
    typed = inventory["relation.alias"]

    assert legacy.status == STATUS_PRODUCTION
    assert legacy.readiness_eligible is False
    assert "PURE_ALIAS" in legacy.limitation
    assert typed.status == STATUS_OPT_IN
    assert typed.readiness_eligible is False
    assert len(typed.writers) == 5
    assert len(typed.readers) == 8
    assert any("lookup_atomic_by_binding" in item for item in typed.readers)
    assert any("discover_surface" in item for item in typed.readers)
    assert "AliasRouteSearchBudget" in typed.sources
    assert "AliasRelationCourseManifest" in typed.sources
    assert "Representation" in typed.sources
    assert "PH2 Core" in typed.limitation
    assert "D-01" in typed.limitation
    assert typed.mechanism_id not in candidates


def test_ph2_relation_use_is_core_event_and_not_memory_use():
    """PH2 Use 必须从 Core 图恢复，不能与 PH3 Memory UsePayload 混为一账。"""
    record = inventory_by_id()["relation.core_use_event"]
    candidates = {item.mechanism_id for item in readiness_candidates()}

    assert record.status == STATUS_OPT_IN
    assert record.readiness_eligible is False
    assert len(record.writers) == 4
    assert len(record.readers) == 4
    assert "RelationUseContext" in record.sources
    assert "graph_statement" in record.recovery
    assert "PH2" in record.limitation
    assert "UsePayload" in record.limitation
    assert record.mechanism_id not in candidates


def test_a01_occurrence_reference_is_opt_in_and_not_readiness():
    """A-01 有真实 F-00 消费者，但默认 caller、A-02 和跨 run 恢复前不计 readiness。"""
    record = inventory_by_id()["semantic.occurrence_reference_resolution"]
    candidates = {item.mechanism_id for item in readiness_candidates()}

    assert record.status == STATUS_OPT_IN
    assert record.readiness_eligible is False
    assert len(record.writers) == 3
    assert len(record.readers) == 3
    assert "speaker/time/context Evidence" in record.sources
    assert "legacy ConceptRef" in record.limitation
    assert "A-02" in record.limitation
    assert record.mechanism_id not in candidates


def test_a02_typed_work_memory_is_opt_in_and_not_long_term_memory():
    """A-02 已有 typed adapter，但无默认 mapper 且内容不得冒充长期 Memory。"""
    record = inventory_by_id()["work_memory.typed_content_state"]
    candidates = {item.mechanism_id for item in readiness_candidates()}

    assert record.status == STATUS_OPT_IN
    assert record.readiness_eligible is False
    assert len(record.writers) == 4
    assert len(record.readers) == 3
    assert "Occurrence graph identity" in record.sources
    assert "explicit supersede" in record.sources
    assert "produced_refs/FIFO" in record.limitation
    assert "长期 Memory" in record.limitation
    assert record.mechanism_id not in candidates


def test_a03_parser_revision_is_opt_in_and_does_not_rebuild_memory():
    """A-03 已有跨版本 Core 修正，但默认 caller 和 A-08 前不得计 readiness。"""
    record = inventory_by_id()["semantic.parser_revision"]
    candidates = {item.mechanism_id for item in readiness_candidates()}

    assert record.status == STATUS_OPT_IN
    assert record.readiness_eligible is False
    assert len(record.writers) == 4
    assert len(record.readers) == 3
    assert "complete_old_active_competition" in record.sources
    assert "H04 decision history" in record.recovery
    assert "A-08" in record.limitation
    assert "post-weaning" in record.limitation
    assert record.mechanism_id not in candidates


def test_a08_memory_revision_rebuild_is_opt_in_and_preserves_history():
    """A-08 已有长期派生协调器，但默认 caller 前不得进入 readiness。"""
    record = inventory_by_id()["memory.parser_revision_rebuild"]
    candidates = {item.mechanism_id for item in readiness_candidates()}

    assert record.status == STATUS_OPT_IN
    assert record.readiness_eligible is False
    assert len(record.writers) == 3
    assert len(record.readers) == 5
    assert "A03_many_to_many_mapping" in record.sources
    assert "old_Use_and_output_event" in record.recovery
    assert "formal_train" in record.limitation
    assert "readiness" in record.limitation
    assert record.mechanism_id not in candidates


def test_learned_relation_cue_has_writer_reader_and_recovery_path():
    record = inventory_by_id()["d11.relation_cue_generation"]
    assert record.status == STATUS_PRODUCTION
    assert record.writers
    assert any("relation_cue_candidates" in reader for reader in record.readers)
    assert "edge" in record.recovery


def test_required_audit_domains_are_present():
    ids = set(inventory_by_id())
    required = {
        "evaluation.split_ledger",
        "evaluation.unified_probe_runner",
        "storage.backend_capability_negotiation",
        "storage.placement_manifest_protocol",
        "hypothesis.conditional_prediction",
        "hypothesis.language_structure_candidates",
        "word_form.schema",
        "language.source_record",
        "language.occurrence",
        "language.order_fact",
        "language.occurrence_order",
        "relation.precedence_structure_closure",
        "relation.event_time_typed",
        "relation.causal_typed_closure",
        "language.recursive_span",
        "language.sentence_boundary",
        "language.structure_boundary_evidence",
        "curriculum.relation_plan",
        "d11.structure_tally_promote",
        "realizes.is_a_causes_label",
        "relation.alias",
        "language.signal_seed_graph",
        "relation.core_use_event",
        "relation.subset_eq_universal",
        "relation.existential",
        "semantic.typed_relation_algebra",
        "semantic.span_candidate_builder",
        "semantic.typed_binding_substitution",
        "semantic.language_course_generation_request",
        "semantic.language_query_recovery",
        "semantic.template_scope_graph",
        "semantic.logic_executor",
        "semantic.reasoning_planner",
        "generation.typed_plan",
        "question.typed_answer_generation_runtime",
        "runtime.post_weaning_dry_run",
        "generation.answer_content_selection",
        "generation.discourse_proposition_syntax",
        "generation.structure_order_execution_bridge",
        "generation.typed_surface_linearization",
        "generation.semantic_structure_alias_connector",
        "generation.connector_candidate_lifecycle",
        "generation.typed_production_bridge",
        "generation.post_surface_verification",
        "generation.typed_language_episode_reward",
        "evaluation.typed_language_h2",
        "evaluation.typed_language_floor",
        "semantic.formal_artifact_bridge",
        "relation.property",
        "relation.mereology_ingest",
        "relation.antonym_ingest",
        "relation.similar",
        "relation.precedes",
        "relation.causes",
        "logic.negation_modal_d11",
        "memory.source_to_use",
        "memory.generation_use_outcome_bridge",
        "memory.overlay_relation_owner",
        "memory.event_schema",
        "memory.source_intake_protocol",
        "memory.source_trust_admission",
        "memory.current_input_query_compiler",
        "memory.core_memory_overlay_resolver",
        "struct_bind.typed_artifact_consumer",
        "struct_bind.generate_consumer",
        "generation.proposition_to_surface",
    }
    assert required <= ids
