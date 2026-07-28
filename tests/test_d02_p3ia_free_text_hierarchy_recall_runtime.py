"""P3-Ia 中文 raw 层级、Evidence center、K-04 recall 与 QA 生产纵切。"""
from __future__ import annotations

import gzip
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.attractor_state import AttractorDependency
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    OwnerScope,
    SourceRef,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    context_scope_identity,
    proposition_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    BindingFailureProtocol,
    PropositionSubstituter,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
    SubstitutionProtocol,
)
from pure_integer_ai.experiments.free_text_hierarchy_recall_evaluator import (
    FreeTextProductionReport,
    IndependentFreeTextHierarchyRecallEvaluator,
)
from pure_integer_ai.experiments.free_text_hierarchy_runtime import (
    MechanicalTextHierarchyFormer,
)
from pure_integer_ai.experiments.free_text_recall_runtime import (
    AclFirstExactRecallReader,
    FreeTextRecallRun,
    FreeTextRecallRuntime,
    LearnedEvidenceCenterFormer,
    LearnedSurfaceFeatureMatcher,
    RecallIndexEntry,
    RecalledFactQuestionExecutor,
    TypedRecallPayload,
    TypedRecallRecordCodec,
    encode_surface_feature_payload,
)
from pure_integer_ai.experiments.free_text_revision_runtime import (
    FreeTextDerivedDependency,
    FreeTextRevisionInvalidator,
)
from pure_integer_ai.experiments.ph2_authored_free_text_hierarchy_recall_course import (
    COURSE_MANIFEST_PATH,
    PACK_NAME,
    SAMPLE_RELATIVE_PATH,
    build_free_text_hierarchy_recall_course_manifest,
    compile_authored_free_text_hierarchy_recall_course,
    default_free_text_hierarchy_recall_sample_bytes,
    read_authored_free_text_hierarchy_recall_seeds,
)
from pure_integer_ai.experiments.ph2_capability_course_contract import (
    read_capability_course_manifest,
    write_capability_course_manifest,
)
from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.experiments.ph2_free_text_hierarchy_recall_contract import (
    RecallBudget,
    SourceDocument,
)
from pure_integer_ai.storage import build_storage_role_registry
from pure_integer_ai.storage.memory_aggregate import (
    MEMORY_AGGREGATE_STORAGE_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.memory_event import (
    MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.memory_query_projection import (
    MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.placement import TemperatureProfile, TemperatureTier
from pure_integer_ai.storage.query_hot_set import (
    QueryHotSetPolicy,
    QueryPrefetchContext,
)
from pure_integer_ai.storage.sealed_segment import OpenHotDelta, SegmentBudget
from pure_integer_ai.storage.segment_dependency import SegmentDependency
from pure_integer_ai.storage.segment_repository import (
    InMemoryObjectRepository,
    OBJECT_KIND_SEGMENT,
)
from pure_integer_ai.storage.tiered_segment_store import TieredSegmentStore
from tests.test_d02_md03_directional_center_adapter import (
    _close as _close_md03,
    _fixture as _md03_fixture,
)
from tests.test_f00_question_answer_runtime import _fixture as _qa_fixture


_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / SAMPLE_RELATIVE_PATH
_FEATURE_REASON = (95100, 1)
_DEPENDENCIES = (
    SegmentDependency(
        MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY, (95101, 1), (95102, 1)),
    SegmentDependency(
        MEMORY_AGGREGATE_STORAGE_DESCRIPTOR_KEY, (95101, 2), (95102, 2)),
)
_PROFILE = TemperatureProfile(
    (95103, 1),
    (
        TemperatureTier((95103, 1), 0),
        TemperatureTier((95103, 2), 1),
    ),
)


class _NeverPrefetch:
    """聚焦测试明确关闭预取，使 exact query 物理计数可直接裁决。"""

    def should_prefetch(self, context: QueryPrefetchContext) -> bool:
        """只核验上下文类型并稳定返回 False。"""
        if not isinstance(context, QueryPrefetchContext):
            raise TypeError("prefetch context 类型错误")
        return False

    def state_key(self) -> tuple[int, ...]:
        """返回注入策略的稳定身份。"""
        return (95104, 1)


class _CountingRepository(InMemoryObjectRepository):
    """只统计 sealed segment 读取，供 ACL-before-payload 反例使用。"""

    def __init__(self) -> None:
        super().__init__()
        self.segment_reads = 0

    def get(self, object_kind: int, identity_key: tuple[int, ...]) -> bytes:
        """委托真实 repository，同时单独累计 segment get。"""
        if object_kind == OBJECT_KIND_SEGMENT:
            self.segment_reads += 1
        return super().get(object_kind, identity_key)


def _bound_target(source: SourceRef):
    """从来源化 typed 身份建立一个不含 surface 规则的 BoundProposition。"""
    failures = BindingFailureProtocol(*tuple(
        minimal_instruction_identity((95110, index))
        for index in range(1, 10)
    ))
    definition = AtomicPropositionDefinition(
        proposition_identity(source, (95111, 1)),
        concept_identity((95111, 2)),
        occurrence_identity(source, start=1, end=2, ordinal=0),
        context_scope_identity(source, (95111, 3)),
        (),
    )
    graph = PropositionTemplateGraph((ScopedPropositionTemplate(
        definition,
        structure_concept_identity((95111, 4)),
    ),))
    return PropositionSubstituter(SubstitutionProtocol(
        minimal_instruction_identity((95111, 5)), failures,
    )).substitute(
        definition.proposition, graph, BindingEnvironment())


def _publish_record(record):
    """把唯一 typed recall record 发布到真实 TieredSegmentStore 冷层。"""
    repository = _CountingRepository()
    store = TieredSegmentStore(
        repository, build_storage_role_registry(), _PROFILE)
    delta = OpenHotDelta(
        MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
        (95120, 1),
        _DEPENDENCIES,
        SegmentBudget(4, 1_000_000),
    )
    delta.append(record)
    store.publish_delta(
        delta,
        segment_key=(95121, 1),
        tier_key=(95103, 2),
        read_fence=1,
        manifest_key=(95122, 1),
        migration_key=(95123, 1),
    )
    repository.segment_reads = 0
    return store, repository


@dataclass
class _RuntimeFixture:
    """保存一次 held-out 候选投影形成的真实 runtime 与关闭句柄。"""

    seed: object
    hierarchy: object
    recall: FreeTextRecallRun
    qa: object
    source: SourceRef
    target: object
    response_scope: object
    repository: _CountingRepository
    reader: AclFirstExactRecallReader
    runtime: FreeTextRecallRuntime
    history: tuple[EvidenceRecord, ...]
    index: tuple[RecallIndexEntry, ...]
    access: MemoryAccessContext
    budget: RecallBudget
    raw_query: str
    md03: dict
    qa_run: object

    def close(self) -> None:
        """关闭 QA surface owner 和 MD-03 query/backend。"""
        self.qa.close()
        _close_md03(self.md03)


def _runtime_fixture() -> _RuntimeFixture:
    """仅从 held-out candidate projection 和 source payload 执行生产闭环。"""
    seed = read_authored_free_text_hierarchy_recall_seeds(_SAMPLE)[8]
    candidate = seed.observation_payload().to_value()
    document_value = candidate["document"]
    query_value = candidate["query"]
    source = SourceRef.from_stable_key(tuple(document_value["source_ref_key"]))
    target = _bound_target(source)
    evidence_scope = document_scope(source)
    response_scope = query_scope(1, parent=evidence_scope)
    target_hypothesis = HypothesisKey(
        (95130, 1),
        target.template.stable_key(),
        (95130, 2),
        evidence_scope,
        source,
    )
    target_evidence = EvidenceRecord(
        95131,
        target_hypothesis,
        EVIDENCE_SUPPORT,
        (95130, 3),
        source,
        1,
    )
    dependency_keys = (
        StableRecordKey((95132, 1)),
        StableRecordKey((95132, 2)),
        StableRecordKey((95132, 3)),
    )
    answer_surface = "竹台"
    citation_start = document_value["raw_text"].index(answer_surface)
    record_key = tuple(candidate["recall_index"][0]["record_key"])
    payload = TypedRecallPayload(
        target,
        (target_evidence,),
        citation_start,
        citation_start + len(answer_surface),
        dependency_keys,
    )
    record = TypedRecallRecordCodec.encode(record_key, payload)
    store, repository = _publish_record(record)
    history = tuple(EvidenceRecord(
        item["evidence_id"],
        HypothesisKey(
            (95133, 1),
            (95133, item["evidence_id"]),
            (95133, 2),
            evidence_scope,
            source,
        ),
        EVIDENCE_SUPPORT,
        _FEATURE_REASON,
        source,
        item["evidence_id"],
        encode_surface_feature_payload(
            item["surface"], StableRecordKey((item["feature_key"],))),
    ) for item in candidate["allowed_history"])
    index = (RecallIndexEntry(
        record_key,
        source,
        evidence_scope,
        tuple(StableRecordKey((value,))
              for value in candidate["recall_index"][0]["feature_keys"]),
        dependency_keys,
    ),)
    md03 = _md03_fixture()
    matcher = LearnedSurfaceFeatureMatcher(_FEATURE_REASON)
    center_former = LearnedEvidenceCenterFormer(matcher, md03["adapter"])
    reader = AclFirstExactRecallReader(
        store,
        MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
        QueryHotSetPolicy(
            SegmentBudget(4, 1_000_000),
            SegmentBudget(2, 500_000),
            _NeverPrefetch(),
            8,
        ),
    )
    budget_value = candidate["recall_budget"]
    budget = RecallBudget(
        budget_value["max_index_gets"],
        budget_value["max_segment_payload_gets"],
        budget_value["max_segment_payload_bytes"],
        budget_value["max_results"],
    )
    runtime = FreeTextRecallRuntime(center_former, reader)
    access = MemoryAccessContext(
        source.owner.tenant_id,
        source.owner.user_id,
        source.owner.session_id,
    )
    recall = runtime.resolve(
        query_value["raw_text"],
        history,
        md03["current"],
        index,
        access,
        budget,
        reader_key=(95140, 1),
    )
    qa = _qa_fixture(
        world=(source, response_scope, target),
        executor_factory=lambda route: RecalledFactQuestionExecutor(
            recall,
            route=route,
            executed_reason=minimal_instruction_identity((95141, 1)),
            trace_prefix=(95141, 2),
        ),
        answer_text="事实",
    )
    qa_run = qa.runtime.run(qa.request)
    document = SourceDocument(
        source,
        StableRecordKey((95150, 1)),
        StableRecordKey((95150, 2)),
        StableRecordKey((95150, 3)),
        document_value["raw_text"],
        document_value["raw_sha256"],
    )
    hierarchy = MechanicalTextHierarchyFormer().form(document)
    return _RuntimeFixture(
        seed, hierarchy, recall, qa, source, target, response_scope,
        repository, reader, runtime, history, index, access, budget,
        query_value["raw_text"], md03, qa_run)


def test_course_sample_is_canonical_split_complete_and_label_private():
    """课程样本必须确定、覆盖完整且 candidate 投影没有正确结果字段。"""
    assert _SAMPLE.read_bytes() == default_free_text_hierarchy_recall_sample_bytes()
    seeds = read_authored_free_text_hierarchy_recall_seeds(_SAMPLE)
    assert len(seeds) == 18
    assert {item.split for item in seeds} == {"held_out", "train"}
    assert any(item.supersedes_seed_id for item in seeds)
    forbidden = {
        "answer", "boundary", "center", "citation", "evaluator",
        "expected", "label", "paragraph", "proposition", "span",
    }

    def keys(value):
        """递归收集 candidate payload 字段名。"""
        if isinstance(value, dict):
            return {str(key).casefold() for key in value} | {
                nested
                for child in value.values()
                for nested in keys(child)
            }
        if isinstance(value, list):
            return {nested for child in value for nested in keys(child)}
        return set()

    assert all(not any(token in key for token in forbidden)
               for seed in seeds
               for key in keys(seed.observation_payload().to_value()))
    assert all(seed.expected_payload.to_value()["center_record_keys"]
               for seed in seeds)


def test_course_compiles_to_four_owner_pack_and_non_overwriting_manifest(tmp_path):
    """正式编译必须物理分账，manifest 保持 COURSE_FROZEN/零执行。"""
    build = compile_authored_free_text_hierarchy_recall_course(
        _SAMPLE, tmp_path / "release")
    assert build.validation.observation_count == 18
    assert build.validation.teacher_evidence_count > 0
    assert build.validation.evaluator_label_count > 0
    assert build.pack_root.name == PACK_NAME
    manifest = build_free_text_hierarchy_recall_course_manifest(_SAMPLE, build)
    target = tmp_path / COURSE_MANIFEST_PATH
    write_capability_course_manifest(manifest, target)
    assert read_capability_course_manifest(target) == manifest
    assert manifest.course_status == "COURSE_FROZEN"
    assert manifest.runtime_status == "NOT_STARTED"
    assert manifest.execution_state.to_value()["formal_training_runs"] == 0
    with gzip.open(build.pack_root / "observations/held_out.jsonl.gz", "rb") as handle:
        held_out = handle.read()
    assert b"expected_payload" not in held_out
    assert b"center_record_keys" not in held_out
    with pytest.raises(Exception, match="已存在|覆盖"):
        compile_authored_free_text_hierarchy_recall_course(
            _SAMPLE, tmp_path / "release")


def test_runtime_forms_hierarchy_center_pages_exact_record_and_runs_real_qa():
    """held-out raw query 必须经 MD-03/K-04 进入现有 QA/G-00..G-03 consumer。"""
    fixture = _runtime_fixture()
    try:
        assert fixture.hierarchy.private_label_read_count == 0
        assert {item.candidate_kind for item in fixture.hierarchy.candidates} == {
            "PARAGRAPH", "PROPOSITION", "SECTION"}
        assert fixture.recall.stop_reason == "RESOLVED"
        assert len(fixture.recall.centers) == 1
        assert fixture.recall.centers[0].md03_center.center.direction == (
            "UNDERSTANDING")
        exact = fixture.recall.exact_read
        assert exact is not None
        assert exact.acl_checked_before_payload == 1
        assert exact.metrics.page_faults == 1
        assert exact.metrics.page_in_records == 1
        assert exact.receipt.unauthorized_payload_read_count == 0
        assert fixture.qa_run.complete
        assert fixture.qa_run.query_result.candidates[0].proposition == fixture.target
        assert fixture.qa_run.query_result.candidates[0].citation_sources == (
            fixture.source,)
    finally:
        fixture.close()


def test_acl_denial_occurs_before_hot_set_payload_read():
    """错误 user 的 recall 必须在创建/读取 K-04 payload 前零读取拒绝。"""
    fixture = _runtime_fixture()
    try:
        run = fixture.recall
        center = run.centers[0]
        reader = run.exact_read
        assert reader is not None
        before = fixture.repository.segment_reads
        denied = fixture.reader.read(
            center,
            fixture.md03["current"],
            MemoryAccessContext(fixture.source.owner.tenant_id,
                                fixture.source.owner.user_id + 1, 0),
            reader.obligation.budget,
            reader_key=(95140, 2),
        )
        assert denied.receipt.stop_reason == "UNAUTHORIZED"
        assert denied.metrics.page_faults == 0
        assert denied.metrics.cold_read_bytes == 0
        assert fixture.repository.segment_reads == before
    finally:
        fixture.close()


def test_real_neighbor_competition_and_multiple_centers_stop_before_payload():
    """错误 source/scope/ACL 邻居不得胜出，两个合法 center 必须 clarify 零读。"""
    fixture = _runtime_fixture()
    try:
        base = fixture.index[0]
        second_valid = RecallIndexEntry(
            (73999, 2),
            base.source,
            base.scope,
            base.required_feature_keys,
            base.dependency_keys,
        )
        before = fixture.repository.segment_reads
        ambiguous = fixture.runtime.resolve(
            fixture.raw_query,
            fixture.history,
            fixture.md03["current"],
            (base, second_valid),
            fixture.access,
            fixture.budget,
            reader_key=(95140, 3),
        )
        assert ambiguous.stop_reason == "CLARIFY"
        assert len(ambiguous.centers) == 2
        assert ambiguous.exact_read is None
        assert fixture.repository.segment_reads == before

        wrong_source = SourceRef(
            fixture.source.source_kind,
            fixture.source.source_id + 1000,
            fixture.source.document_id,
            fixture.source.owner,
            fixture.source.versions,
        )
        wrong_source_neighbor = RecallIndexEntry(
            (73999, 3),
            wrong_source,
            document_scope(wrong_source),
            (base.required_feature_keys[0], StableRecordKey((73999, 30))),
            base.dependency_keys,
        )
        selected = fixture.runtime.resolve(
            fixture.raw_query,
            fixture.history,
            fixture.md03["current"],
            (base, wrong_source_neighbor),
            fixture.access,
            fixture.budget,
            reader_key=(95140, 4),
        )
        assert selected.stop_reason == "RESOLVED"
        assert len(selected.centers) == 1
        assert selected.centers[0].index_entry == base

        wrong_owner = OwnerScope(
            fixture.source.owner.tenant_id,
            fixture.source.owner.user_id + 1,
            fixture.source.owner.session_id,
            fixture.source.owner.visibility,
        )
        wrong_acl_source = SourceRef(
            fixture.source.source_kind,
            fixture.source.source_id + 2000,
            fixture.source.document_id,
            wrong_owner,
            fixture.source.versions,
        )
        wrong_acl_neighbor = RecallIndexEntry(
            (73999, 4),
            wrong_acl_source,
            document_scope(wrong_acl_source),
            base.required_feature_keys,
            base.dependency_keys,
        )
        acl_filtered = fixture.runtime.resolve(
            fixture.raw_query,
            fixture.history,
            fixture.md03["current"],
            (base, wrong_acl_neighbor),
            fixture.access,
            fixture.budget,
            reader_key=(95140, 5),
        )
        assert acl_filtered.stop_reason == "RESOLVED"
        assert len(acl_filtered.centers) == 1
        assert acl_filtered.centers[0].index_entry == base
        assert acl_filtered.rejected_unauthorized_center_keys
        assert acl_filtered.exact_read.receipt.unauthorized_payload_read_count == 0
    finally:
        fixture.close()


def test_revision_invalidates_only_dependent_hierarchy_center_and_claim():
    """后文更正必须精确失效三类依赖项，并逐字节保留无关派生身份。"""
    changed = AttractorDependency(
        minimal_instruction_identity((95200, 1)),
        concept_identity((95200, 2)),
    )
    unrelated = AttractorDependency(
        minimal_instruction_identity((95200, 3)),
        concept_identity((95200, 4)),
    )
    bindings = tuple(sorted((
        FreeTextDerivedDependency(
            StableRecordKey((74015, 1)), "HIERARCHY", (changed,)),
        FreeTextDerivedDependency(
            StableRecordKey((74015, 2)), "CENTER", (changed,)),
        FreeTextDerivedDependency(
            StableRecordKey((74015, 3)), "CLAIM", (changed,)),
        FreeTextDerivedDependency(
            StableRecordKey((75015, 1)), "CLAIM", (unrelated,)),
    )))
    receipt = FreeTextRevisionInvalidator(bindings).invalidate((changed,))
    assert receipt.hierarchy_keys == (StableRecordKey((74015, 1)),)
    assert receipt.center_keys == (StableRecordKey((74015, 2)),)
    assert receipt.claim_keys == (StableRecordKey((74015, 3)),)
    assert receipt.unaffected_bit_identical == 1
    assert receipt.host_learning_write_count == 0

    revision_seed = read_authored_free_text_hierarchy_recall_seeds(_SAMPLE)[14]
    private = revision_seed.expected_payload.to_value()
    assert [item.to_list() for item in receipt.invalidated_keys] == (
        private["invalidated_keys"])
    assert [item.to_list() for item in receipt.preserved_keys] == (
        private["preserved_keys"])


def test_independent_evaluator_passes_then_four_ablations_fail_own_dimensions():
    """私有标签只在 execution 后读取，四个缺组件反例须稳定失败。"""
    fixture = _runtime_fixture()
    try:
        report = FreeTextProductionReport(
            fixture.hierarchy,
            fixture.recall,
            fixture.qa_run,
            (),
            (),
            1,
            0,
            0,
        )
        evaluator = IndependentFreeTextHierarchyRecallEvaluator()
        expected = fixture.seed.expected_payload
        passed = evaluator.evaluate(report, expected)
        assert passed.passed
        assert passed.evaluator_host_write_count == 0

        no_hierarchy = evaluator.evaluate(
            replace(report, hierarchy=None), expected)
        assert no_hierarchy.result("HIERARCHY_FORMATION").passed == 0

        no_centers_run = FreeTextRecallRun(
            (), (), (), None, "UNKNOWN", None, 0, 0)
        no_centers = evaluator.evaluate(replace(
            report, recall=no_centers_run, question=None), expected)
        assert no_centers.result("CENTER_FORMATION").passed == 0

        no_paraphrase_evidence = evaluator.evaluate(replace(
            report, recall=no_centers_run, question=None), expected)
        assert no_paraphrase_evidence.result("RECALL_SELECTION").passed == 0

        no_page_in_run = FreeTextRecallRun(
            fixture.recall.matched_features,
            fixture.recall.centers,
            fixture.recall.rejected_unauthorized_center_keys,
            fixture.recall.centers[0].center_key,
            "NO_MATCH",
            None,
            0,
            0,
        )
        no_page_in = evaluator.evaluate(replace(
            report, recall=no_page_in_run, question=None), expected)
        assert no_page_in.result("CITATION_EXACTNESS").passed == 0
    finally:
        fixture.close()
