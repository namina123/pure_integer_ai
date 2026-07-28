"""R-04 读前精确授权、多中心共享读取和严格生成交付反例。"""
from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.attractor_state import AttractorDependency
from pure_integer_ai.cognition.shared.identity import (
    SourceRef,
    concept_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    query_scope,
)
from pure_integer_ai.experiments.authorized_center_runtime import (
    AuthorizedCenterAgendaRuntime,
    CenterAuthorizationBinding,
    CenterAuthorizationProjection,
)
from pure_integer_ai.experiments.authorized_generation_delivery import (
    AuthorizedGenerationClaim,
    AuthorizedGenerationDeliveryAuthority,
    DELIVERY_AUTHORIZED,
    DELIVERY_CITATION_MISMATCH,
    DELIVERY_CLAIM_BINDING_MISMATCH,
    DELIVERY_CLAIM_COVERAGE_MISMATCH,
    DELIVERY_POSTCHECK_MISSING,
    DELIVERY_REQUIREMENT_NOT_STRICT,
)
from pure_integer_ai.experiments.free_text_recall_runtime import (
    LearnedSurfaceFeatureMatcher,
    RecalledFactQuestionExecutor,
)
from pure_integer_ai.experiments.free_text_revision_runtime import (
    FreeTextDerivedDependency,
    FreeTextRevisionInvalidator,
)
from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.experiments.ph2_authorized_center_generation_catalog import (
    MANIFEST_PATH,
    build_authorized_center_generation_manifest,
)
from pure_integer_ai.experiments.ph2_authorized_center_generation_contract import (
    AuthorizedCenterGenerationContractError,
    AuthorizedCenterGenerationEvidenceFile,
    read_authorized_center_generation_manifest,
    verify_authorized_center_generation_files,
    write_authorized_center_generation_manifest,
)
from pure_integer_ai.storage.location_manifest import ManifestKeyRange
from tests.test_d02_p3ia_free_text_hierarchy_recall_runtime import (
    _runtime_fixture,
)
from tests.test_f00_generation_postcheck import _postcheck_owners
from tests.test_f00_question_answer_runtime import _fixture as _question_fixture


_BASE = 95600
_ROOT = Path(__file__).resolve().parents[1]


def _binding(fixture, center=None) -> CenterAuthorizationBinding:
    """从当前真实 manifest 为一个已形成 center 建立精确授权绑定。"""
    selected = fixture.recall.centers[0] if center is None else center
    manifest = fixture.reader.store.current_manifest()
    assert manifest is not None
    entry = selected.index_entry
    locations = tuple(
        item for item in manifest.entries
        if (item.descriptor_key == fixture.reader.descriptor_key
            and item.key_range.lower_key <= entry.record_key
            and entry.record_key <= item.key_range.upper_key))
    assert len(locations) == 1
    location = locations[0]
    return CenterAuthorizationBinding(
        selected.center_key,
        fixture.reader.descriptor_key,
        entry.record_key,
        entry.source,
        entry.scope,
        location.version_key,
        entry.source.owner,
        location.segment_key,
    )


def _projection(
        fixture,
        bindings,
        *,
        access=None,
        policy_epoch=1,
        manifest_epoch=None,
        manifest_key=None,
        ) -> CenterAuthorizationProjection:
    """绑定当前 location epoch 和调用方访问上下文。"""
    manifest = fixture.reader.store.current_manifest()
    assert manifest is not None
    return CenterAuthorizationProjection(
        StableRecordKey((_BASE + 1, policy_epoch)),
        (_BASE + 2, 1),
        policy_epoch,
        manifest.manifest_key if manifest_key is None else manifest_key,
        manifest.publish_epoch if manifest_epoch is None else manifest_epoch,
        fixture.access if access is None else access,
        tuple(bindings),
    )


def _agenda(fixture, centers=None, bindings=None):
    """运行一次真实 repository 计数的授权中心 agenda。"""
    selected = fixture.recall.centers if centers is None else tuple(centers)
    actual_bindings = (
        tuple(_binding(fixture, item) for item in selected)
        if bindings is None else tuple(bindings))
    return AuthorizedCenterAgendaRuntime(fixture.reader).run(
        selected,
        fixture.md03["current"],
        _projection(fixture, actual_bindings),
        fixture.budget,
        reader_key_prefix=(_BASE + 3, 1),
        current_policy_epoch=1,
    )


def test_authorized_center_checks_exact_binding_then_reads_one_cited_payload():
    """正确绑定必须读一次真实 segment，并返回原 typed payload 和 citation。"""
    fixture = _runtime_fixture()
    try:
        before = fixture.repository.segment_reads

        run = _agenda(fixture)

        assert fixture.repository.segment_reads == before + 1
        assert len(run.record_reads) == 1
        assert len(run.states) == 1
        state = run.states[0]
        assert state.receipt.state == "READY"
        assert state.receipt.physical_payload_gets == 1
        assert state.receipt.reused_payload == 0
        assert state.payload is not None
        assert state.payload.proposition == fixture.target
        assert state.receipt.citations[0].source_ref == fixture.source
        assert state.receipt.citations[0].record_key == StableRecordKey(
            fixture.index[0].record_key)
    finally:
        fixture.close()


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("center", "CENTER_UNBOUND"),
        ("record", "RECORD_MISMATCH"),
        ("source", "SOURCE_MISMATCH"),
        ("scope", "SCOPE_MISMATCH"),
        ("version", "VERSION_MISMATCH"),
        ("acl", "ACL_DENIED"),
        ("manifest", "MANIFEST_STALE"),
        ("policy", "POLICY_STALE"),
    ),
)
def test_identity_source_scope_version_acl_and_epoch_fail_before_payload(
        case, expected):
    """任一授权轴漂移都必须在 repository segment get 前稳定拒绝。"""
    fixture = _runtime_fixture()
    try:
        center = fixture.recall.centers[0]
        binding = _binding(fixture, center)
        access = fixture.access
        manifest_epoch = None
        current_policy_epoch = 1
        if case == "center":
            binding = replace(
                binding, center_key=StableRecordKey((_BASE + 10, 1)))
        elif case == "record":
            binding = replace(binding, record_key=(_BASE + 10, 2))
        elif case == "source":
            source = replace(
                binding.source, source_id=binding.source.source_id + 1)
            binding = replace(
                binding, source=source, scope=document_scope(source))
        elif case == "scope":
            binding = replace(
                binding,
                scope=query_scope(9001, parent=document_scope(binding.source)),
            )
        elif case == "version":
            binding = replace(binding, version_key=(_BASE + 10, 3))
        elif case == "acl":
            access = replace(fixture.access, user_id=fixture.access.user_id + 1)
        elif case == "manifest":
            manifest = fixture.reader.store.current_manifest()
            assert manifest is not None
            manifest_epoch = manifest.publish_epoch + 1
        elif case == "policy":
            current_policy_epoch = 2
        projection = _projection(
            fixture,
            (binding,),
            access=access,
            manifest_epoch=manifest_epoch,
        )
        before = fixture.repository.segment_reads

        run = AuthorizedCenterAgendaRuntime(fixture.reader).run(
            (center,),
            fixture.md03["current"],
            projection,
            fixture.budget,
            reader_key_prefix=(_BASE + 11, 1),
            current_policy_epoch=current_policy_epoch,
        )

        assert fixture.repository.segment_reads == before
        assert run.record_reads == ()
        assert run.states[0].receipt.state == expected
        assert run.states[0].payload is None
        assert run.states[0].receipt.citations == ()
    finally:
        fixture.close()


def test_non_isolated_physical_segment_fails_before_payload(monkeypatch):
    """授权 record 与其他 key 共段时不得先读整段再过滤。"""
    fixture = _runtime_fixture()
    try:
        center = fixture.recall.centers[0]
        binding = _binding(fixture, center)
        manifest = fixture.reader.store.current_manifest()
        assert manifest is not None and len(manifest.entries) == 1
        location = manifest.entries[0]
        widened = replace(
            location,
            key_range=ManifestKeyRange(
                location.key_range.lower_key,
                (*location.key_range.upper_key, 1),
            ),
        )
        monkeypatch.setattr(
            fixture.reader.store,
            "current_manifest",
            lambda: replace(manifest, entries=(widened,)),
        )
        before = fixture.repository.segment_reads

        run = AuthorizedCenterAgendaRuntime(fixture.reader).run(
            (center,),
            fixture.md03["current"],
            _projection(fixture, (binding,)),
            fixture.budget,
            reader_key_prefix=(_BASE + 12, 1),
            current_policy_epoch=1,
        )

        assert fixture.repository.segment_reads == before
        assert run.record_reads == ()
        assert run.states[0].receipt.state == "SEGMENT_NOT_ISOLATED"
    finally:
        fixture.close()


def test_two_centers_share_one_read_but_keep_obligation_frontier_and_stop():
    """同一 record 的两个消费者只读一次，停止其中一个不关闭另一个。"""
    fixture = _runtime_fixture()
    try:
        first = fixture.recall.centers[0]
        second = replace(
            first, center_key=StableRecordKey((_BASE + 20, 1)))
        bindings = (_binding(fixture, first), _binding(fixture, second))
        before = fixture.repository.segment_reads

        run = _agenda(fixture, (first, second), bindings)

        assert fixture.repository.segment_reads == before + 1
        assert len(run.record_reads) == 1
        assert {item.receipt.state for item in run.states} == {"READY"}
        assert sum(item.receipt.physical_payload_gets for item in run.states) == 1
        assert sum(item.receipt.reused_payload for item in run.states) == 1
        assert len({item.obligation.obligation_key for item in run.states}) == 2
        assert len({item.obligation.frontier_key for item in run.states}) == 2
        assert len({item.receipt.receipt_key for item in run.states}) == 2

        stopped = run.stop_center(first.center_key)

        assert stopped.state(first.center_key).receipt.state == "STOPPED"
        assert stopped.state(first.center_key).payload is None
        assert stopped.state(second.center_key).receipt.state == "READY"
        assert stopped.state(second.center_key).payload == run.state(
            second.center_key).payload
        assert stopped.record_reads == run.record_reads
    finally:
        fixture.close()


def test_revision_locality_and_evidence_only_surface_center_formation():
    """后文修正只失效依赖项；移除学习 Evidence 后不形成任何 center。"""
    changed = AttractorDependency(
        minimal_instruction_identity((_BASE + 30, 1)),
        concept_identity((_BASE + 30, 2)),
    )
    unrelated = AttractorDependency(
        minimal_instruction_identity((_BASE + 30, 3)),
        concept_identity((_BASE + 30, 4)),
    )
    bindings = tuple(sorted((
        FreeTextDerivedDependency(
            StableRecordKey((_BASE + 31, 1)), "CENTER", (changed,)),
        FreeTextDerivedDependency(
            StableRecordKey((_BASE + 31, 2)), "CLAIM", (changed,)),
        FreeTextDerivedDependency(
            StableRecordKey((_BASE + 31, 3)), "CLAIM", (unrelated,)),
    )))
    receipt = FreeTextRevisionInvalidator(bindings).invalidate((changed,))
    assert receipt.center_keys == (StableRecordKey((_BASE + 31, 1)),)
    assert receipt.claim_keys == (StableRecordKey((_BASE + 31, 2)),)
    assert receipt.preserved_keys == (StableRecordKey((_BASE + 31, 3)),)
    assert receipt.unaffected_bit_identical == 1

    fixture = _runtime_fixture()
    try:
        former = fixture.runtime.center_former
        assert former.form(
            fixture.raw_query,
            fixture.history,
            fixture.md03["current"],
            fixture.index,
        )
        assert former.form(
            fixture.raw_query,
            (),
            fixture.md03["current"],
            fixture.index,
        ) == ()
        tree = ast.parse(inspect.getsource(LearnedSurfaceFeatureMatcher))
        literal_keys = tuple(
            key.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        )
        assert literal_keys == ()
    finally:
        fixture.close()


def _authorized_generation_fixture():
    """把授权冷读、同次 recalled QA 和完整 G-04 装成一个测试纵切。"""
    fixture = _runtime_fixture()
    agenda = _agenda(fixture)
    mapper, postchecker, _, _, _ = _postcheck_owners()
    qa = _question_fixture(
        world=(fixture.source, fixture.response_scope, fixture.target),
        executor_factory=lambda route: RecalledFactQuestionExecutor(
            fixture.recall,
            route=route,
            executed_reason=minimal_instruction_identity((_BASE + 40, 1)),
            trace_prefix=(_BASE + 40, 2),
        ),
        answer_text="事实",
        postcheck_mapper=mapper,
        postchecker=postchecker,
    )
    run = qa.runtime.run(qa.request)
    assert run.planning_request is not None
    claim = AuthorizedGenerationClaim.from_authorized_center(
        run.planning_request.candidates[0], agenda.states[0])
    return fixture, qa, agenda, run, claim


def test_strict_delivery_requires_authorized_claim_full_g04_and_exact_citations():
    """授权 recall claim 只有在完整 coverage/citation/trust/postcheck 后可交付。"""
    fixture, qa, _, run, claim = _authorized_generation_fixture()
    try:
        decision = AuthorizedGenerationDeliveryAuthority().authorize(
            run, (claim,))

        assert decision.state == DELIVERY_AUTHORIZED
        assert decision.deliverable
        assert decision.envelope is not None
        assert decision.envelope.units == tuple(map(ord, "事实"))
        assert decision.envelope.claims == (claim,)
        assert decision.envelope.cited_sources == claim.evidence_sources
    finally:
        qa.close()
        fixture.close()


def test_missing_postcheck_never_exposes_a_deliverable_surface():
    """旧 G-00..G-03 complete 报告可审计，但不得越过严格交付边界。"""
    fixture = _runtime_fixture()
    try:
        agenda = _agenda(fixture)
        run = fixture.qa_run
        assert run.complete and run.postcheck is None
        assert run.planning_request is not None
        claim = AuthorizedGenerationClaim.from_authorized_center(
            run.planning_request.candidates[0], agenda.states[0])

        decision = AuthorizedGenerationDeliveryAuthority().authorize(
            run, (claim,))

        assert decision.state == DELIVERY_POSTCHECK_MISSING
        assert not decision.deliverable
        assert decision.envelope is None
    finally:
        fixture.close()


def test_citation_scope_requirement_and_unauthorized_attractor_fail_closed():
    """引用、scope、严格要求或额外吸引项漂移都不得暴露 renderer units。"""
    fixture, qa, _, run, claim = _authorized_generation_fixture()
    authority = AuthorizedGenerationDeliveryAuthority()
    try:
        assert run.postcheck is not None
        observation = run.postcheck.parsed.observation
        assert observation is not None

        no_citation_observation = replace(observation, cited_sources=())
        no_citation_postcheck = replace(
            run.postcheck,
            parsed=replace(
                run.postcheck.parsed,
                observation=no_citation_observation,
            ),
        )
        no_citation = authority.authorize(
            replace(run, postcheck=no_citation_postcheck), (claim,))
        assert no_citation.state == DELIVERY_CITATION_MISMATCH
        assert no_citation.envelope is None

        recovered = observation.propositions[0]
        drifted_scope = query_scope(
            9002, parent=document_scope(recovered.source))
        drifted_observation = replace(
            observation,
            propositions=(replace(recovered, scope=drifted_scope),),
        )
        drifted_postcheck = replace(
            run.postcheck,
            parsed=replace(
                run.postcheck.parsed,
                observation=drifted_observation,
            ),
        )
        drifted = authority.authorize(
            replace(run, postcheck=drifted_postcheck), (claim,))
        assert drifted.state == "OBSERVATION_MISMATCH"
        assert drifted.envelope is None

        requirement = run.postcheck.request.source_requirements[0]
        weak_request = replace(
            run.postcheck.request,
            source_requirements=(replace(
                requirement, trust_required=False),),
        )
        weak = authority.authorize(
            replace(run, postcheck=replace(
                run.postcheck, request=weak_request)),
            (claim,),
        )
        assert weak.state == DELIVERY_REQUIREMENT_NOT_STRICT
        assert weak.envelope is None

        unauthorized_attractor = replace(
            claim,
            candidate_key=(_BASE + 41, 1),
            proposition_key=(_BASE + 41, 2),
            authorization_receipt_key=(_BASE + 41, 3),
        )
        injected = authority.authorize(
            run, (claim, unauthorized_attractor))
        assert injected.state == DELIVERY_CLAIM_COVERAGE_MISMATCH
        assert injected.envelope is None

        replaced_claim = replace(
            claim,
            proposition_key=(_BASE + 41, 4),
        )
        replaced_decision = authority.authorize(run, (replaced_claim,))
        assert replaced_decision.state == DELIVERY_CLAIM_BINDING_MISMATCH
        assert replaced_decision.envelope is None
    finally:
        qa.close()
        fixture.close()


@pytest.fixture(scope="module")
def formal_manifest():
    """按当前仓库内容构建一次确定性 R-04 正式 manifest。"""
    return build_authorized_center_generation_manifest(_ROOT)


def test_manifest_round_trip_file_identity_and_zero_execution(
        tmp_path, formal_manifest):
    """R-04 artifact 必须 canonical 回读并逐字节绑定实现和测试。"""
    target = tmp_path / "r04.json"
    assert write_authorized_center_generation_manifest(
        formal_manifest, target) == target
    assert read_authorized_center_generation_manifest(target) == formal_manifest
    verify_authorized_center_generation_files(
        formal_manifest, repository_root=_ROOT)
    assert set(formal_manifest.requirement_decisions.to_value().values()) == {
        "PASS"}
    assert formal_manifest.execution_state.to_value()["formal_training_runs"] == 0
    assert formal_manifest.execution_state.to_value()["teacher_calls"] == 0


def test_manifest_rejects_fake_counterexample_coverage(formal_manifest):
    """任一读前或交付反例被删都必须使固定合同构造失败。"""
    coverage = formal_manifest.counterexample_coverage.to_value()
    coverage["acl_denial_before_segment_get"] = 0
    from pure_integer_ai.experiments.ph2_dataset_contract import (
        CanonicalJsonObject,
    )
    with pytest.raises(
            AuthorizedCenterGenerationContractError,
            match="counterexample coverage 漂移"):
        replace(
            formal_manifest,
            counterexample_coverage=CanonicalJsonObject.from_value(coverage),
        )


def test_manifest_rejects_file_identity_drift(formal_manifest):
    """任何实现或反例文件尺寸/hash 漂移都必须回验失败。"""
    first = formal_manifest.evidence_files[0]
    drifted = AuthorizedCenterGenerationEvidenceFile(
        first.relative_path,
        first.role,
        first.byte_count + 1,
        first.sha256,
    )
    manifest = replace(
        formal_manifest,
        evidence_files=(drifted, *formal_manifest.evidence_files[1:]),
    )
    with pytest.raises(
            AuthorizedCenterGenerationContractError,
            match="evidence 文件身份漂移"):
        verify_authorized_center_generation_files(
            manifest, repository_root=_ROOT)


def test_manifest_is_idempotent_but_non_overwritable(
        tmp_path, formal_manifest):
    """同内容可幂等重放，同版本异内容不可覆盖。"""
    target = tmp_path / "r04.json"
    write_authorized_center_generation_manifest(formal_manifest, target)
    assert write_authorized_center_generation_manifest(
        formal_manifest, target) == target
    target.write_bytes(b"{}\n")
    with pytest.raises(
            AuthorizedCenterGenerationContractError,
            match="已存在且内容不同"):
        write_authorized_center_generation_manifest(formal_manifest, target)


def test_stored_manifest_is_current_readable_and_deterministic(formal_manifest):
    """仓内正式 artifact 必须等于当前 builder，重复构建字节完全一致。"""
    stored = read_authorized_center_generation_manifest(_ROOT / MANIFEST_PATH)
    rebuilt = build_authorized_center_generation_manifest(_ROOT)
    assert stored == formal_manifest == rebuilt
    assert stored.canonical_bytes() == rebuilt.canonical_bytes()
    verify_authorized_center_generation_files(stored, repository_root=_ROOT)
