"""MD-03 三向 center adapter、精确去重和不可覆盖 manifest T0。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.generation_plan import (
    AnswerGenerationGoal,
)
from pure_integer_ai.cognition.shared.identity import (
    OwnerScope,
    ParserVersion,
    VersionBundle,
    VISIBILITY_SESSION,
    concept_identity,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    StableRecordKey,
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_md03_center_adapter import (
    DirectionalCenterAdapterConfig,
    DirectionalCenterProfile,
    DirectionalMemoryCenterAdapter,
    MD03CenterAdapterError,
)
from pure_integer_ai.experiments.ph2_md03_manifest import (
    MD03AdapterManifest,
    MD03ManifestError,
    build_md03_adapter_manifest,
    read_md03_adapter_manifest,
    write_md03_adapter_manifest,
)
from pure_integer_ai.experiments.ph2_memory_dynamics_contract import (
    DIRECTIONS,
    MemoryCenterOrigin,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend
from tests.test_a10_attractor_state import (
    _goals,
    _instruction as _a10_instruction,
)
from tests.test_m06_memory_query import (
    _close_query,
    _current,
    _open_query,
    _source,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MD02_PATH = REPO_ROOT / "data/ph2/manifests/md02_situation_state_adapter_v1.json"
BASELINE_PATH = REPO_ROOT / "data/ph2/manifests/language_capability_baseline_v19.json"
MANIFEST_PATH = REPO_ROOT / "data/ph2/manifests/md03_directional_center_adapter_v1.json"


def _key(*values: int) -> StableRecordKey:
    """建立测试使用的 D-02 正严格整数键。"""
    return StableRecordKey(tuple(values))


def _adapter() -> DirectionalMemoryCenterAdapter:
    """按 MD-01 冻结顺序列全三个方向的独立 profile。"""
    profiles = tuple(
        DirectionalCenterProfile(
            direction,
            _key(30100, ordinal, 1),
            _key(30100, ordinal, 2),
            _key(30100, ordinal, 3),
        )
        for ordinal, direction in enumerate(DIRECTIONS, start=1)
    )
    return DirectionalMemoryCenterAdapter(
        DirectionalCenterAdapterConfig(profiles))


def _fixture():
    """建立真实 M-06 current、A-10 obligation 和 G-00 answer goal。"""
    backend = DictBackend()
    ctx = make_train_context(backend)
    source = _source(83)
    scope = _open_query(ctx, source, local_id=83)
    current = _current(ctx, source, scope, ordinal=3)
    reasoning = _goals(source, scope, count=1)[0]
    generation = AnswerGenerationGoal(
        _a10_instruction(source, 30150),
        reasoning.proposition,
        reasoning.required,
        source,
        scope,
    )
    return {
        "adapter": _adapter(),
        "backend": backend,
        "ctx": ctx,
        "current": current,
        "generation": generation,
        "reasoning": reasoning,
        "source": source,
    }


def _close(fixture) -> None:
    """关闭测试 query 和临时后端。"""
    _close_query(fixture["ctx"])
    fixture["backend"].close()


def _triplet(fixture):
    """形成强度互异的三向 envelope。"""
    return fixture["adapter"].form_triplet(
        fixture["current"],
        fixture["current"].occurrences[0],
        fixture["reasoning"],
        fixture["generation"],
        understanding_strength="CONDITIONAL",
        reasoning_strength="MANDATORY",
        generation_strength="SPECULATIVE",
    )


def test_same_query_forms_three_distinct_payloads_and_shared_proposition_target():
    """同一 query 三向 payload 互异，推理与生成仍共享命题目标身份。"""
    fixture = _fixture()
    try:
        before = fixture["backend"].snapshot()
        report = _triplet(fixture)
        by_direction = {item.center.direction: item for item in report.centers}
        assert set(by_direction) == set(DIRECTIONS)
        assert {item.payload_kind for item in report.centers} == {
            "ANSWER_GENERATION_GOAL", "CURRENT_TYPED_INPUT",
            "EVIDENCE_OBLIGATION",
        }
        assert len({item.payload_key for item in report.centers}) == 3
        assert len({item.input_query_key for item in report.centers}) == 1
        assert by_direction["REASONING"].center.target_key == (
            by_direction["GENERATION"].center.target_key)
        assert {item.center.strength for item in report.centers} == {
            "CONDITIONAL", "MANDATORY", "SPECULATIVE"}
        assert report.host_learning_write_count == 0
        assert fixture["backend"].snapshot() == before
    finally:
        _close(fixture)


def test_exact_dedup_merges_origin_dependencies_without_losing_provenance():
    """完整去重键相同才合并，并按 origin 身份合并全部依赖。"""
    fixture = _fixture()
    try:
        first = fixture["adapter"].from_understanding(
            fixture["current"], fixture["current"].occurrences[0],
            strength="MANDATORY")
        origin = first.center.origins[0]
        extra_dependency = _key(30200, 1)
        expanded_origin = MemoryCenterOrigin(
            origin.origin_kind,
            origin.origin_key,
            tuple(sorted((*origin.dependency_keys, extra_dependency))),
        )
        duplicate = replace(first, center=replace(
            first.center,
            origins=(expanded_origin,),
            dependency_keys=tuple(sorted((
                *first.center.dependency_keys, extra_dependency))),
        ))
        report = fixture["adapter"].deduplicate((first, duplicate))
        assert report.input_center_count == 2
        assert report.merged_duplicate_count == 1
        assert len(report.centers) == 1
        merged = report.centers[0].center
        assert len(merged.origins) == 1
        assert extra_dependency in merged.origins[0].dependency_keys
        assert extra_dependency in merged.dependency_keys
    finally:
        _close(fixture)


def test_direction_strength_and_adoption_condition_are_not_merged():
    """方向、强度或采用条件不同的 envelope 保持独立。"""
    fixture = _fixture()
    try:
        first = fixture["adapter"].from_understanding(
            fixture["current"], fixture["current"].occurrences[0],
            strength="MANDATORY")
        different_strength = replace(
            first, center=replace(first.center, strength="SPECULATIVE"))
        different_condition = replace(
            first,
            adoption_condition_keys=tuple(sorted((
                *first.adoption_condition_keys, _key(30210, 1)))),
        )
        different_direction = fixture["adapter"].from_reasoning(
            fixture["current"], fixture["reasoning"],
            strength="MANDATORY")
        report = fixture["adapter"].deduplicate((
            first, different_strength, different_condition,
            different_direction,
        ))
        assert len(report.centers) == 4
        assert report.merged_duplicate_count == 0
        assert len({item.dedup_key() for item in report.centers}) == 4
    finally:
        _close(fixture)


def test_write_permissions_are_directional_orthogonal_and_fail_closed():
    """三向权限边界互异，activation 不授权 adoption，篡改权限即拒绝。"""
    fixture = _fixture()
    try:
        report = _triplet(fixture)
        allowed = {
            item.center.direction: set(item.write_boundary.allowed_write_kinds)
            for item in report.centers
        }
        assert all(allowed[left].isdisjoint(allowed[right])
                   for left in DIRECTIONS for right in DIRECTIONS
                   if left < right)
        for item in report.centers:
            assert item.center.activation_only == 1
            assert item.write_boundary.activation_authorizes_adoption == 0
            assert item.write_boundary.host_learning_write_count == 0
            assert "CORE_LEARNING_WRITE" in (
                item.write_boundary.forbidden_write_kinds)
            with pytest.raises(MD03CenterAdapterError, match="allowed"):
                replace(item.write_boundary, allowed_write_kinds=())
    finally:
        _close(fixture)


def test_understanding_target_must_be_a_current_typed_anchor():
    """相同 owner/version 的旁路对象也不能冒充当前 typed 输入。"""
    fixture = _fixture()
    try:
        foreign = concept_identity(
            (30220, 1), owner=fixture["source"].owner,
            versions=fixture["source"].versions)
        before = fixture["backend"].snapshot()
        with pytest.raises(MD03CenterAdapterError, match="不属于"):
            fixture["adapter"].from_understanding(
                fixture["current"], foreign, strength="MANDATORY")
        assert fixture["backend"].snapshot() == before
    finally:
        _close(fixture)


@pytest.mark.parametrize("drift", ["source", "scope", "owner", "version"])
def test_foreign_reasoning_boundary_fails_before_any_write(drift):
    """source/scope/owner/version 任一漂移都在 adapter 首写前失败。"""
    fixture = _fixture()
    try:
        source = fixture["source"]
        scope = fixture["current"].scope
        if drift == "scope":
            from pure_integer_ai.cognition.shared.scope_identity import query_scope
            foreign_source = source
            foreign_scope = query_scope(30230, parent=scope.parent)
        else:
            if drift == "source":
                foreign_source = replace(source, document_id=source.document_id + 1)
            elif drift == "owner":
                foreign_source = replace(
                    source,
                    owner=OwnerScope(9, 8, 7, VISIBILITY_SESSION),
                )
            else:
                foreign_source = replace(
                    source,
                    versions=VersionBundle(parser=ParserVersion(1)),
                )
            from pure_integer_ai.cognition.shared.scope_identity import (
                document_scope,
                episode_scope,
                query_scope,
                session_scope,
            )
            session = session_scope(
                30230,
                owner=foreign_source.owner,
                versions=foreign_source.versions,
                source=foreign_source,
            )
            document = document_scope(foreign_source, parent=session)
            episode = episode_scope(30230, parent=document)
            foreign_scope = query_scope(30230, parent=episode)
        foreign = _goals(foreign_source, foreign_scope, count=1)[0]
        before = fixture["backend"].snapshot()
        with pytest.raises(MD03CenterAdapterError, match="source/scope"):
            fixture["adapter"].from_reasoning(
                fixture["current"], foreign, strength="MANDATORY")
        assert fixture["backend"].snapshot() == before
    finally:
        _close(fixture)


def _manifest() -> MD03AdapterManifest:
    """按当前 MD-02 与 v19 不可变 hash 建立 MD-03 artifact。"""
    return build_md03_adapter_manifest(
        md02_manifest_relative_path=(
            "data/ph2/manifests/md02_situation_state_adapter_v1.json"),
        md02_manifest_sha256=hashlib.sha256(MD02_PATH.read_bytes()).hexdigest(),
        baseline_manifest_relative_path=(
            "data/ph2/manifests/language_capability_baseline_v19.json"),
        baseline_manifest_sha256=hashlib.sha256(
            BASELINE_PATH.read_bytes()).hexdigest(),
    )


def test_manifest_round_trip_strict_fields_and_nonoverwrite(tmp_path):
    """MD-03 manifest 规范回读、幂等发布并拒绝覆盖和额外字段。"""
    manifest = _manifest()
    output = tmp_path / "md03.json"
    write_md03_adapter_manifest(manifest, output)
    assert read_md03_adapter_manifest(output) == manifest
    write_md03_adapter_manifest(manifest, output)
    output.write_bytes(canonical_json_line({"damaged": 1}))
    with pytest.raises(MD03ManifestError, match="内容不同"):
        write_md03_adapter_manifest(manifest, output)
    value = manifest.to_dict()
    value["mastered"] = 1
    output.write_bytes(canonical_json_line(value))
    with pytest.raises(MD03ManifestError, match="字段不精确"):
        read_md03_adapter_manifest(output)


def test_manifest_rejects_bad_hash_fake_probe_and_host_write():
    """T0 不得冒充 probe，坏前置或任何宿主学习写都失败。"""
    manifest = _manifest()
    with pytest.raises(MD03ManifestError, match="probe"):
        replace(manifest, probe_status="COMPLETE")
    with pytest.raises(MD03ManifestError, match="host learning write"):
        replace(manifest, host_learning_write_count=1)
    with pytest.raises(MD03ManifestError, match="SHA-256"):
        replace(manifest, md02_manifest_sha256="bad")
    with pytest.raises(MD03ManifestError, match="安全 POSIX"):
        replace(manifest, baseline_manifest_relative_path="../private.json")
    assert manifest.results_observed == 0
    assert all(value == 0 for value in manifest.execution_state.to_value().values())


def test_repository_md03_manifest_matches_current_immutable_prerequisites():
    """正式 MD-03 artifact 必须精确绑定 MD-02 和 v19 基线。"""
    assert read_md03_adapter_manifest(MANIFEST_PATH) == _manifest()
