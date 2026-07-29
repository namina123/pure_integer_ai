"""W-02 v2 当前 Evidence 词干组合、关系消融和恢复对抗测试。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_w02_contract import (
    D03_GLOBAL_MANIFEST_PATH,
    W02_OWNER_KEY,
    W02_RUNNER_KEY,
    W02PayloadFirewall,
    W02RunRequest,
    open_w02_frozen_context,
)
from pure_integer_ai.experiments.ph2_w02_learning import (
    GENERATION_GENERATED,
    GENERATION_UNKNOWN,
    W02LearningError,
    W02MorphologyTarget,
)
from pure_integer_ai.experiments.ph2_w02_learning_v2 import (
    W02MorphologyTargetV2,
    build_w02_morphology_target_v2,
    morphology_target_from_observation_v2,
    morphology_unit_evidence_v2,
    open_w02_learning_runtime_v2,
)
from pure_integer_ai.storage.backend import SQLiteBackend


_REPOSITORY = Path(__file__).resolve().parents[1]
_BASE_REMOTE_COMMIT = "6322ed3d6aedf1a0fceeaffd1990ed5c9015e3f8"


def _training_payload():
    """通过正式 D-03/W-01 firewall 只读两个 public train pack。"""
    context = open_w02_frozen_context(
        _REPOSITORY,
        D03_GLOBAL_MANIFEST_PATH,
        current_remote_commit_sha1=_BASE_REMOTE_COMMIT,
    )
    request = W02RunRequest(
        2, 1, 1, context.stage_key, W02_OWNER_KEY, W02_RUNNER_KEY,
        context.current_remote_commit_sha1, context.stable_key(),
        context.w01_receipt_sha256, (1, 20260729), (1, 1, 20260729),
        1, "fresh",
        tuple(item.relative_path
              for item in context.candidate_payload_bindings),
        tuple(item.relative_path
              for item in context.teacher_evidence_bindings),
    )
    return W02PayloadFirewall.open(
        _REPOSITORY, context, request).read_training_payload()


def _observation(payload, candidate_id: str):
    """按 public candidate_id 找到唯一 LC-02 train Observation。"""
    matches = tuple(
        item for item in payload.observations
        if item.payload_kind == "MorphologyCandidateV1"
        and item.typed_payload.to_value()["candidate_id"] == candidate_id
    )
    assert len(matches) == 1
    return matches[0]


def _unit_evidence(observation, unit_kind: str):
    """从 public Observation 找到唯一指定角色并构造 v2 Evidence。"""
    units = tuple(
        item for item in observation.typed_payload.to_value()["analysis_units"]
        if item["unit_kind"] == unit_kind
    )
    assert len(units) == 1
    return morphology_unit_evidence_v2(observation, units[0]["unit_id"])


def _without_relation(candidates, construction_key: str, relation_kind: str):
    """只删指定 learned construction 的一类关系，保留其余候选字节语义。"""
    result = []
    for candidate in candidates:
        if candidate.payload.get("construction_key") != construction_key:
            result.append(candidate)
            continue
        payload = deepcopy(candidate.payload)
        payload["morphology_relations"] = [
            item for item in payload["morphology_relations"]
            if item["relation_kind"] != relation_kind
        ]
        result.append(replace(candidate, payload=payload))
    return tuple(result)


def test_evidence_bound_novel_stem_generalizes_and_relations_remain_bearing(
        tmp_path, monkeypatch):
    """未入词形表的当前 stem 可组合；裸串、关系缺失和例外外推均失败关闭。"""
    payload = _training_payload()
    unknown = _observation(payload, "teacher-unknown-candidate-v1")
    compound = _observation(payload, "teacher-compound-candidate-v1")
    exception = _observation(
        payload, "teacher-exception-revision-candidate-v1")
    novel_stem = _unit_evidence(unknown, "STEM")
    component = _unit_evidence(compound, "COMPONENT")
    affix_target = build_w02_morphology_target_v2(
        "suffix-hua-construction-v1", novel_stem)
    redup_target = build_w02_morphology_target_v2(
        "redup-aa-construction-v1", novel_stem)
    compound_target = build_w02_morphology_target_v2(
        "modifier-head-construction-v1", novel_stem, (component,))
    novel_exception_target = build_w02_morphology_target_v2(
        "lexical-exception-construction-v2", novel_stem)
    exact_exception_target = morphology_target_from_observation_v2(exception)

    path = tmp_path / "w02-v2.sqlite3"
    backend = SQLiteBackend(str(path))
    try:
        runtime = open_w02_learning_runtime_v2(backend, mode="fresh")
        runtime.consume(payload)
        before = runtime.state_key()
        assert runtime.word_forms.lookup(
            novel_stem.surface, branch=runtime.branch) is None

        affix = runtime.generate(affix_target)
        redup = runtime.generate(redup_target)
        compound_result = runtime.generate(compound_target)
        exact_exception = runtime.generate(exact_exception_target)
        assert affix.status == GENERATION_GENERATED
        assert affix.surfaces == ("清晰化",)
        assert redup.surfaces == ("清晰清晰",)
        assert compound_result.surfaces == ("纸清晰",)
        assert exact_exception.surfaces == ("蝴蝶",)
        assert runtime.generate(
            novel_exception_target).status == GENERATION_UNKNOWN
        assert runtime.state_key() == before

        with pytest.raises(TypeError, match="typed Evidence"):
            runtime.generate(W02MorphologyTarget(
                "suffix-hua-construction-v1", "任意裸串"))
        with pytest.raises(TypeError):
            W02MorphologyTargetV2(
                "suffix-hua-construction-v1", novel_stem, ())

        original = runtime.candidates()
        ablations = (
            (affix_target, "suffix-hua-construction-v1", "HAS_STEM"),
            (affix_target, "suffix-hua-construction-v1", "FILLS_SLOT"),
            (affix_target, "suffix-hua-construction-v1", "ATTACHES_AFFIX"),
            (redup_target, "redup-aa-construction-v1", "REDUPLICATES"),
            (compound_target, "modifier-head-construction-v1",
             "COMPOUND_COMPONENT"),
        )
        for target, construction_key, relation_kind in ablations:
            monkeypatch.setattr(
                runtime,
                "candidates",
                lambda construction_key=construction_key,
                relation_kind=relation_kind: _without_relation(
                    original, construction_key, relation_kind),
            )
            assert runtime.generate(target).status == GENERATION_UNKNOWN
        monkeypatch.setattr(runtime, "candidates", lambda: original)
        state = runtime.state_key()
        expected = (
            affix.stable_key(),
            redup.stable_key(),
            compound_result.stable_key(),
            exact_exception.stable_key(),
        )
    finally:
        backend.close()

    backend = SQLiteBackend(str(path))
    try:
        restored = open_w02_learning_runtime_v2(backend, mode="resume")
        assert restored.state_key() == state
        assert (
            restored.generate(affix_target).stable_key(),
            restored.generate(redup_target).stable_key(),
            restored.generate(compound_target).stable_key(),
            restored.generate(exact_exception_target).stable_key(),
        ) == expected
    finally:
        backend.close()


def test_target_evidence_rejects_detached_stem_component_and_span():
    """当前 Observation 的 role 方向或 span 被改后不得构成 v2 target Evidence。"""
    payload = _training_payload()
    unknown = _observation(payload, "teacher-unknown-candidate-v1")
    compound = _observation(payload, "teacher-compound-candidate-v1")

    cases = []
    for relation_kind, field, replacement_id in (
            ("HAS_STEM", "target_unit_id", "a"),
            ("FILLS_SLOT", "source_unit_id", "a")):
        value = deepcopy(unknown.typed_payload.to_value())
        for relation in value["morphology_relations"]:
            if relation["relation_kind"] == relation_kind:
                relation[field] = replacement_id
        cases.append((
            replace(unknown, typed_payload=CanonicalJsonObject.from_value(value)),
            "s",
        ))

    value = deepcopy(compound.typed_payload.to_value())
    for relation in value["morphology_relations"]:
        if relation["relation_kind"] == "COMPOUND_COMPONENT":
            relation["target_unit_id"] = "s"
    cases.append((
        replace(compound, typed_payload=CanonicalJsonObject.from_value(value)),
        "m",
    ))

    value = deepcopy(unknown.typed_payload.to_value())
    for unit in value["analysis_units"]:
        if unit["unit_id"] == "s":
            unit["surface"] = "清醒"
    cases.append((
        replace(unknown, typed_payload=CanonicalJsonObject.from_value(value)),
        "s",
    ))

    for observation, unit_id in cases:
        with pytest.raises(W02LearningError):
            morphology_unit_evidence_v2(observation, unit_id)
