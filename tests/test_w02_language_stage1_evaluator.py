"""W-02 私有 evaluator 的冻结合取、摘要边界与 host isolation。"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w02_evaluator import (
    GENERATION_HARD_CONJUNCT,
    STATUS_FAIL,
    STATUS_NE,
    STATUS_PASS,
    W02_ABLATION_ORDER,
    W02_EVALUATION_ORDER,
    W02AblationResult,
    W02DimensionResult,
    W02EvaluationError,
    W02PrivateEvaluationReport,
    aggregate_w02_evaluation,
)
from pure_integer_ai.experiments import ph2_w02_evaluator as evaluator_runtime
from pure_integer_ai.experiments.ph2_w02_evaluation_report import (
    W02EvaluationPublicationError,
    publish_w02_private_evaluation_report,
    read_w02_private_evaluation_report,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


_DIMENSIONS = W02_EVALUATION_ORDER[:-1]
_DIGESTS = {
    "core": "1" * 64,
    "cursor": "2" * 64,
    "logical": "3" * 64,
    "memory": "4" * 64,
    "use": "5" * 64,
}


def _dimension(key: str, *, status: str = STATUS_PASS) -> W02DimensionResult:
    """构造不含 private expected 的 1/1 bearing 结果。"""
    return W02DimensionResult(
        dimension_key=key,
        status=status,
        pass_count=int(status == STATUS_PASS),
        case_count=1,
        fail_count=int(status == STATUS_FAIL),
        ne_count=int(status == STATUS_NE),
        evidence_sha256=(key.encode("utf-8").hex() + "0" * 64)[:64],
    )


def _ablations() -> tuple[W02AblationResult, ...]:
    """每个消融只击穿自己对应的 bearing dimension。"""
    values = []
    for ablation, target in zip(W02_ABLATION_ORDER, _DIMENSIONS, strict=True):
        statuses = tuple(
            (dimension, STATUS_FAIL if dimension == target else STATUS_PASS)
            for dimension in _DIMENSIONS
        )
        values.append(W02AblationResult(
            ablation,
            target,
            statuses,
            STATUS_FAIL if target == "W-02-NEW_CONTENT_MORPHOLOGY" else STATUS_PASS,
            (ablation.encode("utf-8").hex() + "f" * 64)[:64],
        ))
    return tuple(values)


def _report() -> W02PrivateEvaluationReport:
    """构造满足 7 private paths、四维、生成和 host 零写的公开报告。"""
    return aggregate_w02_evaluation(
        dimensions=tuple(_dimension(key) for key in _DIMENSIONS),
        generation=_dimension(GENERATION_HARD_CONJUNCT),
        ablations=_ablations(),
        generation_consumer_disabled_status=STATUS_FAIL,
        host_digests_before=_DIGESTS,
        host_digests_after=dict(_DIGESTS),
        private_path_reads=7,
        private_payload_bytes=1234,
        held_out_observation_reads=20,
        evaluator_label_reads=20,
        ud_observation_reads=1,
        evaluator_label_writes=0,
        host_write_count=0,
        evidence_sha256="a" * 64,
    )


def test_preregistered_order_threshold_and_generation_hard_conjunct_are_fixed():
    """四维、四消融和生成附加硬合取不得由首轮结果重排。"""
    assert W02_EVALUATION_ORDER == (
        "W-02-BOUNDARY_WITHDRAWAL",
        "W-02-MULTI_CANDIDATE",
        "W-02-NEW_CONTENT_MORPHOLOGY",
        "W-02-OOV",
        GENERATION_HARD_CONJUNCT,
    )
    assert W02_ABLATION_ORDER == tuple(
        f"{item}-ABLATION" for item in _DIMENSIONS)
    report = _report()
    assert report.status == STATUS_PASS
    assert all(item.min_pass_numerator == item.min_pass_denominator == 1
               for item in report.dimensions)
    assert all(item.max_fail_count == 0 and item.ne_policy == "BLOCK"
               for item in report.dimensions)


@pytest.mark.parametrize("status", (STATUS_FAIL, STATUS_NE))
@pytest.mark.parametrize("index", range(5))
def test_any_bearing_or_generation_fail_ne_blocks_conjunction(status, index):
    """禁止用均值掩盖任一 bearing 或生成硬合取的 FAIL/NE。"""
    dimensions = [_dimension(key) for key in _DIMENSIONS]
    generation = _dimension(GENERATION_HARD_CONJUNCT)
    if index < 4:
        dimensions[index] = _dimension(_DIMENSIONS[index], status=status)
    else:
        generation = _dimension(GENERATION_HARD_CONJUNCT, status=status)
    report = aggregate_w02_evaluation(
        dimensions=tuple(dimensions),
        generation=generation,
        ablations=_ablations(),
        generation_consumer_disabled_status=STATUS_FAIL,
        host_digests_before=_DIGESTS,
        host_digests_after=dict(_DIGESTS),
        private_path_reads=7,
        private_payload_bytes=1,
        held_out_observation_reads=1,
        evaluator_label_reads=1,
        ud_observation_reads=1,
        evaluator_label_writes=0,
        host_write_count=0,
        evidence_sha256="b" * 64,
    )
    assert report.status == status


def test_report_rejects_nonorthogonal_ablation_and_host_or_label_write():
    """消融未击穿本维、host 漂移或 label 写都不能形成可发布报告。"""
    ablations = list(_ablations())
    first = ablations[0]
    ablations[0] = replace(first, dimension_statuses=tuple(
        (key, STATUS_PASS) for key in _DIMENSIONS))
    with pytest.raises(W02EvaluationError, match="ablation"):
        aggregate_w02_evaluation(
            dimensions=tuple(_dimension(key) for key in _DIMENSIONS),
            generation=_dimension(GENERATION_HARD_CONJUNCT),
            ablations=tuple(ablations),
            generation_consumer_disabled_status=STATUS_FAIL,
            host_digests_before=_DIGESTS,
            host_digests_after=dict(_DIGESTS),
            private_path_reads=7,
            private_payload_bytes=1,
            held_out_observation_reads=1,
            evaluator_label_reads=1,
            ud_observation_reads=1,
            evaluator_label_writes=0,
            host_write_count=0,
            evidence_sha256="c" * 64,
        )
    for changes in (
            {"host_digests_after": {**_DIGESTS, "core": "9" * 64}},
            {"evaluator_label_writes": 1},
            {"host_write_count": 1}):
        kwargs = {
            "dimensions": tuple(_dimension(key) for key in _DIMENSIONS),
            "generation": _dimension(GENERATION_HARD_CONJUNCT),
            "ablations": _ablations(),
            "generation_consumer_disabled_status": STATUS_FAIL,
            "host_digests_before": _DIGESTS,
            "host_digests_after": dict(_DIGESTS),
            "private_path_reads": 7,
            "private_payload_bytes": 1,
            "held_out_observation_reads": 1,
            "evaluator_label_reads": 1,
            "ud_observation_reads": 1,
            "evaluator_label_writes": 0,
            "host_write_count": 0,
            "evidence_sha256": "d" * 64,
        }
        kwargs.update(changes)
        with pytest.raises(W02EvaluationError):
            aggregate_w02_evaluation(**kwargs)


def test_failed_first_baseline_stays_reportable_without_ablation_exception():
    """首轮 baseline 已失败时不让消融形态遮蔽正式 FAIL 事实。"""
    dimensions = tuple(
        _dimension(key, status=STATUS_FAIL if index == 0 else STATUS_PASS)
        for index, key in enumerate(_DIMENSIONS)
    )
    non_degrading = tuple(replace(
        item,
        dimension_statuses=tuple(
            (key, STATUS_PASS) for key in _DIMENSIONS),
    ) for item in _ablations())
    report = aggregate_w02_evaluation(
        dimensions=dimensions,
        generation=_dimension(GENERATION_HARD_CONJUNCT),
        ablations=non_degrading,
        generation_consumer_disabled_status=STATUS_PASS,
        host_digests_before=_DIGESTS,
        host_digests_after=dict(_DIGESTS),
        private_path_reads=7,
        private_payload_bytes=1,
        held_out_observation_reads=1,
        evaluator_label_reads=1,
        ud_observation_reads=1,
        evaluator_label_writes=0,
        host_write_count=0,
        evidence_sha256="e" * 64,
    )
    assert report.status == STATUS_FAIL


def test_nonunique_observation_stem_is_ne_instead_of_guessed_or_raised():
    """公开 schema 未保证唯一 STEM 时，target 解析必须失败关闭为 NE 前态。"""
    base = {
        "analysis_units": [
            {"end": 1, "start": 0, "surface": "纸", "unit_kind": "STEM"},
        ],
        "construction_key": "suffix-hua-construction-v1",
    }
    target = evaluator_runtime._morph_target(base)
    assert target is not None and target.stem_surface == "纸"
    assert evaluator_runtime._morph_target({
        **base,
        "analysis_units": [
            *base["analysis_units"],
            {"end": 2, "start": 1, "surface": "化", "unit_kind": "STEM"},
        ],
    }) is None
    assert evaluator_runtime._morph_target({
        **base,
        "analysis_units": [
            {"end": 1, "start": 0, "surface": "纸", "unit_kind": "COMPONENT"},
        ],
    }) is None


def test_public_report_contains_only_counts_status_and_evidence_digests(tmp_path):
    """公开投影不得携带 private expected、surface、Observation 或标签对象。"""
    payload = _report().to_public_dict()
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    forbidden = (
        "expected_payload", "expected_surface", "accepted_surfaces",
        "raw_observation", "evaluator_label_payload",
    )
    assert not any(item in serialized for item in forbidden)
    assert payload["status"] == STATUS_PASS
    assert payload["private_reads"]["private_path_reads"] == 7
    assert payload["private_reads"]["held_out_observation_reads"] == 20
    assert payload["private_reads"]["evaluator_label_reads"] == 20


def test_first_private_evaluation_report_is_canonical_and_cannot_be_overwritten(
        tmp_path):
    """首轮公开摘要持久落盘，后续发布即使同字节也不得覆盖。"""
    path = tmp_path / "w02_private_evaluation_first_run.json"
    report = _report()
    publication = publish_w02_private_evaluation_report(path, report)
    original = path.read_bytes()
    assert original == canonical_json_bytes(report.to_public_dict())
    assert publication.size_bytes == len(original)
    assert publication.status == STATUS_PASS

    value, readback = read_w02_private_evaluation_report(path)
    assert value == report.to_public_dict()
    assert readback == publication
    with pytest.raises(W02EvaluationPublicationError, match="禁止覆盖"):
        publish_w02_private_evaluation_report(path, report)
    assert path.read_bytes() == original
