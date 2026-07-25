from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import pure_integer_ai.experiments.v02_scale_curve as curve_module
from pure_integer_ai.cognition.shared.types import LANG_ZH
from pure_integer_ai.experiments.chinese_semantic_kb_adapter import (
    PARSER_DECIMAL_TAB,
    PARSER_DOCUMENT,
    PARSER_RELATION_MARKER,
    PARSER_SURFACE_LINE,
    PARSER_SYMMETRIC_AT,
    PROFILES,
    build_manifest,
)
from pure_integer_ai.experiments.chinese_semantic_kb_curriculum import (
    SPLIT_TRAIN,
    CourseSplitPolicy,
    build_curriculum_artifacts,
)
from pure_integer_ai.experiments.data_manifest import write_manifest
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.v02_run_store import (
    HostProcessMemory,
    V02RunStore,
)
from pure_integer_ai.experiments.v02_scale_curve import (
    RuntimeLanguageProtocols,
    V02_HOTSPOT_EVENTS,
    V02ScaleConfig,
    _hotspot_review,
    _run_observe_curve,
    _write_summary,
    run_v02_scale_curve,
)
from pure_integer_ai.experiments.v02_scale_types import (
    LanguageProtocolSpec,
    LaneBudget,
    evaluate_lane_budget,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT


class _ManualClock:
    """提供可由伪 workload 显式推进的整数时钟。"""

    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        """返回当前测试时钟值。"""
        return self.value

    def advance(self, amount: int) -> None:
        """把测试时钟推进指定整数纳秒。"""
        self.value += amount


class _IncrementingClock:
    """每次采样递增一纳秒，供真实小课程生成非负阶段时长。"""

    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        """返回当前值后递增，保持所有采样严格确定。"""
        current = self.value
        self.value += 1
        return current


def _memory_sample() -> dict[str, int]:
    """返回固定工作集，并故意给出更高的进程历史峰值。"""
    return {
        "current_working_set_bytes": 123,
        "process_peak_working_set_bytes": 9_999,
    }


def _profile_line(profile) -> str:
    """为课程 adapter 的每种 parser 生成一条最小合法记录。"""
    if profile.parser_kind == PARSER_DOCUMENT:
        return "# V-02 测试来源"
    if profile.parser_kind == PARSER_RELATION_MARKER:
        return f"甲,{profile.relation_marker},乙"
    if profile.parser_kind == PARSER_SYMMETRIC_AT:
        return "上@下"
    if profile.parser_kind == PARSER_DECIMAL_TAB:
        return "很\t2.5"
    if profile.parser_kind == PARSER_SURFACE_LINE:
        return "基础项"
    raise AssertionError(profile.parser_kind)


def _build_tiny_course(tmp_path: Path) -> tuple[Path, Path]:
    """构建所有词形都落 train split 的最小版本化课程。"""
    raw_root = tmp_path / "raw"
    for profile in PROFILES:
        path = raw_root / profile.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_profile_line(profile) + "\n", encoding="utf-8")
    source_manifest, _ = build_manifest(
        raw_root,
        dataset_version="v02-fixture-v1",
        unicode_sequence_family=(91_001,),
    )
    source_path = tmp_path / "manifest" / "source.json"
    write_manifest(source_manifest, source_path, raw_root=raw_root)
    course_root = tmp_path / "course"
    build_curriculum_artifacts(
        source_manifest,
        raw_root,
        course_root,
        split_policy=CourseSplitPolicy(1, 1000, 0, 0),
    )
    return source_path, course_root


def _full_language_protocol() -> LanguageProtocolSpec:
    """构造覆盖 L-02/L-03/L-04/L-06/U-03 的完整开放协议。"""
    return LanguageProtocolSpec.from_json({
        "segmentation": {
            "hypothesis_kind_key": [92_000, 1],
            "lexical_match_reason_key": [92_000, 2],
            "oov_reason_key": [92_000, 3],
            "candidate_limit": 3,
        },
        "occurrence": {
            "candidate_relation_key": [92_010, 1],
            "speaker_relation_key": [92_010, 2],
        },
        "occurrence_order": {
            "relation_key": [92_020, 1],
        },
        "span": {
            "structure_relation_key": [92_030, 1],
            "constituent_relation_key": [92_030, 2],
            "occurrence_relation_key": [92_030, 3],
            "candidate_relation_key": [92_030, 4],
            "document_structure_key": [92_040, 1],
            "part_structure_key": [92_040, 2],
            "candidate_shape_namespace_key": [92_040, 3],
            "atomic_structure_key": [92_040, 4],
        },
        "boundary": {
            "hypothesis_kind_key": [92_050, 1],
            "document_structure_key": [92_050, 2],
            "candidate_structure_key": [92_050, 3],
            "anchor_structure_key": [92_050, 4],
            "candidate_shape_namespace_key": [92_050, 5],
            "selection_relation_key": [92_050, 6],
            "withdrawal_relation_key": [92_050, 7],
            "selection_clock_kind": 92_050_007,
        },
    })


def _config(tmp_path: Path, *, run_id: str = "v02-test",
            scales: tuple[int, ...] = (1,),
            run_provider: bool = False,
            run_observe: bool = True,
            run_curriculum: bool = False,
            measure_evaluation_clone: bool = False,
            measure_dump: bool = False,
            stop_after_scale: int | None = None) -> V02ScaleConfig:
    """构造可被单测按需替换路径的 V-02 配置。"""
    corpus = tmp_path / "corpus.txt"
    if not corpus.exists():
        corpus.write_text(
            "甲 乙\n\n丙 丁\n\n戊 己\n",
            encoding="utf-8",
        )
    return V02ScaleConfig(
        output_root=str(tmp_path / "benchmarks"),
        run_id=run_id,
        course_root=str(tmp_path / "unused-course"),
        source_manifest_path=str(tmp_path / "unused-source.json"),
        corpus_path=str(corpus),
        runtime_language=LANG_ZH,
        source_kind=SOURCE_BARE_TEXT,
        domain=1,
        visible_splits=(SPLIT_TRAIN,),
        source_namespace="v02-test-corpus",
        scales=scales,
        run_provider=run_provider,
        run_observe=run_observe,
        run_curriculum=run_curriculum,
        measure_evaluation_clone=measure_evaluation_clone,
        measure_dump=measure_dump,
        stop_after_scale=stop_after_scale,
    )


def test_run_store_preregistration_and_points_are_strictly_recoverable(
        tmp_path):
    """同 run 只接受完全相同的预注册和 point，拒绝事后改写。"""
    store = V02RunStore(tmp_path, "strict-run")
    preregistration = {"schema_version": 1, "budget": {"limit": 7}}
    store.preregister(preregistration)
    assert json.loads(store.preregistration_path.read_text(
        encoding="utf-8")) == preregistration
    store.preregister(preregistration)
    with pytest.raises(ValueError, match="预注册"):
        store.preregister({"schema_version": 1, "budget": {"limit": 8}})

    point = {"lane": "observe", "n": 1, "elapsed_ns": 3}
    store.write_point("observe", 1, point)
    assert store.has_point("observe", 1)
    assert store.read_point("observe", 1) == point
    store.write_point("observe", 1, point)
    with pytest.raises(ValueError, match="重算结果"):
        store.write_point(
            "observe", 1,
            {"lane": "observe", "n": 1, "elapsed_ns": 4},
        )


@pytest.mark.skipif(os.name != "nt", reason="仅 Windows 提供工作集适配")
def test_host_process_memory_reads_nonzero_windows_working_set() -> None:
    """Windows 外层诊断不得把 API 签名错误静默伪装成零内存。"""
    sample = HostProcessMemory()()
    assert sample["current_working_set_bytes"] > 0
    assert sample["process_peak_working_set_bytes"] >= (
        sample["current_working_set_bytes"])


def test_budget_failure_is_reported_per_dimension() -> None:
    """单个查询维度失败时不得被其他宽松维度或综合均分掩盖。"""
    budget = LaneBudget(
        n=2,
        max_total_elapsed_ns=100,
        max_query_calls_per_item=2,
        max_growth_rows_per_item=10,
        max_candidates_per_item=10,
        max_peak_working_set_bytes=1000,
    )
    result = evaluate_lane_budget(
        budget,
        elapsed_ns=1,
        query_calls=5,
        growth_rows=1,
        candidate_count=1,
        peak_working_set_bytes=1,
    )
    assert not result["passed"]
    assert result["checks"] == {
        "elapsed": True,
        "query_calls_per_item": False,
        "growth_rows_per_item": True,
        "candidates_per_item": True,
        "peak_working_set_bytes": True,
    }


def test_language_protocol_rejects_orphans_and_unknown_fields() -> None:
    """协议依赖不完整或字段拼错时必须 fail closed，不能静默降级。"""
    with pytest.raises(ValueError, match="speaker"):
        LanguageProtocolSpec(occurrence_speaker_relation_key=(1, 2))
    with pytest.raises(ValueError, match="Span"):
        LanguageProtocolSpec(span_atomic_structure_key=(1, 3))
    with pytest.raises(ValueError, match="未知字段"):
        LanguageProtocolSpec.from_json({"segmention": None})
    with pytest.raises(ValueError, match="未知字段"):
        LanguageProtocolSpec.from_json({
            "segmentation": None,
            "occurrence": {"candidate_relation_key": [1], "typo": [2]},
        })
    with pytest.raises(ValueError, match="句界"):
        LanguageProtocolSpec(boundary_hypothesis_kind_key=(1, 4))
    protocol = _full_language_protocol()
    assert LanguageProtocolSpec.from_json(
        protocol.to_json()) == protocol


def test_hotspot_review_keeps_missing_categories_explicit() -> None:
    """未被当前 workload 触发的历史热点必须保留为 missing，不能默认通过。"""
    review = _hotspot_review({
        "event_totals": [{"kind": "hotspot.normalize", "count": 7}],
    })
    assert review["all_observed"] is False
    assert review["observed"] == ["hotspot.normalize"]
    assert set(review["missing"]) == (
        set(V02_HOTSPOT_EVENTS) - {"hotspot.normalize"})


def test_direct_observe_excludes_fixed_costs_and_process_history(
        tmp_path, monkeypatch):
    """direct 曲线只累计 item workload，不计 provider、摘要、clone 或 dump。"""
    config = _config(
        tmp_path,
        scales=(1, 2),
        measure_evaluation_clone=True,
        measure_dump=True,
    )
    store = V02RunStore(config.output_root, config.run_id)
    clock = _ManualClock()

    class _Runner:
        """每个 item 只推进十纳秒，不执行真实认知写入。"""

        def run_round(self, _ctx, _item, _stage, _round_id):
            """模拟单个 direct observe workload。"""
            clock.advance(10)
            return None

    monkeypatch.setattr(curve_module, "DefaultRoundRunner", _Runner)
    monkeypatch.setattr(
        curve_module,
        "_apply_word_form_providers",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(
        curve_module,
        "install_language_graph_protocols",
        lambda *_args, **_kwargs: None,
    )

    def _clone(*_args, **_kwargs):
        """模拟一次一百纳秒的 clone 固定成本。"""
        clock.advance(100)
        return {"elapsed_ns": 100}

    def _dump(*_args, **_kwargs):
        """模拟一次一百纳秒的 dump 固定成本。"""
        clock.advance(100)
        return {"elapsed_ns": 100}

    monkeypatch.setattr(curve_module, "_measure_clone", _clone)
    monkeypatch.setattr(curve_module, "_measure_dump", _dump)

    clock.advance(500)
    _run_observe_curve(
        config,
        RuntimeLanguageProtocols(),
        store,
        make_train_context(DictBackend()),
        object(),
        None,
        clock_ns=clock,
        memory_source=_memory_sample,
    )

    first = store.read_point("observe", 1)
    second = store.read_point("observe", 2)
    assert first["elapsed_ns"] == 10
    assert second["elapsed_ns"] == 20
    assert first["evaluation_clone"] is None
    assert first["dump"] is None
    assert second["evaluation_clone"]["elapsed_ns"] == 100
    assert second["dump"]["elapsed_ns"] == 100
    assert second["budget"]["observed"][
        "peak_working_set_bytes"] == 123


def test_direct_observe_source_identity_does_not_depend_on_run_id(
        tmp_path, monkeypatch):
    """实验输出名变化时，同一输入项必须保持同一 SourceRef。"""
    observed_refs = []

    class _Runner:
        """只截获 direct observe 已分配的来源身份。"""

        def run_round(self, _ctx, item, _stage, _round_id):
            """记录来源身份，不执行任何认知写入。"""
            observed_refs.append(item.source_ref)
            return None

    monkeypatch.setattr(curve_module, "DefaultRoundRunner", _Runner)
    monkeypatch.setattr(
        curve_module,
        "_apply_word_form_providers",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(
        curve_module,
        "install_language_graph_protocols",
        lambda *_args, **_kwargs: None,
    )

    for run_id in ("semantic-run-a", "semantic-run-b"):
        config = _config(tmp_path, run_id=run_id)
        _run_observe_curve(
            config,
            RuntimeLanguageProtocols(),
            V02RunStore(config.output_root, config.run_id),
            make_train_context(DictBackend()),
            object(),
            None,
            clock_ns=_IncrementingClock(),
            memory_source=_memory_sample,
        )

    assert observed_refs[0] is not None
    assert observed_refs[0] == observed_refs[1]


def test_provider_only_summary_is_not_complete_before_provider_result(
        tmp_path):
    """provider-only run 在固定成本结果落盘前不得提前声称全部完成。"""
    config = _config(
        tmp_path,
        run_provider=True,
        run_observe=False,
        run_curriculum=False,
    )
    store = V02RunStore(config.output_root, config.run_id)
    _write_summary(config, store)
    summary = json.loads(store.summary_path.read_text(encoding="utf-8"))
    assert summary["expected"]["provider"] is True
    assert summary["completed"]["provider"] is False
    assert summary["all_expected_complete"] is False
    assert summary["budget_passed"] is False


def test_stop_after_scale_limits_execution_without_changing_frozen_config(
        tmp_path, monkeypatch):
    """单次施工前缀可变化，但完整预注册 scales 和预算身份必须不变。"""
    first = _config(tmp_path, scales=(1, 2), stop_after_scale=1)
    resumed = _config(tmp_path, scales=(1, 2), stop_after_scale=2)
    monkeypatch.setattr(curve_module, "_input_manifest", lambda _config: {})
    monkeypatch.setattr(
        curve_module, "_implementation_manifest", lambda: {})
    assert first.execution_scales() == (1,)
    assert resumed.execution_scales() == (1, 2)
    assert curve_module._preregistration(first) == (
        curve_module._preregistration(resumed))


def test_tiny_course_runs_end_to_end_and_resumes_without_rebuilding(
        tmp_path, monkeypatch):
    """最小真实课程须先预注册，再跑 observe/curriculum，并可整 run 恢复。"""
    source_path, course_root = _build_tiny_course(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("基础项 甲\n", encoding="utf-8")
    config = V02ScaleConfig(
        output_root=str(tmp_path / "benchmarks"),
        run_id="tiny-e2e",
        course_root=str(course_root),
        source_manifest_path=str(source_path),
        corpus_path=str(corpus),
        runtime_language=LANG_ZH,
        source_kind=SOURCE_BARE_TEXT,
        domain=1,
        visible_splits=(SPLIT_TRAIN,),
        source_namespace="tiny-e2e-corpus",
        scales=(1,),
        protocol_spec=_full_language_protocol(),
        run_provider=False,
        run_observe=True,
        run_curriculum=True,
        measure_evaluation_clone=False,
        measure_dump=False,
    )
    preregistration_path = (
        Path(config.output_root) / config.run_id / "preregistered.json")
    original_builder = curve_module.build_word_form_providers

    def _guarded_builder(**kwargs):
        """确认任何课程 artifact 扫描前预注册文件已经原子落盘。"""
        assert preregistration_path.is_file()
        return original_builder(**kwargs)

    monkeypatch.setattr(
        curve_module, "build_word_form_providers", _guarded_builder)
    summary_path = run_v02_scale_curve(
        config,
        clock_ns=_IncrementingClock(),
        memory_source=_memory_sample,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["all_expected_complete"] is True
    assert summary["completed"] == {
        "provider": False,
        "observe": [1],
        "curriculum": [1],
    }
    observe_growth = {
        row["table"]: row["growth"]
        for row in summary["points"]["observe"][0]["table_growth"]
    }
    assert observe_growth["occurrence"] > 0
    assert observe_growth["span"] > 0
    assert "hotspot.normalize" in summary["hotspot_review"]["observed"]
    assert not (summary_path.parent / "provider.json").exists()

    def _unexpected_builder(**_kwargs):
        """恢复路径若重建 provider 就立即暴露。"""
        raise AssertionError("已完成 run 不应重建 provider")

    monkeypatch.setattr(
        curve_module, "build_word_form_providers", _unexpected_builder)
    resumed = run_v02_scale_curve(
        config,
        clock_ns=_IncrementingClock(),
        memory_source=_memory_sample,
    )
    assert resumed == summary_path
