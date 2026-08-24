"""DLG-RAW-02 真实 F-00/G-03/G-04 首句运行桥验证。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PublicFrameCatalog,
    load_public_frame_catalog,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadClosureV1,
    build_public_source_payload_closure_v1,
    public_source_payload_record_from_u8_v1,
)
from pure_integer_ai.experiments.conversation_raw_answer_runtime import (
    run_public_frame_answer,
)
from pure_integer_ai.experiments.conversation_raw_course_prepare import (
    PublicCoursePreparationCache,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    DLG_RAW_REJECT_LEXICAL_MISS,
    DLG_RAW_REJECT_OUTPUT_BUDGET,
    DLG_RAW_REJECT_RUNTIME,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.conversation_raw_lexical_ingress import (
    ingress_raw_lexical_frame,
)
from pure_integer_ai.storage.backend import DictBackend


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_OCCURRENCE_KEY = (65001, 33, 1)


@pytest.fixture(scope="module")
def public_source_payload_closure() -> PublicSourcePayloadClosureV1:
    """仅经 host adapter 构造完整公开资源闭包，core 不接收其物理根。"""
    return load_public_source_payload_closure_from_root(_REPOSITORY_ROOT)


@pytest.fixture(scope="module")
def public_catalog(
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> PublicFrameCatalog:
    """从唯一公开 payload closure 加载首句 frame。"""
    return load_public_frame_catalog(public_source_payload_closure)


def _first_turn_ingress(catalog: PublicFrameCatalog):
    """由 catalog 冻结的 input byte vector 构造真实冷启动请求。"""
    return ingress_raw_lexical_frame(
        intake_raw_conversation_vector(catalog.frames[0].surface_bytes),
        catalog,
        _OCCURRENCE_KEY,
    )


def test_actual_runtime_generates_the_public_first_turn(
        public_catalog: PublicFrameCatalog,
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> None:
    """输出只能来自本次真实 G-03 rendered units，且 G-04 必须完成。"""
    result = run_public_frame_answer(
        _first_turn_ingress(public_catalog),
        source_payload_closure=public_source_payload_closure,
    )

    assert result.result_code == DLG_RAW_ACCEPT
    assert result.accepted is True
    assert result.run is not None
    assert result.run.complete is True
    assert result.run.generation is not None
    assert result.run.generation.rendered is not None
    assert result.run.postcheck is not None
    assert result.run.postcheck.complete is True
    assert result.output_scalars == result.run.generation.rendered.units
    assert bytes(result.output_bytes).decode("utf-8") == "北川站东门于2024年启用。"
    assert result.persistent_state_delta == ()
    assert result.run.selection_commit is None
    assert result.run.outcome_commit is None


def test_lexical_miss_does_not_construct_or_run_answer_runtime(
        monkeypatch: pytest.MonkeyPatch,
        public_catalog: PublicFrameCatalog,
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> None:
    """未学 surface 只返回 RAW-01 miss，绝不能出现语言 fallback。"""
    import pure_integer_ai.experiments.conversation_raw_answer_runtime as runtime

    def forbidden_runtime(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("lexical miss 不得构造实际 answer runtime")

    monkeypatch.setattr(runtime, "_run_actual", forbidden_runtime)
    intake = intake_raw_conversation_vector(
        (*public_catalog.frames[0].surface_bytes, 0x20))
    ingress = ingress_raw_lexical_frame(
        intake,
        public_catalog,
        _OCCURRENCE_KEY,
    )
    result = run_public_frame_answer(
        ingress,
        source_payload_closure=public_source_payload_closure,
    )

    assert ingress.result_code == DLG_RAW_REJECT_LEXICAL_MISS
    assert result.result_code == DLG_RAW_REJECT_LEXICAL_MISS
    assert result.run is None
    assert result.output_scalars == result.output_bytes == ()
    assert result.persistent_state_delta == ()


def test_output_budget_rejects_after_real_runtime_without_output_leak(
        public_catalog: PublicFrameCatalog,
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> None:
    """预算门在真实 G-03/G-04 后拒绝，并且拒绝 record 不携带半句输出。"""
    frame = public_catalog.frames[0]
    constrained_frame = replace(
        frame,
        recipe=replace(frame.recipe, output_max_bytes=1),
    )
    constrained_catalog = PublicFrameCatalog(
        public_catalog.source_sha256,
        (constrained_frame,),
    )

    result = run_public_frame_answer(
        _first_turn_ingress(constrained_catalog),
        source_payload_closure=public_source_payload_closure,
    )

    assert result.result_code == DLG_RAW_REJECT_OUTPUT_BUDGET
    assert result.accepted is False
    assert result.run is None
    assert result.output_scalars == result.output_bytes == ()
    assert result.persistent_state_delta == ()


def test_prepared_course_cache_keeps_actual_answer_record_identical(
        monkeypatch: pytest.MonkeyPatch,
        public_catalog: PublicFrameCatalog,
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> None:
    """热 cache 只能省重复 value 准备，不能改变同一 RAW-02 结果。"""
    import pure_integer_ai.experiments.conversation_raw_answer_runtime as runtime

    created: list[DictBackend] = []
    closed: list[DictBackend] = []

    class TrackingDictBackend(DictBackend):
        """记录本专项中每轮实际 backend 生命周期，不替换任何存储行为。"""

        def __init__(self) -> None:
            super().__init__()
            created.append(self)

        def close(self) -> None:
            closed.append(self)
            super().close()

    monkeypatch.setattr(runtime, "DictBackend", TrackingDictBackend)
    cache = PublicCoursePreparationCache()
    cold = run_public_frame_answer(
        _first_turn_ingress(public_catalog),
        source_payload_closure=public_source_payload_closure,
        preparation_cache=cache,
    )
    hot = run_public_frame_answer(
        _first_turn_ingress(public_catalog),
        source_payload_closure=public_source_payload_closure,
        preparation_cache=cache,
    )

    assert cold.accepted is hot.accepted is True
    assert cold.output_bytes == hot.output_bytes
    assert cold.canonical_record() == hot.canonical_record()
    assert cache.entry_count == 1
    assert cache.miss_count == 1
    assert cache.hit_count == 1

    cache.clear()
    rebuilt = run_public_frame_answer(
        _first_turn_ingress(public_catalog),
        source_payload_closure=public_source_payload_closure,
        preparation_cache=cache,
    )
    assert rebuilt.canonical_record() == cold.canonical_record()
    assert cache.entry_count == 1
    assert cache.miss_count == 2
    assert cache.hit_count == 1
    assert len(created) == len(closed) == 3
    assert len({id(item) for item in created}) == 3
    assert {id(item) for item in closed} == {id(item) for item in created}


def test_course_drift_after_cache_fill_still_rejects_current_bytes(
        monkeypatch: pytest.MonkeyPatch,
        public_catalog: PublicFrameCatalog,
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> None:
    """cache 命中前仍须核验当前 closure，漂移 payload 不得复用旧派生值。"""
    cache = PublicCoursePreparationCache()
    first = run_public_frame_answer(
        _first_turn_ingress(public_catalog),
        source_payload_closure=public_source_payload_closure,
        preparation_cache=cache,
    )
    assert first.accepted is True
    course_key = public_catalog.frames[0].recipe.course_relative_path.encode("ascii")
    drifted_closure = build_public_source_payload_closure_v1(tuple(
        public_source_payload_record_from_u8_v1(
            record.logical_key,
            record.raw_payload + b"x",
        ) if record.logical_key == course_key else record
        for record in public_source_payload_closure.records
    ))

    import pure_integer_ai.experiments.conversation_raw_answer_runtime as runtime

    def forbidden_backend() -> object:
        raise AssertionError("closure 漂移不得构造 run-local backend")

    monkeypatch.setattr(runtime, "DictBackend", forbidden_backend)

    drifted = run_public_frame_answer(
        _first_turn_ingress(public_catalog),
        source_payload_closure=drifted_closure,
        preparation_cache=cache,
    )

    assert drifted.result_code == DLG_RAW_REJECT_RUNTIME
    assert drifted.accepted is False
    assert drifted.output_scalars == drifted.output_bytes == ()
    assert cache.entry_count == 1
    assert cache.miss_count == 1
    assert cache.hit_count == 0
