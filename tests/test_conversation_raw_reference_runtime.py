"""DLG-RAW-05B 公开双命题/reference 实际运行回归。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PublicFrameCatalog,
)
from pure_integer_ai.experiments.conversation_public_reference_catalog import (
    load_public_reference_frame_catalog_from_closure,
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
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    DLG_RAW_REJECT_OUTPUT_BUDGET,
    DLG_RAW_REJECT_RUNTIME,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.conversation_raw_lexical_ingress import (
    ingress_raw_lexical_frame,
)
from pure_integer_ai.experiments.conversation_raw_reference_runtime import (
    ConversationRawReferenceRuntimeError,
    run_public_reference_frame,
)
from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationCourseLoader,
    AliasRelationPreflightCache,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_parser import (
    GroundedAnswerReferenceSurfaceParser,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_EXPECTED_OUTPUT = "北川站东门于2024年启用。前述启用事项已登记入档。"


@pytest.fixture(scope="module")
def public_source_payload_closure() -> PublicSourcePayloadClosureV1:
    """经唯一 host adapter 构造完整公开资源闭包。"""
    return load_public_source_payload_closure_from_root(_REPOSITORY_ROOT)


@pytest.fixture(scope="module")
def public_v3_catalog(
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> PublicFrameCatalog:
    """从闭包加载唯一公开 V3 manifest，不触及 private、formal 或 held-out。"""
    return load_public_reference_frame_catalog_from_closure(
        public_source_payload_closure,
    )


def _ingress(catalog: PublicFrameCatalog, ordinal: int):
    """由冻结 raw u8 surface 构造一次真实 DLG-RAW-01 ingress。"""
    assert len(catalog.frames) == 1
    frame = catalog.frames[0]
    result = ingress_raw_lexical_frame(
        intake_raw_conversation_vector(frame.surface_bytes),
        catalog,
        (65001, 54, 90, ordinal),
    )
    assert result.accepted is True
    return result


@pytest.fixture(scope="module")
def actual_v3_run(
        public_v3_catalog: PublicFrameCatalog,
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        ):
    """只运行一次真实 V3，并保留实际 G-04 reference recovery 供多个断言复用。"""
    original = GroundedAnswerReferenceSurfaceParser.recover_reference
    recovered = []

    def capture_recovery(self, request, preview):
        """保留真实 parser 的回读结果，不替换任何 recovery 语义。"""
        value = original(self, request, preview)
        recovered.append(value)
        return value

    patch = pytest.MonkeyPatch()
    patch.setattr(
        GroundedAnswerReferenceSurfaceParser,
        "recover_reference",
        capture_recovery,
    )
    try:
        run = run_public_reference_frame(
            _ingress(public_v3_catalog, 1),
            source_payload_closure=public_source_payload_closure,
        )
    finally:
        patch.undo()
    return run, tuple(recovered)


def test_v3_actual_runtime_renders_two_sentences_and_recovers_antecedent(
        actual_v3_run) -> None:
    """实际 G-03/G-04 必须生成双句，并从同次 preview 回读唯一前序命题。"""
    run, recovered = actual_v3_run

    assert run.complete is True
    assert run.generation is not None and run.generation.complete is True
    assert run.generation.rendered is not None
    assert "".join(chr(item) for item in run.generation.rendered.units) == (
        _EXPECTED_OUTPUT)
    assert run.postcheck is not None and run.postcheck.complete is True
    assert run.selection_commit is None
    assert run.outcome_commit is None

    assert len(recovered) == 1
    reference = recovered[0]
    assert reference.antecedent is not None
    assert reference.antecedent_candidate_key
    assert reference.reference_proposal_key
    assert reference.surface_proposal_key


def test_v3_preflight_cache_preserves_canonical_output_and_fresh_backend(
        monkeypatch: pytest.MonkeyPatch,
        public_v3_catalog: PublicFrameCatalog,
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> None:
    """R-01 warm hit 只跳过隔离预演，V3 输出和 backend 生命周期仍独立。"""
    import pure_integer_ai.experiments.conversation_raw_reference_runtime as runtime

    preflight_calls = 0
    original_preflight = AliasRelationCourseLoader._preflight_isolated

    def counted_preflight(self):
        nonlocal preflight_calls
        preflight_calls += 1
        return original_preflight(self)

    monkeypatch.setattr(
        AliasRelationCourseLoader,
        "_preflight_isolated",
        counted_preflight,
    )
    created: list[object] = []
    closed: list[object] = []

    class TrackingDictBackend(runtime.DictBackend):
        """仅记录两轮 V3 实际 backend，不改变存储行为。"""

        def __init__(self) -> None:
            super().__init__()
            created.append(self)

        def close(self) -> None:
            closed.append(self)
            super().close()

    monkeypatch.setattr(runtime, "DictBackend", TrackingDictBackend)
    ingress = _ingress(public_v3_catalog, 31)
    cache = AliasRelationPreflightCache()
    cold = run_public_frame_answer(
        ingress,
        source_payload_closure=public_source_payload_closure,
        preflight_cache=cache,
    )
    warm = run_public_frame_answer(
        ingress,
        source_payload_closure=public_source_payload_closure,
        preflight_cache=cache,
    )

    assert cold.accepted is warm.accepted is True
    assert cold.canonical_record() == warm.canonical_record()
    assert cold.run is not None and warm.run is not None
    assert cold.run.selection_commit is None
    assert cold.run.outcome_commit is None
    assert warm.run.selection_commit is None
    assert warm.run.outcome_commit is None
    assert preflight_calls == 1
    assert cache.entry_count == 1
    assert cache.miss_count == 1
    assert cache.hit_count == 1
    assert len(created) == len(closed) == 2
    assert len({id(item) for item in created}) == 2


def test_v3_output_budget_rejects_without_leaking_actual_rendered_units(
        monkeypatch: pytest.MonkeyPatch,
        public_v3_catalog: PublicFrameCatalog,
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        actual_v3_run,
        ) -> None:
    """真实 V3 rendered units 超预算时，RAW-02 只能返回零输出拒绝 record。"""
    import pure_integer_ai.experiments.conversation_raw_answer_runtime as runtime

    run, _recovered = actual_v3_run
    constrained_frame = replace(
        public_v3_catalog.frames[0],
        recipe=replace(public_v3_catalog.frames[0].recipe, output_max_bytes=1),
    )
    constrained_catalog = PublicFrameCatalog(
        public_v3_catalog.source_sha256,
        (constrained_frame,),
    )

    def completed_actual(*_args: object, **_kwargs: object):
        """重放上例已实际完成的同一不可变 QuestionAnswerRun。"""
        return run

    monkeypatch.setattr(runtime, "_run_actual_frame", completed_actual)
    result = run_public_frame_answer(
        _ingress(constrained_catalog, 2),
        source_payload_closure=public_source_payload_closure,
    )

    assert result.result_code == DLG_RAW_REJECT_OUTPUT_BUDGET
    assert result.accepted is False
    assert result.run is None
    assert result.output_scalars == result.output_bytes == ()
    assert result.output_readback is None
    assert result.persistent_state_delta == ()


def test_v3_lexical_source_drift_rejects_before_runtime_construction(
        monkeypatch: pytest.MonkeyPatch,
        public_v3_catalog: PublicFrameCatalog,
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> None:
    """任一 closure 词汇 source 漂移也必须在 backend 前 fail closed。"""
    frame = public_v3_catalog.frames[0]
    source_record = next(
        item for item in frame.source_records
        if item.relative_path != frame.recipe.course_relative_path)
    source_key = source_record.relative_path.encode("ascii")
    drifted_closure = build_public_source_payload_closure_v1(tuple(
        public_source_payload_record_from_u8_v1(
            record.logical_key,
            record.raw_payload + b"x",
        ) if record.logical_key == source_key else record
        for record in public_source_payload_closure.records
    ))

    import pure_integer_ai.experiments.conversation_raw_reference_runtime as runtime

    def forbidden_backend() -> object:
        raise AssertionError("source drift 不得构造 run-local backend")

    monkeypatch.setattr(runtime, "DictBackend", forbidden_backend)
    with pytest.raises(ConversationRawReferenceRuntimeError, match="公开 source"):
        run_public_reference_frame(
            _ingress(public_v3_catalog, 3),
            source_payload_closure=drifted_closure,
        )


def test_v3_course_drift_rejects_without_output_or_state_leak(
        public_v3_catalog: PublicFrameCatalog,
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> None:
    """课程 closure 漂移经 RAW-02 拒绝后，不得泄露 run、输出或状态副作用。"""
    frame = public_v3_catalog.frames[0]
    course_key = frame.recipe.course_relative_path.encode("ascii")
    drifted_closure = build_public_source_payload_closure_v1(tuple(
        public_source_payload_record_from_u8_v1(
            record.logical_key,
            record.raw_payload + b"x",
        ) if record.logical_key == course_key else record
        for record in public_source_payload_closure.records
    ))
    result = run_public_frame_answer(
        _ingress(public_v3_catalog, 4),
        source_payload_closure=drifted_closure,
    )

    assert result.result_code == DLG_RAW_REJECT_RUNTIME
    assert result.accepted is False
    assert result.run is None
    assert result.output_scalars == result.output_bytes == ()
    assert result.output_readback is None
    assert result.persistent_state_delta == ()
