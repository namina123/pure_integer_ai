"""DLG-RAW-05A 的公开 V2 non-answer runtime 回归。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentDecision,
    AnswerContentSelector,
)
from pure_integer_ai.cognition.shared.identity import (
    minimal_instruction_identity,
)
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PUBLIC_FRAME_CATALOG_LOGICAL_KEY_V1,
    PublicFrameCatalog,
    load_public_frame_catalog,
    materialize_public_frame_candidate,
)
from pure_integer_ai.experiments.conversation_public_response_act_catalog import (
    load_public_response_act_frame_catalog_from_closure,
    materialize_public_response_act_planning_from_closure,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadClosureV1,
    build_public_source_payload_closure_v1,
    public_source_payload_record_from_u8_v1,
)
from pure_integer_ai.experiments.conversation_raw_course_prepare import (
    PublicCoursePreparationCache,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_raw_dialogue_session import (
    run_public_frame_dialogue_turn,
    start_public_frame_dialogue,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    DLG_RAW_REJECT_OUTPUT_BUDGET,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.conversation_raw_answer_runtime import (
    run_public_frame_answer,
)
from pure_integer_ai.experiments.conversation_raw_lexical_ingress import (
    ingress_raw_lexical_frame,
)
from pure_integer_ai.experiments.conversation_raw_response_act_runtime import (
    ConversationRawResponseActRuntimeError,
    run_public_response_act_frame,
)
from pure_integer_ai.storage.backend import DictBackend


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_V2_CATALOG_LOGICAL_KEY = (
    b"data/ph2/dlg_raw_public_response_act_frame_v2.jsonl.sample")
_V3_DERIVED_CATALOG_LOGICAL_KEY = (
    b"data/ph2/dlg_raw_public_derived_frame_v3.jsonl.sample")


# object-model: test-double; scope=test-only
class _FixedPolicy:
    """只在 G-01 family gate 回归中注入一个已构造的 policy decision。"""

    def __init__(self, decision: AnswerContentDecision) -> None:
        self.decision = decision

    def select(self, _planning: object, _artifacts: tuple[object, ...]) -> AnswerContentDecision:
        """返回固定决策；selector 仍执行其真实 Evidence 约束。"""
        return self.decision


@pytest.fixture(scope="module")
def public_source_payload_closure() -> PublicSourcePayloadClosureV1:
    """只在 host 边界读取完整公开资源，runtime 仅消费闭包。"""
    return load_public_source_payload_closure_from_root(_REPOSITORY_ROOT)


@pytest.fixture(scope="module")
def public_v2_catalog(
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> PublicFrameCatalog:
    """从完整公开闭包加载唯一 V2 manifest。"""
    return load_public_response_act_frame_catalog_from_closure(
        public_source_payload_closure,
        _V2_CATALOG_LOGICAL_KEY,
    )


def _frame(catalog, frame_key: str):
    """按稳定 V2 frame key 获取一条公开 cold-start frame。"""
    matches = tuple(item for item in catalog.frames if item.frame_key == frame_key)
    assert len(matches) == 1
    return matches[0]


def _ingress(catalog, frame_key: str, ordinal: int):
    """从本 frame 的 raw u8 surface 形成一次真实 RAW-01 ingress。"""
    frame = _frame(catalog, frame_key)
    return ingress_raw_lexical_frame(
        intake_raw_conversation_vector(frame.surface_bytes),
        catalog,
        (65001, 52, 90, ordinal),
    )


@pytest.mark.parametrize(("frame_key", "expected_act", "expected_surface"), (
    (
        "dlg-raw-public-v2-unknown-budget",
        "UNKNOWN",
        "现有来源没有提供所问信息。",
    ),
    (
        "dlg-raw-public-v2-clarify-site",
        "CLARIFY",
        "存在多个候选，请补充更明确的限定。",
    ),
    (
        "dlg-raw-public-v2-conflict-date",
        "CONFLICT",
        "来源说法彼此矛盾，无法给出确定答案。",
    ),
))
def test_v2_actual_runtime_selects_and_renders_the_three_learned_nonanswer_acts(
        public_v2_catalog,
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        frame_key: str,
        expected_act: str,
        expected_surface: str,
        ) -> None:
    """同次 G-01 决定 act，lowest learned pattern 经真实 G-03/G-04 才可返回。"""
    import pure_integer_ai.experiments.conversation_raw_response_act_runtime as runtime

    run = run_public_response_act_frame(
        _ingress(public_v2_catalog, frame_key, 1),
        source_payload_closure=public_source_payload_closure,
    )
    content = runtime._generation_protocols(run.request.target_branch)[0]
    observation = run.postcheck.parsed.observation

    assert run.complete is True
    assert run.status == getattr(content, expected_act.lower())
    assert run.selection is not None
    assert run.selection.stance == run.status
    assert run.generation is not None and run.generation.complete
    assert run.generation.rendered is not None
    assert "".join(chr(item) for item in run.generation.rendered.units) == expected_surface
    assert run.postcheck is not None and run.postcheck.complete
    assert observation is not None
    assert observation.propositions == ()
    assert observation.cited_sources == ()
    assert run.selection_commit is None
    assert run.outcome_commit is None


def test_v3_derived_omission_uses_the_existing_unknown_runtime(
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> None:
    """派生输入只能改变 ingress frame，不得引入答案或 response-act 分支。"""
    catalog = load_public_response_act_frame_catalog_from_closure(
        public_source_payload_closure,
        _V3_DERIVED_CATALOG_LOGICAL_KEY,
    )
    run = run_public_response_act_frame(
        _ingress(catalog, "dlg-raw-public-v3-derived-unknown-budget-omission", 98),
        source_payload_closure=public_source_payload_closure,
    )
    assert run.complete is True
    assert run.selection is not None
    assert run.status == run.selection.stance
    assert run.generation is not None and run.generation.rendered is not None
    assert "".join(chr(item) for item in run.generation.rendered.units) == (
        "现有来源没有提供所问信息。")
    assert run.selection_commit is None
    assert run.outcome_commit is None


def test_v4_contextual_ellipsis_uses_existing_unknown_runtime_after_actual_anchor(
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> None:
    """V4 ingress 只在实际 typed anchor 后进入既有 UNKNOWN runtime。"""
    runtime = build_public_dialogue_runtime_v1(public_source_payload_closure)
    catalog = runtime.active_catalog
    anchor = _frame(
        catalog,
        "dlg-raw-public-v3-derived-unknown-budget-omission",
    )
    contextual = _frame(
        catalog,
        "dlg-raw-public-v4-contextual-unknown-budget-ellipsis",
    )
    first = run_public_frame_dialogue_turn(
        start_public_frame_dialogue((65001, 52, 99, 1)),
        anchor.surface_bytes,
        runtime,
    )
    assert first.answer.result_code == DLG_RAW_ACCEPT
    context_read = first.after.context.read(1)
    ingress = ingress_raw_lexical_frame(
        intake_raw_conversation_vector(contextual.surface_bytes),
        catalog,
        (65001, 52, 99, 2),
        context_read=context_read,
    )
    assert ingress.accepted is True
    assert ingress.context_read == context_read
    run = run_public_response_act_frame(
        ingress,
        source_payload_closure=public_source_payload_closure,
    )
    assert run.complete is True
    assert run.selection is not None
    assert run.status == run.selection.stance
    assert run.generation is not None and run.generation.rendered is not None
    assert "".join(chr(item) for item in run.generation.rendered.units) == (
        "现有来源没有提供所问信息。")
    assert run.selection_commit is None
    assert run.outcome_commit is None


def test_v2_cache_keeps_run_identity_but_each_call_owns_and_closes_a_fresh_backend(
        monkeypatch: pytest.MonkeyPatch,
        public_v2_catalog,
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> None:
    """preparation cache 只能复用内容锁派生值，不能复用 R-01/G-03 runtime owner。"""
    import pure_integer_ai.experiments.conversation_raw_response_act_runtime as runtime

    created: list[DictBackend] = []
    closed: list[DictBackend] = []

    class TrackingDictBackend(DictBackend):
        """记录专项中的 backend 生命周期，不改变实际 DictBackend 语义。"""

        def __init__(self) -> None:
            super().__init__()
            created.append(self)

        def close(self) -> None:
            closed.append(self)
            super().close()

    monkeypatch.setattr(runtime, "DictBackend", TrackingDictBackend)
    cache = PublicCoursePreparationCache()
    ingress = _ingress(
        public_v2_catalog,
        "dlg-raw-public-v2-clarify-site",
        2,
    )
    cold = run_public_response_act_frame(
        ingress,
        source_payload_closure=public_source_payload_closure,
        preparation_cache=cache,
    )
    hot = run_public_response_act_frame(
        ingress,
        source_payload_closure=public_source_payload_closure,
        preparation_cache=cache,
    )

    assert cold.stable_key() == hot.stable_key()
    assert cache.entry_count == 1
    assert cache.miss_count == 1
    assert cache.hit_count == 1
    assert len(created) == len(closed) == 2
    assert len({id(item) for item in created}) == 2
    assert {id(item) for item in closed} == {id(item) for item in created}


def test_v2_uses_the_existing_raw02_output_and_budget_contract(
        public_v2_catalog,
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> None:
    """V2 run 只能经 RAW-02 统一生成 bytes、readback 和预算拒绝。"""
    ingress = _ingress(
        public_v2_catalog,
        "dlg-raw-public-v2-unknown-budget",
        21,
    )
    accepted = run_public_frame_answer(
        ingress,
        source_payload_closure=public_source_payload_closure,
    )

    assert accepted.result_code == DLG_RAW_ACCEPT
    assert accepted.accepted is True
    assert accepted.run is not None
    assert accepted.run.status == accepted.run.selection.stance
    assert accepted.output_readback is not None
    assert accepted.output_scalars == accepted.run.generation.rendered.units

    frame = _frame(
        public_v2_catalog,
        "dlg-raw-public-v2-unknown-budget",
    )
    constrained = replace(
        frame,
        recipe=replace(frame.recipe, output_max_bytes=1),
    )
    constrained_catalog = PublicFrameCatalog(
        public_v2_catalog.source_sha256,
        (constrained,),
    )
    rejected = run_public_frame_answer(
        _ingress(
            constrained_catalog,
            "dlg-raw-public-v2-unknown-budget",
            22,
        ),
        source_payload_closure=public_source_payload_closure,
    )

    assert rejected.result_code == DLG_RAW_REJECT_OUTPUT_BUDGET
    assert rejected.run is None
    assert rejected.output_bytes == ()


def test_v2_course_drift_rejects_before_cached_runtime_can_be_reused(
        monkeypatch: pytest.MonkeyPatch,
        public_v2_catalog,
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> None:
    """materializer/cache 每轮核验 closure；漂移时不构造 backend 或返回 run。"""
    cache = PublicCoursePreparationCache()
    ingress = _ingress(public_v2_catalog, "dlg-raw-public-v2-unknown-budget", 3)
    run_public_response_act_frame(
        ingress,
        source_payload_closure=public_source_payload_closure,
        preparation_cache=cache,
    )
    course_key = _frame(
        public_v2_catalog,
        "dlg-raw-public-v2-unknown-budget",
    ).recipe.course_relative_path.encode("ascii")
    drifted_closure = build_public_source_payload_closure_v1(tuple(
        public_source_payload_record_from_u8_v1(
            record.logical_key,
            record.raw_payload + b"x",
        ) if record.logical_key == course_key else record
        for record in public_source_payload_closure.records
    ))

    import pure_integer_ai.experiments.conversation_raw_response_act_runtime as runtime

    def forbidden_backend() -> object:
        raise AssertionError("course drift 不得构造 run-local backend")

    monkeypatch.setattr(runtime, "DictBackend", forbidden_backend)
    with pytest.raises(ConversationRawResponseActRuntimeError):
        run_public_response_act_frame(
            ingress,
            source_payload_closure=drifted_closure,
            preparation_cache=cache,
        )

    assert cache.entry_count == 1
    assert cache.miss_count == 1
    assert cache.hit_count == 0


def test_answer_and_refuse_cannot_enter_the_v2_nonanswer_family(
        public_v2_catalog,
        public_source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> None:
    """即使 G-01 给出合法 ANSWER/REFUSE，05A mapping 也必须 fail closed。"""
    import pure_integer_ai.experiments.conversation_raw_response_act_runtime as runtime

    clarify_ingress = _ingress(
        public_v2_catalog,
        "dlg-raw-public-v2-clarify-site",
        4,
    )
    clarify_build = materialize_public_response_act_planning_from_closure(
        clarify_ingress.frame,
        clarify_ingress.request,
        public_source_payload_closure,
    )
    content, _selector, *_rest = runtime._generation_protocols(
        clarify_build.planning.goal.target_branch)
    refuse_selector = AnswerContentSelector(
        content,
        _FixedPolicy(AnswerContentDecision(
            content.refuse,
            minimal_instruction_identity((65001, 52, 91, 1)),
            (),
            (),
            (65001, 52, 91, 2),
        )),
    )
    with pytest.raises(ConversationRawResponseActRuntimeError, match="non-answer family"):
        runtime._response_act_for_selection(
            refuse_selector,
            clarify_build.planning,
            content,
        )

    v1_catalog = load_public_frame_catalog(public_source_payload_closure)
    v1_frame = v1_catalog.frames[0]
    v1_ingress = ingress_raw_lexical_frame(
        intake_raw_conversation_vector(v1_frame.surface_bytes),
        v1_catalog,
        (65001, 52, 90, 5),
    )
    _candidate, answer_planning = materialize_public_frame_candidate(
        v1_frame,
        v1_ingress.request,
    )
    answer_content, _answer_selector, *_rest = runtime._generation_protocols(
        answer_planning.goal.target_branch)
    answer_selector = AnswerContentSelector(
        answer_content,
        _FixedPolicy(AnswerContentDecision(
            answer_content.answer,
            minimal_instruction_identity((65001, 52, 91, 3)),
            answer_planning.candidate_keys(),
            (),
            (65001, 52, 91, 4),
        )),
    )
    with pytest.raises(ConversationRawResponseActRuntimeError, match="non-answer family"):
        runtime._response_act_for_selection(
            answer_selector,
            answer_planning,
            answer_content,
        )
