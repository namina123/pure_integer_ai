"""DLG-RAW-03 二进制 terminal transport 的 DLG-RAW-07 runtime 专项。"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PUBLIC_FRAME_CONTEXT_NONE,
    PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR,
    PublicFrameCatalog,
)
from pure_integer_ai.experiments.conversation_public_answer_catalog import (
    load_public_answer_frame_catalog_from_closure,
)
from pure_integer_ai.experiments.conversation_public_reference_catalog import (
    load_public_reference_frame_catalog_from_closure,
)
from pure_integer_ai.experiments.conversation_public_response_act_catalog import (
    load_public_response_act_frame_catalog_from_closure,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_source_bound_slot_catalog import (
    SourceBoundSlotCompositionResolution,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_REJECT_CONSTRUCTION_MISS,
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    encode_utf8_v1,
)
from pure_integer_ai.experiments import conversation_raw_dialogue_session
from pure_integer_ai.experiments.run_public_frame_dialogue import (
    main,
    run_public_frame_dialogue_terminal,
    run_public_terminal_dialogue_act_terminal,
)
from pure_integer_ai.experiments.conversation_raw_terminal_dialogue_act import (
    build_terminal_dialogue_act_runtime_v1,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_V2_CATALOG_LOGICAL_KEY = (
    b"data/ph2/dlg_raw_public_response_act_frame_v2.jsonl.sample")
_DERIVED_V3_CATALOG_LOGICAL_KEY = (
    b"data/ph2/dlg_raw_public_derived_frame_v3.jsonl.sample")
_CONTEXTUAL_V4_CATALOG_LOGICAL_KEY = (
    b"data/ph2/dlg_raw_public_contextual_ellipsis_frame_v4.jsonl.sample")


@pytest.fixture(scope="module")
def public_runtime() -> PublicDialogueRuntimeV1:
    """在唯一 host 边界读取当前冻结 closure 后构造无路径 public dialogue runtime。"""
    closure = load_public_source_payload_closure_from_root(_REPOSITORY_ROOT)
    return build_public_dialogue_runtime_v1(closure)


@pytest.fixture(scope="module")
def public_catalog(public_runtime: PublicDialogueRuntimeV1) -> PublicFrameCatalog:
    """保留 V1 首句/追问测试的明确 base catalog。"""
    return public_runtime.base_catalog


def _source_bound_surface(
        runtime: PublicDialogueRuntimeV1,
        binding_key: str,
        ) -> tuple[int, ...]:
    """从 runtime 的 V2 binding/base 边界派生唯一终端动态输入。"""
    binding = next(item for item in runtime.source_bound_slot_catalog.bindings
                   if item.binding_key == binding_key)
    base = next(item for item in runtime.active_catalog.frames
                if item.frame_key == binding.base_frame_key)
    families = tuple(
        family for family in runtime.source_bound_slot_catalog.families
        if (len(base.surface_scalars) > len(family.suffix_scalars)
            and base.surface_scalars[-len(family.suffix_scalars):]
            == family.suffix_scalars)
    )
    assert len(families) == 1
    family = families[0]
    return (*family.prefix_scalars, *binding.entity_scalars,
            *family.suffix_scalars)


def test_binary_shell_writes_actual_first_turn_output(
        public_runtime: PublicDialogueRuntimeV1,
        public_catalog: PublicFrameCatalog) -> None:
    """终端不经文本重编码，连续转发首轮和显式 context 第二轮的真实输出。"""
    first_frame = next(item for item in public_catalog.frames
                       if item.context_requirement == PUBLIC_FRAME_CONTEXT_NONE)
    follow_up_frame = next(
        item for item in public_catalog.frames
        if item.context_requirement == PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR)
    source = BytesIO(
        bytes(first_frame.surface_bytes) + b"\n"
        + bytes(follow_up_frame.surface_bytes) + b"\n:quit\n")
    target = BytesIO()

    run_public_frame_dialogue_terminal(
        public_runtime,
        source,
        target,
        prompts=False,
    )

    assert target.getvalue() == (
        b"\xe7\xb3\xbb\xe7\xbb\x9f> "
        b"\xe5\x8c\x97\xe5\xb7\x9d\xe7\xab\x99\xe4\xb8\x9c\xe9\x97\xa8"
        b"\xe4\xba\x8e2024\xe5\xb9\xb4\xe5\x90\xaf\xe7\x94\xa8\xe3\x80\x82\n"
        b"\xe7\xb3\xbb\xe7\xbb\x9f> "
        b"\xe5\x8c\x97\xe5\xb7\x9d\xe7\xab\x99\xe4\xb8\x9c\xe9\x97\xa8"
        b"\xe4\xba\x8e2024\xe5\xb9\xb4\xe5\x90\xaf\xe7\x94\xa8\xe3\x80\x82\n"
    )


def test_over_budget_physical_line_is_rejected_then_shell_stops(
        public_runtime: PublicDialogueRuntimeV1,
        public_catalog: PublicFrameCatalog) -> None:
    """预算外的第一个 byte 后不读取余下物理行，也不将其拆为下一轮。"""
    first_frame = next(item for item in public_catalog.frames
                       if item.context_requirement == PUBLIC_FRAME_CONTEXT_NONE)
    source = BytesIO(b"a" * 4097 + b"\n" + bytes(
        first_frame.surface_bytes) + b"\n")
    target = BytesIO()

    run_public_frame_dialogue_terminal(
        public_runtime,
        source,
        target,
        prompts=False,
    )

    assert target.getvalue() == b"\xe7\xb3\xbb\xe7\xbb\x9f> [REJECT:1]\n"
    assert source.tell() == 4097


def test_main_loads_v1_and_v2_for_one_continuous_public_dialogue(
        public_runtime: PublicDialogueRuntimeV1,
        public_catalog: PublicFrameCatalog) -> None:
    """用户入口必须同时加载首句/追问和三类真实 non-answer frame。"""
    v1_first = next(item for item in public_catalog.frames
                    if item.context_requirement == PUBLIC_FRAME_CONTEXT_NONE)
    v1_follow_up = next(
        item for item in public_catalog.frames
        if item.context_requirement == PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR)
    v2_catalog = load_public_response_act_frame_catalog_from_closure(
        public_runtime.source_payload_closure,
        _V2_CATALOG_LOGICAL_KEY,
    )
    assert all(frame in public_runtime.active_catalog.frames
               for frame in v2_catalog.frames)
    source = BytesIO(b"".join((
        bytes(v1_first.surface_bytes), b"\n",
        bytes(v1_follow_up.surface_bytes), b"\n",
        *(item for frame in v2_catalog.frames
          for item in (bytes(frame.surface_bytes), b"\n")),
        b":quit\n",
    )))
    target = BytesIO()

    assert main([], stdin=source, stdout=target) == 0
    rendered = target.getvalue().decode("utf-8")

    assert rendered.count("系统> ") == 5
    assert "[REJECT:" not in rendered
    assert rendered.count("北川站东门于2024年启用。") == 2
    assert "现有来源没有提供所问信息。" in rendered
    assert "存在多个候选，请补充更明确的限定。" in rendered
    assert "来源说法彼此矛盾，无法给出确定答案。" in rendered


def test_main_routes_independent_answer_catalog_through_actual_runtime(
        public_runtime: PublicDialogueRuntimeV1) -> None:
    """独立 ANSWER catalog 必须经 RAW-02 产生首句，不能停在目录装载。"""
    catalog = load_public_answer_frame_catalog_from_closure(
        public_runtime.source_payload_closure)
    assert len(catalog.frames) == 1
    frame = catalog.frames[0]
    assert frame in public_runtime.active_catalog.frames
    target = BytesIO()

    assert main(
        [],
        stdin=BytesIO(bytes(frame.surface_bytes) + b"\n:quit\n"),
        stdout=target,
    ) == 0

    assert target.getvalue() == (
        b"\xe4\xbd\xa0> \xe7\xb3\xbb\xe7\xbb\x9f> "
        b"\xe6\xbe\x84\xe5\xb7\x9d\xe7\xa0\x81\xe5\xa4\xb4"
        b"\xe4\xba\x8e2023\xe5\xb9\xb4\xe5\x90\xaf\xe7\x94\xa8\xe3\x80\x82\n"
        b"\xe4\xbd\xa0> "
    )


def test_main_defaults_to_route_clarification_outer_for_source_bound_ambiguity() -> None:
    """默认入口对真实来源绑定 code-8 输出可重输候选，并完成一轮选择。"""
    target = BytesIO()
    assert main(
        [],
        stdin=BytesIO(
            "东岸入口何时启用？\n"
            "澄川码头何时启用？\n"
            ":quit\n".encode("utf-8"),
        ),
        stdout=target,
    ) == 0
    rendered = target.getvalue().decode("utf-8")
    assert "此输入对应多个已学习路径，请重输其中一个完整问题：" in rendered
    assert "澄川码头何时启用？" in rendered
    assert "北川站东门何时启用？" in rendered
    assert "澄川码头于2023年启用。" in rendered


def test_main_routes_actual_v3_reference_input_to_double_sentence_output(
        public_runtime: PublicDialogueRuntimeV1) -> None:
    """用户入口加载 V3，并将公开输入送入真实双命题 reference runtime。"""
    v3_catalog = load_public_reference_frame_catalog_from_closure(
        public_runtime.source_payload_closure)
    assert len(v3_catalog.frames) == 1
    assert v3_catalog.frames[0] in public_runtime.active_catalog.frames
    source = BytesIO(
        bytes(v3_catalog.frames[0].surface_bytes) + b"\n:quit\n")
    target = BytesIO()

    assert main([], stdin=source, stdout=target) == 0
    rendered = target.getvalue().decode("utf-8")

    assert rendered.count("系统> ") == 1
    assert "[REJECT:" not in rendered
    assert "北川站东门于2024年启用。前述启用事项已登记入档。" in rendered


def test_main_routes_actual_v3_derived_omission_input(
        public_runtime: PublicDialogueRuntimeV1) -> None:
    """终端入口必须装载 05C 派生 frame，不能只在单独 runtime 中可达。"""
    catalog = load_public_response_act_frame_catalog_from_closure(
        public_runtime.source_payload_closure,
        _DERIVED_V3_CATALOG_LOGICAL_KEY,
    )
    assert len(catalog.frames) == 1
    assert catalog.frames[0] in public_runtime.active_catalog.frames
    source = BytesIO(bytes(catalog.frames[0].surface_bytes) + b"\n:quit\n")
    target = BytesIO()

    assert main([], stdin=source, stdout=target) == 0
    rendered = target.getvalue().decode("utf-8")

    assert rendered.count("系统> ") == 1
    assert "[REJECT:" not in rendered
    assert "现有来源没有提供所问信息。" in rendered


def test_main_routes_actual_v4_contextual_ellipsis_after_matching_anchor(
        public_runtime: PublicDialogueRuntimeV1) -> None:
    """用户入口必须让实际 05C typed target 解锁 05D 的省略输入。"""
    derived_catalog = load_public_response_act_frame_catalog_from_closure(
        public_runtime.source_payload_closure,
        _DERIVED_V3_CATALOG_LOGICAL_KEY,
    )
    contextual_catalog = load_public_response_act_frame_catalog_from_closure(
        public_runtime.source_payload_closure,
        _CONTEXTUAL_V4_CATALOG_LOGICAL_KEY,
    )
    assert len(derived_catalog.frames) == len(contextual_catalog.frames) == 1
    source = BytesIO(
        bytes(derived_catalog.frames[0].surface_bytes) + b"\n"
        + bytes(contextual_catalog.frames[0].surface_bytes) + b"\n:quit\n")
    target = BytesIO()

    assert main([], stdin=source, stdout=target) == 0
    rendered = target.getvalue().decode("utf-8")

    assert rendered.count("系统> ") == 2
    assert "[REJECT:" not in rendered
    assert rendered.count("现有来源没有提供所问信息。") == 2


def test_main_routes_source_bound_slot_composition_input(
        public_runtime: PublicDialogueRuntimeV1) -> None:
    """终端入口必须可达 V2 的真实 source-bound 动态回答 frame。"""
    raw_input = bytes(encode_utf8_v1(_source_bound_surface(
        public_runtime,
        "north-east-side-entrance-time-v2",
    )))
    target = BytesIO()

    assert main([], stdin=BytesIO(raw_input + b"\n:quit\n"), stdout=target) == 0
    rendered = target.getvalue().decode("utf-8")

    assert rendered.count("系统> ") == 1
    assert "[REJECT:" not in rendered
    assert "北川站东门于2024年启用。" in rendered


def test_terminal_rejects_v3_multi_target_alias_before_answer_generation(
        public_runtime: PublicDialogueRuntimeV1) -> None:
    """同一来源别名指向不同 target 时必须在 RAW-01 拒绝，不能任取一个回答。"""
    north = _source_bound_surface(
        public_runtime,
        "east-bank-north-gate-time-v3",
    )
    pier = _source_bound_surface(
        public_runtime,
        "east-bank-pier-time-v3",
    )
    binding = next(
        item for item in public_runtime.source_bound_slot_catalog.bindings
        if item.binding_key == "east-bank-pier-time-v3")
    answer = next(
        item for item in public_runtime.active_catalog.frames
        if item.frame_key == "dlg-raw-public-answer-pier-time-v1")
    assert north == pier
    assert binding.base_frame_key == answer.frame_key
    assert binding.base_frame_raw_sha256 == answer.raw_line_sha256
    target = BytesIO()

    run_public_frame_dialogue_terminal(
        public_runtime,
        BytesIO(bytes(encode_utf8_v1(north)) + b"\n:quit\n"),
        target,
        prompts=False,
    )

    assert target.getvalue() == (
        b"\xe7\xb3\xbb\xe7\xbb\x9f> [REJECT:"
        + str(DLG_RAW_REJECT_LEXICAL_AMBIGUOUS).encode("ascii") + b"]\n"
    )


def test_default_terminal_renders_coverage_and_route_act_not_raw_7_8(
        public_runtime: PublicDialogueRuntimeV1) -> None:
    """默认 DLG-RAW-13 shell 仅转换已完成的 raw 7/8，保留单一 response record。"""
    target = BytesIO()

    run_public_terminal_dialogue_act_terminal(
        build_terminal_dialogue_act_runtime_v1(public_runtime),
        BytesIO("这是什么？\n东岸入口何时启用？\n:quit\n".encode("utf-8")),
        target,
        prompts=False,
    )

    assert target.getvalue() == (
        "系统> 当前公开对话资料尚未覆盖此输入。\n"
        "系统> 此输入对应多个已学习路径，请补充限定。\n"
    ).encode("utf-8")


def test_terminal_renders_source_bound_counterevidence_rejection(
        public_runtime: PublicDialogueRuntimeV1) -> None:
    """V2 explicit counterevidence 必须经真实 terminal 路径保留 code 13。"""
    raw_input = bytes(encode_utf8_v1(_source_bound_surface(
        public_runtime,
        "north-east-side-passage-conflict-v2",
    )))
    target = BytesIO()

    run_public_frame_dialogue_terminal(
        public_runtime,
        BytesIO(raw_input + b"\n:quit\n"),
        target,
        prompts=False,
    )
    assert target.getvalue() == b"\xe7\xb3\xbb\xe7\xbb\x9f> [REJECT:13]\n"


def test_terminal_renders_pre_frame_source_bound_construction_miss(
        public_runtime: PublicDialogueRuntimeV1,
        public_catalog: PublicFrameCatalog,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """唯一组合在 frame 物化前失败时，terminal 必须公开转发 REJECT:9。"""
    slot_catalog = public_runtime.source_bound_slot_catalog
    surface = _source_bound_surface(
        public_runtime,
        "north-east-side-entrance-time-v2",
    )
    resolution = SourceBoundSlotCompositionResolution(
        DLG_RAW_REJECT_CONSTRUCTION_MISS,
        1,
        surface,
        slot_catalog,
    )
    monkeypatch.setattr(
        conversation_raw_dialogue_session,
        "resolve_source_bound_slot_composition",
        lambda *_args, **_kwargs: resolution,
    )
    target = BytesIO()

    run_public_frame_dialogue_terminal(
        public_runtime,
        BytesIO(bytes(encode_utf8_v1(surface)) + b"\n:quit\n"),
        target,
        prompts=False,
    )

    assert target.getvalue() == b"\xe7\xb3\xbb\xe7\xbb\x9f> [REJECT:9]\n"


def test_main_defaults_to_v2_and_blocks_target_anchor_after_provider_turn() -> None:
    """默认 CLI 必须写 V2 provider turn，不能跨其回溯旧 Frame anchor。"""
    target = BytesIO()

    assert main(
        [],
        stdin=BytesIO(
            "北川站东门何时启用？\n"
            "寒潮导致什么？\n"
            "它是在什么时候启用的？\n"
            ":quit\n".encode("utf-8"),
        ),
        stdout=target,
    ) == 0

    assert target.getvalue() == (
        b"\xe4\xbd\xa0> \xe7\xb3\xbb\xe7\xbb\x9f> "
        b"\xe5\x8c\x97\xe5\xb7\x9d\xe7\xab\x99\xe4\xb8\x9c\xe9\x97\xa8"
        b"\xe4\xba\x8e2024\xe5\xb9\xb4\xe5\x90\xaf\xe7\x94\xa8\xe3\x80\x82\n"
        b"\xe4\xbd\xa0> \xe7\xb3\xbb\xe7\xbb\x9f> "
        b"\xe5\xaf\x92\xe6\xbd\xae\xe4\xbd\xbf\xe5\xbe\x97\xe8\xb7\xaf\xe9\x9d\xa2\xe7\xbb\x93\xe5\x86\xb0\xe3\x80\x82\n"
        b"\xe4\xbd\xa0> \xe7\xb3\xbb\xe7\xbb\x9f> [REJECT:10]\n"
        b"\xe4\xbd\xa0> "
    )


def test_main_defaults_to_v4_and_runs_source_bound_provider_followup() -> None:
    """默认 terminal 的首个 provider follow-up 只能复用来源 profile 锁定的原因 span。"""
    target = BytesIO()

    assert main(
        [],
        stdin=BytesIO(
            "寒潮导致什么？\n"
            "它的原因是什么？\n"
            ":quit\n".encode("utf-8"),
        ),
        stdout=target,
    ) == 0

    assert target.getvalue() == (
        b"\xe4\xbd\xa0> \xe7\xb3\xbb\xe7\xbb\x9f> "
        b"\xe5\xaf\x92\xe6\xbd\xae\xe4\xbd\xbf\xe5\xbe\x97\xe8\xb7\xaf\xe9\x9d\xa2\xe7\xbb\x93\xe5\x86\xb0\xe3\x80\x82\n"
        b"\xe4\xbd\xa0> \xe7\xb3\xbb\xe7\xbb\x9f> \xe5\xaf\x92\xe6\xbd\xae\n"
        b"\xe4\xbd\xa0> "
    )


def test_main_runs_source_bound_reverse_focus_provider_followup() -> None:
    """默认 terminal 的 V3 ledger 必须连续推进原因、结果再回到原因。"""
    target = BytesIO()

    assert main(
        [],
        stdin=BytesIO(
            "什么导致路面结冰？\n"
            "它的结果是什么？\n"
            "它的原因是什么？\n"
            ":quit\n".encode("utf-8"),
        ),
        stdout=target,
    ) == 0

    assert target.getvalue() == (
        b"\xe4\xbd\xa0> \xe7\xb3\xbb\xe7\xbb\x9f> "
        b"\xe5\xaf\x92\xe6\xbd\xae\xe4\xbd\xbf\xe5\xbe\x97\xe8\xb7\xaf\xe9\x9d\xa2"
        b"\xe7\xbb\x93\xe5\x86\xb0\xe3\x80\x82\n"
        b"\xe4\xbd\xa0> \xe7\xb3\xbb\xe7\xbb\x9f> \xe8\xb7\xaf\xe9\x9d\xa2\xe7\xbb\x93\xe5\x86\xb0\n"
        b"\xe4\xbd\xa0> \xe7\xb3\xbb\xe7\xbb\x9f> \xe5\xaf\x92\xe6\xbd\xae\n"
        b"\xe4\xbd\xa0> "
    )
