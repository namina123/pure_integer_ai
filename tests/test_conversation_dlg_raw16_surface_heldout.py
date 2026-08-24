"""DLG-RAW-16 G2 replay-only held-out diagnostic 的定向回归。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_heldout import (
    G2_HELD_OUT_NOT_READY,
    G2_SURFACE_REPLAY_ONLY,
    SurfaceHeldOutCase,
    SurfaceHeldOutDiagnosticError,
    run_surface_organization_g2_diagnostic,
    semantic_shape_key_for_record,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_compile import (
    load_surface_organization_jsonl,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_selector import (
    SURFACE_NO_MATCH,
    SurfaceSelectionRequest,
    SurfaceSemantic,
    build_surface_organization_selector,
)


_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "data/ph2/dlg_raw16_surface_organization_v1.jsonl.sample"


@pytest.fixture(scope="module")
def selector():
    """G2 只消费已编译 G0 records，不读 terminal 或训练状态。"""
    return build_surface_organization_selector(
        load_surface_organization_jsonl(_SAMPLE.read_bytes()))


def _unseen_case(number: int) -> SurfaceHeldOutCase:
    """两个独立 temporal-causal shape；G0 course 中不存在此完整组合。"""
    proposition = f"p-heldout-temporal-{number}"
    request = SurfaceSelectionRequest(
        "ANSWER",
        SurfaceSemantic(
            proposition,
            "temporal_causal",
            f"潮位{number}",
            "影响",
            f"航道{number}的通行时间",
        ),
        ("topic", "relation", "condition", "effect"),
        "neutral",
        (proposition,),
        (),
        2,
        80,
        number - 1,
        f"heldout-src-{number}",
        f"heldout-ctx-{number}",
        f"heldout-family-{number}",
        f"heldout-template-{number}",
    )
    return SurfaceHeldOutCase(
        f"g2-unseen-temporal-{number}",
        "dlg-raw16-g2-independent-v1",
        (
            "act", "ANSWER", "kind", "temporal_causal", "roles",
            "topic", "relation", "condition", "effect", "required",
            "1", "1", "1", "1",
        ),
        request,
    )


def test_g2_unseen_semantic_shapes_honestly_report_surface_replay_only(selector) -> None:
    """NO_MATCH 证明当前 selector 的 exact 回放边界，而不是泛化成功。"""
    report = run_surface_organization_g2_diagnostic(
        selector, (_unseen_case(1), _unseen_case(2)))

    assert report.status == G2_SURFACE_REPLAY_ONLY
    assert report.total_cases == 2
    assert report.shape_unseen_cases == 2
    assert report.no_match_cases == 2
    assert report.unexpected_match_cases == 0
    assert report.ready == 0
    assert report.pass_ == 0
    assert tuple(item.observed_status_code for item in report.observations) == (
        SURFACE_NO_MATCH, SURFACE_NO_MATCH)
    assert report.canonical_record()


def test_g2_diagnostic_is_deterministic_for_the_same_selector_and_cases(selector) -> None:
    """完整 report 只依赖 typed case 内容，不依赖对象地址或执行历史。"""
    cases = (_unseen_case(1), _unseen_case(2))
    first = run_surface_organization_g2_diagnostic(selector, cases)
    second = run_surface_organization_g2_diagnostic(selector, cases)

    assert first.canonical_record() == second.canonical_record()
    assert first.trace == second.trace


def test_g2_rejects_a_shape_that_was_already_in_the_course(selector) -> None:
    """只改 entity/source 不能伪装为独立 held-out semantic shape。"""
    record = next(item for item in selector.records if item.sample_id == "s03")
    request = SurfaceSelectionRequest(
        "ANSWER",
        SurfaceSemantic(
            "p-fake-heldout", record.proposition_kind, "新暴雨", "导致", "新河水上涨"),
        tuple(item.slot_id for item in record.clause_slots),
        record.register,
        ("p-fake-heldout",),
        (),
        record.min_chars,
        record.max_chars,
        0,
        "heldout-src-fake",
        "heldout-ctx-fake",
        "heldout-family-fake",
        "heldout-template-fake",
    )
    leaked_shape = SurfaceHeldOutCase(
        "g2-fake-shape", "dlg-raw16-g2-independent-v1",
        semantic_shape_key_for_record(record), request)

    with pytest.raises(SurfaceHeldOutDiagnosticError, match="semantic-shape 泄漏"):
        run_surface_organization_g2_diagnostic(selector, (leaked_shape,))


def test_g2_empty_input_is_not_ready_not_pass(selector) -> None:
    """没有独立 typed case 时不允许形成任何正向结论。"""
    report = run_surface_organization_g2_diagnostic(selector, ())

    assert report.status == G2_HELD_OUT_NOT_READY
    assert report.total_cases == 0
    assert report.ready == 0
    assert report.pass_ == 0
