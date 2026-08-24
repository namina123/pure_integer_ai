"""DLG-RAW-16 G1 只读表层选择器的定向回归。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_compile import (
    load_surface_organization_jsonl,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_schema import (
    SurfaceVariant,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_selector import (
    SURFACE_AMBIGUOUS_RECORD,
    SURFACE_NO_LEGAL_VARIANT,
    SURFACE_NO_MATCH,
    SURFACE_SELECTED,
    SurfaceSelectionRequest,
    SurfaceSemantic,
    SurfaceSelectorError,
    build_surface_organization_selector,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_COURSE_PATH = _REPOSITORY_ROOT / "data/ph2/dlg_raw16_surface_organization_v1.jsonl.sample"


@pytest.fixture(scope="module")
def course():
    """G0 sample 只由 host 读取一次；G1 只消费 compiled records。"""
    return load_surface_organization_jsonl(_COURSE_PATH.read_bytes())


def _record(course, sample_id: str):
    matches = tuple(item.record for item in course if item.record.sample_id == sample_id)
    assert len(matches) == 1
    return matches[0]


def _request(record, *, ordinal: int = 0, **changes) -> SurfaceSelectionRequest:
    """将完整已确认的 G0 semantic/obligation 投影为 G1 输入。"""
    values = {
        "dialogue_act": record.dialogue_act,
        "semantic": SurfaceSemantic(
            record.proposition_id,
            record.proposition_kind,
            record.proposition_subject,
            record.proposition_predicate,
            record.proposition_object,
        ),
        "ordered_clause_slots": tuple(item.slot_id for item in record.clause_slots),
        "register": record.register,
        "required_proposition_ids": record.required_proposition_ids,
        "forbidden_proposition_ids": record.forbidden_proposition_ids,
        "min_chars": record.min_chars,
        "max_chars": record.max_chars,
        "selection_ordinal": ordinal,
        "source_id": record.source_id,
        "context_id": record.context_id,
        "family_id": record.family_id,
        "template_family": record.template_family,
    }
    values.update(changes)
    return SurfaceSelectionRequest(**values)


def test_selects_multiple_legal_surfaces_in_explicit_stable_ordinal_order(course) -> None:
    """不能依赖 accepted JSON 顺序或 Python hash 顺序；ordinal 是唯一变元。"""
    record = _record(course, "s03")
    selector_a = build_surface_organization_selector(course)
    selector_b = build_surface_organization_selector(tuple(reversed(course)))

    first = selector_a.select(_request(record, ordinal=0))
    second = selector_a.select(_request(record, ordinal=1))
    wrapped = selector_a.select(_request(record, ordinal=2))
    replay = selector_b.select(_request(record, ordinal=0))

    assert first.status_code == SURFACE_SELECTED
    assert first.record_id == "s03"
    assert first.variant_id == "a01"
    assert first.surface == "暴雨导致河水上涨。"
    assert first.output_bytes == tuple(first.surface.encode("utf-8"))
    assert second.status_code == SURFACE_SELECTED
    assert second.variant_id == "a02"
    assert second.surface == "因为暴雨，河水上涨了。"
    assert wrapped.variant_id == first.variant_id
    assert replay.canonical_record() == first.canonical_record()
    assert first.trace == replay.trace


def test_selector_fails_closed_for_mismatched_contract_or_tight_budget(course) -> None:
    """语义、语域、义务和预算任一不满足时均不泄漏半句输出。"""
    record = _record(course, "s05")
    selector = build_surface_organization_selector(course)

    register_miss = selector.select(_request(record, register="neutral"))
    too_small = selector.select(_request(record, max_chars=2))

    assert register_miss.status_code == SURFACE_NO_MATCH
    assert register_miss.surface is None
    assert register_miss.output_bytes == ()
    assert too_small.status_code == SURFACE_NO_LEGAL_VARIANT
    assert too_small.surface is None
    assert too_small.output_scalars == ()


def test_nonanswer_cannot_emit_a_candidate_that_claims_a_forbidden_proposition(course) -> None:
    """UNKNOWN/CLARIFY/REPAIR 的 accepted 也要在 runtime 再次拒绝事实 claim。"""
    record = _record(course, "s07")
    # schema/compiler 已在 G0 拒绝这种漂移；这里用 frozen dataclass 的
    # 低层篡改模拟边界上游已经被破坏，确认 G1 仍然 fail closed。
    altered = replace(record)
    object.__setattr__(
        altered,
        "accepted",
        tuple(
            SurfaceVariant(
                item.variant_id,
                item.surface,
                ("p-invented",),
                item.clause_order,
                item.register,
                (),
            )
            for item in record.accepted
        ),
    )
    selector = build_surface_organization_selector((altered,))

    result = selector.select(_request(altered))

    assert result.status_code == SURFACE_NO_LEGAL_VARIANT
    assert result.surface is None
    assert result.output_bytes == ()


def test_selector_preserves_ambiguity_until_a_record_identity_is_explicit(course) -> None:
    """同一 semantic/act 不能由 selector 通过偶然顺序私选 record。"""
    record = _record(course, "s04")
    shadow = replace(
        record,
        sample_id="s04-shadow",
        source_id="src-s04-shadow",
        context_id="ctx-s04-shadow",
        family_id="fam-causal-02-shadow",
        template_family="tmpl-causal-shadow",
    )
    selector = build_surface_organization_selector((record, shadow))

    ambiguous = selector.select(_request(
        record,
        source_id=None,
        context_id=None,
        family_id=None,
        template_family=None,
    ))
    resolved = selector.select(_request(record))

    assert ambiguous.status_code == SURFACE_AMBIGUOUS_RECORD
    assert ambiguous.candidate_count == 2
    assert ambiguous.surface is None
    assert resolved.status_code == SURFACE_SELECTED
    assert resolved.record_id == "s04"


def test_answer_request_cannot_omit_its_confirmed_semantic_proposition(course) -> None:
    """表层层不能用空 obligation 把事实回答伪装成普通措辞。"""
    record = _record(course, "s01")
    with pytest.raises(SurfaceSelectorError, match="ANSWER request.required"):
        _request(record, required_proposition_ids=())
