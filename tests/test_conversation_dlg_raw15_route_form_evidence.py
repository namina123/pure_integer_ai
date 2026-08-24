"""DLG-RAW-15 route-form 证据层的有界回归（不是 G1 formal gate）。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.experiments.conversation_dlg_raw15_route_form_evidence import (
    DlgRaw15RouteFormEvidenceError,
    build_g1_route_form_evidence_v1,
    verify_g1_route_form_evidence_v1,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
)
from pure_integer_ai.experiments.conversation_source_bound_slot_catalog import (
    resolve_source_bound_slot_composition,
)
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


_INPUT = tuple(ord(item) for item in "东岸入口何时启用？")
_OPTIONS = (
    "澄川码头何时启用？",
    "北川站东门何时启用？",
)


def _resolution():
    """用真实 public V3 ambiguity 运行 evidence helper；不宣称 G1 已通过。"""
    closure = load_public_source_payload_closure_from_root(_ROOT)
    runtime = build_public_dialogue_runtime_v1(closure)
    result = resolve_source_bound_slot_composition(
        runtime.source_bound_slot_catalog,
        runtime.base_catalog,
        runtime.active_catalog,
        _INPUT,
        closure,
    )
    assert result.result_code == DLG_RAW_REJECT_LEXICAL_AMBIGUOUS
    return result


def _build():
    """构造两份独立 raw surface payload。"""
    output = "请选择一个完整问题：\n" + "\n".join(_OPTIONS)
    surface_a = ("independent authored witness A\n" + output + "\n").encode("utf-8")
    surface_b = ("independent authored witness B\n" + output + "\n").encode("utf-8")
    return build_g1_route_form_evidence_v1(
        _resolution(),
        form_id="g1-independent-route-form-v1",
        prompt_surface="请选择一个完整问题：",
        option_surfaces=_OPTIONS,
        course_logical_key=b"data/ph2/dlg_raw15_g1_route_form_v1.jsonl.sample",
        surface_a_logical_key=b"data/ph2/dlg_raw15_g1_route_surface_v1_a.txt.sample",
        surface_b_logical_key=b"data/ph2/dlg_raw15_g1_route_surface_v1_b.txt.sample",
        surface_a_payload=surface_a,
        surface_b_payload=surface_b,
        course_attribution="独立 G1 authored route course",
        surface_a_attribution="独立 G1 authored surface A",
        surface_b_attribution="独立 G1 authored surface B",
    )


def test_route_form_evidence_closes_candidate_route_course_and_surfaces() -> None:
    """真实 V3 ambiguity 可生成可回读的独立 route/course/A-B 证据。"""
    evidence = _build()
    assert evidence.matched_frame_count == 2
    assert len(evidence.candidate_identities_u8) == 2
    assert len(set(evidence.candidate_identities_u8)) == 2
    assert len(evidence.route_identity_u8) == 32
    assert evidence.course_payload.endswith(b"\n")
    assert evidence.surface_a_payload != evidence.surface_b_payload
    assert evidence.form_identity_u8 == evidence.form_identity_u8
    assert verify_g1_route_form_evidence_v1(evidence) is evidence


def test_route_form_evidence_rejects_route_or_surface_drift() -> None:
    """route identity、candidate identity、raw surface 漂移均 fail closed。"""
    evidence = _build()
    with pytest.raises(DlgRaw15RouteFormEvidenceError):
        replace(evidence, route_identity_u8=(0,) * 32)
    with pytest.raises(DlgRaw15RouteFormEvidenceError):
        replace(evidence, candidate_identities_u8=(
            (0,) * 32,
            evidence.candidate_identities_u8[1],
        ))
    with pytest.raises(DlgRaw15RouteFormEvidenceError):
        replace(evidence, surface_b_payload=evidence.surface_a_payload)
