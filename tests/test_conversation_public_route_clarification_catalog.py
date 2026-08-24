"""DLG-RAW-14 公开 route clarification course/catalog 的有界回归。"""
from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import pytest

from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_route_clarification_catalog import (
    PublicRouteClarificationCatalogError,
    candidate_identity_v1,
    load_public_route_clarification_catalog_from_closure,
    public_route_clarification_course_parser_identity_v1,
    public_route_clarification_course_parser_record_v1,
    route_identity_from_source_bound_resolution_v1,
    route_identity_record_v1,
    route_identity_v1,
    source_bound_route_candidate_identities_v1,
    validate_public_route_clarification_catalog_v1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
)
from pure_integer_ai.experiments.conversation_source_bound_slot_catalog import (
    resolve_source_bound_slot_composition,
)


_ROOT = Path(__file__).resolve().parents[1]
_COURSE = (
    b"data/ph2/dlg_raw_public_route_clarification_course_v1.jsonl.sample")
_SURFACE_A = (
    b"data/ph2/dlg_raw_public_route_clarification_surface_v1_a.txt.sample")
_EXPECTED_ROUTE_IDENTITY_HEX = (
    "185cb124c327c22a145e3601db9f655eb3815166a111c499b583474cfc048658")
_EXPECTED_CANDIDATE_IDENTITIES_HEX = (
    "5b6673cb297449d491784ff2a47f62e765661cc1455e3e567d4d0f7f4124d251",
    "067b04bd493ce50e1493aae5949a946af61d352556cb212a6942d7708727c4b1",
)


def _copy_closure(target: Path) -> Path:
    """按 registry 完整复制 raw payload，避免物理路径或 mtime 参与语义。"""
    for logical_key in PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1:
        relative = logical_key.decode("ascii")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(_ROOT / relative, destination)
    return target


def _real_ambiguity_resolution():
    """从真实公开 runtime 重演当前 V3 两候选 ambiguity，不读取 private 数据。"""
    closure = load_public_source_payload_closure_from_root(_ROOT)
    runtime = build_public_dialogue_runtime_v1(closure)
    resolution = resolve_source_bound_slot_composition(
        runtime.source_bound_slot_catalog,
        runtime.base_catalog,
        runtime.active_catalog,
        tuple(ord(item) for item in "东岸入口何时启用？"),
        closure,
    )
    return closure, resolution


def test_course_compiles_one_source_bound_route_with_complete_reentry_options() -> None:
    """单条课程必须给出完整提示与 resolver canonical 顺序的两个完整重输问句。"""
    closure = load_public_source_payload_closure_from_root(_ROOT)
    catalog = load_public_route_clarification_catalog_from_closure(closure)
    form = catalog.forms[0]

    assert len(catalog.forms) == 1
    assert form.result_code == DLG_RAW_REJECT_LEXICAL_AMBIGUOUS
    assert form.matched_frame_count == 2
    assert bytes(form.route_identity_u8).hex() == _EXPECTED_ROUTE_IDENTITY_HEX
    assert bytes(form.output_u8).decode("utf-8") == (
        "此输入对应多个已学习路径，请重输其中一个完整问题：\n"
        "澄川码头何时启用？\n"
        "北川站东门何时启用？")
    assert form.output_readback.output_u8 == form.output_u8
    assert tuple(bytes(option.candidate_identity_u8).hex()
                 for option in form.options) == _EXPECTED_CANDIDATE_IDENTITIES_HEX
    assert tuple(bytes(option.option_surface_u8).decode("utf-8")
                 for option in form.options) == (
        "澄川码头何时启用？",
        "北川站东门何时启用？",
    )
    assert catalog.form_for_route_identity_u8(form.route_identity_u8) == form
    assert validate_public_route_clarification_catalog_v1(catalog, closure) == catalog


def test_route_identity_helpers_match_real_v3_resolution_without_closure_cycle() -> None:
    """candidate 只取自身 record；route 只取 code/count/input/有序 candidate identities。"""
    _closure, resolution = _real_ambiguity_resolution()
    identities = source_bound_route_candidate_identities_v1(resolution)

    assert resolution.result_code == DLG_RAW_REJECT_LEXICAL_AMBIGUOUS
    assert tuple(bytes(identity).hex() for identity in identities) == (
        _EXPECTED_CANDIDATE_IDENTITIES_HEX)
    assert tuple(candidate_identity_v1(candidate)
                 for candidate in resolution.target_candidates) == identities
    record = route_identity_record_v1(
        resolution.result_code,
        resolution.matched_frame_count,
        resolution.input_scalars,
        identities,
    )
    assert record[:4] == (1, DLG_RAW_REJECT_LEXICAL_AMBIGUOUS, 2, 9)
    assert route_identity_v1(
        resolution.result_code,
        resolution.matched_frame_count,
        resolution.input_scalars,
        identities,
    ) == route_identity_from_source_bound_resolution_v1(resolution)
    assert bytes(route_identity_from_source_bound_resolution_v1(resolution)).hex() == (
        _EXPECTED_ROUTE_IDENTITY_HEX)


def test_catalog_finds_the_only_form_for_real_source_bound_resolution() -> None:
    """catalog lookup 必须同时锁 route identity 与 candidate identity 的原有顺序。"""
    closure, resolution = _real_ambiguity_resolution()
    catalog = load_public_route_clarification_catalog_from_closure(closure)

    form = catalog.form_for_source_bound_resolution(resolution)

    assert form is not None
    assert form.route_identity_u8 == route_identity_from_source_bound_resolution_v1(
        resolution)
    assert tuple(option.candidate_identity_u8 for option in form.options) == (
        source_bound_route_candidate_identities_v1(resolution))
    for source in (
            form.course_source,
            form.output_course_source,
            *form.output_surface_sources,
            *(source for option in form.options
              for source in (option.candidate_course_source,
                             *option.surface_sources))):
        payload = closure.payload_for(bytes(source.logical_key_u8))
        assert payload[source.span_start:source.span_end] == bytes(source.span_u8)


def test_course_or_surface_drift_fails_closed_before_runtime(tmp_path: Path) -> None:
    """candidate vector、course canonical bytes 或 A/B witness 变化均不得形成 selector form。"""
    candidate_root = _copy_closure(tmp_path / "candidate-drift")
    candidate_path = candidate_root / _COURSE.decode("ascii")
    original = bytes.fromhex(_EXPECTED_CANDIDATE_IDENTITIES_HEX[0]).hex().encode("ascii")
    forged = b"0" + original[1:]
    candidate_path.write_bytes(candidate_path.read_bytes().replace(original, forged, 1))
    with pytest.raises(PublicRouteClarificationCatalogError):
        load_public_route_clarification_catalog_from_closure(
            load_public_source_payload_closure_from_root(candidate_root))

    course_root = _copy_closure(tmp_path / "course-drift")
    course_path = course_root / _COURSE.decode("ascii")
    course_path.write_bytes(course_path.read_bytes() + b" ")
    with pytest.raises(PublicRouteClarificationCatalogError):
        load_public_route_clarification_catalog_from_closure(
            load_public_source_payload_closure_from_root(course_root))

    surface_root = _copy_closure(tmp_path / "surface-drift")
    surface_path = surface_root / _SURFACE_A.decode("ascii")
    surface_path.write_bytes(surface_path.read_bytes() + b"\n")
    with pytest.raises(PublicRouteClarificationCatalogError):
        load_public_route_clarification_catalog_from_closure(
            load_public_source_payload_closure_from_root(surface_root))


def test_parser_contract_is_a_fixed_portable_integer_record() -> None:
    """移植实现必须先复现本课程 parser record 与 raw-u8 identity。"""
    assert public_route_clarification_course_parser_record_v1() == (
        1, 1, 1, 1, 1, 1, 1)
    assert public_route_clarification_course_parser_identity_v1() == (
        173, 249, 31, 145, 51, 46, 87, 244,
        223, 220, 39, 162, 186, 37, 196, 153,
        3, 38, 135, 25, 165, 57, 184, 162,
        123, 227, 6, 87, 109, 146, 7, 251,
    )
