"""DLG-RAW-13 terminal meta-act 课程/来源闭包的有界回归。"""
from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import pytest

from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1,
)
from pure_integer_ai.experiments.conversation_public_terminal_dialogue_act_catalog import (
    PUBLIC_TERMINAL_DIALOGUE_ACT_COVERAGE_UNSUPPORTED_V1,
    PUBLIC_TERMINAL_DIALOGUE_ACT_ROUTE_CLARIFICATION_V1,
    PublicTerminalDialogueActCatalogError,
    load_public_terminal_dialogue_act_catalog_from_closure,
    terminal_dialogue_act_course_parser_identity_v1,
    terminal_dialogue_act_course_parser_record_v1,
    validate_public_terminal_dialogue_act_catalog_v1,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    DLG_RAW_REJECT_LEXICAL_MISS,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    parse_canonical_json_bytes,
)


_ROOT = Path(__file__).resolve().parents[1]
_COURSE = (
    b"data/ph2/dlg_raw_public_terminal_dialogue_act_course_v1.jsonl.sample")
_SURFACE_A = (
    b"data/ph2/dlg_raw_public_terminal_dialogue_act_surface_v1_a.txt.sample")
_CONFORMANCE = (
    _ROOT / "tests/fixtures/dlg_raw_public_terminal_dialogue_act_v1_conformance.json")


def _copy_closure(target: Path) -> Path:
    """以完整 registry 建立独立 physical root，避免路径/mtime 参与逻辑结果。"""
    for key in PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1:
        relative = key.decode("ascii")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(_ROOT / relative, destination)
    return target


def test_course_compiles_two_source_bound_terminal_acts() -> None:
    """课程只允许 7/8 两条显式 mapping，surface 必须来自两份 CC0 witness。"""
    closure = load_public_source_payload_closure_from_root(_ROOT)
    catalog = load_public_terminal_dialogue_act_catalog_from_closure(closure)

    assert tuple((form.base_result_code, form.act_code) for form in catalog.forms) == (
        (DLG_RAW_REJECT_LEXICAL_MISS,
         PUBLIC_TERMINAL_DIALOGUE_ACT_COVERAGE_UNSUPPORTED_V1),
        (DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
         PUBLIC_TERMINAL_DIALOGUE_ACT_ROUTE_CLARIFICATION_V1),
    )
    assert tuple(bytes(form.output_u8).decode("utf-8") for form in catalog.forms) == (
        "当前公开对话资料尚未覆盖此输入。",
        "此输入对应多个已学习路径，请补充限定。",
    )
    assert all(len(form.surface_sources) == 2 for form in catalog.forms)
    assert validate_public_terminal_dialogue_act_catalog_v1(catalog, closure) == (
        catalog)


def test_course_parser_contract_has_stable_integer_identity() -> None:
    """移植实现必须先复现 parser 子集 record 与 raw-u8 identity。"""
    assert terminal_dialogue_act_course_parser_record_v1() == (
        1, 1, 1, 1, 1, 1, 1)
    assert terminal_dialogue_act_course_parser_identity_v1() == (
        103, 199, 218, 132, 110, 251, 170, 118,
        67, 169, 204, 150, 44, 104, 125, 180,
        228, 81, 238, 238, 179, 139, 191, 109,
        97, 37, 255, 229, 107, 178, 36, 233,
    )


def test_raw_byte_conformance_vector_matches_course_forms() -> None:
    """移植实现可只消费此规范 JSON vector 复现 parser 与两个 output bytes。"""
    payload = _CONFORMANCE.read_bytes()
    assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
    vector = parse_canonical_json_bytes(payload[:-1], require_object=True)
    catalog = load_public_terminal_dialogue_act_catalog_from_closure(
        load_public_source_payload_closure_from_root(_ROOT))

    assert vector["schema"] == 1
    assert tuple(vector["parser_record"]) == terminal_dialogue_act_course_parser_record_v1()
    assert vector["parser_identity_hex"] == bytes(
        terminal_dialogue_act_course_parser_identity_v1()).hex()
    assert tuple((item["base_result_code"], item["act_code"],
                  item["output_utf8_hex"]) for item in vector["cases"]) == tuple(
        (form.base_result_code, form.act_code, bytes(form.output_u8).hex())
        for form in catalog.forms)


def test_course_or_surface_byte_drift_fails_closed(tmp_path: Path) -> None:
    """课程 canonical JSON 或 witness SHA/span 漂移不得形成未验证 act。"""
    course_root = _copy_closure(tmp_path / "course-drift")
    course_path = course_root / _COURSE.decode("ascii")
    course_path.write_bytes(course_path.read_bytes() + b" ")
    with pytest.raises(PublicTerminalDialogueActCatalogError):
        load_public_terminal_dialogue_act_catalog_from_closure(
            load_public_source_payload_closure_from_root(course_root))

    surface_root = _copy_closure(tmp_path / "surface-drift")
    surface_path = surface_root / _SURFACE_A.decode("ascii")
    surface_path.write_bytes(surface_path.read_bytes() + b"\n")
    with pytest.raises(PublicTerminalDialogueActCatalogError):
        load_public_terminal_dialogue_act_catalog_from_closure(
            load_public_source_payload_closure_from_root(surface_root))
