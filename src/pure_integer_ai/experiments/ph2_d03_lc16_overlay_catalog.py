"""从公开冻结物构建 D-03 LC-16 后继 overlay。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from pure_integer_ai.experiments.ph2_carrier_directional_manifest_contract import (
    read_carrier_directional_manifest,
    verify_carrier_directional_files,
)
from pure_integer_ai.experiments.ph2_carrier_projection_mapper_contract import (
    read_carrier_projection_mapper_manifest,
    verify_carrier_projection_mapper_files,
)
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_d03_lc16_overlay_contract import (
    D03Lc16SuccessorOverlay,
    expected_coverage_cells,
    expected_evaluator_boundaries,
    expected_failure_dependencies,
    expected_generation_accounts,
    expected_resource_budgets,
    expected_scope_records,
)
from pure_integer_ai.experiments.ph2_d03_lc16_overlay_records import (
    D03Lc16OverlayError,
    OverlayCarrierCourse,
    OverlayCourseCase,
    OverlayEvidenceFile,
    OverlayFileIdentity,
)
from pure_integer_ai.experiments.ph2_d03_lc16_overlay_specs import (
    ARTIFACT_STATUS,
    ARTIFACT_VERSION,
    COURSE_STATE,
    EVIDENCE_ROLES,
    EXECUTION_STATE,
    FORMAT_VERSION,
    PARSER_IDENTITIES,
    RENDERER_KEY,
    RENDERER_MODE,
    RENDERER_VERSION,
)
from pure_integer_ai.experiments.ph2_d03_release_catalog import (
    FORMAL_GLOBAL_MANIFEST_PATH,
    FORMAL_RECEIPT_PATH,
)
from pure_integer_ai.experiments.ph2_d03_release_reader import D03ReleaseReader
from pure_integer_ai.experiments.ph2_document_container_carrier_contract import (
    read_document_container_carrier_manifest,
    verify_document_container_carrier_files,
)
from pure_integer_ai.experiments.ph2_html_carrier_contract import (
    read_html_carrier_manifest,
    verify_html_carrier_files,
)
from pure_integer_ai.experiments.ph2_language_coverage_v2_contract import (
    IN_SCOPE_CARRIER_KEYS,
    W02_RECEIPT_SHA256,
)
from pure_integer_ai.experiments.ph2_markdown_carrier_contract import (
    read_markdown_carrier_manifest,
    verify_markdown_carrier_files,
)
from pure_integer_ai.experiments.ph2_math_notation_carrier_contract import (
    read_math_notation_carrier_manifest,
    verify_math_notation_carrier_files,
)
from pure_integer_ai.experiments.ph2_plain_text_carrier_contract import (
    read_plain_text_carrier_manifest,
    verify_plain_text_carrier_files,
)
from pure_integer_ai.experiments.ph2_reference_link_embed_carrier_contract import (
    read_reference_link_embed_carrier_manifest,
    verify_reference_link_embed_carrier_files,
)
from pure_integer_ai.experiments.ph2_source_code_carrier_contract import (
    read_source_code_carrier_manifest,
    verify_source_code_carrier_files,
)
from pure_integer_ai.experiments.ph2_table_grid_carrier_contract import (
    read_table_grid_carrier_manifest,
    verify_table_grid_carrier_files,
)
from pure_integer_ai.experiments.ph2_transcribed_ocr_asr_carrier_contract import (
    read_transcribed_ocr_asr_carrier_manifest,
    verify_transcribed_ocr_asr_carrier_files,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_contract import (
    SAMPLE_KINDS,
    read_typed_carrier_pack_manifest,
    verify_typed_carrier_pack_files,
)


OVERLAY_MANIFEST_PATH = (
    "data/ph2/manifests/d03_lc16_successor_overlay_v1.json")
DIRECTIONAL_PARENT_PATH = (
    "data/ph2/manifests/lc16_carrier_directional_runtime_v1.json")
DIRECTIONAL_PARENT_SHA256 = (
    "c7119639340c9baa5d80c8b582df8131376c3c8dd182f9414717f553a942985e")
TYPED_CARRIER_PACK_PATH = (
    "data/ph2/manifests/lc16_typed_carrier_pack_v1.json")
TYPED_CARRIER_PACK_SHA256 = (
    "29d1d54e6c547c5f375f807c92c16c9c9c69c97eb143f697df073747be8b0aa2")
MAPPER_MANIFEST_PATH = (
    "data/ph2/manifests/lc16_carrier_projection_mapper_v1.json")
W03_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v1/w03_runtime_evidence_receipt_v1.json")
_EVIDENCE_PATHS = (
    (
        "src/pure_integer_ai/experiments/ph2_d03_lc16_overlay_catalog.py",
        "CATALOG",
    ),
    (
        "src/pure_integer_ai/experiments/ph2_d03_lc16_overlay_contract.py",
        "CONTRACT",
    ),
    (
        "src/pure_integer_ai/experiments/ph2_d03_lc16_overlay_records.py",
        "RECORDS",
    ),
    (
        "src/pure_integer_ai/experiments/ph2_d03_lc16_overlay_specs.py",
        "SPECS",
    ),
    ("tests/test_d03_lc16_successor_overlay.py", "TEST"),
)

_Reader = Callable[[str | Path], Any]
_Verifier = Callable[..., None]
_CARRIER_PROTOCOLS: dict[str, tuple[_Reader, _Verifier, str, str]] = {
    "DOCUMENT_CONTAINER": (
        read_document_container_carrier_manifest,
        verify_document_container_carrier_files,
        *PARSER_IDENTITIES["DOCUMENT_CONTAINER"],
    ),
    "HTML": (
        read_html_carrier_manifest,
        verify_html_carrier_files,
        *PARSER_IDENTITIES["HTML"],
    ),
    "MARKDOWN": (
        read_markdown_carrier_manifest,
        verify_markdown_carrier_files,
        *PARSER_IDENTITIES["MARKDOWN"],
    ),
    "MATH_NOTATION": (
        read_math_notation_carrier_manifest,
        verify_math_notation_carrier_files,
        *PARSER_IDENTITIES["MATH_NOTATION"],
    ),
    "PLAIN_TEXT": (
        read_plain_text_carrier_manifest,
        verify_plain_text_carrier_files,
        *PARSER_IDENTITIES["PLAIN_TEXT"],
    ),
    "REFERENCE_LINK_EMBED": (
        read_reference_link_embed_carrier_manifest,
        verify_reference_link_embed_carrier_files,
        *PARSER_IDENTITIES["REFERENCE_LINK_EMBED"],
    ),
    "SOURCE_CODE": (
        read_source_code_carrier_manifest,
        verify_source_code_carrier_files,
        *PARSER_IDENTITIES["SOURCE_CODE"],
    ),
    "TABLE_GRID": (
        read_table_grid_carrier_manifest,
        verify_table_grid_carrier_files,
        *PARSER_IDENTITIES["TABLE_GRID"],
    ),
    "TRANSCRIBED_OCR_ASR": (
        read_transcribed_ocr_asr_carrier_manifest,
        verify_transcribed_ocr_asr_carrier_files,
        *PARSER_IDENTITIES["TRANSCRIBED_OCR_ASR"],
    ),
}


class D03Lc16OverlayCatalogError(RuntimeError):
    """overlay 的 parent、历史发布或课程输入发生漂移。"""


def _path(root: Path, relative_path: str) -> Path:
    """在仓库根内解析一个必须存在的相对文件。"""
    target = (root / Path(*relative_path.split("/"))).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise D03Lc16OverlayCatalogError("overlay catalog 路径逃逸") from error
    if not target.is_file():
        raise D03Lc16OverlayCatalogError(
            f"overlay catalog 文件缺失: {relative_path}")
    return target


def _identity(root: Path, relative_path: str) -> OverlayFileIdentity:
    """计算仓内文件的严格身份。"""
    payload = _path(root, relative_path).read_bytes()
    return OverlayFileIdentity(
        relative_path, len(payload), hashlib.sha256(payload).hexdigest())


def _verify_frozen_identity(
        identity: OverlayFileIdentity, expected_sha256: str, *, where: str,
        ) -> None:
    """拒绝冻结 parent 被无声替换。"""
    if identity.sha256 != expected_sha256:
        raise D03Lc16OverlayCatalogError(f"{where} SHA-256 漂移")


def _verify_parent_chain(root: Path) -> None:
    """严格回验方向 parent、D-03 v1 发布和 mapper/carrier 依赖链。"""
    directional_identity = _identity(root, DIRECTIONAL_PARENT_PATH)
    _verify_frozen_identity(
        directional_identity, DIRECTIONAL_PARENT_SHA256,
        where="directional parent",
    )
    try:
        directional = read_carrier_directional_manifest(
            _path(root, DIRECTIONAL_PARENT_PATH))
        verify_carrier_directional_files(directional, repository_root=root)
        mapper = read_carrier_projection_mapper_manifest(
            _path(root, MAPPER_MANIFEST_PATH))
        verify_carrier_projection_mapper_files(mapper, repository_root=root)
        D03ReleaseReader.open(root, FORMAL_GLOBAL_MANIFEST_PATH)
    except Exception as error:
        raise D03Lc16OverlayCatalogError(
            "overlay parent 或 D-03 v1 无法严格回验") from error


def _w03_receipt(root: Path) -> tuple[OverlayFileIdentity, dict[str, Any]]:
    """回读 W-03 公开 receipt，并提取 W-02 的不可公开内容承诺 SHA。"""
    identity = _identity(root, W03_RECEIPT_PATH)
    try:
        payload = json.loads(_path(root, W03_RECEIPT_PATH).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise D03Lc16OverlayCatalogError("W-03 receipt 损坏") from error
    if (not isinstance(payload, dict)
            or payload.get("status") != "RUNTIME_EVIDENCED"
            or payload.get("w02_receipt_sha256") != W02_RECEIPT_SHA256
            or payload.get("d03_global_manifest_sha256")
            != _identity(root, FORMAL_GLOBAL_MANIFEST_PATH).sha256
            or payload.get("d03_receipt_sha256")
            != _identity(root, FORMAL_RECEIPT_PATH).sha256
            or not isinstance(payload.get("execution_state"), dict)
            or payload["execution_state"].get("W04_STARTED") != 0):
        raise D03Lc16OverlayCatalogError(
            "W-03 receipt、W-02 承诺或 W-04 边界漂移")
    return identity, payload


def _carrier_course(
        root: Path,
        carrier_key: str,
        dependency: Any,
        parent_cases: dict[Any, Any],
        parent_budgets: dict[str, Any],
        ) -> OverlayCarrierCourse:
    """把 parent 的 owner/split 与已物化 adapter payload 合并为单载体课程。"""
    reader, verifier, parser_package, parser_version = _CARRIER_PROTOCOLS[
        carrier_key]
    path = _path(root, dependency.relative_path)
    try:
        manifest = reader(path)
        verifier(manifest, repository_root=root)
    except Exception as error:
        raise D03Lc16OverlayCatalogError(
            f"{carrier_key} carrier manifest 无法严格回验") from error
    manifest_identity = _identity(root, dependency.relative_path)
    if (manifest_identity.size_bytes != dependency.byte_count
            or manifest_identity.sha256 != dependency.sha256
            or manifest.carrier_key != carrier_key):
        raise D03Lc16OverlayCatalogError(
            f"{carrier_key} mapper dependency identity 漂移")
    actual_parser_package = getattr(manifest, "parser_package", parser_package)
    actual_parser_version = getattr(manifest, "parser_version", parser_version)
    if (actual_parser_package != parser_package
            or actual_parser_version != parser_version):
        raise D03Lc16OverlayCatalogError(
            f"{carrier_key} parser identity 漂移")
    sample_identity = _identity(root, manifest.sample_relative_path)
    if sample_identity.sha256 != manifest.sample_sha256:
        raise D03Lc16OverlayCatalogError(
            f"{carrier_key} sample identity 漂移")
    materializations = {item.case_key: item for item in manifest.materializations}
    cases = []
    for case_key in manifest.case_keys:
        parent_case = parent_cases.get(case_key)
        materialization = materializations.get(case_key)
        if (parent_case is None or materialization is None
                or parent_case.carrier_key != carrier_key):
            raise D03Lc16OverlayCatalogError(
                f"{carrier_key} case、owner 或 materialization 缺失")
        cases.append(OverlayCourseCase(
            case_key,
            parent_case.sample_kind,
            parent_case.owner_key,
            parent_case.split,
            parent_case.directions,
            materialization.byte_count,
            materialization.sha256,
        ))
    if tuple(item.sample_kind for item in cases) != SAMPLE_KINDS:
        raise D03Lc16OverlayCatalogError(
            f"{carrier_key} 七类样本顺序漂移")
    budget = parent_budgets.get(carrier_key)
    if budget is None or manifest.budget != budget:
        raise D03Lc16OverlayCatalogError(
            f"{carrier_key} parent/adapter budget 漂移")
    return OverlayCarrierCourse(
        carrier_key,
        COURSE_STATE,
        manifest_identity,
        sample_identity,
        parser_package,
        parser_version,
        RENDERER_KEY,
        RENDERER_VERSION,
        RENDERER_MODE,
        CanonicalJsonObject.from_value(budget.to_dict()),
        tuple(cases),
    )


def _carrier_courses(root: Path) -> tuple[OverlayCarrierCourse, ...]:
    """严格回验 typed parent、mapper 当前九依赖并构造 63-case 课程。"""
    pack_identity = _identity(root, TYPED_CARRIER_PACK_PATH)
    _verify_frozen_identity(
        pack_identity, TYPED_CARRIER_PACK_SHA256, where="typed carrier pack")
    try:
        pack = read_typed_carrier_pack_manifest(
            _path(root, TYPED_CARRIER_PACK_PATH))
        verify_typed_carrier_pack_files(pack, repository_root=root)
        mapper = read_carrier_projection_mapper_manifest(
            _path(root, MAPPER_MANIFEST_PATH))
    except Exception as error:
        raise D03Lc16OverlayCatalogError(
            "typed parent 或 mapper 无法严格回验") from error
    dependencies = {item.carrier_key: item for item in mapper.dependencies}
    if tuple(sorted(dependencies)) != IN_SCOPE_CARRIER_KEYS:
        raise D03Lc16OverlayCatalogError("mapper 九类 dependency 未闭合")
    parent_cases = {item.case_key: item for item in pack.cases}
    parent_budgets = {item.carrier_key: item for item in pack.budgets}
    return tuple(
        _carrier_course(
            root, carrier_key, dependencies[carrier_key],
            parent_cases, parent_budgets,
        )
        for carrier_key in IN_SCOPE_CARRIER_KEYS
    )


def _evidence_files(root: Path) -> tuple[OverlayEvidenceFile, ...]:
    """绑定 overlay 的拆分规格、记录、合同、目录和专项测试。"""
    result = tuple(
        OverlayEvidenceFile(role, _identity(root, relative_path))
        for relative_path, role in _EVIDENCE_PATHS
    )
    if tuple(item.role for item in result) != EVIDENCE_ROLES:
        raise D03Lc16OverlayCatalogError("overlay evidence role 漂移")
    return result


def build_d03_lc16_successor_overlay(
        repository_root: str | Path,
        ) -> D03Lc16SuccessorOverlay:
    """构造只冻结课程与资格协议、绝不启动训练的 append-only overlay。"""
    root = Path(repository_root).resolve()
    _verify_parent_chain(root)
    w03_identity, _ = _w03_receipt(root)
    try:
        return D03Lc16SuccessorOverlay(
            FORMAT_VERSION,
            ARTIFACT_VERSION,
            ARTIFACT_STATUS,
            _identity(root, DIRECTIONAL_PARENT_PATH),
            _identity(root, FORMAL_GLOBAL_MANIFEST_PATH),
            _identity(root, FORMAL_RECEIPT_PATH),
            _identity(root, TYPED_CARRIER_PACK_PATH),
            W02_RECEIPT_SHA256,
            w03_identity,
            _carrier_courses(root),
            expected_scope_records(),
            expected_coverage_cells(),
            expected_evaluator_boundaries(),
            expected_resource_budgets(),
            expected_failure_dependencies(),
            expected_generation_accounts(),
            CanonicalJsonObject.from_value(EXECUTION_STATE),
            _evidence_files(root),
        )
    except D03Lc16OverlayError as error:
        raise D03Lc16OverlayCatalogError("overlay 构造失败") from error


__all__ = [
    "DIRECTIONAL_PARENT_PATH", "DIRECTIONAL_PARENT_SHA256",
    "D03Lc16OverlayCatalogError", "MAPPER_MANIFEST_PATH",
    "OVERLAY_MANIFEST_PATH", "TYPED_CARRIER_PACK_PATH",
    "TYPED_CARRIER_PACK_SHA256", "W03_RECEIPT_PATH",
    "build_d03_lc16_successor_overlay",
]
