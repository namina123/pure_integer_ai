"""DLG-05 v4 候选侧 metadata-only handoff 审计。

本模块只检查公开 K 盘 source bundle 的文件闭包、确定性身份、freeze/manifest
一致性和物理文件边界。它不读取 owner label，不创建 guard，不执行 query 或
formal，也不把 Markdown/HTML 投影导入运行时。默认只接受 K 盘路径；测试可显式
关闭盘符门，但生产调用不得关闭。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

from pure_integer_ai.experiments.conversation_heldout_v4_bundle import (
    ConversationHeldOutV4SourceBundle,
    digest_hex,
)
from pure_integer_ai.experiments.conversation_heldout_v4_family import (
    ConversationHeldOutV4Family,
    build_v4_family,
)
from pure_integer_ai.experiments.conversation_heldout_v4_freeze import (
    ConversationHeldOutV4Freeze,
    freeze_v4_bundle,
    verify_v4_freeze,
)
from pure_integer_ai.experiments.conversation_heldout_v4_projection import (
    render_v4_html,
    render_v4_markdown,
)


class ConversationHeldOutV4MetadataAuditError(RuntimeError):
    """v4 source bundle 无法满足 owner handoff 的只读元数据合同。"""


_EXPECTED_FILES = frozenset({
    Path("bundle.canonical.ints"),
    Path("freeze.json"),
    Path("artifact_manifest.json"),
    Path("projection") / "dlg05_v4_reading.md",
    Path("projection") / "dlg05_v4_reading.html",
})
_OPTIONAL_REPORT = Path("metadata_audit.json")
_REPARSE_POINT = 0x0400
_FORBIDDEN_PROJECTION_TOKENS = (
    b"selected_candidate",
    b"response_act",
    b"expected",
    b"pass",
    b"fail",
    b"labels",
)


def _fail(message: str) -> None:
    raise ConversationHeldOutV4MetadataAuditError(message)


def _root_path(root: str | Path, *, require_k_drive: bool) -> Path:
    """解析并限制审计根；生产默认拒绝非 K 盘回退。"""
    path = Path(root).resolve()
    if require_k_drive and path.drive.upper() != "K:":
        _fail("v4 metadata audit 生产根必须位于 K 盘")
    if not path.is_dir() or path.is_symlink():
        _fail("v4 metadata audit 根不存在或是链接")
    root_stat = os.stat(path, follow_symlinks=False)
    if getattr(root_stat, "st_file_attributes", 0) & _REPARSE_POINT:
        _fail("v4 metadata audit 根不得是 reparse point")
    return path


def _relative_files(root: Path) -> tuple[Path, ...]:
    """枚举根内文件并拒绝越界链接、重解析点和非单硬链接。"""
    files: list[Path] = []
    for path in root.rglob("*"):
        stat = os.stat(path, follow_symlinks=False)
        if getattr(stat, "st_file_attributes", 0) & _REPARSE_POINT:
            _fail("v4 metadata audit 拒绝 reparse point")
        if path.is_dir():
            if path.is_symlink():
                _fail("v4 metadata audit 发现目录链接")
            continue
        if not path.is_file() or path.is_symlink():
            _fail("v4 metadata audit 发现非普通文件")
        if getattr(stat, "st_nlink", 1) != 1:
            _fail("v4 metadata audit 文件硬链接数必须为 1")
        relative = path.relative_to(root)
        if (root / relative).resolve() != path.resolve():
            _fail("v4 metadata audit 文件路径越界")
        files.append(relative)
    return tuple(sorted(files))


def _canonical_json(value: object) -> bytes:
    """生成 metadata report 使用的确定性 JSON 字节。"""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    """只读取公开 manifest/freeze 元数据并拒绝尾随结构。"""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"{label} 不是可读规范 JSON")
        raise AssertionError from exc
    if not isinstance(value, dict):
        _fail(f"{label} 顶层必须是 object")
    return value


def _expected_freeze_doc(
        bundle: ConversationHeldOutV4SourceBundle,
        freeze: ConversationHeldOutV4Freeze,
        ) -> dict[str, object]:
    """从 typed 对象生成 freeze 的公开元数据形状。"""
    dependencies = bundle.dependencies
    return {
        "bundle_index": freeze.bundle_index,
        "dependencies": {
            "artifact_sha256": digest_hex(dependencies.artifact_sha256),
            "document_sha256": digest_hex(dependencies.document_sha256),
            "inventory_sha256": digest_hex(dependencies.inventory_sha256),
        },
        "payload_sha256": digest_hex(freeze.bundle_payload_sha256),
        "payload_size": freeze.bundle_payload_size,
        "schema": "dlg05-v4-freeze-v1",
        "source_count": freeze.source_count,
        "turn_count": freeze.turn_count,
    }


def _expected_manifest_doc(
        bundle: ConversationHeldOutV4SourceBundle,
        ) -> dict[str, object]:
    """从 typed bundle 生成 artifact manifest 的公开元数据形状。"""
    return {
        "bundle_index": bundle.index,
        "bundle_payload_size": bundle.payload_size,
        "bundle_sha256": digest_hex(bundle.payload_sha256),
        "candidate_count": sum(len(item.candidates) for item in bundle.turns),
        "projection": ["dlg05_v4_reading.md", "dlg05_v4_reading.html"],
        "schema": "dlg05-v4-family-artifact-manifest-v1",
        "source_count": len(bundle.sources),
        "turn_count": len(bundle.turns),
    }


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4MetadataAudit:
    """可供 owner handoff 读取的只读整数资格摘要。"""

    schema_version: int
    bundle_payload_size: int
    bundle_payload_sha256: tuple[int, ...]
    bundle_index: int
    turn_count: int
    source_count: int
    candidate_count: int
    file_count: int
    hard_link_clear: int
    manifest_clear: int
    freeze_clear: int
    projection_boundary_clear: int

    def __post_init__(self) -> None:
        values = (
            self.schema_version, self.bundle_payload_size, self.bundle_index,
            self.turn_count, self.source_count, self.candidate_count,
            self.file_count, self.hard_link_clear, self.manifest_clear,
            self.freeze_clear, self.projection_boundary_clear,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ConversationHeldOutV4MetadataAuditError(
                "metadata audit 字段必须是非负严格整数")
        if self.schema_version != 1 or len(self.bundle_payload_sha256) != 32:
            raise ConversationHeldOutV4MetadataAuditError(
                "metadata audit schema 或 bundle SHA 非法")
        if any(type(value) is not int or value < 0 or value > 255
               for value in self.bundle_payload_sha256):
            raise ConversationHeldOutV4MetadataAuditError(
                "metadata audit bundle SHA 含非法字节")
        if not all((self.hard_link_clear, self.manifest_clear,
                    self.freeze_clear, self.projection_boundary_clear)):
            raise ConversationHeldOutV4MetadataAuditError(
                "metadata audit 未达到 handoff ready")

    @property
    def status(self) -> str:
        """返回只读 owner handoff 状态；不表示能力或 formal 结果。"""
        return "READY_FOR_OWNER_HANDOFF"

    def stable_key(self) -> tuple[int, ...]:
        """返回不含路径和文本的完整整数资格键。"""
        return (
            self.schema_version, self.bundle_payload_size, self.bundle_index,
            self.turn_count, self.source_count, self.candidate_count,
            self.file_count, self.hard_link_clear, self.manifest_clear,
            self.freeze_clear, self.projection_boundary_clear,
            *self.bundle_payload_sha256,
        )

    def document(self) -> dict[str, object]:
        """生成可供独立 owner 会话回读的 metadata-only 文档。"""
        return {
            "bundle_index": self.bundle_index,
            "bundle_payload_sha256": digest_hex(self.bundle_payload_sha256),
            "bundle_payload_size": self.bundle_payload_size,
            "candidate_count": self.candidate_count,
            "file_count": self.file_count,
            "freeze_clear": self.freeze_clear,
            "hard_link_clear": self.hard_link_clear,
            "manifest_clear": self.manifest_clear,
            "projection_boundary_clear": self.projection_boundary_clear,
            "schema": "dlg05-v4-metadata-audit-v1",
            "source_count": self.source_count,
            "status": self.status,
            "turn_count": self.turn_count,
        }


def audit_v4_family_artifacts(
        root: str | Path,
        *,
        require_k_drive: bool = True,
        family: ConversationHeldOutV4Family | None = None,
        ) -> ConversationHeldOutV4MetadataAudit:
    """只读核对 v4 K 盘 artifact，并返回 handoff 元数据资格摘要。"""
    target = _root_path(root, require_k_drive=require_k_drive)
    files = _relative_files(target)
    allowed = _EXPECTED_FILES | ({_OPTIONAL_REPORT}
                                  if _OPTIONAL_REPORT in files else set())
    if set(files) != allowed:
        _fail("v4 metadata audit 文件闭包与预期不一致")
    current = family if family is not None else build_v4_family()
    bundle = current.bundle
    freeze = current.freeze
    payload = (" ".join(str(value) for value in bundle.canonical_payload) + "\n").encode()
    if (target / "bundle.canonical.ints").read_bytes() != payload:
        _fail("v4 metadata audit canonical payload 漂移")
    if _read_json(target / "freeze.json", label="freeze") != _expected_freeze_doc(bundle, freeze):
        _fail("v4 metadata audit freeze metadata 漂移")
    if _read_json(target / "artifact_manifest.json", label="manifest") != _expected_manifest_doc(bundle):
        _fail("v4 metadata audit artifact manifest 漂移")
    verify_v4_freeze(bundle, freeze)
    expected_projection = {
        Path("projection/dlg05_v4_reading.md"): render_v4_markdown(bundle).encode("utf-8"),
        Path("projection/dlg05_v4_reading.html"): render_v4_html(bundle).encode("utf-8"),
    }
    for relative, expected in expected_projection.items():
        actual = (target / relative).read_bytes()
        if actual != expected:
            _fail(f"v4 metadata audit projection 漂移：{relative}")
        lowered = actual.lower()
        if any(token in lowered for token in _FORBIDDEN_PROJECTION_TOKENS):
            _fail(f"v4 metadata audit projection 含标签字段：{relative}")
    result = ConversationHeldOutV4MetadataAudit(
        1,
        bundle.payload_size,
        bundle.payload_sha256,
        bundle.index,
        len(bundle.turns),
        len(bundle.sources),
        sum(len(item.candidates) for item in bundle.turns),
        len(_EXPECTED_FILES),
        1,
        1,
        1,
        1,
    )
    if _OPTIONAL_REPORT in files:
        expected_report = _canonical_json(result.document())
        if (target / _OPTIONAL_REPORT).read_bytes() != expected_report:
            _fail("v4 metadata audit report 漂移")
    return result


def write_v4_metadata_audit(
        root: str | Path,
        audit: ConversationHeldOutV4MetadataAudit,
        ) -> Path:
    """在 bundle 根幂等写入 metadata-only handoff 摘要，不创建 owner receipt。"""
    target = _root_path(root, require_k_drive=True)
    if not isinstance(audit, ConversationHeldOutV4MetadataAudit):
        raise TypeError("audit 类型错误")
    path = target / _OPTIONAL_REPORT
    payload = _canonical_json(audit.document())
    if path.exists() and path.read_bytes() != payload:
        _fail("metadata audit report 已存在且内容不同")
    if not path.exists():
        path.write_bytes(payload)
    return path


__all__ = [
    "ConversationHeldOutV4MetadataAudit",
    "ConversationHeldOutV4MetadataAuditError",
    "audit_v4_family_artifacts",
    "write_v4_metadata_audit",
]
