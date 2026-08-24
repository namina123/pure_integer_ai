"""DLG-05 v4 external-capsule runtime artifact 的无标签过渡闭包。

本模块只接受由 ``read_v4_external_input_capsule`` 导入、随后经过实际 H-00、G-01
和 G-00 至 G-03 的 runtime result。它在另一此前不存在的 K 盘根发布 bundle、receipt、
freeze 与阅读投影，并以 manifest-last 闭合。external capsule 只是输入 transport 声明；
本模块明确不产生独立来源、held-out、owner、label、guard、formal 或能力资格。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

from pure_integer_ai.experiments.conversation_heldout_v4_bundle import (
    digest_hex,
)
from pure_integer_ai.experiments.conversation_heldout_v4_candidate_runtime import (
    ConversationHeldOutV4CandidateRuntimeResult,
    ConversationHeldOutV4RuntimeSourceCapsule,
    V4_RUNTIME_CODE_CLOSURE_SCHEMA,
    V4_RUNTIME_SOURCE_ORIGIN_EXTERNAL,
    read_v4_runtime_inventory,
    run_v4_candidate_runtime,
)
from pure_integer_ai.experiments.conversation_heldout_v4_external_input_capsule import (
    ConversationHeldOutV4ExternalCapsuleError,
    read_v4_external_input_capsule,
)
from pure_integer_ai.experiments.conversation_heldout_v4_freeze import (
    verify_v4_freeze,
)
from pure_integer_ai.experiments.conversation_heldout_v4_projection import (
    render_v4_html,
    render_v4_markdown,
)
from pure_integer_ai.experiments.ph2_dataset_core import canonical_json_bytes
from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    decode_integer_tuple,
    encode_integer_tuple,
)


V4_RUNTIME_ARTIFACT_KIND = "DLG05_V4_EXTERNAL_CAPSULE_RUNTIME_ARTIFACT_V1"
V4_RUNTIME_ARTIFACT_SCHEMA = "dlg05-v4-external-capsule-runtime-artifact-v1"
V4_RUNTIME_FREEZE_SCHEMA = "dlg05-v4-runtime-freeze-v1"
V4_RUNTIME_ARTIFACT_UNQUALIFIED = "EXTERNAL_CAPSULE_RUNTIME_UNQUALIFIED"

_BUNDLE_FILE = Path("bundle.canonical.ints")
_RECEIPT_FILE = Path("runtime_receipt.canonical.ints")
_FREEZE_FILE = Path("freeze.json")
_PROJECTION_DIRECTORY = Path("projection")
_MARKDOWN_FILE = _PROJECTION_DIRECTORY / "dlg05_v4_reading.md"
_HTML_FILE = _PROJECTION_DIRECTORY / "dlg05_v4_reading.html"
_MANIFEST_FILE = Path("artifact_manifest.json")
_PRE_MANIFEST_FILES = frozenset({
    _BUNDLE_FILE,
    _RECEIPT_FILE,
    _FREEZE_FILE,
    _MARKDOWN_FILE,
    _HTML_FILE,
})
_EXPECTED_FILES = _PRE_MANIFEST_FILES | {_MANIFEST_FILE}
_REPARSE_POINT = 0x0400


# object-model: exception
class ConversationHeldOutV4RuntimeArtifactError(RuntimeError):
    """v4 runtime artifact 的来源、运行身份、文件闭包或审计结果不完整。"""


def _fail(message: str) -> None:
    """统一产生本模块的 fail-closed 错误。"""
    raise ConversationHeldOutV4RuntimeArtifactError(message)


def _sha256(value: bytes) -> tuple[int, ...]:
    """以整数 tuple 返回输入字节的 SHA-256。"""
    return tuple(hashlib.sha256(value).digest())


def _sha256_hex(value: bytes) -> str:
    """返回 manifest 文件身份使用的小写 SHA-256。"""
    return hashlib.sha256(value).hexdigest()


def _encoded_key_sha256(value: tuple[int, ...]) -> str:
    """对完整整数键的规范 codec 字节求 SHA，不以摘要替代原键。"""
    return _sha256_hex(encode_integer_tuple(value))


def _is_reparse(path: Path) -> bool:
    """在 resolve 前检查 Windows reparse attribute，普通平台保持 false。"""
    try:
        stat = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return bool(getattr(stat, "st_file_attributes", 0) & _REPARSE_POINT)


def _existing_normal_directory(path: Path, *, label: str) -> Path:
    """逐级拒绝 link/reparse 后解析一个已存在普通目录。"""
    if not path.is_absolute() or ".." in path.parts:
        _fail(f"{label} 必须是不含 .. 的绝对路径")
    for current in (path, *path.parents):
        if current.is_symlink() or _is_reparse(current):
            _fail(f"{label} 含链接或 reparse point")
        if not current.exists():
            _fail(f"{label} 不存在")
        if current == current.parent:
            break
    resolved = path.resolve()
    if (not resolved.is_dir() or resolved.is_symlink()
            or _is_reparse(resolved)):
        _fail(f"{label} 不是普通目录")
    return resolved


def _read_root(
        root: str | Path, *, require_k_drive: bool, label: str,
        ) -> Path:
    """读取前验证 source/artifact 根位于普通目录和规定数据盘。"""
    target = _existing_normal_directory(Path(root), label=label)
    if require_k_drive and target.drive.upper() != "K:":
        _fail(f"{label} 必须位于 K 盘")
    return target


def _new_root(
        root: str | Path, *, require_k_drive: bool,
        ) -> Path:
    """只接受此前不存在、父级已验证且生产位于 K 盘的 artifact 根。"""
    raw = Path(root)
    if not raw.is_absolute() or ".." in raw.parts:
        _fail("runtime artifact root 必须是不含 .. 的绝对路径")
    if os.path.lexists(raw) or raw.is_symlink() or _is_reparse(raw):
        _fail("runtime artifact root 必须此前不存在且不是链接")
    parent = _existing_normal_directory(
        raw.parent, label="runtime artifact parent")
    target = parent / raw.name
    if require_k_drive and target.drive.upper() != "K:":
        _fail("runtime artifact 生产根必须位于 K 盘")
    if target.exists() or os.path.lexists(target):
        _fail("runtime artifact root 已存在")
    return target


def _require_disjoint(source: Path, artifact: Path) -> None:
    """拒绝 source 与 artifact 根相同、嵌套或借父级覆盖彼此。"""
    if (source == artifact or source.is_relative_to(artifact)
            or artifact.is_relative_to(source)):
        _fail("external capsule root 与 runtime artifact root 不得重叠")


def _relative_path(root: Path, relative: Path) -> Path:
    """只将固定相对文件名解析到 root 内，拒绝额外路径成分。"""
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        _fail("runtime artifact 文件相对路径非法")
    path = root.joinpath(*relative.parts)
    if not path.is_relative_to(root):
        _fail("runtime artifact 文件路径越界")
    return path


def _normal_relative_directory(root: Path, relative: Path) -> Path:
    """验证 root 内既有子目录无 symlink/reparse，且解析后仍不越界。"""
    directory = _relative_path(root, relative)
    if (not directory.is_dir() or directory.is_symlink()
            or _is_reparse(directory)):
        _fail(f"runtime artifact 目录非法: {relative.as_posix()}")
    try:
        resolved = directory.resolve()
    except OSError as exc:
        _fail(f"runtime artifact 目录无法解析: {relative.as_posix()}")
        raise AssertionError from exc
    if not resolved.is_relative_to(root):
        _fail(f"runtime artifact 目录路径越界: {relative.as_posix()}")
    return directory


def _plain_file(root: Path, relative: Path) -> Path:
    """返回 root 内唯一普通单硬链接文件，拒绝链接、reparse 和路径逃逸。"""
    path = _relative_path(root, relative)
    if relative.parent != Path("."):
        _normal_relative_directory(root, relative.parent)
    if not path.is_file() or path.is_symlink() or _is_reparse(path):
        _fail(f"runtime artifact 文件不是普通文件: {relative.as_posix()}")
    try:
        stat = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        _fail(f"runtime artifact 文件不可读取: {relative.as_posix()}")
        raise AssertionError from exc
    if getattr(stat, "st_nlink", 1) != 1:
        _fail(f"runtime artifact 文件硬链接数必须为 1: {relative.as_posix()}")
    try:
        resolved = path.resolve()
    except OSError as exc:
        _fail(f"runtime artifact 文件无法解析: {relative.as_posix()}")
        raise AssertionError from exc
    if not resolved.is_relative_to(root):
        _fail(f"runtime artifact 文件路径越界: {relative.as_posix()}")
    return path


def _collect_files(root: Path) -> frozenset[Path]:
    """递归枚举所有叶文件，同时拒绝任何非 projection 的目录或重解析点。"""
    found: set[Path] = set()

    def visit(directory: Path, relative: Path) -> None:
        try:
            children = tuple(directory.iterdir())
        except OSError as exc:
            _fail("runtime artifact root 不可枚举")
            raise AssertionError from exc
        for child in children:
            child_relative = relative / child.name
            if child.is_symlink() or _is_reparse(child):
                _fail("runtime artifact 不允许链接或 reparse point")
            if child.is_dir():
                if child_relative != _PROJECTION_DIRECTORY:
                    _fail("runtime artifact 只允许 projection 子目录")
                _normal_relative_directory(root, child_relative)
                visit(child, child_relative)
                continue
            if not child.is_file():
                _fail("runtime artifact 含非普通文件")
            _plain_file(root, child_relative)
            found.add(child_relative)

    visit(root, Path("."))
    return frozenset(found)


def _require_file_set(root: Path, *, published: bool) -> None:
    """强制 pre-manifest 或完整 published 文件闭包，不允许缓存、标签或额外目录。"""
    expected = _EXPECTED_FILES if published else _PRE_MANIFEST_FILES
    if _collect_files(root) != expected:
        _fail("runtime artifact 文件闭包与预期不一致")


def _read_plain_bytes(root: Path, relative: Path) -> bytes:
    """在读取前后均复核文件物理边界，避免普通文件被静默替换。"""
    path = _plain_file(root, relative)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        _fail(f"runtime artifact 文件读取失败: {relative.as_posix()}")
        raise AssertionError from exc
    _plain_file(root, relative)
    if not payload:
        _fail(f"runtime artifact 文件不能为空: {relative.as_posix()}")
    return payload


def _write_exclusive(root: Path, relative: Path, payload: bytes) -> None:
    """以 xb+fsync 写入一个新文件，并立即核验内容和普通文件边界。"""
    if not isinstance(payload, bytes) or not payload:
        _fail("runtime artifact 写入 payload 不能为空")
    path = _relative_path(root, relative)
    if relative.parent != Path("."):
        _normal_relative_directory(root, relative.parent)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        _fail(f"runtime artifact 不可覆盖写入失败: {relative.as_posix()}")
        raise AssertionError from exc
    _plain_file(root, relative)
    if _read_plain_bytes(root, relative) != payload:
        _fail(f"runtime artifact 写入后内容漂移: {relative.as_posix()}")


def _dependencies_document(result: ConversationHeldOutV4CandidateRuntimeResult) -> dict[str, str]:
    """导出 result 已冻结的三项 provenance SHA。"""
    dependencies = result.bundle.dependencies
    return {
        "artifact_sha256": digest_hex(dependencies.artifact_sha256),
        "document_sha256": digest_hex(dependencies.document_sha256),
        "inventory_sha256": digest_hex(dependencies.inventory_sha256),
    }


def _freeze_document(result: ConversationHeldOutV4CandidateRuntimeResult) -> dict[str, object]:
    """生成包含完整 typed freeze 身份的 canonical JSON，而非只落摘要。"""
    freeze = result.freeze
    bundle = result.bundle
    return {
        "artifact_kind": V4_RUNTIME_ARTIFACT_KIND,
        "bundle": {
            "index": bundle.index,
            "payload_sha256": digest_hex(bundle.payload_sha256),
            "payload_size": bundle.payload_size,
        },
        "dependencies": _dependencies_document(result),
        "format_version": 1,
        "freeze_stable_key": list(freeze.stable_key()),
        "schema": V4_RUNTIME_FREEZE_SCHEMA,
        "source_count": len(bundle.sources),
        "turn_count": len(bundle.turns),
    }


def _runtime_inventory_document(
        result: ConversationHeldOutV4CandidateRuntimeResult,
        ) -> dict[str, object]:
    """导出 runtime 同次冻结的完整执行代码闭包和公开 sample identity。"""
    inventory = result.identity.runtime_inventory
    return {
        "execution_code": {
            "closure_schema": V4_RUNTIME_CODE_CLOSURE_SCHEMA,
            "closure_sha256": digest_hex(
                inventory.execution_code_closure_sha256),
            "file_count": len(inventory.execution_code),
            "files": [
                {
                    "path": item.relative_path,
                    "sha256": digest_hex(item.sha256),
                    "size": item.size,
                }
                for item in inventory.execution_code
            ],
            "total_size": inventory.execution_code_total_size,
        },
        "surface_sample": {
            "path": inventory.surface_sample_path,
            "sha256": digest_hex(inventory.surface_sample_sha256),
            "size": inventory.surface_sample_size,
        },
    }


def _projection_payloads(
        result: ConversationHeldOutV4CandidateRuntimeResult,
        ) -> dict[Path, bytes]:
    """从同一 bundle 纯函数重算人类阅读投影，不能回读旧投影。"""
    return {
        _MARKDOWN_FILE: render_v4_markdown(result.bundle).encode("utf-8"),
        _HTML_FILE: render_v4_html(result.bundle).encode("utf-8"),
    }


def _artifact_payloads(
        result: ConversationHeldOutV4CandidateRuntimeResult,
        ) -> dict[Path, bytes]:
    """构造 manifest 之前的全部无标签文件字节，所有 payload 均来自同次 result。"""
    verify_v4_freeze(result.bundle, result.freeze)
    payloads = {
        _BUNDLE_FILE: encode_integer_tuple(result.bundle.canonical_payload),
        _RECEIPT_FILE: encode_integer_tuple(result.receipt.canonical_payload),
        _FREEZE_FILE: canonical_json_bytes(_freeze_document(result)),
    }
    payloads.update(_projection_payloads(result))
    if set(payloads) != _PRE_MANIFEST_FILES:
        _fail("runtime artifact payload 集合漂移")
    return payloads


def _manifest_document(
        result: ConversationHeldOutV4CandidateRuntimeResult,
        payloads: dict[Path, bytes],
        ) -> dict[str, object]:
    """绑定 capsule、真实 runtime、bundle/freeze、投影及五个 manifest 前 payload 文件。"""
    capsule = result.capsule
    identity = result.identity
    bundle = result.bundle
    receipt = result.receipt
    if capsule.external_producer_key is None:
        _fail("external capsule 缺 producer key")
    if capsule.external_producer_declaration is None:
        _fail("external capsule 缺 producer declaration")
    files = {
        relative.as_posix(): {
            "sha256": _sha256_hex(payload),
            "size": len(payload),
        }
        for relative, payload in sorted(
            payloads.items(), key=lambda item: item[0].as_posix())
    }
    markdown_payload = payloads[_MARKDOWN_FILE]
    html_payload = payloads[_HTML_FILE]
    return {
        "artifact_kind": V4_RUNTIME_ARTIFACT_KIND,
        "bundle": {
            "candidate_count": sum(len(turn.candidates) for turn in bundle.turns),
            "case_count": len({turn.case_key for turn in bundle.turns}),
            "index": bundle.index,
            "payload_sha256": digest_hex(bundle.payload_sha256),
            "payload_size": bundle.payload_size,
            "source_count": len(bundle.sources),
            "turn_count": len(bundle.turns),
        },
        "external_capsule": {
            "input_sha256": digest_hex(identity.input_sha256),
            "manifest_sha256": digest_hex(capsule.manifest_sha256),
            "origin": capsule.origin,
            "producer": {
                "declaration": capsule.external_producer_declaration,
                "producer_key": list(capsule.external_producer_key.components),
            },
            "provenance": _dependencies_document(result),
            "stable_key_sha256": _encoded_key_sha256(capsule.stable_key()),
        },
        "files": files,
        "files_scope": "PRE_MANIFEST_PAYLOAD_FILES_ONLY",
        "format_version": 1,
        "freeze": {
            "schema_version": result.freeze.schema_version,
            "stable_key_sha256": _encoded_key_sha256(result.freeze.stable_key()),
        },
        "projection": {
            "html_sha256": _sha256_hex(html_payload),
            "markdown_sha256": _sha256_hex(markdown_payload),
        },
        "qualification_status": V4_RUNTIME_ARTIFACT_UNQUALIFIED,
        "runtime": {
            "candidate_realization_count": sum(
                len(frame.candidate_realizations) for frame in result.frames),
            "executor_call_count": sum(
                frame.executor_calls for frame in result.frames),
            "execution_reason_key": list(identity.execution_reason.stable_key()),
            "family_key": list(receipt.family_key.components),
            "frame_count": len(result.frames),
            "identity_stable_key_sha256": _encoded_key_sha256(
                identity.stable_key()),
            "protocol_key": list(identity.protocol.stable_key()),
            "receipt_integer_count": len(receipt.canonical_payload),
            "receipt_payload_sha256": digest_hex(receipt.payload_sha256),
            "renderer_sha256": digest_hex(identity.renderer_sha256),
            "route_key": list(identity.route.stable_key()),
            "surface_model_sha256": digest_hex(identity.surface_model_sha256),
        },
        "runtime_inventory": _runtime_inventory_document(result),
        "schema": V4_RUNTIME_ARTIFACT_SCHEMA,
    }


def _read_source_capsule(
        source_root: Path, *, require_k_drive: bool,
        ) -> ConversationHeldOutV4RuntimeSourceCapsule:
    """仅通过 R02 reader 导入 external capsule，不能直接构造 origin 标记。"""
    try:
        capsule = read_v4_external_input_capsule(
            source_root, require_k_drive=require_k_drive)
    except ConversationHeldOutV4ExternalCapsuleError as exc:
        _fail(f"external source capsule 不可验证: {exc}")
        raise AssertionError from exc
    if capsule.origin != V4_RUNTIME_SOURCE_ORIGIN_EXTERNAL or not capsule.is_external:
        _fail("runtime artifact 只接受 external source capsule")
    return capsule


def _validate_result(
        result: ConversationHeldOutV4CandidateRuntimeResult,
        capsule: ConversationHeldOutV4RuntimeSourceCapsule,
        ) -> None:
    """确保写入对象来自刚刚只读导入的 capsule 和当前 runtime code/sample。"""
    if not isinstance(result, ConversationHeldOutV4CandidateRuntimeResult):
        raise TypeError("runtime artifact 必须接收 CandidateRuntimeResult")
    if result.capsule != capsule:
        _fail("runtime result 与 external source capsule 身份不一致")
    if result.capsule.origin != V4_RUNTIME_SOURCE_ORIGIN_EXTERNAL:
        _fail("synthetic runtime result 不得进入 runtime artifact")
    if result.identity.runtime_inventory != read_v4_runtime_inventory():
        _fail("runtime code 或公开 surface sample 已在 result 后漂移")
    verify_v4_freeze(result.bundle, result.freeze)


def _make_paths(root: Path) -> "ConversationHeldOutV4RuntimeArtifactPaths":
    """从已验证 artifact root 生成固定闭包的路径结果。"""
    return ConversationHeldOutV4RuntimeArtifactPaths(
        root,
        root / _BUNDLE_FILE,
        root / _RECEIPT_FILE,
        root / _FREEZE_FILE,
        root / _MARKDOWN_FILE,
        root / _HTML_FILE,
        root / _MANIFEST_FILE,
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeArtifactPaths:
    """一个已发布 runtime artifact 闭包的固定路径集合。"""

    root: Path
    bundle: Path
    runtime_receipt: Path
    freeze: Path
    markdown: Path
    html: Path
    manifest: Path

    def __post_init__(self) -> None:
        """确保返回路径只覆盖本模块声明的六个固定文件。"""
        if not isinstance(self.root, Path) or not self.root.is_absolute():
            raise TypeError("runtime artifact paths root 类型错误")
        expected = (
            self.root / _BUNDLE_FILE,
            self.root / _RECEIPT_FILE,
            self.root / _FREEZE_FILE,
            self.root / _MARKDOWN_FILE,
            self.root / _HTML_FILE,
            self.root / _MANIFEST_FILE,
        )
        if (self.bundle, self.runtime_receipt, self.freeze, self.markdown,
                self.html, self.manifest) != expected:
            raise ConversationHeldOutV4RuntimeArtifactError(
                "runtime artifact paths 文件集合漂移")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeArtifactAudit:
    """只读 audit 的无标签结果；状态永远不升级为 source/owner/formal 资格。"""

    schema_version: int
    qualification_status: str
    external_capsule_bound: int
    file_count: int
    frame_count: int
    candidate_realization_count: int
    executor_call_count: int
    bundle_payload_sha256: tuple[int, ...]
    runtime_receipt_payload_sha256: tuple[int, ...]

    def __post_init__(self) -> None:
        """只接受 R03 的 unqualified 只读审计结论和完整 digest。"""
        if self.schema_version != 1:
            _fail("runtime artifact audit schema 非法")
        if self.qualification_status != V4_RUNTIME_ARTIFACT_UNQUALIFIED:
            _fail("runtime artifact audit 不得升级资格状态")
        if self.external_capsule_bound != 1:
            _fail("runtime artifact audit 必须绑定 external capsule")
        for label, value in (
                ("file count", self.file_count),
                ("frame count", self.frame_count),
                ("executor call count", self.executor_call_count)):
            if type(value) is not int or value <= 0:
                _fail(f"runtime artifact audit {label} 必须为正严格整数")
        if (type(self.candidate_realization_count) is not int
                or self.candidate_realization_count < 0):
            _fail("runtime artifact audit candidate realization count 必须为非负严格整数")
        for label, digest in (
                ("bundle payload sha256", self.bundle_payload_sha256),
                ("runtime receipt payload sha256",
                 self.runtime_receipt_payload_sha256)):
            if (not isinstance(digest, tuple) or len(digest) != 32
                    or any(type(item) is not int or item < 0 or item > 255
                           for item in digest)):
                _fail(f"runtime artifact audit {label} 非法")


def write_v4_runtime_artifact(
        artifact_root: str | Path,
        source_capsule_root: str | Path,
        *,
        require_k_drive: bool = True,
        ) -> ConversationHeldOutV4RuntimeArtifactPaths:
    """在新 K 盘根实际运行 capsule，并以 bundle、receipt、freeze、投影、manifest 顺序发布闭包。

    writer 不接受调用方构造的 result，避免手工拼接 receipt/bundle 冒充同次 runtime。它在写前和
    manifest 前两次只读 capsule，第一次导入后自行执行无标签 runtime；任一失败保留无 manifest 的
    partial root，绝不清理、覆盖或回退到 D 盘。
    """
    source = _read_root(
        source_capsule_root, require_k_drive=require_k_drive,
        label="external source capsule root")
    target = _new_root(artifact_root, require_k_drive=require_k_drive)
    _require_disjoint(source, target)
    source_before = _read_source_capsule(
        source, require_k_drive=require_k_drive)
    try:
        result = run_v4_candidate_runtime(source_before)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        _fail(f"external capsule runtime 无法运行: {exc}")
        raise AssertionError from exc
    _validate_result(result, source_before)
    payloads = _artifact_payloads(result)
    manifest_payload = canonical_json_bytes(_manifest_document(result, payloads))
    try:
        target.mkdir()
    except OSError as exc:
        _fail("runtime artifact root 创建失败")
        raise AssertionError from exc
    _existing_normal_directory(target, label="runtime artifact root")
    try:
        (target / _PROJECTION_DIRECTORY).mkdir()
    except OSError as exc:
        _fail("runtime artifact projection 目录创建失败")
        raise AssertionError from exc
    _normal_relative_directory(target, _PROJECTION_DIRECTORY)
    for relative in (
            _BUNDLE_FILE,
            _RECEIPT_FILE,
            _FREEZE_FILE,
            _MARKDOWN_FILE,
            _HTML_FILE):
        _write_exclusive(target, relative, payloads[relative])
    _require_file_set(target, published=False)
    source_after = _read_source_capsule(
        source, require_k_drive=require_k_drive)
    if source_after != source_before:
        _fail("external source capsule 在 runtime artifact 发布期间漂移")
    _validate_result(result, source_after)
    # manifest 是唯一发布标志，必须在 source 二次核验和其余文件闭合后最后写入。
    _write_exclusive(target, _MANIFEST_FILE, manifest_payload)
    _require_file_set(target, published=True)
    return _make_paths(target)


def audit_v4_runtime_artifact(
        artifact_root: str | Path,
        source_capsule_root: str | Path,
        *,
        require_k_drive: bool = True,
        ) -> ConversationHeldOutV4RuntimeArtifactAudit:
    """只读重跑 external capsule 的无标签 runtime，并逐字节比对 immutable artifact。

    审计不会写 report，也不会读取 label/private/formal/owner 内容；它只是验证 artifact
    仍可由当前公开 runtime 和同一 external capsule 复现，返回状态始终为 unqualified。
    """
    source = _read_root(
        source_capsule_root, require_k_drive=require_k_drive,
        label="external source capsule root")
    artifact = _read_root(
        artifact_root, require_k_drive=require_k_drive,
        label="runtime artifact root")
    _require_disjoint(source, artifact)
    source_before = _read_source_capsule(
        source, require_k_drive=require_k_drive)
    try:
        result = run_v4_candidate_runtime(source_before)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        _fail(f"external capsule runtime 无法重跑: {exc}")
        raise AssertionError from exc
    _validate_result(result, source_before)
    payloads = _artifact_payloads(result)
    manifest_payload = canonical_json_bytes(_manifest_document(result, payloads))
    _require_file_set(artifact, published=True)
    for relative, expected in payloads.items():
        actual = _read_plain_bytes(artifact, relative)
        if actual != expected:
            _fail(f"runtime artifact 文件内容漂移: {relative.as_posix()}")
    actual_manifest = _read_plain_bytes(artifact, _MANIFEST_FILE)
    if actual_manifest != manifest_payload:
        _fail("runtime artifact manifest 漂移")
    try:
        if decode_integer_tuple(_read_plain_bytes(artifact, _BUNDLE_FILE)) != (
                result.bundle.canonical_payload):
            _fail("runtime artifact bundle canonical ints 漂移")
        if decode_integer_tuple(_read_plain_bytes(artifact, _RECEIPT_FILE)) != (
                result.receipt.canonical_payload):
            _fail("runtime artifact receipt canonical ints 漂移")
    except (IntegerCodecError, TypeError, ValueError) as exc:
        _fail("runtime artifact canonical ints 非法")
        raise AssertionError from exc
    _require_file_set(artifact, published=True)
    source_after = _read_source_capsule(
        source, require_k_drive=require_k_drive)
    if source_after != source_before:
        _fail("external source capsule 在 runtime artifact audit 期间漂移")
    _validate_result(result, source_after)
    return ConversationHeldOutV4RuntimeArtifactAudit(
        1,
        V4_RUNTIME_ARTIFACT_UNQUALIFIED,
        1,
        len(_EXPECTED_FILES),
        len(result.frames),
        sum(len(frame.candidate_realizations) for frame in result.frames),
        sum(frame.executor_calls for frame in result.frames),
        result.bundle.payload_sha256,
        result.receipt.payload_sha256,
    )


__all__ = [
    "ConversationHeldOutV4RuntimeArtifactAudit",
    "ConversationHeldOutV4RuntimeArtifactError",
    "ConversationHeldOutV4RuntimeArtifactPaths",
    "V4_RUNTIME_ARTIFACT_KIND",
    "V4_RUNTIME_ARTIFACT_SCHEMA",
    "V4_RUNTIME_ARTIFACT_UNQUALIFIED",
    "audit_v4_runtime_artifact",
    "write_v4_runtime_artifact",
]
