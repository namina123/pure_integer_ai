"""J-F2 公开 Core/artifact bundle manifest 合同。

该模块只绑定公开仓库中的可复核文件，以及 W09 receipt 已公开的安全承诺。
W09 的 learned dump 仍属于仓库外封存材料；本 manifest 不读取、不复制该材料。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any

from pure_integer_ai.experiments.j_f1_facility_receipt import (
    read_j_f1_facility_receipt,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)


FORMAT_VERSION = 1
ARTIFACT_KIND = "PH2_J_F2_CORE_ARTIFACT_BUNDLE"
ARTIFACT_VERSION = "J-F2-CORE-ARTIFACT-20260806-A"
MANIFEST_PATH = "data/ph2/manifests/j_f2_core_artifact_manifest_v1.json"
J_F1_RECEIPT_PATH = "data/ph2/manifests/j_f1_facility_receipt_v1.json"
D03_RECEIPT_PATH = "data/ph2/manifests/d03_v1/ph2_d03_post_publication_receipt_v1.json"
D03_GLOBAL_PATH = "data/ph2/manifests/d03_v1/ph2_global_course_manifest_v1.json"
W09_RECEIPT_PATH = "data/ph2/manifests/d03_v1/w09_runtime_evidence_receipt_v1.json"
W09_STAGE_PATH = "data/ph2/manifests/d03_v1/stages/w09_stage_manifest_v1.json"

_W_RECEIPT_PATHS = (
    ("W01_PROTOCOL", "data/ph2/manifests/w01_v1/ph2_w01_stage0_receipt_v2.json"),
    ("W02_FORMAL_COMMITMENT", "data/ph2/manifests/d03_v1/w03_runtime_evidence_receipt_v1.json"),
    ("W02_LC16_SUPPLEMENTAL", "data/ph2/manifests/w02_lc16_supplemental_runtime_receipt_v1.json"),
    ("W03_LC16_SUPPLEMENTAL", "data/ph2/manifests/w03_lc16_supplemental_runtime_receipt_v1.json"),
    ("W03_RUNTIME", "data/ph2/manifests/d03_v1/w03_runtime_evidence_receipt_v1.json"),
    ("W04_RUNTIME", "data/ph2/manifests/d03_v1/w04_runtime_evidence_receipt_v1.json"),
    ("W05_RUNTIME", "data/ph2/manifests/d03_v1/w05_runtime_evidence_receipt_v1.json"),
    ("W06_RUNTIME", "data/ph2/manifests/d03_v1/w06_runtime_evidence_receipt_v1.json"),
    ("W07_RUNTIME", "data/ph2/manifests/d03_v1/w07_runtime_evidence_receipt_v1.json"),
    ("W08_RUNTIME", "data/ph2/manifests/d03_v1/w08_runtime_evidence_receipt_v1.json"),
    ("W09_RUNTIME", W09_RECEIPT_PATH),
)
_D03_STAGE_PATHS = tuple(
    f"data/ph2/manifests/d03_v1/stages/w{ordinal:02d}_stage_manifest_v1.json"
    for ordinal in range(1, 10)
)
_EXCLUDED_PATHS = frozenset({
    MANIFEST_PATH,
    "src/pure_integer_ai/experiments/ph2_j_f2_contract.py",
    "src/pure_integer_ai/experiments/j_f2_core_artifact_manifest.py",
    "src/pure_integer_ai/experiments/run_j_f2_core_artifact_manifest.py",
})
_REQUIRED_ROLES = frozenset({
    "BACKEND_CAPABILITY",
    "CORE_IMPLEMENTATION",
    "D03_PUBLICATION",
    "LC16",
    "PRIMITIVE_IMPLEMENTATION",
    "REQUIRED_REPORT",
    "SCHEMA_COURSE",
    "SEGMENT_LOCATION_RECOVERY",
    "W_RECEIPTS",
})
_SAFE_W09_COMMITMENT_KEYS = (
    "candidate_contract_sha256",
    "candidate_first_run_guard_sha256",
    "candidate_host_freeze_sha256",
    "candidate_inference_state_sha256",
    "candidate_public_head_commit_sha1",
    "candidate_terminal_seal_sha256",
    "dump_manifest_sha256",
)
_COMMITMENT_ONLY_BOUNDARY = (
    "W09 learned state is Git-external and remains sealed; "
    "only public commitments are bound here."
)


class CoreArtifactManifestError(RuntimeError):
    """公开 Core/artifact manifest 缺失、漂移或违反封存边界。"""


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    """校验对象字段集合精确，拒绝静默扩展和删减。"""
    if not isinstance(value, dict) or set(value) != keys:
        raise CoreArtifactManifestError(f"{where} 字段不精确")
    return value


def _relative(value: str, *, where: str) -> str:
    """校验仓库相对 POSIX 路径不越界、不含 Windows 分隔符。"""
    path = PurePosixPath(value)
    if (not isinstance(value, str) or not value or path.is_absolute()
            or ".." in path.parts or "\\" in value):
        raise CoreArtifactManifestError(f"{where} 相对路径非法")
    return value


def _sha256(value: str, *, where: str) -> str:
    """校验小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise CoreArtifactManifestError(f"{where} SHA-256 非法")
    return value


def _sha1(value: str, *, where: str) -> str:
    """校验小写 SHA-1 commit 身份。"""
    if (not isinstance(value, str) or len(value) != 40
            or any(char not in "0123456789abcdef" for char in value)):
        raise CoreArtifactManifestError(f"{where} SHA-1 非法")
    return value


def _integer(value: Any, *, where: str, minimum: int = 0) -> int:
    """校验 manifest 中的离散整数。"""
    if type(value) is not int or value < minimum:
        raise CoreArtifactManifestError(f"{where} 整数非法")
    return value


def _identity(root: Path, relative_path: str) -> tuple[int, str] | None:
    """读取公开文件的大小和 SHA-256；越界或缺失返回 None。"""
    _relative(relative_path, where="file identity")
    target = (root / Path(*relative_path.split("/"))).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        return None
    payload = target.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _read_json(root: Path, relative_path: str) -> dict[str, Any]:
    """读取一个公开 canonical JSON 对象，不接受多余尾换行。"""
    identity = _identity(root, relative_path)
    if identity is None:
        raise CoreArtifactManifestError(f"公开 JSON 缺失: {relative_path}")
    path = root / Path(*relative_path.split("/"))
    payload = path.read_bytes()
    body = payload[:-1] if payload.endswith(b"\n") else payload
    if payload.endswith(b"\n\n"):
        raise CoreArtifactManifestError(f"公开 JSON 多余尾换行: {relative_path}")
    try:
        value = parse_canonical_json_bytes(body, require_object=True)
    except Exception as error:
        raise CoreArtifactManifestError(f"公开 JSON 非 canonical: {relative_path}") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != body:
        raise CoreArtifactManifestError(f"公开 JSON 字节漂移: {relative_path}")
    return value


def _tracked_public_paths(root: Path) -> tuple[str, ...]:
    """取得 Git 已跟踪的公开文件，避免把工作树临时材料写入 manifest。"""
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CoreArtifactManifestError("无法读取公开 Git 文件清单") from error
    values = tuple(item.decode("utf-8") for item in result.stdout.split(b"\0") if item)
    return tuple(sorted(values))


def _roles(relative_path: str) -> tuple[str, ...]:
    """按稳定的公开路径规则赋予 artifact bundle 角色。"""
    roles = {"REQUIRED_REPORT"} if relative_path.startswith("data/ph2/") else set()
    if "/" not in relative_path:
        roles.add("PUBLIC_CONFIGURATION")
    if relative_path.startswith("src/pure_integer_ai/"):
        roles.add("CORE_IMPLEMENTATION")
    if relative_path.startswith("data/ph2/"):
        name = PurePosixPath(relative_path).name.lower()
        if "/d03_v1/" in relative_path or name.startswith("j_lg_d03"):
            roles.add("D03_PUBLICATION")
        if name.startswith("lc16_"):
            roles.add("LC16")
        if ("course" in name or "schema" in name or name.startswith("lc")
                or name.startswith("gg") or name.startswith("j_lc")
                or name.startswith("language_capability")
                or name.startswith("authored_")):
            roles.add("SCHEMA_COURSE")
        if ("receipt" in name or name.startswith("w0")
                or name.startswith("j_f1_facility")):
            roles.add("W_RECEIPTS")
    source_name = PurePosixPath(relative_path).name.lower()
    if relative_path.startswith("src/pure_integer_ai/"):
        if any(token in source_name for token in ("primitive", "symbol_domain", "edge_types")):
            roles.add("PRIMITIVE_IMPLEMENTATION")
        if any(token in source_name for token in ("backend", "storage_backend")):
            roles.add("BACKEND_CAPABILITY")
        if any(token in source_name for token in (
                "segment", "location", "recovery", "sharded_delta")):
            roles.add("SEGMENT_LOCATION_RECOVERY")
    return tuple(sorted(roles))


@dataclass(frozen=True, order=True)
class CoreFileBinding:
    """一个公开 artifact 文件的稳定身份和职责角色。"""

    relative_path: str
    sha256: str
    size_bytes: int
    roles: tuple[str, ...]

    def __post_init__(self) -> None:
        """校验文件身份、路径和角色的离散不变量。"""
        _relative(self.relative_path, where="Core file")
        _sha256(self.sha256, where=self.relative_path)
        _integer(self.size_bytes, where=self.relative_path)
        if not self.roles or tuple(sorted(set(self.roles))) != self.roles:
            raise CoreArtifactManifestError("Core file roles 必须唯一稳定排序")

    def to_dict(self) -> dict[str, Any]:
        """返回 canonical 公开文件身份。"""
        return {
            "relative_path": self.relative_path,
            "roles": list(self.roles),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, order=True)
class ArtifactIdentity:
    """一个 receipt 或 manifest 的仓库内身份。"""

    relative_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        """校验身份路径与摘要。"""
        _relative(self.relative_path, where="artifact identity")
        _sha256(self.sha256, where=self.relative_path)
        _integer(self.size_bytes, where=self.relative_path)

    def to_dict(self) -> dict[str, Any]:
        """返回 canonical 文件身份。"""
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, order=True)
class ReceiptBinding:
    """一个公开阶段 receipt 的身份和已审计状态。"""

    role: str
    identity: ArtifactIdentity
    status: str

    def __post_init__(self) -> None:
        """校验 receipt 角色和状态。"""
        if not self.role or self.status not in {
                "PASS", "POST_PUBLISH_VERIFIED", "RUNTIME_EVIDENCED",
                "W01_PROTOCOL_VERIFIED"}:
            raise CoreArtifactManifestError("receipt binding 状态非法")

    def to_dict(self) -> dict[str, Any]:
        """返回 canonical receipt binding。"""
        return {
            "identity": self.identity.to_dict(),
            "role": self.role,
            "status": self.status,
        }


def _identity_object(root: Path, relative_path: str) -> ArtifactIdentity:
    """形成一个现存公开文件的身份对象。"""
    value = _identity(root, relative_path)
    if value is None:
        raise CoreArtifactManifestError(f"artifact 文件缺失: {relative_path}")
    size_bytes, sha256 = value
    return ArtifactIdentity(relative_path, sha256, size_bytes)


def _file_bindings(root: Path) -> tuple[CoreFileBinding, ...]:
    """从 Git 跟踪文件中形成固定的公开 artifact 文件闭包。"""
    selected = []
    for relative_path in _tracked_public_paths(root):
        normalized = relative_path.replace("\\", "/")
        if normalized in _EXCLUDED_PATHS or normalized.startswith("tests/"):
            continue
        if not (normalized.startswith("src/pure_integer_ai/")
                or normalized.startswith("data/ph2/")
                or "/" not in normalized):
            continue
        identity = _identity(root, normalized)
        if identity is None:
            raise CoreArtifactManifestError(f"tracked artifact 文件缺失: {normalized}")
        size_bytes, sha256 = identity
        selected.append(CoreFileBinding(
            normalized, sha256, size_bytes, _roles(normalized)))
    bindings = tuple(sorted(selected))
    if len({item.relative_path for item in bindings}) != len(bindings):
        raise CoreArtifactManifestError("Core file identity 重复")
    roles = {role for item in bindings for role in item.roles}
    if not _REQUIRED_ROLES.issubset(roles):
        raise CoreArtifactManifestError("Core artifact 角色覆盖不完整")
    return bindings


def _d03_identities(root: Path) -> tuple[ArtifactIdentity, ...]:
    """形成 D-03 发布、global、stage 和 invalidation 身份。"""
    paths = (D03_RECEIPT_PATH, D03_GLOBAL_PATH,
             "data/ph2/manifests/d03_v1/stage_invalidation_graph_v1.json",
             *_D03_STAGE_PATHS)
    return tuple(_identity_object(root, item) for item in paths)


def _receipt_bindings(root: Path) -> tuple[ReceiptBinding, ...]:
    """读取并形成 W01-W09 与 LC-16 公开 receipt 身份。"""
    result: list[ReceiptBinding] = []
    for role, path in _W_RECEIPT_PATHS:
        value = _read_json(root, path)
        status = str(value.get("status", ""))
        expected = {
            "W01_PROTOCOL": "W01_PROTOCOL_VERIFIED",
            "W02_FORMAL_COMMITMENT": "RUNTIME_EVIDENCED",
        }.get(role, "PASS" if "SUPPLEMENTAL" in role else "RUNTIME_EVIDENCED")
        if role == "W02_FORMAL_COMMITMENT":
            expected = "RUNTIME_EVIDENCED"
        if status != expected:
            raise CoreArtifactManifestError(f"{role} receipt 状态不是 {expected}")
        if role == "W02_FORMAL_COMMITMENT":
            if value.get("w02_receipt_sha256") != (
                    "6b1344bfb226ea2488760987a838b4a7d4016f14831d6ed58c78b9ff0e45a2eb"):
                raise CoreArtifactManifestError("W02 commitment 漂移")
        result.append(ReceiptBinding(role, _identity_object(root, path), status))
    return tuple(result)


def _safe_w09_commitments(root: Path) -> dict[str, Any]:
    """从公开 W09 receipt 提取不含原始 payload 的安全承诺。"""
    value = _read_json(root, W09_RECEIPT_PATH)
    candidate = value.get("candidate_evidence")
    if not isinstance(candidate, dict):
        raise CoreArtifactManifestError("W09 candidate evidence 缺失")
    result = {key: candidate.get(key) for key in _SAFE_W09_COMMITMENT_KEYS}
    for key in _SAFE_W09_COMMITMENT_KEYS:
        if key.endswith("_sha256"):
            _sha256(result[key], where=f"W09 {key}")
        else:
            _sha1(result[key], where=f"W09 {key}")
    result.update({
        "learned_payload_redistributed": 0,
        "commitment_only_boundary": _COMMITMENT_ONLY_BOUNDARY,
    })
    return result


def _runtime_bindings(root: Path) -> dict[str, Any]:
    """提取 fresh/resume/clone/worker 和 Core 身份的公开运行承诺。"""
    w09 = _read_json(root, W09_RECEIPT_PATH)
    resource = w09.get("resource", {})
    rollback = w09.get("rollback", {})
    recovery = w09.get("candidate_evidence", {}).get("recovery_protocol", {})
    if (resource.get("status") != "PASS"
            or resource.get("fresh_resume_equivalent") != 1
            or resource.get("worker_1_2_4_invariant") != 1
            or rollback.get("status") != "PASS"):
        raise CoreArtifactManifestError("W09 recovery/worker 承诺未通过")
    receipt = read_j_f1_facility_receipt(root, verify_runtime=False)
    receipt_value = receipt.to_dict()
    core_before = receipt_value["identity_bindings"]["core_before"]
    core_after = receipt_value["identity_bindings"]["core_after"]
    if core_before["sha256"] != core_after["sha256"]:
        raise CoreArtifactManifestError("J-F1 Core 非 bit-identical")
    return {
        "j_f1": {
            "core_after_sha256": core_after["sha256"],
            "core_before_sha256": core_before["sha256"],
            "core_bit_identical": 1,
            "worker_counts": list(receipt_value["facility_evidence"]["worker_counts"]),
        },
        "w09": {
            "fresh_resume_equivalent": resource["fresh_resume_equivalent"],
            "recovery_protocol": recovery,
            "rollback_status": rollback["status"],
            "worker_1_2_4_invariant": resource["worker_1_2_4_invariant"],
        },
    }


@dataclass(frozen=True)
class CoreArtifactManifest:
    """完整公开 Core/artifact bundle manifest。"""

    format_version: int
    artifact_kind: str
    artifact_version: str
    manifest_relative_path: str
    manifest_self_excluded: int
    file_bindings: tuple[CoreFileBinding, ...]
    d03_identities: tuple[ArtifactIdentity, ...]
    receipt_bindings: tuple[ReceiptBinding, ...]
    j_f1_identity: ArtifactIdentity
    w09_commitments: dict[str, Any]
    runtime_bindings: dict[str, Any]

    def __post_init__(self) -> None:
        """保持公开封存身份、排序和边界状态不可变。"""
        if (self.format_version != FORMAT_VERSION
                or self.artifact_kind != ARTIFACT_KIND
                or self.artifact_version != ARTIFACT_VERSION):
            raise CoreArtifactManifestError("Core artifact identity 漂移")
        if self.manifest_relative_path != MANIFEST_PATH or self.manifest_self_excluded != 1:
            raise CoreArtifactManifestError("manifest self-excluded 边界非法")
        if tuple(sorted(self.file_bindings)) != self.file_bindings:
            raise CoreArtifactManifestError("Core file bindings 未排序")
        if len({item.relative_path for item in self.file_bindings}) != len(self.file_bindings):
            raise CoreArtifactManifestError("Core file bindings 路径重复")
        if any(item.relative_path in _EXCLUDED_PATHS for item in self.file_bindings):
            raise CoreArtifactManifestError("Core file bindings 含 J-F2 施工文件")
        roles = {role for item in self.file_bindings for role in item.roles}
        if not _REQUIRED_ROLES.issubset(roles):
            raise CoreArtifactManifestError("Core artifact 角色覆盖不完整")
        if tuple(sorted(self.d03_identities)) != self.d03_identities:
            raise CoreArtifactManifestError("D03 identities 未排序")
        expected_d03_paths = tuple(sorted((
            D03_RECEIPT_PATH, D03_GLOBAL_PATH,
            "data/ph2/manifests/d03_v1/stage_invalidation_graph_v1.json",
            *_D03_STAGE_PATHS,
        )))
        if tuple(item.relative_path for item in self.d03_identities) != expected_d03_paths:
            raise CoreArtifactManifestError("D03 identity 路径漂移")
        if tuple(item.role for item in self.receipt_bindings) != tuple(
                role for role, _ in _W_RECEIPT_PATHS):
            raise CoreArtifactManifestError("receipt binding 顺序漂移")
        if tuple(item.identity.relative_path for item in self.receipt_bindings) != tuple(
                path for _, path in _W_RECEIPT_PATHS):
            raise CoreArtifactManifestError("receipt binding 路径漂移")
        if self.j_f1_identity.relative_path != J_F1_RECEIPT_PATH:
            raise CoreArtifactManifestError("J-F1 identity 路径漂移")
        expected = set(_SAFE_W09_COMMITMENT_KEYS) | {
            "learned_payload_redistributed", "commitment_only_boundary"}
        if set(self.w09_commitments) != expected:
            raise CoreArtifactManifestError("W09 commitment 字段不精确")
        for key in _SAFE_W09_COMMITMENT_KEYS:
            if key.endswith("_sha256"):
                _sha256(self.w09_commitments[key], where=f"W09 {key}")
            else:
                _sha1(self.w09_commitments[key], where=f"W09 {key}")
        if (self.w09_commitments["learned_payload_redistributed"] != 0
                or self.w09_commitments["commitment_only_boundary"]
                != _COMMITMENT_ONLY_BOUNDARY):
            raise CoreArtifactManifestError("W09 commitment 边界漂移")

    def to_dict(self) -> dict[str, Any]:
        """返回 canonical、公开且不含原始 learned payload 的 manifest。"""
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_version": self.artifact_version,
            "d03_identities": [item.to_dict() for item in self.d03_identities],
            "file_bindings": [item.to_dict() for item in self.file_bindings],
            "format_version": self.format_version,
            "j_f1_identity": self.j_f1_identity.to_dict(),
            "manifest_relative_path": self.manifest_relative_path,
            "manifest_self_excluded": self.manifest_self_excluded,
            "receipt_bindings": [item.to_dict() for item in self.receipt_bindings],
            "runtime_bindings": self.runtime_bindings,
            "w09_commitments": self.w09_commitments,
        }

    def canonical_bytes(self) -> bytes:
        """返回带单尾换行的 canonical JSON。"""
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        """返回 manifest 的 SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def build_core_artifact_manifest(repository_root: str | Path) -> CoreArtifactManifest:
    """从公开 receipt 和 Git 跟踪文件形成 J-F2 Core manifest。"""
    root = Path(repository_root).resolve()
    d03_value = _read_json(root, D03_RECEIPT_PATH)
    if d03_value.get("status") != "POST_PUBLISH_VERIFIED":
        raise CoreArtifactManifestError("D03 尚未完成公开发布核验")
    if d03_value.get("publication_state", {}).get("d03_published") != 1:
        raise CoreArtifactManifestError("D03 published gate 未通过")
    _read_json(root, D03_GLOBAL_PATH)
    _read_json(root, W09_STAGE_PATH)
    j_f1_identity = _identity_object(root, J_F1_RECEIPT_PATH)
    read_j_f1_facility_receipt(root, verify_runtime=False)
    return CoreArtifactManifest(
        FORMAT_VERSION, ARTIFACT_KIND, ARTIFACT_VERSION, MANIFEST_PATH, 1,
        _file_bindings(root), _d03_identities(root), _receipt_bindings(root),
        j_f1_identity, _safe_w09_commitments(root), _runtime_bindings(root))


def read_core_artifact_manifest(
        repository_root: str | Path,
        path: str | Path = MANIFEST_PATH,
        *, verify_files: bool = True) -> CoreArtifactManifest:
    """严格读取并回验公开 Core manifest 及其安全承诺。"""
    root = Path(repository_root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / Path(*str(target).replace("\\", "/").split("/"))
    payload = target.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise CoreArtifactManifestError("Core manifest newline 非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except Exception as error:
        raise CoreArtifactManifestError("Core manifest JSON 非 canonical") from error
    raw = _exact(value, {
        "artifact_kind", "artifact_version", "d03_identities", "file_bindings",
        "format_version", "j_f1_identity", "manifest_relative_path",
        "manifest_self_excluded", "receipt_bindings", "runtime_bindings",
        "w09_commitments",
    }, where="CoreArtifactManifest")
    files: list[CoreFileBinding] = []
    for item in raw["file_bindings"]:
        entry = _exact(item, {"relative_path", "roles", "sha256", "size_bytes"}, where="CoreFileBinding")
        files.append(CoreFileBinding(
            str(entry["relative_path"]), str(entry["sha256"]),
            entry["size_bytes"], tuple(str(role) for role in entry["roles"])))
    d03: list[ArtifactIdentity] = []
    for item in raw["d03_identities"]:
        entry = _exact(item, {"relative_path", "sha256", "size_bytes"}, where="D03 identity")
        d03.append(ArtifactIdentity(str(entry["relative_path"]), str(entry["sha256"]), entry["size_bytes"]))
    receipts: list[ReceiptBinding] = []
    for item in raw["receipt_bindings"]:
        entry = _exact(item, {"identity", "role", "status"}, where="ReceiptBinding")
        identity = _exact(entry["identity"], {"relative_path", "sha256", "size_bytes"}, where="Receipt identity")
        receipts.append(ReceiptBinding(
            str(entry["role"]), ArtifactIdentity(
                str(identity["relative_path"]), str(identity["sha256"]), identity["size_bytes"]),
            str(entry["status"])))
    j_f1 = _exact(raw["j_f1_identity"], {"relative_path", "sha256", "size_bytes"}, where="J-F1 identity")
    manifest = CoreArtifactManifest(
        raw["format_version"], str(raw["artifact_kind"]), str(raw["artifact_version"]),
        str(raw["manifest_relative_path"]), raw["manifest_self_excluded"],
        tuple(files), tuple(d03), tuple(receipts),
        ArtifactIdentity(str(j_f1["relative_path"]), str(j_f1["sha256"]), j_f1["size_bytes"]),
        dict(raw["w09_commitments"]), dict(raw["runtime_bindings"]))
    if manifest.canonical_bytes() != payload:
        raise CoreArtifactManifestError("Core manifest canonical bytes 漂移")
    if verify_files:
        for item in manifest.file_bindings:
            if _identity(root, item.relative_path) != (item.size_bytes, item.sha256):
                raise CoreArtifactManifestError(f"Core 文件身份漂移: {item.relative_path}")
        for item in (*manifest.d03_identities, manifest.j_f1_identity,
                     *(binding.identity for binding in manifest.receipt_bindings)):
            if _identity(root, item.relative_path) != (item.size_bytes, item.sha256):
                raise CoreArtifactManifestError(f"公开 artifact identity 漂移: {item.relative_path}")
        _validate_public_commitments(root, manifest)
    return manifest


def _validate_public_commitments(root: Path, manifest: CoreArtifactManifest) -> None:
    """把 manifest 中的阶段状态和安全承诺重新绑定到公开 receipt。"""
    if manifest.w09_commitments != _safe_w09_commitments(root):
        raise CoreArtifactManifestError("W09 commitment 漂移")
    if manifest.runtime_bindings != _runtime_bindings(root):
        raise CoreArtifactManifestError("公开 runtime binding 漂移")


def publish_core_artifact_manifest(
        repository_root: str | Path,
        *, target: str | Path = MANIFEST_PATH) -> CoreArtifactManifest:
    """首次以 xb 发布 manifest，拒绝覆盖既有封存材料。"""
    root = Path(repository_root).resolve()
    manifest = build_core_artifact_manifest(root)
    destination = Path(target)
    if not destination.is_absolute():
        destination = root / Path(*str(destination).replace("\\", "/").split("/"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(manifest.canonical_bytes())
    except FileExistsError as error:
        raise CoreArtifactManifestError("Core manifest 已发布，禁止覆盖") from error
    restored = read_core_artifact_manifest(root, destination)
    if restored != manifest:
        raise CoreArtifactManifestError("Core manifest 发布回读不一致")
    return restored


__all__ = [
    "ARTIFACT_KIND", "ARTIFACT_VERSION", "CoreArtifactManifest",
    "CoreArtifactManifestError", "CoreFileBinding", "D03_GLOBAL_PATH",
    "D03_RECEIPT_PATH", "FORMAT_VERSION", "J_F1_RECEIPT_PATH", "MANIFEST_PATH",
    "ReceiptBinding", "build_core_artifact_manifest", "publish_core_artifact_manifest",
    "read_core_artifact_manifest",
]
