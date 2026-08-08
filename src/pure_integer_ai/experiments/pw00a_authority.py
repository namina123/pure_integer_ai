"""PW-00A 正式装载 authority 的构建、排他发布与无 Git 回读。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any

from pure_integer_ai.experiments.artifact_verification_mode import (
    ARCHIVE_IDENTITY_VERIFY,
    CURRENT_HEAD_COMPATIBILITY_VERIFY,
    require_artifact_verification_mode,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)


FORMAT_VERSION = 1
ARTIFACT_KIND = "PURE_INTEGER_AI_PW00A_FORMAL_LOAD_AUTHORITY"
ARTIFACT_VERSION = "PW00A-FORMAL-LOAD-AUTHORITY-20260807-A"
RECEIPT_PATH = "data/ph2/manifests/pw00a_formal_load_authority_v1.json"
STATUS = "PW00A_FORMAL_LOAD_AUTHORITY_EVIDENCED"
PARENT_COMMIT = "e8c86f996da1983eb92844558159e49fd4a30135"

_DEPENDENCIES = (
    ("data/ph2/manifests/j_f2_core_artifact_manifest_v1.json", 232887,
     "d68e8e27f3d0cfe0632f3d51ff56adfe0087b1546a1a58fc5fe6f5062e5e6759"),
    ("data/ph2/manifests/j_f2_final_joint_seal_v1.json", 9068,
     "88f03b9c4b7a110ac031e99ae6f0d1bcc54b0523d6f35903628c65190c7d516f"),
    ("data/ph2/manifests/performance_p3_sqlite_trial_receipt_v2.json", 1153,
     "3a2ef9aae797e39c91e6dda95b3b68ffb112203768abb4d6482fbe12325b4665"),
    ("data/ph2/manifests/performance_successor_receipt_v1.json", 2117,
     "01ecdb29437d3ce7ac88e126cc7a4ccff206fc458290cf7ad65f80457d9ecb17"),
    ("data/ph2/manifests/performance_successor_receipt_v2.json", 2129,
     "53162d1a89da5f0c3e9dfb85384bfbab285d560999ec778a57eeed8df4b7a055"),
    ("data/ph2/manifests/performance_successor_receipt_v3.json", 1734,
     "13ca4b1d2256b097c7c93481ca632702b24336a161fa683250229e025ec13a8e"),
    ("data/ph2/manifests/performance_successor_receipt_v4.json", 1773,
     "71cc8c02c1dd7d663cc3b551e5c46d3584cf9fb37ed6072f3ea611c34b61e97f"),
    ("data/ph2/manifests/performance_successor_receipt_v5.json", 1952,
     "ddc0cf1861723ba68a946b2f3bd44a23edaa2b020c1a2834b759da04f21b36ca"),
    ("data/ph2/manifests/performance_successor_receipt_v6.json", 1655,
     "54949d483ed4f591ad51790f983c93fc16d0327058daceaa07fa1ba92669de1c"),
    ("data/ph2/manifests/performance_successor_receipt_v7.json", 1709,
     "03f06f5f75956c38fb3923ac9e48fe67a96f02c13a60e00ee522fed3915f13bf"),
    ("data/ph2/manifests/pw00a_w09_inference_state_v1.json", 101321,
     "6cab544a8b7c7c3be094c7d8972cfd5d6e7bbcf0fe6e4084be1b982cdcd60325"),
    ("data/ph2/manifests/source_successor_receipt_v1.json", 2846,
     "6b4d042bb82f5ec5f467a0d253254bb13e252d339b2294c3d234e38b9ff5977e"),
    ("data/ph2/manifests/struct_layout_successor_receipt_v1.json", 2439,
     "2b34bbed1c7dab67e0cadcfb0ff00f64fd28754e67d023c9e718f8633944d2d7"),
    ("data/ph2/manifests/struct_layout_successor_receipt_v2.json", 3008,
     "1047e03de245d89ce07a8e8b35119b1a38ebc79490ddb2b2e7d2239920e6733f"),
)

_ALLOWED_SOURCE_PATHS = (
    "src/pure_integer_ai/cognition/shared/formal_post_weaning.py",
    "src/pure_integer_ai/cognition/understanding/observe.py",
    "src/pure_integer_ai/crosscut/integer/valtypes.py",
    "src/pure_integer_ai/experiments/arithmetic_structure_runtime.py",
    "src/pure_integer_ai/experiments/collection.py",
    "src/pure_integer_ai/experiments/evaluation_runtime.py",
    "src/pure_integer_ai/experiments/language_structure_runtime.py",
    "src/pure_integer_ai/experiments/ph2_w09_inference.py",
    "src/pure_integer_ai/experiments/post_weaning_runtime.py",
    "src/pure_integer_ai/experiments/pw00a_authority.py",
    "src/pure_integer_ai/experiments/pw00a_formal_runtime.py",
    "src/pure_integer_ai/experiments/pw00a_formal_transaction.py",
    "src/pure_integer_ai/experiments/pw00a_inference_artifact.py",
    "src/pure_integer_ai/experiments/round_runtime.py",
    "src/pure_integer_ai/experiments/source_successor_receipt.py",
    "src/pure_integer_ai/storage/backend.py",
    "src/pure_integer_ai/storage/integer_codec.py",
    "src/pure_integer_ai/storage/query_hot_set.py",
    "src/pure_integer_ai/storage/sealed_segment.py",
    "src/pure_integer_ai/storage/segment_cache.py",
)

_V5_PATH = "src/pure_integer_ai/storage/query_hot_set.py"
_V5_RECORDED_PARENT = (
    "acc48c407048082866274126745eafbd106c4e49266feec8cc46c1ad9546f1c"
)
_V5_CORRECT_PARENT = (
    "acc48c407048082866274126745eafbd106c4e49266feec8cc46c1ad9546f1ca"
)
_V5_CURRENT = (
    "88ce77e8274c8dc5bede0f8ab786667de21872185f8c9e61ea5a0b9b6600219a"
)
_V5_PARENT_COMMIT = "86704b541635ef67902f201e34f5b34d945c4263"
_V5_CHANGE_COMMIT = "7cb4a3c080b885a96deddd1e5bf3906666fb5e51"


# object-model: exception
class PW00AAuthorityError(RuntimeError):
    """PW-00A authority 依赖、Git 连续性或规范字节不闭合。"""


def _sha256(payload: bytes) -> str:
    """返回字节的十六进制 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _strict_sha(value: object, *, label: str) -> str:
    """核验小写十六进制 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise PW00AAuthorityError(f"{label} SHA-256 非法")
    return value


def _relative(value: object, *, label: str) -> str:
    """拒绝绝对路径、反斜杠与目录上跳。"""
    if not isinstance(value, str) or not value:
        raise PW00AAuthorityError(f"{label} 相对路径非法")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise PW00AAuthorityError(f"{label} 相对路径非法")
    return value


def _target(root: Path, relative_path: str) -> Path:
    """解析并限制一个仓库内相对文件。"""
    _relative(relative_path, label="authority target")
    target = (root / Path(*relative_path.split("/"))).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise PW00AAuthorityError(f"authority 文件缺失: {relative_path}")
    return target


def _identity(root: Path, relative_path: str) -> tuple[int, str]:
    """返回仓库内文件的字节数和 SHA-256。"""
    payload = _target(root, relative_path).read_bytes()
    return len(payload), _sha256(payload)


def _git(root: Path, *args: str) -> bytes:
    """只供构建期执行固定 Git 只读命令。"""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise PW00AAuthorityError("PW00A authority Git 只读命令失败") from error


def _canonical_object(payload: bytes, *, label: str) -> dict[str, Any]:
    """严格解析单换行 canonical JSON object。"""
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise PW00AAuthorityError(f"{label} newline 非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except Exception as error:
        raise PW00AAuthorityError(f"{label} JSON 非 canonical") from error
    if canonical_json_bytes(value) + b"\n" != payload:
        raise PW00AAuthorityError(f"{label} canonical bytes 漂移")
    return value


def _validate_dependency_chain(root: Path) -> list[dict[str, Any]]:
    """核验固定依赖字节以及 performance v1-v7 的逐代前驱。"""
    bindings = []
    dependency_by_path = {path: digest for path, _, digest in _DEPENDENCIES}
    for path, size, digest in _DEPENDENCIES:
        if _identity(root, path) != (size, digest):
            raise PW00AAuthorityError(f"PW00A dependency 漂移: {path}")
        value = _canonical_object(_target(root, path).read_bytes(), label=path)
        transition = value.get("readiness_transition")
        if (transition is not None
                and path not in {
                    "data/ph2/manifests/j_f2_core_artifact_manifest_v1.json",
                    "data/ph2/manifests/j_f2_final_joint_seal_v1.json"}
                and transition.get("PW00A_STARTED") != 0):
            raise PW00AAuthorityError(f"历史依赖提前启动 PW00A: {path}")
        bindings.append({
            "relative_path": path,
            "sha256": digest,
            "size_bytes": size,
        })
    previous = dependency_by_path[
        "data/ph2/manifests/struct_layout_successor_receipt_v2.json"]
    for version in range(1, 8):
        path = f"data/ph2/manifests/performance_successor_receipt_v{version}.json"
        value = _canonical_object(_target(root, path).read_bytes(), label=path)
        prior = value.get("prior_successor_receipt")
        if not isinstance(prior, dict) or prior.get("sha256") != previous:
            raise PW00AAuthorityError(f"performance v{version} 前驱漂移")
        previous = dependency_by_path[path]
    return bindings


def _source_changes(root: Path, head: str) -> dict[str, str]:
    """返回 parent 到 HEAD 的精确 src 增删改集合。"""
    output = _git(
        root,
        "diff",
        "--name-status",
        PARENT_COMMIT,
        head,
        "--",
        "src",
    ).decode("utf-8")
    result: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) != 2 or parts[0] not in {"A", "M"}:
            raise PW00AAuthorityError("PW00A src 存在删除、重命名或未知变化")
        result[parts[1].replace("\\", "/")] = parts[0]
    if tuple(sorted(result)) != _ALLOWED_SOURCE_PATHS:
        raise PW00AAuthorityError("PW00A src change set 不等于固定 allowlist")
    return result


def _source_bindings(
        root: Path,
        changes: dict[str, str],
        ) -> list[dict[str, Any]]:
    """绑定 parent blob 与当前工作树字节；新增文件显式无 parent。"""
    bindings = []
    for path in _ALLOWED_SOURCE_PATHS:
        current_size, current_sha = _identity(root, path)
        kind = changes[path]
        parent_payload = b""
        if kind == "M":
            parent_payload = _git(root, "show", f"{PARENT_COMMIT}:{path}")
        bindings.append({
            "change_kind": kind,
            "current_sha256": current_sha,
            "current_size_bytes": current_size,
            "parent_sha256": _sha256(parent_payload) if parent_payload else None,
            "parent_size_bytes": len(parent_payload),
            "relative_path": path,
        })
    return bindings


def _v5_correction(root: Path) -> dict[str, Any]:
    """以 Core base 和两个 Git blob 证明 v5 的 63 字符 metadata 笔误。"""
    value = _canonical_object(
        _target(
            root,
            "data/ph2/manifests/performance_successor_receipt_v5.json",
        ).read_bytes(),
        label="performance v5",
    )
    bindings = value.get("source_bindings")
    if not isinstance(bindings, list):
        raise PW00AAuthorityError("performance v5 source bindings 非法")
    matches = tuple(
        item for item in bindings
        if isinstance(item, dict) and item.get("relative_path") == _V5_PATH)
    if (len(matches) != 1
            or matches[0].get("parent_sha256") != _V5_RECORDED_PARENT
            or matches[0].get("current_sha256") != _V5_CURRENT):
        raise PW00AAuthorityError("performance v5 metadata 不是已知单字符截断")
    parent_blob = _git(root, "show", f"{_V5_PARENT_COMMIT}:{_V5_PATH}")
    change_blob = _git(root, "show", f"{_V5_CHANGE_COMMIT}:{_V5_PATH}")
    if (_sha256(parent_blob) != _V5_CORRECT_PARENT
            or _sha256(change_blob) != _V5_CURRENT
            or _identity(root, _V5_PATH)[1] != _V5_CURRENT):
        raise PW00AAuthorityError("performance v5 Git blob correction 不闭合")
    return {
        "change_blob_sha256": _V5_CURRENT,
        "change_commit": _V5_CHANGE_COMMIT,
        "correct_parent_sha256": _V5_CORRECT_PARENT,
        "current_leaf_sha256": _V5_CURRENT,
        "parent_blob_sha256": _V5_CORRECT_PARENT,
        "parent_commit": _V5_PARENT_COMMIT,
        "recorded_parent_sha256": _V5_RECORDED_PARENT,
        "relative_path": _V5_PATH,
        "status": "METADATA_TRUNCATION_CORRECTED_BY_GIT_BLOB",
    }


def build_pw00a_formal_load_authority(
        repository_root: str | Path,
        ) -> dict[str, Any]:
    """在 clean committed HEAD 上形成唯一正式装载 authority。"""
    root = Path(repository_root).resolve()
    if _git(root, "status", "--porcelain=v1", "--", "src").strip():
        raise PW00AAuthorityError("PW00A authority 要求 src worktree clean")
    head = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    if len(head) != 40:
        raise PW00AAuthorityError("PW00A authority HEAD 非 SHA-1")
    dependencies = _validate_dependency_chain(root)
    changes = _source_changes(root, head)
    sources = _source_bindings(root, changes)
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": ARTIFACT_VERSION,
        "dependency_bindings": dependencies,
        "format_version": FORMAT_VERSION,
        "head_commit": head,
        "parent_commit": PARENT_COMMIT,
        "private_boundaries": {
            "candidate_root_reads": 0,
            "evaluator_label_reads": 0,
            "private_root_reads": 0,
            "teacher_api_calls": 0,
        },
        "readiness_transition": {
            "LANGUAGE_CAPABILITY_MASTERED": 1,
            "LANGUAGE_READINESS_REPUBLISHED": 1,
            "PW00A_STARTED": 0,
        },
        "receipt_relative_path": RECEIPT_PATH,
        "receipt_self_excluded": 1,
        "source_bindings": sources,
        "status": STATUS,
        "v5_metadata_correction": _v5_correction(root),
    }


def _source_drift_paths(value: dict[str, Any], root: Path) -> tuple[str, ...]:
    """返回 authority 所绑定 source 与当前工作树之间的全部漂移路径。"""
    return tuple(
        item["relative_path"]
        for item in value["source_bindings"]
        if _identity(root, item["relative_path"])
        != (item["current_size_bytes"], item["current_sha256"])
    )


def _validate(
        value: dict[str, Any],
        root: Path,
        verification_mode: str,
        ) -> None:
    """核验固定结构，并按模式决定是否要求当前 source 兼容。"""
    try:
        mode = require_artifact_verification_mode(verification_mode)
    except ValueError as error:
        raise PW00AAuthorityError(str(error)) from error
    if set(value) != {
            "artifact_kind", "artifact_version", "dependency_bindings",
            "format_version", "head_commit", "parent_commit",
            "private_boundaries", "readiness_transition",
            "receipt_relative_path", "receipt_self_excluded",
            "source_bindings", "status", "v5_metadata_correction"}:
        raise PW00AAuthorityError("PW00A authority 字段不精确")
    if (value["artifact_kind"] != ARTIFACT_KIND
            or value["artifact_version"] != ARTIFACT_VERSION
            or value["format_version"] != FORMAT_VERSION
            or value["parent_commit"] != PARENT_COMMIT
            or value["receipt_relative_path"] != RECEIPT_PATH
            or value["receipt_self_excluded"] != 1
            or value["status"] != STATUS):
        raise PW00AAuthorityError("PW00A authority 固定身份漂移")
    head_commit = value["head_commit"]
    if (not isinstance(head_commit, str) or len(head_commit) != 40
            or any(char not in "0123456789abcdef" for char in head_commit)):
        raise PW00AAuthorityError("PW00A authority head commit 非法")
    if value["readiness_transition"] != {
            "LANGUAGE_CAPABILITY_MASTERED": 1,
            "LANGUAGE_READINESS_REPUBLISHED": 1,
            "PW00A_STARTED": 0}:
        raise PW00AAuthorityError("PW00A authority readiness 漂移")
    if value["private_boundaries"] != {
            "candidate_root_reads": 0,
            "evaluator_label_reads": 0,
            "private_root_reads": 0,
            "teacher_api_calls": 0}:
        raise PW00AAuthorityError("PW00A authority private 边界漂移")
    dependencies = value["dependency_bindings"]
    if not isinstance(dependencies, list) or len(dependencies) != len(_DEPENDENCIES):
        raise PW00AAuthorityError("PW00A dependency 数量漂移")
    expected_dependencies = {
        path: (size, digest) for path, size, digest in _DEPENDENCIES}
    for item in dependencies:
        if not isinstance(item, dict) or set(item) != {
                "relative_path", "sha256", "size_bytes"}:
            raise PW00AAuthorityError("PW00A dependency binding 字段漂移")
        path = _relative(item["relative_path"], label="dependency")
        if path not in expected_dependencies:
            raise PW00AAuthorityError("PW00A dependency 不在 allowlist")
        size, digest = expected_dependencies[path]
        if item != {
                "relative_path": path,
                "sha256": digest,
                "size_bytes": size} or _identity(root, path) != (size, digest):
            raise PW00AAuthorityError(f"PW00A dependency leaf 漂移: {path}")
    sources = value["source_bindings"]
    if not isinstance(sources, list) or len(sources) != len(_ALLOWED_SOURCE_PATHS):
        raise PW00AAuthorityError("PW00A source 数量漂移")
    if tuple(item.get("relative_path") for item in sources) != _ALLOWED_SOURCE_PATHS:
        raise PW00AAuthorityError("PW00A source allowlist 顺序漂移")
    for item in sources:
        if not isinstance(item, dict) or set(item) != {
                "change_kind", "current_sha256", "current_size_bytes",
                "parent_sha256", "parent_size_bytes", "relative_path"}:
            raise PW00AAuthorityError("PW00A source binding 字段漂移")
        path = _relative(item["relative_path"], label="source")
        _strict_sha(item["current_sha256"], label=path)
        if (type(item["current_size_bytes"]) is not int
                or item["current_size_bytes"] <= 0):
            raise PW00AAuthorityError("PW00A source current size 非法")
        if item["change_kind"] == "A":
            if item["parent_sha256"] is not None or item["parent_size_bytes"] != 0:
                raise PW00AAuthorityError("PW00A added source parent 非空")
        elif item["change_kind"] == "M":
            _strict_sha(item["parent_sha256"], label=f"{path} parent")
            if type(item["parent_size_bytes"]) is not int or item["parent_size_bytes"] <= 0:
                raise PW00AAuthorityError("PW00A modified source parent size 非法")
        else:
            raise PW00AAuthorityError("PW00A source change kind 非法")
    if mode == CURRENT_HEAD_COMPATIBILITY_VERIFY:
        drift_paths = _source_drift_paths(value, root)
        if drift_paths:
            raise PW00AAuthorityError(
                "PW00A source leaf 漂移: " + ", ".join(drift_paths))
    correction = value["v5_metadata_correction"]
    expected_correction = {
        "change_blob_sha256": _V5_CURRENT,
        "change_commit": _V5_CHANGE_COMMIT,
        "correct_parent_sha256": _V5_CORRECT_PARENT,
        "current_leaf_sha256": _V5_CURRENT,
        "parent_blob_sha256": _V5_CORRECT_PARENT,
        "parent_commit": _V5_PARENT_COMMIT,
        "recorded_parent_sha256": _V5_RECORDED_PARENT,
        "relative_path": _V5_PATH,
        "status": "METADATA_TRUNCATION_CORRECTED_BY_GIT_BLOB",
    }
    if correction != expected_correction:
        raise PW00AAuthorityError("PW00A v5 metadata correction 漂移")
    if (mode == CURRENT_HEAD_COMPATIBILITY_VERIFY
            and _identity(root, _V5_PATH)[1] != _V5_CURRENT):
        raise PW00AAuthorityError("PW00A v5 current leaf 漂移")


def read_pw00a_formal_load_authority(
        repository_root: str | Path,
        path: str | Path = RECEIPT_PATH,
        *,
        verification_mode: str = CURRENT_HEAD_COMPATIBILITY_VERIFY,
        ) -> dict[str, Any]:
    """不调用 Git，按显式历史或当前语义回读 authority。"""
    root = Path(repository_root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / Path(*str(target).replace("\\", "/").split("/"))
    payload = target.read_bytes()
    value = _canonical_object(payload, label="PW00A authority")
    _validate(value, root, verification_mode)
    return value


def audit_pw00a_current_source_drift(
        repository_root: str | Path,
        path: str | Path = RECEIPT_PATH,
        ) -> tuple[str, ...]:
    """在 archive 身份成立后，结构化列出当前 HEAD 的 source 漂移。"""
    root = Path(repository_root).resolve()
    value = read_pw00a_formal_load_authority(
        root,
        path,
        verification_mode=ARCHIVE_IDENTITY_VERIFY,
    )
    return _source_drift_paths(value, root)


def publish_pw00a_formal_load_authority(
        repository_root: str | Path,
        *,
        target: str | Path = RECEIPT_PATH,
        ) -> dict[str, Any]:
    """排他发布 authority；任何既有目标均拒绝覆盖。"""
    root = Path(repository_root).resolve()
    destination = Path(target)
    if not destination.is_absolute():
        destination = root / Path(*str(destination).replace("\\", "/").split("/"))
    if destination.exists():
        raise PW00AAuthorityError("PW00A authority 已存在，禁止覆盖")
    value = build_pw00a_formal_load_authority(root)
    payload = canonical_json_bytes(value) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise PW00AAuthorityError("PW00A authority 已存在，禁止覆盖") from error
    restored = read_pw00a_formal_load_authority(root, destination)
    if restored != value:
        raise PW00AAuthorityError("PW00A authority 发布回读漂移")
    return restored


__all__ = [
    "ARTIFACT_KIND",
    "ARTIFACT_VERSION",
    "FORMAT_VERSION",
    "PARENT_COMMIT",
    "RECEIPT_PATH",
    "STATUS",
    "PW00AAuthorityError",
    "ARCHIVE_IDENTITY_VERIFY",
    "CURRENT_HEAD_COMPATIBILITY_VERIFY",
    "audit_pw00a_current_source_drift",
    "build_pw00a_formal_load_authority",
    "publish_pw00a_formal_load_authority",
    "read_pw00a_formal_load_authority",
]
