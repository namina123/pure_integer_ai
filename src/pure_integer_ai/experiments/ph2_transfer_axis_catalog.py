"""从当前正式 D-02 pack 构建 LC-09 跨轴迁移只读账。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    ObservationRecord,
    SourceRefRecord,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import read_artifact_manifest
from pure_integer_ai.experiments.ph2_source_pack_catalog import (
    SOURCE_PACK_COVERAGE_PATH,
)
from pure_integer_ai.experiments.ph2_source_pack_compiler import (
    read_source_pack_coverage_manifest,
    read_source_pack_view,
)
from pure_integer_ai.experiments.ph2_transfer_axis_contract import (
    ARTIFACT_STATUS,
    EXECUTION_STATE,
    FORMAT_VERSION,
    RUNTIME_STATUS,
    SCOPE_CONTRACTION_PROTOCOLS,
    TRANSFER_AXIS_KEYS,
    VERIFIER_DIMENSIONS,
    VERIFIER_NE_CONDITIONS,
    BlockedTransferSource,
    LanguageTransferAxisManifest,
    TransferAxisContractError,
    TransferPackAudit,
    TransferSplitProbe,
    evaluate_transfer_split_probe,
)


LC09_MANIFEST_PATH = Path(
    "data/ph2/manifests/lc09_transfer_axis_manifest_v1.json")
LC09_ARTIFACT_VERSION = "LC-09-transfer-axis-manifest-v1"
FORMAL_ARTIFACT_ROOT = "ph2_dataset_artifacts"
_COURSE_ARTIFACT_KINDS = {
    "PH2_CAPABILITY_COURSE_FREEZE",
    "PH2_LANGUAGE_COURSE_FREEZE",
}
_SOURCE_AXIS_MAP = {
    "code_switch": "CODE_SWITCH",
    "dialect": "DIALECT",
    "domain": "DOMAIN",
    "era": "ERA",
    "genre": "GENRE",
    "language": "LANGUAGE",
    "length": "LENGTH",
    "register": "REGISTER",
    "script_orthography": "SCRIPT",
    "source": "SOURCE",
}
_DECLARED_NE_VALUES = {"UNASSESSED", "UNDECLARED", "UNKNOWN"}


class TransferAxisCatalogError(RuntimeError):
    """正式 pack inventory、hash、owner 视图或标准轴发生漂移。"""


def _sha256_path(path: Path) -> str:
    """以固定块大小计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _read_canonical_object(path: Path) -> dict[str, Any]:
    """回读带单一换行的规范 JSON object。"""
    payload = path.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise TransferAxisCatalogError("repository manifest newline 非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except ValueError as error:
        raise TransferAxisCatalogError("repository manifest JSON 损坏") from error
    if not isinstance(value, dict):
        raise TransferAxisCatalogError("repository manifest 非 object")
    return value


def _repository_pack_references(
        repository_root: Path,
        ) -> tuple[
            dict[str, tuple[str, tuple[str, ...]]],
            tuple[BlockedTransferSource, ...],
        ]:
    """汇合来源覆盖账与课程 manifest 中的正式 pack 引用。"""
    coverage_path = repository_root / SOURCE_PACK_COVERAGE_PATH
    coverage = read_source_pack_coverage_manifest(coverage_path)
    references: dict[str, tuple[str, tuple[str, ...]]] = {}
    blocked: list[BlockedTransferSource] = []
    for entry in coverage.entries:
        if entry.status == "BLOCKED":
            blocked.append(BlockedTransferSource(
                entry.source_key,
                entry.blocker_code,
                tuple(sorted(entry.evidence_refs)),
            ))
            continue
        references[entry.pack_manifest_relative_path] = (
            entry.pack_manifest_sha256,
            tuple(sorted({
                SOURCE_PACK_COVERAGE_PATH.as_posix(),
                *entry.evidence_refs,
                entry.pack_manifest_relative_path,
            })),
        )
    manifest_root = repository_root / "data" / "ph2" / "manifests"
    for path in sorted(manifest_root.glob("*.json")):
        value = _read_canonical_object(path)
        if value.get("artifact_kind") not in _COURSE_ARTIFACT_KINDS:
            continue
        relative_path = value.get("pack_manifest_relative_path")
        digest = value.get("pack_manifest_sha256")
        if not isinstance(relative_path, str) or not isinstance(digest, str):
            raise TransferAxisCatalogError("course manifest 缺 pack 引用")
        evidence = tuple(sorted({
            path.relative_to(repository_root).as_posix(),
            relative_path,
        }))
        prior = references.get(relative_path)
        if prior is not None and prior != (digest, evidence):
            raise TransferAxisCatalogError("pack 引用 hash/evidence 冲突")
        references[relative_path] = (digest, evidence)
    return references, tuple(sorted(blocked, key=lambda item: item.source_key))


def _formal_pack_inventory(workspace_root: Path) -> tuple[str, ...]:
    """枚举工作区当前全部正式 pack manifest 相对路径。"""
    root = workspace_root / FORMAL_ARTIFACT_ROOT
    if not root.is_dir():
        raise TransferAxisCatalogError("正式 artifact root 缺失")
    paths = tuple(sorted(
        path.relative_to(workspace_root).as_posix()
        for path in root.rglob("manifest.json")
        if path.is_file()
    ))
    if not paths:
        raise TransferAxisCatalogError("正式 pack inventory 为空")
    return paths


def _observation_axis_row(
        observation: ObservationRecord,
        source: SourceRefRecord,
        ) -> dict[str, str]:
    """从共用 record 和标准来源 payload 提取可直接证明的十轴值。"""
    row = {"SOURCE": source.source_key}
    if observation.payload_kind != "RAW_SOURCE_OBSERVATION_V1":
        row["LANGUAGE"] = observation.language
        return row
    payload = observation.typed_payload.to_value()
    raw_axes = payload.get("combination_axes")
    if not isinstance(raw_axes, dict):
        raise TransferAxisCatalogError("source pack 缺 combination_axes")
    for raw_key, axis in _SOURCE_AXIS_MAP.items():
        value = raw_axes.get(raw_key)
        if not isinstance(value, str) or not value or value.strip() != value:
            raise TransferAxisCatalogError("source pack 标准轴缺失或非法")
        if axis == "SOURCE" and value != source.source_key:
            raise TransferAxisCatalogError("record 与 combination axis 漂移")
        row[axis] = value
    return row


def _axis_state(values: tuple[str, ...]) -> str:
    """按值数量与显式未知标记归类 pack 轴声明状态。"""
    if not values:
        return "UNDECLARED"
    if all(value in _DECLARED_NE_VALUES for value in values):
        return "DECLARED_NE"
    if len(values) == 1:
        return "BASELINE_ONLY"
    return "VARIATION_OBSERVED"


def _pack_audit(
        workspace_root: Path,
        relative_path: str,
        expected_sha256: str,
        evidence_refs: tuple[str, ...],
        ) -> TransferPackAudit:
    """逐 owner 回读一个 pack，冻结直接可证的十轴与组合 split 事实。"""
    manifest_path = workspace_root / Path(*relative_path.split("/"))
    if (not manifest_path.is_file()
            or _sha256_path(manifest_path) != expected_sha256):
        raise TransferAxisCatalogError("正式 pack manifest hash 漂移")
    pack_root = manifest_path.parent
    manifest = read_artifact_manifest(manifest_path)
    if manifest.sha256() != expected_sha256:
        raise TransferAxisCatalogError("正式 pack manifest 规范 hash 漂移")
    sources = read_source_pack_view(
        pack_root, reader_kind="source_audit")
    if any(not isinstance(item, SourceRefRecord) for item in sources):
        raise TransferAxisCatalogError("source_audit view 类型漂移")
    source_by_key = {item.stable_key: item for item in sources}
    observations: list[ObservationRecord] = []
    for split in manifest.splits:
        records = read_source_pack_view(
            pack_root, reader_kind="student", split=split)
        if any(not isinstance(item, ObservationRecord) for item in records):
            raise TransferAxisCatalogError("student view 类型漂移")
        observations.extend(records)
    if not observations:
        raise TransferAxisCatalogError("正式 pack Observation 为空")
    rows: list[tuple[str, dict[str, str]]] = []
    for observation in observations:
        source = source_by_key.get(observation.source_ref_key)
        if source is None:
            raise TransferAxisCatalogError("Observation source_ref 无法恢复")
        rows.append((
            observation.split,
            _observation_axis_row(observation, source),
        ))
    axis_values = {
        axis: tuple(sorted({row[axis] for _, row in rows if axis in row}))
        for axis in TRANSFER_AXIS_KEYS
    }
    held_out_axis_values = {
        axis: tuple(sorted({
            row[axis] for split, row in rows
            if split == "held_out" and axis in row
        }))
        for axis in TRANSFER_AXIS_KEYS
    }
    states = {
        axis: _axis_state(axis_values[axis]) for axis in TRANSFER_AXIS_KEYS
    }
    complete = all(
        all(axis in row for axis in TRANSFER_AXIS_KEYS)
        for _, row in rows
    )
    train_combinations = {
        tuple(row[axis] for axis in TRANSFER_AXIS_KEYS)
        for split, row in rows
        if split == "train" and all(axis in row for axis in TRANSFER_AXIS_KEYS)
    }
    held_out_combinations = {
        tuple(row[axis] for axis in TRANSFER_AXIS_KEYS)
        for split, row in rows
        if split == "held_out" and all(axis in row for axis in TRANSFER_AXIS_KEYS)
    }
    if not complete:
        combination_state = "NE_AXIS_UNDECLARED"
    elif not train_combinations or not held_out_combinations:
        combination_state = "NE_SINGLE_SPLIT"
    elif train_combinations & held_out_combinations:
        combination_state = "NE_COMBINATION_OVERLAP"
    else:
        combination_state = "FROZEN"
    pack_kind = (
        "SOURCE_PACK"
        if all(item.payload_kind == "RAW_SOURCE_OBSERVATION_V1"
               for item in observations)
        else "AUTHORED_COURSE"
    )
    return TransferPackAudit(
        relative_path,
        expected_sha256,
        pack_kind,
        tuple(sorted({item.source_key for item in sources})),
        tuple(sorted({item.license_id for item in sources})),
        tuple(sorted(manifest.splits)),
        CanonicalJsonObject.from_value({
            axis: list(axis_values[axis]) for axis in TRANSFER_AXIS_KEYS
        }),
        CanonicalJsonObject.from_value({
            axis: list(held_out_axis_values[axis])
            for axis in TRANSFER_AXIS_KEYS
        }),
        CanonicalJsonObject.from_value(states),
        combination_state,
        len(train_combinations),
        len(held_out_combinations),
        "NE",
        evidence_refs,
    )


def _combination(prefix: str) -> dict[str, str]:
    """构造列全十轴的基线组合。"""
    return {axis: f"{prefix}_{axis}_A" for axis in TRANSFER_AXIS_KEYS}


def _probe(
        key: str,
        kind: str,
        isolated_axes: tuple[str, ...],
        train: tuple[dict[str, str], ...],
        held_out: tuple[dict[str, str], ...],
        ) -> TransferSplitProbe:
    """执行并冻结一个零写 split fixture 的直接 verdict。"""
    train_objects = tuple(
        CanonicalJsonObject.from_value(item) for item in train)
    held_objects = tuple(
        CanonicalJsonObject.from_value(item) for item in held_out)
    verdict, failure = evaluate_transfer_split_probe(
        kind, isolated_axes, train_objects, held_objects)
    return TransferSplitProbe(
        key, kind, isolated_axes, train_objects, held_objects,
        verdict, failure, 0)


def build_transfer_split_probes() -> tuple[TransferSplitProbe, ...]:
    """构建单轴、双轴和完整组合三类正交 held-out fixture。"""
    single_train = _combination("SINGLE")
    single_held = dict(single_train)
    single_held["DOMAIN"] = "SINGLE_DOMAIN_B"

    double_base = _combination("DOUBLE")
    double_train: list[dict[str, str]] = []
    for register, genre in (("R1", "G1"), ("R2", "G1"), ("R1", "G2")):
        item = dict(double_base)
        item["REGISTER"] = register
        item["GENRE"] = genre
        double_train.append(item)
    double_held = dict(double_base)
    double_held["REGISTER"] = "R2"
    double_held["GENRE"] = "G2"

    full_base = _combination("FULL")
    full_train = [dict(full_base)]
    for axis in TRANSFER_AXIS_KEYS:
        item = dict(full_base)
        item[axis] = f"FULL_{axis}_B"
        full_train.append(item)
    full_held = {
        axis: f"FULL_{axis}_B" for axis in TRANSFER_AXIS_KEYS
    }
    return tuple(sorted((
        _probe(
            "LC09_DOUBLE_AXIS_REGISTER_GENRE_V1",
            "DOUBLE_AXIS",
            ("GENRE", "REGISTER"),
            tuple(double_train),
            (double_held,),
        ),
        _probe(
            "LC09_FULL_COMBINATION_V1",
            "FULL_COMBINATION",
            TRANSFER_AXIS_KEYS,
            tuple(full_train),
            (full_held,),
        ),
        _probe(
            "LC09_SINGLE_AXIS_DOMAIN_V1",
            "SINGLE_AXIS",
            ("DOMAIN",),
            (single_train,),
            (single_held,),
        ),
    ), key=lambda item: item.probe_kind))


def build_repository_transfer_axis_manifest(
        repository_root: str | Path,
        workspace_root: str | Path,
        ) -> LanguageTransferAxisManifest:
    """枚举全部正式 pack 并构建 LC-09 十轴、NE 与 split probe 账。"""
    repository = Path(repository_root).resolve()
    workspace = Path(workspace_root).resolve()
    if (not (repository / "src" / "pure_integer_ai").is_dir()
            or repository.parent != workspace):
        raise TransferAxisCatalogError("repository/workspace 边界非法")
    references, blocked = _repository_pack_references(repository)
    actual = _formal_pack_inventory(workspace)
    if tuple(sorted(references)) != actual:
        missing = tuple(sorted(set(actual) - set(references)))
        stale = tuple(sorted(set(references) - set(actual)))
        raise TransferAxisCatalogError(
            f"正式 pack inventory 未闭合 missing={missing} stale={stale}")
    audits = tuple(
        _pack_audit(workspace, path, references[path][0], references[path][1])
        for path in actual
    )
    try:
        return LanguageTransferAxisManifest(
            FORMAT_VERSION,
            LC09_ARTIFACT_VERSION,
            ARTIFACT_STATUS,
            RUNTIME_STATUS,
            "LC-09",
            TRANSFER_AXIS_KEYS,
            len(audits),
            audits,
            blocked,
            build_transfer_split_probes(),
            VERIFIER_DIMENSIONS,
            VERIFIER_NE_CONDITIONS,
            SCOPE_CONTRACTION_PROTOCOLS,
            0,
            CanonicalJsonObject.from_value(EXECUTION_STATE),
        )
    except TransferAxisContractError as error:
        raise TransferAxisCatalogError("LC-09 manifest 构建失败") from error


__all__ = [
    "FORMAL_ARTIFACT_ROOT",
    "LC09_ARTIFACT_VERSION",
    "LC09_MANIFEST_PATH",
    "TransferAxisCatalogError",
    "build_repository_transfer_axis_manifest",
    "build_transfer_split_probes",
]
