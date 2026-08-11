"""FT26 compact public sense artifact 的 fail-closed 加载与只读查询。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sysconfig

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_contract import (
    W03_PUBLIC_SENSE_ARTIFACT_VERSION,
    W03_PUBLIC_SENSE_FORMAT,
    W03_PUBLIC_SENSE_MAX_ARTIFACT_BYTES,
    W03_PUBLIC_SENSE_SCHEMA_VERSION,
    W03PublicSenseArtifact,
    W03PublicSenseCandidate,
    W03PublicSenseContractError,
    W03PublicSenseQuery,
    W03PublicSenseQueryResult,
)


REPOSITORY = Path(__file__).resolve().parents[3]
PUBLIC_W03_SENSE_ARTIFACT_RELATIVE_PATH = Path(
    "data/ph2/w03_public_sense_runtime_v1.json")
PUBLIC_W03_SENSE_ARTIFACT = (
    REPOSITORY / PUBLIC_W03_SENSE_ARTIFACT_RELATIVE_PATH)
PUBLIC_W03_SENSE_DISTRIBUTION_SUBDIRECTORY = Path("share/pure_integer_ai")
PUBLIC_W03_SENSE_ARTIFACT_SHA256 = (
    "7e0e1ae1b4c7bb334d9581c887f880949c5a43c64ca68aad4a9e05a6206e3792")


# object-model: exception
class W03PublicSenseRuntimeError(RuntimeError):
    """compact artifact 缺失、损坏、非规范或查询索引不一致。"""


def _exact(
        value: object,
        keys: tuple[str, ...],
        *,
        where: str,
        ) -> dict[str, object]:
    """要求 JSON object 字段集合精确。"""
    if not isinstance(value, dict) or set(value) != set(keys):
        raise W03PublicSenseRuntimeError(f"{where} 字段集合漂移")
    return value


def _installed_resource_roots() -> tuple[Path, ...]:
    """枚举当前与 user install 的标准 data scheme 根。"""
    data_roots = []
    current = sysconfig.get_path("data")
    if current:
        data_roots.append(Path(current))
    for scheme in sysconfig.get_scheme_names():
        if not scheme.endswith("_user"):
            continue
        try:
            value = sysconfig.get_path("data", scheme=scheme)
        except (KeyError, TypeError, ValueError):
            continue
        if value:
            data_roots.append(Path(value))
    values = []
    seen = set()
    for root in data_roots:
        candidate = (root / PUBLIC_W03_SENSE_DISTRIBUTION_SUBDIRECTORY).resolve()
        if candidate not in seen:
            seen.add(candidate)
            values.append(candidate)
    return tuple(values)


def _candidate_artifact_paths() -> tuple[Path, ...]:
    """返回 source checkout 与安装 data scheme 中的 artifact 候选。"""
    return tuple(
        root / PUBLIC_W03_SENSE_ARTIFACT_RELATIVE_PATH
        for root in (REPOSITORY, *_installed_resource_roots())
    )


def _language_matches(candidate: str, requested: str) -> bool:
    """允许 base language 查询命中其显式变体。"""
    return candidate == requested or candidate.startswith(requested + "-")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03PublicSenseRuntime:
    """不可变 artifact、查询索引与完整字节身份。"""

    artifact: W03PublicSenseArtifact
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, W03PublicSenseArtifact):
            raise TypeError("public sense runtime artifact 类型非法")
        if (not isinstance(self.artifact_sha256, str)
                or len(self.artifact_sha256) != 64):
            raise W03PublicSenseRuntimeError(
                "public sense runtime artifact SHA 非法")
        alias_targets: dict[tuple[str, str], str] = {}
        for alias in self.artifact.aliases:
            key = alias.language, alias.alias_surface
            prior = alias_targets.get(key)
            if prior is not None and prior != alias.target_surface:
                raise W03PublicSenseRuntimeError(
                    "public sense alias 存在多个 target")
            alias_targets[key] = alias.target_surface
        for key in alias_targets:
            seen = set()
            current = key
            while current in alias_targets:
                if current in seen:
                    raise W03PublicSenseRuntimeError(
                        "public sense alias chain 成环")
                seen.add(current)
                current = current[0], alias_targets[current]


def load_w03_public_sense_artifact(
        path: str | Path | None = None,
        ) -> W03PublicSenseRuntime:
    """严格恢复 canonical artifact，缺失或任一 commitment 漂移即拒绝。"""
    if path is None:
        source = next(
            (item for item in _candidate_artifact_paths() if item.is_file()),
            PUBLIC_W03_SENSE_ARTIFACT,
        )
    else:
        source = Path(path).resolve()
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise W03PublicSenseRuntimeError(
            "public sense artifact 缺失或不可读") from error
    if (not raw.endswith(b"\n")
            or not 0 < len(raw) <= W03_PUBLIC_SENSE_MAX_ARTIFACT_BYTES):
        raise W03PublicSenseRuntimeError(
            "public sense artifact 大小或终止换行非法")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != PUBLIC_W03_SENSE_ARTIFACT_SHA256:
        raise W03PublicSenseRuntimeError(
            "public sense artifact canonical SHA 漂移")
    try:
        value = json.loads(raw[:-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise W03PublicSenseRuntimeError(
            "public sense artifact JSON 非法") from error
    if canonical_json_bytes(value) + b"\n" != raw:
        raise W03PublicSenseRuntimeError(
            "public sense artifact 不是规范 JSON 字节")
    envelope = _exact(value, (
        "artifact_version", "experimental", "formal_mastery_claim", "format",
        "mastery", "payload", "payload_sha256", "readiness",
        "schema_version", "w02_runtime_evidenced", "w03_started",
    ), where="public sense envelope")
    if (envelope["format"] != W03_PUBLIC_SENSE_FORMAT
            or envelope["schema_version"] != W03_PUBLIC_SENSE_SCHEMA_VERSION
            or envelope["artifact_version"] != W03_PUBLIC_SENSE_ARTIFACT_VERSION
            or (
                envelope["experimental"],
                envelope["formal_mastery_claim"],
                envelope["w02_runtime_evidenced"],
                envelope["w03_started"],
                envelope["mastery"],
                envelope["readiness"],
            ) != (1, 0, 0, 0, 0, 0)):
        raise W03PublicSenseRuntimeError(
            "public sense experimental/formal boundary 漂移")
    payload = envelope["payload"]
    payload_sha256 = envelope["payload_sha256"]
    if (not isinstance(payload_sha256, str)
            or hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
            != payload_sha256):
        raise W03PublicSenseRuntimeError(
            "public sense payload commitment 漂移")
    try:
        artifact = W03PublicSenseArtifact.from_payload_value(payload)
    except W03PublicSenseContractError as error:
        raise W03PublicSenseRuntimeError(
            "public sense payload 合同非法") from error
    if artifact.payload_sha256() != payload_sha256:
        raise W03PublicSenseRuntimeError(
            "public sense payload roundtrip 漂移")
    return W03PublicSenseRuntime(artifact, actual_sha256)


def _matched_entries(
        runtime: W03PublicSenseRuntime,
        query: W03PublicSenseQuery,
        ) -> tuple[tuple[object, ...], tuple[str, ...]]:
    """读取 direct 候选并沿唯一 redirect chain 累积 target 候选。"""
    aliases = {
        (item.language, item.alias_surface): item.target_surface
        for item in runtime.artifact.aliases
    }
    surfaces = [query.surface]
    current = query.surface
    for _ in range(16):
        targets = {
            target
            for (language, alias), target in aliases.items()
            if alias == current and _language_matches(language, query.language)
        }
        if not targets:
            break
        if len(targets) != 1:
            raise W03PublicSenseRuntimeError(
                "public sense query alias target 非唯一")
        target = next(iter(targets))
        if target in surfaces:
            raise W03PublicSenseRuntimeError(
                "public sense query alias chain 成环")
        surfaces.append(target)
        current = target
    else:
        raise W03PublicSenseRuntimeError(
            "public sense query alias chain 超预算")
    entries = tuple(sorted(
        (
            item for item in runtime.artifact.entries
            if item.active == 1
            and item.surface in surfaces
            and _language_matches(item.language, query.language)
        ),
        key=lambda item: item.entry_key,
    ))
    deduplicated = {}
    for item in entries:
        key = (
            item.surface,
            item.canonical_surface,
            item.relation_kind,
            item.definition_text,
            item.concept_key,
            item.source_ref.stable_key,
        )
        prior = deduplicated.get(key)
        if prior is None:
            deduplicated[key] = item
            continue
        prior_rank = (
            int(prior.language != query.language),
            prior.language,
            prior.entry_key,
        )
        current_rank = (
            int(item.language != query.language),
            item.language,
            item.entry_key,
        )
        if current_rank < prior_rank:
            deduplicated[key] = item
    return tuple(sorted(
        deduplicated.values(), key=lambda item: item.entry_key)), tuple(surfaces)


def query_w03_public_sense(
        runtime: W03PublicSenseRuntime,
        query: W03PublicSenseQuery,
        ) -> W03PublicSenseQueryResult:
    """按 exact term/context 返回 typed candidate、SourceRef 与 trace commitment。"""
    if (not isinstance(runtime, W03PublicSenseRuntime)
            or not isinstance(query, W03PublicSenseQuery)):
        raise TypeError("public sense query 输入类型非法")
    visible, alias_path = _matched_entries(runtime, query)
    if query.context_text is not None:
        exact = tuple(
            item for item in visible
            if item.definition_text == query.context_text)
        if exact:
            visible = exact
        elif len({item.concept_key for item in visible}) <= 1:
            visible = ()
    concepts = {item.concept_key for item in visible}
    sources = {item.source_ref.source_key for item in visible}
    conflict_kind = None
    if not visible:
        status = "UNKNOWN"
    elif query.context_text is not None and not any(
            item.definition_text == query.context_text for item in visible):
        status = "CLARIFY"
    elif len(concepts) == 1:
        status = "UNIQUE"
    elif len(sources) > 1:
        status = "CONFLICT"
        conflict_kind = "UNRESOLVED_SOURCE_PARTITION"
    else:
        status = "AMBIGUOUS"
    candidates = tuple(
        W03PublicSenseCandidate(item, item.surface) for item in visible)
    trace_value = {
        "alias_path": list(alias_path),
        "artifact_sha256": runtime.artifact_sha256,
        "candidate_entry_keys": [
            list(item.entry.entry_key) for item in candidates],
        "conflict_kind": conflict_kind,
        "query": query.to_dict(),
        "status": status,
    }
    trace = hashlib.sha256(canonical_json_bytes(trace_value)).hexdigest()
    return W03PublicSenseQueryResult(
        query,
        status,
        candidates,
        alias_path,
        conflict_kind,
        runtime.artifact_sha256,
        trace,
    )


def query_w03_public_senses(
        runtime: W03PublicSenseRuntime,
        queries: tuple[W03PublicSenseQuery, ...],
        ) -> tuple[W03PublicSenseQueryResult, ...]:
    """复用同一 immutable runtime 查询一个有界 term batch。"""
    if (not isinstance(queries, tuple) or not queries
            or any(not isinstance(item, W03PublicSenseQuery)
                   for item in queries)):
        raise TypeError("public sense query batch 非法")
    return tuple(query_w03_public_sense(runtime, item) for item in queries)


__all__ = [
    "PUBLIC_W03_SENSE_ARTIFACT",
    "PUBLIC_W03_SENSE_ARTIFACT_RELATIVE_PATH",
    "PUBLIC_W03_SENSE_ARTIFACT_SHA256",
    "PUBLIC_W03_SENSE_DISTRIBUTION_SUBDIRECTORY",
    "W03PublicSenseRuntime",
    "W03PublicSenseRuntimeError",
    "load_w03_public_sense_artifact",
    "query_w03_public_sense",
    "query_w03_public_senses",
]
