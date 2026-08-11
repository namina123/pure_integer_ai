"""FT26 public source sense artifact 与查询结果的纯结构合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)


W03_PUBLIC_SENSE_FORMAT = "PURE_INTEGER_AI_W03_PUBLIC_SENSE_RUNTIME"
W03_PUBLIC_SENSE_SCHEMA_VERSION = 1
W03_PUBLIC_SENSE_ARTIFACT_VERSION = 1
W03_PUBLIC_SENSE_MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
W03_PUBLIC_SENSE_STATUSES = (
    "UNIQUE",
    "AMBIGUOUS",
    "CLARIFY",
    "UNKNOWN",
    "CONFLICT",
)
W03_PUBLIC_SENSE_RELATIONS = (
    "ALIAS",
    "DEFINITION",
    "LABEL",
)


# object-model: exception
class W03PublicSenseContractError(ValueError):
    """artifact、SourceRef、候选或查询投影不满足冻结合同。"""


def _text(value: object, *, where: str, allow_empty: bool = False) -> str:
    """要求字符串无首尾空白，并按调用边界控制空值。"""
    if not isinstance(value, str) or value.strip() != value:
        raise W03PublicSenseContractError(f"{where} 必须是无首尾空白文本")
    if not value and not allow_empty:
        raise W03PublicSenseContractError(f"{where} 不能为空")
    return value


def _integer(value: object, *, where: str) -> int:
    """要求严格整数，拒绝 bool 与可转换字符串。"""
    if type(value) is not int:
        raise W03PublicSenseContractError(f"{where} 必须是严格整数")
    return value


def _sha256(value: object, *, where: str) -> str:
    """要求小写十六进制 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise W03PublicSenseContractError(f"{where} 不是规范 SHA-256")
    return value


def _key(value: object, *, where: str) -> tuple[int, ...]:
    """要求非空严格整数 key，拒绝 bool。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W03PublicSenseContractError(f"{where} 不是严格整数 key")
    return value


def _key_from_value(value: object, *, where: str) -> tuple[int, ...]:
    """从 JSON list 恢复严格整数 key。"""
    if not isinstance(value, list):
        raise W03PublicSenseContractError(f"{where} 不是 key list")
    return _key(tuple(value), where=where)


def _exact(
        value: object,
        keys: tuple[str, ...],
        *,
        where: str,
        ) -> dict[str, Any]:
    """要求 JSON object 字段集合精确。"""
    if not isinstance(value, dict) or set(value) != set(keys):
        raise W03PublicSenseContractError(f"{where} 字段集合漂移")
    return value


def _sha_value(value: object) -> str:
    """返回规范 JSON 值 SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03PublicSenseSourceRef:
    """compact artifact 内可公开回溯的 SourceRef 投影。"""

    stable_key: tuple[int, ...]
    source_key: str
    snapshot_id: str
    revision_id: str
    source_identity: str
    official_url: str
    license_id: str
    attribution: str
    source_commitment_sha256: str

    def __post_init__(self) -> None:
        _key(self.stable_key, where="SourceRef stable_key")
        for name in (
                "source_key", "snapshot_id", "revision_id",
                "source_identity", "official_url", "license_id",
                "attribution"):
            _text(getattr(self, name), where=f"SourceRef {name}")
        _sha256(
            self.source_commitment_sha256,
            where="SourceRef source commitment",
        )

    def to_dict(self) -> dict[str, object]:
        """导出规范 SourceRef JSON 值。"""
        return {
            "attribution": self.attribution,
            "license_id": self.license_id,
            "official_url": self.official_url,
            "revision_id": self.revision_id,
            "snapshot_id": self.snapshot_id,
            "source_commitment_sha256": self.source_commitment_sha256,
            "source_identity": self.source_identity,
            "source_key": self.source_key,
            "stable_key": list(self.stable_key),
        }

    @classmethod
    def from_dict(cls, value: object) -> "W03PublicSenseSourceRef":
        """从字段精确的 JSON object 恢复 SourceRef。"""
        raw = _exact(value, (
            "attribution", "license_id", "official_url", "revision_id",
            "snapshot_id", "source_commitment_sha256", "source_identity",
            "source_key", "stable_key",
        ), where="SourceRef")
        return cls(
            _key_from_value(raw["stable_key"], where="SourceRef stable_key"),
            _text(raw["source_key"], where="SourceRef source_key"),
            _text(raw["snapshot_id"], where="SourceRef snapshot_id"),
            _text(raw["revision_id"], where="SourceRef revision_id"),
            _text(raw["source_identity"], where="SourceRef source_identity"),
            _text(raw["official_url"], where="SourceRef official_url"),
            _text(raw["license_id"], where="SourceRef license_id"),
            _text(raw["attribution"], where="SourceRef attribution"),
            _sha256(
                raw["source_commitment_sha256"],
                where="SourceRef source commitment",
            ),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03PublicSenseEntry:
    """一个由公开 Observation 提取的词义、label 或 alias 候选。"""

    entry_key: tuple[int, ...]
    surface: str
    canonical_surface: str
    language: str
    relation_kind: str
    definition_text: str | None
    sense_key: tuple[int, ...]
    concept_key: tuple[int, ...]
    observation_key: tuple[int, ...]
    source_ref: W03PublicSenseSourceRef
    field_roles: tuple[str, ...]
    active: int
    supersedes_entry_keys: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        for name in ("entry_key", "sense_key", "concept_key", "observation_key"):
            _key(getattr(self, name), where=f"sense entry {name}")
        for name in ("surface", "canonical_surface", "language"):
            _text(getattr(self, name), where=f"sense entry {name}")
        if self.relation_kind not in W03_PUBLIC_SENSE_RELATIONS:
            raise W03PublicSenseContractError("sense entry relation_kind 漂移")
        if self.definition_text is not None:
            _text(self.definition_text, where="sense entry definition")
        if not isinstance(self.source_ref, W03PublicSenseSourceRef):
            raise TypeError("sense entry source_ref 类型非法")
        if (not isinstance(self.field_roles, tuple)
                or any(not isinstance(item, str) or not item
                       for item in self.field_roles)
                or tuple(sorted(set(self.field_roles))) != self.field_roles):
            raise W03PublicSenseContractError("sense entry field_roles 非规范")
        if self.active not in {0, 1}:
            raise W03PublicSenseContractError("sense entry active 非二值")
        if (not isinstance(self.supersedes_entry_keys, tuple)
                or any(not isinstance(item, tuple)
                       for item in self.supersedes_entry_keys)):
            raise W03PublicSenseContractError(
                "sense entry supersedes_entry_keys 类型非法")
        for item in self.supersedes_entry_keys:
            _key(item, where="sense entry superseded key")
        if tuple(sorted(set(self.supersedes_entry_keys))) != (
                self.supersedes_entry_keys):
            raise W03PublicSenseContractError(
                "sense entry supersedes_entry_keys 非规范")

    def to_dict(self) -> dict[str, object]:
        """导出规范候选 JSON 值。"""
        return {
            "active": self.active,
            "canonical_surface": self.canonical_surface,
            "concept_key": list(self.concept_key),
            "definition_text": self.definition_text,
            "entry_key": list(self.entry_key),
            "field_roles": list(self.field_roles),
            "language": self.language,
            "observation_key": list(self.observation_key),
            "relation_kind": self.relation_kind,
            "sense_key": list(self.sense_key),
            "source_ref": self.source_ref.to_dict(),
            "supersedes_entry_keys": [
                list(item) for item in self.supersedes_entry_keys],
            "surface": self.surface,
        }

    @classmethod
    def from_dict(cls, value: object) -> "W03PublicSenseEntry":
        """从字段精确的 JSON object 恢复候选。"""
        raw = _exact(value, (
            "active", "canonical_surface", "concept_key", "definition_text",
            "entry_key", "field_roles", "language", "observation_key",
            "relation_kind", "sense_key", "source_ref",
            "supersedes_entry_keys", "surface",
        ), where="sense entry")
        roles = raw["field_roles"]
        supersedes = raw["supersedes_entry_keys"]
        if not isinstance(roles, list) or not isinstance(supersedes, list):
            raise W03PublicSenseContractError(
                "sense entry list 字段类型非法")
        definition = raw["definition_text"]
        if definition is not None and not isinstance(definition, str):
            raise W03PublicSenseContractError("sense entry definition 类型非法")
        return cls(
            _key_from_value(raw["entry_key"], where="sense entry key"),
            _text(raw["surface"], where="sense entry surface"),
            _text(
                raw["canonical_surface"],
                where="sense entry canonical_surface",
            ),
            _text(raw["language"], where="sense entry language"),
            _text(
                raw["relation_kind"], where="sense entry relation_kind"),
            definition,
            _key_from_value(raw["sense_key"], where="sense key"),
            _key_from_value(raw["concept_key"], where="concept key"),
            _key_from_value(raw["observation_key"], where="observation key"),
            W03PublicSenseSourceRef.from_dict(raw["source_ref"]),
            tuple(
                _text(item, where="sense entry field role")
                for item in roles),
            _integer(raw["active"], where="sense entry active"),
            tuple(
                _key_from_value(item, where="superseded entry key")
                for item in supersedes),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03PublicSenseAlias:
    """由公开 redirect 形成的 surface REFERS 边。"""

    alias_surface: str
    target_surface: str
    language: str
    source_ref: W03PublicSenseSourceRef
    observation_key: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("alias_surface", "target_surface", "language"):
            _text(getattr(self, name), where=f"sense alias {name}")
        if self.alias_surface == self.target_surface:
            raise W03PublicSenseContractError("sense alias 不得自环")
        if not isinstance(self.source_ref, W03PublicSenseSourceRef):
            raise TypeError("sense alias source_ref 类型非法")
        _key(self.observation_key, where="sense alias observation_key")

    def to_dict(self) -> dict[str, object]:
        """导出规范 alias JSON 值。"""
        return {
            "alias_surface": self.alias_surface,
            "language": self.language,
            "observation_key": list(self.observation_key),
            "source_ref": self.source_ref.to_dict(),
            "target_surface": self.target_surface,
        }

    @classmethod
    def from_dict(cls, value: object) -> "W03PublicSenseAlias":
        """从字段精确的 JSON object 恢复 alias。"""
        raw = _exact(value, (
            "alias_surface", "language", "observation_key", "source_ref",
            "target_surface",
        ), where="sense alias")
        return cls(
            _text(raw["alias_surface"], where="sense alias surface"),
            _text(raw["target_surface"], where="sense alias target"),
            _text(raw["language"], where="sense alias language"),
            W03PublicSenseSourceRef.from_dict(raw["source_ref"]),
            _key_from_value(
                raw["observation_key"], where="sense alias observation key"),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03PublicSenseSourcePackIdentity:
    """compact artifact 消费的完整 source-pack manifest 身份。"""

    relative_path: str
    manifest_sha256: str
    artifact_key: tuple[int, ...]
    source_key: str
    license_id: str
    snapshot_id: str
    source_ref_count: int
    observation_count: int

    def __post_init__(self) -> None:
        for name in ("relative_path", "source_key", "license_id", "snapshot_id"):
            _text(getattr(self, name), where=f"source pack {name}")
        if (self.relative_path.startswith(("/", "../"))
                or "\\" in self.relative_path):
            raise W03PublicSenseContractError(
                "source pack relative_path 非安全 POSIX 路径")
        _sha256(self.manifest_sha256, where="source pack manifest")
        _key(self.artifact_key, where="source pack artifact key")
        if (type(self.source_ref_count) is not int
                or type(self.observation_count) is not int
                or self.source_ref_count <= 0 or self.observation_count <= 0):
            raise W03PublicSenseContractError("source pack count 非正整数")

    def to_dict(self) -> dict[str, object]:
        """导出规范 source-pack identity JSON 值。"""
        return {
            "artifact_key": list(self.artifact_key),
            "license_id": self.license_id,
            "manifest_sha256": self.manifest_sha256,
            "observation_count": self.observation_count,
            "relative_path": self.relative_path,
            "snapshot_id": self.snapshot_id,
            "source_key": self.source_key,
            "source_ref_count": self.source_ref_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "W03PublicSenseSourcePackIdentity":
        """从字段精确的 JSON object 恢复 source-pack identity。"""
        raw = _exact(value, (
            "artifact_key", "license_id", "manifest_sha256",
            "observation_count", "relative_path", "snapshot_id",
            "source_key", "source_ref_count",
        ), where="source pack identity")
        return cls(
            _text(raw["relative_path"], where="source pack relative_path"),
            _sha256(raw["manifest_sha256"], where="source pack manifest"),
            _key_from_value(raw["artifact_key"], where="source pack key"),
            _text(raw["source_key"], where="source pack source_key"),
            _text(raw["license_id"], where="source pack license_id"),
            _text(raw["snapshot_id"], where="source pack snapshot_id"),
            _integer(
                raw["source_ref_count"], where="source pack source_ref_count"),
            _integer(
                raw["observation_count"],
                where="source pack observation_count",
            ),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03PublicSenseSourceRevision:
    """一个真实来源 registry 的 append-only supersede 身份。"""

    revision_kind: str
    active_sha256: str
    supersedes_sha256: str

    def __post_init__(self) -> None:
        _text(self.revision_kind, where="source revision kind")
        _sha256(self.active_sha256, where="source revision active")
        _sha256(self.supersedes_sha256, where="source revision supersedes")
        if self.active_sha256 == self.supersedes_sha256:
            raise W03PublicSenseContractError("source revision 不得自 supersede")

    def to_dict(self) -> dict[str, object]:
        """导出规范 source revision JSON 值。"""
        return {
            "active_sha256": self.active_sha256,
            "revision_kind": self.revision_kind,
            "supersedes_sha256": self.supersedes_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "W03PublicSenseSourceRevision":
        """从字段精确的 JSON object 恢复 source revision。"""
        raw = _exact(value, (
            "active_sha256", "revision_kind", "supersedes_sha256",
        ), where="source revision")
        return cls(
            _text(raw["revision_kind"], where="source revision kind"),
            _sha256(raw["active_sha256"], where="source revision active"),
            _sha256(
                raw["supersedes_sha256"],
                where="source revision supersedes",
            ),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03PublicSenseArtifact:
    """只含 learned projection 与来源身份的 compact runtime artifact。"""

    source_packs: tuple[W03PublicSenseSourcePackIdentity, ...]
    source_revisions: tuple[W03PublicSenseSourceRevision, ...]
    entries: tuple[W03PublicSenseEntry, ...]
    aliases: tuple[W03PublicSenseAlias, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.source_packs, tuple) or not self.source_packs
                or any(not isinstance(item, W03PublicSenseSourcePackIdentity)
                       for item in self.source_packs)):
            raise W03PublicSenseContractError("artifact source_packs 非法")
        if (not isinstance(self.source_revisions, tuple)
                or any(not isinstance(item, W03PublicSenseSourceRevision)
                       for item in self.source_revisions)):
            raise W03PublicSenseContractError("artifact source_revisions 非法")
        if (not isinstance(self.entries, tuple) or not self.entries
                or any(not isinstance(item, W03PublicSenseEntry)
                       for item in self.entries)):
            raise W03PublicSenseContractError("artifact entries 非法")
        if (not isinstance(self.aliases, tuple)
                or any(not isinstance(item, W03PublicSenseAlias)
                       for item in self.aliases)):
            raise W03PublicSenseContractError("artifact aliases 非法")
        pack_keys = tuple(
            (item.source_key, item.relative_path) for item in self.source_packs)
        entry_keys = tuple(item.entry_key for item in self.entries)
        alias_keys = tuple(
            (item.language, item.alias_surface, item.target_surface,
             item.observation_key) for item in self.aliases)
        if (pack_keys != tuple(sorted(set(pack_keys)))
                or entry_keys != tuple(sorted(set(entry_keys)))
                or alias_keys != tuple(sorted(set(alias_keys)))):
            raise W03PublicSenseContractError("artifact 顺序或唯一性漂移")
        by_key = {item.entry_key: item for item in self.entries}
        superseded = {
            key
            for item in self.entries if item.active == 1
            for key in item.supersedes_entry_keys
        }
        if any(key not in by_key for key in superseded):
            raise W03PublicSenseContractError("artifact supersede 引用缺失")
        if any(by_key[key].active != 0 for key in superseded):
            raise W03PublicSenseContractError(
                "artifact 被 supersede 候选仍处于 active")
        if any(
                item.active == 0 and item.entry_key not in superseded
                for item in self.entries):
            raise W03PublicSenseContractError(
                "artifact inactive 候选缺少 active superseder")

    def payload_value(self) -> dict[str, object]:
        """导出不含 envelope 的规范 payload。"""
        return {
            "aliases": [item.to_dict() for item in self.aliases],
            "entries": [item.to_dict() for item in self.entries],
            "source_packs": [item.to_dict() for item in self.source_packs],
            "source_revisions": [
                item.to_dict() for item in self.source_revisions],
        }

    def payload_sha256(self) -> str:
        """返回 compact payload commitment。"""
        return _sha_value(self.payload_value())

    @classmethod
    def from_payload_value(cls, value: object) -> "W03PublicSenseArtifact":
        """从字段精确的 payload 恢复完整 artifact。"""
        raw = _exact(value, (
            "aliases", "entries", "source_packs", "source_revisions",
        ), where="sense artifact payload")
        for key in raw:
            if not isinstance(raw[key], list):
                raise W03PublicSenseContractError(
                    f"sense artifact {key} 不是 list")
        return cls(
            tuple(W03PublicSenseSourcePackIdentity.from_dict(item)
                  for item in raw["source_packs"]),
            tuple(W03PublicSenseSourceRevision.from_dict(item)
                  for item in raw["source_revisions"]),
            tuple(W03PublicSenseEntry.from_dict(item)
                  for item in raw["entries"]),
            tuple(W03PublicSenseAlias.from_dict(item)
                  for item in raw["aliases"]),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03PublicSenseQuery:
    """不携带 expected answer 的原始 term/短语查询。"""

    surface: str
    context_text: str | None = None
    language: str = "zh"

    def __post_init__(self) -> None:
        _text(self.surface, where="sense query surface")
        _text(self.language, where="sense query language")
        if self.context_text is not None:
            _text(self.context_text, where="sense query context")

    def to_dict(self) -> dict[str, object]:
        """导出规范 query JSON 值。"""
        return {
            "context_text": self.context_text,
            "language": self.language,
            "surface": self.surface,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03PublicSenseCandidate:
    """查询返回的 active typed candidate 与完整 SourceRef。"""

    entry: W03PublicSenseEntry
    matched_surface: str

    def __post_init__(self) -> None:
        if not isinstance(self.entry, W03PublicSenseEntry):
            raise TypeError("query candidate entry 类型非法")
        if self.entry.active != 1:
            raise W03PublicSenseContractError("query candidate 不是 active")
        _text(self.matched_surface, where="query candidate matched_surface")

    def to_dict(self) -> dict[str, object]:
        """导出规范 query candidate JSON 值。"""
        return {
            "entry": self.entry.to_dict(),
            "matched_surface": self.matched_surface,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03PublicSenseQueryResult:
    """诚实标注唯一、多义、未知、澄清或未合并来源冲突。"""

    query: W03PublicSenseQuery
    status: str
    candidates: tuple[W03PublicSenseCandidate, ...]
    alias_path: tuple[str, ...]
    conflict_kind: str | None
    artifact_sha256: str
    trace_commitment_sha256: str
    experimental: int = 1
    formal_mastery_claim: int = 0
    w03_started: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.query, W03PublicSenseQuery):
            raise TypeError("sense query result query 类型非法")
        if self.status not in W03_PUBLIC_SENSE_STATUSES:
            raise W03PublicSenseContractError("sense query status 漂移")
        if (not isinstance(self.candidates, tuple)
                or any(not isinstance(item, W03PublicSenseCandidate)
                       for item in self.candidates)):
            raise W03PublicSenseContractError("sense query candidates 非法")
        if (not isinstance(self.alias_path, tuple)
                or any(not isinstance(item, str) or not item
                       for item in self.alias_path)):
            raise W03PublicSenseContractError("sense query alias_path 非法")
        if self.conflict_kind is not None:
            _text(self.conflict_kind, where="sense query conflict_kind")
        if ((self.status == "CONFLICT")
                != (self.conflict_kind is not None)):
            raise W03PublicSenseContractError(
                "sense query conflict status/kind 不一致")
        if self.status == "UNKNOWN" and self.candidates:
            raise W03PublicSenseContractError("UNKNOWN 不得返回候选")
        if self.status != "UNKNOWN" and not self.candidates:
            raise W03PublicSenseContractError("非 UNKNOWN 必须返回候选")
        _sha256(self.artifact_sha256, where="sense query artifact")
        _sha256(self.trace_commitment_sha256, where="sense query trace")
        if (self.experimental, self.formal_mastery_claim,
                self.w03_started) != (1, 0, 0):
            raise W03PublicSenseContractError("sense query formal boundary 漂移")

    def to_dict(self) -> dict[str, object]:
        """导出规范 query result JSON 值。"""
        return {
            "alias_path": list(self.alias_path),
            "artifact_sha256": self.artifact_sha256,
            "candidates": [item.to_dict() for item in self.candidates],
            "conflict_kind": self.conflict_kind,
            "experimental": self.experimental,
            "formal_mastery_claim": self.formal_mastery_claim,
            "query": self.query.to_dict(),
            "status": self.status,
            "trace_commitment_sha256": self.trace_commitment_sha256,
            "w03_started": self.w03_started,
        }

    def sha256(self) -> str:
        """返回完整查询结果 commitment。"""
        return _sha_value(self.to_dict())


__all__ = [
    "W03_PUBLIC_SENSE_ARTIFACT_VERSION",
    "W03_PUBLIC_SENSE_FORMAT",
    "W03_PUBLIC_SENSE_MAX_ARTIFACT_BYTES",
    "W03_PUBLIC_SENSE_RELATIONS",
    "W03_PUBLIC_SENSE_SCHEMA_VERSION",
    "W03_PUBLIC_SENSE_STATUSES",
    "W03PublicSenseAlias",
    "W03PublicSenseArtifact",
    "W03PublicSenseCandidate",
    "W03PublicSenseContractError",
    "W03PublicSenseEntry",
    "W03PublicSenseQuery",
    "W03PublicSenseQueryResult",
    "W03PublicSenseSourcePackIdentity",
    "W03PublicSenseSourceRef",
    "W03PublicSenseSourceRevision",
]
