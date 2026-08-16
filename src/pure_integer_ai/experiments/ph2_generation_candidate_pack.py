"""E-05 generation candidate 的不可变模型 pack 与规范回读边界。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    CurriculumVersion,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    ScopeIdentity,
    document_scope,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    canonical_json_bytes,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    REFERENCE_STRATEGIES,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    PATTERN_CLAIM,
    PATTERN_LITERAL,
    GroundedAnswerSurfaceModel,
    LearnedSurfacePattern,
    SurfacePatternPart,
)


ARTIFACT_KIND = "PH2_GENERATION_CANDIDATE_PACK_V1"
PACK_NAME = "PH2_GENERATION_CANDIDATE"
SCHEMA_VERSION = 1
RULE_CLAIM = "CLAIM_FROM_VISIBLE_EVIDENCE"
RULE_LITERAL = "LITERAL_FROM_TRAIN_PATTERN"
RULE_REFERENCE = "REFERENCE_FROM_VISIBLE_INPUT"
RULE_RESPONSE_ACT = "RESPONSE_ACT_FROM_TRAIN_PATTERN"
REPRESENTATION_RULES = (
    RULE_CLAIM,
    RULE_LITERAL,
    RULE_REFERENCE,
    RULE_RESPONSE_ACT,
)
_NAMESPACE = 22010
_TOP_LEVEL_FIELDS = frozenset({
    "artifact_kind", "payload", "payload_sha256", "schema_version",
})
_PAYLOAD_FIELDS = frozenset({
    "candidate_version", "minimum_forming_sources", "model",
    "owner_scope_key", "owner_source_key", "reference_strategies",
    "representation_rules", "training_artifact_sha256",
})
_MODEL_FIELDS = frozenset({"patterns"})
_PATTERN_FIELDS = frozenset({
    "carrier_kind", "claim_count", "forming_evidence_keys", "parts",
    "pattern_id", "response_act", "support_episode_ids",
})
_PART_FIELDS = frozenset({"claim_ordinal", "kind", "literal"})


# object-model: exception
class GenerationCandidatePackError(RuntimeError):
    """candidate pack 字段、内容锁、规范字节或发布边界不一致。"""


def _exact(
        value: Any, fields: frozenset[str], *, where: str,
        ) -> dict[str, Any]:
    """核验规范 JSON object 使用精确字段集合。"""
    if not isinstance(value, dict) or set(value) != fields:
        raise GenerationCandidatePackError(f"{where} 字段集合漂移")
    return value


def _strict_key(value: Any, *, where: str) -> tuple[int, ...]:
    """从 JSON list 或 tuple 恢复非空严格整数键。"""
    if not isinstance(value, (list, tuple)) or not value:
        raise GenerationCandidatePackError(f"{where} 必须是非空整数序列")
    result = tuple(value)
    if any(type(item) is not int for item in result):
        raise GenerationCandidatePackError(f"{where} 必须使用严格整数")
    return result


def _sha256(value: str, *, where: str) -> str:
    """规范化并核验小写 SHA-256。"""
    if not isinstance(value, str):
        raise GenerationCandidatePackError(f"{where} 类型错误")
    digest = value.lower()
    if (len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise GenerationCandidatePackError(f"{where} 格式错误")
    return digest


def _stable_positive_int(payload: bytes) -> int:
    """从规范字节导出稳定正整数 SourceRef id。"""
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    value &= (1 << 63) - 1
    return value if value > 0 else 1


def _pack_directory_name(pack: "GenerationCandidatePack") -> str:
    """把 candidate version 与 payload 锁编码进不可覆盖目录身份。"""
    version = "-".join(str(item) for item in pack.candidate_version)
    return f"{PACK_NAME}--v{version}--{pack.sha256()[:16]}"


def _pattern_to_dict(pattern: LearnedSurfacePattern) -> dict[str, object]:
    """把已学 pattern 导出为不含原 episode surface 集的规范值。"""
    return {
        "carrier_kind": pattern.carrier_kind,
        "claim_count": pattern.claim_count,
        "forming_evidence_keys": [
            list(item) for item in pattern.support_teacher_keys
        ],
        "parts": [
            {
                "claim_ordinal": item.claim_ordinal,
                "kind": item.kind,
                "literal": item.literal,
            }
            for item in pattern.parts
        ],
        "pattern_id": pattern.pattern_id,
        "response_act": pattern.response_act,
        "support_episode_ids": list(pattern.support_episode_ids),
    }


def _pattern_from_dict(value: Any) -> LearnedSurfacePattern:
    """从 pack 恢复一个已学 pattern，并复用现有模型不变量。"""
    raw = _exact(value, _PATTERN_FIELDS, where="candidate pattern")
    if (not isinstance(raw["parts"], list)
            or not isinstance(raw["support_episode_ids"], list)
            or not isinstance(raw["forming_evidence_keys"], list)):
        raise GenerationCandidatePackError("candidate pattern list 字段非法")
    parts = []
    for value_part in raw["parts"]:
        part = _exact(value_part, _PART_FIELDS, where="candidate pattern part")
        parts.append(SurfacePatternPart(
            part["kind"], part["literal"], part["claim_ordinal"]))
    return LearnedSurfacePattern(
        raw["pattern_id"],
        raw["response_act"],
        raw["carrier_kind"],
        raw["claim_count"],
        tuple(parts),
        tuple(raw["support_episode_ids"]),
        tuple(_strict_key(item, where="forming evidence key")
              for item in raw["forming_evidence_keys"]),
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationCandidatePack:
    """保存已学 surface model、owner/scope、表示规则和训练来源内容锁。"""

    candidate_version: tuple[int, ...]
    training_artifact_sha256: str
    owner_source: SourceRef
    owner_scope: ScopeIdentity
    model: GroundedAnswerSurfaceModel
    representation_rules: tuple[str, ...] = REPRESENTATION_RULES
    reference_strategies: tuple[str, ...] = REFERENCE_STRATEGIES
    minimum_forming_sources: int = 1

    def __post_init__(self) -> None:
        version = _strict_key(
            self.candidate_version, where="candidate version")
        object.__setattr__(self, "candidate_version", version)
        object.__setattr__(
            self,
            "training_artifact_sha256",
            _sha256(
                self.training_artifact_sha256,
                where="training artifact SHA-256",
            ),
        )
        if not isinstance(self.owner_source, SourceRef):
            raise TypeError("candidate pack owner_source 类型错误")
        if (not isinstance(self.owner_scope, ScopeIdentity)
                or self.owner_scope.source != self.owner_source):
            raise GenerationCandidatePackError(
                "candidate pack owner_scope 必须绑定 owner_source")
        if not isinstance(self.model, GroundedAnswerSurfaceModel):
            raise TypeError("candidate pack model 类型错误")
        if self.representation_rules != REPRESENTATION_RULES:
            raise GenerationCandidatePackError("candidate 表示规则漂移")
        if self.reference_strategies != REFERENCE_STRATEGIES:
            raise GenerationCandidatePackError("candidate reference 策略漂移")
        if (type(self.minimum_forming_sources) is not int
                or self.minimum_forming_sources <= 0):
            raise GenerationCandidatePackError(
                "candidate minimum forming sources 必须为严格正整数")
        parts = tuple(
            part for pattern in self.model.patterns for part in pattern.parts)
        if (not any(part.kind == PATTERN_CLAIM for part in parts)
                or not any(part.kind == PATTERN_LITERAL for part in parts)
                or not any(pattern.response_act != "ANSWER"
                           for pattern in self.model.patterns)):
            raise GenerationCandidatePackError(
                "candidate model 未覆盖 claim、literal 和 response-act")

    def pattern(self, pattern_id: int) -> LearnedSurfacePattern:
        """按稳定 pattern id 返回 pack 内唯一已学模式。"""
        matches = tuple(
            item for item in self.model.patterns
            if item.pattern_id == pattern_id)
        if len(matches) != 1:
            raise GenerationCandidatePackError(
                "请求的 pattern 不属于 candidate pack")
        return matches[0]

    def literal_inventory(self) -> tuple[str, ...]:
        """返回全部 TRAIN learned literal，供 grammar fragment 严格核验。"""
        return tuple(sorted({
            part.literal
            for pattern in self.model.patterns
            for part in pattern.parts
            if part.kind == PATTERN_LITERAL
        }))

    def payload_dict(self) -> dict[str, object]:
        """导出内容锁覆盖的完整 candidate payload。"""
        return {
            "candidate_version": list(self.candidate_version),
            "minimum_forming_sources": self.minimum_forming_sources,
            "model": {
                "patterns": [
                    _pattern_to_dict(item) for item in self.model.patterns
                ],
            },
            "owner_scope_key": list(self.owner_scope.stable_key()),
            "owner_source_key": list(self.owner_source.stable_key()),
            "reference_strategies": list(self.reference_strategies),
            "representation_rules": list(self.representation_rules),
            "training_artifact_sha256": self.training_artifact_sha256,
        }

    def sha256(self) -> str:
        """返回不依赖路径和对象地址的 candidate payload SHA-256。"""
        return hashlib.sha256(
            canonical_json_bytes(self.payload_dict())).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """导出带自校验内容锁的单文件 pack manifest。"""
        return {
            "artifact_kind": ARTIFACT_KIND,
            "payload": self.payload_dict(),
            "payload_sha256": self.sha256(),
            "schema_version": SCHEMA_VERSION,
        }

    def canonical_bytes(self) -> bytes:
        """返回以单换行结束的 canonical JSON bytes。"""
        return canonical_json_line(self.to_dict())

    @classmethod
    def from_dict(cls, value: Any) -> "GenerationCandidatePack":
        """从精确 JSON object 恢复 pack 并复核 payload 内容锁。"""
        raw = _exact(value, _TOP_LEVEL_FIELDS, where="candidate pack")
        if (raw["artifact_kind"] != ARTIFACT_KIND
                or raw["schema_version"] != SCHEMA_VERSION):
            raise GenerationCandidatePackError(
                "candidate pack kind/schema 漂移")
        payload = _exact(
            raw["payload"], _PAYLOAD_FIELDS, where="candidate payload")
        model_value = _exact(
            payload["model"], _MODEL_FIELDS, where="candidate model")
        if not isinstance(model_value["patterns"], list):
            raise GenerationCandidatePackError(
                "candidate model patterns 类型错误")
        expected = _sha256(
            raw["payload_sha256"], where="candidate payload SHA-256")
        actual = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if actual != expected:
            raise GenerationCandidatePackError(
                "candidate pack payload 内容锁漂移")
        pack = cls(
            _strict_key(payload["candidate_version"],
                        where="candidate version"),
            payload["training_artifact_sha256"],
            SourceRef.from_stable_key(_strict_key(
                payload["owner_source_key"], where="owner source key")),
            ScopeIdentity.from_stable_key(_strict_key(
                payload["owner_scope_key"], where="owner scope key")),
            GroundedAnswerSurfaceModel(tuple(
                _pattern_from_dict(item)
                for item in model_value["patterns"]
            )),
            tuple(payload["representation_rules"]),
            tuple(payload["reference_strategies"]),
            payload["minimum_forming_sources"],
        )
        if pack.sha256() != expected:
            raise GenerationCandidatePackError(
                "candidate pack readback 内容锁不一致")
        return pack


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class LoadedGenerationCandidatePack:
    """返回 pack 物理根、规范 manifest 路径和已核验模型。"""

    pack_root: Path
    manifest_path: Path
    pack: GenerationCandidatePack


def build_generation_candidate_pack(
        model: GroundedAnswerSurfaceModel,
        training_artifact_sha256: str,
        *, candidate_version: tuple[int, ...] = (1,),
        ) -> GenerationCandidatePack:
    """从已学模型和训练 artifact 内容锁建立路径无关 candidate pack。"""
    if not isinstance(model, GroundedAnswerSurfaceModel):
        raise TypeError("candidate pack builder model 类型错误")
    digest = _sha256(
        training_artifact_sha256, where="training artifact SHA-256")
    version = _strict_key(candidate_version, where="candidate version")
    owner_id = _stable_positive_int(canonical_json_bytes({
        "candidate_version": list(version),
        "training_artifact_sha256": digest,
    }))
    owner = SourceRef(
        _NAMESPACE,
        owner_id,
        0,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(curriculum=CurriculumVersion(version[-1])),
    )
    return GenerationCandidatePack(
        version,
        digest,
        owner,
        document_scope(owner),
        model,
    )


def publish_generation_candidate_pack(
        pack: GenerationCandidatePack,
        release_root: str | Path,
        ) -> LoadedGenerationCandidatePack:
    """以 manifest-last 单文件形式独占发布 candidate pack。"""
    if not isinstance(pack, GenerationCandidatePack):
        raise TypeError("candidate pack publisher 输入类型错误")
    root = (
        Path(release_root).resolve()
        / "packs"
        / _pack_directory_name(pack)
    ).resolve()
    if root.exists():
        raise GenerationCandidatePackError(
            "candidate pack 已存在，必须提升 candidate version")
    root.mkdir(parents=True, exist_ok=False)
    target = root / "manifest.json"
    handle = tempfile.NamedTemporaryFile(
        prefix=".manifest.json.building-",
        dir=root,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        handle.write(pack.canonical_bytes())
        handle.close()
        os.replace(temporary, target)
    finally:
        if not handle.closed:
            handle.close()
        if temporary.exists():
            temporary.unlink()
    return LoadedGenerationCandidatePack(root, target, pack)


def read_generation_candidate_pack(
        path: str | Path,
        *, expected_sha256: str | None = None,
        ) -> LoadedGenerationCandidatePack:
    """严格回读 pack manifest、canonical bytes 和可选外部内容锁。"""
    supplied = Path(path).resolve()
    target = supplied / "manifest.json" if supplied.is_dir() else supplied
    try:
        payload = target.read_bytes()
    except OSError as error:
        raise GenerationCandidatePackError(
            "candidate pack manifest 无法读取") from error
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise GenerationCandidatePackError(
            "candidate pack manifest 换行非法")
    try:
        value = parse_canonical_json_bytes(
            payload[:-1], require_object=True)
    except DatasetContractError as error:
        raise GenerationCandidatePackError(
            "candidate pack manifest 非 canonical JSON") from error
    if canonical_json_line(value) != payload:
        raise GenerationCandidatePackError(
            "candidate pack manifest 字节非规范")
    pack = GenerationCandidatePack.from_dict(value)
    if (expected_sha256 is not None
            and pack.sha256() != _sha256(
                expected_sha256, where="expected candidate SHA-256")):
        raise GenerationCandidatePackError(
            "candidate pack 外部内容锁漂移")
    return LoadedGenerationCandidatePack(target.parent, target, pack)


__all__ = [
    "ARTIFACT_KIND",
    "GenerationCandidatePack",
    "GenerationCandidatePackError",
    "LoadedGenerationCandidatePack",
    "PACK_NAME",
    "REPRESENTATION_RULES",
    "RULE_CLAIM",
    "RULE_LITERAL",
    "RULE_REFERENCE",
    "RULE_RESPONSE_ACT",
    "build_generation_candidate_pack",
    "publish_generation_candidate_pack",
    "read_generation_candidate_pack",
]
