"""显式 prerequisite 授权的 W-04 到 W-05 公开 bridge 合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_query_contract import (
    W04V2PublicQueryResult,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_query_contract import (
    W05V2PublicQueryResult,
)


W04_W05_PUBLIC_BRIDGE_STATUSES = {"BRIDGED", "UNKNOWN", "CLARIFY"}


# object-model: exception
class W04W05PublicBridgeError(ValueError):
    """bridge query、stage result 或显式 Observation link 漂移。"""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _text(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise W04W05PublicBridgeError(f"{where} 不是规范文本")
    return value


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W04W05PublicBridgeError(f"{where} 不是严格整数键")
    return value


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise W04W05PublicBridgeError(f"{where} 不是 SHA-256")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W04W05PublicBridgeQuery:
    """分别查询 primitive 表层与完整 Proposition 表层，不携带预期答案。"""

    primitive_surface: str
    context_text: str
    proposition_surface: str
    allow_generation: int = 1

    def __post_init__(self) -> None:
        _text(self.primitive_surface, where="bridge primitive surface")
        _text(self.context_text, where="bridge context")
        _text(self.proposition_surface, where="bridge Proposition surface")
        if self.allow_generation not in {0, 1}:
            raise W04W05PublicBridgeError(
                "bridge allow_generation 必须是零或一")

    def to_dict(self) -> dict[str, object]:
        return {
            "allow_generation": self.allow_generation,
            "context_text": self.context_text,
            "primitive_surface": self.primitive_surface,
            "proposition_surface": self.proposition_surface,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W04W05PublicBridgeLink:
    """同源 W04 Observation 到 W05 Proposition 的显式 typed link。"""

    source_record_key: tuple[int, ...]
    source_commitment: str
    w04_observation_key: tuple[int, ...]
    w05_observation_key: tuple[int, ...]
    primitive_registry: str
    primitive_kind: int
    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    predicate_occurrence_key: tuple[int, ...]
    context_key: tuple[int, ...]
    occurrence_order: tuple[tuple[int, ...], ...]
    role_binding_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        for name in (
                "source_record_key", "w04_observation_key",
                "w05_observation_key", "proposition_key", "predicate_key",
                "predicate_occurrence_key", "context_key"):
            _strict_key(getattr(self, name), where=f"bridge link {name}")
        _sha256(self.source_commitment, where="bridge source commitment")
        _text(self.primitive_registry, where="bridge primitive registry")
        if type(self.primitive_kind) is not int or self.primitive_kind <= 0:
            raise W04W05PublicBridgeError("bridge primitive kind 漂移")
        for name in ("occurrence_order", "role_binding_keys"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise W04W05PublicBridgeError(f"bridge {name} 为空")
            for item in values:
                _strict_key(item, where=f"bridge {name} item")
        if self.predicate_occurrence_key not in self.occurrence_order:
            raise W04W05PublicBridgeError(
                "bridge predicate occurrence 不在 occurrence order")

    def to_dict(self) -> dict[str, object]:
        return {
            "context_key": list(self.context_key),
            "occurrence_order": [list(item) for item in self.occurrence_order],
            "predicate_key": list(self.predicate_key),
            "predicate_occurrence_key": list(self.predicate_occurrence_key),
            "primitive_kind": self.primitive_kind,
            "primitive_registry": self.primitive_registry,
            "proposition_key": list(self.proposition_key),
            "role_binding_keys": [list(item) for item in self.role_binding_keys],
            "source_commitment": self.source_commitment,
            "source_record_key": list(self.source_record_key),
            "w04_observation_key": list(self.w04_observation_key),
            "w05_observation_key": list(self.w05_observation_key),
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W04W05PublicBridgeResult:
    """两个 stage 的查询结果与可选 prerequisite 授权 link。"""

    query: W04W05PublicBridgeQuery
    status: str
    w04_result: W04V2PublicQueryResult
    w05_result: W05V2PublicQueryResult
    link: W04W05PublicBridgeLink | None
    w04_source_binding_sha256: str
    w05_source_binding_sha256: str
    experimental: int = 1
    formal_mastery_claim: int = 0
    w04_started: int = 0
    w05_started: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.query, W04W05PublicBridgeQuery)
                or self.status not in W04_W05_PUBLIC_BRIDGE_STATUSES
                or not isinstance(self.w04_result, W04V2PublicQueryResult)
                or not isinstance(self.w05_result, W05V2PublicQueryResult)):
            raise W04W05PublicBridgeError("bridge result 投影漂移")
        if self.status == "BRIDGED":
            if (not isinstance(self.link, W04W05PublicBridgeLink)
                    or self.w04_result.status != "UNIQUE"
                    or self.w05_result.status != "UNIQUE"):
                raise W04W05PublicBridgeError(
                    "BRIDGED result 缺两个 UNIQUE stage 或 link")
        elif self.link is not None:
            raise W04W05PublicBridgeError(
                "非 BRIDGED result 不得发布 link")
        _sha256(self.w04_source_binding_sha256, where="W04 source binding")
        _sha256(self.w05_source_binding_sha256, where="W05 source binding")
        if (self.experimental, self.formal_mastery_claim,
                self.w04_started, self.w05_started) != (1, 0, 0, 0):
            raise W04W05PublicBridgeError("bridge result 边界漂移")

    def to_dict(self) -> dict[str, object]:
        return {
            "experimental": self.experimental,
            "formal_mastery_claim": self.formal_mastery_claim,
            "link": None if self.link is None else self.link.to_dict(),
            "query": self.query.to_dict(),
            "status": self.status,
            "w04_result": self.w04_result.to_dict(),
            "w04_source_binding_sha256": self.w04_source_binding_sha256,
            "w04_started": self.w04_started,
            "w05_result": self.w05_result.to_dict(),
            "w05_source_binding_sha256": self.w05_source_binding_sha256,
            "w05_started": self.w05_started,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


__all__ = [
    "W04_W05_PUBLIC_BRIDGE_STATUSES",
    "W04W05PublicBridgeError",
    "W04W05PublicBridgeLink",
    "W04W05PublicBridgeQuery",
    "W04W05PublicBridgeResult",
]
