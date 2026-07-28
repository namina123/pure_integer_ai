"""MD-03 将三向既有输入薄映射为 MD-01 center，不执行召回或写入。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.generation_plan import (
    AnswerGenerationGoal,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    OBJECT_SPAN,
    ObjectIdentity,
    TypedRef,
)
from pure_integer_ai.cognition.shared.memory_query import MemoryCurrentQuery
from pure_integer_ai.cognition.shared.reasoning_planner import (
    ReasoningObligation,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.experiments.ph2_memory_dynamics_contract import (
    CENTER_STRENGTHS,
    DIRECTIONS,
    MemoryAttentionCenter,
    MemoryCenterOrigin,
    MemoryDynamicsBoundary,
)


FORMAT_VERSION = 1
PAYLOAD_KINDS = (
    "ANSWER_GENERATION_GOAL",
    "CURRENT_TYPED_INPUT",
    "EVIDENCE_OBLIGATION",
)

_PAYLOAD_BY_DIRECTION = {
    "GENERATION": "ANSWER_GENERATION_GOAL",
    "REASONING": "EVIDENCE_OBLIGATION",
    "UNDERSTANDING": "CURRENT_TYPED_INPUT",
}
_ALLOWED_WRITES = {
    "GENERATION": (
        "DIMENSION_OUTCOME",
        "OUTPUT_PLAN",
        "SELECTION_USE",
        "SURFACE_OCCURRENCE",
    ),
    "REASONING": (
        "CONFLICT_RECEIPT",
        "DERIVATION_RECEIPT",
        "PROVISIONAL_PROJECTION",
        "UNRESOLVED_PREMISE",
    ),
    "UNDERSTANDING": (
        "INPUT_INTERPRETATION_CANDIDATE",
        "REFERENCE_REVISION_CANDIDATE",
        "SOURCED_CURRENT_PROJECTION_UPDATE",
    ),
}
_COMMON_FORBIDDEN_WRITES = (
    "COMPANION_WRITE",
    "CORE_LEARNING_WRITE",
    "MEMORY_LEARNING_WRITE",
    "USE_WRITE_WITHOUT_EXPLICIT_COMMIT",
)
_DIRECTION_FORBIDDEN_WRITES = {
    "GENERATION": (
        "ACTIVATION_AS_FACT",
        "INPUT_FACT_BACKWRITE",
        "RENDERER_SUCCESS_AS_POSTCHECK_PASS",
    ),
    "REASONING": (
        "ABDUCTIVE_CAUSES_ASSERTION",
        "DEFINITIVE_WITHOUT_EVIDENCE_GROUNDING",
    ),
    "UNDERSTANDING": (
        "ORIGINAL_OBSERVATION_REWRITE",
        "PRAGMATIC_CANDIDATE_AS_PRIVATE_TRUTH",
    ),
}


class MD03CenterAdapterError(ValueError):
    """三向 center 输入、边界、权限或去重不闭合。"""


def _positive_component(value: int) -> int:
    """把任意严格整数可逆映射为 StableRecordKey 要求的正整数。"""
    if type(value) is not int:
        raise MD03CenterAdapterError("center identity 必须使用严格整数")
    return value * 2 + 1 if value >= 0 else -value * 2


def _record_key(values: tuple[int, ...], *, domain: int) -> StableRecordKey:
    """按显式 domain 把现有完整稳定键编码为 D-02 共用整数键。"""
    if type(domain) is not int or domain <= 0:
        raise MD03CenterAdapterError("center key domain 非法")
    if not isinstance(values, tuple) or not values:
        raise MD03CenterAdapterError("center identity key 不得为空")
    return StableRecordKey((domain, *tuple(
        _positive_component(value) for value in values)))


def _normalized_keys(
        values: tuple[StableRecordKey, ...],
        *,
        where: str,
        ) -> tuple[StableRecordKey, ...]:
    """要求 D-02 稳定键非空、唯一并稳定排序。"""
    if (not isinstance(values, tuple) or not values
            or any(not isinstance(item, StableRecordKey) for item in values)):
        raise MD03CenterAdapterError(f"{where} 必须是非空 StableRecordKey tuple")
    ordered = tuple(sorted(values))
    if len(set(ordered)) != len(ordered):
        raise MD03CenterAdapterError(f"{where} 不得重复")
    return ordered


def _text_key(value: str) -> tuple[int, ...]:
    """把开放枚举文本编码为不含零的稳定整数键。"""
    if not isinstance(value, str) or not value:
        raise MD03CenterAdapterError("center text key 不得为空")
    return len(value), *tuple(ord(char) + 1 for char in value)


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(value), *value


def _center_values(
        *,
        direction: str,
        strength: str,
        origins: tuple[MemoryCenterOrigin, ...],
        obligation_kind_key: StableRecordKey,
        target_key: StableRecordKey,
        boundary: MemoryDynamicsBoundary,
        relation_profile_key: StableRecordKey,
        expansion_profile_key: StableRecordKey,
        dependency_keys: tuple[StableRecordKey, ...],
        ) -> tuple[int, ...]:
    """返回除 center_key 外的全部语义身份，供构造和复核共用。"""
    result = [
        FORMAT_VERSION,
        DIRECTIONS.index(direction) + 1,
        CENTER_STRENGTHS.index(strength) + 1,
        len(origins),
    ]
    for origin in origins:
        result.extend(_packed(_text_key(origin.origin_kind)))
        result.extend(_packed(origin.origin_key.stable_key()))
        result.append(len(origin.dependency_keys))
        for dependency in origin.dependency_keys:
            result.extend(_packed(dependency.stable_key()))
    for key in (
            obligation_kind_key,
            target_key,
            boundary.owner_key,
            boundary.scope_key,
            boundary.source_key,
            boundary.version_key,
            relation_profile_key,
            expansion_profile_key):
        result.extend(_packed(key.stable_key()))
    result.append(len(dependency_keys))
    for dependency in dependency_keys:
        result.extend(_packed(dependency.stable_key()))
    return tuple(result)


def _build_center(
        *,
        direction: str,
        strength: str,
        origins: tuple[MemoryCenterOrigin, ...],
        obligation_kind_key: StableRecordKey,
        target_key: StableRecordKey,
        boundary: MemoryDynamicsBoundary,
        relation_profile_key: StableRecordKey,
        expansion_profile_key: StableRecordKey,
        dependency_keys: tuple[StableRecordKey, ...],
        ) -> MemoryAttentionCenter:
    """规范合并 origin/dependency 后建立 activation-only center。"""
    if direction not in DIRECTIONS:
        raise MD03CenterAdapterError("center direction 非法")
    if strength not in CENTER_STRENGTHS:
        raise MD03CenterAdapterError("center strength 非法")
    origin_dependencies: dict[
        tuple[str, StableRecordKey], set[StableRecordKey]] = {}
    for origin in origins:
        if not isinstance(origin, MemoryCenterOrigin):
            raise TypeError("center origins 类型错误")
        origin_dependencies.setdefault(
            (origin.origin_kind, origin.origin_key), set()).update(
                origin.dependency_keys)
    origins = tuple(
        MemoryCenterOrigin(kind, key, tuple(sorted(dependencies)))
        for (kind, key), dependencies in sorted(origin_dependencies.items())
    )
    dependency_keys = _normalized_keys(
        dependency_keys, where="center dependencies")
    values = _center_values(
        direction=direction,
        strength=strength,
        origins=origins,
        obligation_kind_key=obligation_kind_key,
        target_key=target_key,
        boundary=boundary,
        relation_profile_key=relation_profile_key,
        expansion_profile_key=expansion_profile_key,
        dependency_keys=dependency_keys,
    )
    digest = integer_tuple_fingerprint(
        values, domain="ph2.md03.memory_attention_center.v1")
    center_key = _record_key(digest, domain=99)
    return MemoryAttentionCenter(
        center_key,
        direction,
        strength,
        origins,
        obligation_kind_key,
        target_key,
        boundary,
        relation_profile_key,
        expansion_profile_key,
        dependency_keys,
        "ACTIVE",
        1,
    )


def _boundary(current: MemoryCurrentQuery) -> MemoryDynamicsBoundary:
    """从真实 query 恢复 owner/scope/source/version 四个完整边界键。"""
    return MemoryDynamicsBoundary(
        _record_key(current.source.owner.stable_key(), domain=11),
        _record_key(current.scope.stable_key(), domain=12),
        _record_key(current.source.stable_key(), domain=13),
        _record_key(current.source.versions.stable_key(), domain=14),
    )


@dataclass(frozen=True)
class DirectionalCenterProfile:
    """一个方向注入的 obligation、typed relation 与 expansion profile。"""

    direction: str
    obligation_kind_key: StableRecordKey
    relation_profile_key: StableRecordKey
    expansion_profile_key: StableRecordKey

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise MD03CenterAdapterError("directional profile direction 非法")
        for name in (
                "obligation_kind_key", "relation_profile_key",
                "expansion_profile_key"):
            if not isinstance(getattr(self, name), StableRecordKey):
                raise TypeError(f"directional profile {name} 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回方向和三类注入 profile 完整键。"""
        return (
            FORMAT_VERSION,
            DIRECTIONS.index(self.direction) + 1,
            *_packed(self.obligation_kind_key.stable_key()),
            *_packed(self.relation_profile_key.stable_key()),
            *_packed(self.expansion_profile_key.stable_key()),
        )


@dataclass(frozen=True)
class DirectionalCenterAdapterConfig:
    """列全 Understanding/Reasoning/Generation 的注入 profile。"""

    profiles: tuple[DirectionalCenterProfile, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.profiles, tuple)
                or any(not isinstance(item, DirectionalCenterProfile)
                       for item in self.profiles)):
            raise TypeError("directional adapter profiles 类型错误")
        directions = tuple(item.direction for item in self.profiles)
        if directions != DIRECTIONS:
            raise MD03CenterAdapterError("directional adapter 必须按冻结顺序列全三向")

    def profile(self, direction: str) -> DirectionalCenterProfile:
        """按精确方向返回唯一 profile。"""
        matches = tuple(item for item in self.profiles
                        if item.direction == direction)
        if len(matches) != 1:
            raise MD03CenterAdapterError("directional profile 不唯一")
        return matches[0]

    def stable_key(self) -> tuple[int, ...]:
        """返回列全三向的配置键。"""
        result = [FORMAT_VERSION, len(self.profiles)]
        for profile in self.profiles:
            result.extend(_packed(profile.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class DirectionalWriteBoundary:
    """一个方向未来可写/禁止写范围及本 adapter 的实际零写事实。"""

    direction: str
    allowed_write_kinds: tuple[str, ...]
    forbidden_write_kinds: tuple[str, ...]
    activation_authorizes_adoption: int
    host_learning_write_count: int

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise MD03CenterAdapterError("write boundary direction 非法")
        expected_allowed = tuple(sorted(_ALLOWED_WRITES[self.direction]))
        expected_forbidden = tuple(sorted({
            *_COMMON_FORBIDDEN_WRITES,
            *_DIRECTION_FORBIDDEN_WRITES[self.direction],
        }))
        if self.allowed_write_kinds != expected_allowed:
            raise MD03CenterAdapterError("write boundary allowed kinds 未精确列全")
        if self.forbidden_write_kinds != expected_forbidden:
            raise MD03CenterAdapterError("write boundary forbidden kinds 未精确列全")
        if set(self.allowed_write_kinds) & set(self.forbidden_write_kinds):
            raise MD03CenterAdapterError("write boundary 允许与禁止范围重叠")
        if self.activation_authorizes_adoption != 0:
            raise MD03CenterAdapterError("activation 不得授权 adoption")
        if self.host_learning_write_count != 0:
            raise MD03CenterAdapterError("center adapter 不得产生 host learning write")

    @classmethod
    def for_direction(cls, direction: str) -> "DirectionalWriteBoundary":
        """建立冻结的方向权限边界，实际写数恒为零。"""
        if direction not in DIRECTIONS:
            raise MD03CenterAdapterError("write boundary direction 非法")
        return cls(
            direction,
            tuple(sorted(_ALLOWED_WRITES[direction])),
            tuple(sorted({
                *_COMMON_FORBIDDEN_WRITES,
                *_DIRECTION_FORBIDDEN_WRITES[direction],
            })),
            0,
            0,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回方向、权限集合和零写事实。"""
        result = [FORMAT_VERSION, DIRECTIONS.index(self.direction) + 1]
        for values in (self.allowed_write_kinds, self.forbidden_write_kinds):
            result.append(len(values))
            for value in values:
                result.extend(_packed(_text_key(value)))
        result.extend((
            self.activation_authorizes_adoption,
            self.host_learning_write_count,
        ))
        return tuple(result)


@dataclass(frozen=True)
class DirectionalMemoryCenter:
    """一个 MD-01 center 及其方向 payload、采用条件和写权限。"""

    center: MemoryAttentionCenter
    input_query_key: StableRecordKey
    payload_kind: str
    payload_key: StableRecordKey
    adoption_condition_keys: tuple[StableRecordKey, ...]
    write_boundary: DirectionalWriteBoundary

    def __post_init__(self) -> None:
        if not isinstance(self.center, MemoryAttentionCenter):
            raise TypeError("directional center 类型错误")
        if not isinstance(self.input_query_key, StableRecordKey):
            raise TypeError("directional input query key 类型错误")
        if self.payload_kind not in PAYLOAD_KINDS:
            raise MD03CenterAdapterError("directional payload kind 非法")
        if self.payload_kind != _PAYLOAD_BY_DIRECTION[self.center.direction]:
            raise MD03CenterAdapterError("direction 与 payload kind 不一致")
        if not isinstance(self.payload_key, StableRecordKey):
            raise TypeError("directional payload key 类型错误")
        object.__setattr__(self, "adoption_condition_keys", _normalized_keys(
            self.adoption_condition_keys, where="adoption conditions"))
        if not isinstance(self.write_boundary, DirectionalWriteBoundary):
            raise TypeError("directional write boundary 类型错误")
        if self.write_boundary.direction != self.center.direction:
            raise MD03CenterAdapterError("center 与 write boundary 方向不一致")
        if self.center.activation_only != 1:
            raise MD03CenterAdapterError("directional center 必须 activation-only")

    def dedup_key(self) -> tuple[int, ...]:
        """返回严格去重键；不含可合并的 origin/dependency 和 center_key。"""
        center = self.center
        result = [
            FORMAT_VERSION,
            DIRECTIONS.index(center.direction) + 1,
            CENTER_STRENGTHS.index(center.strength) + 1,
            *_packed(center.obligation_kind_key.stable_key()),
            *_packed(center.target_key.stable_key()),
        ]
        for key in (
                center.boundary.owner_key,
                center.boundary.scope_key,
                center.boundary.source_key,
                center.boundary.version_key,
                center.relation_profile_key,
                center.expansion_profile_key,
                self.input_query_key,
                self.payload_key):
            result.extend(_packed(key.stable_key()))
        result.extend(_packed(_text_key(self.payload_kind)))
        result.append(len(self.adoption_condition_keys))
        for key in self.adoption_condition_keys:
            result.extend(_packed(key.stable_key()))
        result.extend(_packed(self.write_boundary.stable_key()))
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回 center 完整身份、方向 payload 和权限边界。"""
        return (
            FORMAT_VERSION,
            *_packed(self.center.center_key.stable_key()),
            *_packed(_center_values(
                direction=self.center.direction,
                strength=self.center.strength,
                origins=self.center.origins,
                obligation_kind_key=self.center.obligation_kind_key,
                target_key=self.center.target_key,
                boundary=self.center.boundary,
                relation_profile_key=self.center.relation_profile_key,
                expansion_profile_key=self.center.expansion_profile_key,
                dependency_keys=self.center.dependency_keys,
            )),
            *_packed(self.dedup_key()),
        )


@dataclass(frozen=True)
class MemoryCenterFormationReport:
    """三向形成和精确去重的只读报告。"""

    input_center_count: int
    centers: tuple[DirectionalMemoryCenter, ...]
    merged_duplicate_count: int
    host_learning_write_count: int

    def __post_init__(self) -> None:
        if (type(self.input_center_count) is not int
                or self.input_center_count <= 0):
            raise MD03CenterAdapterError("formation input count 非法")
        if (not isinstance(self.centers, tuple) or not self.centers
                or any(not isinstance(item, DirectionalMemoryCenter)
                       for item in self.centers)):
            raise TypeError("formation centers 类型错误")
        keys = tuple(item.dedup_key() for item in self.centers)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise MD03CenterAdapterError("formation envelope keys 未稳定去重")
        if (type(self.merged_duplicate_count) is not int
                or self.merged_duplicate_count < 0
                or len(self.centers) + self.merged_duplicate_count
                != self.input_center_count):
            raise MD03CenterAdapterError("formation 去重计数不闭合")
        if self.host_learning_write_count != 0:
            raise MD03CenterAdapterError("formation 不得产生 host learning write")

    def stable_key(self) -> tuple[int, ...]:
        """返回输入数、去重后 center 和零写事实。"""
        result = [
            FORMAT_VERSION,
            self.input_center_count,
            self.merged_duplicate_count,
            self.host_learning_write_count,
            len(self.centers),
        ]
        for center in self.centers:
            result.extend(_packed(center.stable_key()))
        return tuple(result)


class DirectionalMemoryCenterAdapter:
    """从真实三向输入形成 center envelope，不接召回、consumer 或 committer。"""

    def __init__(self, config: DirectionalCenterAdapterConfig) -> None:
        if not isinstance(config, DirectionalCenterAdapterConfig):
            raise TypeError("directional center adapter config 类型错误")
        self.config = config

    @staticmethod
    def _validate_current(current: MemoryCurrentQuery) -> None:
        """要求 typed 当前输入保持自身 source/scope/owner/version 合同。"""
        if not isinstance(current, MemoryCurrentQuery):
            raise TypeError("directional center current 类型错误")

    @staticmethod
    def _validate_target(
            current: MemoryCurrentQuery,
            target: ObjectIdentity | TypedRef,
            ) -> None:
        """理解目标必须来自当前 typed 输入且 owner/version 完全一致。"""
        if not isinstance(target, (ObjectIdentity, TypedRef)):
            raise TypeError("understanding target 类型错误")
        anchors = {
            *current.occurrences,
            *current.spans,
            *current.semantic_objects,
            *current.structures,
            current.domain,
            current.intent,
        }
        if current.speaker is not None:
            anchors.add(current.speaker)
        if current.task is not None:
            anchors.add(current.task)
        if target not in anchors:
            raise MD03CenterAdapterError("understanding target 不属于当前 typed 输入")
        if (target.owner != current.source.owner
                or target.versions != current.source.versions):
            raise MD03CenterAdapterError("understanding target owner/version 越权")

    def _envelope(
            self,
            current: MemoryCurrentQuery,
            *,
            direction: str,
            strength: str,
            origins: tuple[MemoryCenterOrigin, ...],
            target_key: StableRecordKey,
            payload_key: StableRecordKey,
            adoption_condition_keys: tuple[StableRecordKey, ...],
            dependency_keys: tuple[StableRecordKey, ...],
            ) -> DirectionalMemoryCenter:
        """按方向 profile 建立零写 envelope。"""
        profile = self.config.profile(direction)
        center = _build_center(
            direction=direction,
            strength=strength,
            origins=origins,
            obligation_kind_key=profile.obligation_kind_key,
            target_key=target_key,
            boundary=_boundary(current),
            relation_profile_key=profile.relation_profile_key,
            expansion_profile_key=profile.expansion_profile_key,
            dependency_keys=dependency_keys,
        )
        return DirectionalMemoryCenter(
            center,
            _record_key(current.stable_key(), domain=20),
            _PAYLOAD_BY_DIRECTION[direction],
            payload_key,
            adoption_condition_keys,
            DirectionalWriteBoundary.for_direction(direction),
        )

    def from_understanding(
            self,
            current: MemoryCurrentQuery,
            target: ObjectIdentity | TypedRef,
            *,
            strength: str,
            ) -> DirectionalMemoryCenter:
        """由当前 typed anchor 形成解释中心，不改写 Observation。"""
        self._validate_current(current)
        self._validate_target(current, target)
        target_key = _record_key(target.stable_key(), domain=21)
        query_key = _record_key(current.stable_key(), domain=20)
        if isinstance(target, TypedRef) and target.object_kind == OBJECT_OCCURRENCE:
            origin_kind = "OCCURRENCE"
        elif isinstance(target, TypedRef) and target.object_kind == OBJECT_SPAN:
            origin_kind = "SPAN"
        elif target.object_kind == OBJECT_PROPOSITION:
            origin_kind = "PROPOSITION"
        else:
            origin_kind = "QUERY"
        origin = MemoryCenterOrigin(
            origin_kind,
            target_key,
            tuple(sorted((query_key, target_key))),
        )
        conditions = {
            _record_key(current.intent.stable_key(), domain=22),
            _record_key(current.domain.stable_key(), domain=23),
        }
        if current.task is not None:
            conditions.add(_record_key(current.task.stable_key(), domain=24))
        return self._envelope(
            current,
            direction="UNDERSTANDING",
            strength=strength,
            origins=(origin,),
            target_key=target_key,
            payload_key=query_key,
            adoption_condition_keys=tuple(sorted(conditions)),
            dependency_keys=tuple(sorted((query_key, target_key))),
        )

    def from_reasoning(
            self,
            current: MemoryCurrentQuery,
            obligation: ReasoningObligation,
            *,
            strength: str,
            ) -> DirectionalMemoryCenter:
        """由证据方向 obligation 形成推理中心，不产生 definitive 事实。"""
        self._validate_current(current)
        if not isinstance(obligation, ReasoningObligation):
            raise TypeError("reasoning obligation 类型错误")
        if (obligation.source != current.source
                or obligation.scope != current.scope):
            raise MD03CenterAdapterError("reasoning obligation source/scope 越权")
        query_key = _record_key(current.stable_key(), domain=20)
        payload_key = _record_key(obligation.stable_key(), domain=31)
        target_key = _record_key(
            obligation.proposition.stable_key(), domain=32)
        required_key = _record_key(
            obligation.required.stable_key(), domain=33)
        origin = MemoryCenterOrigin(
            "PROPOSITION",
            target_key,
            tuple(sorted((query_key, payload_key, required_key))),
        )
        return self._envelope(
            current,
            direction="REASONING",
            strength=strength,
            origins=(origin,),
            target_key=target_key,
            payload_key=payload_key,
            adoption_condition_keys=(required_key,),
            dependency_keys=tuple(sorted((
                query_key, payload_key, target_key, required_key))),
        )

    def from_generation(
            self,
            current: MemoryCurrentQuery,
            goal: AnswerGenerationGoal,
            *,
            strength: str,
            ) -> DirectionalMemoryCenter:
        """由回答 goal 形成生成中心，不形成 Use 或 postcheck PASS。"""
        self._validate_current(current)
        if not isinstance(goal, AnswerGenerationGoal):
            raise TypeError("generation goal 类型错误")
        if goal.source != current.source or goal.scope != current.scope:
            raise MD03CenterAdapterError("generation goal source/scope 越权")
        query_key = _record_key(current.stable_key(), domain=20)
        payload_key = _record_key(goal.stable_key(), domain=41)
        target_key = _record_key(goal.proposition.stable_key(), domain=32)
        required_key = _record_key(goal.required.stable_key(), domain=43)
        goal_kind_key = _record_key(goal.goal_kind.stable_key(), domain=44)
        conditions = {required_key, goal_kind_key}
        if goal.target_branch is not None:
            conditions.add(_record_key(
                goal.target_branch.stable_key(), domain=45))
        origin = MemoryCenterOrigin(
            "GOAL",
            payload_key,
            tuple(sorted((query_key, payload_key, target_key))),
        )
        return self._envelope(
            current,
            direction="GENERATION",
            strength=strength,
            origins=(origin,),
            target_key=target_key,
            payload_key=payload_key,
            adoption_condition_keys=tuple(sorted(conditions)),
            dependency_keys=tuple(sorted((
                query_key, payload_key, target_key, *conditions))),
        )

    def form_triplet(
            self,
            current: MemoryCurrentQuery,
            understanding_target: ObjectIdentity | TypedRef,
            reasoning: ReasoningObligation,
            generation: AnswerGenerationGoal,
            *,
            understanding_strength: str,
            reasoning_strength: str,
            generation_strength: str,
            ) -> MemoryCenterFormationReport:
        """对同一 query 形成三向不同 payload，并执行严格去重。"""
        if (not isinstance(reasoning, ReasoningObligation)
                or not isinstance(generation, AnswerGenerationGoal)):
            raise TypeError("triplet reasoning/generation 类型错误")
        if reasoning.proposition != generation.proposition:
            raise MD03CenterAdapterError("triplet 推理与生成目标 Proposition 不一致")
        return self.deduplicate((
            self.from_generation(
                current, generation, strength=generation_strength),
            self.from_reasoning(
                current, reasoning, strength=reasoning_strength),
            self.from_understanding(
                current, understanding_target, strength=understanding_strength),
        ))

    @staticmethod
    def deduplicate(
            centers: tuple[DirectionalMemoryCenter, ...],
            ) -> MemoryCenterFormationReport:
        """只合并完整调度/采用/权限键相同项，并保留全部 origin/dependency。"""
        if (not isinstance(centers, tuple) or not centers
                or any(not isinstance(item, DirectionalMemoryCenter)
                       for item in centers)):
            raise TypeError("deduplicate centers 类型错误")
        groups: dict[tuple[int, ...], list[DirectionalMemoryCenter]] = {}
        for center in centers:
            groups.setdefault(center.dedup_key(), []).append(center)
        merged = []
        for key in sorted(groups):
            group = groups[key]
            first = group[0]
            origins = tuple({
                origin for item in group for origin in item.center.origins})
            dependencies = tuple({
                dependency
                for item in group for dependency in item.center.dependency_keys})
            rebuilt = _build_center(
                direction=first.center.direction,
                strength=first.center.strength,
                origins=origins,
                obligation_kind_key=first.center.obligation_kind_key,
                target_key=first.center.target_key,
                boundary=first.center.boundary,
                relation_profile_key=first.center.relation_profile_key,
                expansion_profile_key=first.center.expansion_profile_key,
                dependency_keys=dependencies,
            )
            merged.append(DirectionalMemoryCenter(
                rebuilt,
                first.input_query_key,
                first.payload_kind,
                first.payload_key,
                first.adoption_condition_keys,
                first.write_boundary,
            ))
        ordered = tuple(sorted(merged, key=lambda item: item.dedup_key()))
        return MemoryCenterFormationReport(
            len(centers), ordered, len(centers) - len(ordered), 0)


__all__ = [
    "DirectionalCenterAdapterConfig",
    "DirectionalCenterProfile",
    "DirectionalMemoryCenter",
    "DirectionalMemoryCenterAdapter",
    "DirectionalWriteBoundary",
    "FORMAT_VERSION",
    "MD03CenterAdapterError",
    "MemoryCenterFormationReport",
    "PAYLOAD_KINDS",
]
