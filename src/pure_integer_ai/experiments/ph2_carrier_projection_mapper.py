"""LC-16 九类 carrier-local Observation 到共享 projection 输入的通用 mapper。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.cognition.shared.artifact_envelope import (
    ArtifactAnchor,
    ArtifactEnvelope,
    ArtifactStructureNode,
)
from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    SourceRef,
    concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.experiments.ph2_carrier_projection_mapper_contract import (
    CarrierProjectionRule,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    StableRecordKey,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_contract import (
    TypedCarrierPackManifest,
)


class CarrierProjectionMapperError(RuntimeError):
    """carrier identity、receipt selector 或共享 feature 映射失败。"""


def _pack(values: tuple[int, ...]) -> tuple[int, ...]:
    return (len(values), *values)


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int or item <= 0 for item in value)):
        raise CarrierProjectionMapperError(f"{where} 必须是正严格整数 tuple")
    return value


def _select_value(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for component in path:
        if not isinstance(current, dict) or component not in current:
            raise CarrierProjectionMapperError(
                f"receipt selector path 缺失: {'/'.join(path)}")
        current = current[component]
    return current


@dataclass(frozen=True)
class MappedCarrierFeature:
    """一项 carrier-local 对象经 data-only 规则得到的共享结构特征。"""

    rule: CarrierProjectionRule
    item_identity: ObjectIdentity
    feature_identity: ObjectIdentity
    selected_values: CanonicalJsonObject

    def __post_init__(self) -> None:
        if not isinstance(self.rule, CarrierProjectionRule):
            raise CarrierProjectionMapperError("mapped feature rule 类型非法")
        if not isinstance(self.item_identity, ObjectIdentity):
            raise CarrierProjectionMapperError("mapped feature item identity 非法")
        if self.feature_identity != concept_identity(
                self.rule.feature_key.stable_key()):
            raise CarrierProjectionMapperError("mapped feature identity 漂移")
        if self.selected_values != self.rule.expected_values:
            raise CarrierProjectionMapperError("mapped feature selected values 漂移")

    def stable_key(self) -> tuple[int, ...]:
        rule_key = self.rule.rule_key.stable_key()
        item_key = self.item_identity.stable_key()
        feature_key = self.feature_identity.stable_key()
        selected = tuple(self.selected_values.payload)
        return (
            1,
            *_pack(rule_key),
            *_pack(item_key),
            *_pack(feature_key),
            *_pack(selected),
        )


@dataclass(frozen=True)
class CarrierProjectionInput:
    """供共享 semantic projection 生命周期消费的 carrier-neutral 输入。"""

    carrier_key: str
    case_key: StableRecordKey
    source: SourceRef
    scope: ScopeIdentity
    envelope: ArtifactEnvelope
    anchor_identities: tuple[ObjectIdentity, ...]
    structure_node_identities: tuple[ObjectIdentity, ...]
    features: tuple[MappedCarrierFeature, ...]
    input_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.case_key, StableRecordKey):
            raise CarrierProjectionMapperError("projection input case_key 非法")
        if not isinstance(self.source, SourceRef):
            raise CarrierProjectionMapperError("projection input source 非法")
        if not isinstance(self.scope, ScopeIdentity):
            raise CarrierProjectionMapperError("projection input scope 非法")
        if not isinstance(self.envelope, ArtifactEnvelope):
            raise CarrierProjectionMapperError("projection input envelope 非法")
        if (self.envelope.source != self.source
                or self.envelope.scope != self.scope):
            raise CarrierProjectionMapperError("projection input envelope context 漂移")
        for name in ("anchor_identities", "structure_node_identities"):
            values = getattr(self, name)
            if (not isinstance(values, tuple)
                    or values != tuple(sorted(
                        set(values), key=ObjectIdentity.stable_key))):
                raise CarrierProjectionMapperError(
                    f"projection input {name} 必须排序去重")
        if not self.anchor_identities and not self.structure_node_identities:
            raise CarrierProjectionMapperError("projection input local objects 不能为空")
        if (not isinstance(self.features, tuple) or not self.features
                or any(not isinstance(item, MappedCarrierFeature)
                       for item in self.features)):
            raise CarrierProjectionMapperError("projection input features 非法")
        local_ids = set(self.anchor_identities) | set(self.structure_node_identities)
        if any(item.item_identity not in local_ids for item in self.features):
            raise CarrierProjectionMapperError("mapped feature 未绑定选中 local object")
        if any(item.rule.carrier_key != self.carrier_key for item in self.features):
            raise CarrierProjectionMapperError("mapped feature carrier 漂移")
        _strict_key(self.input_key, where="projection input_key")

    @property
    def feature_identities(self) -> tuple[ObjectIdentity, ...]:
        return tuple(sorted(
            {item.feature_identity for item in self.features},
            key=ObjectIdentity.stable_key,
        ))

    @property
    def visible_inputs(self) -> tuple[ObjectIdentity, ...]:
        return tuple(sorted(
            set(self.anchor_identities)
            | set(self.structure_node_identities)
            | set(self.feature_identities),
            key=ObjectIdentity.stable_key,
        ))

    def stable_key(self) -> tuple[int, ...]:
        result = [
            1,
            *_pack(tuple(self.carrier_key.encode("utf-8"))),
            *_pack(self.case_key.stable_key()),
            *_pack(self.source.stable_key()),
            *_pack(self.scope.stable_key()),
            *_pack(self.envelope.stable_key()),
            len(self.anchor_identities),
        ]
        for item in self.anchor_identities:
            result.extend(_pack(item.stable_key()))
        result.append(len(self.structure_node_identities))
        for item in self.structure_node_identities:
            result.extend(_pack(item.stable_key()))
        result.append(len(self.features))
        for item in self.features:
            result.extend(_pack(item.stable_key()))
        result.extend(_pack(self.input_key))
        return tuple(result)


class CarrierProjectionMapper:
    """只按冻结 parent identity 与数据规则映射，不含 carrier 类型分支。"""

    def __init__(self, parent: TypedCarrierPackManifest) -> None:
        if not isinstance(parent, TypedCarrierPackManifest):
            raise TypeError("projection mapper parent 类型非法")
        self.parent = parent
        self._carrier_by_case = {
            item.case_key: item.carrier_key for item in parent.cases
        }

    def map(
            self,
            carrier_key: str,
            materialization: Any,
            rule: CarrierProjectionRule,
            *,
            item_indices: tuple[int, ...],
            input_key: tuple[int, ...],
            envelope_index: int = 0,
            ) -> CarrierProjectionInput:
        """匹配 selector 后建立共享输入；未知结构通过新增规则数据进入。"""
        if not isinstance(rule, CarrierProjectionRule):
            raise TypeError("projection mapper rule 类型非法")
        if rule.carrier_key != carrier_key:
            raise CarrierProjectionMapperError("rule 与请求 carrier 不一致")
        try:
            record = materialization.record
            case_key = record.case_key
            sources = materialization.sources
            scopes = materialization.scopes
            envelopes = materialization.envelopes
            anchors = materialization.anchors
        except AttributeError as error:
            raise CarrierProjectionMapperError("materialization 缺少共享只读字段") from error
        if self._carrier_by_case.get(case_key) != carrier_key:
            raise CarrierProjectionMapperError("materialization case 与 parent carrier 不一致")
        if (type(envelope_index) is not int
                or not 0 <= envelope_index < len(envelopes)):
            raise CarrierProjectionMapperError("envelope_index 越界")
        if (not isinstance(item_indices, tuple) or not item_indices
                or any(type(item) is not int or item < 0 for item in item_indices)
                or item_indices != tuple(sorted(set(item_indices)))):
            raise CarrierProjectionMapperError("item_indices 必须排序去重")
        source = sources[envelope_index]
        scope = scopes[envelope_index]
        envelope = envelopes[envelope_index]
        if rule.input_kind == "ANCHOR":
            collection = anchors
            expected_type = ArtifactAnchor
        else:
            collection = getattr(materialization, "structure_nodes", ())
            expected_type = ArtifactStructureNode
        try:
            items = tuple(collection[index] for index in item_indices)
        except (IndexError, TypeError) as error:
            raise CarrierProjectionMapperError("item_indices 越界") from error
        if any(not isinstance(item, expected_type) for item in items):
            raise CarrierProjectionMapperError("selected item 类型漂移")
        if any(item.envelope_identity != envelope.identity for item in items):
            raise CarrierProjectionMapperError("selected item 跨 envelope")
        mapped = []
        for item in items:
            if isinstance(item, ArtifactAnchor):
                receipt = {"anchor_kind": item.anchor_kind}
            else:
                try:
                    receipt = parse_canonical_json_bytes(
                        bytes(item.qualifiers), require_object=True)
                except Exception as error:
                    raise CarrierProjectionMapperError(
                        "structure node receipt 不是 canonical JSON") from error
            selected = CanonicalJsonObject.from_value({
                "values": [
                    _select_value(receipt, path)
                    for path in rule.selector_paths
                ],
            })
            if selected != rule.expected_values:
                raise CarrierProjectionMapperError("receipt 与 data-only rule 不匹配")
            mapped.append(MappedCarrierFeature(
                rule,
                item.identity,
                concept_identity(rule.feature_key.stable_key()),
                selected,
            ))
        if rule.input_kind == "ANCHOR":
            anchor_ids = tuple(sorted(
                {item.identity for item in items},
                key=ObjectIdentity.stable_key,
            ))
            node_ids: tuple[ObjectIdentity, ...] = ()
        else:
            anchor_ids = tuple(sorted(
                {item.anchor_identity for item in items},
                key=ObjectIdentity.stable_key,
            ))
            node_ids = tuple(sorted(
                {item.identity for item in items},
                key=ObjectIdentity.stable_key,
            ))
        return CarrierProjectionInput(
            carrier_key,
            case_key,
            source,
            scope,
            envelope,
            anchor_ids,
            node_ids,
            tuple(mapped),
            _strict_key(input_key, where="projection input_key"),
        )


__all__ = [
    "CarrierProjectionInput", "CarrierProjectionMapper",
    "CarrierProjectionMapperError", "MappedCarrierFeature",
]
