"""W-09 34-pack registry 与九载体共享 typed engine 审计。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_w09_authority import W09_CARRIER_KEYS
from pure_integer_ai.experiments.ph2_w09_contract import W09FrozenContract
from pure_integer_ai.experiments.ph2_w09_firewall import W09TrainingPayload


W09_SHARED_TYPED_ENGINE_KEY = "PH2_W09_SHARED_TYPED_CARRIER_ENGINE_V1"


class W09RegistryError(RuntimeError):
    """W-09 pack 归属、Evidence 覆盖或 carrier engine 发生漂移。"""


@dataclass(frozen=True)
class W09PackRegistryEntry:
    pack_key: str
    candidate_train: int
    dev_calibration: int
    held_out_evaluator: int

    def __post_init__(self) -> None:
        if not isinstance(self.pack_key, str) or not self.pack_key:
            raise W09RegistryError("W-09 pack registry key 非法")
        if any(value not in {0, 1} for value in (
            self.candidate_train,
            self.dev_calibration,
            self.held_out_evaluator,
        )):
            raise W09RegistryError("W-09 pack registry flag 非法")
        if not any((
            self.candidate_train,
            self.dev_calibration,
            self.held_out_evaluator,
        )):
            raise W09RegistryError("W-09 pack registry entry 无可见相位")


@dataclass(frozen=True)
class W09CarrierBinding:
    carrier_key: str
    carrier_adapter_key: str
    semantic_engine_key: str

    def __post_init__(self) -> None:
        if self.carrier_key not in W09_CARRIER_KEYS:
            raise W09RegistryError("W-09 carrier key 非法")
        if not isinstance(self.carrier_adapter_key, str) or not self.carrier_adapter_key:
            raise W09RegistryError("W-09 carrier adapter key 非法")
        if self.semantic_engine_key != W09_SHARED_TYPED_ENGINE_KEY:
            raise W09RegistryError("W-09 carrier 复制了专用语义内核")


@dataclass(frozen=True)
class W09Registry:
    pack_entries: tuple[W09PackRegistryEntry, ...]
    carrier_bindings: tuple[W09CarrierBinding, ...]

    def __post_init__(self) -> None:
        if len(self.pack_entries) != 37:
            raise W09RegistryError("W-09 pack registry 必须有 37 项")
        if len({item.pack_key for item in self.pack_entries}) != 37:
            raise W09RegistryError("W-09 pack registry key 重复")
        if tuple(item.carrier_key for item in self.carrier_bindings) != W09_CARRIER_KEYS:
            raise W09RegistryError("W-09 carrier registry 不完整")
        if len({item.carrier_adapter_key for item in self.carrier_bindings}) != 9:
            raise W09RegistryError("W-09 carrier adapter identity 必须独立")
        if {item.semantic_engine_key for item in self.carrier_bindings} != {
            W09_SHARED_TYPED_ENGINE_KEY
        }:
            raise W09RegistryError("W-09 carrier 未共享唯一 typed engine")


def build_w09_registry(context: W09FrozenContract) -> W09Registry:
    if not isinstance(context, W09FrozenContract):
        raise W09RegistryError("W-09 registry requires frozen contract")
    candidate = set(context.candidate_pack_keys)
    dev = set(context.dev_pack_keys)
    held_out = set(context.held_out_pack_keys)
    entries = tuple(
        W09PackRegistryEntry(
            pack_key,
            int(pack_key in candidate),
            int(pack_key in dev),
            int(pack_key in held_out),
        )
        for pack_key in context.held_out_pack_keys
    )
    carriers = tuple(
        W09CarrierBinding(
            carrier_key,
            f"PH2_W09_{carrier_key}_ADAPTER_V1",
            W09_SHARED_TYPED_ENGINE_KEY,
        )
        for carrier_key in W09_CARRIER_KEYS
    )
    return W09Registry(entries, carriers)


@dataclass(frozen=True)
class W09RegistryAuditReport:
    pack_counts: tuple[tuple[str, int], ...]
    source_ref_count: int
    observation_count: int
    training_evidence_count: int
    expected_or_label_field_count: int
    shared_typed_engine_count: int


def _binding_for_artifact(artifact_key, bindings):
    matches = tuple(
        item
        for item in bindings
        if item.artifact_key == artifact_key
    )
    if len(matches) != 1:
        raise W09RegistryError("W-09 record 不属于唯一 pack identity range")
    return matches[0]


def _contains_label_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("expected", "label", "evaluator")):
                return True
            if _contains_label_key(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_label_key(item) for item in value)
    return False


def audit_w09_registry_payload(
    payload: W09TrainingPayload,
    context: W09FrozenContract,
) -> W09RegistryAuditReport:
    """按 manifest key range 核验 34 pack、Evidence 覆盖和零 label field。"""
    if not isinstance(payload, W09TrainingPayload) or not isinstance(
        context, W09FrozenContract
    ):
        raise W09RegistryError("W-09 registry audit input 类型非法")
    registry = build_w09_registry(context)
    observation_bindings = tuple(
        item
        for item in context.candidate_bindings
        if item.identity.owner_kind == "observation"
    )
    evidence_bindings = context.training_material_bindings
    counts = {key: 0 for key in context.candidate_pack_keys}
    observation_by_key = {item.stable_key: item for item in payload.observations}
    source_keys = {item.stable_key for item in payload.source_refs}
    if (
        len(observation_by_key) != len(payload.observations)
        or len(source_keys) != len(payload.source_refs)
    ):
        raise W09RegistryError("W-09 train stable key 重复")
    label_fields = 0
    for observation in payload.observations:
        binding = _binding_for_artifact(observation.artifact_key, observation_bindings)
        if observation.source_ref_key not in source_keys or observation.split != "train":
            raise W09RegistryError("W-09 Observation owner/split/source 不闭合")
        counts[binding.pack_key] += 1
        label_fields += int(_contains_label_key(observation.typed_payload.to_value()))
    for binding in observation_bindings:
        if counts[binding.pack_key] != binding.identity.record_count:
            raise W09RegistryError("W-09 pack Observation count 与 manifest 漂移")
    seen_evidence = set()
    for evidence in payload.training_evidence:
        binding = _binding_for_artifact(evidence.artifact_key, evidence_bindings)
        observation = observation_by_key.get(evidence.observation_key)
        if (
            evidence.stable_key in seen_evidence
            or observation is None
            or evidence.source_ref_key != observation.source_ref_key
        ):
            raise W09RegistryError("W-09 training Evidence 引用不闭合")
        seen_evidence.add(evidence.stable_key)
        if binding.pack_key != _binding_for_artifact(
            observation.artifact_key, observation_bindings
        ).pack_key:
            raise W09RegistryError("W-09 Evidence 跨 pack 绑定")
    if len(payload.training_evidence) != len(payload.observations):
        raise W09RegistryError("W-09 training Evidence 覆盖不完整")
    if any(count == 0 for count in counts.values()):
        raise W09RegistryError("W-09 candidate registry 含空 train pack")
    if label_fields:
        raise W09RegistryError("W-09 candidate payload 含 expected/label/evaluator 字段")
    return W09RegistryAuditReport(
        tuple((key, counts[key]) for key in context.candidate_pack_keys),
        len(payload.source_refs),
        len(payload.observations),
        len(payload.training_evidence),
        label_fields,
        len({item.semantic_engine_key for item in registry.carrier_bindings}),
    )


__all__ = [
    "W09_SHARED_TYPED_ENGINE_KEY",
    "W09CarrierBinding",
    "W09PackRegistryEntry",
    "W09Registry",
    "W09RegistryAuditReport",
    "W09RegistryError",
    "audit_w09_registry_payload",
    "build_w09_registry",
]
