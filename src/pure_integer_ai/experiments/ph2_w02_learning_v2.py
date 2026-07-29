"""W-02 v2：把生产性形态目标绑定到当前 typed Observation Evidence。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from pure_integer_ai.experiments.ph2_authored_morphology_course import (
    PAYLOAD_KIND,
    AuthoredMorphologyCourseError,
    validate_morphology_payload,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w02_learning import (
    GENERATION_CONFLICT,
    GENERATION_GENERATED,
    GENERATION_UNKNOWN,
    W02GenerationResult,
    W02LearningError,
    W02LearningRuntime,
    open_w02_learning_runtime,
)
from pure_integer_ai.storage.backend import StorageBackend


W02_MORPHOLOGY_ADAPTER_VERSION = "PH2-W02-evidence-bound-morphology-v2"
_EVIDENCE_VERSION = 2
_EVIDENCE_PERMIT = object()
_TARGET_PERMIT = object()
_TARGET_UNIT_KINDS = ("STEM", "COMPONENT")
_FAMILY_RELATIONS = (
    "ATTACHES_AFFIX",
    "COMPOUND_COMPONENT",
    "EXCEPTION_TO",
    "REDUPLICATES",
)


def _text_key(value: str) -> tuple[int, ...]:
    """把 Evidence 中的开放文本无损编码为稳定整数键。"""
    if not isinstance(value, str) or not value:
        raise W02LearningError("W-02 v2 文本键必须非空")
    payload = value.encode("utf-8")
    return _EVIDENCE_VERSION, len(payload), *payload


def _record_key(value) -> tuple[int, ...]:
    """复制统一资料整数键，避免 Evidence 持有可变外部对象。"""
    components = getattr(value, "components", None)
    if not isinstance(components, tuple) or not components:
        raise W02LearningError("W-02 v2 Observation 整数键非法")
    if any(type(item) is not int or item < 0 for item in components):
        raise W02LearningError("W-02 v2 Observation 整数键分量非法")
    return components


def _payload_sha256(observation: ObservationRecord) -> str:
    """摘要当前 typed payload，使 unit Evidence 与被验证字节不可分离。"""
    return hashlib.sha256(canonical_json_bytes(
        observation.typed_payload.to_value())).hexdigest()


def _validated_observation(
        observation: ObservationRecord,
        ) -> dict[str, Any]:
    """验证当前 Observation 的阶段、语言和 LC-02 typed schema。"""
    if not isinstance(observation, ObservationRecord):
        raise TypeError("W-02 v2 target Evidence 必须来自 ObservationRecord")
    if (observation.w_stage != "W-02"
            or observation.payload_kind != PAYLOAD_KIND
            or observation.language != "zh"
            or observation.split not in {"train", "held_out"}):
        raise W02LearningError("W-02 v2 target Observation 作用域非法")
    try:
        validate_morphology_payload(observation.typed_payload)
    except (AuthoredMorphologyCourseError, KeyError, TypeError, ValueError) as exc:
        raise W02LearningError(
            "W-02 v2 target Observation morphology schema 损坏") from exc
    return observation.typed_payload.to_value()


@dataclass(frozen=True, init=False)
class W02MorphologyUnitEvidenceV2:
    """一个由当前 Observation 的精确 span 和关系角色承载的表层单位。"""

    dataset_key: tuple[int, ...]
    artifact_key: tuple[int, ...]
    source_ref_key: tuple[int, ...]
    observation_key: tuple[int, ...]
    split: str
    payload_sha256: str
    unit_id: str
    unit_kind: str
    start: int
    end: int
    surface: str

    def __init__(
            self,
            *,
            dataset_key: tuple[int, ...],
            artifact_key: tuple[int, ...],
            source_ref_key: tuple[int, ...],
            observation_key: tuple[int, ...],
            split: str,
            payload_sha256: str,
            unit_id: str,
            unit_kind: str,
            start: int,
            end: int,
            surface: str,
            _permit: object,
            ) -> None:
        if _permit is not _EVIDENCE_PERMIT:
            raise W02LearningError(
                "W-02 v2 unit Evidence 只能由已验证 Observation 构造")
        values = {
            "dataset_key": dataset_key,
            "artifact_key": artifact_key,
            "source_ref_key": source_ref_key,
            "observation_key": observation_key,
            "split": split,
            "payload_sha256": payload_sha256,
            "unit_id": unit_id,
            "unit_kind": unit_kind,
            "start": start,
            "end": end,
            "surface": surface,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def stable_key(self) -> tuple[int, ...]:
        """返回包含来源、payload 和 span 的确定性 Evidence 身份。"""
        payload_digest = bytes.fromhex(self.payload_sha256)
        unit_id = _text_key(self.unit_id)
        unit_kind = _text_key(self.unit_kind)
        surface = _text_key(self.surface)
        split = _text_key(self.split)
        values: list[int] = [_EVIDENCE_VERSION]
        for key in (
                self.dataset_key,
                self.artifact_key,
                self.source_ref_key,
                self.observation_key,
                split,
                unit_id,
                unit_kind,
                surface):
            values.extend((len(key), *key))
        values.extend((len(payload_digest), *payload_digest, self.start, self.end))
        return tuple(values)


def morphology_unit_evidence_v2(
        observation: ObservationRecord,
        unit_id: str,
        ) -> W02MorphologyUnitEvidenceV2:
    """从当前 morphology Observation 提取一个关系承载的 STEM/COMPONENT。"""
    if not isinstance(unit_id, str) or not unit_id:
        raise W02LearningError("W-02 v2 unit_id 必须非空")
    value = _validated_observation(observation)
    units = tuple(
        item for item in value["analysis_units"] if item["unit_id"] == unit_id)
    if len(units) != 1:
        raise W02LearningError("W-02 v2 target unit_id 不唯一或不存在")
    unit = units[0]
    kind = unit["unit_kind"]
    if kind not in _TARGET_UNIT_KINDS:
        raise W02LearningError("W-02 v2 target 只接受 STEM/COMPONENT Evidence")
    relations = value["morphology_relations"]
    if kind == "STEM":
        has_stem = any(
            item["relation_kind"] == "HAS_STEM"
            and item["target_unit_id"] == unit_id
            for item in relations)
        fills_slot = any(
            item["relation_kind"] == "FILLS_SLOT"
            and item["source_unit_id"] == unit_id
            for item in relations)
        if not has_stem or not fills_slot:
            raise W02LearningError(
                "W-02 v2 STEM 未被 HAS_STEM/FILLS_SLOT 双关系承载")
    elif not any(
            item["relation_kind"] == "COMPOUND_COMPONENT"
            and item["target_unit_id"] == unit_id
            for item in relations):
        raise W02LearningError(
            "W-02 v2 COMPONENT 未被 COMPOUND_COMPONENT 承载")
    return W02MorphologyUnitEvidenceV2(
        dataset_key=_record_key(observation.dataset_key),
        artifact_key=_record_key(observation.artifact_key),
        source_ref_key=_record_key(observation.source_ref_key),
        observation_key=_record_key(observation.stable_key),
        split=observation.split,
        payload_sha256=_payload_sha256(observation),
        unit_id=unit["unit_id"],
        unit_kind=kind,
        start=unit["start"],
        end=unit["end"],
        surface=unit["surface"],
        _permit=_EVIDENCE_PERMIT,
    )


@dataclass(frozen=True, init=False)
class W02MorphologyTargetV2:
    """由当前 unit Evidence 和已学 construction key 组成的无答案目标。"""

    construction_key: str
    stem_evidence: W02MorphologyUnitEvidenceV2
    component_evidence: tuple[W02MorphologyUnitEvidenceV2, ...]

    def __init__(
            self,
            construction_key: str,
            stem_evidence: W02MorphologyUnitEvidenceV2,
            component_evidence: tuple[W02MorphologyUnitEvidenceV2, ...],
            *,
            _permit: object,
            ) -> None:
        if _permit is not _TARGET_PERMIT:
            raise W02LearningError(
                "W-02 v2 target 只能由 typed Evidence builder 构造")
        object.__setattr__(self, "construction_key", construction_key)
        object.__setattr__(self, "stem_evidence", stem_evidence)
        object.__setattr__(self, "component_evidence", component_evidence)

    @property
    def stem_surface(self) -> str:
        """返回已绑定 Evidence 的 stem surface。"""
        return self.stem_evidence.surface

    @property
    def component_surfaces(self) -> tuple[str, ...]:
        """按 target Evidence 顺序返回 component surface。"""
        return tuple(item.surface for item in self.component_evidence)

    def stable_key(self) -> tuple[int, ...]:
        """返回不含 expected surface 的来源化请求身份。"""
        construction = _text_key(self.construction_key)
        stem = self.stem_evidence.stable_key()
        values = [
            _EVIDENCE_VERSION,
            len(construction), *construction,
            len(stem), *stem,
            len(self.component_evidence),
        ]
        for component in self.component_evidence:
            key = component.stable_key()
            values.extend((len(key), *key))
        return tuple(values)


def build_w02_morphology_target_v2(
        construction_key: str,
        stem_evidence: W02MorphologyUnitEvidenceV2,
        component_evidence: tuple[W02MorphologyUnitEvidenceV2, ...] = (),
        ) -> W02MorphologyTargetV2:
    """把一个已验证 STEM 与零或多个 COMPONENT 组成 evidence-bound target。"""
    if not isinstance(construction_key, str) or not construction_key:
        raise W02LearningError("W-02 v2 construction_key 必须非空")
    if (not isinstance(stem_evidence, W02MorphologyUnitEvidenceV2)
            or stem_evidence.unit_kind != "STEM"):
        raise W02LearningError("W-02 v2 target 缺合法 STEM Evidence")
    if (not isinstance(component_evidence, tuple)
            or any(not isinstance(item, W02MorphologyUnitEvidenceV2)
                   or item.unit_kind != "COMPONENT"
                   for item in component_evidence)):
        raise W02LearningError("W-02 v2 target COMPONENT Evidence 非法")
    if len({item.stable_key() for item in component_evidence}) != len(
            component_evidence):
        raise W02LearningError("W-02 v2 target COMPONENT Evidence 重复")
    return W02MorphologyTargetV2(
        construction_key,
        stem_evidence,
        component_evidence,
        _permit=_TARGET_PERMIT,
    )


def morphology_target_from_observation_v2(
        observation: ObservationRecord,
        ) -> W02MorphologyTargetV2:
    """从单条当前 Observation 提取唯一 STEM、顺序 COMPONENT 和构式请求。"""
    value = _validated_observation(observation)
    ordered = tuple(sorted(
        value["analysis_units"],
        key=lambda item: (item["start"], item["end"], item["unit_id"]),
    ))
    stems = tuple(item for item in ordered if item["unit_kind"] == "STEM")
    if len(stems) != 1:
        raise W02LearningError("W-02 v2 target Observation 必须有唯一 STEM")
    stem = morphology_unit_evidence_v2(observation, stems[0]["unit_id"])
    components = tuple(
        morphology_unit_evidence_v2(observation, item["unit_id"])
        for item in ordered
        if item["unit_kind"] == "COMPONENT"
    )
    return build_w02_morphology_target_v2(
        value["construction_key"], stem, components)


def _learned_role_units(value: dict[str, Any]) -> tuple[dict[str, Any], ...] | None:
    """验证 learned Candidate 的 construction/stem/slot 拓扑并返回表层角色序。"""
    units = {item["unit_id"]: item for item in value["analysis_units"]}
    constructions = tuple(
        item for item in units.values() if item["unit_kind"] == "CONSTRUCTION")
    stems = tuple(item for item in units.values() if item["unit_kind"] == "STEM")
    if len(constructions) != 1 or len(stems) != 1:
        return None
    construction = constructions[0]
    stem = stems[0]
    relations = value["morphology_relations"]
    if not any(
            item["relation_kind"] == "HAS_STEM"
            and item["source_unit_id"] == construction["unit_id"]
            and item["target_unit_id"] == stem["unit_id"]
            for item in relations):
        return None
    if not any(
            item["relation_kind"] == "FILLS_SLOT"
            and item["source_unit_id"] == stem["unit_id"]
            and item["target_unit_id"] == construction["unit_id"]
            for item in relations):
        return None
    return tuple(sorted(
        (item for item in units.values()
         if item["unit_kind"] in {"COMPONENT", "STEM"}),
        key=lambda item: (item["start"], item["end"], item["unit_id"]),
    ))


class W02LearningRuntimeV2(W02LearningRuntime):
    """复用冻结 v1 学习状态，只替换为 evidence-bound 形态生成入口。"""

    def generate(
            self,
            target: W02MorphologyTargetV2,
            *,
            morphology_consumer_enabled: bool = True,
            ) -> W02GenerationResult:
        """仅让 active learned construction 消费当前 typed unit Evidence。"""
        if not isinstance(target, W02MorphologyTargetV2):
            raise TypeError("W-02 v2 generate 拒绝无 typed Evidence 的 target")
        if type(morphology_consumer_enabled) is not bool:
            raise TypeError("morphology_consumer_enabled 必须是 bool")
        if not morphology_consumer_enabled:
            return W02GenerationResult(GENERATION_UNKNOWN, (), ())
        surfaces: dict[str, tuple[int, ...]] = {}
        for learned in self.candidates():
            if not learned.active or learned.payload_kind != PAYLOAD_KIND:
                continue
            value = learned.payload
            if (value["construction_key"] != target.construction_key
                    or value["baseline_kind"] == "DICTIONARY_REPLAY_ONLY"
                    or value["candidate_kind"] in {
                        "UNKNOWN", "SEGMENTATION", "GENERATION"}):
                continue
            generated = self._apply_evidence_rule(value, target)
            if generated is not None:
                surfaces[generated] = learned.observation_key
        ordered = tuple(sorted(surfaces))
        if not ordered:
            return W02GenerationResult(GENERATION_UNKNOWN, (), ())
        status = (
            GENERATION_GENERATED if len(ordered) == 1
            else GENERATION_CONFLICT)
        return W02GenerationResult(
            status,
            ordered,
            tuple(surfaces[item] for item in ordered),
        )

    def _apply_evidence_rule(
            self,
            value: dict[str, Any],
            target: W02MorphologyTargetV2,
            ) -> str | None:
        """按完整 learned relation 拓扑组合 Evidence surface，坏状态一律拒绝。"""
        try:
            validate_morphology_payload(value)
        except (AuthoredMorphologyCourseError, KeyError, TypeError, ValueError):
            return None
        role_units = _learned_role_units(value)
        if role_units is None:
            return None
        units = {item["unit_id"]: item for item in value["analysis_units"]}
        relations = value["morphology_relations"]
        family_kinds = {
            item["relation_kind"] for item in relations
            if item["relation_kind"] in _FAMILY_RELATIONS
        }
        if len(family_kinds) != 1:
            return None
        family = next(iter(family_kinds))
        learned_stem = next(
            item for item in units.values() if item["unit_kind"] == "STEM")

        if family == "REDUPLICATES":
            relation = next(
                item for item in relations
                if item["relation_kind"] == family)
            reduplicant = units[relation["source_unit_id"]]
            if (reduplicant["unit_kind"] != "REDUPLICANT"
                    or relation["target_unit_id"] != learned_stem["unit_id"]
                    or target.component_evidence):
                return None
            return target.stem_surface + target.stem_surface

        if family == "ATTACHES_AFFIX":
            relation = next(
                item for item in relations
                if item["relation_kind"] == family)
            affix = units[relation["source_unit_id"]]
            if (affix["unit_kind"] != "AFFIX"
                    or relation["target_unit_id"] != learned_stem["unit_id"]
                    or target.component_evidence
                    or self.word_forms.lookup(
                        affix["surface"], branch=self.branch) is None):
                return None
            return (
                target.stem_surface + affix["surface"]
                if affix["start"] >= learned_stem["end"]
                else affix["surface"] + target.stem_surface)

        if family == "COMPOUND_COMPONENT":
            learned_components = tuple(
                item for item in role_units if item["unit_kind"] == "COMPONENT")
            if len(learned_components) != len(target.component_evidence):
                return None
            for component in learned_components:
                if not any(
                        item["relation_kind"] == family
                        and item["target_unit_id"] == component["unit_id"]
                        for item in relations):
                    return None
            component_iter = iter(target.component_surfaces)
            output = []
            for unit in role_units:
                output.append(
                    target.stem_surface
                    if unit["unit_kind"] == "STEM"
                    else next(component_iter))
            return "".join(output)

        if family == "EXCEPTION_TO":
            relation = next(
                item for item in relations
                if item["relation_kind"] == family)
            exception = units[relation["source_unit_id"]]
            construction = next(
                item for item in units.values()
                if item["unit_kind"] == "CONSTRUCTION")
            observed = value["observed_surface"]["text"]
            if (exception["unit_kind"] != "EXCEPTION_FORM"
                    or relation["target_unit_id"] != construction["unit_id"]
                    or target.component_evidence):
                return None
            return observed if target.stem_surface == learned_stem["surface"] else None
        return None


def open_w02_learning_runtime_v2(
        backend: StorageBackend,
        *,
        mode: str,
        ) -> W02LearningRuntimeV2:
    """用冻结 v1 状态装配 evidence-bound v2 consumer，不迁移或重写状态。"""
    base = open_w02_learning_runtime(backend, mode=mode)
    return W02LearningRuntimeV2(
        base.backend,
        base.candidate_runtime,
        base.word_forms,
        base.branch,
        base.envelopes,
        base.use_outcomes,
        mode=base.mode,
    )


__all__ = [
    "W02_MORPHOLOGY_ADAPTER_VERSION",
    "W02LearningRuntimeV2",
    "W02MorphologyTargetV2",
    "W02MorphologyUnitEvidenceV2",
    "build_w02_morphology_target_v2",
    "morphology_target_from_observation_v2",
    "morphology_unit_evidence_v2",
    "open_w02_learning_runtime_v2",
]
