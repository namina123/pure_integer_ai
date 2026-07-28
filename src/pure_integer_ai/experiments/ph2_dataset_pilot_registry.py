"""D-02F 小批 pilot 的固定资料包注册表和动态 compiler 边界。"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


class DatasetPilotRegistryError(RuntimeError):
    """pilot 注册表、compiler 常量或仓库相对路径发生漂移。"""


@dataclass(frozen=True, order=True)
class PilotPackSpec:
    """声明一个 D-02F 输入包的稳定顺序、compiler 和冻结课程身份。"""

    pack_id: int
    module_name: str
    compiler_name: str
    pack_name: str
    stage: str
    substage: str
    sample_relative_path: str | None
    synthetic: bool = False

    def __post_init__(self) -> None:
        if type(self.pack_id) is not int or self.pack_id <= 0:
            raise DatasetPilotRegistryError("pilot pack_id 必须是正严格整数")
        for name, value in (
                ("module_name", self.module_name),
                ("compiler_name", self.compiler_name),
                ("pack_name", self.pack_name),
                ("stage", self.stage),
                ("substage", self.substage)):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise DatasetPilotRegistryError(f"pilot {name} 非法")
        if self.synthetic != (self.sample_relative_path is None):
            raise DatasetPilotRegistryError(
                "synthetic pack 与 sample_relative_path 声明不一致")
        if self.sample_relative_path is not None:
            relative = Path(self.sample_relative_path)
            if (relative.is_absolute() or ".." in relative.parts
                    or "\\" in self.sample_relative_path):
                raise DatasetPilotRegistryError("pilot sample 路径必须是安全相对路径")

    def sample_path(self, repository_root: str | Path) -> Path | None:
        """解析并约束 sample 必须位于权威 Git 仓库内。"""
        if self.sample_relative_path is None:
            return None
        root = Path(repository_root).resolve()
        path = (root / self.sample_relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise DatasetPilotRegistryError(
                f"pilot sample 缺失或逃逸仓库: {self.sample_relative_path}")
        return path

    def load_compiler(self) -> Callable[..., object]:
        """加载 compiler，并反查模块常量防止 registry 静默过期。"""
        try:
            module = importlib.import_module(self.module_name)
        except ImportError as error:
            raise DatasetPilotRegistryError(
                f"pilot compiler module 无法导入: {self.module_name}") from error
        for name, expected in (
                ("PACK_NAME", self.pack_name),
                ("STAGE", self.stage),
                ("SUBSTAGE", self.substage)):
            if getattr(module, name, None) != expected:
                raise DatasetPilotRegistryError(
                    f"pilot registry 与 {self.module_name}.{name} 漂移")
        compiler = getattr(module, self.compiler_name, None)
        if not callable(compiler):
            raise DatasetPilotRegistryError(
                f"pilot compiler 不可调用: {self.compiler_name}")
        return compiler


_PREFIX = "pure_integer_ai.experiments."


PILOT_PACK_SPECS = (
    PilotPackSpec(
        1, _PREFIX + "ph2_authored_atomic_course",
        "compile_authored_atomic_course",
        "AUTHORED_CC0_V1--CC0-1.0--atomic-v1", "W-05",
        "OCCURRENCE_ROLE_ATOMIC_PROPOSITION",
        "data/ph2/authored_atomic_seed_v1.jsonl.sample"),
    PilotPackSpec(
        2, _PREFIX + "ph2_authored_primitive_course",
        "compile_authored_primitive_course",
        "AUTHORED_CC0_V1--CC0-1.0--primitive-v1", "W-04",
        "PRIMITIVE_SURFACE_MAPPING",
        "data/ph2/authored_primitive_seed_v1.jsonl.sample"),
    PilotPackSpec(
        3, _PREFIX + "ph2_authored_sense_course",
        "compile_authored_sense_course",
        "AUTHORED_CC0_V1--CC0-1.0--sense-v1", "W-03",
        "SENSE_CONCEPT_BOUNDARY",
        "data/ph2/authored_sense_seed_v1.jsonl.sample"),
    PilotPackSpec(
        4, _PREFIX + "ph2_authored_alias_refers_course",
        "compile_authored_alias_refers_course",
        "AUTHORED_CC0_V1--CC0-1.0--alias-refers-v1", "W-06",
        "PURE_ALIAS_REFERS",
        "data/ph2/authored_relation_alias_refers_seed_v1.jsonl.sample"),
    PilotPackSpec(
        5, _PREFIX + "ph2_authored_subset_member_course",
        "compile_authored_subset_member_course",
        "AUTHORED_CC0_V1--CC0-1.0--subset-member-v1", "W-06",
        "SUBSET_MEMBER",
        "data/ph2/authored_relation_subset_member_seed_v1.jsonl.sample"),
    PilotPackSpec(
        6, _PREFIX + "ph2_authored_property_course",
        "compile_authored_property_course",
        "AUTHORED_CC0_V1--CC0-1.0--property-v1", "W-06",
        "PROPERTY",
        "data/ph2/authored_relation_property_seed_v1.jsonl.sample"),
    PilotPackSpec(
        7, _PREFIX + "ph2_authored_mereology_course",
        "compile_authored_mereology_course",
        "AUTHORED_CC0_V1--CC0-1.0--mereology-v1", "W-06",
        "MEREOLOGY",
        "data/ph2/authored_relation_mereology_seed_v1.jsonl.sample"),
    PilotPackSpec(
        8, _PREFIX + "ph2_authored_semantic_pair_course",
        "compile_authored_semantic_pair_course",
        "AUTHORED_CC0_V1--CC0-1.0--similar-antonym-v1", "W-06",
        "SIMILAR_ANTONYM",
        "data/ph2/authored_relation_similar_antonym_seed_v1.jsonl.sample"),
    PilotPackSpec(
        9, _PREFIX + "ph2_authored_precedes_course",
        "compile_authored_precedes_course",
        "AUTHORED_CC0_V1--CC0-1.0--precedes-v1", "W-06",
        "PRECEDES",
        "data/ph2/authored_relation_precedes_seed_v1.jsonl.sample"),
    PilotPackSpec(
        10, _PREFIX + "ph2_authored_causes_course",
        "compile_authored_causes_course",
        "AUTHORED_CC0_V1--CC0-1.0--causes-v1", "W-06",
        "CAUSES",
        "data/ph2/authored_relation_causes_seed_v1.jsonl.sample"),
    PilotPackSpec(
        11, _PREFIX + "ph2_authored_not_course",
        "compile_authored_not_course",
        "AUTHORED_CC0_V1--CC0-1.0--not-v1", "W-07", "NOT",
        "data/ph2/authored_logic_not_seed_v1.jsonl.sample"),
    PilotPackSpec(
        12, _PREFIX + "ph2_authored_and_or_course",
        "compile_authored_and_or_course",
        "AUTHORED_CC0_V1--CC0-1.0--and-or-v1", "W-07", "AND_OR",
        "data/ph2/authored_logic_and_or_seed_v1.jsonl.sample"),
    PilotPackSpec(
        13, _PREFIX + "ph2_authored_condition_course",
        "compile_authored_condition_course",
        "AUTHORED_CC0_V1--CC0-1.0--condition-v1", "W-07", "CONDITION",
        "data/ph2/authored_logic_condition_seed_v1.jsonl.sample"),
    PilotPackSpec(
        14, _PREFIX + "ph2_authored_exists_course",
        "compile_authored_exists_course",
        "AUTHORED_CC0_V1--CC0-1.0--exists-v1", "W-07", "EXISTS",
        "data/ph2/authored_logic_exists_seed_v1.jsonl.sample"),
    PilotPackSpec(
        15, _PREFIX + "ph2_authored_forall_course",
        "compile_authored_forall_course",
        "AUTHORED_CC0_V1--CC0-1.0--forall-v1", "W-07", "FORALL",
        "data/ph2/authored_logic_forall_seed_v1.jsonl.sample"),
    PilotPackSpec(
        16, _PREFIX + "ph2_authored_modal_course",
        "compile_authored_modal_course",
        "AUTHORED_CC0_V1--CC0-1.0--modal-v1", "W-07", "MODAL",
        "data/ph2/authored_logic_modal_seed_v1.jsonl.sample"),
    PilotPackSpec(
        17, _PREFIX + "ph2_authored_nested_course",
        "compile_authored_nested_course",
        "AUTHORED_CC0_V1--CC0-1.0--nested-scope-v1", "W-07",
        "NESTED_SCOPE",
        "data/ph2/authored_logic_nested_scope_seed_v1.jsonl.sample"),
    PilotPackSpec(
        18, _PREFIX + "ph2_authored_discourse_course",
        "compile_authored_discourse_course",
        "AUTHORED_CC0_V1--CC0-1.0--discourse-revision-v1", "W-08",
        "DISCOURSE_REVISION",
        "data/ph2/authored_discourse_revision_seed_v1.jsonl.sample"),
    PilotPackSpec(
        19, _PREFIX + "ph2_authored_qa_course",
        "compile_authored_qa_course",
        "AUTHORED_CC0_V1--CC0-1.0--question-answer-v1", "W-09",
        "QUESTION_ANSWER",
        "data/ph2/authored_question_answer_seed_v1.jsonl.sample"),
    PilotPackSpec(
        20, _PREFIX + "ph2_authored_generation_course",
        "compile_authored_generation_course",
        "AUTHORED_CC0_V1--CC0-1.0--generation-postcheck-v1", "W-09",
        "GENERATION_POSTCHECK",
        "data/ph2/authored_generation_postcheck_seed_v1.jsonl.sample"),
    PilotPackSpec(
        21, _PREFIX + "ph2_dataset_pilot_probe",
        "compile_pilot_split_probe",
        "PILOT_CC0_V1--CC0-1.0--split-isolation-probe-v1", "W-01",
        "DATASET_SPLIT_ISOLATION", None, True),
)


def validate_pilot_registry(
        specs: tuple[PilotPackSpec, ...] = PILOT_PACK_SPECS,
        ) -> tuple[PilotPackSpec, ...]:
    """要求注册表顺序连续且 pack/compiler/sample 身份不重复。"""
    if not isinstance(specs, tuple) or not specs:
        raise DatasetPilotRegistryError("pilot registry 必须是非空 tuple")
    if any(not isinstance(spec, PilotPackSpec) for spec in specs):
        raise DatasetPilotRegistryError("pilot registry 含非法 spec")
    ids = tuple(spec.pack_id for spec in specs)
    if ids != tuple(range(1, len(specs) + 1)):
        raise DatasetPilotRegistryError("pilot pack_id 必须从 1 连续递增")
    for values, label in (
            ((spec.pack_name for spec in specs), "pack_name"),
            ((spec.module_name for spec in specs), "module_name"),
            ((spec.compiler_name for spec in specs), "compiler_name")):
        materialized = tuple(values)
        if len(set(materialized)) != len(materialized):
            raise DatasetPilotRegistryError(f"pilot {label} 重复")
    samples = tuple(
        spec.sample_relative_path for spec in specs
        if spec.sample_relative_path is not None)
    if len(set(samples)) != len(samples):
        raise DatasetPilotRegistryError("pilot sample_relative_path 重复")
    return specs


validate_pilot_registry()


__all__ = [
    "DatasetPilotRegistryError",
    "PILOT_PACK_SPECS",
    "PilotPackSpec",
    "validate_pilot_registry",
]
