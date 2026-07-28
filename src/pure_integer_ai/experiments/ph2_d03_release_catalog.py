"""从公开 D-02 编译器构建 D-03 全局课程、九阶段和失效图。"""
from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    D03ContractError,
    D03FileIdentity,
    D03PublicationState,
    D03ReleaseIdentity,
    STAGE_KEYS,
    W06_SUBSTAGE_KEYS,
    W07_SUBSTAGE_KEYS,
    ZERO_EXECUTION_STATE,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_global_contract import (
    D03ArtifactReference,
    D03ExcludedSource,
    D03GlobalManifest,
    D03PackBinding,
    GLOBAL_ARTIFACT_KIND,
    GLOBAL_ARTIFACT_STATUS,
)
from pure_integer_ai.experiments.ph2_d03_invalidation import (
    INVALIDATION_ARTIFACT_KIND,
    StageDependencyEdge,
    StageInvalidationGraph,
    StageInvalidationRule,
)
from pure_integer_ai.experiments.ph2_d03_stage_contract import (
    AGGREGATION_POLICY,
    D03StageIdentity,
    D03StageManifest,
    EvaluationThreshold,
    OWNER_KEYS,
    REQUIRED_FAILURE_POINTS,
    RUN_ID_POLICY,
    STAGE_ARTIFACT_KIND,
    StageDataVisibility,
    StageEvaluationBinding,
    StageRecoveryBinding,
    StageResourceBudget,
)
from pure_integer_ai.experiments.ph2_dataset_io import read_artifact_manifest


RELEASE_KEY = "PH2-D03-V1"
RELEASE_VERSION = "PH2-D03-formal-release-v1"
GLOBAL_ARTIFACT_VERSION = "PH2-D03-global-course-manifest-v1"
INVALIDATION_ARTIFACT_VERSION = "PH2-D03-stage-invalidation-graph-v1"
COURSE_RELEASE_ROOT = "ph2_d03_dataset_artifacts/course_freeze_v1"
FORMAL_MANIFEST_ROOT = "data/ph2/manifests/d03_v1"
FORMAL_GLOBAL_MANIFEST_PATH = (
    FORMAL_MANIFEST_ROOT + "/ph2_global_course_manifest_v1.json"
)
FORMAL_INVALIDATION_GRAPH_PATH = (
    FORMAL_MANIFEST_ROOT + "/stage_invalidation_graph_v1.json"
)
FORMAL_RECEIPT_PATH = (
    FORMAL_MANIFEST_ROOT + "/ph2_d03_post_publication_receipt_v1.json"
)
PARENT_GATE_PATH = "data/ph2/manifests/j_lg_d03_gate_v4.json"
CAPABILITY_BASELINE_PATH = (
    "data/ph2/manifests/language_capability_baseline_v41.json"
)
SOURCE_COVERAGE_PATH = "data/ph2/manifests/d02_source_pack_coverage_v1.json"
HISTORICAL_HOLD_RECEIPT_PATH = (
    "data/ph2/manifests/j_lg_d03_gate_v4_git_publication_v1.json"
)
CC_CEDICT_BLOCKER_PATH = (
    "data/ph2/manifests/cc_cedict_20260725.license_reconciliation_v1.json"
)
SOURCE_PACK_MANIFEST_PATHS = (
    "ph2_dataset_artifacts/d02_source_pack_v1/packs/"
    "CONCEPTNET_5_7_0--CC-BY-4.0--source-pack-v1/manifest.json",
    "ph2_dataset_artifacts/d02_source_pack_v1/packs/"
    "CONCEPTNET_5_7_0--CC-BY-SA-4.0--source-pack-v1/manifest.json",
    "ph2_dataset_artifacts/d02_source_pack_v1/packs/"
    "UD_ZH_GSDSIMP_R2_18--CC-BY-SA-4.0--source-pack-v1/manifest.json",
    "ph2_dataset_artifacts/d02_source_pack_v1/packs/"
    "WIKIDATA_REVISION_V1--CC0-1.0--source-pack-v1/manifest.json",
    "ph2_dataset_artifacts/d02_source_pack_v1/packs/"
    "ZHWIKIPEDIA_20260701--CC-BY-SA-4.0--source-pack-v1/manifest.json",
    "ph2_dataset_artifacts/d02_source_pack_v1/packs/"
    "ZHWIKTIONARY_20260701--CC-BY-SA-4.0--source-pack-v1/manifest.json",
)


@dataclass(frozen=True)
class CourseCompilerSpec:
    """冻结一个 D-02 原创课程编译器、输入 seed 和阶段身份。"""

    module_name: str
    compiler_name: str
    sample_relative_path: str
    stage_key: str
    substage_key: str
    pack_name: str


def _course(
        module: str,
        compiler: str,
        sample: str,
        stage: str,
        substage: str,
        pack: str,
        ) -> CourseCompilerSpec:
    """用统一命名空间构造冻结课程编译条目。"""
    return CourseCompilerSpec(
        "pure_integer_ai.experiments." + module,
        compiler,
        "data/ph2/" + sample,
        stage,
        substage,
        pack,
    )


COURSE_COMPILER_SPECS = (
    _course("ph2_authored_text_fidelity_course", "compile_authored_text_fidelity_course", "authored_text_fidelity_seed_v1.jsonl.sample", "W-02", "LC01_TEXT_FIDELITY", "AUTHORED_CC0_V1--CC0-1.0--lc01-text-fidelity-v1"),
    _course("ph2_authored_morphology_course", "compile_authored_morphology_course", "authored_morphology_seed_v1.jsonl.sample", "W-02", "LC02_MORPHOLOGY_WORD_FORM", "AUTHORED_CC0_V1--CC0-1.0--lc02-morphology-v1"),
    _course("ph2_authored_construction_course", "compile_authored_construction_course", "authored_construction_seed_v1.jsonl.sample", "W-03", "LC03_MULTIWORD_CONSTRUCTION", "AUTHORED_CC0_V1--CC0-1.0--lc03-construction-v1"),
    _course("ph2_authored_sense_course", "compile_authored_sense_course", "authored_sense_seed_v1.jsonl.sample", "W-03", "SENSE_CONCEPT_BOUNDARY", "AUTHORED_CC0_V1--CC0-1.0--sense-v1"),
    _course("ph2_authored_primitive_course", "compile_authored_primitive_course", "authored_primitive_seed_v1.jsonl.sample", "W-04", "PRIMITIVE_SURFACE_MAPPING", "AUTHORED_CC0_V1--CC0-1.0--primitive-v1"),
    _course("ph2_authored_atomic_course", "compile_authored_atomic_course", "authored_atomic_seed_v1.jsonl.sample", "W-05", "OCCURRENCE_ROLE_ATOMIC_PROPOSITION", "AUTHORED_CC0_V1--CC0-1.0--atomic-v1"),
    _course("ph2_authored_recursive_parse_course", "compile_authored_recursive_parse_course", "authored_recursive_parse_seed_v1.jsonl.sample", "W-05", "LC04_RECURSIVE_PARSE", "AUTHORED_CC0_V1--CC0-1.0--lc04-recursive-parse-v1"),
    _course("ph2_authored_event_time_aspect_course", "compile_authored_event_time_aspect_course", "authored_event_time_aspect_seed_v1.jsonl.sample", "W-05", "LC05_EVENT_TIME_ASPECT", "AUTHORED_CC0_V1--CC0-1.0--lc05-event-time-aspect-v1"),
    _course("ph2_authored_comparison_quantity_course", "compile_authored_comparison_quantity_course", "authored_comparison_quantity_seed_v1.jsonl.sample", "W-05", "LC06_COMPARISON_QUANTITY_MEASURE", "AUTHORED_CC0_V1--CC0-1.0--lc06-comparison-quantity-v1"),
    _course("ph2_authored_alias_refers_course", "compile_authored_alias_refers_course", "authored_relation_alias_refers_seed_v1.jsonl.sample", "W-06", "PURE_ALIAS_REFERS", "AUTHORED_CC0_V1--CC0-1.0--alias-refers-v1"),
    _course("ph2_authored_subset_member_course", "compile_authored_subset_member_course", "authored_relation_subset_member_seed_v1.jsonl.sample", "W-06", "SUBSET_MEMBER", "AUTHORED_CC0_V1--CC0-1.0--subset-member-v1"),
    _course("ph2_authored_property_course", "compile_authored_property_course", "authored_relation_property_seed_v1.jsonl.sample", "W-06", "PROPERTY", "AUTHORED_CC0_V1--CC0-1.0--property-v1"),
    _course("ph2_authored_mereology_course", "compile_authored_mereology_course", "authored_relation_mereology_seed_v1.jsonl.sample", "W-06", "MEREOLOGY", "AUTHORED_CC0_V1--CC0-1.0--mereology-v1"),
    _course("ph2_authored_semantic_pair_course", "compile_authored_semantic_pair_course", "authored_relation_similar_antonym_seed_v1.jsonl.sample", "W-06", "SIMILAR_ANTONYM", "AUTHORED_CC0_V1--CC0-1.0--similar-antonym-v1"),
    _course("ph2_authored_precedes_course", "compile_authored_precedes_course", "authored_relation_precedes_seed_v1.jsonl.sample", "W-06", "PRECEDES", "AUTHORED_CC0_V1--CC0-1.0--precedes-v1"),
    _course("ph2_authored_causes_course", "compile_authored_causes_course", "authored_relation_causes_seed_v1.jsonl.sample", "W-06", "CAUSES", "AUTHORED_CC0_V1--CC0-1.0--causes-v1"),
    _course("ph2_authored_not_course", "compile_authored_not_course", "authored_logic_not_seed_v1.jsonl.sample", "W-07", "NOT", "AUTHORED_CC0_V1--CC0-1.0--not-v1"),
    _course("ph2_authored_and_or_course", "compile_authored_and_or_course", "authored_logic_and_or_seed_v1.jsonl.sample", "W-07", "AND_OR", "AUTHORED_CC0_V1--CC0-1.0--and-or-v1"),
    _course("ph2_authored_condition_course", "compile_authored_condition_course", "authored_logic_condition_seed_v1.jsonl.sample", "W-07", "CONDITION", "AUTHORED_CC0_V1--CC0-1.0--condition-v1"),
    _course("ph2_authored_exists_course", "compile_authored_exists_course", "authored_logic_exists_seed_v1.jsonl.sample", "W-07", "EXISTS", "AUTHORED_CC0_V1--CC0-1.0--exists-v1"),
    _course("ph2_authored_forall_course", "compile_authored_forall_course", "authored_logic_forall_seed_v1.jsonl.sample", "W-07", "FORALL", "AUTHORED_CC0_V1--CC0-1.0--forall-v1"),
    _course("ph2_authored_modal_course", "compile_authored_modal_course", "authored_logic_modal_seed_v1.jsonl.sample", "W-07", "MODAL", "AUTHORED_CC0_V1--CC0-1.0--modal-v1"),
    _course("ph2_authored_nested_course", "compile_authored_nested_course", "authored_logic_nested_scope_seed_v1.jsonl.sample", "W-07", "NESTED_SCOPE", "AUTHORED_CC0_V1--CC0-1.0--nested-scope-v1"),
    _course("ph2_authored_discourse_information_course", "compile_authored_discourse_information_course", "authored_discourse_information_seed_v1.jsonl.sample", "W-08", "LC07_DISCOURSE_INFORMATION_STRUCTURE", "AUTHORED_CC0_V1--CC0-1.0--lc07-discourse-information-v1"),
    _course("ph2_authored_open_set_clarification_course", "compile_authored_open_set_clarification_course", "authored_open_set_clarification_seed_v1.jsonl.sample", "W-08", "LC08_OPEN_SET_CLARIFICATION", "AUTHORED_CC0_V1--CC0-1.0--lc08-open-set-clarification-v1"),
    _course("ph2_authored_attribution_quotation_course", "compile_authored_attribution_quotation_course", "authored_attribution_quotation_seed_v1.jsonl.sample", "W-08", "LC14_ATTRIBUTION_QUOTATION_PERSPECTIVE", "AUTHORED_CC0_V1--CC0-1.0--lc14-attribution-quotation-v1"),
    _course("ph2_authored_discourse_course", "compile_authored_discourse_course", "authored_discourse_revision_seed_v1.jsonl.sample", "W-08", "DISCOURSE_REVISION", "AUTHORED_CC0_V1--CC0-1.0--discourse-revision-v1"),
    _course("ph2_authored_qa_course", "compile_authored_qa_course", "authored_question_answer_seed_v1.jsonl.sample", "W-09", "QUESTION_ANSWER", "AUTHORED_CC0_V1--CC0-1.0--question-answer-v1"),
    _course("ph2_authored_generation_course", "compile_authored_generation_course", "authored_generation_postcheck_seed_v1.jsonl.sample", "W-09", "GENERATION_POSTCHECK", "AUTHORED_CC0_V1--CC0-1.0--generation-postcheck-v1"),
    _course("ph2_authored_generation_generalization_course", "compile_authored_generation_generalization_course", "authored_generation_generalization_seed_v1.jsonl.sample", "W-09", "GG03_GENERATION_GENERALIZATION", "AUTHORED_CC0_V1--CC0-1.0--gg03-generation-generalization-v1"),
    _course("ph2_authored_free_text_hierarchy_recall_course", "compile_authored_free_text_hierarchy_recall_course", "authored_free_text_hierarchy_recall_course_seed_v1.jsonl.sample", "W-09", "FREE_TEXT_HIERARCHY_RECALL", "AUTHORED_CC0_V1--CC0-1.0--free-text-hierarchy-recall-v1"),
)


EVALUATION_DIMENSIONS = {
    "W-01": ("VERSION_ASSEMBLY", "VISIBILITY_ZERO_READ", "FRESH_RESUME", "WORKER_EQUIVALENCE", "FAULT_RECOVERY"),
    "W-02": ("NEW_CONTENT_MORPHOLOGY", "BOUNDARY_WITHDRAWAL", "OOV", "MULTI_CANDIDATE"),
    "W-03": ("POLYSEMY_COMPETITION", "CONCEPT_SPLIT", "SUPERSEDE", "SOURCE_CONFLICT"),
    "W-04": ("CUE_REPLACEMENT", "CONTENT_REPLACEMENT", "SEED_ABLATION", "EVIDENCE_ABLATION"),
    "W-05": ("OCCURRENCE_IDENTITY", "ROLE_SWAP", "PROPOSITION_CONSUMER", "SCOPE"),
    "W-06": W06_SUBSTAGE_KEYS,
    "W-07": W07_SUBSTAGE_KEYS,
    "W-08": ("CHINESE_VARIATION", "DISCOURSE", "P3IA", "LONG_CONTEXT", "LOCAL_RECOMPUTE"),
    "W-09": ("TEACHER_ZERO_WINDOW", "V06_CLONE", "ROLLBACK", "DIMENSIONAL_PASS", "RESOURCE_STOP", "W1_PHYSICAL_GROUNDING", "W2_DEFINITIVE_TRUTH"),
}


@dataclass(frozen=True)
class D03CandidateBuild:
    """返回已写入候选根的全局、九阶段、失效图和全部 pack 绑定。"""

    output_root: Path
    global_manifest: D03GlobalManifest
    stages: tuple[D03StageManifest, ...]
    invalidation_graph: StageInvalidationGraph


def _repo_path(root: Path, relative: str) -> Path:
    """把 POSIX 相对路径安全解析到指定根。"""
    target = (root / Path(*PurePosixPath(relative).parts)).resolve()
    if not target.is_relative_to(root):
        raise D03ContractError("catalog 路径逃逸 root")
    return target


def _file_identity(root: Path, relative: str) -> D03FileIdentity:
    """读取一个文件并形成 D-03 精确身份。"""
    path = _repo_path(root, relative)
    if not path.is_file():
        raise D03ContractError(f"catalog 依赖文件缺失: {relative}")
    payload = path.read_bytes()
    return D03FileIdentity(relative, len(payload), hashlib.sha256(payload).hexdigest())


def _pack_binding(root: Path, manifest_relative_path: str) -> D03PackBinding:
    """从规范 D-02 ArtifactManifest 形成全路径 owner 分账。"""
    manifest_path = _repo_path(root, manifest_relative_path)
    manifest = read_artifact_manifest(manifest_path)
    prefix = PurePosixPath(manifest_relative_path).parent.as_posix()

    def full_path(relative: str) -> str:
        """把 pack 内相对路径提升为仓库相对路径。"""
        return PurePosixPath(prefix, relative).as_posix()

    def selected(owner: str, split: str | None = None) -> tuple[str, ...]:
        """按物理 owner 和可选 split 返回文件路径。"""
        return tuple(sorted(
            full_path(item.relative_path)
            for item in manifest.files
            if item.owner_kind == owner and (split is None or item.split == split)
        ))

    return D03PackBinding(
        PurePosixPath(prefix).name,
        manifest.source_key,
        manifest.license_partition,
        manifest.earliest_invalidated_stage,
        _file_identity(root, manifest_relative_path),
        selected("source"),
        selected("observation", "train"),
        selected("observation", "dev"),
        selected("observation", "held_out"),
        selected("teacher"),
        selected("evaluator"),
        sum(item.record_count for item in manifest.files),
        len(manifest.source_cluster_keys),
    )


def _load_compiler(spec: CourseCompilerSpec) -> Callable[[Path, Path], Any]:
    """加载并核对冻结课程编译器的公开常量和入口。"""
    module = importlib.import_module(spec.module_name)
    if (getattr(module, "PACK_NAME", None) != spec.pack_name
            or getattr(module, "STAGE", None) != spec.stage_key
            or getattr(module, "SUBSTAGE", None) != spec.substage_key):
        raise D03ContractError("D-02 course compiler identity 漂移")
    compiler = getattr(module, spec.compiler_name, None)
    if not callable(compiler):
        raise D03ContractError("D-02 course compiler 入口缺失")
    return compiler


def _compile_course_packs(source_root: Path, output_root: Path) -> tuple[D03PackBinding, ...]:
    """逐项编译 31 个 owner-separated D-02 课程 pack。"""
    release_root = _repo_path(output_root, COURSE_RELEASE_ROOT)
    bindings: list[D03PackBinding] = []
    for spec in COURSE_COMPILER_SPECS:
        sample = _repo_path(source_root, spec.sample_relative_path)
        compiler = _load_compiler(spec)
        build = compiler(sample, release_root)
        if build.pack_root.name != spec.pack_name:
            raise D03ContractError("D-02 course compiler 产物 pack name 漂移")
        relative_manifest = PurePosixPath(
            COURSE_RELEASE_ROOT, "packs", spec.pack_name, "manifest.json"
        ).as_posix()
        bindings.append(_pack_binding(output_root, relative_manifest))
    return tuple(bindings)


def _release_identity(source_root: Path) -> D03ReleaseIdentity:
    """从现场 v4/v41/D-02 覆盖账形成正式 D-03 发布身份。"""
    baseline = read_canonical_object(
        _repo_path(source_root, CAPABILITY_BASELINE_PATH))
    versions = dict(baseline["version_keys"])
    versions.update({
        "course_version": "PH2-D03-COURSE-FREEZE-V1",
        "data_version": "PH2-D02-OWNER-SEPARATED-DATA-V1",
        "parser_version": "PH2-D02-PARSER-BUNDLE-V1",
        "primitive_version": "PH2-D02-TYPED-PRIMITIVE-REGISTRY-V1",
    })
    return D03ReleaseIdentity(
        1,
        RELEASE_KEY,
        RELEASE_VERSION,
        PARENT_GATE_PATH,
        _file_identity(source_root, PARENT_GATE_PATH).sha256,
        CAPABILITY_BASELINE_PATH,
        _file_identity(source_root, CAPABILITY_BASELINE_PATH).sha256,
        SOURCE_COVERAGE_PATH,
        _file_identity(source_root, SOURCE_COVERAGE_PATH).sha256,
        tuple(versions.items()),
    )


def _thresholds(stage_key: str) -> tuple[EvaluationThreshold, ...]:
    """形成逐维预注册阈值，并保留 W1/W2 墙维非承重 NE。"""
    result: list[EvaluationThreshold] = []
    for dimension in EVALUATION_DIMENSIONS[stage_key]:
        wall = dimension in {"W1_PHYSICAL_GROUNDING", "W2_DEFINITIVE_TRUTH"}
        result.append(EvaluationThreshold(
            f"{stage_key}-{dimension}",
            0 if wall else 1,
            1,
            0,
            0 if wall else 1,
            "ALLOW_NON_BEARING" if wall else "BLOCK",
            1,
        ))
    return tuple(result)


def _stage_manifest(stage_key: str, packs: tuple[D03PackBinding, ...]) -> D03StageManifest:
    """按阶段形成累计 train 白名单、future 明拒和私有 evaluator 集。"""
    ordinal = STAGE_KEYS.index(stage_key) + 1
    current_index = ordinal - 1
    available = tuple(
        item for item in packs
        if STAGE_KEYS.index(item.earliest_stage) <= current_index
    )
    future = tuple(
        item for item in packs
        if STAGE_KEYS.index(item.earliest_stage) > current_index
    )
    train_keys = tuple(item.pack_key for item in available if item.train_observation_paths)
    dev_keys = tuple(item.pack_key for item in available if item.dev_observation_paths)
    held_out_keys = tuple(
        item.pack_key for item in available if item.held_out_observation_paths)
    evaluator_keys = tuple(
        item.pack_key for item in available if item.evaluator_label_paths)
    substages: tuple[str, ...] = ()
    if stage_key == "W-06":
        substages = W06_SUBSTAGE_KEYS
    elif stage_key == "W-07":
        substages = W07_SUBSTAGE_KEYS
    prerequisites = () if ordinal == 1 else (STAGE_KEYS[ordinal - 2],)
    scale = ordinal * 100000
    return D03StageManifest(
        1,
        STAGE_ARTIFACT_KIND,
        f"PH2-D03-{stage_key.replace('-', '')}-stage-manifest-v1",
        RELEASE_KEY,
        D03StageIdentity(stage_key, ordinal, prerequisites, substages),
        StageDataVisibility(
            train_keys,
            tuple(item.pack_key for item in future),
            dev_keys,
            held_out_keys,
            evaluator_keys,
            *OWNER_KEYS,
            ("train",),
            ("dev", "held_out", "adversarial", "wall"),
        ),
        StageEvaluationBinding(
            f"PH2-{stage_key}-PRIVATE-EVALUATOR",
            f"PH2-{stage_key}-PRIVATE-EVALUATOR-V1",
            "PH2_PRIVATE_EVALUATOR",
            AGGREGATION_POLICY,
            _thresholds(stage_key),
            tuple(f"{stage_key}-{dimension}-ABLATION"
                  for dimension in EVALUATION_DIMENSIONS[stage_key]),
            3 if stage_key == "W-09" else 1,
        ),
        StageResourceBudget(
            scale,
            ordinal * 4096,
            ordinal * 65536,
            ordinal * 64 * 1024 * 1024,
            ordinal * 1000000,
            ordinal * 100000,
            4,
            ordinal * 256,
        ),
        StageRecoveryBinding(
            RUN_ID_POLICY,
            "PH2-D03-CURSOR-V1",
            1,
            16,
            (1, 2, 4),
            "PH2-D03-STABLE-MERGE-BARRIER-V1",
            REQUIRED_FAILURE_POINTS,
            1,
        ),
        dict(ZERO_EXECUTION_STATE),
    )


def _invalidation_graph(packs: tuple[D03PackBinding, ...]) -> StageInvalidationGraph:
    """形成 pack、许可、source、evaluator 和全局版本的精确失效规则。"""
    rules: list[StageInvalidationRule] = []
    for pack in packs:
        suffix = STAGE_KEYS[STAGE_KEYS.index(pack.earliest_stage):]
        for change_kind in ("PACK_CONTENT", "SOURCE_SET", "LICENSE"):
            rules.append(StageInvalidationRule(
                change_kind, pack.pack_key, pack.earliest_stage, suffix))
    for stage_key in STAGE_KEYS:
        rules.append(StageInvalidationRule(
            "EVALUATOR_VERSION", stage_key, stage_key,
            STAGE_KEYS[STAGE_KEYS.index(stage_key):],
        ))
    global_rules = {
        "BACKEND_VERSION": "W-01",
        "CODE_VERSION": "W-01",
        "COURSE_VERSION": "W-02",
        "DATA_VERSION": "W-02",
        "LOCATION_VERSION": "W-01",
        "PARSER_VERSION": "W-02",
        "PRIMITIVE_VERSION": "W-04",
        "SCHEMA_VERSION": "W-01",
        "SEGMENT_VERSION": "W-01",
    }
    for change_kind, earliest in global_rules.items():
        rules.append(StageInvalidationRule(
            change_kind, "GLOBAL", earliest,
            STAGE_KEYS[STAGE_KEYS.index(earliest):],
        ))
    return StageInvalidationGraph(
        1,
        INVALIDATION_ARTIFACT_KIND,
        INVALIDATION_ARTIFACT_VERSION,
        RELEASE_KEY,
        STAGE_KEYS,
        tuple(StageDependencyEdge(STAGE_KEYS[index], STAGE_KEYS[index - 1])
              for index in range(1, len(STAGE_KEYS))),
        tuple(rules),
    )


def _write_model(value: dict[str, Any], output_root: Path, relative: str) -> D03FileIdentity:
    """不可覆盖写一个模型并返回其精确文件身份。"""
    write_immutable_json(value, _repo_path(output_root, relative))
    return _file_identity(output_root, relative)


def build_d03_candidate(
        repository_root: str | Path,
        output_root: str | Path | None = None,
        ) -> D03CandidateBuild:
    """构建 owner-separated pack、九阶段、失效图和全局 candidate，绝不训练。"""
    source_root = Path(repository_root).resolve()
    target_root = Path(output_root).resolve() if output_root is not None else source_root
    target_root.mkdir(parents=True, exist_ok=True)
    course_packs = _compile_course_packs(source_root, target_root)
    source_packs = tuple(
        _pack_binding(source_root, path) for path in SOURCE_PACK_MANIFEST_PATHS)
    packs = tuple(sorted((*course_packs, *source_packs)))
    if len(packs) != 37:
        raise D03ContractError("D-03 必须冻结 31 个课程 pack 和 6 个 source pack")
    stages = tuple(_stage_manifest(key, packs) for key in STAGE_KEYS)
    stage_refs: list[D03ArtifactReference] = []
    for stage in stages:
        key = stage.stage_identity.stage_key
        relative = (
            FORMAL_MANIFEST_ROOT + "/stages/"
            + key.casefold().replace("-", "") + "_stage_manifest_v1.json"
        )
        stage_refs.append(D03ArtifactReference(
            key, _write_model(stage.to_dict(), target_root, relative)))
    graph = _invalidation_graph(packs)
    graph_ref = D03ArtifactReference(
        "STAGE_INVALIDATION_GRAPH",
        _write_model(graph.to_dict(), target_root, FORMAL_INVALIDATION_GRAPH_PATH),
    )
    global_manifest = D03GlobalManifest(
        1,
        GLOBAL_ARTIFACT_KIND,
        GLOBAL_ARTIFACT_VERSION,
        GLOBAL_ARTIFACT_STATUS,
        _release_identity(source_root),
        tuple(stage_refs),
        graph_ref,
        packs,
        (D03ExcludedSource(
            "CC_CEDICT_20260725",
            "BLOCKED",
            "OFFICIAL_LICENSE_EVIDENCE_DIVERGENCE",
            _file_identity(source_root, CC_CEDICT_BLOCKER_PATH),
        ),),
        _file_identity(source_root, HISTORICAL_HOLD_RECEIPT_PATH),
        (
            _file_identity(source_root, "paper/main.pdf"),
            _file_identity(source_root, "paper/main.tex"),
        ),
        D03PublicationState("CANDIDATE_VERIFIED", 0, "", 0),
        dict(ZERO_EXECUTION_STATE),
    )
    _write_model(global_manifest.to_dict(), target_root, FORMAL_GLOBAL_MANIFEST_PATH)
    return D03CandidateBuild(target_root, global_manifest, stages, graph)


__all__ = [
    "CAPABILITY_BASELINE_PATH",
    "COURSE_COMPILER_SPECS",
    "COURSE_RELEASE_ROOT",
    "CourseCompilerSpec",
    "D03CandidateBuild",
    "FORMAL_GLOBAL_MANIFEST_PATH",
    "FORMAL_INVALIDATION_GRAPH_PATH",
    "FORMAL_MANIFEST_ROOT",
    "FORMAL_RECEIPT_PATH",
    "HISTORICAL_HOLD_RECEIPT_PATH",
    "PARENT_GATE_PATH",
    "RELEASE_KEY",
    "RELEASE_VERSION",
    "SOURCE_COVERAGE_PATH",
    "SOURCE_PACK_MANIFEST_PATHS",
    "build_d03_candidate",
]
