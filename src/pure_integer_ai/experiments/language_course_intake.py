"""把 D-01 只读课程目录装配为正式多语言词形 provider。

装配期只物化 LanguageBranch；大词典保持在只读课程目录和内存 FMM 中。真实输入命中后，
provider 才经 WordFormIndex 惰性物化 Representation 并建立 legacy 桥。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.cognition.shared.language_object_index import (
    LanguageObjectIndex,
)
from pure_integer_ai.cognition.understanding.word_form_provider import (
    VisibleWordForm,
    WordFormProvider,
    WordFormProviderRegistry,
)
from pure_integer_ai.cognition.understanding.segmentation_hypothesis import (
    SegmentationProtocol,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.chinese_semantic_kb_curriculum import (
    SPLIT_DEV,
    SPLIT_HELD_OUT,
    SPLIT_TRAIN,
    ChineseSemanticKBCurriculum,
    read_curriculum_manifest,
)
from pure_integer_ai.experiments.data_manifest import (
    ManifestIntegrityError,
    read_manifest,
)

_VALID_SPLITS = frozenset({SPLIT_TRAIN, SPLIT_DEV, SPLIT_HELD_OUT})
_CATALOG_HASHER = Hasher("language_course_intake.catalog.v1")


@dataclass(frozen=True)
class LanguageCourseIntakeReport:
    """一次课程目录装配的版本、可见 split 和去重计数。"""

    course_manifest_sha256: str
    source_manifest_sha256: str
    runtime_language: int
    selected_splits: tuple[int, ...]
    source_rows: int
    selected_rows: int
    unique_forms: int
    duplicate_category_rows: int
    retokenized_items: int = 0


def _single_binding(manifest, name: str) -> int:
    """读取要求为单整数的源 manifest 绑定。"""
    values = manifest.binding(name)
    if len(values) != 1 or values[0] <= 0:
        raise ManifestIntegrityError(
            f"语言课程绑定 {name!r} 必须是单个正整数")
    return values[0]


def _validated_splits(values: tuple[int, ...]) -> tuple[int, ...]:
    """规范化调用方注入的课程可见 split，并拒绝空集或未知值。"""
    if not isinstance(values, tuple) or not values:
        raise ValueError("语言课程可见 split 必须是非空 tuple")
    assert_int(*values, _where="language_course_intake.splits")
    if any(type(value) is not int or value not in _VALID_SPLITS
           for value in values):
        raise ValueError("语言课程可见 split 含未知值")
    return tuple(sorted(set(values)))


def build_word_form_providers(
        *, backend, concept_index, ontology,
        course_root: str | Path,
        source_manifest_path: str | Path,
        runtime_language: int,
        visible_splits: tuple[int, ...],
        segmentation_protocol: SegmentationProtocol | None = None,
        ) -> tuple[WordFormProviderRegistry, LanguageCourseIntakeReport]:
    """核验 D-00/D-01 版本链并构造一个正式课程词形 provider。"""
    assert_int(runtime_language, _where="language_course_intake.runtime_language")
    if runtime_language <= 0:
        raise ValueError("运行期语言键必须为正")
    selected_splits = _validated_splits(visible_splits)
    root = Path(course_root).resolve()
    course_manifest = read_curriculum_manifest(root / "manifest.json")
    source_manifest = read_manifest(source_manifest_path)
    if source_manifest.sha256() != course_manifest.source_manifest_sha256:
        raise ManifestIntegrityError("D-01 课程与 D-00 源 manifest 摘要不匹配")
    if (source_manifest.dataset_name != course_manifest.dataset_name
            or source_manifest.dataset_version != course_manifest.dataset_version
            or source_manifest.adapter_version
            != course_manifest.source_adapter_version
            or source_manifest.parser_version
            != course_manifest.source_parser_version):
        raise ManifestIntegrityError("D-01 课程与 D-00 源版本链不匹配")

    course = ChineseSemanticKBCurriculum(course_manifest, root)
    expected_source_kind = _single_binding(
        source_manifest, "dataset_source_kind")
    epistemic_origin = _single_binding(source_manifest, "epistemic_origin")
    catalog: dict[str, VisibleWordForm] = {}
    split_by_surface: dict[str, int] = {}
    source_rows = 0
    selected_rows = 0
    duplicate_rows = 0
    for item in course.iter_word_forms():
        source_rows += 1
        old_split = split_by_surface.setdefault(item.surface, item.split)
        if old_split != item.split:
            raise ManifestIntegrityError(
                "同一词形在不同类别跨越课程 split")
        if item.split not in selected_splits:
            continue
        selected_rows += 1
        if item.source_ref.source_kind != expected_source_kind:
            raise ManifestIntegrityError("课程词形 SourceRef 类型与源 manifest 不一致")
        visible = VisibleWordForm(
            item.source_ref,
            item.split,
            expected_source_kind,
            epistemic_origin,
            course_manifest.course_version,
        )
        existing = catalog.get(item.surface)
        if existing is not None:
            duplicate_rows += 1
            continue
        catalog[item.surface] = visible

    objects = LanguageObjectIndex(ontology)
    branch = objects.ensure_branch(source_manifest.binding("language_branch"))
    digest_key = _CATALOG_HASHER.h63((
        course_manifest.sha256(), selected_splits)) or 1
    provider = WordFormProvider(
        backend=backend,
        concept_index=concept_index,
        ontology=ontology,
        branch=branch,
        runtime_language=runtime_language,
        unicode_family_key=source_manifest.binding("unicode_sequence_family"),
        inventory_relation_key=source_manifest.binding(
            "word_inventory_relation"),
        catalog_identity=(
            course_manifest.course_version,
            course_manifest.split_policy.version,
            digest_key,
        ),
        catalog=catalog,
        segmentation_protocol=segmentation_protocol,
    )
    registry = WordFormProviderRegistry()
    registry.register(provider)
    report = LanguageCourseIntakeReport(
        course_manifest.sha256(),
        source_manifest.sha256(),
        runtime_language,
        selected_splits,
        source_rows,
        selected_rows,
        len(catalog),
        duplicate_rows,
    )
    return registry, report


__all__ = [
    "LanguageCourseIntakeReport",
    "build_word_form_providers",
]
