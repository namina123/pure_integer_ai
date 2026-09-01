"""公开对话训练状态驱动的表层组织消费者。

该模块把 DLG-RAW-16 的公开结构课程接到真实回答侧：先读取 K 盘
``formal_train`` 运行摘要和 SQLite 计数，随后用独立 family 学得的
typed slot 结构重建可读完整句。运行时只消费调用方提供的 typed semantic
和结构槽位，不从某一语言的表面词形猜测关系；无法安全分解的答案原样保留，
绝不猜测事实或把模板回放冒充通用生成。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_compile import (
    load_surface_organization_jsonl,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    SurfaceStructureModel,
    SurfaceStructureRequest,
    SurfaceStructureResult,
    SurfaceSemantic,
    STRUCTURE_SELECTED,
    learn_surface_structure_model,
    load_surface_evidence_jsonl,
    realize_surface_structure,
)
from pure_integer_ai.experiments.conversation_raw_t1_surface_variants import (
    SurfaceVariantModel,
    learn_surface_variant_model,
    realize_surface_variants,
    VARIANT_SELECTED,
)
from pure_integer_ai.experiments.conversation_raw_t1_surface_order import (
    ORDER_SELECTED,
    SurfaceOrderModel,
    learn_surface_order_model,
    realize_surface_order,
)
from pure_integer_ai.experiments.conversation_runtime_material_generation_context import (
    RuntimeMaterialGenerationContext,
    validate_runtime_material_generation_context,
)
from pure_integer_ai.experiments.conversation_response_organization import (
    ResponseOrganizationModel,
    organize_response_surface,
)
if TYPE_CHECKING:
    from pure_integer_ai.experiments.conversation_dialogue_scale_showcase import (
        TrainingObservation,
    )


class TrainedSurfaceRuntimeError(ValueError):
    """训练状态、公开结构课程或表层重建发生漂移。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _surface_evidence_paths_for_run(
        project_root: Path, training_run_root: Path,
        default_path: Path,
        ) -> tuple[Path, ...]:
    """读取新 run 的 evidence commitment；旧 run 使用固定公开 evidence。"""
    manifest_path = training_run_root / "dialogue_pack_manifest.json"
    if not manifest_path.is_file():
        raise TrainedSurfaceRuntimeError("training run 缺少 dialogue pack manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainedSurfaceRuntimeError("training run manifest 不可回读") from error
    rows = manifest.get("surface_evidence_files") if isinstance(manifest, dict) else None
    if rows is None:
        return (default_path,)
    if not isinstance(rows, list) or not rows:
        raise TrainedSurfaceRuntimeError("surface evidence commitment 非法")
    result: list[Path] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 2:
            raise TrainedSurfaceRuntimeError("surface evidence commitment 记录非法")
        candidate = Path(str(row[0]))
        if not candidate.is_absolute():
            candidate = project_root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(project_root)
        except ValueError as error:
            raise TrainedSurfaceRuntimeError(
                "surface evidence 越出 project_root") from error
        if not candidate.is_file() or _sha256_file(candidate) != str(row[1]):
            raise TrainedSurfaceRuntimeError("surface evidence digest 漂移")
        result.append(candidate)
    if len(result) != len(set(result)):
        raise TrainedSurfaceRuntimeError("surface evidence commitment 重复")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SurfaceRenderResult:
    """一次回答侧表层消费的可审计值。"""

    surface: str
    used: bool
    pattern_id: int
    reason: str
    run_id: str
    graph_size: int
    trace: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class TrainedSurfaceRuntime:
    """绑定公开 pack、K 盘训练状态和可迁移结构模型的只读消费者。"""

    observation: "TrainingObservation"
    models: Mapping[tuple[str, str], SurfaceStructureModel]
    variant_models: Mapping[tuple[str, str], SurfaceVariantModel] = field(
        default_factory=dict)
    order_models: Mapping[tuple[str, str], SurfaceOrderModel] = field(
        default_factory=dict)
    organization_model: ResponseOrganizationModel | None = None
    # 由公开 slot evidence 学得的角色词位提示；不是代码内置词表。
    slot_value_hints: Mapping[tuple[str, str, str], tuple[str, ...]] = field(
        default_factory=dict)

    def __post_init__(self) -> None:
        if not hasattr(self.observation, "run_id"):
            raise TypeError("surface runtime observation 类型错误")
        if self.observation.graph_size <= 0 or self.observation.training_item_count <= 0:
            raise TrainedSurfaceRuntimeError("surface runtime 缺少真实训练状态")
        if not isinstance(self.models, Mapping):
            raise TypeError("surface runtime models 类型错误")
        if not isinstance(self.variant_models, Mapping):
            raise TypeError("surface runtime variant_models 类型错误")
        if not isinstance(self.order_models, Mapping):
            raise TypeError("surface runtime order_models 类型错误")
        if (self.organization_model is not None
                and not isinstance(
                    self.organization_model, ResponseOrganizationModel)):
            raise TypeError("surface runtime organization_model 类型错误")
        if not isinstance(self.slot_value_hints, Mapping):
            raise TypeError("surface runtime slot_value_hints 类型错误")
        for key, values in self.slot_value_hints.items():
            if (not isinstance(key, tuple) or len(key) != 3
                    or any(not isinstance(item, str) or not item for item in key)
                    or not isinstance(values, tuple)
                    or any(not isinstance(item, str) or not item for item in values)
                    or values != tuple(sorted(set(values)))):
                raise TrainedSurfaceRuntimeError(
                    "surface runtime slot_value_hints 非规范")

    def render(self, answer: str, *, response_act: str = "ANSWER",
               source_title: str | None = None,
               ordinal: int = 0,
               generation_context: RuntimeMaterialGenerationContext | None = None,
               ) -> SurfaceRenderResult:
        """消费一条已确认回答；不匹配时返回原表层和显式未消费原因。"""
        if not isinstance(answer, str) or not answer.strip():
            raise TrainedSurfaceRuntimeError("surface answer 不能为空")
        original = answer.strip()
        if response_act not in {"ANSWER", "CLARIFY", "UNKNOWN", "REPAIR"}:
            return SurfaceRenderResult(
                original, False, 0, "non_answer", self.observation.run_id,
                self.observation.graph_size)
        if generation_context is not None:
            try:
                validate_runtime_material_generation_context(
                    generation_context, response_act=response_act)
            except ValueError as error:
                raise TrainedSurfaceRuntimeError(
                    "generation context 与 surface response-act 不一致") from error
            # Runtime 结构证据参与候选变体的确定性选择；消费完整上下文
            # 身份的整数折叠值，不把 digest 或内部状态转成用户可见文本。
            if ordinal == 0:
                folded = 0
                for value in generation_context.identity_key:
                    folded = ((folded * 257) + value) & 0x7FFFFFFF
                ordinal = folded or 1
        # 从学习到的角色词位中恢复可唯一分段的输入。这里不假定任何语言
        # 的关系词；词位必须来自公开课程 evidence，且重建后须逐字符相等。
        for (act, register), model in sorted(self.models.items()):
            if act != response_act:
                continue
            for pattern in model.patterns:
                role_hints = {
                    role: self.slot_value_hints.get((act, register, role), ())
                    for role in pattern.roles
                }
                if any(not values for values in role_hints.values()):
                    continue
                values = _match_learned_pattern(
                    original, pattern.roles, pattern.gaps, role_hints)
                if values is None:
                    continue
                semantic = _semantic_from_slots(
                    response_act, pattern.roles, values)
                result = self.render_typed(
                    semantic,
                    response_act=response_act,
                    register=register,
                    ordered_roles=pattern.roles,
                    slot_values=values,
                    source_id=source_title or "runtime-source",
                    context_id="runtime-context",
                    family_id="runtime-surface-family",
                    ordinal=ordinal,
                )
                if result.used:
                    return result
        for (act, register), model in sorted(self.variant_models.items()):
            if act != response_act:
                continue
            for pattern in model.patterns:
                for option_index, option in enumerate(pattern.gap_options):
                    values = _match_pattern(original, pattern.roles, option)
                    if values is None:
                        continue
                    semantic = _semantic_from_slots(
                        response_act, pattern.roles, values)
                    variant_request = SurfaceStructureRequest(
                        semantic, response_act, register, pattern.roles,
                        1, 4096, (option_index + 1) % len(pattern.gap_options),
                        source_title or "runtime-source", "runtime-context",
                        "runtime-variant-family", values,
                    )
                    variant_result = realize_surface_variants(
                        model, variant_request)
                    if (variant_result.status_code == VARIANT_SELECTED
                            and variant_result.surface
                            and variant_result.surface != original):
                        return SurfaceRenderResult(
                            variant_result.surface, True,
                            variant_result.selected_pattern_id,
                            "variant_selected", self.observation.run_id,
                            self.observation.graph_size,
                            variant_result.trace)
        for (act, register), model in sorted(self.order_models.items()):
            if act != response_act:
                continue
            for pattern in model.patterns:
                for source_index, source_option in enumerate(pattern.options):
                    values = _match_pattern(original, source_option.roles,
                                            source_option.gaps)
                    if values is None or len(values) != len(source_option.roles):
                        continue
                    if len(pattern.options) < 2:
                        continue
                    target_option = pattern.options[
                        (source_index + 1) % len(pattern.options)]
                    by_role = dict(zip(source_option.roles, values))
                    target_values = tuple(
                        by_role.get(role, "") for role in target_option.roles)
                    if any(not value for value in target_values):
                        continue
                    order_request = SurfaceStructureRequest(
                        _semantic_from_slots(response_act, target_option.roles,
                                             target_values),
                        response_act, register, target_option.roles,
                        1, 4096, 0,
                        source_title or "runtime-source", "runtime-context",
                        "runtime-order-family", target_values,
                    )
                    order_result = realize_surface_order(model, order_request)
                    if (order_result.status_code == ORDER_SELECTED
                            and order_result.surface
                            and order_result.surface != original):
                        return SurfaceRenderResult(
                            order_result.surface, True,
                            order_result.selected_pattern_id,
                            "order_variant_selected", self.observation.run_id,
                            self.observation.graph_size, order_result.trace)
        for (act, register), model in sorted(self.models.items()):
            if act != response_act:
                continue
            for pattern in model.patterns:
                values = _match_pattern(original, pattern.roles, pattern.gaps)
                if values is None:
                    continue
                try:
                    candidates = tuple(item for item in model.patterns
                                       if item.dialogue_act == response_act
                                       and item.register == register
                                       and item.roles == pattern.roles)
                    matched_index = next(
                        index for index, item in enumerate(candidates)
                        if item.pattern_id == pattern.pattern_id)
                    variant_ordinal = (
                        (matched_index + 1) % len(candidates)
                        if len(candidates) > 1 else 0)
                    semantic = _semantic_from_slots(
                        response_act, pattern.roles, values)
                    result = self.render_typed(
                        semantic, response_act=response_act, register=register,
                        ordered_roles=pattern.roles, slot_values=values,
                        source_id=source_title or "runtime-source",
                        context_id="runtime-context",
                        family_id="runtime-surface-family",
                        ordinal=variant_ordinal,
                    )
                except (RuntimeError, TypeError, ValueError):
                    continue
                if result.used and result.surface == original:
                    return result
        if self.organization_model is not None:
            organized = organize_response_surface(
                self.organization_model, original)
            if organized.used:
                return SurfaceRenderResult(
                    organized.surface, True, organized.pattern_id,
                    organized.reason, self.observation.run_id,
                    self.observation.graph_size, organized.trace)
        return SurfaceRenderResult(
            original, False, 0, "no_learned_surface_shape",
            self.observation.run_id, self.observation.graph_size)

    def render_typed(
            self,
            semantic: SurfaceSemantic,
            *,
            response_act: str,
            register: str,
            ordered_roles: tuple[str, ...],
            slot_values: tuple[str, ...] = (),
            source_id: str = "runtime-source",
            context_id: str = "runtime-context",
            family_id: str = "runtime-surface-family",
            ordinal: int = 0,
            ) -> SurfaceRenderResult:
        """用已授权 typed semantic/slot 值重建新的可读表层。"""
        model = self.models.get((response_act, register))
        if model is None:
            return SurfaceRenderResult(
                "", False, 0, "no_learned_response_model",
                self.observation.run_id, self.observation.graph_size)
        request = SurfaceStructureRequest(
            semantic, response_act, register, ordered_roles,
            1, 4096, ordinal, source_id, context_id, family_id,
            slot_values,
        )
        result: SurfaceStructureResult = realize_surface_structure(model, request)
        if result.status_code != STRUCTURE_SELECTED or result.surface is None:
            return SurfaceRenderResult(
                "", False, 0, "structure_rejected",
                self.observation.run_id, self.observation.graph_size)
        return SurfaceRenderResult(
            result.surface, True, result.selected_pattern_id, "selected",
            self.observation.run_id, self.observation.graph_size,
            result.trace)

    def render_order_typed(
            self,
            semantic: SurfaceSemantic,
            *,
            response_act: str,
            register: str,
            ordered_roles: tuple[str, ...],
            slot_values: tuple[str, ...] = (),
            source_id: str = "runtime-source",
            context_id: str = "runtime-context",
            family_id: str = "runtime-order-family",
            ) -> SurfaceRenderResult:
        """用已授权 typed slots 消费 opt-in G9 角色排列模型。"""
        model = self.order_models.get((response_act, register))
        if model is None:
            return SurfaceRenderResult(
                "", False, 0, "no_learned_order_model",
                self.observation.run_id, self.observation.graph_size)
        request = SurfaceStructureRequest(
            semantic, response_act, register, ordered_roles,
            1, 4096, 0, source_id, context_id, family_id, slot_values,
        )
        result = realize_surface_order(model, request)
        if result.status_code != ORDER_SELECTED or result.surface is None:
            return SurfaceRenderResult(
                "", False, 0, "order_structure_rejected",
                self.observation.run_id, self.observation.graph_size)
        return SurfaceRenderResult(
            result.surface, True, result.selected_pattern_id,
            "order_selected", self.observation.run_id,
            self.observation.graph_size, result.trace)


def load_trained_surface_runtime(*, project_root: str | Path,
                                 training_run_root: str | Path,
                                 expected_pack_sha256: str | None = None,
                                 require_k_drive: bool = True,
                                 extra_course_paths: tuple[str | Path, ...] = (),
                                 extra_evidence_paths: tuple[str | Path, ...] = (),
                                 extra_variant_course_paths: tuple[str | Path, ...] = (),
                                 extra_variant_evidence_paths: tuple[str | Path, ...] = (),
                                 extra_order_course_paths: tuple[str | Path, ...] = (),
                                 extra_order_evidence_paths: tuple[str | Path, ...] = (),
                                 response_organization_artifact_root: str | Path | None = None,
                                 ) -> TrainedSurfaceRuntime:
    """从公开课程和 K 盘训练 run 建立只读表层消费者。"""
    root = Path(project_root).resolve()
    course_path = root / "data" / "ph2" / "dlg_raw16_surface_organization_v1.jsonl.sample"
    default_evidence_path = (
        root / "data" / "ph2" / "dlg_raw16_surface_slot_evidence_v1.jsonl.sample")
    evidence_paths = _surface_evidence_paths_for_run(
        root, Path(training_run_root).resolve(), default_evidence_path)
    course_paths = (course_path, *tuple(
        Path(item).resolve() for item in extra_course_paths))
    evidence_paths = (*evidence_paths, *tuple(
        Path(item).resolve() for item in extra_evidence_paths))
    if (any(not item.is_file() for item in course_paths)
            or any(not item.is_file() for item in evidence_paths)):
        raise TrainedSurfaceRuntimeError("DLG-RAW-16 公开课程缺失")
    records = tuple(
        item.record for path in course_paths
        for item in load_surface_organization_jsonl(path.read_bytes()))
    evidence_packs = tuple(
        load_surface_evidence_jsonl(path.read_bytes()) for path in evidence_paths)
    if len({item.source_namespace for item in evidence_packs}) != len(evidence_packs):
        raise TrainedSurfaceRuntimeError("表层 evidence source namespace 重复")
    from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
        SurfaceEvidencePack,
    )
    evidence = SurfaceEvidencePack(
        "dlg-raw16-combined-v1", "CC0-1.0",
        tuple(item for pack in evidence_packs for item in pack.entries),
    )
    slot_hints: dict[tuple[str, str, str], set[str]] = {}
    records_by_id = {item.sample_id: item for item in records}
    for entry in evidence.entries:
        record = records_by_id.get(entry.record_id)
        if record is None:
            continue
        variant = next(
            (item for item in record.accepted
             if item.variant_id == entry.variant_id), None)
        if variant is None or entry.end > len(variant.surface):
            continue
        value = (entry.surface_text
                 or variant.surface[entry.start:entry.end]).strip()
        if value:
            slot_hints.setdefault(
                (record.dialogue_act, record.register, entry.role),
                set()).add(value)
    variant_models: dict[tuple[str, str], SurfaceVariantModel] = {}
    variant_course_paths = tuple(Path(item).resolve()
                                 for item in extra_variant_course_paths)
    variant_evidence_paths = tuple(Path(item).resolve()
                                   for item in extra_variant_evidence_paths)
    if len(variant_course_paths) != len(variant_evidence_paths):
        raise TrainedSurfaceRuntimeError(
            "variant course/evidence 数量必须一一对应")
    for course, evidence_path in zip(variant_course_paths,
                                    variant_evidence_paths):
        if not course.is_file() or not evidence_path.is_file():
            raise TrainedSurfaceRuntimeError("variant 课程或 evidence 缺失")
        variant_records = tuple(
            item.record for item in load_surface_organization_jsonl(
                course.read_bytes()))
        variant_evidence = load_surface_evidence_jsonl(evidence_path.read_bytes())
        for key in sorted({(item.dialogue_act, item.register)
                            for item in variant_records}):
            selected = tuple(item for item in variant_records
                             if (item.dialogue_act, item.register) == key)
            variant_models[key] = learn_surface_variant_model(
                selected, variant_evidence)
    order_models: dict[tuple[str, str], SurfaceOrderModel] = {}
    order_course_paths = tuple(Path(item).resolve()
                               for item in extra_order_course_paths)
    order_evidence_paths = tuple(Path(item).resolve()
                                 for item in extra_order_evidence_paths)
    if len(order_course_paths) != len(order_evidence_paths):
        raise TrainedSurfaceRuntimeError(
            "order course/evidence 数量必须一一对应")
    for course, evidence_path in zip(order_course_paths, order_evidence_paths):
        if not course.is_file() or not evidence_path.is_file():
            raise TrainedSurfaceRuntimeError("order 课程或 evidence 缺失")
        order_records = tuple(item.record for item in
                              load_surface_organization_jsonl(course.read_bytes()))
        order_evidence = load_surface_evidence_jsonl(evidence_path.read_bytes())
        for key in sorted({(item.dialogue_act, item.register)
                            for item in order_records}):
            selected = tuple(item for item in order_records
                             if (item.dialogue_act, item.register) == key)
            order_models[key] = learn_surface_order_model(selected, order_evidence)
    models: dict[tuple[str, str], SurfaceStructureModel] = {}
    for key in sorted({(item.dialogue_act, item.register) for item in records}):
        selected = tuple(item for item in records
                         if (item.dialogue_act, item.register) == key)
        try:
            models[key] = learn_surface_structure_model(selected, evidence)
        except (RuntimeError, TypeError, ValueError):
            # 某些 act/register 只有一个 family，保持 fail-closed；它们仍可
            # 由原回答侧输出，不能伪造为已学结构。
            continue
    if ("ANSWER", "neutral") not in models:
        raise TrainedSurfaceRuntimeError("缺少跨 family ANSWER/neutral 结构")
    # load_training_observation 同时核对 pack SHA、summary 和 SQLite 表计数，
    # 因此这里不是只读取一个孤立 JSON 状态字段。
    from pure_integer_ai.experiments.conversation_dialogue_scale_showcase import (
        load_training_observation,
    )
    observation = load_training_observation(
        training_run_root, expected_pack_sha256=expected_pack_sha256,
        require_k_drive=require_k_drive,
    )
    if observation.graph_size <= 0 or observation.concept_node_count <= 0:
        raise TrainedSurfaceRuntimeError("训练 SQLite 图为空")
    organization_model = None
    if response_organization_artifact_root is not None:
        from pure_integer_ai.experiments.build_response_organization_artifact import (
            load_response_organization_artifact,
        )
        artifact = load_response_organization_artifact(
            response_organization_artifact_root,
            expected_run_id=observation.run_id,
            expected_pack_sha256=observation.pack_sha256,
            require_k_drive=require_k_drive,
        )
        organization_model = artifact.model
    frozen_slot_hints = {
        key: tuple(sorted(values))
        for key, values in slot_hints.items()
    }
    return TrainedSurfaceRuntime(
        observation, models, variant_models, order_models, organization_model,
        frozen_slot_hints,
    )


__all__ = [
    "SurfaceRenderResult", "TrainedSurfaceRuntime",
    "TrainedSurfaceRuntimeError", "load_trained_surface_runtime",
]


def _match_pattern(surface: str, roles: tuple[str, ...],
                   gaps: tuple[str, ...]) -> tuple[str, ...] | None:
    """按已学习 literal gap 提取 typed slot，不使用语义猜测。"""
    if len(gaps) != len(roles) + 1 or not surface.startswith(gaps[0]):
        return None
    cursor = len(gaps[0])
    values: list[str] = []
    for index in range(len(roles)):
        following = gaps[index + 1]
        if index == len(roles) - 1:
            if not surface.endswith(following):
                return None
            end = len(surface) - len(following)
            if end < cursor:
                return None
            values.append(surface[cursor:end])
            cursor = end
            continue
        if following:
            end = surface.find(following, cursor)
            if end < cursor:
                return None
            values.append(surface[cursor:end])
            cursor = end + len(following)
        else:
            next_gap = next((item for item in gaps[index + 2:] if item), "")
            if not next_gap:
                values.append(surface[cursor:])
                cursor = len(surface)
            else:
                end = surface.find(next_gap, cursor)
                if end < cursor:
                    return None
                values.append(surface[cursor:end])
                cursor = end
    if not surface.endswith(gaps[-1]) or cursor > len(surface) - len(gaps[-1]):
        return None
    if cursor != len(surface) - len(gaps[-1]):
        return None
    # Consecutive empty gaps do not provide a unique boundary for adjacent
    # slots.  Reject the extraction instead of inventing an empty slot or
    # silently merging two typed values.
    if any(not value for value in values):
        return None
    return tuple(values)


def _match_learned_pattern(
        surface: str,
        roles: tuple[str, ...],
        gaps: tuple[str, ...],
        hints: Mapping[str, tuple[str, ...]],
        ) -> tuple[str, ...] | None:
    """按 evidence 学到的 slot hints 解析一条完整结构。

    ``gaps`` 是课程投影的 literal；slot 值只能来自独立 evidence 中的
    词位，因而这里不需要也不允许任何语言特定 cue 表。
    """
    if len(gaps) != len(roles) + 1:
        return None

    matches: list[tuple[str, ...]] = []

    def visit(index: int, cursor: int, values: tuple[str, ...]) -> None:
        gap = gaps[index]
        if not surface.startswith(gap, cursor):
            return
        cursor += len(gap)
        if index == len(roles):
            if cursor == len(surface):
                matches.append(values)
            return
        for value in hints.get(roles[index], ()):
            if surface.startswith(value, cursor):
                visit(
                    index + 1,
                    cursor + len(value),
                    (*values, value),
                )
                if len(matches) > 1:
                    return

    visit(0, 0, ())
    return matches[0] if len(matches) == 1 else None


def _semantic_from_slots(response_act: str, roles: tuple[str, ...],
                         values: tuple[str, ...]) -> SurfaceSemantic:
    """从已显式提取的 slot 值建立最小 typed semantic 外壳。"""
    fields = {"subject": "runtime", "predicate": "runtime",
              "object": "runtime"}
    for role, value in zip(roles, values):
        if role in {"subject", "topic", "cause"}:
            fields["subject"] = value
        elif role in {"predicate", "relation"}:
            fields["predicate"] = value
        elif role in {"object", "claim", "effect"}:
            fields["object"] = value
    return SurfaceSemantic(
        f"runtime-{response_act.lower()}", response_act.lower(),
        fields["subject"], fields["predicate"], fields["object"],
    )
