"""DLG-05 v4 的公开、无标签问答 runtime 与 synthetic fixture 边界。

本模块不读取旧 preflight catalog、历史 observation、owner label 或 formal artifact。
runtime 只消费调用方显式传入的 typed source capsule；模块内的六 case helper 明确是
synthetic fixture，仅用于验证 H-00、G-01 与 G-00 至 G-03 的生产接线，绝不能成为
独立来源、held-out source、owner 或 formal 输入。source bundle 从完整 execution
candidate 集合正向导出；每个 candidate 的阅读 surface 与 turn-level response 分开留证。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentProtocol,
    AnswerContentSelection,
    AnswerContentSelector,
)
from pure_integer_ai.cognition.shared.generation_execution import (
    TypedGenerationExecution,
    TypedGenerationExecutor,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    AnswerGenerationGoal,
    GenerationCandidate,
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
    occurrence_identity,
    representation_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.question_answer import (
    EvidenceAnswerPolicy,
    EvidenceAnswerPolicyProtocol,
    FactQuestionExecutor,
    QuestionExecutionResult,
    QuestionQuery,
    QuestionRequest,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    RenderedSurface,
    UnicodeRepresentationRenderer,
    representation_parts,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    ScopeIdentity,
    document_scope,
    episode_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    context_scope_identity,
    entity_identity,
    event_identity as semantic_event_identity,
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    BindingFailureProtocol,
    BoundProposition,
    PropositionSubstituter,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
    SubstitutionProtocol,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.conversation_heldout_v4_bundle import (
    ConversationHeldOutV4DependencyBinding,
    ConversationHeldOutV4ExecutionInput,
    ConversationHeldOutV4Representation,
    ConversationHeldOutV4SourceBundle,
    ConversationHeldOutV4SourceRecord,
    build_v4_source_bundle_from_executions,
    unicode_scalars,
)
from pure_integer_ai.experiments.conversation_heldout_v4_freeze import (
    ConversationHeldOutV4Freeze,
    freeze_v4_bundle,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.facility_generation_scenario import (
    FacilityStructureOrderOwner,
    build_facility_alias_fixture,
    build_facility_generation_plan_protocol,
    build_facility_generation_postcheck_protocol,
    build_facility_generation_surface_protocol,
    build_facility_structure_order_owner,
    build_supporting_generation_verifier,
)
from pure_integer_ai.experiments.ph2_grounded_answer_compile import (
    compile_grounded_answer_training_records_from_payload,
)
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerClaimInput,
    GroundedAnswerConnectorTarget,
    compile_grounded_answer_connectors,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    GroundedAnswerSurfaceModel,
    learn_grounded_answer_surface_model,
)
from pure_integer_ai.experiments.ph2_grounded_answer_parser import (
    GroundedAnswerParserProtocol,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_compile import (
    GroundedResponseActCompileTarget,
    compile_grounded_response_act_patterns,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_parser import (
    GroundedResponseActParserProtocol,
    GroundedResponseActStructureVerifier,
    GroundedResponseActTaskVerifier,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_runtime_factory import (
    GroundedResponseActQuestionInput,
    GroundedResponseActRunLocalBuild,
    GroundedResponseActRunLocalComponents,
    GroundedResponseActRunLocalFactory,
)
from pure_integer_ai.experiments.ph2_grounded_answer_runtime_factory import (
    GroundedAnswerRunLocalBuild,
    GroundedAnswerRunLocalComponents,
    GroundedAnswerRunLocalFactory,
)
from pure_integer_ai.experiments.ph2_grounded_answer_verification import (
    GroundedAnswerEvidenceSourceVerifier,
    GroundedAnswerStructureVerifier,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    EvidenceQuestionPostcheckMapper,
    QuestionAnswerProtocol,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationStructureLayerProtocol,
)
from pure_integer_ai.storage.integer_codec import encode_integer_tuple
from pure_integer_ai.storage.k_run_boundary import (
    KRunBoundaryError,
    KRunFileIdentity,
    KRunRoot,
    capture_plain_file_identity,
    open_existing_run_root,
    open_plain_binary,
    require_plain_file_identity,
)


_NAMESPACE = (20260821, 405, 5)
V4_RUNTIME_FAMILY_KEY = ProtocolKey((*_NAMESPACE, 1))
V4_RUNTIME_INPUT_FAMILY = (*_NAMESPACE, 2)
V4_RUNTIME_PROTOCOL = minimal_instruction_identity((*_NAMESPACE, 3, 1))
V4_RUNTIME_ROUTE = minimal_instruction_identity((*_NAMESPACE, 3, 2))
V4_RUNTIME_EXECUTION_REASON = minimal_instruction_identity((*_NAMESPACE, 3, 3))
_HYPOTHESIS_KIND = (*_NAMESPACE, 4)
V4_RUNTIME_SOURCE_ORIGIN_SYNTHETIC = "SYNTHETIC_RUNTIME_FIXTURE"
V4_RUNTIME_SOURCE_ORIGIN_EXTERNAL = "EXTERNAL_SOURCE_CAPSULE"
V4_RUNTIME_CODE_RELATIVE_PATH = (
    "src/pure_integer_ai/experiments/conversation_heldout_v4_candidate_runtime.py")
V4_RUNTIME_CODE_CLOSURE_SCHEMA = "EXPLICIT_LOCAL_IMPORT_CLOSURE_V1"
V4_RUNTIME_SURFACE_SAMPLE_RELATIVE_PATH = (
    "data/ph2/grounded_answer_train_v1.jsonl.sample")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
V4_RUNTIME_EXECUTION_CODE_RELATIVE_PATHS = (
    "src/pure_integer_ai/__init__.py",
    "src/pure_integer_ai/algorithm/__init__.py",
    "src/pure_integer_ai/algorithm/closure.py",
    "src/pure_integer_ai/algorithm/graph_algebra.py",
    "src/pure_integer_ai/cognition/__init__.py",
    "src/pure_integer_ai/cognition/process/__init__.py",
    "src/pure_integer_ai/cognition/process/abstraction.py",
    "src/pure_integer_ai/cognition/result/__init__.py",
    "src/pure_integer_ai/cognition/result/graph_view.py",
    "src/pure_integer_ai/cognition/shared/__init__.py",
    "src/pure_integer_ai/cognition/shared/alias_resolution.py",
    "src/pure_integer_ai/cognition/shared/attractor_state.py",
    "src/pure_integer_ai/cognition/shared/candidate_projection.py",
    "src/pure_integer_ai/cognition/shared/candidate_runtime.py",
    "src/pure_integer_ai/cognition/shared/candidate_verifier.py",
    "src/pure_integer_ai/cognition/shared/concept_index.py",
    "src/pure_integer_ai/cognition/shared/edge_types.py",
    "src/pure_integer_ai/cognition/shared/evidence_candidate.py",
    "src/pure_integer_ai/cognition/shared/formal_artifact.py",
    "src/pure_integer_ai/cognition/shared/formal_artifact_bridge.py",
    "src/pure_integer_ai/cognition/shared/generation_content.py",
    "src/pure_integer_ai/cognition/shared/generation_execution.py",
    "src/pure_integer_ai/cognition/shared/generation_plan.py",
    "src/pure_integer_ai/cognition/shared/generation_response.py",
    "src/pure_integer_ai/cognition/shared/generation_structure_execution.py",
    "src/pure_integer_ai/cognition/shared/generation_structure_plan.py",
    "src/pure_integer_ai/cognition/shared/generation_surface.py",
    "src/pure_integer_ai/cognition/shared/generation_verification.py",
    "src/pure_integer_ai/cognition/shared/graph_ontology.py",
    "src/pure_integer_ai/cognition/shared/hub_detect.py",
    "src/pure_integer_ai/cognition/shared/hypothesis.py",
    "src/pure_integer_ai/cognition/shared/hypothesis_resolution.py",
    "src/pure_integer_ai/cognition/shared/identity.py",
    "src/pure_integer_ai/cognition/shared/logic_executor.py",
    "src/pure_integer_ai/cognition/shared/memory_aggregate.py",
    "src/pure_integer_ai/cognition/shared/memory_batch.py",
    "src/pure_integer_ai/cognition/shared/memory_event.py",
    "src/pure_integer_ai/cognition/shared/memory_event_log.py",
    "src/pure_integer_ai/cognition/shared/memory_generation.py",
    "src/pure_integer_ai/cognition/shared/memory_overlay.py",
    "src/pure_integer_ai/cognition/shared/memory_owner.py",
    "src/pure_integer_ai/cognition/shared/memory_query.py",
    "src/pure_integer_ai/cognition/shared/memory_resolver.py",
    "src/pure_integer_ai/cognition/shared/order_hypothesis.py",
    "src/pure_integer_ai/cognition/shared/question_answer.py",
    "src/pure_integer_ai/cognition/shared/reasoning_planner.py",
    "src/pure_integer_ai/cognition/shared/relation_closure.py",
    "src/pure_integer_ai/cognition/shared/relation_use.py",
    "src/pure_integer_ai/cognition/shared/representation_rendering.py",
    "src/pure_integer_ai/cognition/shared/scope_identity.py",
    "src/pure_integer_ai/cognition/shared/scoped_persistence.py",
    "src/pure_integer_ai/cognition/shared/semantic_graph.py",
    "src/pure_integer_ai/cognition/shared/semantic_object.py",
    "src/pure_integer_ai/cognition/shared/source_trust.py",
    "src/pure_integer_ai/cognition/shared/structure_order.py",
    "src/pure_integer_ai/cognition/shared/structure_order_consumer.py",
    "src/pure_integer_ai/cognition/shared/structure_order_lifecycle.py",
    "src/pure_integer_ai/cognition/shared/training_hypothesis.py",
    "src/pure_integer_ai/cognition/shared/typed_binding.py",
    "src/pure_integer_ai/cognition/shared/typed_relation.py",
    "src/pure_integer_ai/cognition/shared/types.py",
    "src/pure_integer_ai/cognition/shared/unicode_representation.py",
    "src/pure_integer_ai/cognition/shared/work_memory.py",
    "src/pure_integer_ai/cognition/shared/work_memory_content.py",
    "src/pure_integer_ai/cognition/understanding/__init__.py",
    "src/pure_integer_ai/cognition/understanding/emergent_role.py",
    "src/pure_integer_ai/cognition/understanding/memory_intake.py",
    "src/pure_integer_ai/cognition/understanding/modification_direction.py",
    "src/pure_integer_ai/cognition/understanding/order_constraint_promotion.py",
    "src/pure_integer_ai/cognition/understanding/role_scheme.py",
    "src/pure_integer_ai/cognition/understanding/source_intake.py",
    "src/pure_integer_ai/config/__init__.py",
    "src/pure_integer_ai/config/gates.py",
    "src/pure_integer_ai/crosscut/__init__.py",
    "src/pure_integer_ai/crosscut/determinism/__init__.py",
    "src/pure_integer_ai/crosscut/determinism/audit_event.py",
    "src/pure_integer_ai/crosscut/determinism/fingerprint.py",
    "src/pure_integer_ai/crosscut/determinism/hasher.py",
    "src/pure_integer_ai/crosscut/environment.py",
    "src/pure_integer_ai/crosscut/guards/__init__.py",
    "src/pure_integer_ai/crosscut/guards/float_guard.py",
    "src/pure_integer_ai/crosscut/guards/int_blocker.py",
    "src/pure_integer_ai/crosscut/integer/__init__.py",
    "src/pure_integer_ai/crosscut/integer/unicode_codec.py",
    "src/pure_integer_ai/crosscut/integer/valtypes.py",
    "src/pure_integer_ai/experiments/__init__.py",
    "src/pure_integer_ai/experiments/alias_relation_runtime.py",
    "src/pure_integer_ai/experiments/collection.py",
    "src/pure_integer_ai/experiments/conversation_heldout_v4_bundle.py",
    "src/pure_integer_ai/experiments/conversation_heldout_v4_candidate_runtime.py",
    "src/pure_integer_ai/experiments/conversation_heldout_v4_freeze.py",
    "src/pure_integer_ai/experiments/corpus_identity.py",
    "src/pure_integer_ai/experiments/evaluation_protocol.py",
    "src/pure_integer_ai/experiments/facility_generation_scenario.py",
    "src/pure_integer_ai/experiments/generation_surface_runtime.py",
    "src/pure_integer_ai/experiments/generation_verification_runtime.py",
    "src/pure_integer_ai/experiments/language_generation_connector.py",
    "src/pure_integer_ai/experiments/ph2_dataset_contract.py",
    "src/pure_integer_ai/experiments/ph2_dataset_core.py",
    "src/pure_integer_ai/experiments/ph2_dataset_manifest.py",
    "src/pure_integer_ai/experiments/ph2_dataset_owner_records.py",
    "src/pure_integer_ai/experiments/ph2_dataset_records.py",
    "src/pure_integer_ai/experiments/ph2_dataset_validation.py",
    "src/pure_integer_ai/experiments/ph2_generation_choice_contract.py",
    "src/pure_integer_ai/experiments/ph2_grounded_answer_choice.py",
    "src/pure_integer_ai/experiments/ph2_grounded_answer_compile.py",
    "src/pure_integer_ai/experiments/ph2_grounded_answer_connector.py",
    "src/pure_integer_ai/experiments/ph2_grounded_answer_course.py",
    "src/pure_integer_ai/experiments/ph2_grounded_answer_learning.py",
    "src/pure_integer_ai/experiments/ph2_grounded_answer_order.py",
    "src/pure_integer_ai/experiments/ph2_grounded_answer_parser.py",
    "src/pure_integer_ai/experiments/ph2_grounded_answer_response_act_choice_use.py",
    "src/pure_integer_ai/experiments/ph2_grounded_answer_response_act_compile.py",
    "src/pure_integer_ai/experiments/ph2_grounded_answer_response_act_parser.py",
    "src/pure_integer_ai/experiments/ph2_grounded_answer_response_act_runtime_factory.py",
    "src/pure_integer_ai/experiments/ph2_grounded_answer_runtime_factory.py",
    "src/pure_integer_ai/experiments/ph2_grounded_answer_verification.py",
    "src/pure_integer_ai/experiments/question_answer_runtime.py",
    "src/pure_integer_ai/experiments/relation_closure_runtime.py",
    "src/pure_integer_ai/experiments/train_context.py",
    "src/pure_integer_ai/experiments/verification_orchestration.py",
    "src/pure_integer_ai/storage/__init__.py",
    "src/pure_integer_ai/storage/abstract_mark.py",
    "src/pure_integer_ai/storage/assertion_identity.py",
    "src/pure_integer_ai/storage/assertion_record.py",
    "src/pure_integer_ai/storage/audit.py",
    "src/pure_integer_ai/storage/backend.py",
    "src/pure_integer_ai/storage/backend_capability.py",
    "src/pure_integer_ai/storage/chapter_seq.py",
    "src/pure_integer_ai/storage/composes_attr.py",
    "src/pure_integer_ai/storage/concept_correspondence.py",
    "src/pure_integer_ai/storage/concept_identity.py",
    "src/pure_integer_ai/storage/curriculum_mastery.py",
    "src/pure_integer_ai/storage/discipline.py",
    "src/pure_integer_ai/storage/edge_store.py",
    "src/pure_integer_ai/storage/edge_types.py",
    "src/pure_integer_ai/storage/experience_count.py",
    "src/pure_integer_ai/storage/graph_object.py",
    "src/pure_integer_ai/storage/graph_object_identity.py",
    "src/pure_integer_ai/storage/graph_statement.py",
    "src/pure_integer_ai/storage/integer_codec.py",
    "src/pure_integer_ai/storage/k_run_boundary.py",
    "src/pure_integer_ai/storage/location_manifest.py",
    "src/pure_integer_ai/storage/memory_aggregate.py",
    "src/pure_integer_ai/storage/memory_batch.py",
    "src/pure_integer_ai/storage/memory_event.py",
    "src/pure_integer_ai/storage/memory_forget.py",
    "src/pure_integer_ai/storage/memory_overlay.py",
    "src/pure_integer_ai/storage/memory_query_projection.py",
    "src/pure_integer_ai/storage/node_store.py",
    "src/pure_integer_ai/storage/occurrence.py",
    "src/pure_integer_ai/storage/op_confidence.py",
    "src/pure_integer_ai/storage/placement.py",
    "src/pure_integer_ai/storage/pronoun_resolution_count.py",
    "src/pure_integer_ai/storage/sealed_segment.py",
    "src/pure_integer_ai/storage/segment_cache.py",
    "src/pure_integer_ai/storage/segment_commit.py",
    "src/pure_integer_ai/storage/segment_dependency.py",
    "src/pure_integer_ai/storage/segment_release.py",
    "src/pure_integer_ai/storage/segment_repository.py",
    "src/pure_integer_ai/storage/segment_write_intent.py",
    "src/pure_integer_ai/storage/selection_pref_count.py",
    "src/pure_integer_ai/storage/sense_candidates.py",
    "src/pure_integer_ai/storage/source_record.py",
    "src/pure_integer_ai/storage/source_trust.py",
    "src/pure_integer_ai/storage/spaces/__init__.py",
    "src/pure_integer_ai/storage/spaces/abstract_space.py",
    "src/pure_integer_ai/storage/spaces/companion.py",
    "src/pure_integer_ai/storage/spaces/memory_space.py",
    "src/pure_integer_ai/storage/spaces/registry.py",
    "src/pure_integer_ai/storage/span.py",
    "src/pure_integer_ai/storage/storage_role.py",
    "src/pure_integer_ai/storage/structure_match_count.py",
    "src/pure_integer_ai/storage/telemetry.py",
    "src/pure_integer_ai/storage/tiered_segment_store.py",
    "src/pure_integer_ai/storage/training_candidate_event.py",
    "src/pure_integer_ai/storage/word_form_index.py",
    "src/pure_integer_ai/storage/write_guard.py",
    "src/pure_integer_ai/teacher/__init__.py",
    "src/pure_integer_ai/teacher/probe_set.py",
    "src/pure_integer_ai/teacher/weaning.py",
    "src/pure_integer_ai/teacher/weaning_calibration.py",
)


class ConversationHeldOutV4RuntimeError(RuntimeError):
    """v4 runtime、来源 capsule、选择、renderer 或 receipt 未闭合。"""


def _sha256_bytes(value: bytes) -> tuple[int, ...]:
    """返回规范 SHA-256 的整数序列。"""
    return tuple(hashlib.sha256(value).digest())


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """向整数 receipt 追加带长度边界的一个键。"""
    result.extend((len(value), *value))


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验开放协议键不含 bool、浮点或空值。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise ConversationHeldOutV4RuntimeError(
            f"{label} 必须是非空严格整数 tuple")
    return value


def _digest(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验 receipt 所用 digest 是完整 SHA-256 字节序列。"""
    if (not isinstance(value, tuple) or len(value) != 32
            or any(type(item) is not int or item < 0 or item > 255
                   for item in value)):
        raise ConversationHeldOutV4RuntimeError(f"{label} 必须是 SHA-256")
    return value


def _positive_id(values: tuple[int, ...], *, domain: str) -> int:
    """为 append-only Evidence 建立稳定且严格正的事件 id。"""
    digest = integer_tuple_fingerprint(values, domain=domain)
    value = int.from_bytes(bytes(digest[:8]), "big") & ((1 << 63) - 1)
    return value or 1


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeCodeFile:
    """一个显式登记的本地执行代码文件，不以单一树摘要替代逐文件身份。"""

    relative_path: str
    size: int
    sha256: tuple[int, ...]

    def __post_init__(self) -> None:
        """拒绝绝对路径、非现役包路径、空文件和非 SHA-256 身份。"""
        parts = self.relative_path.split("/") if isinstance(
            self.relative_path, str) else ()
        if (not isinstance(self.relative_path, str)
                or not self.relative_path.startswith("src/pure_integer_ai/")
                or not self.relative_path.endswith(".py")
                or any(part in {"", ".", ".."} for part in parts)):
            raise ConversationHeldOutV4RuntimeError(
                "v4 execution code relative path 非法")
        if type(self.size) is not int or self.size <= 0:
            raise ConversationHeldOutV4RuntimeError(
                "v4 execution code size 必须为正严格整数")
        _digest(self.sha256, label="v4 execution code sha256")

    def stable_key(self) -> tuple[int, ...]:
        """返回路径、长度和完整 SHA 的纯整数文件身份。"""
        result = [self.size]
        _pack(result, tuple(ord(item) for item in self.relative_path))
        _pack(result, self.sha256)
        return tuple(result)


def _execution_code_closure_payload(
        files: tuple[ConversationHeldOutV4RuntimeCodeFile, ...],
        ) -> tuple[int, ...]:
    """为固定代码闭包构造带顺序和边界的完整整数 payload。"""
    result = [1, len(files)]
    for file in files:
        _pack(result, file.stable_key())
    return tuple(result)


def _execution_code_closure_sha256(
        files: tuple[ConversationHeldOutV4RuntimeCodeFile, ...],
        ) -> tuple[int, ...]:
    """对完整整数闭包编码求 SHA-256，便于 manifest 快速比对。"""
    return _sha256_bytes(encode_integer_tuple(
        _execution_code_closure_payload(files)))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeInventory:
    """固定本地执行代码闭包和公开 surface sample 的可复核内容身份。"""

    schema_version: int
    execution_code: tuple[ConversationHeldOutV4RuntimeCodeFile, ...]
    execution_code_total_size: int
    execution_code_closure_sha256: tuple[int, ...]
    surface_sample_path: str
    surface_sample_size: int
    surface_sample_sha256: tuple[int, ...]

    def __post_init__(self) -> None:
        """拒绝未登记、重排、缩减或与聚合身份不一致的执行代码闭包。"""
        if self.schema_version != 2:
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime inventory schema 非法")
        if (not isinstance(self.execution_code, tuple)
                or not self.execution_code
                or any(not isinstance(item,
                                      ConversationHeldOutV4RuntimeCodeFile)
                       for item in self.execution_code)):
            raise ConversationHeldOutV4RuntimeError(
                "v4 execution code inventory 类型非法")
        paths = tuple(item.relative_path for item in self.execution_code)
        if paths != V4_RUNTIME_EXECUTION_CODE_RELATIVE_PATHS:
            raise ConversationHeldOutV4RuntimeError(
                "v4 execution code inventory 与冻结闭包不一致")
        if type(self.execution_code_total_size) is not int or (
                self.execution_code_total_size <= 0):
            raise ConversationHeldOutV4RuntimeError(
                "v4 execution code total size 必须为正严格整数")
        if self.execution_code_total_size != sum(
                item.size for item in self.execution_code):
            raise ConversationHeldOutV4RuntimeError(
                "v4 execution code total size 不一致")
        _digest(self.execution_code_closure_sha256,
                label="v4 execution code closure sha256")
        if self.execution_code_closure_sha256 != _execution_code_closure_sha256(
                self.execution_code):
            raise ConversationHeldOutV4RuntimeError(
                "v4 execution code closure sha256 不一致")
        if self.surface_sample_path != V4_RUNTIME_SURFACE_SAMPLE_RELATIVE_PATH:
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime inventory surface sample path 未注册")
        if type(self.surface_sample_size) is not int or (
                self.surface_sample_size <= 0):
            raise ConversationHeldOutV4RuntimeError(
                "v4 surface sample size 必须为正严格整数")
        _digest(self.surface_sample_sha256,
                label="v4 runtime surface sample sha256")

    def stable_key(self) -> tuple[int, ...]:
        """返回不含绝对路径的完整执行代码闭包和公开 sample 身份。"""
        result = [
            self.schema_version,
            self.execution_code_total_size,
            self.surface_sample_size,
        ]
        _pack(result, tuple(ord(item) for item in
                            V4_RUNTIME_CODE_CLOSURE_SCHEMA))
        _pack(result, self.execution_code_closure_sha256)
        for file in self.execution_code:
            _pack(result, file.stable_key())
        _pack(result, tuple(ord(item) for item in self.surface_sample_path))
        _pack(result, self.surface_sample_sha256)
        return tuple(result)


V4_RUNTIME_STATIC_ASSET_READ_BUDGET_SCHEMA = 1
_RUNTIME_STATIC_ASSET_READ_CHUNK_BYTES = 64 * 1024


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeStaticAssetReadBudget:
    """限制 runtime 固定代码闭包与公开 surface sample 的物理读取。"""

    max_execution_code_file_count: int
    max_execution_code_file_bytes: int
    max_execution_code_total_bytes: int
    max_surface_sample_bytes: int

    def __post_init__(self) -> None:
        """要求所有静态资产预算均为可审计的正严格整数。"""
        values = (
            self.max_execution_code_file_count,
            self.max_execution_code_file_bytes,
            self.max_execution_code_total_bytes,
            self.max_surface_sample_bytes,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("v4 runtime static asset budget 必须是正严格整数")
        if self.max_execution_code_file_bytes > self.max_execution_code_total_bytes:
            raise ValueError("v4 runtime 单个代码文件预算不得超过总代码预算")

    def integer_stream(self) -> tuple[int, ...]:
        """返回版本化预算键，供 C3 receipt/manifest 绑定而不含本机路径。"""
        return (
            V4_RUNTIME_STATIC_ASSET_READ_BUDGET_SCHEMA,
            self.max_execution_code_file_count,
            self.max_execution_code_file_bytes,
            self.max_execution_code_total_bytes,
            self.max_surface_sample_bytes,
        )

    def stable_key(self) -> tuple[int, ...]:
        """兼容 value identity 消费者，稳定键等同版本化整数预算。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeStaticAssetReadCounts:
    """记录同次静态资产物理读取的文件数和实际字节，不能代替内容身份。"""

    execution_code_file_count: int
    execution_code_total_bytes: int
    surface_sample_bytes: int
    static_asset_total_bytes: int

    def __post_init__(self) -> None:
        """闭合代码、sample 与总读取字节，拒绝零长度静态 closure。"""
        values = self.integer_stream()
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("v4 runtime static asset counts 必须是非负严格整数")
        if (self.execution_code_file_count <= 0
                or self.execution_code_total_bytes <= 0
                or self.surface_sample_bytes <= 0
                or self.static_asset_total_bytes != (
                    self.execution_code_total_bytes + self.surface_sample_bytes)):
            raise ValueError("v4 runtime static asset counts 不闭合")

    def integer_stream(self) -> tuple[int, ...]:
        """返回固定顺序的公开聚合计数。"""
        return (
            self.execution_code_file_count,
            self.execution_code_total_bytes,
            self.surface_sample_bytes,
            self.static_asset_total_bytes,
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeStaticAssets:
    """一次受预算读取后的 runtime inventory 与内部可消费 surface payload。"""

    budget: ConversationHeldOutV4RuntimeStaticAssetReadBudget
    inventory: ConversationHeldOutV4RuntimeInventory
    counts: ConversationHeldOutV4RuntimeStaticAssetReadCounts
    test_transport: bool
    surface_sample_payload: bytes = field(repr=False)

    def __post_init__(self) -> None:
        """将 payload、inventory、计数和预算交叉闭合，禁止伪造路径重读输入。"""
        if not isinstance(self.budget, ConversationHeldOutV4RuntimeStaticAssetReadBudget):
            raise TypeError("v4 runtime static asset budget 类型错误")
        if not isinstance(self.inventory, ConversationHeldOutV4RuntimeInventory):
            raise TypeError("v4 runtime static asset inventory 类型错误")
        if not isinstance(self.counts, ConversationHeldOutV4RuntimeStaticAssetReadCounts):
            raise TypeError("v4 runtime static asset counts 类型错误")
        if type(self.test_transport) is not bool:
            raise TypeError("v4 runtime static asset test_transport 必须是 bool")
        if not isinstance(self.surface_sample_payload, bytes) or not self.surface_sample_payload:
            raise ValueError("v4 runtime static asset surface payload 必须是非空 bytes")
        if (self.counts.execution_code_file_count != len(self.inventory.execution_code)
                or self.counts.execution_code_total_bytes
                != self.inventory.execution_code_total_size
                or self.counts.surface_sample_bytes
                != self.inventory.surface_sample_size
                or self.inventory.surface_sample_size
                != len(self.surface_sample_payload)
                or self.inventory.surface_sample_sha256
                != _sha256_bytes(self.surface_sample_payload)):
            raise ValueError("v4 runtime static asset identity/counts 不一致")
        if (self.counts.execution_code_file_count
                > self.budget.max_execution_code_file_count
                or any(file.size > self.budget.max_execution_code_file_bytes
                       for file in self.inventory.execution_code)
                or self.counts.execution_code_total_bytes
                > self.budget.max_execution_code_total_bytes
                or self.counts.surface_sample_bytes
                > self.budget.max_surface_sample_bytes):
            raise ValueError("v4 runtime static asset 超过读取预算")

    def stable_key(self) -> tuple[int, ...]:
        """返回不含正文和绝对路径的 C3 可绑定静态资产身份。"""
        result = [1 if self.test_transport else 0]
        _pack(result, self.budget.integer_stream())
        _pack(result, self.counts.integer_stream())
        _pack(result, self.inventory.stable_key())
        return tuple(result)


V4_RUNTIME_DEFAULT_STATIC_ASSET_READ_BUDGET = (
    ConversationHeldOutV4RuntimeStaticAssetReadBudget(
        len(V4_RUNTIME_EXECUTION_CODE_RELATIVE_PATHS),
        1 * 1024 * 1024,
        16 * 1024 * 1024,
        1 * 1024 * 1024,
    ))


def _runtime_static_asset_root(*, test_transport: bool) -> KRunRoot:
    """以 capability 打开现役仓库；D 盘只允许调用方显式标记 test transport。"""
    if type(test_transport) is not bool:
        raise TypeError("v4 runtime static asset test_transport 必须是 bool")
    try:
        return open_existing_run_root(
            _REPOSITORY_ROOT,
            require_k_drive=not test_transport,
            label="v4 runtime static asset root",
        )
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4RuntimeError(
            "v4 runtime static asset root 物理边界未通过") from exc


def _read_bounded_runtime_static_asset_payload(
        root: KRunRoot,
        relative_path: str,
        identity: KRunFileIdentity,
        *,
        maximum_bytes: int,
        label: str,
        ) -> bytes:
    """只经 K-run capability 流式读取一个已预检身份的静态文件并在读后复核。"""
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("v4 runtime static asset maximum_bytes 必须是正严格整数")
    if not isinstance(identity, KRunFileIdentity):
        raise TypeError("v4 runtime static asset identity 类型错误")
    if identity.byte_count <= 0 or identity.byte_count > maximum_bytes:
        raise ConversationHeldOutV4RuntimeError(
            f"v4 runtime {label} 超过读取预算或为空")
    remaining = identity.byte_count
    chunks: list[bytes] = []
    try:
        with open_plain_binary(
                root,
                relative_path,
                label=f"v4 runtime {label}",
                expected_identity=identity,
                ) as stream:
            while remaining:
                chunk = stream.read(min(
                    _RUNTIME_STATIC_ASSET_READ_CHUNK_BYTES, remaining))
                if not isinstance(chunk, bytes) or not chunk:
                    raise ConversationHeldOutV4RuntimeError(
                        f"v4 runtime {label} 流式读取长度漂移")
                if len(chunk) > remaining:
                    raise ConversationHeldOutV4RuntimeError(
                        f"v4 runtime {label} 流式读取越过预算")
                chunks.append(chunk)
                remaining -= len(chunk)
            if stream.read(1):
                raise ConversationHeldOutV4RuntimeError(
                    f"v4 runtime {label} 流式读取长度漂移")
        require_plain_file_identity(
            root,
            relative_path,
            identity,
            label=f"v4 runtime {label}",
        )
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4RuntimeError(
            f"v4 runtime {label} 文件身份漂移或物理边界未通过") from exc
    except OSError as exc:
        raise ConversationHeldOutV4RuntimeError(
            f"v4 runtime {label} 流式读取失败") from exc
    payload = b"".join(chunks)
    if len(payload) != identity.byte_count:
        raise ConversationHeldOutV4RuntimeError(
            f"v4 runtime {label} 流式读取长度不闭合")
    return payload


def read_v4_runtime_static_assets(
        budget: ConversationHeldOutV4RuntimeStaticAssetReadBudget,
        *,
        test_transport: bool,
        ) -> ConversationHeldOutV4RuntimeStaticAssets:
    """受预算读取冻结代码闭包和 sample；所有预算在任一 payload 打开前已检查。"""
    if not isinstance(budget, ConversationHeldOutV4RuntimeStaticAssetReadBudget):
        raise TypeError("v4 runtime static asset budget 类型错误")
    paths = V4_RUNTIME_EXECUTION_CODE_RELATIVE_PATHS
    if (not isinstance(paths, tuple) or not paths
            or paths != tuple(sorted(paths))
            or len(set(paths)) != len(paths)
            or V4_RUNTIME_CODE_RELATIVE_PATH not in paths):
        raise ConversationHeldOutV4RuntimeError(
            "v4 execution code 冻结清单非法")
    if len(paths) > budget.max_execution_code_file_count:
        raise ConversationHeldOutV4RuntimeError(
            "v4 runtime execution code file count 超过读取预算")
    root = _runtime_static_asset_root(test_transport=test_transport)
    try:
        code_identities = tuple((
            relative_path,
            capture_plain_file_identity(
                root,
                relative_path,
                label="v4 runtime execution code",
            ),
        ) for relative_path in paths)
        sample_identity = capture_plain_file_identity(
            root,
            V4_RUNTIME_SURFACE_SAMPLE_RELATIVE_PATH,
            label="v4 runtime surface train sample",
        )
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4RuntimeError(
            "v4 runtime static asset 预检物理边界未通过") from exc
    code_total_bytes = sum(identity.byte_count for _, identity in code_identities)
    if (any(identity.byte_count > budget.max_execution_code_file_bytes
            for _, identity in code_identities)
            or code_total_bytes > budget.max_execution_code_total_bytes
            or sample_identity.byte_count > budget.max_surface_sample_bytes):
        raise ConversationHeldOutV4RuntimeError(
            "v4 runtime static asset 预检超过读取预算")
    execution_code = tuple(ConversationHeldOutV4RuntimeCodeFile(
        relative_path,
        len(payload := _read_bounded_runtime_static_asset_payload(
            root,
            relative_path,
            identity,
            maximum_bytes=budget.max_execution_code_file_bytes,
            label="execution code",
        )),
        _sha256_bytes(payload),
    ) for relative_path, identity in code_identities)
    surface_sample_payload = _read_bounded_runtime_static_asset_payload(
        root,
        V4_RUNTIME_SURFACE_SAMPLE_RELATIVE_PATH,
        sample_identity,
        maximum_bytes=budget.max_surface_sample_bytes,
        label="surface train sample",
    )
    inventory = ConversationHeldOutV4RuntimeInventory(
        2,
        execution_code,
        sum(file.size for file in execution_code),
        _execution_code_closure_sha256(execution_code),
        V4_RUNTIME_SURFACE_SAMPLE_RELATIVE_PATH,
        len(surface_sample_payload),
        _sha256_bytes(surface_sample_payload),
    )
    return ConversationHeldOutV4RuntimeStaticAssets(
        budget,
        inventory,
        ConversationHeldOutV4RuntimeStaticAssetReadCounts(
            len(execution_code),
            inventory.execution_code_total_size,
            inventory.surface_sample_size,
            inventory.execution_code_total_size + inventory.surface_sample_size,
        ),
        test_transport,
        surface_sample_payload,
    )


def read_v4_runtime_inventory() -> ConversationHeldOutV4RuntimeInventory:
    """兼容旧 inventory 入口；实际读取已改为显式 test transport 有界 static loader。"""
    return read_v4_runtime_static_assets(
        V4_RUNTIME_DEFAULT_STATIC_ASSET_READ_BUDGET,
        test_transport=True,
    ).inventory


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeEvidencePlan:
    """一个授权目标的 Evidence stance 及其来源原文范围。"""

    target: BoundProposition
    competition_key: tuple[int, ...]
    stances: tuple[int, ...]
    source: SourceRef
    source_span_start: int
    source_span_end: int

    def __post_init__(self) -> None:
        """拒绝非 typed target、无来源范围、空计划或重复方向。"""
        if not isinstance(self.target, BoundProposition):
            raise TypeError("v4 runtime evidence target 类型错误")
        _strict_key(self.competition_key, label="v4 runtime competition key")
        if (not isinstance(self.stances, tuple) or not self.stances
                or any(item not in {
                    EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN,
                } for item in self.stances)
                or len(set(self.stances)) != len(self.stances)):
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime evidence stances 非法")
        if not isinstance(self.source, SourceRef):
            raise TypeError("v4 runtime evidence source 类型错误")
        if (type(self.source_span_start) is not int
                or type(self.source_span_end) is not int
                or self.source_span_start < 0
                or self.source_span_start >= self.source_span_end):
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime evidence source span 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回 target、竞争组、stance 和来源范围的完整输入身份。"""
        result = []
        for value in (
                self.target.stable_key(), self.competition_key, self.stances,
                self.source.stable_key(),
                (self.source_span_start, self.source_span_end)):
            _pack(result, value)
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeInput:
    """一个 v4 turn 的公开 typed 输入和 ledger Evidence 计划。"""

    case_key: ProtocolKey
    turn_key: ProtocolKey
    ordinal: int
    request: QuestionRequest
    representations: tuple[ConversationHeldOutV4Representation, ...]
    source_records: tuple[ConversationHeldOutV4SourceRecord, ...]
    evidence_plans: tuple[ConversationHeldOutV4RuntimeEvidencePlan, ...]

    def __post_init__(self) -> None:
        """闭合 source、target 和输入 Representation，不携带 label 或 expected。"""
        if not isinstance(self.case_key, ProtocolKey):
            raise TypeError("v4 runtime case_key 类型错误")
        if not isinstance(self.turn_key, ProtocolKey):
            raise TypeError("v4 runtime turn_key 类型错误")
        if type(self.ordinal) is not int or self.ordinal <= 0:
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime ordinal 必须为正整数")
        if not isinstance(self.request, QuestionRequest):
            raise TypeError("v4 runtime request 类型错误")
        if (not isinstance(self.representations, tuple)
                or not self.representations
                or any(not isinstance(item, ConversationHeldOutV4Representation)
                       for item in self.representations)):
            raise TypeError("v4 runtime representations 类型错误")
        if tuple(item.ordinal for item in self.representations) != tuple(
                range(len(self.representations))):
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime representation ordinal 不连续")
        if (not isinstance(self.source_records, tuple)
                or not self.source_records
                or any(not isinstance(item, ConversationHeldOutV4SourceRecord)
                       for item in self.source_records)):
            raise TypeError("v4 runtime source_records 类型错误")
        source_map = {item.source: item for item in self.source_records}
        if len(source_map) != len(self.source_records):
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime source_records 不得重复")
        if self.request.source not in source_map:
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime request source 缺 SourceRecord")
        if (not isinstance(self.evidence_plans, tuple)
                or any(not isinstance(item, ConversationHeldOutV4RuntimeEvidencePlan)
                       for item in self.evidence_plans)):
            raise TypeError("v4 runtime evidence_plans 类型错误")
        for plan in self.evidence_plans:
            record = source_map.get(plan.source)
            if record is None:
                raise ConversationHeldOutV4RuntimeError(
                    "v4 runtime Evidence 缺少 SourceRecord")
            if plan.source != self.request.source:
                raise ConversationHeldOutV4RuntimeError(
                    "v4 runtime 当前不接受跨请求来源 Evidence")
            if plan.source_span_end > len(record.raw_text_scalars):
                raise ConversationHeldOutV4RuntimeError(
                    "v4 runtime Evidence source span 越过原文")
        targets = self.request.authorized_candidate_targets or (
            self.request.target,)
        scope = self.request.response_scope
        boundary_values = [
            self.request.target_branch,
            *(value for target in targets for value in (
                target.template, target.predicate, target.structure)),
        ]
        if any(value is None or value.owner != scope.owner
               or value.versions != scope.versions
               for value in boundary_values):
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime request target crosses response scope owner/version")
        planned = tuple(item.target for item in self.evidence_plans)
        if len(set(planned)) != len(planned):
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime evidence plan target 不得重复")
        if planned and set(planned) != set(targets):
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime partial candidate Evidence 不得进入执行")

    def stable_key(self) -> tuple[int, ...]:
        """返回完整公开输入、来源和 ledger 计划的确定性身份。"""
        result = []
        _pack(result, self.case_key.components)
        _pack(result, self.turn_key.components)
        result.append(self.ordinal)
        for value in (self.request.stable_key(),):
            _pack(result, value)
        result.append(len(self.representations))
        for item in self.representations:
            _pack(result, item.stable_key())
        result.append(len(self.source_records))
        for item in self.source_records:
            _pack(result, item.stable_key())
        result.append(len(self.evidence_plans))
        for item in self.evidence_plans:
            _pack(result, item.stable_key())
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeSourceCapsule:
    """一次 runtime 可消费的完整 typed 输入及其来源冻结身份。

    ``SYNTHETIC_RUNTIME_FIXTURE`` 只允许公开回归使用；未来 artifact/owner/formal
    路径必须显式要求 ``EXTERNAL_SOURCE_CAPSULE``，并由专门的只读导入器建立。
    """

    origin: str
    manifest_sha256: tuple[int, ...]
    dependencies: ConversationHeldOutV4DependencyBinding
    inputs: tuple[ConversationHeldOutV4RuntimeInput, ...]
    external_producer_key: ProtocolKey | None = None
    external_producer_declaration: str | None = None

    def __post_init__(self) -> None:
        """闭合输入序、来源依赖和 manifest，拒绝默认或隐式 fixture 回落。"""
        if self.origin not in {
                V4_RUNTIME_SOURCE_ORIGIN_SYNTHETIC,
                V4_RUNTIME_SOURCE_ORIGIN_EXTERNAL}:
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime source capsule origin 未注册")
        _digest(self.manifest_sha256, label="v4 runtime capsule manifest_sha256")
        if not isinstance(
                self.dependencies, ConversationHeldOutV4DependencyBinding):
            raise TypeError("v4 runtime source capsule dependencies 类型错误")
        if (not isinstance(self.inputs, tuple) or not self.inputs
                or any(not isinstance(item, ConversationHeldOutV4RuntimeInput)
                       for item in self.inputs)):
            raise TypeError("v4 runtime source capsule inputs 必须非空")
        identities = tuple((item.case_key, item.turn_key) for item in self.inputs)
        if len(set(identities)) != len(identities):
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime source capsule case/turn 不得重复")
        if tuple(item.ordinal for item in self.inputs) != tuple(
                range(1, len(self.inputs) + 1)):
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime source capsule ordinal 必须连续")
        if self.origin == V4_RUNTIME_SOURCE_ORIGIN_EXTERNAL:
            if not isinstance(self.external_producer_key, ProtocolKey):
                raise ConversationHeldOutV4RuntimeError(
                    "external runtime source capsule 缺 producer key")
            if (not isinstance(self.external_producer_declaration, str)
                    or not self.external_producer_declaration):
                raise ConversationHeldOutV4RuntimeError(
                    "external runtime source capsule 缺 producer declaration")
        elif (self.external_producer_key is not None
              or self.external_producer_declaration is not None):
            raise ConversationHeldOutV4RuntimeError(
                "synthetic runtime source capsule 不得携带 external producer")

    @property
    def is_external(self) -> bool:
        """返回该 capsule 是否声明为未来外部来源导入边界。"""
        return self.origin == V4_RUNTIME_SOURCE_ORIGIN_EXTERNAL

    def stable_key(self) -> tuple[int, ...]:
        """返回 origin、manifest、依赖和完整 typed input 的规范身份。"""
        result = [1]
        _pack(result, tuple(ord(item) for item in self.origin))
        _pack(result, self.manifest_sha256)
        _pack(result, self.dependencies.stable_key())
        if self.external_producer_key is None:
            result.append(0)
        else:
            result.append(1)
            _pack(result, self.external_producer_key.components)
            _pack(result, tuple(ord(item) for item in (
                self.external_producer_declaration or "")))
        result.append(len(self.inputs))
        for item in self.inputs:
            _pack(result, item.stable_key())
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeSelectionProtocol:
    """一个 G-01 scope 的实际 content/policy 身份绑定。"""

    scope: ScopeIdentity
    content: AnswerContentProtocol
    policy: EvidenceAnswerPolicyProtocol

    def __post_init__(self) -> None:
        """拒绝把另一个 owner 或版本的 selector 混入当前 G-01 scope。"""
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("v4 runtime selector scope 类型错误")
        if not isinstance(self.content, AnswerContentProtocol):
            raise TypeError("v4 runtime selector content 类型错误")
        if not isinstance(self.policy, EvidenceAnswerPolicyProtocol):
            raise TypeError("v4 runtime selector policy 类型错误")
        identities = (
            *self.content.stances(),
            self.policy.answer_reason,
            self.policy.clarify_reason,
            self.policy.unknown_reason,
            self.policy.conflict_reason,
        )
        if any(item.owner != self.scope.owner
               or item.versions != self.scope.versions
               for item in identities):
            raise ValueError("v4 runtime selector crosses scope owner/version")

    def stable_key(self) -> tuple[int, ...]:
        """返回 scope 与全部实际 G-01 协议身份的完整整数键。"""
        result: list[int] = []
        _pack(result, self.scope.stable_key())
        for item in (
                *self.content.stances(),
                self.policy.answer_reason,
                self.policy.clarify_reason,
                self.policy.unknown_reason,
                self.policy.conflict_reason):
            _pack(result, item.stable_key())
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeIdentity:
    """来源 capsule、各 scope 的 G-01、实际 renderer 和 surface model 身份。"""

    protocol: ObjectIdentity
    route: ObjectIdentity
    execution_reason: ObjectIdentity
    selection_protocols: tuple[ConversationHeldOutV4RuntimeSelectionProtocol, ...]
    capsule: ConversationHeldOutV4RuntimeSourceCapsule
    runtime_inventory: ConversationHeldOutV4RuntimeInventory
    input_sha256: tuple[int, ...]
    surface_model_sha256: tuple[int, ...]
    renderer_sha256: tuple[int, ...]

    def __post_init__(self) -> None:
        """要求运行协议和每个实际 scope 的 selector identity 完整存在。"""
        for label, value in (
                ("protocol", self.protocol),
                ("route", self.route),
                ("execution reason", self.execution_reason)):
            if not isinstance(value, ObjectIdentity):
                raise TypeError(f"v4 runtime {label} 类型错误")
        if (not isinstance(self.selection_protocols, tuple)
                or not self.selection_protocols
                or any(not isinstance(item,
                                      ConversationHeldOutV4RuntimeSelectionProtocol)
                       for item in self.selection_protocols)):
            raise TypeError("v4 runtime selector protocols 类型错误")
        protocols = tuple(sorted(
            self.selection_protocols,
            key=lambda item: item.scope.stable_key(),
        ))
        if len({item.scope for item in protocols}) != len(protocols):
            raise ValueError("v4 runtime selector scope 不得重复")
        object.__setattr__(self, "selection_protocols", protocols)
        if not isinstance(self.capsule, ConversationHeldOutV4RuntimeSourceCapsule):
            raise TypeError("v4 runtime source capsule 类型错误")
        if not isinstance(self.runtime_inventory,
                          ConversationHeldOutV4RuntimeInventory):
            raise TypeError("v4 runtime inventory 类型错误")
        _digest(self.input_sha256, label="v4 runtime input_sha256")
        _digest(self.surface_model_sha256, label="v4 runtime surface_model_sha256")
        _digest(self.renderer_sha256, label="v4 runtime renderer_sha256")

    def stable_key(self) -> tuple[int, ...]:
        """返回足以绑定 runtime、G-01、来源 capsule 和表面模型的整数键。"""
        result = []
        identities = (self.protocol, self.route, self.execution_reason)
        for item in identities:
            _pack(result, item.stable_key())
        result.append(len(self.selection_protocols))
        for protocol in self.selection_protocols:
            _pack(result, protocol.stable_key())
        _pack(result, self.capsule.stable_key())
        _pack(result, self.runtime_inventory.stable_key())
        _pack(result, self.input_sha256)
        _pack(result, self.surface_model_sha256)
        _pack(result, self.renderer_sha256)
        return tuple(result)

    def dependencies(self) -> ConversationHeldOutV4DependencyBinding:
        """返回来源 capsule 冻结的三项依赖，不用 runtime 摘要伪装来源。"""
        return self.capsule.dependencies


# object-model: runtime
class _CountingFactQuestionExecutor(FactQuestionExecutor):
    """在不改变 H-00 读取逻辑的前提下记录每 turn 的真实 execute 次数。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.execute_calls = 0

    def execute(self, query: QuestionQuery) -> QuestionExecutionResult:
        """委托共享 FactQuestionExecutor，并记录一次实际 route 调用。"""
        self.execute_calls += 1
        return super().execute(query)


# object-model: resource-factory; state=mutable
class _AnswerAliasFactory:
    """只从当前 learned ANSWER variant 的 alias requirement 建立 R-01 owner。"""

    def __init__(self, branch: ObjectIdentity) -> None:
        self.branch = branch
        self.fixture = None

    def build(self, variant):
        """从 actual variant 物化全部 slot realization，不查候选表面映射。"""
        if self.fixture is not None:
            raise ConversationHeldOutV4RuntimeError(
                "v4 ANSWER alias factory 不得重复 build")
        self.fixture = build_facility_alias_fixture(
            self.branch,
            tuple((item.filler, item.representation)
                  for item in variant.aliases),
        )
        return self.fixture.runtime

    def close(self) -> None:
        """关闭本 turn 的独占 alias backend。"""
        if self.fixture is not None:
            self.fixture.close()


# object-model: resource-factory; state=mutable
class _ResponseActAliasFactory:
    """只从当前 learned non-answer variant 物化 stance 的实际 realization。"""

    def __init__(self, branch: ObjectIdentity) -> None:
        self.branch = branch
        self.fixture = None

    def build(self, variant):
        """把当前 G-01 stance 接入其 learned Representation。"""
        if self.fixture is not None:
            raise ConversationHeldOutV4RuntimeError(
                "v4 response-act alias factory 不得重复 build")
        self.fixture = build_facility_alias_fixture(
            self.branch,
            ((variant.template.stance, variant.representation),),
        )
        return self.fixture.runtime

    def close(self) -> None:
        """关闭本 turn 的独占 alias backend。"""
        if self.fixture is not None:
            self.fixture.close()


# object-model: resource-owner; state=mutable
@dataclass(slots=True)
class _GenerationOwner:
    """持有一次 actual G-00 至 G-03 的 executor 与两类可关闭图 owner。"""

    executor: TypedGenerationExecutor
    structure: FacilityStructureOrderOwner
    aliases: _AnswerAliasFactory | _ResponseActAliasFactory

    def close(self) -> None:
        """逆序关闭 R-01 与 S-07 资源，不改变已生成 typed receipt。"""
        self.aliases.close()
        self.structure.close()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4CandidateRealization:
    """一个 execution candidate 的独立 G-01/G-00 至 G-03 阅读表面证明。

    它不是整回合的 response：完整回合仍由 ``RuntimeFrame.generation`` 表示。该
    realization 只避免将一个澄清、冲突或拒答表面静态复制给多个 candidate。
    """

    candidate: GenerationCandidate
    planning: GenerationPlanningRequest
    selection: AnswerContentSelection
    generation: TypedGenerationExecution

    def __post_init__(self) -> None:
        """要求单 candidate 计划、同次 G-01 和实际 renderer 全部闭合。"""
        if not isinstance(self.candidate, GenerationCandidate):
            raise TypeError("v4 candidate realization candidate 类型错误")
        if not isinstance(self.planning, GenerationPlanningRequest):
            raise TypeError("v4 candidate realization planning 类型错误")
        if self.planning.candidates != (self.candidate,):
            raise ConversationHeldOutV4RuntimeError(
                "v4 candidate realization 必须只消费一个 execution candidate")
        if (self.planning.goal.proposition != self.candidate.proposition
                or self.planning.goal.source != self.candidate.source
                or self.planning.goal.scope != self.candidate.scope):
            raise ConversationHeldOutV4RuntimeError(
                "v4 candidate realization goal 与 candidate 漂移")
        if not isinstance(self.selection, AnswerContentSelection):
            raise TypeError("v4 candidate realization selection 类型错误")
        if self.selection.request != self.planning:
            raise ConversationHeldOutV4RuntimeError(
                "v4 candidate realization G-01 未绑定单 candidate planning")
        allowed = {self.candidate.stable_key()}
        if not set(self.selection.selected_candidate_keys).issubset(allowed):
            raise ConversationHeldOutV4RuntimeError(
                "v4 candidate realization G-01 选择了计划外 candidate")
        if not isinstance(self.generation, TypedGenerationExecution):
            raise TypeError("v4 candidate realization generation 类型错误")
        if self.generation.plan.request != self.planning:
            raise ConversationHeldOutV4RuntimeError(
                "v4 candidate realization G-00 替换了 planning")
        if not self.generation.complete or self.generation.rendered is None:
            raise ConversationHeldOutV4RuntimeError(
                "v4 candidate realization 未形成实际 renderer 输出")
        if len(self.generation.plan.layers) < 2:
            raise ConversationHeldOutV4RuntimeError(
                "v4 candidate realization 缺少 G-01 stance/content layers")
        stance_layer, content_layer = self.generation.plan.layers[:2]
        selection_key = self.selection.stable_key()
        if (not stance_layer.executed or not content_layer.executed
                or stance_layer.payload != selection_key
                or content_layer.payload != selection_key):
            raise ConversationHeldOutV4RuntimeError(
                "v4 candidate realization G-00 未消费实际 G-01")

    @property
    def rendered(self) -> RenderedSurface:
        """返回这个 candidate 在同次 G-03 中实际生成的阅读表面。"""
        rendered = self.generation.rendered
        if rendered is None:
            raise RuntimeError("v4 candidate realization 缺少 rendered surface")
        return rendered

    def stable_key(self) -> tuple[int, ...]:
        """返回 candidate、计划、G-01、G-00 至 G-03 的完整 receipt 键。"""
        result = []
        for value in (
                self.candidate.stable_key(), self.planning.stable_key(),
                self.selection.stable_key(), self.generation.stable_key(),
                self.rendered.stable_key()):
            _pack(result, value)
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeFrame:
    """一个 turn 的完整 execution、回合 response 与逐候选 realization 证据。"""

    input: ConversationHeldOutV4RuntimeInput
    query: QuestionQuery
    execution: QuestionExecutionResult
    selection: AnswerContentSelection
    selection_protocol: ConversationHeldOutV4RuntimeSelectionProtocol
    generation: TypedGenerationExecution
    candidate_realizations: tuple[ConversationHeldOutV4CandidateRealization, ...]
    executor_calls: int

    def __post_init__(self) -> None:
        """逐段核验完整 candidate、回合 response 和阅读 realization 不串用。"""
        if not isinstance(self.input, ConversationHeldOutV4RuntimeInput):
            raise TypeError("v4 runtime frame input 类型错误")
        if not isinstance(self.query, QuestionQuery):
            raise TypeError("v4 runtime frame query 类型错误")
        if self.query.request != self.input.request:
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime frame query 替换了 request")
        if not isinstance(self.execution, QuestionExecutionResult):
            raise TypeError("v4 runtime frame execution 类型错误")
        if self.execution.query != self.query:
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime frame execution 替换了 query")
        planning = self.execution.planning_request()
        if not isinstance(self.selection, AnswerContentSelection):
            raise TypeError("v4 runtime frame selection 类型错误")
        if self.selection.request != planning:
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime frame G-01 未绑定同次 execution")
        if not isinstance(self.selection_protocol,
                          ConversationHeldOutV4RuntimeSelectionProtocol):
            raise TypeError("v4 runtime frame selection_protocol 类型错误")
        if (planning.goal.scope != self.input.request.response_scope
                or self.selection_protocol.scope != planning.goal.scope
                or self.selection.protocol != self.selection_protocol.content
                or self.selection.reason not in (
                    self.selection_protocol.policy.answer_reason,
                    self.selection_protocol.policy.clarify_reason,
                    self.selection_protocol.policy.unknown_reason,
                    self.selection_protocol.policy.conflict_reason)):
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime frame G-01 selector scope 或 policy 漂移")
        if not isinstance(self.generation, TypedGenerationExecution):
            raise TypeError("v4 runtime frame generation 类型错误")
        if self.generation.plan.request != planning:
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime frame G-00 替换了 execution candidates")
        if not self.generation.complete or self.generation.rendered is None:
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime frame 未形成实际 renderer 输出")
        if len(self.generation.plan.layers) < 2:
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime frame 缺少 G-01 stance/content layers")
        selection_key = self.selection.stable_key()
        stance_layer, content_layer = self.generation.plan.layers[:2]
        if (not stance_layer.executed or not content_layer.executed
                or stance_layer.payload != selection_key
                or content_layer.payload != selection_key):
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime frame G-00 未消费实际 G-01 选择")
        candidate_keys = {item.stable_key() for item in self.execution.candidates}
        if not set(self.selection.selected_candidate_keys).issubset(candidate_keys):
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime G-01 选择了 execution 外 candidate")
        if (not isinstance(self.candidate_realizations, tuple)
                or any(not isinstance(item, ConversationHeldOutV4CandidateRealization)
                       for item in self.candidate_realizations)):
            raise TypeError("v4 runtime frame candidate_realizations 类型错误")
        realization_map = {
            item.candidate.stable_key(): item
            for item in self.candidate_realizations
        }
        if (len(realization_map) != len(self.candidate_realizations)
                or set(realization_map) != candidate_keys):
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime frame 必须为每个 execution candidate 保留唯一 realization")
        if any(item.candidate not in self.execution.candidates
               for item in self.candidate_realizations):
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime realization candidate 不属于同次 execution")
        if any(
                item.planning.goal.scope != self.selection_protocol.scope
                or item.selection.protocol != self.selection_protocol.content
                or item.selection.reason not in (
                    self.selection_protocol.policy.answer_reason,
                    self.selection_protocol.policy.clarify_reason,
                    self.selection_protocol.policy.unknown_reason,
                    self.selection_protocol.policy.conflict_reason)
                for item in self.candidate_realizations):
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime realization selector scope 或 policy 漂移")
        if type(self.executor_calls) is not int or self.executor_calls != 1:
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime 每 turn 必须恰好调用一次真实 executor")

    @property
    def rendered(self) -> RenderedSurface:
        """返回 G-03 已实际生成的同次 Unicode renderer 输出。"""
        rendered = self.generation.rendered
        if rendered is None:
            raise RuntimeError("v4 runtime frame 缺少 rendered surface")
        return rendered

    @property
    def surface_representations(self) -> tuple[
            ConversationHeldOutV4Representation, ...]:
        """合并逐候选实际 renderer 的 Representation，绝不导入回合 response。"""
        values = []
        seen = set()
        for realization in self.candidate_realizations:
            units = []
            for representation in realization.rendered.representations:
                _family, content = representation_parts(representation)
                units.extend(content)
                if representation not in seen:
                    seen.add(representation)
                    values.append(ConversationHeldOutV4Representation(
                        representation, len(values), content))
            if tuple(units) != realization.rendered.units:
                raise ConversationHeldOutV4RuntimeError(
                    "v4 candidate realization renderer units 与 Representation 漂移")
        return tuple(values)

    def render_candidate(
            self, candidate,
            ) -> RenderedSurface:
        """返回 candidate 自己的实际 realization 表面，不复用回合 response。"""
        if not isinstance(candidate, GenerationCandidate):
            raise TypeError("v4 runtime candidate 类型错误")
        if candidate not in self.execution.candidates:
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime candidate 不属于同次 execution")
        matches = tuple(
            item for item in self.candidate_realizations
            if item.candidate == candidate)
        if len(matches) != 1:
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime candidate 缺少唯一 actual realization")
        return matches[0].rendered

    def stable_key(self) -> tuple[int, ...]:
        """返回 query、完整 execution、response 和逐候选 realization 的 receipt 键。"""
        result = []
        for value in (
                self.input.stable_key(), self.query.stable_key(),
                self.execution.stable_key(), self.selection.stable_key(),
                self.selection_protocol.stable_key(),
                self.generation.stable_key(), self.rendered.stable_key(),
                (self.executor_calls,)):
            _pack(result, value)
        result.append(len(self.candidate_realizations))
        for item in self.candidate_realizations:
            _pack(result, item.stable_key())
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeReceipt:
    """不含 owner label 的 runtime receipt，保留实际 selection 与 renderer 证据。"""

    family_key: ProtocolKey
    identity: ConversationHeldOutV4RuntimeIdentity
    frames: tuple[ConversationHeldOutV4RuntimeFrame, ...]
    bundle: ConversationHeldOutV4SourceBundle
    _payload: tuple[int, ...] = field(init=False, repr=False, compare=False)
    payload_sha256: tuple[int, ...] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """绑定全部真实 frame 与同一无标签 bundle，拒绝漏 turn 或 freeze 前替换。"""
        if not isinstance(self.family_key, ProtocolKey):
            raise TypeError("v4 runtime receipt family_key 类型错误")
        if not isinstance(self.identity, ConversationHeldOutV4RuntimeIdentity):
            raise TypeError("v4 runtime receipt identity 类型错误")
        if (not isinstance(self.frames, tuple) or not self.frames
                or any(not isinstance(item, ConversationHeldOutV4RuntimeFrame)
                       for item in self.frames)):
            raise TypeError("v4 runtime receipt frames 类型错误")
        if not isinstance(self.bundle, ConversationHeldOutV4SourceBundle):
            raise TypeError("v4 runtime receipt bundle 类型错误")
        if self.bundle.family_key != self.family_key:
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime receipt bundle family 漂移")
        if self.identity.input_sha256 != _input_sha256(self.identity.capsule.inputs):
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime receipt input identity 漂移")
        if self.identity.renderer_sha256 != _renderer_sha256(self.frames):
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime receipt renderer identity 漂移")
        frame_protocols = tuple(sorted(
            {frame.selection_protocol for frame in self.frames},
            key=lambda item: item.scope.stable_key(),
        ))
        if self.identity.selection_protocols != frame_protocols:
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime receipt selector protocol coverage 漂移")
        if self.bundle.dependencies != self.identity.dependencies():
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime receipt bundle runtime identity 漂移")
        keys = tuple((item.input.case_key, item.input.turn_key)
                     for item in self.frames)
        if len(set(keys)) != len(keys):
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime receipt frame 不得重复")
        if set(keys) != {(item.case_key, item.turn_key)
                         for item in self.bundle.turns}:
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime receipt 与 bundle turn 覆盖漂移")
        for frame in self.frames:
            turn = self.bundle.turn_for(
                frame.input.case_key, frame.input.turn_key)
            if (turn.request != frame.input.request
                    or tuple(item.candidate for item in turn.candidates)
                    != frame.execution.candidates
                    or turn.surface_representations
                    != frame.surface_representations):
                raise ConversationHeldOutV4RuntimeError(
                    "v4 runtime receipt 未从同次 execution/renderer 导出 bundle")
            for exported in turn.candidates:
                rendered = frame.render_candidate(exported.candidate)
                if (exported.surface_scalars != rendered.units
                        or exported.surface_representations
                        != rendered.representations):
                    raise ConversationHeldOutV4RuntimeError(
                        "v4 runtime receipt candidate surface 未绑定其 actual realization")
        payload = self._build_payload()
        object.__setattr__(self, "_payload", payload)
        object.__setattr__(self, "payload_sha256", _sha256_bytes(
            bytes(" ".join(str(value) for value in payload) + "\n", "ascii")))

    def _build_payload(self) -> tuple[int, ...]:
        """构造不含 label 的完整 runtime receipt 整数载荷。"""
        result = [1]
        _pack(result, self.family_key.components)
        _pack(result, self.identity.stable_key())
        result.append(len(self.frames))
        for item in self.frames:
            _pack(result, item.stable_key())
        _pack(result, self.bundle.canonical_payload)
        return tuple(result)

    @property
    def canonical_payload(self) -> tuple[int, ...]:
        """返回完整 receipt payload；摘要永不替代实际执行对象。"""
        return self._payload


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4CandidateRuntimeResult:
    """一次来源 capsule runtime slice 的无标签 bundle、freeze 和真实 receipt。"""

    capsule: ConversationHeldOutV4RuntimeSourceCapsule
    identity: ConversationHeldOutV4RuntimeIdentity
    frames: tuple[ConversationHeldOutV4RuntimeFrame, ...]
    bundle: ConversationHeldOutV4SourceBundle
    freeze: ConversationHeldOutV4Freeze
    receipt: ConversationHeldOutV4RuntimeReceipt

    def __post_init__(self) -> None:
        """确认 receipt、bundle 与 freeze 全部来自同一 actual runtime slice。"""
        if not isinstance(self.capsule, ConversationHeldOutV4RuntimeSourceCapsule):
            raise TypeError("v4 candidate runtime capsule 类型错误")
        if self.identity.capsule != self.capsule:
            raise ConversationHeldOutV4RuntimeError(
                "v4 candidate runtime identity 替换了 source capsule")
        if tuple(item.input for item in self.frames) != self.capsule.inputs:
            raise ConversationHeldOutV4RuntimeError(
                "v4 candidate runtime frame 输入替换了 source capsule")
        if self.bundle.dependencies != self.identity.dependencies():
            raise ConversationHeldOutV4RuntimeError(
                "v4 candidate runtime dependencies 漂移")
        if self.receipt.identity != self.identity or self.receipt.frames != self.frames:
            raise ConversationHeldOutV4RuntimeError(
                "v4 candidate runtime receipt 漂移")
        if self.receipt.bundle != self.bundle:
            raise ConversationHeldOutV4RuntimeError(
                "v4 candidate runtime receipt bundle 漂移")
        if self.freeze != freeze_v4_bundle(self.bundle):
            raise ConversationHeldOutV4RuntimeError(
                "v4 candidate runtime freeze 漂移")


def _source(case: int) -> SourceRef:
    """为代码内 synthetic fixture 建立互异 SourceRef，不声称来源独立。"""
    return SourceRef(
        _NAMESPACE[0], 2_000 + case, case,
        GLOBAL_OWNER_SCOPE, VersionBundle())


def _build_request(
        case: int,
        turn: int,
        *,
        candidate_count: int,
        event_target: bool = False,
        relation_target: bool = False,
        ) -> QuestionRequest:
    """从代码内 synthetic 定义建立完整 QuestionRequest，不接旧 catalog。"""
    if type(candidate_count) is not int or candidate_count <= 0:
        raise ValueError("v4 runtime candidate_count 必须为正整数")
    if event_target and relation_target:
        raise ValueError("v4 runtime target 不得同时声明 event/relation")
    source = _source(case)
    evidence_scope = document_scope(source)
    response_scope = query_scope(
        30_000 + case * 100 + turn,
        parent=episode_scope(40_000 + case * 100 + turn, parent=evidence_scope),
    )
    if event_target:
        bindings = (AtomicRoleBinding(
            role_identity((*_NAMESPACE, 10, case, 1)),
            semantic_event_identity(source, (*_NAMESPACE, 11, case, 1)),
        ),)
    elif relation_target:
        bindings = tuple(
            AtomicRoleBinding(
                role_identity((*_NAMESPACE, 12, case, ordinal)),
                entity_identity(source, (*_NAMESPACE, 13, case, ordinal)),
            )
            for ordinal in (1, 2)
        )
    else:
        bindings = ()
    definitions = tuple(
        AtomicPropositionDefinition(
            proposition_identity(source, (*_NAMESPACE, 20, case, turn, ordinal)),
            concept_identity((*_NAMESPACE, 21, case, turn, ordinal)),
            occurrence_identity(
                source,
                start=case * 10 + turn + ordinal,
                end=case * 10 + turn + ordinal + 1,
                ordinal=0,
            ),
            context_scope_identity(source, (*_NAMESPACE, 22, case, turn, ordinal)),
            bindings,
        )
        for ordinal in range(1, candidate_count + 1)
    )
    graph = PropositionTemplateGraph(tuple(
        ScopedPropositionTemplate(
            definition,
            structure_concept_identity((*_NAMESPACE, 23, case, turn, ordinal)),
        )
        for ordinal, definition in enumerate(definitions, start=1)
    ))
    failures = BindingFailureProtocol(*tuple(
        minimal_instruction_identity((*_NAMESPACE, 24, case, index))
        for index in range(1, 10)
    ))
    substituter = PropositionSubstituter(SubstitutionProtocol(
        minimal_instruction_identity((*_NAMESPACE, 25, case)), failures))
    targets = tuple(
        substituter.substitute(
            definition.proposition, graph, BindingEnvironment())
        for definition in definitions
    )
    return QuestionRequest(
        minimal_instruction_identity((*_NAMESPACE, 26, case, 1)),
        minimal_instruction_identity((*_NAMESPACE, 26, case, 2)),
        minimal_instruction_identity((*_NAMESPACE, 26, case, 3)),
        targets[0],
        LogicEvidenceState(True, False),
        evidence_scope,
        response_scope,
        (*_NAMESPACE, 27, case, turn),
        language_branch_identity((*_NAMESPACE, 28, case)),
        targets if candidate_count > 1 else (),
    )


def _representation(
        text: str, ordinal: int,
        ) -> ConversationHeldOutV4Representation:
    """将公开输入文本正向编为权威 Unicode Representation 映射。"""
    scalars = unicode_scalars(text, allow_empty=False)
    return ConversationHeldOutV4Representation(
        representation_identity(V4_RUNTIME_INPUT_FAMILY, scalars),
        ordinal,
        scalars,
    )


def _input_representations(case: int, turn: int) -> tuple[
        ConversationHeldOutV4Representation, ...]:
    """返回六 case 的公开输入表示；它们不是期望答案或 evaluator label。"""
    texts = {
        (1, 1): ("请核对", "第一案例事实。"),
        (1, 2): ("请确认", "第一案例事实。"),
        (2, 1): ("第二案例", "命题一与命题二。"),
        (2, 2): ("继续上一问", "请给出结论。"),
        (3, 1): ("记录了一个事件", "请确认它。"),
        (3, 2): ("这个事件", "后来怎样？"),
        (4, 1): ("关系组合", "请检查。"),
        (4, 2): ("同一关系", "证据是否一致？"),
        (5, 1): ("第五案例", "查找记录。"),
        (5, 2): ("换一个范围", "再次查找。"),
        (6, 1): ("第六案例", "先记录。"),
        (6, 2): ("再问一次", "保持原问题。"),
    }
    return tuple(
        _representation(text, ordinal)
        for ordinal, text in enumerate(texts[(case, turn)])
    )


def _source_record(case: int) -> ConversationHeldOutV4SourceRecord:
    """为每个 synthetic case 生成公开 fixture 原文、许可、归属和内容 hash。"""
    raw = f"公开 v4 runtime 来源记录：第{case}案例的 typed 事实与结构。"
    return ConversationHeldOutV4SourceRecord(
        _source(case),
        unicode_scalars(raw),
        _sha256_bytes(raw.encode("utf-8")),
        unicode_scalars("CC-BY-4.0"),
        unicode_scalars("pure_integer_ai v4 public runtime successor"),
        unicode_scalars(
            f"urn:pure-integer-ai:dlg05:v4:runtime-source:{case}"),
    )


def _build_v4_synthetic_runtime_inputs() -> tuple[ConversationHeldOutV4RuntimeInput, ...]:
    """构造六 case、十二 turn 的代码内公开 synthetic runtime fixture 输入。"""
    inputs = []
    ordinal = 0
    for case in range(1, 7):
        source = _source_record(case)
        candidate_count = 2 if case == 2 else 1
        for turn in (1, 2):
            ordinal += 1
            request = _build_request(
                case,
                turn,
                candidate_count=candidate_count,
                event_target=case == 3,
                relation_target=case == 4,
            )
            targets = request.authorized_candidate_targets or (request.target,)
            plans = ()
            if case != 5:
                plans = tuple(
                    ConversationHeldOutV4RuntimeEvidencePlan(
                        target,
                        ((*_NAMESPACE, 60, case, turn)
                         if case == 2
                         else (*_NAMESPACE, 61, case, turn, target_ordinal)),
                        ((EVIDENCE_SUPPORT, EVIDENCE_REFUTE)
                         if case == 4 else (EVIDENCE_SUPPORT,)),
                        request.source,
                        0,
                        1,
                    )
                    for target_ordinal, target in enumerate(targets, start=1)
                )
            inputs.append(ConversationHeldOutV4RuntimeInput(
                ProtocolKey((*_NAMESPACE, 70, case)),
                ProtocolKey((*_NAMESPACE, 70, case, turn)),
                ordinal,
                request,
                _input_representations(case, turn),
                (source,),
                plans,
            ))
    return tuple(inputs)


def _model_sha256(model: GroundedAnswerSurfaceModel) -> tuple[int, ...]:
    """从公开训练结果的完整 pattern 值派生 surface model 内容身份。"""
    if not isinstance(model, GroundedAnswerSurfaceModel):
        raise TypeError("v4 runtime model 类型错误")
    payload = json.dumps(
        [item.stable_value() for item in model.patterns],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return _sha256_bytes(payload)


def _input_sha256(
        inputs: tuple[ConversationHeldOutV4RuntimeInput, ...],
        ) -> tuple[int, ...]:
    """从全部 typed 输入完整键派生 runtime 输入身份，不证明来源独立。"""
    values = []
    for item in inputs:
        _pack(values, item.stable_key())
    return _sha256_bytes(bytes(" ".join(str(value) for value in values), "ascii"))


def build_v4_synthetic_runtime_fixture() -> ConversationHeldOutV4RuntimeSourceCapsule:
    """返回显式标记的代码内 fixture，供公开 runtime 回归而非 artifact/formal 使用。"""
    inputs = _build_v4_synthetic_runtime_inputs()
    input_sha256 = _input_sha256(inputs)
    manifest_sha256 = _sha256_bytes(
        b"PIA-DLG05-V4-synthetic-runtime-fixture-manifest-v1\\n"
        + bytes(input_sha256))
    dependencies = ConversationHeldOutV4DependencyBinding(
        _sha256_bytes(b"PIA-DLG05-V4-synthetic-fixture-artifact-v1\\n"
                      + bytes(manifest_sha256)),
        _sha256_bytes(b"PIA-DLG05-V4-synthetic-fixture-inventory-v1\\n"
                      + bytes(manifest_sha256)),
        _sha256_bytes(b"PIA-DLG05-V4-synthetic-fixture-document-v1\\n"
                      + bytes(manifest_sha256)),
    )
    return ConversationHeldOutV4RuntimeSourceCapsule(
        V4_RUNTIME_SOURCE_ORIGIN_SYNTHETIC,
        manifest_sha256,
        dependencies,
        inputs,
    )


def _selector(
        scope: ScopeIdentity,
        ) -> tuple[
        AnswerContentProtocol,
        EvidenceAnswerPolicyProtocol,
        AnswerContentSelector,
        ]:
    """在一个实际 query scope 内建立不接 evaluator label 的 G-01。"""
    if not isinstance(scope, ScopeIdentity):
        raise TypeError("v4 runtime selector scope 类型错误")
    content = AnswerContentProtocol(*tuple(
        minimal_instruction_identity(
            (*_NAMESPACE, 80, index),
            owner=scope.owner,
            versions=scope.versions,
        )
        for index in range(1, 6)
    ))
    policy = EvidenceAnswerPolicyProtocol(*tuple(
        minimal_instruction_identity(
            (*_NAMESPACE, 81, index),
            owner=scope.owner,
            versions=scope.versions,
        )
        for index in range(1, 5)
    ))
    return content, policy, AnswerContentSelector(
        content, EvidenceAnswerPolicy(content, policy))


def _scoped_instruction(
        scope: ScopeIdentity,
        key: tuple[int, ...],
        ) -> ObjectIdentity:
    """为一个实际 generation scope 建立局部最小指令身份。"""
    return minimal_instruction_identity(
        key, owner=scope.owner, versions=scope.versions)


def _generation_scope(
        item: ConversationHeldOutV4RuntimeInput,
        planning: GenerationPlanningRequest,
        ) -> ScopeIdentity:
    """核验 runtime 输入、H-00 planning 和语言目标共享同一边界。"""
    scope = planning.goal.scope
    if scope != item.request.response_scope:
        raise ConversationHeldOutV4RuntimeError(
            "v4 runtime planning scope 与 request response scope 不一致")
    branch = item.request.target_branch
    if branch is None:
        raise ConversationHeldOutV4RuntimeError(
            "v4 runtime generation 缺少目标语言分支")
    values = (
        branch,
        planning.goal.proposition.template,
        planning.goal.proposition.predicate,
        planning.goal.proposition.structure,
    )
    if any(value.owner != scope.owner or value.versions != scope.versions
           for value in values):
        raise ConversationHeldOutV4RuntimeError(
            "v4 runtime generation input crosses owner/version boundary")
    return scope


def _runtime_identity(
        capsule: ConversationHeldOutV4RuntimeSourceCapsule,
        inventory: ConversationHeldOutV4RuntimeInventory,
        model: GroundedAnswerSurfaceModel,
        frames: tuple[ConversationHeldOutV4RuntimeFrame, ...],
        ) -> ConversationHeldOutV4RuntimeIdentity:
    """绑定来源 capsule、G-01、实际 renderer 与 learned surface model identity。"""
    selection_protocols = tuple(sorted(
        {frame.selection_protocol for frame in frames},
        key=lambda item: item.scope.stable_key(),
    ))
    return ConversationHeldOutV4RuntimeIdentity(
        V4_RUNTIME_PROTOCOL,
        V4_RUNTIME_ROUTE,
        V4_RUNTIME_EXECUTION_REASON,
        selection_protocols,
        capsule,
        inventory,
        _input_sha256(capsule.inputs),
        _model_sha256(model),
        _renderer_sha256(frames),
    )


def _renderer_sha256(
        frames: tuple[ConversationHeldOutV4RuntimeFrame, ...],
        ) -> tuple[int, ...]:
    """从回合 response 与逐候选 renderer identity 派生严格内容绑定。"""
    values = []
    for frame in frames:
        _pack(values, frame.rendered.stable_key())
        for realization in frame.candidate_realizations:
            _pack(values, realization.rendered.stable_key())
    return _sha256_bytes(bytes(" ".join(str(value) for value in values), "ascii"))


def _populate_ledger(
        item: ConversationHeldOutV4RuntimeInput,
        ) -> HypothesisLedger:
    """只将当前 typed Evidence 计划追加到新的 in-memory H-00 ledger。"""
    ledger = HypothesisLedger()
    for target_ordinal, plan in enumerate(item.evidence_plans, start=1):
        hypothesis = ledger.register(HypothesisKey(
            _HYPOTHESIS_KIND,
            plan.target.template.stable_key(),
            plan.competition_key,
            item.request.evidence_scope,
            item.request.source,
        ))
        for stance_ordinal, stance in enumerate(plan.stances, start=1):
            evidence_id = _positive_id(
                (*_NAMESPACE, item.ordinal, target_ordinal, stance_ordinal,
                 stance),
                domain="dlg05.v4.runtime.evidence.v1",
            )
            ledger.append_evidence(EvidenceRecord(
                evidence_id,
                hypothesis,
                stance,
                (*_NAMESPACE, 82, item.ordinal, target_ordinal, stance_ordinal),
                plan.source,
                item.ordinal * 100 + target_ordinal * 10 + stance_ordinal,
                (*_NAMESPACE, 83, item.ordinal, target_ordinal, stance_ordinal,
                 plan.source_span_start, plan.source_span_end),
            ))
    return ledger


def _claim_input(
        candidate,
        source_records: tuple[ConversationHeldOutV4SourceRecord, ...],
        ) -> GroundedAnswerClaimInput:
    """从同次 candidate Evidence 的公开 SourceRecord 恢复唯一 claim 原文。"""
    source_map = {item.source: item for item in source_records}
    sources = {candidate.source, *(item.source for item in candidate.evidence)}
    try:
        texts = {source_map[source].raw_text for source in sources}
    except KeyError as exc:
        raise ConversationHeldOutV4RuntimeError(
            "v4 ANSWER candidate 缺少公开 SourceRecord") from exc
    if len(texts) != 1:
        raise ConversationHeldOutV4RuntimeError(
            "v4 ANSWER candidate 多来源原文不一致")
    return GroundedAnswerClaimInput(next(iter(texts)))


def _answer_generation_owner(
        item: ConversationHeldOutV4RuntimeInput,
        planning: GenerationPlanningRequest,
        selection: AnswerContentSelection,
        model: GroundedAnswerSurfaceModel,
        content: AnswerContentProtocol,
        selector: AnswerContentSelector,
        slot: int,
        ) -> _GenerationOwner:
    """按实际 ANSWER selection 装配 learned connector，不采用预写 surface。"""
    if type(slot) is not int or slot <= 0:
        raise ConversationHeldOutV4RuntimeError("v4 ANSWER generation slot 非法")
    if len(selection.selected_candidate_keys) != 1:
        raise ConversationHeldOutV4RuntimeError(
            "v4 ANSWER selection 必须精确选择一个 candidate")
    scope = _generation_scope(item, planning)
    candidates = {candidate.stable_key(): candidate for candidate in planning.candidates}
    candidate = candidates[selection.selected_candidate_keys[0]]
    family = (*_NAMESPACE, 90, slot)
    surface = build_facility_generation_surface_protocol(
        _NAMESPACE[0] + slot,
        owner=scope.owner,
        versions=scope.versions,
    )
    target = GroundedAnswerConnectorTarget(
        candidate.proposition, item.request.target_branch, family)
    compilation = compile_grounded_answer_connectors(
        model, _claim_input(candidate, item.source_records), target, surface)
    variant = min(compilation.variants, key=lambda value: value.option.pattern_id)
    structure = build_facility_structure_order_owner(
        _NAMESPACE[0] + 100 + slot,
        owner=scope.owner,
        versions=scope.versions,
    )
    aliases = _AnswerAliasFactory(item.request.target_branch)
    renderer_identity = minimal_instruction_identity(
        (*_NAMESPACE, 91, slot), owner=scope.owner, versions=scope.versions)
    components = GroundedAnswerRunLocalComponents(
        selector,
        build_facility_generation_plan_protocol(
            _NAMESPACE[0] + 200 + slot,
            owner=scope.owner,
            versions=scope.versions,
        ),
        GenerationStructureLayerProtocol(*tuple(
            _scoped_instruction(scope, (*_NAMESPACE, 92, slot, index))
            for index in range(1, 4)
        )),
        aliases,
        UnicodeRepresentationRenderer(family, renderer_identity),
        renderer_identity,
        build_facility_generation_postcheck_protocol(
            owner=scope.owner,
            versions=scope.versions,
        ),
        GroundedAnswerStructureVerifier(
            _scoped_instruction(scope, (*_NAMESPACE, 93, slot, 1)),
            _scoped_instruction(scope, (*_NAMESPACE, 93, slot, 2)),
        ),
        GroundedAnswerEvidenceSourceVerifier(
            _scoped_instruction(scope, (*_NAMESPACE, 94, slot, 1)),
            _scoped_instruction(scope, (*_NAMESPACE, 94, slot, 2)),
        ),
        QuestionAnswerProtocol(*tuple(
            _scoped_instruction(scope, (*_NAMESPACE, 95, slot, index))
            for index in range(1, 4)
        )),
        EvidenceQuestionPostcheckMapper(
            (*_NAMESPACE, 96, slot),
            citation_required=True,
            trust_required=True,
        ),
    )
    installation = GroundedAnswerRunLocalFactory(
        surface, structure.lifecycle, components).build(
            GroundedAnswerRunLocalBuild(
                model,
                _claim_input(candidate, item.source_records),
                target,
                planning,
                candidate,
                variant.option.structure_id,
                variant.option.pattern_id,
                GroundedAnswerParserProtocol(
                    *tuple(_scoped_instruction(
                        scope, (*_NAMESPACE, 97, slot, index))
                        for index in range(1, 6)),
                    content.answer,
                ),
                item.request.query_kind,
                V4_RUNTIME_ROUTE,
                V4_RUNTIME_EXECUTION_REASON,
                (*_NAMESPACE, 98, slot),
            ))
    return _GenerationOwner(installation.executor, structure, aliases)


def _response_act_generation_owner(
        item: ConversationHeldOutV4RuntimeInput,
        planning: GenerationPlanningRequest,
        selection: AnswerContentSelection,
        model: GroundedAnswerSurfaceModel,
        content: AnswerContentProtocol,
        selector: AnswerContentSelector,
        slot: int,
        ) -> _GenerationOwner:
    """按实际 UNKNOWN/CLARIFY/CONFLICT selection 装配 learned response act。"""
    if type(slot) is not int or slot <= 0:
        raise ConversationHeldOutV4RuntimeError("v4 response-act generation slot 非法")
    response_acts = {
        content.unknown: "UNKNOWN",
        content.clarify: "CLARIFY",
        content.conflict: "CONFLICT",
    }
    response_act = response_acts.get(selection.stance)
    if response_act is None:
        raise ConversationHeldOutV4RuntimeError(
            "v4 runtime G-01 返回未注册 response act")
    scope = _generation_scope(item, planning)
    family = (*_NAMESPACE, 100, slot)
    target = GroundedResponseActCompileTarget(
        response_act, selection.stance, item.request.target_branch, family)
    compilation = compile_grounded_response_act_patterns(model, target)
    variant = min(compilation.variants, key=lambda value: value.pattern_id)
    structure = build_facility_structure_order_owner(
        _NAMESPACE[0] + 300 + slot,
        owner=scope.owner,
        versions=scope.versions,
    )
    aliases = _ResponseActAliasFactory(item.request.target_branch)
    renderer_identity = minimal_instruction_identity(
        (*_NAMESPACE, 101, slot), owner=scope.owner, versions=scope.versions)
    components = GroundedResponseActRunLocalComponents(
        selector,
        build_facility_generation_plan_protocol(
            _NAMESPACE[0] + 400 + slot,
            owner=scope.owner,
            versions=scope.versions,
        ),
        GenerationStructureLayerProtocol(*tuple(
            _scoped_instruction(scope, (*_NAMESPACE, 102, slot, index))
            for index in range(1, 4)
        )),
        build_facility_generation_surface_protocol(
            _NAMESPACE[0] + 500 + slot,
            owner=scope.owner,
            versions=scope.versions,
        ),
        aliases,
        UnicodeRepresentationRenderer(family, renderer_identity),
        renderer_identity,
        build_facility_generation_postcheck_protocol(
            owner=scope.owner,
            versions=scope.versions,
        ),
        GroundedResponseActStructureVerifier(
            _scoped_instruction(scope, (*_NAMESPACE, 103, slot, 1)),
            _scoped_instruction(scope, (*_NAMESPACE, 103, slot, 2)),
        ),
        build_supporting_generation_verifier(_NAMESPACE[0] + 600 + slot),
        GroundedResponseActTaskVerifier(
            _scoped_instruction(scope, (*_NAMESPACE, 104, slot, 1)),
            _scoped_instruction(scope, (*_NAMESPACE, 104, slot, 2)),
        ),
        QuestionAnswerProtocol(*tuple(
            _scoped_instruction(scope, (*_NAMESPACE, 105, slot, index))
            for index in range(1, 4)
        )),
    )
    installation = GroundedResponseActRunLocalFactory(
        structure.lifecycle, components).build(
            GroundedResponseActRunLocalBuild(
                model,
                GroundedResponseActQuestionInput(response_act),
                target,
                planning,
                variant.pattern_id,
                GroundedResponseActParserProtocol(*tuple(
                    _scoped_instruction(
                        scope, (*_NAMESPACE, 106, slot, index))
                    for index in range(1, 4)
                )),
                item.request.query_kind,
                V4_RUNTIME_ROUTE,
                V4_RUNTIME_EXECUTION_REASON,
                (*_NAMESPACE, 107, slot),
            ))
    return _GenerationOwner(installation.executor, structure, aliases)


def _generation_owner(
        item: ConversationHeldOutV4RuntimeInput,
        planning: GenerationPlanningRequest,
        selection: AnswerContentSelection,
        model: GroundedAnswerSurfaceModel,
        content: AnswerContentProtocol,
        selector: AnswerContentSelector,
        slot: int,
        ) -> _GenerationOwner:
    """根据已经发生的真实 G-01 selection 选择实际 response runtime。"""
    if selection.stance == content.answer:
        return _answer_generation_owner(
            item, planning, selection, model, content, selector, slot)
    return _response_act_generation_owner(
        item, planning, selection, model, content, selector, slot)


def _generation_slot(
        item: ConversationHeldOutV4RuntimeInput,
        candidate_ordinal: int,
        ) -> int:
    """为回合 response 或逐候选 realization 建立无碰撞的正整数设施槽位。"""
    if type(candidate_ordinal) is not int or candidate_ordinal < 0:
        raise ConversationHeldOutV4RuntimeError("v4 runtime candidate ordinal 非法")
    total = item.ordinal + candidate_ordinal
    return (total * (total + 1)) // 2 + candidate_ordinal + 1


def _candidate_planning(
        item: ConversationHeldOutV4RuntimeInput,
        candidate: GenerationCandidate,
        ) -> GenerationPlanningRequest:
    """从同次 execution candidate 投影单 candidate 阅读计划，不改写回合 response。"""
    if not isinstance(candidate, GenerationCandidate):
        raise TypeError("v4 candidate planning candidate 类型错误")
    return GenerationPlanningRequest(
        AnswerGenerationGoal(
            item.request.goal_kind,
            candidate.proposition,
            item.request.required,
            candidate.source,
            candidate.scope,
            item.request.target_branch,
        ),
        (candidate,),
    )


def _run_candidate_realization(
        item: ConversationHeldOutV4RuntimeInput,
        candidate: GenerationCandidate,
        candidate_ordinal: int,
        model: GroundedAnswerSurfaceModel,
        content: AnswerContentProtocol,
        selector: AnswerContentSelector,
        ) -> ConversationHeldOutV4CandidateRealization:
    """为一个完整 execution candidate 实际运行局部 G-01/G-00 至 G-03 阅读链。"""
    planning = _candidate_planning(item, candidate)
    selection = selector.select(planning)
    owner = _generation_owner(
        item,
        planning,
        selection,
        model,
        content,
        selector,
        _generation_slot(item, candidate_ordinal),
    )
    try:
        generation = owner.executor.execute(planning)
    finally:
        owner.close()
    return ConversationHeldOutV4CandidateRealization(
        candidate, planning, selection, generation)


def _run_input(
        item: ConversationHeldOutV4RuntimeInput,
        model: GroundedAnswerSurfaceModel,
        selection_protocol: ConversationHeldOutV4RuntimeSelectionProtocol,
        selector: AnswerContentSelector,
        ) -> ConversationHeldOutV4RuntimeFrame:
    """执行一个 turn：真实 H-00 查询、G-01 选择和 G-00 至 G-03 renderer。"""
    content = selection_protocol.content
    ledger = _populate_ledger(item)
    executor = _CountingFactQuestionExecutor(
        ledger,
        route=V4_RUNTIME_ROUTE,
        hypothesis_kind=_HYPOTHESIS_KIND,
        executed_reason=V4_RUNTIME_EXECUTION_REASON,
    )
    query = QuestionQuery(
        item.request,
        V4_RUNTIME_ROUTE,
        (*_NAMESPACE, 110, item.ordinal),
    )
    execution = executor.execute(query)
    planning = execution.planning_request()
    if planning.goal.scope != item.request.response_scope:
        raise ConversationHeldOutV4RuntimeError(
            "v4 runtime H-00 planning scope 与 request response scope 不一致")
    selection = selector.select(planning)
    owner = _generation_owner(
        item,
        planning,
        selection,
        model,
        content,
        selector,
        _generation_slot(item, 0),
    )
    try:
        generation = owner.executor.execute(planning)
    finally:
        owner.close()
    realizations = tuple(
        _run_candidate_realization(
            item,
            candidate,
            candidate_ordinal,
            model,
            content,
            selector,
        )
        for candidate_ordinal, candidate in enumerate(
            execution.candidates, start=1)
    )
    return ConversationHeldOutV4RuntimeFrame(
        item,
        query,
        execution,
        selection,
        selection_protocol,
        generation,
        realizations,
        executor.execute_calls,
    )


def run_v4_candidate_runtime(
        capsule: ConversationHeldOutV4RuntimeSourceCapsule,
        *,
        static_assets: ConversationHeldOutV4RuntimeStaticAssets | None = None,
        ) -> ConversationHeldOutV4CandidateRuntimeResult:
    """运行显式 capsule；surface compiler 只消费已核验的 static asset payload。"""
    if not isinstance(capsule, ConversationHeldOutV4RuntimeSourceCapsule):
        raise TypeError("v4 candidate runtime 必须显式接收 RuntimeSourceCapsule")
    if static_assets is None:
        static_assets = read_v4_runtime_static_assets(
            V4_RUNTIME_DEFAULT_STATIC_ASSET_READ_BUDGET,
            test_transport=True,
        )
    if not isinstance(static_assets, ConversationHeldOutV4RuntimeStaticAssets):
        raise TypeError("v4 candidate runtime static_assets 类型错误")
    runtime_inputs = capsule.inputs
    inventory = static_assets.inventory
    model, _report = learn_grounded_answer_surface_model(
        compile_grounded_answer_training_records_from_payload(
            static_assets.surface_sample_payload,
            source_relative_path=V4_RUNTIME_SURFACE_SAMPLE_RELATIVE_PATH,
        ))
    selectors: dict[ScopeIdentity, tuple[
        ConversationHeldOutV4RuntimeSelectionProtocol,
        AnswerContentSelector,
    ]] = {}
    frames = []
    for item in runtime_inputs:
        scope = item.request.response_scope
        selected = selectors.get(scope)
        if selected is None:
            content, policy, selector = _selector(scope)
            binding = ConversationHeldOutV4RuntimeSelectionProtocol(
                scope, content, policy)
            selected = binding, selector
            selectors[scope] = selected
        binding, selector = selected
        frame = _run_input(item, model, binding, selector)
        if frame.selection_protocol != binding:
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime frame 未绑定当前 scope selector")
        frames.append(frame)
    frames = tuple(frames)
    identity = _runtime_identity(
        capsule,
        inventory,
        model,
        frames,
    )
    by_candidate_key = {
        candidate.stable_key(): frame
        for frame in frames
        for candidate in frame.execution.candidates
    }
    if len(by_candidate_key) != sum(
            len(frame.execution.candidates) for frame in frames):
        raise ConversationHeldOutV4RuntimeError(
            "v4 runtime execution candidate identity 不得跨 turn 重复")

    def render_candidate(candidate) -> RenderedSurface:
        """只返回实际 frame 中该 candidate 自己的 G-03 realization surface。"""
        frame = by_candidate_key.get(candidate.stable_key())
        if frame is None:
            raise ConversationHeldOutV4RuntimeError(
                "v4 runtime bundle 收到 execution 外 candidate")
        return frame.render_candidate(candidate)

    dependencies = identity.dependencies()
    execution_inputs = tuple(ConversationHeldOutV4ExecutionInput(
        item.case_key,
        item.turn_key,
        item.ordinal,
        item.request,
        frame.execution,
        item.representations,
        frame.surface_representations,
        item.source_records,
        dependencies,
    ) for item, frame in zip(runtime_inputs, frames, strict=True))
    bundle = build_v4_source_bundle_from_executions(
        version=1,
        family_key=V4_RUNTIME_FAMILY_KEY,
        inputs=execution_inputs,
        render_candidate=render_candidate,
    )
    freeze = freeze_v4_bundle(bundle)
    receipt = ConversationHeldOutV4RuntimeReceipt(
        V4_RUNTIME_FAMILY_KEY, identity, frames, bundle)
    return ConversationHeldOutV4CandidateRuntimeResult(
        capsule, identity, frames, bundle, freeze, receipt)


def run_v4_synthetic_candidate_runtime(
        ) -> ConversationHeldOutV4CandidateRuntimeResult:
    """运行显式 synthetic fixture，仅供公开 runtime 接线回归，绝不产生来源资格。"""
    return run_v4_candidate_runtime(build_v4_synthetic_runtime_fixture())


__all__ = [
    "ConversationHeldOutV4CandidateRuntimeResult",
    "ConversationHeldOutV4CandidateRealization",
    "ConversationHeldOutV4RuntimeEvidencePlan",
    "ConversationHeldOutV4RuntimeError",
    "ConversationHeldOutV4RuntimeFrame",
    "ConversationHeldOutV4RuntimeIdentity",
    "ConversationHeldOutV4RuntimeCodeFile",
    "ConversationHeldOutV4RuntimeInventory",
    "ConversationHeldOutV4RuntimeInput",
    "ConversationHeldOutV4RuntimeSelectionProtocol",
    "ConversationHeldOutV4RuntimeSourceCapsule",
    "ConversationHeldOutV4RuntimeReceipt",
    "ConversationHeldOutV4RuntimeStaticAssetReadBudget",
    "ConversationHeldOutV4RuntimeStaticAssetReadCounts",
    "ConversationHeldOutV4RuntimeStaticAssets",
    "V4_RUNTIME_FAMILY_KEY",
    "V4_RUNTIME_CODE_CLOSURE_SCHEMA",
    "V4_RUNTIME_CODE_RELATIVE_PATH",
    "V4_RUNTIME_DEFAULT_STATIC_ASSET_READ_BUDGET",
    "V4_RUNTIME_EXECUTION_CODE_RELATIVE_PATHS",
    "V4_RUNTIME_SOURCE_ORIGIN_EXTERNAL",
    "V4_RUNTIME_SOURCE_ORIGIN_SYNTHETIC",
    "V4_RUNTIME_STATIC_ASSET_READ_BUDGET_SCHEMA",
    "V4_RUNTIME_SURFACE_SAMPLE_RELATIVE_PATH",
    "build_v4_synthetic_runtime_fixture",
    "read_v4_runtime_static_assets",
    "run_v4_candidate_runtime",
    "run_v4_synthetic_candidate_runtime",
    "read_v4_runtime_inventory",
]
