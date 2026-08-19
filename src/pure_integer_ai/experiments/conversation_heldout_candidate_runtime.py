"""DLG-05 六案例公开候选的独立、无标签生产运行时。

本模块拥有候选的 Memory、G-00 至 G-04、SQLite 恢复和 rollback 生命周期。
入口不导入测试模块、不接收 evaluator label，也不构造 expected answer。JSON 仅
用于公开 observation/audit artifact；权威运行状态仍位于 SQLite 和纯整数对象。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentProtocol,
    AnswerContentSelector,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationStructureLayerProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    SourceRef,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_resolver import (
    MemoryAggregateFilter,
    RESOLUTION_ORIGIN_MEMORY,
)
from pure_integer_ai.cognition.shared.question_answer import (
    EvidenceAnswerPolicy,
    EvidenceAnswerPolicyProtocol,
    QuestionQuery,
    QuestionRequest,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    UnicodeRepresentationRenderer,
)
from pure_integer_ai.experiments.conversation_heldout_answer_runtime import (
    claim_input_from_candidate,
)
from pure_integer_ai.experiments.conversation_heldout_family import (
    ConversationHeldOutInputCatalog,
)
from pure_integer_ai.experiments.conversation_heldout_preflight import (
    ConversationHeldOutPreflightQuestionExecutor,
    audit_dlg05_preflight_axis_inputs,
    build_dlg05_preflight_evidence_plans,
    build_dlg05_preflight_language_compiler,
    build_dlg05_typed_preflight_catalog,
    build_dlg05_typed_preflight_manifest,
    build_dlg05_unseen_relation_compiler,
    build_dlg05_unseen_source_compiler,
)
from pure_integer_ai.experiments.conversation_heldout_preflight_runtime import (
    ConversationHeldOutCatalogTurnFactory,
    MappedConversationHeldOutResponseActResolver,
)
from pure_integer_ai.experiments.conversation_heldout_protocol import (
    AXIS_MEMORY_MISS,
    RESPONSE_ANSWER,
    RESPONSE_CLARIFY,
    RESPONSE_CONFLICT,
    RESPONSE_UNKNOWN,
    ConversationHeldOutManifest,
)
from pure_integer_ai.experiments.conversation_heldout_qualification import (
    ConversationHeldOutQualificationReceipt,
    ConversationHeldOutRollbackRecoveryReceipt,
    conversation_heldout_rollback_fault_key,
    qualify_dlg05_preflight,
)
from pure_integer_ai.experiments.conversation_heldout_runtime import (
    ConversationHeldOutMemoryPlan,
    ConversationHeldOutSelectionReceipt,
    run_real_selection_first,
    run_real_selection_first_receipt,
)
from pure_integer_ai.experiments.conversation_memory_demand_runtime import (
    ConversationMemoryQuestionExecutor,
    MemoryDemandRead,
)
from pure_integer_ai.experiments.evaluation_isolation import (
    clone_backend,
    clone_train_context,
)
from pure_integer_ai.experiments.facility_generation_scenario import (
    FacilityStructureOrderOwner,
    build_facility_alias_fixture,
    build_facility_generation_plan_protocol,
    build_facility_generation_postcheck_protocol,
    build_facility_generation_surface_protocol,
    build_facility_structure_order_owner,
    build_supporting_generation_verifier,
)
from pure_integer_ai.experiments.facility_readiness_scenarios import (
    FacilityConversationMemoryOwner,
    build_facility_conversation_memory_owner,
    facility_conversation_memory_owner_from_context,
)
from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.experiments.ph2_dataset_core import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_grounded_answer_compile import (
    compile_grounded_answer_training_records,
)
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerConnectorTarget,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    learn_grounded_answer_surface_model,
    surface_pattern_structure_id,
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
from pure_integer_ai.experiments.ph2_md03_center_adapter import (
    DirectionalCenterAdapterConfig,
    DirectionalCenterProfile,
    DirectionalMemoryCenterAdapter,
)
from pure_integer_ai.experiments.ph2_memory_dynamics_contract import (
    DIRECTIONS,
    EXPANSION_CHANNELS,
    MemoryExpansionChannelBudget,
    MemoryExpansionProfile,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    EvidenceQuestionPostcheckMapper,
    QuestionAnswerProtocol,
    QuestionAnswerRuntime,
)
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.source_record import (
    SourceRecordMetadata,
    SourceRecordRepository,
)


_BASE = 31_005
_SAMPLE = (
    Path(__file__).resolve().parents[3]
    / "data" / "ph2" / "grounded_answer_train_v1.jsonl.sample"
)
DLG05_PUBLIC_CANDIDATE_OBSERVATION_SCHEMA = (
    "dlg05-public-candidate-observation-v3"
)


# object-model: exception
class ConversationHeldOutCandidateError(RuntimeError):
    """DLG-05 独立候选的 owner、运行或 observation 合同未闭合。"""


def _snapshot_key(snapshot: Any) -> tuple[int, ...]:
    """把 canonical backend snapshot 压成固定纯整数资格键。"""
    return tuple(hashlib.sha256(canonical_json_bytes(snapshot)).digest())


def _record_key(*values: int) -> StableRecordKey:
    """建立 MD-01/MD-03 生产候选使用的正严格整数键。"""
    return StableRecordKey(tuple(values))


def _channel_budget(channel: str) -> MemoryExpansionChannelBudget:
    """为一个扩域 channel 建立冻结的小规模有界预算。"""
    return MemoryExpansionChannelBudget(
        channel, 1, 16, 8, 4, 4, 2, 2, 32, 4096)


def _understanding_profile() -> MemoryExpansionProfile:
    """把 UNDERSTANDING expansion key 绑定到列全 channel 的 profile。"""
    return MemoryExpansionProfile(
        _record_key(30_100, 3, 3),
        (_record_key(30_100, 3, 1),),
        EXPANSION_CHANNELS,
        tuple(_channel_budget(channel) for channel in EXPANSION_CHANNELS),
        (_record_key(30_100, 30), _record_key(30_100, 31)),
        ("GRAPH_DISTANCE", "LOGICAL_TIME", "SCOPE_DISTANCE"),
        ("OBLIGATION_MATCH", "SOURCE_INDEPENDENCE", "SUPPORT_REFUTE"),
        ("EVIDENCE_STATE", "GROUNDING_BOUNDARY", "SCOPE_AUTHORIZATION"),
        ("ACCESS", "HELD_OUT", "OWNER", "REFUTE", "VERSION"),
        1,
        1,
        0,
        0,
    )


def _center_adapter() -> DirectionalMemoryCenterAdapter:
    """按冻结顺序列全三个方向，并为本候选选择 UNDERSTANDING。"""
    profiles = tuple(
        DirectionalCenterProfile(
            direction,
            _record_key(30_100, ordinal, 1),
            _record_key(30_100, ordinal, 2),
            _record_key(30_100, ordinal, 3),
        )
        for ordinal, direction in enumerate(DIRECTIONS, start=1)
    )
    return DirectionalMemoryCenterAdapter(
        DirectionalCenterAdapterConfig(profiles))


def _answer_selector() -> tuple[AnswerContentProtocol, AnswerContentSelector]:
    """构造四个实际 response act 共用的纯 typed G-01 selector。"""
    content = AnswerContentProtocol(*tuple(
        minimal_instruction_identity((_BASE, 1, index))
        for index in range(1, 6)
    ))
    selector = AnswerContentSelector(
        content,
        EvidenceAnswerPolicy(
            content,
            EvidenceAnswerPolicyProtocol(*tuple(
                minimal_instruction_identity((_BASE, 2, index))
                for index in range(1, 5)
            )),
        ),
    )
    return content, selector


# object-model: resource-factory; state=mutable
class _AnswerAliasFactory:
    """按 selected ANSWER variant 建立并持有独占 R-01 owner。"""

    def __init__(self, branch: ObjectIdentity) -> None:
        self.branch = branch
        self.fixture = None

    def build(self, variant: Any) -> Any:
        """只从 variant alias requirements 建立所需 realization。"""
        if self.fixture is not None:
            raise ConversationHeldOutCandidateError(
                "ANSWER alias factory 不得重复 build")
        self.fixture = build_facility_alias_fixture(
            self.branch,
            tuple((item.filler, item.representation)
                  for item in variant.aliases),
        )
        return self.fixture.runtime

    def close(self) -> None:
        """关闭已经建立的 alias 后端。"""
        if self.fixture is not None:
            self.fixture.close()


# object-model: resource-factory; state=mutable
class _ResponseActAliasFactory:
    """按 selected response-act variant 建立独占 R-01 owner。"""

    def __init__(self, branch: ObjectIdentity) -> None:
        self.branch = branch
        self.fixture = None

    def build(self, variant: Any) -> Any:
        """只物化 stance 到 learned Representation 的唯一 realization。"""
        if self.fixture is not None:
            raise ConversationHeldOutCandidateError(
                "response-act alias factory 不得重复 build")
        self.fixture = build_facility_alias_fixture(
            self.branch,
            ((variant.template.stance, variant.representation),),
        )
        return self.fixture.runtime

    def close(self) -> None:
        """关闭已经建立的 alias 后端。"""
        if self.fixture is not None:
            self.fixture.close()


# object-model: resource-owner; state=mutable
@dataclass(slots=True)
class _QuestionRuntimeOwner:
    """持有一个实际 QuestionAnswerRuntime 的 S-07 与 alias 资源。"""

    runtime: QuestionAnswerRuntime
    structure: FacilityStructureOrderOwner
    aliases: _AnswerAliasFactory | _ResponseActAliasFactory

    def close(self) -> None:
        """关闭 runtime 的两个独占图 owner。"""
        self.aliases.close()
        self.structure.close()


# object-model: runtime-state; state=mutable
@dataclass(slots=True)
class _CandidateMemoryState:
    """保存可在 host、clone 与 resume 之间替换的 Memory owner。"""

    owner: FacilityConversationMemoryOwner
    source_records: SourceRecordRepository

    def replace(self, owner: FacilityConversationMemoryOwner) -> None:
        """切换到同一候选的 clone/resume owner 与其 SourceRecord 仓库。"""
        self.owner = owner
        self.source_records = SourceRecordRepository(owner.backend)


# object-model: provider; state=immutable
class _EmptyCoreBaseline:
    """为 MEMORY_MISS 轴关闭 Core fallback，不改变 Memory 存储。"""

    def candidates(self, request: Any) -> tuple[Any, ...]:
        """返回空 Core 候选集合。"""
        del request
        return ()

    def state_key(self) -> tuple[int, ...]:
        """返回固定 miss baseline 身份。"""
        return (_BASE, 90, 1)


# object-model: provider; state=immutable-after-init
class _NoMatchMemoryFilter:
    """把 MEMORY_MISS 查询限制到不存在的精确来源。"""

    def __init__(self, source: SourceRef) -> None:
        self.source = source

    def filters(self, request: Any) -> tuple[MemoryAggregateFilter, ...]:
        """返回唯一、来源精确且不命中的 aggregate filter。"""
        del request
        return (MemoryAggregateFilter(source=self.source),)

    def state_key(self) -> tuple[int, ...]:
        """返回包含完整虚构来源身份的过滤器键。"""
        return (_BASE, 90, 2, *self.source.stable_key())


def _ensure_memory_source(
        repository: SourceRecordRepository,
        trace: Any,
        ordinal: int,
        ) -> None:
    """为 resolver 返回的真实来源补齐可引用 SourceRecord。"""
    record = repository.find(trace.source.stable_key())
    if record is not None:
        if not record.metadata_complete:
            raise ConversationHeldOutCandidateError(
                "既有 Memory 来源缺少完整 SourceRecord metadata")
        return
    repository.put_complete(
        trace.source.stable_key(),
        f"公开候选来源{ordinal}",
        metadata=SourceRecordMetadata(
            "public-candidate-license",
            ordinal,
            700 + ordinal,
            800 + ordinal,
            900 + ordinal,
        ),
    )


# object-model: runtime-factory; state=mutable
class _CandidateQuestionRuntimeBuilder:
    """从同次 query/Memory read 装配 learned ANSWER 或 response-act runtime。"""

    def __init__(
            self,
            state: _CandidateMemoryState,
            executor: ConversationHeldOutPreflightQuestionExecutor,
            route: ObjectIdentity,
            content: AnswerContentProtocol,
            selector: AnswerContentSelector,
            ) -> None:
        self.state = state
        self.executor = executor
        self.route = route
        self.content = content
        self.selector = selector
        self.model, _ = learn_grounded_answer_surface_model(
            compile_grounded_answer_training_records(_SAMPLE))
        self.owners: list[_QuestionRuntimeOwner] = []

    @staticmethod
    def _ordinal(request: QuestionRequest) -> int:
        """从 catalog trace 读取稳定回合 ordinal。"""
        if len(request.trace) < 2 or type(request.trace[-2]) is not int:
            raise ConversationHeldOutCandidateError(
                "DLG-05 request trace 缺少回合 ordinal")
        return request.trace[-2]

    def _answer_runtime(
            self,
            request: QuestionRequest,
            planning: Any,
            candidate: Any,
            claim: Any,
            ) -> _QuestionRuntimeOwner:
        """装配当前候选的 learned ANSWER connector、citation 与 G-04。"""
        ordinal = self._ordinal(request)
        selected_pattern = next(
            pattern for pattern in self.model.patterns
            if any(part.literal == "档案显示，" for part in pattern.parts)
        )
        surface = build_facility_generation_surface_protocol(
            _BASE + 900 + ordinal)
        family = (_BASE, 910, ordinal)
        target = GroundedAnswerConnectorTarget(
            candidate.proposition, request.target_branch, family)
        structure = build_facility_structure_order_owner(
            _BASE + 920 + ordinal)
        aliases = _AnswerAliasFactory(request.target_branch)
        renderer_identity = minimal_instruction_identity(
            (_BASE, 911, ordinal))
        components = GroundedAnswerRunLocalComponents(
            self.selector,
            build_facility_generation_plan_protocol(
                _BASE + 930 + ordinal),
            GenerationStructureLayerProtocol(*tuple(
                minimal_instruction_identity((_BASE, 940 + ordinal, index))
                for index in range(1, 4)
            )),
            aliases,
            UnicodeRepresentationRenderer(family, renderer_identity),
            renderer_identity,
            build_facility_generation_postcheck_protocol(),
            GroundedAnswerStructureVerifier(
                minimal_instruction_identity((_BASE, 950 + ordinal, 1)),
                minimal_instruction_identity((_BASE, 950 + ordinal, 2)),
            ),
            GroundedAnswerEvidenceSourceVerifier(
                minimal_instruction_identity((_BASE, 960 + ordinal, 1)),
                minimal_instruction_identity((_BASE, 960 + ordinal, 2)),
            ),
            QuestionAnswerProtocol(*tuple(
                minimal_instruction_identity((_BASE, 970 + ordinal, index))
                for index in range(1, 4)
            )),
            EvidenceQuestionPostcheckMapper(
                (_BASE, 975 + ordinal, 1),
                citation_required=True,
                trust_required=True,
            ),
        )
        installation = GroundedAnswerRunLocalFactory(
            surface, structure.lifecycle, components).build(
                GroundedAnswerRunLocalBuild(
                    self.model,
                    claim,
                    target,
                    planning,
                    candidate,
                    surface_pattern_structure_id(selected_pattern),
                    selected_pattern.pattern_id,
                    GroundedAnswerParserProtocol(
                        *tuple(minimal_instruction_identity(
                            (_BASE, 980 + ordinal, index))
                               for index in range(1, 6)),
                        self.content.answer,
                    ),
                    request.query_kind,
                    minimal_instruction_identity((_BASE, 990 + ordinal, 1)),
                    minimal_instruction_identity((_BASE, 990 + ordinal, 2)),
                    (_BASE, 990 + ordinal, 3),
                ))
        return _QuestionRuntimeOwner(installation.runtime, structure, aliases)

    def _response_act_runtime(
            self,
            request: QuestionRequest,
            planning: Any,
            response_act: str,
            stance: ObjectIdentity,
            ) -> _QuestionRuntimeOwner:
        """装配当前 G-01 stance 对应的 learned non-answer runtime。"""
        ordinal = self._ordinal(request)
        family = (_BASE, 720, ordinal)
        target = GroundedResponseActCompileTarget(
            response_act, stance, request.target_branch, family)
        selected = compile_grounded_response_act_patterns(
            self.model, target).variants[0]
        structure = build_facility_structure_order_owner(
            _BASE + 730 + ordinal)
        aliases = _ResponseActAliasFactory(request.target_branch)
        renderer_identity = minimal_instruction_identity(
            (_BASE, 740, ordinal))
        components = GroundedResponseActRunLocalComponents(
            self.selector,
            build_facility_generation_plan_protocol(_BASE + 750 + ordinal),
            GenerationStructureLayerProtocol(*tuple(
                minimal_instruction_identity((_BASE, 760 + ordinal, index))
                for index in range(1, 4)
            )),
            build_facility_generation_surface_protocol(_BASE + 770 + ordinal),
            aliases,
            UnicodeRepresentationRenderer(
                target.representation_family, renderer_identity),
            renderer_identity,
            build_facility_generation_postcheck_protocol(),
            GroundedResponseActStructureVerifier(
                minimal_instruction_identity((_BASE, 780 + ordinal, 1)),
                minimal_instruction_identity((_BASE, 780 + ordinal, 2)),
            ),
            build_supporting_generation_verifier(790 + ordinal),
            GroundedResponseActTaskVerifier(
                minimal_instruction_identity((_BASE, 800 + ordinal, 1)),
                minimal_instruction_identity((_BASE, 800 + ordinal, 2)),
            ),
            QuestionAnswerProtocol(*tuple(
                minimal_instruction_identity((_BASE, 810 + ordinal, index))
                for index in range(1, 4)
            )),
        )
        installation = GroundedResponseActRunLocalFactory(
            structure.lifecycle, components).build(
                GroundedResponseActRunLocalBuild(
                    self.model,
                    GroundedResponseActQuestionInput(response_act),
                    target,
                    planning,
                    selected.pattern_id,
                    GroundedResponseActParserProtocol(*tuple(
                        minimal_instruction_identity(
                            (_BASE, 820 + ordinal, index))
                        for index in range(1, 4)
                    )),
                    request.query_kind,
                    minimal_instruction_identity((_BASE, 830 + ordinal, 1)),
                    minimal_instruction_identity((_BASE, 830 + ordinal, 2)),
                    (_BASE, 830 + ordinal, 3),
                ))
        return _QuestionRuntimeOwner(installation.runtime, structure, aliases)

    def build(
            self,
            request: QuestionRequest,
            context_read: Any,
            memory_read: MemoryDemandRead | None,
            ) -> QuestionAnswerRuntime:
        """从同次 typed query/read 重新计算 stance 并建立完整 runtime。"""
        del context_read
        ordinal = self._ordinal(request)
        query = QuestionQuery(
            request, self.route, (_BASE, 1_201, ordinal))
        if memory_read is None:
            result = self.executor.execute(query)
        else:
            repository = self.state.source_records
            for candidate_set in memory_read.resolution.sets:
                for resolved in candidate_set.candidates:
                    if resolved.origin_kind != RESOLUTION_ORIGIN_MEMORY:
                        continue
                    for trace in resolved.memory_source_traces:
                        _ensure_memory_source(
                            repository,
                            trace,
                            700 + repository.source_count(),
                        )
            eligible = tuple(
                resolved.stable_key()
                for candidate_set in memory_read.resolution.sets
                for resolved in candidate_set.candidates
                if resolved.origin_kind == RESOLUTION_ORIGIN_MEMORY
            )
            result = ConversationMemoryQuestionExecutor(
                memory_read,
                request.target,
                authorized_candidate_keys=eligible,
                executed_reason=minimal_instruction_identity(
                    (_BASE, 1_202, ordinal)),
                binding_reason=minimal_instruction_identity(
                    (_BASE, 1_203, ordinal)),
                trace_prefix=(_BASE, 1_204, ordinal),
                source_records=repository,
            ).execute(query)
        planning = result.planning_request()
        selection = self.selector.select(planning)
        if selection.stance == self.content.answer:
            selected = next(
                item for item in planning.candidates
                if item.stable_key() == selection.selected_candidate_keys[0]
            )
            owner = self._answer_runtime(
                request,
                planning,
                selected,
                claim_input_from_candidate(selected, self.state.source_records),
            )
        else:
            response_act = {
                self.content.unknown: "UNKNOWN",
                self.content.clarify: "CLARIFY",
                self.content.conflict: "CONFLICT",
            }.get(selection.stance)
            if response_act is None:
                raise ConversationHeldOutCandidateError(
                    "G-01 返回未注册 response act")
            owner = self._response_act_runtime(
                request, planning, response_act, selection.stance)
        self.owners.append(owner)
        return owner.runtime

    def close(self) -> None:
        """逆序关闭本候选运行期间建立的全部问答 runtime owner。"""
        while self.owners:
            self.owners.pop().close()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class Dlg05PublicCandidateResult:
    """独立生产 runner 返回的 catalog、manifest 与无标签资格证据。"""

    catalog: ConversationHeldOutInputCatalog
    manifest: ConversationHeldOutManifest
    qualification: ConversationHeldOutQualificationReceipt

    def __post_init__(self) -> None:
        """核验结果三者身份闭合且已经通过公开 qualification。"""
        qualify_dlg05_preflight(
            self.catalog, self.manifest, self.qualification)

    @property
    def execution(self) -> ConversationHeldOutSelectionReceipt:
        """返回首次 selection-first execution receipt。"""
        return self.qualification.execution


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class Dlg05PublicCandidateSelection:
    """一次且仅一次 selection-first 候选执行的无标签结果。"""

    catalog: ConversationHeldOutInputCatalog
    manifest: ConversationHeldOutManifest
    execution: ConversationHeldOutSelectionReceipt

    def __post_init__(self) -> None:
        """核验 selection receipt 绑定当前 manifest 并完整覆盖六案例。"""
        if self.execution.manifest_key != self.manifest.stable_key():
            raise ConversationHeldOutCandidateError(
                "candidate selection manifest key 漂移")
        self.catalog.assert_manifest_rebuildable(self.manifest)
        _validate_selection_execution(self.manifest, self.execution)


# object-model: resource-owner; state=mutable
@dataclass(slots=True)
class _CandidateAssembly:
    """持有一次 DLG-05 候选装配的全部生产 owner 与编译器。"""

    catalog: ConversationHeldOutInputCatalog
    manifest: ConversationHeldOutManifest
    language_compiler: Any
    relation_compiler: Any
    source_compiler: Any
    initial_backend: SQLiteBackend
    state: _CandidateMemoryState
    builder: _CandidateQuestionRuntimeBuilder
    factory: ConversationHeldOutCatalogTurnFactory

    def close(self) -> None:
        """关闭候选期间建立的问答 owner、Memory 生命周期和 SQLite。"""
        self.builder.close()
        self.state.owner.close()


def _seed_catalog_sources(
        catalog: ConversationHeldOutInputCatalog,
        repository: SourceRecordRepository,
        ) -> None:
    """为每个 typed held-out source 建立可回查、无 label 的公开原文。"""
    for case in catalog.cases:
        case_number = case.case_key.components[-1]
        for turn in case.turns:
            repository.put_complete(
                turn.request.source.stable_key(),
                f"公开候选输入来源{case_number}",
                metadata=SourceRecordMetadata(
                    "public-candidate-license",
                    case_number,
                    400 + case_number,
                    500 + case_number,
                    600 + case_number,
                ),
            )


def _validate_selection_execution(
        manifest: ConversationHeldOutManifest,
        execution: ConversationHeldOutSelectionReceipt,
        ) -> None:
    """核验单次 selection-first 完整覆盖六案例和全部声明轴。"""
    observations = execution.observations
    if (len(observations) != 6
            or any(item.context_revision != 2 for item in observations)
            or any(not item.turn_response_acts for item in observations)
            or not any(RESPONSE_ANSWER in item.turn_response_acts
                       for item in observations)
            or sum(bool(item.memory_receipt_keys) for item in observations) != 5
            or {axis for item in observations for axis in item.proven_axis_keys}
            != set(manifest.required_axes)):
        raise ConversationHeldOutCandidateError(
            "DLG-05 单次六案例 execution 不完整")


def _build_candidate_assembly(database: Path) -> _CandidateAssembly:
    """在新 SQLite 上装配可单次执行或继续资格重放的生产候选。"""
    catalog = build_dlg05_typed_preflight_catalog()
    language_compiler = build_dlg05_preflight_language_compiler(catalog)
    relation_compiler = build_dlg05_unseen_relation_compiler(catalog)
    source_compiler = build_dlg05_unseen_source_compiler(catalog)
    manifest = build_dlg05_typed_preflight_manifest()
    plans = build_dlg05_preflight_evidence_plans(catalog)
    route = minimal_instruction_identity((_BASE, 1_200, 1))
    reason = minimal_instruction_identity((_BASE, 1_200, 2))
    executor = ConversationHeldOutPreflightQuestionExecutor(
        route, reason, plans)
    content, selector = _answer_selector()
    initial_backend = SQLiteBackend(str(database))
    owner = build_facility_conversation_memory_owner(initial_backend)
    state = _CandidateMemoryState(
        owner, SourceRecordRepository(initial_backend))
    builder = _CandidateQuestionRuntimeBuilder(
        state, executor, route, content, selector)
    profile = _understanding_profile()
    adapter = _center_adapter()
    _seed_catalog_sources(catalog, state.source_records)

    def memory_plan_factory(case: Any, turn: Any, context_read: Any) -> Any:
        """为当前 request 打开精确 source/scope 的真实 DLG-04 read。"""
        del context_read
        request = catalog.turn_for(case.case_key, turn.turn_key).request
        active_owner = state.owner
        current = active_owner.begin_current(
            request.source, request.response_scope)
        center = adapter.from_understanding(
            current, current.occurrences[0], strength="CONDITIONAL")
        resolver_owner = active_owner.ctx.memory_resolver_runtime.resolver
        original_baseline = resolver_owner.baseline_provider
        original_filter = resolver_owner.index_filter_provider
        is_miss = AXIS_MEMORY_MISS in case.axis_keys
        if is_miss:
            resolver_owner.baseline_provider = _EmptyCoreBaseline()
            resolver_owner.index_filter_provider = _NoMatchMemoryFilter(SourceRef(
                request.source.source_kind,
                request.source.source_id + 90_000,
                request.source.document_id + 90_000,
                request.source.owner,
                request.source.versions,
            ))

        def release() -> None:
            """恢复本回合 resolver profile 并关闭 query 生命周期。"""
            if is_miss:
                resolver_owner.baseline_provider = original_baseline
                resolver_owner.index_filter_provider = original_filter
            active_owner.close_active()

        return ConversationHeldOutMemoryPlan(
            active_owner.consumer,
            current,
            center,
            profile,
            MemoryAccessContext(1, 2, 3),
            release=release,
        )

    resolver = MappedConversationHeldOutResponseActResolver((
        (content.unknown, RESPONSE_UNKNOWN),
        (content.clarify, RESPONSE_CLARIFY),
        (content.conflict, RESPONSE_CONFLICT),
        (content.answer, RESPONSE_ANSWER),
    ))
    factory = ConversationHeldOutCatalogTurnFactory(
        catalog,
        builder,
        memory_plan_factory=memory_plan_factory,
        response_act_resolver=resolver,
        question_input_compiler=language_compiler,
        relation_input_compiler=relation_compiler,
        source_input_compiler=source_compiler,
    )
    return _CandidateAssembly(
        catalog,
        manifest,
        language_compiler,
        relation_compiler,
        source_compiler,
        initial_backend,
        state,
        builder,
        factory,
    )


def run_dlg05_public_candidate(
        database_path: str | Path,
        ) -> Dlg05PublicCandidateSelection:
    """在新 SQLite 上执行一次六案例 selection-first，不做资格重放。"""
    database = Path(database_path).resolve()
    if database.exists():
        raise ConversationHeldOutCandidateError(
            "DLG-05 candidate database 必须是未存在的新路径")
    database.parent.mkdir(parents=True, exist_ok=True)
    assembly = _build_candidate_assembly(database)
    try:
        execution = run_real_selection_first_receipt(
            assembly.manifest, assembly.factory)
        return Dlg05PublicCandidateSelection(
            assembly.catalog, assembly.manifest, execution)
    finally:
        assembly.close()


def qualify_dlg05_public_candidate(
        database_path: str | Path,
        ) -> Dlg05PublicCandidateResult:
    """直接运行六案例生产候选并完成 fresh/clone/resume/rollback 资格闭环。"""
    database = Path(database_path).resolve()
    if database.exists():
        raise ConversationHeldOutCandidateError(
            "DLG-05 candidate database 必须是未存在的新路径")
    database.parent.mkdir(parents=True, exist_ok=True)
    assembly = _build_candidate_assembly(database)
    catalog = assembly.catalog
    manifest = assembly.manifest
    language_compiler = assembly.language_compiler
    relation_compiler = assembly.relation_compiler
    source_compiler = assembly.source_compiler
    initial_backend = assembly.initial_backend
    current_owner = assembly.state.owner
    state = assembly.state
    builder = assembly.builder
    factory = assembly.factory
    cloned_owner = None
    closed_initial = False
    try:
        execution = run_real_selection_first_receipt(manifest, factory)
        _validate_selection_execution(manifest, execution)
        observations = execution.observations
        frozen_backend = state.owner.backend.snapshot()
        fresh_execution = run_real_selection_first_receipt(manifest, factory)
        if (fresh_execution.observations != observations
                or state.owner.backend.snapshot() != frozen_backend):
            raise ConversationHeldOutCandidateError(
                "DLG-05 fresh replay 或存储发生漂移")

        host_owner = state.owner
        cloned_backend = clone_backend(host_owner.backend)
        cloned_ctx = clone_train_context(
            host_owner.ctx,
            cloned_backend,
            label="dlg05-public-candidate",
        )
        cloned_owner = facility_conversation_memory_owner_from_context(
            cloned_ctx, cloned_backend)
        state.replace(cloned_owner)
        cloned_before = cloned_backend.snapshot()
        clone_execution = run_real_selection_first_receipt(manifest, factory)
        if (clone_execution.observations != observations
                or cloned_backend.snapshot() != cloned_before):
            raise ConversationHeldOutCandidateError(
                "DLG-05 clone replay 或存储发生漂移")
        state.replace(host_owner)
        cloned_owner.close()
        cloned_owner = None

        host_owner.backend.commit()
        persisted_backend = host_owner.backend.snapshot()
        if persisted_backend != frozen_backend:
            raise ConversationHeldOutCandidateError(
                "DLG-05 SQLite commit 前后漂移")
        host_owner.close()
        closed_initial = True
        reopened_backend = SQLiteBackend(str(database))
        current_owner = build_facility_conversation_memory_owner(
            reopened_backend, seed=False)
        state.replace(current_owner)
        if current_owner.backend.snapshot() != persisted_backend:
            raise ConversationHeldOutCandidateError(
                "DLG-05 SQLite resume 初始 snapshot 漂移")
        resumed_before = current_owner.backend.snapshot()
        resumed_execution = run_real_selection_first_receipt(manifest, factory)
        if (resumed_execution.observations != observations
                or current_owner.backend.snapshot() != resumed_before):
            raise ConversationHeldOutCandidateError(
                "DLG-05 resume replay 或存储发生漂移")

        rollback_case = manifest.cases[-1]

        def faulting_factory(case: Any, turn: Any, context_read: Any) -> Any:
            """只在 rollback case 第二回合注入一次真实持久化半写。"""
            plan = factory(case, turn, context_read)
            if case.case_key != rollback_case.case_key or turn.ordinal != 2:
                return plan

            def fail_after_read(memory_read: MemoryDemandRead) -> Any:
                """在真实 read 后写入污染并抛错，要求 runner 原子恢复。"""
                if not isinstance(memory_read, MemoryDemandRead):
                    raise TypeError("rollback fault 缺少真实 Memory read")
                state.owner.backend.insert("source_record", {
                    "source_hash": 9_876_543_210,
                    "text_hash": 1,
                    "codepoint_count": 5,
                    "source_kind": 9,
                    "source_id": 9,
                    "document_id": 9,
                    "corpus_version": 0,
                    "parser_version": 0,
                    "license_id": "",
                    "batch_id": 0,
                    "companion_type_hash": 0,
                    "companion_name_hash": 0,
                    "companion_assoc_id": 0,
                    "raw_text": "fault",
                })
                raise RuntimeError("DLG05_INJECTED_ROLLBACK_FAULT")

            return replace(plan, runtime_factory=fail_after_read)

        fault_before = current_owner.backend.snapshot()
        caught = None
        try:
            run_real_selection_first(manifest, faulting_factory)
        except RuntimeError as error:
            if str(error) != "DLG05_INJECTED_ROLLBACK_FAULT":
                raise
            caught = error
        if caught is None:
            raise ConversationHeldOutCandidateError(
                "DLG-05 rollback fault 未被执行")
        if (current_owner.ctx.work_memory.active_query_scope is not None
                or current_owner.backend.snapshot() != fault_before):
            raise ConversationHeldOutCandidateError(
                "DLG-05 rollback 未恢复 query/storage")
        fault_after = current_owner.backend.snapshot()
        recovered_execution = run_real_selection_first_receipt(
            manifest, factory)
        if (recovered_execution.observations != observations
                or current_owner.backend.snapshot() != fault_before):
            raise ConversationHeldOutCandidateError(
                "DLG-05 rollback 后确定重试漂移")
        recovered_after = current_owner.backend.snapshot()

        axis_audit = audit_dlg05_preflight_axis_inputs(
            catalog,
            compiler=language_compiler,
            relation_compiler=relation_compiler,
            source_compiler=source_compiler,
        )
        qualification = ConversationHeldOutQualificationReceipt(
            manifest.stable_key(),
            catalog.stable_key(),
            execution,
            fresh_execution,
            clone_execution,
            resumed_execution,
            ConversationHeldOutRollbackRecoveryReceipt(
                conversation_heldout_rollback_fault_key(caught),
                _snapshot_key(fault_before),
                _snapshot_key(fault_after),
                _snapshot_key(recovered_after),
                recovered_execution,
            ),
            (
                _snapshot_key(frozen_backend),
                _snapshot_key(current_owner.backend.snapshot()),
                _snapshot_key(cloned_before),
                _snapshot_key(persisted_backend),
                _snapshot_key(resumed_before),
                _snapshot_key(fault_before),
            ),
            axis_audit,
        )
        return Dlg05PublicCandidateResult(catalog, manifest, qualification)
    finally:
        builder.close()
        if cloned_owner is not None:
            cloned_owner.close()
        if current_owner is not None:
            if current_owner is not state.owner:
                current_owner.close()
            elif not (closed_initial and current_owner.backend is initial_backend):
                current_owner.close()


def build_dlg05_candidate_observation_document(
        result: Dlg05PublicCandidateSelection,
        ) -> dict[str, Any]:
    """序列化无 label、无表面文本的独立候选 observation。"""
    if not isinstance(result, Dlg05PublicCandidateSelection):
        raise TypeError("DLG-05 candidate result 类型错误")
    document: dict[str, Any] = {
        "schema": DLG05_PUBLIC_CANDIDATE_OBSERVATION_SCHEMA,
        "authority": "public-selection-first-only",
        "labels_included": 0,
        "formal_run": 0,
        "selection_run_count": 1,
        "catalog_key": list(result.catalog.stable_key()),
        "manifest_key": list(result.manifest.stable_key()),
        "contract_key": list(result.execution.contract_key),
        "execution_key": list(result.execution.stable_key()),
        "observation_keys": [
            list(item.stable_key())
            for item in result.execution.observations
        ],
    }
    document["document_sha256"] = hashlib.sha256(
        canonical_json_bytes(document)).hexdigest()
    return document


def write_dlg05_candidate_observation(
        target: str | Path,
        repository_root: str | Path,
        result: Dlg05PublicCandidateSelection,
        ) -> Path:
    """不可覆盖写出公开 selection-first observation artifact。"""
    root = Path(repository_root).resolve()
    path = Path(target).resolve()
    expected_parent = (root / "data" / "ph2" / "manifests").resolve()
    if path.parent != expected_parent:
        raise ConversationHeldOutCandidateError(
            "candidate observation 必须位于 data/ph2/manifests")
    payload = canonical_json_bytes(
        build_dlg05_candidate_observation_document(result)) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise ConversationHeldOutCandidateError(
                "candidate observation 已存在且内容不同，不允许覆盖")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def verify_dlg05_candidate_observation(
        target: str | Path,
        ) -> dict[str, Any]:
    """只读验证公开 observation schema、边界与 document SHA。"""
    path = Path(target).resolve()
    payload = path.read_bytes()
    if not payload.endswith(b"\n"):
        raise ConversationHeldOutCandidateError(
            "candidate observation 必须以单个换行结束")
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    if not isinstance(value, dict):
        raise ConversationHeldOutCandidateError(
            "candidate observation 顶层必须是 object")
    expected_keys = {
        "schema", "authority", "labels_included", "formal_run",
        "selection_run_count", "catalog_key", "manifest_key", "contract_key",
        "execution_key", "observation_keys", "document_sha256",
    }
    if set(value) != expected_keys:
        raise ConversationHeldOutCandidateError(
            "candidate observation 字段不精确")
    if (value["schema"] != DLG05_PUBLIC_CANDIDATE_OBSERVATION_SCHEMA
            or value["authority"] != "public-selection-first-only"
            or value["labels_included"] != 0
            or value["formal_run"] != 0
            or value["selection_run_count"] != 1):
        raise ConversationHeldOutCandidateError(
            "candidate observation 公开边界或资格状态漂移")
    declared_sha = value.pop("document_sha256")
    actual_document_sha = hashlib.sha256(
        canonical_json_bytes(value)).hexdigest()
    if declared_sha != actual_document_sha:
        raise ConversationHeldOutCandidateError(
            "candidate observation document SHA 漂移")
    return {
        "path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "document_sha256": actual_document_sha,
        "observation_count": len(value["observation_keys"]),
        "verified": 1,
    }


__all__ = [
    "ConversationHeldOutCandidateError",
    "DLG05_PUBLIC_CANDIDATE_OBSERVATION_SCHEMA",
    "Dlg05PublicCandidateResult",
    "Dlg05PublicCandidateSelection",
    "build_dlg05_candidate_observation_document",
    "qualify_dlg05_public_candidate",
    "run_dlg05_public_candidate",
    "verify_dlg05_candidate_observation",
    "write_dlg05_candidate_observation",
]
