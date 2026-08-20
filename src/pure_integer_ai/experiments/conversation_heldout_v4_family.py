"""DLG-05 v4 新 successor 的 typed family 构造点。

本模块只从公开 S-00/S-03/F-00/G-00 类型正向构造新的六 case family。
每个 turn 在同一构造点生成 ``QuestionRequest``、``QuestionExecutionResult``、
完整候选集合、输入/输出 Representation 和 SourceRecord；不读取旧 catalog、
旧 observation、private label 或论文。它是 source-bundle 的生产前置，不是
owner/formal runner，也不写入仓库内的大体量材料。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from pure_integer_ai.cognition.shared.generation_plan import GenerationCandidate
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
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
from pure_integer_ai.cognition.shared.logic_executor import (
    LogicEvidenceState,
)
from pure_integer_ai.cognition.shared.question_answer import (
    QuestionExecutionResult,
    QuestionQuery,
    QuestionRequest,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    RenderedSurface,
    UnicodeRepresentationRenderer,
)
from pure_integer_ai.cognition.shared.scope_identity import (
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
    PropositionSubstituter,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
    SubstitutionProtocol,
    BoundProposition,
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
from pure_integer_ai.experiments.conversation_heldout_v4_projection import (
    publish_v4_projection,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey


_NAMESPACE = (20260821, 405, 4)
V4_FAMILY_KEY = ProtocolKey((*_NAMESPACE, 1))
V4_INPUT_FAMILY = (*_NAMESPACE, 2)
V4_OUTPUT_FAMILY = (*_NAMESPACE, 3)
V4_RENDERER = minimal_instruction_identity((*_NAMESPACE, 4, 1))
V4_ROUTE = minimal_instruction_identity((*_NAMESPACE, 5, 1))
V4_EXECUTED_REASON = minimal_instruction_identity((*_NAMESPACE, 5, 2))


def _sha256_bytes(value: bytes) -> tuple[int, ...]:
    return tuple(hashlib.sha256(value).digest())


def _digest_hex(value: tuple[int, ...]) -> str:
    return bytes(value).hex()


def _dependency_binding() -> ConversationHeldOutV4DependencyBinding:
    """绑定本 successor 的公开 artifact/inventory/document 依赖身份。"""
    return ConversationHeldOutV4DependencyBinding(
        _sha256_bytes(b"pure-integer-ai:dlg05:v4:typed-family-artifact:1\n"),
        _sha256_bytes(b"pure-integer-ai:dlg05:v4:typed-family-inventory:1\n"),
        _sha256_bytes(b"pure-integer-ai:dlg05:v4:typed-family-document:1\n"),
    )


def _source(case: int) -> SourceRef:
    return SourceRef(
        _NAMESPACE[0], 1_000 + case, case,
        GLOBAL_OWNER_SCOPE, VersionBundle())


def _request(
        case: int,
        turn: int,
        *,
        candidate_count: int,
        event_target: bool = False,
        relation_target: bool = False,
        ) -> QuestionRequest:
    """从全新 source 正向编译一个完整 typed QuestionRequest。"""
    if type(candidate_count) is not int or candidate_count <= 0:
        raise ValueError("v4 candidate_count 必须为正整数")
    if event_target and relation_target:
        raise ValueError("v4 target 不得同时声明 event/relation")
    source = _source(case)
    evidence_scope = document_scope(source)
    response_scope = query_scope(
        20_000 + case * 100 + turn,
        parent=episode_scope(30_000 + case * 100 + turn, parent=evidence_scope),
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


def _evidence_id(case: int, turn: int, candidate: int, stance: int) -> int:
    """为同次执行中的完整 Evidence 生成稳定、正整数事件号。"""
    payload = (*_NAMESPACE, case, turn, candidate, stance)
    digest = integer_tuple_fingerprint(payload, domain="dlg05.v4.evidence.v1")
    value = int.from_bytes(bytes(digest[:8]), "big")
    return value or 1


def _execution(
        case: int,
        turn: int,
        request: QuestionRequest,
        *,
        conflict: bool,
        empty: bool,
        ) -> QuestionExecutionResult:
    """在同一 query execution 内产生完整候选集合和 Evidence。"""
    query = QuestionQuery(
        request, V4_ROUTE, (*_NAMESPACE, 30, case, turn))
    if empty:
        candidates: tuple[GenerationCandidate, ...] = ()
    else:
        targets = request.authorized_candidate_targets or (request.target,)
        built = []
        for ordinal, target in enumerate(targets, start=1):
            state = LogicEvidenceState(True, conflict)
            hypothesis = HypothesisKey(
                (*_NAMESPACE, 31),
                target.template.stable_key(),
                (*_NAMESPACE, 32, case, turn),
                request.evidence_scope,
                request.source,
            )
            stances = (EVIDENCE_SUPPORT, EVIDENCE_REFUTE) if conflict else (
                EVIDENCE_SUPPORT,)
            evidence = tuple(
                EvidenceRecord(
                    _evidence_id(case, turn, ordinal, stance),
                    hypothesis,
                    stance,
                    (*_NAMESPACE, 33, stance),
                    request.source,
                    case * 100 + turn,
                    (case, turn, ordinal, stance),
                )
                for stance in stances
            )
            built.append(GenerationCandidate(
                target, state, request.source, request.response_scope, evidence))
        candidates = tuple(built)
    return QuestionExecutionResult(
        query,
        V4_EXECUTED_REASON,
        candidates,
        (*_NAMESPACE, 34, case, turn, len(candidates)),
    )


def _representation(
        family: tuple[int, ...],
        text: str,
        ordinal: int,
        ) -> ConversationHeldOutV4Representation:
    scalars = unicode_scalars(text, allow_empty=False)
    identity = representation_identity(family, scalars)
    return ConversationHeldOutV4Representation(identity, ordinal, scalars)


def _input_representations(case: int, turn: int) -> tuple[
        ConversationHeldOutV4Representation, ...]:
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
        _representation(V4_INPUT_FAMILY, text, ordinal)
        for ordinal, text in enumerate(texts[(case, turn)])
    )


def _surface_text(case: int, turn: int, ordinal: int, conflict: bool) -> str:
    if conflict:
        return f"证据冲突：第{case}案例第{turn}回合。"
    if ordinal == 1:
        return f"档案显示，第{case}案例第{turn}回合成立。"
    return f"档案显示，第{case}案例第{turn}回合候选{ordinal}成立。"


def _source_record(case: int) -> ConversationHeldOutV4SourceRecord:
    source = _source(case)
    raw = f"公开 v4 来源记录：第{case}案例的 typed 事实与结构。"
    return ConversationHeldOutV4SourceRecord(
        source,
        unicode_scalars(raw),
        _sha256_bytes(raw.encode("utf-8")),
        unicode_scalars("CC-BY-4.0"),
        unicode_scalars("pure_integer_ai v4 public successor"),
        unicode_scalars(f"urn:pure-integer-ai:dlg05:v4:generated-source:{case}"),
    )


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4Family:
    """新 v4 family 的完整 execution、bundle 和 freeze 只读汇合。"""

    executions: tuple[QuestionExecutionResult, ...]
    bundle: ConversationHeldOutV4SourceBundle
    freeze: ConversationHeldOutV4Freeze
    dependencies: ConversationHeldOutV4DependencyBinding


def build_v4_family() -> ConversationHeldOutV4Family:
    """构造六 case/十二 turn successor，并从同次 execution 导出 bundle。"""
    dependencies = _dependency_binding()
    executions = []
    inputs = []
    surfaces_by_candidate: dict[tuple[int, ...], tuple[ObjectIdentity, ...]] = {}
    input_records: list[tuple[int, int, QuestionRequest, tuple[
        ConversationHeldOutV4Representation, ...], bool, bool]] = []
    for case in range(1, 7):
        candidate_count = 2 if case == 2 else 1
        for turn in (1, 2):
            request = _request(
                case,
                turn,
                candidate_count=candidate_count,
                event_target=case == 3,
                relation_target=case == 4,
            )
            conflict = case == 4
            empty = case == 5
            execution = _execution(
                case, turn, request, conflict=conflict, empty=empty)
            executions.append(execution)
            input_records.append((
                case, turn, request, _input_representations(case, turn),
                conflict, empty))
            for ordinal, candidate in enumerate(execution.candidates, start=1):
                surface = _representation(
                    V4_OUTPUT_FAMILY,
                    _surface_text(case, turn, ordinal, conflict),
                    ordinal - 1,
                )
                surfaces_by_candidate[candidate.stable_key()] = (surface.representation,)
    renderer = UnicodeRepresentationRenderer(V4_OUTPUT_FAMILY, V4_RENDERER)

    def render_candidate(candidate: GenerationCandidate) -> RenderedSurface:
        representations = surfaces_by_candidate.get(candidate.stable_key())
        if representations is None:
            raise ValueError("v4 candidate 不在同次 family surface map")
        return renderer.render(representations)

    for index, (case, turn, request, representations, conflict, empty) in enumerate(
            input_records, start=1):
        execution = executions[index - 1]
        surface_representations = tuple(
            ConversationHeldOutV4Representation(
                representation,
                ordinal,
                tuple(renderer.render((representation,)).units),
            )
            for ordinal, representation in enumerate(
                dict.fromkeys(
                    item
                    for candidate in execution.candidates
                    for item in surfaces_by_candidate[candidate.stable_key()]
                )
            )
        )
        del conflict, empty
        inputs.append(ConversationHeldOutV4ExecutionInput(
            ProtocolKey((*_NAMESPACE, 40, case)),
            ProtocolKey((*_NAMESPACE, 40, case, turn)),
            turn,
            request,
            execution,
            representations,
            surface_representations,
            (_source_record(case),),
            dependencies,
        ))
    bundle = build_v4_source_bundle_from_executions(
        version=1,
        family_key=V4_FAMILY_KEY,
        inputs=tuple(inputs),
        render_candidate=render_candidate,
    )
    return ConversationHeldOutV4Family(
        tuple(executions), bundle, freeze_v4_bundle(bundle), dependencies)


def write_v4_family_artifacts(
        family: ConversationHeldOutV4Family,
        root: str | Path,
        ) -> dict[str, Path]:
    """把可重建的 bundle/freeze/projection 审计材料写入指定 K 盘目录。"""
    if not isinstance(family, ConversationHeldOutV4Family):
        raise TypeError("family 类型错误")
    target = Path(root).resolve()
    target.mkdir(parents=True, exist_ok=True)
    payload_path = target / "bundle.canonical.ints"
    payload = (" ".join(str(value) for value in family.bundle.canonical_payload) + "\n").encode()
    if payload_path.exists() and payload_path.read_bytes() != payload:
        raise ValueError("bundle canonical payload 已存在且内容不同")
    if not payload_path.exists():
        payload_path.write_bytes(payload)
    freeze_path = target / "freeze.json"
    freeze_doc = {
        "schema": "dlg05-v4-freeze-v1",
        "payload_size": family.freeze.bundle_payload_size,
        "payload_sha256": _digest_hex(family.freeze.bundle_payload_sha256),
        "bundle_index": family.freeze.bundle_index,
        "turn_count": family.freeze.turn_count,
        "source_count": family.freeze.source_count,
        "dependencies": {
            "artifact_sha256": _digest_hex(family.dependencies.artifact_sha256),
            "inventory_sha256": _digest_hex(family.dependencies.inventory_sha256),
            "document_sha256": _digest_hex(family.dependencies.document_sha256),
        },
    }
    freeze_payload = (json.dumps(freeze_doc, ensure_ascii=False,
                                  sort_keys=True, separators=(",", ":")) + "\n").encode()
    if freeze_path.exists() and freeze_path.read_bytes() != freeze_payload:
        raise ValueError("freeze 已存在且内容不同")
    if not freeze_path.exists():
        freeze_path.write_bytes(freeze_payload)
    paths = publish_v4_projection(family.bundle, target / "projection")
    manifest_path = target / "artifact_manifest.json"
    manifest_doc = {
        "schema": "dlg05-v4-family-artifact-manifest-v1",
        "bundle_sha256": _digest_hex(family.bundle.payload_sha256),
        "bundle_payload_size": family.bundle.payload_size,
        "bundle_index": family.bundle.index,
        "turn_count": len(family.bundle.turns),
        "source_count": len(family.bundle.sources),
        "candidate_count": sum(len(turn.candidates) for turn in family.bundle.turns),
        "projection": [path.name for path in paths],
    }
    manifest_payload = (json.dumps(manifest_doc, ensure_ascii=False,
                                    sort_keys=True, separators=(",", ":")) + "\n").encode()
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_payload:
        raise ValueError("artifact manifest 已存在且内容不同")
    if not manifest_path.exists():
        manifest_path.write_bytes(manifest_payload)
    return {
        "payload": payload_path,
        "freeze": freeze_path,
        "manifest": manifest_path,
        "markdown": paths[0],
        "html": paths[1],
    }


__all__ = [
    "ConversationHeldOutV4Family",
    "V4_FAMILY_KEY",
    "build_v4_family",
    "write_v4_family_artifacts",
]
