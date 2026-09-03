"""Typed 对话课程到 S-02 semantic lesson 的数据驱动适配器。

该模块只消费已登记的 canonical typed payload。普通 surface、HTML、Markdown、
代码和表格不会进入这里；payload 中没有的语义也不会由词面猜测。输出沿用
``LanguageSemanticCourseProtocol``，因此正式入口仍经过 occurrence/span、S-02、
H-00 和 G-00 请求链。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ENTITY,
    OBJECT_PROPOSITION,
    ObjectIdentity,
    TypedRef,
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.semantic_object import context_scope_identity
from pure_integer_ai.cognition.understanding.semantic_builder import (
    SemanticBuildPlan,
    SemanticObjectSpec,
    SemanticPropositionSpec,
)
from pure_integer_ai.experiments.language_semantic_course import (
    LanguageSemanticCourseDecision,
    LanguageSemanticCourseInput,
    LanguageSemanticCourseMapper,
    LanguageSemanticCourseProtocol,
    LanguageSemanticLesson,
    SemanticCourseEvidenceSpec,
    SemanticCourseTemplateScope,
)
from pure_integer_ai.experiments.language_semantic_query import (
    LanguageSemanticQueryDecision,
    LanguageSemanticQueryInput,
    LanguageSemanticQueryMapper,
    LanguageSemanticQueryProtocol,
)
from pure_integer_ai.cognition.shared.typed_binding import BindingEnvironment
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject


_NAMESPACE = (21401, 1)
_SUPPORTED_KINDS = frozenset({
    "GenerationAdoptionPostcheckQuery",
    "GenerationGeneralizationCandidateV1",
})
_EVIDENCE_HASHER = Hasher("typed.dialogue.semantic.evidence.v1")


def _key(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} 必须是非空整数列表")
    if any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{label} 必须使用非负严格整数")
    return tuple(value)


def _text_key(value: str, *, domain: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not value:
        raise ValueError("typed identity 文本键不能为空")
    result = integer_tuple_fingerprint(
        tuple(value.encode("utf-8")), domain=domain)
    return result or (1,)


def _legacy_text_key(value: str, *, domain: str) -> tuple[int, ...]:
    """恢复早期 fingerprint v1 的文本键，供已发布 checkpoint 只读兼容。

    早期运行在同一版本号下把每个整数写成 ``size + bytes``；现役实现
    使用小整数定长表以减少编码成本。两者都保持纯整数输入和 SHA-256
    输出，不能混用，也不能把这个兼容键写回新课程。
    """
    if not isinstance(value, str) or not value:
        raise ValueError("typed legacy identity 文本键不能为空")
    domain_bytes = domain.encode("utf-8")
    values = tuple(value.encode("utf-8"))
    digest = hashlib.sha256()
    digest.update(b"pure_integer_ai.integer_tuple_fingerprint.v1\x00")
    digest.update(len(domain_bytes).to_bytes(8, "big"))
    digest.update(domain_bytes)
    digest.update(len(values).to_bytes(8, "big"))
    for value in values:
        size = max(1, (value.bit_length() + 8) // 8)
        digest.update(size.to_bytes(8, "big"))
        digest.update(value.to_bytes(size, "big", signed=True))
    return (1, len(values), *digest.digest())


def _value(payload: CanonicalJsonObject) -> dict[str, Any]:
    raw = payload.to_value()
    if not isinstance(raw, dict):
        raise ValueError("typed semantic payload 必须是 object")
    return raw


def _candidate_rows(kind: str, raw: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    if kind == "GenerationAdoptionPostcheckQuery":
        rows = raw.get("candidate_propositions")
    else:
        rows = raw.get("choice_candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("typed semantic payload 缺少候选集合")
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("typed candidate 必须是 object")
        result.append(row)
    return tuple(result)


def _structure_family_value(kind: str, raw: dict[str, Any]) -> str:
    """返回不含来源、表面或答案的 typed 课程结构族值。"""
    # ADOPTION and POSTCHECK are deliberately separate contracts.  Adoption
    # selects a candidate; postcheck verifies a completed generation.  Sharing
    # their structure index makes a read-only held-out query recover multiple
    # stage candidates and would force an unauthorized choice.
    family_kind = kind
    task_kind = raw.get("task_kind")
    family = (raw.get("generation_case")
              if kind == "GenerationAdoptionPostcheckQuery"
              else raw.get("candidate_case"))
    if not isinstance(family, str) or not family:
        family = task_kind
    if not isinstance(family, str) or not family:
        family = "UNSPECIFIED"
    if not isinstance(family, str) or not family:
        raise ValueError("typed semantic payload 缺少结构族")
    return f"{family_kind}:{family}"


def _structure_family_key(kind: str, raw: dict[str, Any]) -> tuple[int, ...]:
    """返回现役 typed 课程结构族整数键。"""
    return _text_key(
        _structure_family_value(kind, raw),
        domain="typed.dialogue.semantic.structure.v1",
    )


def _candidate_key(kind: str, row: dict[str, Any], ordinal: int) -> tuple[int, ...]:
    if kind == "GenerationAdoptionPostcheckQuery":
        raw = row.get("candidate_key")
        if raw is not None:
            return _key(raw, label="candidate_key")
        candidate_id = row.get("candidate_id")
    else:
        candidate_id = row.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("typed candidate 缺少 candidate_id")
    return _text_key(
        f"{ordinal}:{candidate_id}",
        domain="typed.dialogue.semantic.candidate.v1",
    )


def _contains_packed(value: tuple[int, ...], part: tuple[int, ...]) -> bool:
    """查找长度封装的局部键，避免把它误认成完整 Proposition 身份。"""
    packed = (len(part), *part)
    width = len(packed)
    return any(value[index:index + width] == packed
               for index in range(len(value) - width + 1))


def _state(kind: str, row: dict[str, Any], raw: dict[str, Any]) -> LogicEvidenceState:
    if kind == "GenerationAdoptionPostcheckQuery":
        state = row.get("state")
        if not isinstance(state, dict):
            raise ValueError("adoption candidate 缺少 state")
        support = state.get("support") == 1
        refute = state.get("refute") == 1
        if not support and not refute:
            raise ValueError("adoption candidate 缺少 support/refute")
        return LogicEvidenceState(support, refute)
    # Generalization positive/negative partition is an explicit course label;
    # negative rows never reach the adapter as positive requests.
    if raw.get("sample_family") != "POSITIVE":
        raise ValueError("generalization semantic lesson 只接受 POSITIVE family")
    return LogicEvidenceState(True, False)


@dataclass(frozen=True, slots=True)
class TypedDialogueSemanticMapper:
    """把显式 typed payload 映射为来源化 semantic lesson。"""

    semantic_hypothesis_kind: tuple[int, ...]
    provenance_kind: int
    version: int = 1

    def __post_init__(self) -> None:
        if (not isinstance(self.semantic_hypothesis_kind, tuple)
                or not self.semantic_hypothesis_kind
                or any(type(item) is not int or item < 0
                       for item in self.semantic_hypothesis_kind)):
            raise ValueError("semantic_hypothesis_kind 非法")
        if type(self.provenance_kind) is not int or self.provenance_kind <= 0:
            raise ValueError("provenance_kind 必须为正整数")

    def map(self, input_value: LanguageSemanticCourseInput):
        if not isinstance(input_value, LanguageSemanticCourseInput):
            raise TypeError("typed semantic mapper 输入类型错误")
        reason = minimal_instruction_identity(
            (*_NAMESPACE, 1), owner=input_value.source.owner,
            versions=input_value.source.versions)
        if input_value.payload_kind is None:
            return LanguageSemanticCourseDecision(reason, (1, 0, 0))
        if input_value.payload_kind not in _SUPPORTED_KINDS:
            raise ValueError("typed semantic payload kind 未注册")
        if not isinstance(input_value.typed_payload, CanonicalJsonObject):
            raise TypeError("typed semantic payload 类型错误")
        raw = _value(input_value.typed_payload)
        # 正式 V-00 只读路径只能由 query mapper 从已学图恢复；正向 lesson
        # 在这里必须停止，避免把评测样本重新注入 S-02/H-00。
        if input_value.read_only:
            return LanguageSemanticCourseDecision(reason, (1, 0, 0))
        if (input_value.payload_kind == "GenerationGeneralizationCandidateV1"
                and raw.get("sample_family") != "POSITIVE"):
            # Negative/ambiguous partitions remain evaluation material.  They
            # must not create a positive semantic lesson or be silently
            # reinterpreted from their surface.
            return LanguageSemanticCourseDecision(reason, (1, 0, 0))
        if (input_value.payload_kind == "GenerationAdoptionPostcheckQuery"
                and raw.get("postcheck", {}).get("enabled") == 1
                and not all(
                    isinstance(row, dict)
                    and row.get("source_match") == 1
                    and not row.get("refuted_source_ids")
                    for row in raw.get("postcheck", {}).get("requirements", ()))):
            return LanguageSemanticCourseDecision(reason, (1, 0, 0))
        if (input_value.payload_kind == "GenerationAdoptionPostcheckQuery"
                and raw.get("task_kind") == "ADOPTION"):
            # Only an explicit support-only adoption is a positive lesson.
            # Refute-only, conflict, and empty states are authored negative
            # outcomes and must remain evaluation material.
            rows = _candidate_rows(input_value.payload_kind, raw)
            if any(
                    not isinstance(row.get("state"), dict)
                    or row["state"].get("support") != 1
                    or row["state"].get("refute") != 0
                    for row in rows):
                return LanguageSemanticCourseDecision(reason, (1, 0, 0))
        rows = _candidate_rows(input_value.payload_kind, raw)
        structure_family = _structure_family_key(input_value.payload_kind, raw)
        anchors = input_value.spans or input_value.occurrences
        if not anchors:
            raise ValueError("typed semantic lesson 缺少 occurrence/span anchor")
        root_anchor = anchors[0]
        if not isinstance(root_anchor, TypedRef):
            raise TypeError("semantic anchor 必须是 TypedRef")
        source = input_value.source
        owner = source.owner
        versions = source.versions
        upstream = HypothesisKey(
            self.semantic_hypothesis_kind,
            (*_NAMESPACE, 2, *root_anchor.stable_key()),
            (*_NAMESPACE, 3),
            input_value.occurrence_scope,
            source,
        )
        objects = []
        propositions = []
        local_refs = []
        states = []
        for ordinal, row in enumerate(rows, start=1):
            candidate_key = _candidate_key(
                input_value.payload_kind, row, ordinal)
            local_key = (*_NAMESPACE, 10, *candidate_key)
            predicate = concept_identity(
                (*_NAMESPACE, 20, ordinal), owner=owner, versions=versions)
            # 结构索引是可迁移的候选槽位，不包含来源、surface、答案或
            # candidate_id；来源和 query scope 在运行时重新绑定。
            structure = structure_concept_identity(
                (*_NAMESPACE, 30, 1, *structure_family),
                owner=owner, versions=versions)
            propositions.append(SemanticPropositionSpec(
                local_key,
                (*_NAMESPACE, 40, ordinal),
                predicate,
                structure,
                (),
                root_anchor,
            ))
            local_refs.append(propositions[-1].local_ref)
            states.append(_state(input_value.payload_kind, row, raw))
            objects.append(SemanticObjectSpec(
                OBJECT_ENTITY, (*_NAMESPACE, 50, ordinal)))
        plan = SemanticBuildPlan(
            upstream,
            (*_NAMESPACE, 60, *root_anchor.stable_key()),
            tuple(objects),
            tuple(propositions),
        )
        evidence = []
        scopes = []
        for ordinal, (local, state) in enumerate(zip(local_refs, states), start=1):
            stance = EVIDENCE_REFUTE if state.refute and not state.support else EVIDENCE_SUPPORT
            evidence.append(SemanticCourseEvidenceSpec(
                local,
                (_EVIDENCE_HASHER.h63((source.stable_key(), ordinal)) or
                 ordinal),
                stance,
                (*_NAMESPACE, 70, ordinal),
                source,
                ordinal,
            ))
            scopes.append(SemanticCourseTemplateScope(
                local,
                context_scope_identity(source, (*_NAMESPACE, 80, ordinal)),
            ))
        goal = local_refs[0]
        goal_kind = minimal_instruction_identity(
            (*_NAMESPACE, 90, 1), owner=owner, versions=versions)
        branch = language_branch_identity(
            (*_NAMESPACE, 91, 1), owner=owner, versions=versions)
        lesson = LanguageSemanticLesson(
            root_anchor,
            plan,
            tuple(evidence),
            tuple(scopes),
            BindingEnvironment(),
            goal,
            tuple(local_refs),
            goal_kind,
            LogicEvidenceState(True, False),
            branch,
        )
        trace = (1, len(rows), *root_anchor.stable_key())
        return LanguageSemanticCourseDecision(reason, trace, lesson)

    def clone_for_evaluation(self) -> "TypedDialogueSemanticMapper":
        return self

    def state_key(self) -> tuple:
        return (self.version, self.semantic_hypothesis_kind, self.provenance_kind)


@dataclass(frozen=True, slots=True)
class TypedDialogueSemanticQueryMapper:
    """从已学候选恢复 typed generation 请求，不重新写入课程 Evidence。

    只读评测没有机会重新调用正向 lesson mapper；这里仅依据当前输入携带的
    typed payload 形状和 recovered 候选数量做选择。多候选、负例、歧义或候选
    缺失均返回无请求，避免把索引顺序或表层文字当作答案。
    """

    semantic_hypothesis_kind: tuple[int, ...]
    version: int = 1

    def __post_init__(self) -> None:
        if (not isinstance(self.semantic_hypothesis_kind, tuple)
                or not self.semantic_hypothesis_kind
                or any(type(item) is not int or item < 0
                       for item in self.semantic_hypothesis_kind)):
            raise ValueError("semantic query hypothesis kind 非法")
        if type(self.version) is not int or self.version <= 0:
            raise ValueError("semantic query version 必须为正整数")

    def map(self, input_value: LanguageSemanticQueryInput):
        if not isinstance(input_value, LanguageSemanticQueryInput):
            raise TypeError("typed semantic query 输入类型错误")
        reason = minimal_instruction_identity(
            (*_NAMESPACE, 101), owner=input_value.current.source.owner,
            versions=input_value.current.source.versions)
        payload_kind = input_value.current.payload_kind
        if payload_kind not in _SUPPORTED_KINDS:
            return LanguageSemanticQueryDecision(reason, (1, 0, 0))
        payload = input_value.current.typed_payload
        if not isinstance(payload, CanonicalJsonObject):
            raise TypeError("typed semantic query payload 类型错误")
        raw = _value(payload)
        if (payload_kind == "GenerationGeneralizationCandidateV1"
                and raw.get("sample_family") != "POSITIVE"):
            return LanguageSemanticQueryDecision(reason, (1, 0, 0))
        if (payload_kind == "GenerationAdoptionPostcheckQuery"
                and raw.get("postcheck", {}).get("enabled") == 1
                and not all(
                    isinstance(row, dict)
                    and row.get("source_match") == 1
                    and not row.get("refuted_source_ids")
                    for row in raw.get("postcheck", {}).get("requirements", ()))):
            return LanguageSemanticQueryDecision(reason, (1, 0, 0))
        rows = _candidate_rows(payload_kind, raw)
        recoverable = tuple(
            item for item in input_value.candidates if item.recoverable
        )
        # Adoption payloads carry the complete authored proposition identity.
        # When a trained graph contains several candidates in the same
        # structure family, use that explicit identity to disambiguate; never
        # choose a candidate from storage order.  Other payloads retain the
        # original unique-candidate requirement.
        selected = ()
        if len(rows) == 1 and payload_kind == "GenerationAdoptionPostcheckQuery":
            row = rows[0]
            authored_key = row.get("candidate_key")
            if authored_key is not None:
                expected_key = _key(authored_key, label="candidate_key")
                # The authored key is embedded as the local declaration key
                # inside the runtime Proposition identity.  Match that
                # length-delimited segment rather than comparing it with the
                # full source-bound identity.
                local_key = (*_NAMESPACE, 10, *expected_key)
                selected = tuple(
                    item for item in recoverable
                    if _contains_packed(
                        item.hypothesis.candidate_key, local_key)
                )
            # The authored adoption key belongs to the pre-course fixture and
            # is not itself the runtime proposition identity.  Its bound
            # proposition still carries an exact source anchor, predicate and
            # structure identity, which are sufficient to select the matching
            # learned candidate across a shared structure family.
            if not selected:
                bound = row.get("bound_proposition")
                if isinstance(bound, dict):
                    expected_predicate = bound.get("predicate_key")
                    expected_structure = bound.get("structure_key")
                    expected_anchor = bound.get("source_anchor_key")
                    if all(isinstance(value, list) for value in (
                            expected_predicate, expected_structure,
                            expected_anchor)):
                        predicate_key = _key(
                            expected_predicate, label="predicate_key")
                        structure_key = _key(
                            expected_structure, label="structure_key")
                        anchor_key = _key(
                            expected_anchor, label="source_anchor_key")
                        selected = tuple(
                            item for item in recoverable
                            if item.materialized.atomic.definition.predicate.stable_key()
                            == predicate_key
                            and item.materialized.structure.stable_key()
                            == structure_key
                            and item.materialized.atomic.definition.source_anchor.stable_key()
                            == anchor_key
                        )
        if not selected and len(rows) == 1 and len(input_value.candidates) == 1:
            selected = recoverable
        if len(selected) != 1:
            return LanguageSemanticQueryDecision(
                reason, (1, len(rows), len(input_value.candidates),
                         len(recoverable), len(selected)))
        selected = selected[0]
        owner = input_value.current.source.owner
        versions = input_value.current.source.versions
        goal_kind = minimal_instruction_identity(
            (*_NAMESPACE, 90, 1), owner=owner, versions=versions)
        branch = language_branch_identity(
            (*_NAMESPACE, 91, 1), owner=owner, versions=versions)
        required = LogicEvidenceState.from_status(selected.snapshot.epistemic_status)
        return LanguageSemanticQueryDecision(
            reason,
            (1, len(rows), len(recoverable), *selected.hypothesis.stable_key()),
            goal=selected.hypothesis,
            candidates=(selected.hypothesis,),
            goal_kind=goal_kind,
            required=required,
            target_branch=branch,
        )

    def lookup_structures(self, current: LanguageSemanticCourseInput):
        """返回当前 typed payload 的现役及历史结构索引。

        结构索引是候选族的来源无关身份。旧 checkpoint 在 fingerprint
        编码修订前写入过现役族；本查询只携带现役族与 fingerprint v1
        兼容键，不把不同契约（ADOPTION vs POSTCHECK）的族混入彼此恢复。
        """
        if not isinstance(current, LanguageSemanticCourseInput):
            raise TypeError("typed semantic structure lookup 输入类型错误")
        if current.payload_kind not in _SUPPORTED_KINDS:
            return ()
        payload = current.typed_payload
        if not isinstance(payload, CanonicalJsonObject):
            raise TypeError("typed semantic structure lookup payload 类型错误")
        raw = _value(payload)
        if (current.payload_kind == "GenerationGeneralizationCandidateV1"
                and raw.get("sample_family") != "POSITIVE"):
            return ()
        _candidate_rows(current.payload_kind, raw)
        family_value = _structure_family_value(current.payload_kind, raw)
        # POSTCHECK was historically aliased onto the ADOPTION family so a
        # held-out postcheck could read a candidate taught as an earlier
        # adoption.  With several authored POSTCHECK courses sharing the same
        # structure family, that alias pulls distinct ADOPTION sources into
        # the POSTCHECK recovery and makes the authored candidate-key match
        # ambiguous (decision_goal=False).  Each POSTCHECK lesson is taught
        # under its own POSTCHECK family with a full authored candidate_key,
        # so the alias is not needed to recover a POSTCHECK goal; restore the
        # per-contract separation and read only the declared family.
        family_values = [family_value]
        owner = current.source.owner
        versions = current.source.versions
        identities = []
        for family in family_values:
            for family_key in (
                    _text_key(
                        family,
                        domain="typed.dialogue.semantic.structure.v1"),
                    _legacy_text_key(
                        family,
                        domain="typed.dialogue.semantic.structure.v1"),
            ):
                identities.append(structure_concept_identity(
                    (*_NAMESPACE, 30, 1, *family_key),
                    owner=owner, versions=versions,
                ))
        return tuple(sorted(set(identities), key=ObjectIdentity.stable_key))

    def clone_for_evaluation(self) -> "TypedDialogueSemanticQueryMapper":
        return self

    def state_key(self) -> tuple:
        return (self.version, self.semantic_hypothesis_kind)


def build_typed_dialogue_semantic_protocol(
        mapper: TypedDialogueSemanticMapper,
        *,
        builder_identity,
        atomic_predicates,
        trace_predicates,
        scope_predicates,
        substitution,
        provenance_kind: int,
        ) -> LanguageSemanticCourseProtocol:
    """用调用方注入的 S-02 predicate/builders 建立正式 semantic protocol。"""
    return LanguageSemanticCourseProtocol(
        builder_identity,
        tuple(atomic_predicates),
        tuple(trace_predicates),
        tuple(scope_predicates),
        substitution,
        mapper,
        provenance_kind,
    )


def build_typed_dialogue_semantic_query_protocol(
        mapper: TypedDialogueSemanticQueryMapper,
        ) -> LanguageSemanticQueryProtocol:
    """建立只读 typed query protocol；正向课程与评测 mapper 彼此隔离。"""
    if not isinstance(mapper, TypedDialogueSemanticQueryMapper):
        raise TypeError("typed semantic query mapper 类型错误")
    return LanguageSemanticQueryProtocol(mapper)


__all__ = [
    "TypedDialogueSemanticMapper",
    "TypedDialogueSemanticQueryMapper",
    "build_typed_dialogue_semantic_protocol",
    "build_typed_dialogue_semantic_query_protocol",
]
