"""DLG-05 语言表面到 typed QuestionRequest 的证据化整数输入边界。

本模块不保存原始字符串，也不把 alias key 当作语义。用户输入由有序
Representation 身份组成；每个词形必须经来源化 lexical route 投影为
LanguageAtom，随后由显式结构绑定构造 ``QuestionRequest``。缺路由、歧义、
分支漂移或未注册构式都 fail closed。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_ATOM,
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_REPRESENTATION,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition


# object-model: exception
class ConversationLanguageInputError(RuntimeError):
    """语言输入、词形证据或语义构式不能唯一闭合。"""


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验非空严格整数键。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise ConversationLanguageInputError(f"{label} 必须是非负严格整数 tuple")
    return value


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """把可变长整数键加入长度边界。"""
    result.extend((len(value), *value))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationLexicalEvidence:
    """一个词形到 LanguageAtom 路由的独立来源证据。"""

    source: SourceRef
    evidence_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验完整来源与证据身份。"""
        if not isinstance(self.source, SourceRef):
            raise TypeError("lexical evidence source 类型错误")
        _strict_key(self.evidence_key, label="lexical evidence key")

    def stable_key(self) -> tuple[int, ...]:
        """返回完整来源化证据键。"""
        result: list[int] = [1]
        _pack(result, self.source.stable_key())
        _pack(result, self.evidence_key)
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationLexicalRoute:
    """一个 Representation 经独立来源证据指向一个 LanguageAtom。"""

    branch: ObjectIdentity
    visible_form: ObjectIdentity
    semantic_atom: ObjectIdentity
    evidence: tuple[ConversationLexicalEvidence, ...]

    def __post_init__(self) -> None:
        """拒绝无来源、单源伪独立或跨分支的 lexical route。"""
        if (not isinstance(self.branch, ObjectIdentity)
                or self.branch.object_kind != OBJECT_LANGUAGE_BRANCH):
            raise ValueError("lexical route branch 必须是 LanguageBranch")
        if (not isinstance(self.visible_form, ObjectIdentity)
                or self.visible_form.object_kind != OBJECT_REPRESENTATION):
            raise ValueError("lexical route visible form 必须是 Representation")
        if (not isinstance(self.semantic_atom, ObjectIdentity)
                or self.semantic_atom.object_kind != OBJECT_LANGUAGE_ATOM):
            raise ValueError("lexical route semantic atom 必须是 LanguageAtom")
        if (not isinstance(self.evidence, tuple) or len(self.evidence) < 2
                or any(not isinstance(item, ConversationLexicalEvidence)
                       for item in self.evidence)):
            raise ConversationLanguageInputError(
                "lexical route 至少需要两个独立来源证据")
        ordered = tuple(sorted(self.evidence, key=lambda item: item.stable_key()))
        if ordered != self.evidence:
            raise ConversationLanguageInputError("lexical evidence 必须规范排序")
        if len({item.source.stable_key() for item in self.evidence}) < 2:
            raise ConversationLanguageInputError("lexical route 来源不独立")
        if len({item.stable_key() for item in self.evidence}) != len(self.evidence):
            raise ConversationLanguageInputError("lexical evidence 不得重复")

    def stable_key(self) -> tuple[int, ...]:
        """返回词形、语义原子和全部来源证据的整数键。"""
        result: list[int] = [1]
        for value in (
                self.branch.stable_key(),
                self.visible_form.stable_key(),
                self.semantic_atom.stable_key()):
            _pack(result, value)
        result.append(len(self.evidence))
        for item in self.evidence:
            _pack(result, item.stable_key())
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationTypedUtterance:
    """一个语言分支内的有序整数 Representation 输入。"""

    branch: ObjectIdentity
    visible_forms: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        """核验输入非空、有序且只含 Representation 身份。"""
        if (not isinstance(self.branch, ObjectIdentity)
                or self.branch.object_kind != OBJECT_LANGUAGE_BRANCH):
            raise ValueError("typed utterance branch 必须是 LanguageBranch")
        if (not isinstance(self.visible_forms, tuple) or not self.visible_forms
                or any(not isinstance(item, ObjectIdentity)
                       or item.object_kind != OBJECT_REPRESENTATION
                       for item in self.visible_forms)):
            raise ConversationLanguageInputError(
                "typed utterance 必须包含有序 Representation")

    def stable_key(self) -> tuple[int, ...]:
        """返回保留输入顺序的纯整数键。"""
        result: list[int] = [1]
        _pack(result, self.branch.stable_key())
        result.append(len(self.visible_forms))
        for item in self.visible_forms:
            _pack(result, item.stable_key())
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationNormalizedUtterance:
    """一次实际 lexical resolution 的语义原子序列与证据路径。"""

    utterance_key: tuple[int, ...]
    semantic_atoms: tuple[ObjectIdentity, ...]
    route_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """核验每个输入位置都有一个语义原子和一条来源化路由。"""
        _strict_key(self.utterance_key, label="normalized utterance key")
        if (not isinstance(self.semantic_atoms, tuple)
                or not self.semantic_atoms
                or any(not isinstance(item, ObjectIdentity)
                       or item.object_kind != OBJECT_LANGUAGE_ATOM
                       for item in self.semantic_atoms)):
            raise ConversationLanguageInputError(
                "normalized utterance semantic atoms 非法")
        if (not isinstance(self.route_keys, tuple)
                or len(self.route_keys) != len(self.semantic_atoms)):
            raise ConversationLanguageInputError(
                "normalized utterance route 数量不闭合")
        for key in self.route_keys:
            _strict_key(key, label="normalized lexical route key")

    def stable_key(self) -> tuple[int, ...]:
        """返回输入、语义序列和实际路由的稳定键。"""
        result: list[int] = [1]
        _pack(result, self.utterance_key)
        result.append(len(self.semantic_atoms))
        for atom, route in zip(self.semantic_atoms, self.route_keys):
            _pack(result, atom.stable_key())
            _pack(result, route)
        return tuple(result)


# object-model: runtime-owner; owns=immutable-route-index
class ConversationLexicalNormalizer:
    """按 branch+Representation 唯一解析证据化 lexical route。"""

    def __init__(self, routes: tuple[ConversationLexicalRoute, ...]) -> None:
        """建立只读路由表；同词形多语义选项保持歧义并在读取时拒绝。"""
        if (not isinstance(routes, tuple) or not routes
                or any(not isinstance(item, ConversationLexicalRoute)
                       for item in routes)):
            raise TypeError("lexical normalizer routes 非法")
        if len({item.stable_key() for item in routes}) != len(routes):
            raise ConversationLanguageInputError("lexical routes 不得重复")
        self.routes = tuple(sorted(routes, key=lambda item: item.stable_key()))
        grouped: dict[tuple[ObjectIdentity, ObjectIdentity], list] = {}
        for route in self.routes:
            grouped.setdefault(
                (route.branch, route.visible_form), []).append(route)
        self._routes = {key: tuple(value) for key, value in grouped.items()}

    def normalize(
            self,
            utterance: ConversationTypedUtterance,
            ) -> ConversationNormalizedUtterance:
        """逐位置解析 lexical route；缺失或多语义分叉时 fail closed。"""
        if not isinstance(utterance, ConversationTypedUtterance):
            raise TypeError("lexical normalizer utterance 类型错误")
        atoms = []
        route_keys = []
        for visible in utterance.visible_forms:
            routes = self._routes.get((utterance.branch, visible), ())
            semantic = {item.semantic_atom for item in routes}
            if not routes:
                raise ConversationLanguageInputError("typed utterance 缺少 lexical route")
            if len(semantic) != 1:
                raise ConversationLanguageInputError("typed utterance lexical route 歧义")
            # 多条独立 route 若同意同一 atom，保留全部 route 身份作为审计路径。
            route_key: list[int] = [len(routes)]
            for route in routes:
                _pack(route_key, route.stable_key())
            atoms.append(next(iter(semantic)))
            route_keys.append(tuple(route_key))
        return ConversationNormalizedUtterance(
            utterance.stable_key(), tuple(atoms), tuple(route_keys))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationQuestionRequestFrame:
    """一个 normalized semantic sequence 对应的 QuestionRequest 构式。"""

    construction: ObjectIdentity
    branch: ObjectIdentity
    semantic_atoms: tuple[ObjectIdentity, ...]
    query_kind: ObjectIdentity
    intent: ObjectIdentity
    goal_kind: ObjectIdentity
    target: BoundProposition
    required: LogicEvidenceState
    evidence_scope: ScopeIdentity
    response_scope: ScopeIdentity
    trace_prefix: tuple[int, ...]
    target_branch: ObjectIdentity | None = None
    authorized_candidate_targets: tuple[BoundProposition, ...] = ()

    def __post_init__(self) -> None:
        """以一次真实 QuestionRequest 构造核验全部 typed 字段。"""
        if (not isinstance(self.construction, ObjectIdentity)
                or self.construction.object_kind != OBJECT_STRUCTURE_CONCEPT):
            raise ValueError("question frame construction 必须是 StructureConcept")
        if self.branch.object_kind != OBJECT_LANGUAGE_BRANCH:
            raise ValueError("question frame branch 必须是 LanguageBranch")
        if (not isinstance(self.semantic_atoms, tuple) or not self.semantic_atoms
                or any(item.object_kind != OBJECT_LANGUAGE_ATOM
                       for item in self.semantic_atoms)):
            raise ConversationLanguageInputError("question frame semantic atoms 非法")
        _strict_key(self.trace_prefix, label="question frame trace prefix")
        self.request_for((1,))

    def request_for(self, occurrence_key: tuple[int, ...]) -> QuestionRequest:
        """用本次输入 occurrence 构造 QuestionRequest，不复用表面字符串。"""
        occurrence = _strict_key(
            occurrence_key, label="question input occurrence key")
        return QuestionRequest(
            self.query_kind,
            self.intent,
            self.goal_kind,
            self.target,
            self.required,
            self.evidence_scope,
            self.response_scope,
            (*self.trace_prefix, *occurrence),
            self.target_branch,
            self.authorized_candidate_targets,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回结构、语义序列与请求构式的纯整数键。"""
        result: list[int] = [1]
        for value in (
                self.construction.stable_key(),
                self.branch.stable_key(),
                self.query_kind.stable_key(),
                self.intent.stable_key(),
                self.goal_kind.stable_key(),
                self.target.stable_key(),
                self.required.stable_key(),
                self.evidence_scope.stable_key(),
                self.response_scope.stable_key(),
                self.trace_prefix):
            _pack(result, value)
        result.append(len(self.semantic_atoms))
        for item in self.semantic_atoms:
            _pack(result, item.stable_key())
        result.append(0 if self.target_branch is None else 1)
        if self.target_branch is not None:
            _pack(result, self.target_branch.stable_key())
        result.append(len(self.authorized_candidate_targets))
        for item in self.authorized_candidate_targets:
            _pack(result, item.stable_key())
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationQuestionLanguageInput:
    """一个实际用户 utterance、构式身份和 occurrence 序。"""

    utterance: ConversationTypedUtterance
    construction: ObjectIdentity
    occurrence_key: tuple[int, ...]
    provided_positions: tuple[int, ...] = ()
    context_target_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验输入只引用 typed utterance 和显式构式。"""
        if not isinstance(self.utterance, ConversationTypedUtterance):
            raise TypeError("question language input utterance 类型错误")
        if (not isinstance(self.construction, ObjectIdentity)
                or self.construction.object_kind != OBJECT_STRUCTURE_CONCEPT):
            raise ValueError("question language input construction 非法")
        _strict_key(self.occurrence_key, label="question language occurrence")
        if self.provided_positions:
            if (len(set(self.provided_positions)) != len(self.provided_positions)
                    or len(self.provided_positions)
                    != len(self.utterance.visible_forms)
                    or tuple(self.provided_positions)
                    != tuple(sorted(self.provided_positions))
                    or any(type(item) is not int or item < 0
                           for item in self.provided_positions)):
                raise ConversationLanguageInputError(
                    "question language provided positions 非法")
            if not self.context_target_key:
                raise ConversationLanguageInputError(
                    "省略输入缺少 context target anchor")
            _strict_key(
                self.context_target_key,
                label="question language context target")

    def stable_key(self) -> tuple[int, ...]:
        """返回用户输入、构式和 occurrence 的完整整数身份。"""
        result: list[int] = [1]
        for value in (
                self.utterance.stable_key(),
                self.construction.stable_key(),
                self.occurrence_key,
                self.provided_positions,
                self.context_target_key):
            _pack(result, value)
        return tuple(result)


# object-model: runtime-owner; owns=immutable-frame-index
class ConversationQuestionInputCompiler:
    """执行 utterance normalization 与结构绑定，构造 QuestionRequest。"""

    def __init__(
            self,
            normalizer: ConversationLexicalNormalizer,
            frames: tuple[ConversationQuestionRequestFrame, ...],
            ) -> None:
        """绑定只读 lexical normalizer 和唯一 semantic construction frames。"""
        if not isinstance(normalizer, ConversationLexicalNormalizer):
            raise TypeError("question input compiler normalizer 类型错误")
        if (not isinstance(frames, tuple) or not frames
                or any(not isinstance(item, ConversationQuestionRequestFrame)
                       for item in frames)):
            raise TypeError("question input compiler frames 非法")
        keys = tuple(self._frame_key(item) for item in frames)
        if len(set(keys)) != len(keys):
            raise ConversationLanguageInputError("question input frame 不得歧义")
        self.normalizer = normalizer
        self.frames = tuple(sorted(frames, key=lambda item: item.stable_key()))
        self._frames = {
            self._frame_key(item): item for item in self.frames
        }

    @staticmethod
    def _frame_key(frame: ConversationQuestionRequestFrame) -> tuple:
        """返回结构、分支和 normalized semantic sequence 的查找键。"""
        return frame.construction, frame.branch, frame.semantic_atoms

    def compile(
            self,
            value: ConversationQuestionLanguageInput,
            *,
            context_read=None,
            ) -> QuestionRequest:
        """从实际整数 utterance 唯一解析并构造 QuestionRequest。"""
        if not isinstance(value, ConversationQuestionLanguageInput):
            raise TypeError("question input compiler value 类型错误")
        if context_read is not None:
            from pure_integer_ai.experiments.conversation_context_runtime import (
                ConversationContextRead,
            )
            if not isinstance(context_read, ConversationContextRead):
                raise TypeError("question input compiler context read 类型错误")
        normalized = self.normalizer.normalize(value.utterance)
        if value.provided_positions:
            if context_read is None or not getattr(context_read, "turns", ()):
                raise ConversationLanguageInputError(
                    "省略输入需要非空 ConversationContextRead")
            if context_read.turns[-1].target_key != value.context_target_key:
                raise ConversationLanguageInputError(
                    "省略输入 context target anchor 漂移")
            candidates = []
            for frame in self.frames:
                if (frame.construction != value.construction
                        or frame.branch != value.utterance.branch
                        or len(frame.semantic_atoms)
                        <= len(normalized.semantic_atoms)):
                    continue
                if any(position >= len(frame.semantic_atoms)
                       for position in value.provided_positions):
                    continue
                if tuple(
                        frame.semantic_atoms[position]
                        for position in value.provided_positions
                ) == normalized.semantic_atoms:
                    candidates.append(frame)
            if len(candidates) != 1:
                raise ConversationLanguageInputError(
                    "省略输入不能唯一绑定 context 补全 frame")
            frame = candidates[0]
        else:
            key = (
                value.construction,
                value.utterance.branch,
                normalized.semantic_atoms,
            )
            frame = self._frames.get(key)
            if frame is None:
                raise ConversationLanguageInputError(
                    "normalized utterance 没有已注册 question construction")
        return frame.request_for(value.occurrence_key)


__all__ = [
    "ConversationLanguageInputError",
    "ConversationLexicalEvidence",
    "ConversationLexicalNormalizer",
    "ConversationLexicalRoute",
    "ConversationNormalizedUtterance",
    "ConversationQuestionInputCompiler",
    "ConversationQuestionLanguageInput",
    "ConversationQuestionRequestFrame",
    "ConversationTypedUtterance",
]
