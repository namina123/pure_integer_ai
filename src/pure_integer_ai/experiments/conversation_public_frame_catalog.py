"""DLG-RAW-01 的公开 Frame 目录、递归请求和候选物化边界。

本模块只消费 DLG-RAW-07 public payload closure 中经内容锁的课程和词汇证据。
JSONL 是来源 transport；一经解析，入口和后续 runtime 只消费完整整数 record、
typed ``QuestionRequest`` 与 ``GenerationCandidate``。这里不读取 held-out、
private、终端历史或答案表层，也不接触物理文件系统。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateDefinition,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    AnswerGenerationGoal,
    GenerationCandidate,
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_BINDER,
    OBJECT_LANGUAGE_ATOM,
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_REPRESENTATION,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.cognition.shared.representation_rendering import (
    representation_parts,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.cognition.shared.typed_binding import (
    BoundProposition,
    BoundRoleBinding,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    encode_utf8_v1,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadClosureV1,
    PublicSourcePayloadProviderError,
    public_source_payload_sha256_v1,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    DatasetContractError,
    parse_canonical_json_bytes,
)


PUBLIC_FRAME_CATALOG_SCHEMA_V1 = 1
PUBLIC_FRAME_CATALOG_RECORD_V1 = 1
PUBLIC_FRAME_RECORD_V1 = 1
PUBLIC_FRAME_RECORD_V2 = 2
PUBLIC_FRAME_RECORD_V3 = 3
PUBLIC_FRAME_SOURCE_RECORD_V1 = 1
PUBLIC_FRAME_CONSTRUCTION_RECORD_V1 = 1
PUBLIC_FRAME_ROUTE_RECORD_V1 = 1
PUBLIC_FRAME_QUESTION_RECORD_V1 = 1
PUBLIC_FRAME_RECIPE_RECORD_V1 = 1
PUBLIC_FRAME_RESPONSE_ACT_RECIPE_RECORD_V2 = 2
PUBLIC_FRAME_REFERENCE_RECIPE_RECORD_V3 = 3
PUBLIC_FRAME_PATTERN_SELECTION_LOWEST_VALID_V1 = 1
PUBLIC_FRAME_REFERENCE_SELECTION_LOWEST_COST_V1 = 1

PUBLIC_FRAME_CONTEXT_NONE = 0
PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR = 1

# DLG-RAW-07 的 manifest 只能由已验证的逻辑 payload 闭包取得；它不是
# 当前安装目录或任何物理文件名的语义输入。
PUBLIC_FRAME_CATALOG_LOGICAL_KEY_V1 = (
    b"data/ph2/dlg_raw_public_frame_v1.jsonl.sample")

_TOP_LEVEL_FIELDS = frozenset({
    "catalog_schema",
    "construction",
    "context_requirement",
    "context_target_key",
    "frame_key",
    "lexical_routes",
    "question_frame",
    "runtime_recipe",
    "source_records",
    "surface",
})
_SOURCE_RECORD_FIELDS = frozenset({
    "attribution",
    "license_id",
    "raw_sha256",
    "record_id",
    "relative_path",
    "source_ref_key",
    "span",
    "span_utf8_hex",
})
_ROUTE_FIELDS = frozenset({
    "atom_identity_key",
    "branch_identity_key",
    "evidence_source_record_ids",
    "position",
    "representation_identity_key",
    "scalar_span",
})
_CONSTRUCTION_FIELDS = frozenset({
    "identity_key",
    "semantic_atom_identity_keys",
})
_QUESTION_FIELDS = frozenset({
    "authorized_candidate_target_records",
    "evidence_scope_identity_key",
    "goal_kind_identity_key",
    "intent_identity_key",
    "query_kind_identity_key",
    "required",
    "response_scope_identity_key",
    "target_branch_identity_key",
    "target_branch_present",
    "target_record",
    "trace_prefix",
})
_BOUND_FIELDS = frozenset({
    "applied_variable_identity_keys",
    "bindings",
    "context_identity_key",
    "instruction_identity_key",
    "introduced_binder_identity_keys",
    "predicate_identity_key",
    "source_anchor_identity_key",
    "structure_identity_key",
    "template_identity_key",
})
_BOUND_ROLE_FIELDS = frozenset({
    "filler_bound_proposition",
    "filler_identity_key",
    "ordinal",
    "role_identity_key",
})
_RECIPE_FIELDS = frozenset({
    "candidate_evidence_keys",
    "candidate_evidence_source_record_ids",
    "candidate_state",
    "claim_source_record_id",
    "claim_utf8_hex",
    "course_raw_sha256",
    "course_relative_path",
    "episode_id",
    "g04_required",
    "output_max_bytes",
    "pattern_id",
    "recipe_identity_key",
    "structure_id",
})
_SURFACE_FIELDS = frozenset({"scalars", "utf8_hex"})
_TARGET_BINDING_FIELDS = frozenset({
    "applied_variable_identity_keys",
    "bindings",
    "context_identity_key",
    "instruction_identity_key",
    "introduced_binder_identity_keys",
    "predicate_identity_key",
    "source_anchor_identity_key",
    "structure_identity_key",
    "template_identity_key",
})
_HEX = frozenset("0123456789abcdef")


# object-model: exception; interop=public-catalog-validation
class PublicFrameCatalogError(ValueError):
    """公开 frame 的来源、整数 record 或闭合关系不符合合同。"""


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """将可变长度整数序列以显式长度前缀加入规范 record。"""
    result.extend((len(value), *value))


def _text_u8(value: str, *, label: str) -> tuple[int, ...]:
    """把 transport 文本降为 UTF-8 byte vector，不作为核心匹配依据。"""
    if not isinstance(value, str) or not value:
        raise PublicFrameCatalogError(f"{label} 必须是非空文本")
    return tuple(value.encode("utf-8"))


def _ascii_id(value: Any, *, label: str) -> str:
    """约束 transport record id 为有限 ASCII，避免宿主规范化影响索引。"""
    if (not isinstance(value, str) or not value
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
                   for character in value)):
        raise PublicFrameCatalogError(f"{label} 必须是小写 ASCII record id")
    return value


def _ascii_label(value: Any, *, label: str) -> str:
    """校验公开元数据标签为可跨语言处理的有限 ASCII 字节序列。"""
    if (not isinstance(value, str) or not value
            or any(ord(character) < 0x20 or ord(character) > 0x7E
                   for character in value)):
        raise PublicFrameCatalogError(f"{label} 必须是非空 ASCII 标签")
    return value


def _exact(value: Any, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    """拒绝 JSON transport 的缺字段、尾随字段和非 object 形态。"""
    if not isinstance(value, dict) or set(value) != fields:
        raise PublicFrameCatalogError(f"{label} 字段集合漂移")
    return value


def _strict_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    """拒绝 bool 与整数子类，确保 record 的整数语义可跨语言复现。"""
    if type(value) is not int or (minimum is not None and value < minimum):
        raise PublicFrameCatalogError(f"{label} 不是合法严格整数")
    return value


def _int_vector(
        value: Any,
        *,
        label: str,
        allow_empty: bool,
        minimum: int | None = None,
        ) -> tuple[int, ...]:
    """从 JSON list 恢复有限严格整数序列。"""
    if not isinstance(value, list) or (not allow_empty and not value):
        raise PublicFrameCatalogError(f"{label} 必须是整数 list")
    result = tuple(value)
    for item in result:
        _strict_int(item, label=label, minimum=minimum)
    return result


def _scalar_vector(value: Any, *, label: str, allow_empty: bool) -> tuple[int, ...]:
    """核验 Unicode scalar 序列，拒绝 surrogate 和越界值。"""
    result = _int_vector(value, label=label, allow_empty=allow_empty)
    if any(item > 0x10FFFF or 0xD800 <= item <= 0xDFFF for item in result):
        raise PublicFrameCatalogError(f"{label} 含非法 Unicode scalar")
    return result


def _hex_bytes(value: Any, *, label: str, expected_size: int | None = None) -> tuple[int, ...]:
    """手工把小写 hex transport 还原为 0..255 整数，避免隐式编码。"""
    if (not isinstance(value, str) or len(value) % 2
            or any(character not in _HEX for character in value)):
        raise PublicFrameCatalogError(f"{label} 不是小写 hex")
    result = []
    for cursor in range(0, len(value), 2):
        high = int(value[cursor], 16)
        low = int(value[cursor + 1], 16)
        result.append((high << 4) | low)
    if expected_size is not None and len(result) != expected_size:
        raise PublicFrameCatalogError(f"{label} 长度不符合合同")
    return tuple(result)


def _identity(
        value: Any,
        *,
        label: str,
        expected_kind: int | None = None,
        ) -> ObjectIdentity:
    """从完整 stable key 恢复一等对象并核验分型。"""
    key = _int_vector(value, label=label, allow_empty=False)
    try:
        result = ObjectIdentity.from_stable_key(key)
    except (TypeError, ValueError) as error:
        raise PublicFrameCatalogError(f"{label} 不是完整 ObjectIdentity") from error
    if expected_kind is not None and result.object_kind != expected_kind:
        raise PublicFrameCatalogError(f"{label} object kind 漂移")
    return result


def _source_ref(value: Any, *, label: str) -> SourceRef:
    """从完整 stable key 恢复 SourceRef，禁止用局部 id 代替来源本体。"""
    key = _int_vector(value, label=label, allow_empty=False)
    try:
        return SourceRef.from_stable_key(key)
    except (TypeError, ValueError) as error:
        raise PublicFrameCatalogError(f"{label} 不是完整 SourceRef") from error


def _scope(value: Any, *, label: str) -> ScopeIdentity:
    """从完整 stable key 恢复 scope，保留 source、owner 与版本。"""
    key = _int_vector(value, label=label, allow_empty=False)
    try:
        return ScopeIdentity.from_stable_key(key)
    except (TypeError, ValueError) as error:
        raise PublicFrameCatalogError(f"{label} 不是完整 ScopeIdentity") from error


def _logic_state(value: Any, *, label: str) -> LogicEvidenceState:
    """从显式 support/refute bit 建立四态，不接受 Python truthiness。"""
    bits = _int_vector(value, label=label, allow_empty=False)
    if len(bits) != 2 or any(item not in (0, 1) for item in bits):
        raise PublicFrameCatalogError(f"{label} 必须是两个 0/1 bit")
    try:
        result = LogicEvidenceState(bool(bits[0]), bool(bits[1]))
    except (TypeError, ValueError) as error:
        raise PublicFrameCatalogError(f"{label} 无法形成 EvidenceState") from error
    if not result.support and not result.refute:
        raise PublicFrameCatalogError(f"{label} 至少需要一个 Evidence 方向")
    return result


def _logical_payload_key(value: Any, *, label: str) -> tuple[str, bytes]:
    """将 transport 中的规范相对名还原为 closure 的 ASCII logical key。

    这里只验证冻结逻辑命名空间，不解析宿主路径。注册表完整性、payload
    长度与 SHA 由 ``PublicSourcePayloadClosureV1`` 负责闭合。
    """
    if not isinstance(value, str) or not value or "\\" in value:
        raise PublicFrameCatalogError(f"{label} 不是规范 POSIX logical key")
    try:
        logical_key = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise PublicFrameCatalogError(f"{label} 不是 ASCII logical key") from error
    parts = logical_key.split(b"/")
    if (len(parts) != 3 or tuple(parts[:2]) != (b"data", b"ph2")
            or any(part in (b"", b".", b"..") for part in parts)):
        raise PublicFrameCatalogError(f"{label} 不在冻结 data/ph2 logical namespace")
    return value, logical_key


def _closure_payload(
        closure: PublicSourcePayloadClosureV1,
        value: Any,
        *,
        label: str,
        ) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    """从已验证 closure 取得一项 raw u8[]，不接触物理 I/O 或目录状态。"""
    relative_path, logical_key = _logical_payload_key(value, label=label)
    try:
        record = closure.record_for(logical_key)
    except PublicSourcePayloadProviderError as error:
        raise PublicFrameCatalogError(
            f"{label} 未绑定到已登记 public payload") from error
    return relative_path, tuple(record.raw_payload), tuple(record.raw_sha256)


def _decode_source_scalars(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """复用 DLG-RAW-00 的严格 UTF-8 状态机恢复公开 source span。"""
    intake = intake_raw_conversation_vector(value)
    if not intake.accepted:
        raise PublicFrameCatalogError(f"{label} 不是可接受 UTF-8 source span")
    return intake.unicode_scalars


def _bound_proposition(value: Any, *, label: str) -> BoundProposition:
    """递归重建完整 BoundProposition，stable key 绝不充当反序列化输入。"""
    raw = _exact(value, _BOUND_FIELDS, label=label)
    template = _identity(
        raw["template_identity_key"], label=f"{label}.template",
        expected_kind=7)
    instruction = _identity(
        raw["instruction_identity_key"], label=f"{label}.instruction",
        expected_kind=OBJECT_MINIMAL_INSTRUCTION)
    predicate = _identity(
        raw["predicate_identity_key"], label=f"{label}.predicate",
        expected_kind=4)
    structure = _identity(
        raw["structure_identity_key"], label=f"{label}.structure",
        expected_kind=OBJECT_STRUCTURE_CONCEPT)
    source_anchor = _identity(
        raw["source_anchor_identity_key"], label=f"{label}.source_anchor")
    context = _identity(
        raw["context_identity_key"], label=f"{label}.context")
    binders = tuple(_identity(
        item, label=f"{label}.introduced_binder", expected_kind=OBJECT_BINDER)
        for item in _list(raw["introduced_binder_identity_keys"],
                          label=f"{label}.introduced_binders"))
    bindings = []
    for ordinal, item in enumerate(_list(raw["bindings"], label=f"{label}.bindings")):
        binding = _exact(item, _BOUND_ROLE_FIELDS,
                         label=f"{label}.bindings[{ordinal}]")
        role = _identity(
            binding["role_identity_key"],
            label=f"{label}.bindings[{ordinal}].role",
            expected_kind=20)
        filler_identity = binding["filler_identity_key"]
        filler_bound = binding["filler_bound_proposition"]
        if (filler_identity is None) == (filler_bound is None):
            raise PublicFrameCatalogError(
                f"{label}.bindings[{ordinal}] 必须有且只有一种 filler")
        filler = (
            _identity(filler_identity,
                      label=f"{label}.bindings[{ordinal}].filler")
            if filler_bound is None else _bound_proposition(
                filler_bound, label=f"{label}.bindings[{ordinal}].nested"))
        bindings.append(BoundRoleBinding(
            role,
            filler,
            _strict_int(binding["ordinal"],
                        label=f"{label}.bindings[{ordinal}].ordinal",
                        minimum=0),
        ))
    variables = tuple(_identity(
        item, label=f"{label}.applied_variable", expected_kind=19)
        for item in _list(raw["applied_variable_identity_keys"],
                          label=f"{label}.applied_variables"))
    try:
        return BoundProposition(
            template,
            instruction,
            predicate,
            structure,
            source_anchor,
            context,
            binders,
            tuple(bindings),
            variables,
        )
    except (TypeError, ValueError) as error:
        raise PublicFrameCatalogError(f"{label} 无法形成 BoundProposition") from error


def _list(value: Any, *, label: str) -> list[Any]:
    """为 JSON array 提供显式 type 边界，防止 tuple/dict 隐式进入 transport。"""
    if not isinstance(value, list):
        raise PublicFrameCatalogError(f"{label} 必须是 JSON list")
    return value


def _branch_contains_atom(branch: ObjectIdentity, atom: ObjectIdentity) -> bool:
    """核验 LanguageAtom 完整 identity 内嵌相同 branch components/owner/version。"""
    if atom.owner != branch.owner or atom.versions != branch.versions:
        return False
    components = atom.components
    size = len(branch.components)
    return (
        len(components) > size + 1
        and components[0] == size
        and components[1:1 + size] == branch.components
    )


# object-model: value; representation=struct; interop=public-source-record
@dataclass(frozen=True, slots=True)
class PublicFrameSourceRecord:
    """一条完整 SourceRef、物理字节、许可和 span 的公开来源记录。"""

    record_id: str
    source: SourceRef
    relative_path: str
    raw_sha256: tuple[int, ...]
    license_id: str
    attribution: str
    span: tuple[int, int]
    span_bytes: tuple[int, ...]
    span_scalars: tuple[int, ...]

    def canonical_record(self) -> tuple[int, ...]:
        """导出来源本体而非仅来源 hash 的长度前缀整数 record。"""
        result = [PUBLIC_FRAME_SOURCE_RECORD_V1]
        for value in (
                _text_u8(self.record_id, label="source record id"),
                self.source.stable_key(),
                _text_u8(self.relative_path, label="source relative path"),
                self.raw_sha256,
                _text_u8(self.license_id, label="source license"),
                _text_u8(self.attribution, label="source attribution"),
                self.span_bytes,
                self.span_scalars):
            _pack(result, value)
        result.extend(self.span)
        return tuple(result)


# object-model: value; representation=struct; interop=public-lexical-route
@dataclass(frozen=True, slots=True)
class PublicFrameLexicalRoute:
    """一个 scalar span 到 Representation/LanguageAtom 的双来源词汇路由。"""

    position: int
    scalar_span: tuple[int, int]
    branch: ObjectIdentity
    representation: ObjectIdentity
    atom: ObjectIdentity
    evidence: tuple[PublicFrameSourceRecord, ...]
    scalars: tuple[int, ...]

    def canonical_record(self) -> tuple[int, ...]:
        """保留位置、表层 scalar、两个以上完整来源和语言对象 identity。"""
        result = [PUBLIC_FRAME_ROUTE_RECORD_V1, self.position, *self.scalar_span]
        for value in (
                self.branch.stable_key(),
                self.representation.stable_key(),
                self.atom.stable_key(),
                self.scalars):
            _pack(result, value)
        result.append(len(self.evidence))
        for source in self.evidence:
            _pack(result, source.canonical_record())
        return tuple(result)


# object-model: value; representation=struct; interop=F00-template
@dataclass(frozen=True, slots=True)
class PublicFrameQuestionTemplate:
    """完整十字段 QuestionRequest 的无 occurrence 模板。"""

    query_kind: ObjectIdentity
    intent: ObjectIdentity
    goal_kind: ObjectIdentity
    target: BoundProposition
    required: LogicEvidenceState
    evidence_scope: ScopeIdentity
    response_scope: ScopeIdentity
    trace_prefix: tuple[int, ...]
    target_branch: ObjectIdentity | None
    authorized_candidate_targets: tuple[BoundProposition, ...]

    def __post_init__(self) -> None:
        """在 catalog load 时闭合来源、branch 和完整授权 target 集。"""
        for identity, label in (
                (self.query_kind, "question query kind"),
                (self.intent, "question intent"),
                (self.goal_kind, "question goal kind")):
            if identity.object_kind != OBJECT_MINIMAL_INSTRUCTION:
                raise PublicFrameCatalogError(f"{label} 不是 MinimalInstruction")
        if not self.trace_prefix or any(type(item) is not int
                                        for item in self.trace_prefix):
            raise PublicFrameCatalogError("question trace prefix 非法")
        source = semantic_source(self.target.template)
        if (self.evidence_scope.source != source
                or self.response_scope.source != source):
            raise PublicFrameCatalogError("question scope 与 target source 漂移")
        if (not self.authorized_candidate_targets
                or self.target not in self.authorized_candidate_targets
                or len(set(self.authorized_candidate_targets))
                != len(self.authorized_candidate_targets)):
            raise PublicFrameCatalogError("question authorized targets 不闭合")
        if any(semantic_source(item.template) != source
               for item in self.authorized_candidate_targets):
            raise PublicFrameCatalogError("question authorized target source 漂移")
        if self.target_branch is None:
            raise PublicFrameCatalogError("公开首句必须显式携带 target branch")
        if self.target_branch.object_kind != OBJECT_LANGUAGE_BRANCH:
            raise PublicFrameCatalogError("question target branch 类型错误")
        for identity in (
                self.target.template,
                self.target.predicate,
                self.target.structure):
            if (identity.owner != self.target_branch.owner
                    or identity.versions != self.target_branch.versions):
                raise PublicFrameCatalogError("target branch owner/version 漂移")
        object.__setattr__(self, "authorized_candidate_targets", tuple(sorted(
            self.authorized_candidate_targets,
            key=BoundProposition.stable_key,
        )))

    def request_for(self, occurrence_key: tuple[int, ...]) -> QuestionRequest:
        """只追加调用方给定的单调 occurrence，不读取终端行号或墙钟。"""
        occurrence = _int_vector(
            list(occurrence_key), label="question occurrence", allow_empty=False)
        try:
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
        except (TypeError, ValueError) as error:
            raise PublicFrameCatalogError("完整 QuestionRequest 无法物化") from error

    def canonical_record(self) -> tuple[int, ...]:
        """输出含递归 target/binder/role 的十字段完整整数 record。"""
        result = [PUBLIC_FRAME_QUESTION_RECORD_V1]
        for value in (
                self.query_kind.stable_key(),
                self.intent.stable_key(),
                self.goal_kind.stable_key(),
                self.target.stable_key(),
                self.required.stable_key(),
                self.evidence_scope.stable_key(),
                self.response_scope.stable_key(),
                self.trace_prefix):
            _pack(result, value)
        result.append(0 if self.target_branch is None else 1)
        if self.target_branch is not None:
            _pack(result, self.target_branch.stable_key())
        result.append(len(self.authorized_candidate_targets))
        for target in self.authorized_candidate_targets:
            _pack(result, target.stable_key())
        return tuple(result)


# object-model: value; representation=struct; interop=label-free-runtime-recipe
@dataclass(frozen=True, slots=True)
class PublicFrameRuntimeRecipe:
    """不含答案表层或 expected label 的公开候选与 G-03/G-04 配方。"""

    identity: ObjectIdentity
    course_relative_path: str
    course_raw_sha256: tuple[int, ...]
    episode_id: str
    candidate_state: LogicEvidenceState
    candidate_evidence: tuple[EvidenceRecord, ...]
    candidate_evidence_source_record_ids: tuple[str, ...]
    claim_source_record_id: str
    claim_scalars: tuple[int, ...]
    pattern_id: int
    structure_id: int
    output_max_bytes: int
    g04_required: int

    def __post_init__(self) -> None:
        """冻结 evidence/source 对位、course 内容锁与无标签生成预算。"""
        if self.identity.object_kind != OBJECT_MINIMAL_INSTRUCTION:
            raise PublicFrameCatalogError("runtime recipe identity 类型错误")
        if len(self.course_raw_sha256) != 32:
            raise PublicFrameCatalogError("runtime recipe course SHA 长度错误")
        if (not self.episode_id or not self.candidate_evidence
                or len(self.candidate_evidence)
                != len(self.candidate_evidence_source_record_ids)):
            raise PublicFrameCatalogError("runtime recipe evidence/source 不闭合")
        if any(not isinstance(item, EvidenceRecord)
               for item in self.candidate_evidence):
            raise PublicFrameCatalogError("runtime recipe evidence 类型错误")
        if (self.pattern_id <= 0 or self.structure_id <= 0
                or self.output_max_bytes <= 0 or self.g04_required != 1):
            raise PublicFrameCatalogError("runtime recipe 参数非法")
        if not self.claim_scalars:
            raise PublicFrameCatalogError("runtime recipe claim 不得为空")

    def canonical_record(self) -> tuple[int, ...]:
        """导出无自然语言答案的候选、claim、pattern 和输出预算 record。"""
        result = [PUBLIC_FRAME_RECIPE_RECORD_V1]
        for value in (
                self.identity.stable_key(),
                _text_u8(self.course_relative_path, label="recipe course path"),
                self.course_raw_sha256,
                _text_u8(self.episode_id, label="recipe episode id"),
                self.candidate_state.stable_key(),
                _text_u8(self.claim_source_record_id, label="recipe claim source"),
                self.claim_scalars):
            _pack(result, value)
        result.extend((self.pattern_id, self.structure_id,
                       self.output_max_bytes, self.g04_required,
                       len(self.candidate_evidence)))
        for evidence, source_id in zip(
                self.candidate_evidence,
                self.candidate_evidence_source_record_ids,
                strict=True):
            _pack(result, evidence.stable_key())
            _pack(result, _text_u8(source_id, label="recipe evidence source"))
        return tuple(result)


# object-model: value; representation=struct; interop=label-free-response-act-runtime
@dataclass(frozen=True, slots=True)
class PublicFrameResponseActRuntimeRecipe:
    """V2 公开课程派生的非回答配方，不保存 expected stance 或表面。"""

    identity: ObjectIdentity
    course_relative_path: str
    course_raw_sha256: tuple[int, ...]
    episode_id: str
    planning_input_record: tuple[int, ...]
    course_source_record_id: str
    evidence_source_record_ids: tuple[str, ...]
    output_max_bytes: int
    g04_required: int
    pattern_selection_policy: int = (
        PUBLIC_FRAME_PATTERN_SELECTION_LOWEST_VALID_V1)

    def __post_init__(self) -> None:
        """冻结课程、无标签 planning input 和确定性 pattern 选择规则。"""
        if self.identity.object_kind != OBJECT_MINIMAL_INSTRUCTION:
            raise PublicFrameCatalogError("response-act recipe identity 类型错误")
        if (not isinstance(self.course_relative_path, str)
                or not self.course_relative_path
                or len(self.course_raw_sha256) != 32
                or any(type(item) is not int or item < 0 or item > 255
                       for item in self.course_raw_sha256)
                or not self.episode_id
                or not isinstance(self.planning_input_record, tuple)
                or not self.planning_input_record
                or any(type(item) is not int
                       for item in self.planning_input_record)
                or not self.course_source_record_id):
            raise PublicFrameCatalogError("response-act recipe 内容锁非法")
        if (not isinstance(self.evidence_source_record_ids, tuple)
                or any(not isinstance(item, str) or not item
                       for item in self.evidence_source_record_ids)
                or self.evidence_source_record_ids
                != tuple(sorted(set(self.evidence_source_record_ids)))):
            raise PublicFrameCatalogError("response-act recipe Evidence source 非法")
        if (type(self.output_max_bytes) is not int
                or self.output_max_bytes <= 0
                or self.g04_required != 1
                or self.pattern_selection_policy
                != PUBLIC_FRAME_PATTERN_SELECTION_LOWEST_VALID_V1):
            raise PublicFrameCatalogError("response-act recipe 参数非法")

    def canonical_record(self) -> tuple[int, ...]:
        """导出没有 answer label/surface 的 V2 runtime 输入 record。"""
        result = [PUBLIC_FRAME_RESPONSE_ACT_RECIPE_RECORD_V2]
        for value in (
                self.identity.stable_key(),
                _text_u8(self.course_relative_path,
                         label="response-act recipe course path"),
                self.course_raw_sha256,
                _text_u8(self.episode_id,
                         label="response-act recipe episode id"),
                self.planning_input_record,
                _text_u8(self.course_source_record_id,
                         label="response-act recipe course source")):
            _pack(result, value)
        result.append(len(self.evidence_source_record_ids))
        for source_id in self.evidence_source_record_ids:
            _pack(result, _text_u8(source_id,
                                   label="response-act recipe evidence source"))
        result.extend((
            self.output_max_bytes,
            self.g04_required,
            self.pattern_selection_policy,
        ))
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-05B
@dataclass(frozen=True, slots=True)
class PublicFrameReferenceRuntimeRecipe:
    """V3 双 claim/reference 配方，只保存无标签 planning 与来源化结构输入。"""

    identity: ObjectIdentity
    course_relative_path: str
    course_raw_sha256: tuple[int, ...]
    episode_id: str
    planning_input_record: tuple[int, ...]
    ordered_proposition_ids: tuple[str, ...]
    antecedent_proposition_id: str
    referring_proposition_id: str
    relation_kind_code: int
    course_source_record_id: str
    evidence_source_record_ids: tuple[str, ...]
    antecedent_reference_source_record_ids: tuple[str, ...]
    explicit_repetition_source_record_ids: tuple[str, ...]
    output_max_bytes: int
    g04_required: int
    strategy_selection_policy: int = (
        PUBLIC_FRAME_REFERENCE_SELECTION_LOWEST_COST_V1)

    def __post_init__(self) -> None:
        """冻结两个 claim 的顺序、结构词汇来源和无标签运行期预算。"""
        if self.identity.object_kind != OBJECT_MINIMAL_INSTRUCTION:
            raise PublicFrameCatalogError("reference recipe identity 类型错误")
        if (not isinstance(self.course_relative_path, str)
                or not self.course_relative_path
                or len(self.course_raw_sha256) != 32
                or any(type(item) is not int or item < 0 or item > 255
                       for item in self.course_raw_sha256)
                or not isinstance(self.episode_id, str)
                or not self.episode_id
                or not isinstance(self.planning_input_record, tuple)
                or not self.planning_input_record
                or any(type(item) is not int
                       for item in self.planning_input_record)
                or not isinstance(self.course_source_record_id, str)
                or not self.course_source_record_id):
            raise PublicFrameCatalogError("reference recipe 内容锁非法")
        if (not isinstance(self.ordered_proposition_ids, tuple)
                or len(self.ordered_proposition_ids) != 2
                or any(not isinstance(item, str) or not item
                       for item in self.ordered_proposition_ids)
                or len(set(self.ordered_proposition_ids)) != 2):
            raise PublicFrameCatalogError("reference recipe 必须有两个有序 Proposition id")
        if (self.antecedent_proposition_id != self.ordered_proposition_ids[0]
                or self.referring_proposition_id
                != self.ordered_proposition_ids[1]
                or type(self.relation_kind_code) is not int
                or self.relation_kind_code != 1):
            raise PublicFrameCatalogError(
                "reference recipe antecedent/referring relation 漂移")
        for label, record_ids, minimum_count in (
                ("evidence", self.evidence_source_record_ids, 1),
                ("antecedent reference",
                 self.antecedent_reference_source_record_ids, 2),
                ("explicit repetition",
                 self.explicit_repetition_source_record_ids, 2)):
            if (not isinstance(record_ids, tuple)
                    or len(record_ids) < minimum_count
                    or any(not isinstance(item, str) or not item
                           for item in record_ids)
                    or record_ids != tuple(sorted(set(record_ids)))):
                raise PublicFrameCatalogError(
                    f"reference recipe {label} source record 非法")
        if (type(self.output_max_bytes) is not int
                or self.output_max_bytes <= 0
                or type(self.g04_required) is not int
                or self.g04_required != 1
                or type(self.strategy_selection_policy) is not int
                or self.strategy_selection_policy
                != PUBLIC_FRAME_REFERENCE_SELECTION_LOWEST_COST_V1):
            raise PublicFrameCatalogError("reference recipe 参数非法")

    def canonical_record(self) -> tuple[int, ...]:
        """导出可跨语言重建的 V3 public reference runtime 输入。"""
        result = [PUBLIC_FRAME_REFERENCE_RECIPE_RECORD_V3]
        for value in (
                self.identity.stable_key(),
                _text_u8(self.course_relative_path,
                         label="reference recipe course path"),
                self.course_raw_sha256,
                _text_u8(self.episode_id,
                         label="reference recipe episode id"),
                self.planning_input_record,
                _text_u8(self.course_source_record_id,
                         label="reference recipe course source")):
            _pack(result, value)
        result.append(len(self.ordered_proposition_ids))
        for proposition_id in self.ordered_proposition_ids:
            _pack(result, _text_u8(
                proposition_id,
                label="reference recipe proposition id"))
        for proposition_id in (
                self.antecedent_proposition_id,
                self.referring_proposition_id):
            _pack(result, _text_u8(
                proposition_id,
                label="reference recipe relation proposition id"))
        result.append(self.relation_kind_code)
        for record_ids in (
                self.evidence_source_record_ids,
                self.antecedent_reference_source_record_ids,
                self.explicit_repetition_source_record_ids):
            result.append(len(record_ids))
            for record_id in record_ids:
                _pack(result, _text_u8(
                    record_id,
                    label="reference recipe source record"))
        result.extend((
            self.output_max_bytes,
            self.g04_required,
            self.strategy_selection_policy,
        ))
        return tuple(result)


# object-model: value; representation=struct; interop=public-frame
@dataclass(frozen=True, slots=True)
class PublicFrame:
    """一个完整 raw scalar surface、词汇证据、构式和 F-00 请求的公开 frame。"""

    frame_key: str
    raw_line_sha256: tuple[int, ...]
    surface_bytes: tuple[int, ...]
    surface_scalars: tuple[int, ...]
    source_records: tuple[PublicFrameSourceRecord, ...]
    routes: tuple[PublicFrameLexicalRoute, ...]
    construction: ObjectIdentity
    construction_atoms: tuple[ObjectIdentity, ...]
    question: PublicFrameQuestionTemplate
    recipe: (
        PublicFrameRuntimeRecipe
        | PublicFrameResponseActRuntimeRecipe
        | PublicFrameReferenceRuntimeRecipe
    )
    context_requirement: int
    context_target_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """验证 lexical 全覆盖、构式 atom 序和 context 的冷启动规则。"""
        if not self.frame_key or len(self.raw_line_sha256) != 32:
            raise PublicFrameCatalogError("public frame identity 不闭合")
        if encode_utf8_v1(self.surface_scalars) != self.surface_bytes:
            raise PublicFrameCatalogError("public frame surface UTF-8 漂移")
        if (not self.source_records
                or self.source_records != tuple(sorted(
                    self.source_records,
                    key=lambda item: item.source.stable_key()))
                or len({item.record_id for item in self.source_records})
                != len(self.source_records)
                or len({item.source.stable_key() for item in self.source_records})
                != len(self.source_records)):
            raise PublicFrameCatalogError("public frame source records 不闭合")
        source_index = {item.record_id: item for item in self.source_records}
        if (not self.routes or self.routes != tuple(sorted(
                self.routes, key=lambda item: item.position))):
            raise PublicFrameCatalogError("public frame routes 未按位置排序")
        cursor = 0
        branch = self.routes[0].branch
        family = representation_parts(self.routes[0].representation)[0]
        atoms = []
        for position, route in enumerate(self.routes):
            if (route.position != position or route.scalar_span[0] != cursor
                    or route.scalar_span[1] <= cursor
                    or route.scalar_span[1] > len(self.surface_scalars)
                    or route.scalars != self.surface_scalars[
                        route.scalar_span[0]:route.scalar_span[1]]
                    or route.branch != branch
                    or route.branch.object_kind != OBJECT_LANGUAGE_BRANCH
                    or route.representation.object_kind != OBJECT_REPRESENTATION
                    or route.atom.object_kind != OBJECT_LANGUAGE_ATOM
                    or not _branch_contains_atom(branch, route.atom)):
                raise PublicFrameCatalogError("public frame lexical route 漂移")
            route_family, content = representation_parts(route.representation)
            if (route_family != family
                    or content != (route.position, len(route.scalars),
                                   *route.scalars)):
                raise PublicFrameCatalogError("Representation 未完整绑定 raw scalar")
            if len(route.evidence) < 2 or len({item.source.stable_key()
                                               for item in route.evidence}) < 2:
                raise PublicFrameCatalogError("lexical route 缺两个独立 SourceRef")
            if any(source_index.get(item.record_id) != item
                   for item in route.evidence):
                raise PublicFrameCatalogError("lexical route 引用了外部 source record")
            if any(item.span_scalars != route.scalars for item in route.evidence):
                raise PublicFrameCatalogError("lexical evidence span 与 Representation 漂移")
            cursor = route.scalar_span[1]
            atoms.append(route.atom)
        if cursor != len(self.surface_scalars):
            raise PublicFrameCatalogError("public frame scalar 未被 lexical route 完整覆盖")
        if (self.construction.object_kind != OBJECT_STRUCTURE_CONCEPT
                or not isinstance(self.construction_atoms, tuple)
                or any(not isinstance(item, ObjectIdentity)
                       or item.object_kind != OBJECT_LANGUAGE_ATOM
                       for item in self.construction_atoms)
                or tuple(atoms) != self.construction_atoms):
            raise PublicFrameCatalogError("public frame construction 不闭合")
        if isinstance(self.recipe, PublicFrameRuntimeRecipe):
            recipe_sources = {
                self.recipe.claim_source_record_id,
                *self.recipe.candidate_evidence_source_record_ids,
            }
        elif isinstance(self.recipe, PublicFrameResponseActRuntimeRecipe):
            recipe_sources = {
                self.recipe.course_source_record_id,
                *self.recipe.evidence_source_record_ids,
            }
        elif isinstance(self.recipe, PublicFrameReferenceRuntimeRecipe):
            recipe_sources = {
                self.recipe.course_source_record_id,
                *self.recipe.evidence_source_record_ids,
                *self.recipe.antecedent_reference_source_record_ids,
                *self.recipe.explicit_repetition_source_record_ids,
            }
        else:
            raise TypeError("public frame runtime recipe 类型错误")
        if any(source_id not in source_index for source_id in recipe_sources):
            raise PublicFrameCatalogError("runtime recipe 引用了外部 source record")
        if self.question.target_branch != branch:
            raise PublicFrameCatalogError("question target branch 与 lexical branch 漂移")
        if self.context_requirement == PUBLIC_FRAME_CONTEXT_NONE:
            if self.context_target_key:
                raise PublicFrameCatalogError("NONE frame 不得携带 context anchor")
        elif self.context_requirement == PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR:
            if not self.context_target_key:
                raise PublicFrameCatalogError("TARGET_ANCHOR frame 缺 target key")
        else:
            raise PublicFrameCatalogError("context requirement 未注册")

    def canonical_record(self) -> tuple[int, ...]:
        """导出 source、lexical、构式、十字段请求和无标签 recipe 的完整 record。"""
        result = [
            (PUBLIC_FRAME_RECORD_V1
             if isinstance(self.recipe, PublicFrameRuntimeRecipe)
             else (PUBLIC_FRAME_RECORD_V2
                   if isinstance(self.recipe,
                                 PublicFrameResponseActRuntimeRecipe)
                   else PUBLIC_FRAME_RECORD_V3))
        ]
        for value in (
                _text_u8(self.frame_key, label="frame key"),
                self.raw_line_sha256,
                self.surface_bytes,
                self.surface_scalars,
                self.question.canonical_record(),
                self.recipe.canonical_record(),
                self.context_target_key):
            _pack(result, value)
        construction = [PUBLIC_FRAME_CONSTRUCTION_RECORD_V1]
        _pack(construction, self.construction.stable_key())
        construction.append(len(self.construction_atoms))
        for atom in self.construction_atoms:
            _pack(construction, atom.stable_key())
        _pack(result, tuple(construction))
        result.append(len(self.source_records))
        for source in self.source_records:
            _pack(result, source.canonical_record())
        result.extend((self.context_requirement, len(self.routes)))
        for route in self.routes:
            _pack(result, route.canonical_record())
        return tuple(result)


# object-model: value; representation=struct; interop=public-frame-catalog
@dataclass(frozen=True, slots=True)
class PublicFrameCatalog:
    """只读公开 frame 向量；索引可删除并由规范 frame vector 恢复。"""

    source_sha256: tuple[int, ...]
    frames: tuple[PublicFrame, ...]
    construction_keys: tuple[tuple[int, ...], ...] = ()
    _surface_index: tuple[tuple[tuple[int, ...], tuple[PublicFrame, ...]], ...] = field(
        init=False, repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        """冻结 raw source identity、frame 规范排序和可删重建的 exact-scalar 索引。"""
        if len(self.source_sha256) != 32 or any(type(item) is not int
                                                or item < 0 or item > 255
                                                for item in self.source_sha256):
            raise PublicFrameCatalogError("catalog SHA-256 非法")
        if (not self.frames or self.frames != tuple(sorted(
                self.frames, key=PublicFrame.canonical_record))):
            raise PublicFrameCatalogError("catalog frames 未规范排序")
        keys = self.construction_keys or tuple(sorted({
            item.construction.stable_key() for item in self.frames
        }))
        if (not isinstance(keys, tuple)
                or any(not isinstance(item, tuple) or not item
                       or any(type(value) is not int for value in item)
                       for item in keys)
                or keys != tuple(sorted(set(keys)))):
            raise PublicFrameCatalogError("catalog construction registry 非法")
        target_keys = {item.question.target.stable_key() for item in self.frames}
        if any(item.context_requirement == PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR
               and item.context_target_key not in target_keys
               for item in self.frames):
            raise PublicFrameCatalogError(
                "TARGET_ANCHOR 未绑定 catalog 内完整 target")
        object.__setattr__(self, "construction_keys", keys)
        grouped: dict[tuple[int, ...], list[PublicFrame]] = {}
        for frame in self.frames:
            grouped.setdefault(frame.surface_scalars, []).append(frame)
        object.__setattr__(self, "_surface_index", tuple(
            (surface, tuple(value))
            for surface, value in sorted(grouped.items())
        ))

    def matching_frames(self, scalars: tuple[int, ...]) -> tuple[PublicFrame, ...]:
        """按 scalar 精确匹配 frame，不调用宿主文本等价或隐式 normalize。"""
        for surface, frames in self._surface_index:
            if surface == scalars:
                return frames
        return ()

    def has_construction(self, construction: ObjectIdentity) -> bool:
        """查询已注册构式；缺失是可观察 construction miss，而非 lexical miss。"""
        return construction.stable_key() in self.construction_keys

    def canonical_record(self) -> tuple[int, ...]:
        """返回可跨语言重建的完整 catalog record，不依赖 Python index。"""
        result = [PUBLIC_FRAME_CATALOG_RECORD_V1]
        _pack(result, self.source_sha256)
        result.append(len(self.construction_keys))
        for key in self.construction_keys:
            _pack(result, key)
        result.append(len(self.frames))
        for frame in self.frames:
            _pack(result, frame.canonical_record())
        return tuple(result)


def _parse_source_records(
        value: Any,
        *,
        closure: PublicSourcePayloadClosureV1,
        ) -> tuple[PublicFrameSourceRecord, ...]:
    """校验 closure 内每条 source 的 SHA、许可和不可截断 raw span。"""
    rows = _list(value, label="source_records")
    if not rows:
        raise PublicFrameCatalogError("source_records 不能为空")
    records = []
    for ordinal, item in enumerate(rows):
        raw = _exact(item, _SOURCE_RECORD_FIELDS,
                     label=f"source_records[{ordinal}]")
        record_id = _ascii_id(raw["record_id"],
                              label=f"source_records[{ordinal}].record_id")
        relative_path, source_bytes, closure_sha256 = _closure_payload(
            closure,
            raw["relative_path"],
            label=f"source_records[{ordinal}].relative_path")
        source = _source_ref(raw["source_ref_key"],
                             label=f"source_records[{ordinal}].source_ref")
        digest = _hex_bytes(raw["raw_sha256"],
                            label=f"source_records[{ordinal}].raw_sha256",
                            expected_size=32)
        if closure_sha256 != digest:
            raise PublicFrameCatalogError("公开 source raw SHA-256 漂移")
        span = _int_vector(raw["span"], label=f"source_records[{ordinal}].span",
                           allow_empty=False, minimum=0)
        if len(span) != 2 or span[0] >= span[1] or span[1] > len(source_bytes):
            raise PublicFrameCatalogError("公开 source span 越界")
        span_bytes = _hex_bytes(raw["span_utf8_hex"],
                                label=f"source_records[{ordinal}].span_hex")
        if source_bytes[span[0]:span[1]] != span_bytes:
            raise PublicFrameCatalogError("公开 source span bytes 漂移")
        license_id = _ascii_label(raw["license_id"],
                                  label=f"source_records[{ordinal}].license")
        attribution = raw["attribution"]
        if not isinstance(attribution, str) or not attribution:
            raise PublicFrameCatalogError("公开 source attribution 为空")
        records.append(PublicFrameSourceRecord(
            record_id,
            source,
            relative_path,
            digest,
            license_id,
            attribution,
            (span[0], span[1]),
            span_bytes,
            _decode_source_scalars(
                span_bytes, label=f"source_records[{ordinal}].span"),
        ))
    if len({item.record_id for item in records}) != len(records):
        raise PublicFrameCatalogError("source record id 重复")
    if len({item.source.stable_key() for item in records}) != len(records):
        raise PublicFrameCatalogError("source SourceRef 重复")
    return tuple(sorted(records, key=lambda item: item.source.stable_key()))


def _parse_route(
        value: Any,
        *,
        source_index: dict[str, PublicFrameSourceRecord],
        surface_scalars: tuple[int, ...],
        label: str,
        ) -> PublicFrameLexicalRoute:
    """从 raw scalar span 与两个公开来源恢复唯一 lexical route。"""
    raw = _exact(value, _ROUTE_FIELDS, label=label)
    position = _strict_int(raw["position"], label=f"{label}.position", minimum=0)
    span = _int_vector(raw["scalar_span"], label=f"{label}.scalar_span",
                       allow_empty=False, minimum=0)
    if len(span) != 2 or span[0] >= span[1] or span[1] > len(surface_scalars):
        raise PublicFrameCatalogError(f"{label}.scalar_span 越界")
    branch = _identity(raw["branch_identity_key"], label=f"{label}.branch",
                       expected_kind=OBJECT_LANGUAGE_BRANCH)
    representation = _identity(
        raw["representation_identity_key"], label=f"{label}.representation",
        expected_kind=OBJECT_REPRESENTATION)
    atom = _identity(raw["atom_identity_key"], label=f"{label}.atom",
                     expected_kind=OBJECT_LANGUAGE_ATOM)
    ids = tuple(_ascii_id(item, label=f"{label}.evidence id")
                for item in _list(raw["evidence_source_record_ids"],
                                  label=f"{label}.evidence ids"))
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise PublicFrameCatalogError(f"{label} 缺独立 lexical evidence")
    try:
        evidence = tuple(source_index[item] for item in ids)
    except KeyError as error:
        raise PublicFrameCatalogError(f"{label} 引用了未知 source record") from error
    if evidence != tuple(sorted(evidence, key=lambda item: item.source.stable_key())):
        raise PublicFrameCatalogError(f"{label} evidence 未按 SourceRef 排序")
    scalars = surface_scalars[span[0]:span[1]]
    return PublicFrameLexicalRoute(
        position, (span[0], span[1]), branch, representation, atom,
        evidence, scalars)


def _parse_question(value: Any, *, label: str) -> PublicFrameQuestionTemplate:
    """直接解码递归 target 与十字段 F-00 request 模板。"""
    raw = _exact(value, _QUESTION_FIELDS, label=label)
    query_kind = _identity(raw["query_kind_identity_key"],
                           label=f"{label}.query_kind",
                           expected_kind=OBJECT_MINIMAL_INSTRUCTION)
    intent = _identity(raw["intent_identity_key"], label=f"{label}.intent",
                       expected_kind=OBJECT_MINIMAL_INSTRUCTION)
    goal_kind = _identity(raw["goal_kind_identity_key"],
                          label=f"{label}.goal_kind",
                          expected_kind=OBJECT_MINIMAL_INSTRUCTION)
    target = _bound_proposition(raw["target_record"], label=f"{label}.target")
    required = _logic_state(raw["required"], label=f"{label}.required")
    evidence_scope = _scope(raw["evidence_scope_identity_key"],
                            label=f"{label}.evidence_scope")
    response_scope = _scope(raw["response_scope_identity_key"],
                            label=f"{label}.response_scope")
    trace = _int_vector(raw["trace_prefix"], label=f"{label}.trace_prefix",
                        allow_empty=False)
    present = _strict_int(raw["target_branch_present"],
                          label=f"{label}.target_branch_present", minimum=0)
    if present not in (0, 1):
        raise PublicFrameCatalogError(f"{label}.target_branch_present 非法")
    branch_key = raw["target_branch_identity_key"]
    if present == 0:
        if branch_key != []:
            raise PublicFrameCatalogError(f"{label} target branch presence 漂移")
        branch = None
    else:
        branch = _identity(branch_key, label=f"{label}.target_branch",
                           expected_kind=OBJECT_LANGUAGE_BRANCH)
    authorized = tuple(_bound_proposition(
        item, label=f"{label}.authorized_target")
        for item in _list(raw["authorized_candidate_target_records"],
                          label=f"{label}.authorized_targets"))
    return PublicFrameQuestionTemplate(
        query_kind,
        intent,
        goal_kind,
        target,
        required,
        evidence_scope,
        response_scope,
        trace,
        branch,
        authorized,
    )


def _parse_recipe(
        value: Any,
        *,
        source_index: dict[str, PublicFrameSourceRecord],
        closure: PublicSourcePayloadClosureV1,
        label: str,
        ) -> PublicFrameRuntimeRecipe:
    """从 closure 校验不含答案表层的实际候选与完整公开课程锁。"""
    raw = _exact(value, _RECIPE_FIELDS, label=label)
    identity = _identity(raw["recipe_identity_key"], label=f"{label}.identity",
                         expected_kind=OBJECT_MINIMAL_INSTRUCTION)
    relative_path, _course_bytes, closure_sha256 = _closure_payload(
        closure,
        raw["course_relative_path"],
        label=f"{label}.course_relative_path")
    digest = _hex_bytes(raw["course_raw_sha256"],
                        label=f"{label}.course_raw_sha256", expected_size=32)
    if closure_sha256 != digest:
        raise PublicFrameCatalogError("runtime recipe course SHA 漂移")
    episode_id = _ascii_id(raw["episode_id"], label=f"{label}.episode_id")
    state = _logic_state(raw["candidate_state"], label=f"{label}.candidate_state")
    evidence_keys = _list(raw["candidate_evidence_keys"],
                          label=f"{label}.candidate_evidence_keys")
    if not evidence_keys:
        raise PublicFrameCatalogError("runtime recipe 缺 candidate evidence")
    evidence = []
    for ordinal, item in enumerate(evidence_keys):
        key = _int_vector(item, label=f"{label}.evidence[{ordinal}]",
                          allow_empty=False)
        try:
            evidence.append(EvidenceRecord.from_stable_key(key))
        except (TypeError, ValueError) as error:
            raise PublicFrameCatalogError("runtime recipe Evidence record 损坏") from error
    source_ids = tuple(_ascii_id(
        item, label=f"{label}.candidate_evidence_source")
        for item in _list(raw["candidate_evidence_source_record_ids"],
                          label=f"{label}.candidate_evidence_sources"))
    if len(source_ids) != len(evidence) or len(set(source_ids)) != len(source_ids):
        raise PublicFrameCatalogError("runtime recipe Evidence source 对位漂移")
    for record, source_id in zip(evidence, source_ids, strict=True):
        source = source_index.get(source_id)
        if source is None or source.source != record.source:
            raise PublicFrameCatalogError("runtime recipe Evidence SourceRef 漂移")
    claim_source_id = _ascii_id(raw["claim_source_record_id"],
                                label=f"{label}.claim_source")
    claim_source = source_index.get(claim_source_id)
    if claim_source is None:
        raise PublicFrameCatalogError("runtime recipe claim source 缺失")
    claim_bytes = _hex_bytes(raw["claim_utf8_hex"], label=f"{label}.claim")
    claim_scalars = _decode_source_scalars(claim_bytes, label=f"{label}.claim")
    if claim_scalars != claim_source.span_scalars:
        raise PublicFrameCatalogError("runtime recipe claim/source span 漂移")
    return PublicFrameRuntimeRecipe(
        identity,
        relative_path,
        digest,
        episode_id,
        state,
        tuple(evidence),
        source_ids,
        claim_source_id,
        claim_scalars,
        _strict_int(raw["pattern_id"], label=f"{label}.pattern_id", minimum=1),
        _strict_int(raw["structure_id"], label=f"{label}.structure_id", minimum=1),
        _strict_int(raw["output_max_bytes"], label=f"{label}.output_max_bytes", minimum=1),
        _strict_int(raw["g04_required"], label=f"{label}.g04_required", minimum=0),
    )


def _parse_frame(
        value: Any,
        *,
        raw_line_sha256: tuple[int, ...],
        closure: PublicSourcePayloadClosureV1,
        ) -> PublicFrame:
    """从 closure 解析一行 canonical JSONL，物化完整公开 frame。"""
    raw = _exact(value, _TOP_LEVEL_FIELDS, label="public frame")
    if _strict_int(raw["catalog_schema"], label="catalog schema") != PUBLIC_FRAME_CATALOG_SCHEMA_V1:
        raise PublicFrameCatalogError("public frame schema 未注册")
    frame_key = _ascii_id(raw["frame_key"], label="frame key")
    surface = _exact(raw["surface"], _SURFACE_FIELDS, label="surface")
    scalars = _scalar_vector(surface["scalars"], label="surface scalars",
                             allow_empty=False)
    surface_bytes = _hex_bytes(surface["utf8_hex"], label="surface utf8")
    if encode_utf8_v1(scalars) != surface_bytes:
        raise PublicFrameCatalogError("surface scalar/UTF-8 不一致")
    sources = _parse_source_records(raw["source_records"], closure=closure)
    source_index = {item.record_id: item for item in sources}
    routes = tuple(_parse_route(
        item,
        source_index=source_index,
        surface_scalars=scalars,
        label=f"lexical_routes[{ordinal}]",
    ) for ordinal, item in enumerate(_list(raw["lexical_routes"],
                                            label="lexical_routes")))
    construction_raw = _exact(raw["construction"], _CONSTRUCTION_FIELDS,
                              label="construction")
    construction = _identity(construction_raw["identity_key"],
                             label="construction identity",
                             expected_kind=OBJECT_STRUCTURE_CONCEPT)
    atom_keys = tuple(_identity(
        item, label="construction atom", expected_kind=OBJECT_LANGUAGE_ATOM)
        for item in _list(construction_raw["semantic_atom_identity_keys"],
                          label="construction atoms"))
    if atom_keys != tuple(item.atom for item in routes):
        raise PublicFrameCatalogError("construction atom 序与 lexical route 漂移")
    question = _parse_question(raw["question_frame"], label="question_frame")
    recipe = _parse_recipe(raw["runtime_recipe"], source_index=source_index,
                           closure=closure,
                           label="runtime_recipe")
    context_text = raw["context_requirement"]
    if context_text == "NONE":
        context_requirement = PUBLIC_FRAME_CONTEXT_NONE
    elif context_text == "TARGET_ANCHOR":
        context_requirement = PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR
    else:
        raise PublicFrameCatalogError("context requirement 未注册")
    context_target = _int_vector(raw["context_target_key"],
                                 label="context target key", allow_empty=True)
    frame = PublicFrame(
        frame_key,
        raw_line_sha256,
        surface_bytes,
        scalars,
        sources,
        routes,
        construction,
        atom_keys,
        question,
        recipe,
        context_requirement,
        context_target,
    )
    return frame


def load_public_frame_catalog_from_closure(
        closure: PublicSourcePayloadClosureV1,
        ) -> PublicFrameCatalog:
    """由完整 public payload closure 加载 V1 frame catalog 纯核心。

    manifest、本体 source 和 runtime course 均经同一个 closure 的固定 logical
    key 查得。这里不接收物理位置、目录状态或 host 读取异常。
    """
    if type(closure) is not PublicSourcePayloadClosureV1:
        raise PublicFrameCatalogError("public frame loader 需要完整 payload closure")
    try:
        payload = closure.payload_for(PUBLIC_FRAME_CATALOG_LOGICAL_KEY_V1)
    except PublicSourcePayloadProviderError as error:
        raise PublicFrameCatalogError("public frame manifest 不在 payload closure") from error
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise PublicFrameCatalogError("public frame catalog JSONL 换行非法")
    lines = payload[:-1].split(b"\n")
    if not lines or any(not line for line in lines):
        raise PublicFrameCatalogError("public frame catalog 含空记录")
    frames = []
    for ordinal, line in enumerate(lines):
        try:
            value = parse_canonical_json_bytes(line, require_object=True)
        except DatasetContractError as error:
            raise PublicFrameCatalogError("public frame 不是 canonical JSON") from error
        frames.append(_parse_frame(
            value,
            raw_line_sha256=tuple(public_source_payload_sha256_v1(line)),
            closure=closure,
        ))
    return PublicFrameCatalog(tuple(public_source_payload_sha256_v1(payload)),
                              tuple(sorted(frames, key=PublicFrame.canonical_record)))


def load_public_frame_catalog_from_host_root(
        host_resource_root: object,
        *,
        expected_closure_identity: bytes | None = None,
        ) -> PublicFrameCatalog:
    """为暂未迁移的 host 调用者构造 closure 后立即转入纯 loader。

    唯一物理读取由 payload host adapter 执行；本模块不解析、枚举或读取
    ``host_resource_root``。新的 production 调用者应自行持有 closure，并直接
    调用 ``load_public_frame_catalog_from_closure``。
    """
    from pure_integer_ai.experiments.conversation_public_source_payload_host import (
        load_public_source_payload_closure_from_root,
    )

    return load_public_frame_catalog_from_closure(
        load_public_source_payload_closure_from_root(
            host_resource_root,
            expected_closure_identity=expected_closure_identity,
        ))


def load_public_frame_catalog(
        closure: PublicSourcePayloadClosureV1,
        ) -> PublicFrameCatalog:
    """旧同名入口的 closure-only 兼容别名。

    路径式加载已废止；调用者必须先通过 host adapter 取得完整 closure。保留
    此名仅避免纯 caller 在迁移过程中被迫依赖物理 I/O 边界。
    """
    if type(closure) is not PublicSourcePayloadClosureV1:
        raise PublicFrameCatalogError(
            "路径式 public frame catalog 加载已废止；必须提供 payload closure")
    return load_public_frame_catalog_from_closure(closure)


def materialize_public_frame_candidate(
        frame: PublicFrame,
        request: QuestionRequest,
        ) -> tuple[GenerationCandidate, GenerationPlanningRequest]:
    """从 catalog 的完整 Evidence record 形成单候选 F-00 planning，不读答案标签。"""
    if not isinstance(frame, PublicFrame) or not isinstance(request, QuestionRequest):
        raise TypeError("public frame candidate 需要 frame 和 QuestionRequest")
    if (request.target != frame.question.target
            or request.authorized_candidate_targets
            != frame.question.authorized_candidate_targets
            or request.source != semantic_source(frame.question.target.template)
            or request.response_scope != frame.question.response_scope):
        raise PublicFrameCatalogError("candidate materialization 收到漂移 QuestionRequest")
    evidence = frame.recipe.candidate_evidence
    for item in evidence:
        if item.hypothesis.observation != request.source:
            try:
                aggregate = EvidenceCandidateDefinition.from_stable_key(
                    item.hypothesis.candidate_key)
            except (TypeError, ValueError) as error:
                raise PublicFrameCatalogError("跨来源 Evidence 缺 aggregate 定义") from error
            if aggregate.candidate != request.target.template:
                raise PublicFrameCatalogError("跨来源 Evidence 指向其他 target")
    candidate = GenerationCandidate(
        request.target,
        frame.recipe.candidate_state,
        request.source,
        request.response_scope,
        evidence,
    )
    planning = GenerationPlanningRequest(
        AnswerGenerationGoal(
            request.goal_kind,
            request.target,
            request.required,
            request.source,
            request.response_scope,
            request.target_branch,
        ),
        (candidate,),
    )
    return candidate, planning


__all__ = [
    "PUBLIC_FRAME_CATALOG_SCHEMA_V1",
    "PUBLIC_FRAME_PATTERN_SELECTION_LOWEST_VALID_V1",
    "PUBLIC_FRAME_REFERENCE_SELECTION_LOWEST_COST_V1",
    "PUBLIC_FRAME_CONTEXT_NONE",
    "PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR",
    "PUBLIC_FRAME_RECORD_V2",
    "PUBLIC_FRAME_RECORD_V3",
    "PUBLIC_FRAME_RESPONSE_ACT_RECIPE_RECORD_V2",
    "PUBLIC_FRAME_REFERENCE_RECIPE_RECORD_V3",
    "PublicFrame",
    "PublicFrameCatalog",
    "PublicFrameCatalogError",
    "PublicFrameLexicalRoute",
    "PublicFrameQuestionTemplate",
    "PublicFrameResponseActRuntimeRecipe",
    "PublicFrameReferenceRuntimeRecipe",
    "PublicFrameRuntimeRecipe",
    "PublicFrameSourceRecord",
    "PUBLIC_FRAME_CATALOG_LOGICAL_KEY_V1",
    "load_public_frame_catalog",
    "load_public_frame_catalog_from_closure",
    "load_public_frame_catalog_from_host_root",
    "materialize_public_frame_candidate",
]
