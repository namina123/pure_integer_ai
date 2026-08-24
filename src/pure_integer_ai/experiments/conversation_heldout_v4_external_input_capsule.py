"""DLG-05 v4 外部冻结输入 capsule 的无标签 transport 合同。

本模块只负责外部输入的完整 typed 表达、K 盘不可覆盖三文件闭包及只读重建。它不
生成 candidate、G-01 选择、renderer、owner receipt、label、guard 或 formal 状态。代码内
synthetic fixture 不能借本模块获得来源资格；独立性仍需由输入方和后续物理隔离审计证明。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Any

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_OCCURRENCE,
    OBJECT_SPAN,
    ObjectIdentity,
    SourceRef,
    occurrence_identity,
    span_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import (
    validate_semantic_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BoundProposition,
    BoundRoleBinding,
)
from pure_integer_ai.experiments.conversation_heldout_v4_bundle import (
    ConversationHeldOutV4DependencyBinding,
    ConversationHeldOutV4Representation,
    ConversationHeldOutV4SourceRecord,
    digest_from_hex,
    digest_hex,
)
from pure_integer_ai.experiments.conversation_heldout_v4_candidate_runtime import (
    ConversationHeldOutV4RuntimeEvidencePlan,
    ConversationHeldOutV4RuntimeInput,
    ConversationHeldOutV4RuntimeSourceCapsule,
    V4_RUNTIME_FAMILY_KEY,
    V4_RUNTIME_SOURCE_ORIGIN_EXTERNAL,
)
from pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_structure import (
    ConversationHeldOutV4RuntimeTaskStructureError,
    validate_v4_bound_proposition_source,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.ph2_dataset_core import (
    DatasetContractError,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    decode_integer_tuple,
    encode_integer_tuple,
)


V4_EXTERNAL_INPUT_CAPSULE_KIND = "DLG05_V4_RUNTIME_INPUT_CAPSULE_V1"
V4_EXTERNAL_INPUT_CAPSULE_SCHEMA = "dlg05-v4-runtime-input-capsule-v1"
V4_EXTERNAL_INPUT_MANIFEST_SCHEMA = "dlg05-v4-runtime-input-manifest-v1"
V4_EXTERNAL_PRODUCER_DECLARATION = "DECLARED_EXTERNAL_INPUT_AUTHORITY"

_INPUT_FILE = Path("input_capsule.json")
_INTS_FILE = Path("input.canonical.ints")
_MANIFEST_FILE = Path("manifest.json")
_EXPECTED_FILES = frozenset({_INPUT_FILE, _INTS_FILE, _MANIFEST_FILE})
_REPARSE_POINT = 0x0400
# object-model: exception
class ConversationHeldOutV4ExternalCapsuleError(RuntimeError):
    """外部输入 capsule 的来源、typed 结构、文件闭包或物理边界不完整。"""


def _fail(message: str) -> None:
    """统一产生本模块的 fail-closed 错误。"""
    raise ConversationHeldOutV4ExternalCapsuleError(message)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ExternalCapsuleBudget:
    """限制 C1c 在 external capsule 互操作前可生成的三个 v1 文件。"""

    max_input_payload_bytes: int
    max_integer_payload_bytes: int
    max_manifest_payload_bytes: int
    max_total_payload_bytes: int

    def __post_init__(self) -> None:
        """要求每项与总物理字节预算均为正整数且单项不超过总额。"""
        values = (
            self.max_input_payload_bytes,
            self.max_integer_payload_bytes,
            self.max_manifest_payload_bytes,
            self.max_total_payload_bytes,
        )
        if any(type(item) is not int or item <= 0 for item in values):
            raise ValueError("external capsule budget must use positive strict integers")
        if any(item > self.max_total_payload_bytes for item in values[:3]):
            raise ValueError("external capsule file budget exceeds total budget")

    def integer_tuple(self) -> tuple[int, ...]:
        """返回固定顺序的整数预算，供上层 receipt 绑定。"""
        return (
            self.max_input_payload_bytes,
            self.max_integer_payload_bytes,
            self.max_manifest_payload_bytes,
            self.max_total_payload_bytes,
        )


def _sha256(value: bytes) -> tuple[int, ...]:
    """以整数 tuple 返回输入字节的 SHA-256。"""
    return tuple(hashlib.sha256(value).digest())


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """为可变长度整数段增加显式边界。"""
    result.extend((len(value), *value))


def _strict_int_list(
        value: Any, *, label: str, allow_empty: bool = False,
        ) -> tuple[int, ...]:
    """从 JSON 严格读取整数列表，拒绝 bool、浮点和隐式转换。"""
    if (not isinstance(value, list) or (not allow_empty and not value)
            or any(type(item) is not int for item in value)):
        _fail(f"{label} 必须是{'可空' if allow_empty else '非空'}严格整数列表")
    return tuple(value)


def _strict_key(value: Any, *, label: str) -> tuple[int, ...]:
    """读取协议/trace 键，要求至少一个严格整数。"""
    return _strict_int_list(value, label=label)


def _exact_fields(value: Any, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    """拒绝未知、缺失或非 object 字段，避免标签类内容夹带进入输入。"""
    if not isinstance(value, dict) or set(value) != fields:
        _fail(f"{label} 字段集合漂移")
    return value


def _identity_from_document(value: Any, *, label: str) -> ObjectIdentity:
    """从完整稳定键重建 ObjectIdentity。"""
    try:
        return ObjectIdentity.from_stable_key(_strict_key(value, label=label))
    except (TypeError, ValueError) as exc:
        _fail(f"{label} identity 非法")
        raise AssertionError from exc


def _source_from_document(value: Any, *, label: str) -> SourceRef:
    """从完整稳定键重建 SourceRef。"""
    try:
        return SourceRef.from_stable_key(_strict_key(value, label=label))
    except (TypeError, ValueError) as exc:
        _fail(f"{label} SourceRef 非法")
        raise AssertionError from exc


def _scope_from_document(value: Any, *, label: str) -> ScopeIdentity:
    """从完整稳定键重建递归 ScopeIdentity。"""
    try:
        return ScopeIdentity.from_stable_key(_strict_key(value, label=label))
    except (TypeError, ValueError) as exc:
        _fail(f"{label} scope 非法")
        raise AssertionError from exc


def _protocol_key_from_document(value: Any, *, label: str) -> ProtocolKey:
    """从严格整数列表重建 ProtocolKey。"""
    try:
        return ProtocolKey(_strict_key(value, label=label))
    except (TypeError, ValueError) as exc:
        _fail(f"{label} ProtocolKey 非法")
        raise AssertionError from exc


def _digest_from_document(value: Any, *, label: str) -> tuple[int, ...]:
    """读取小写十六进制 SHA-256，transport 不接受截断或大写变体。"""
    if (not isinstance(value, str) or len(value) != 64
            or value != value.lower()):
        _fail(f"{label} 必须是小写 SHA-256")
    try:
        return digest_from_hex(value, label=label)
    except (TypeError, ValueError) as exc:
        _fail(f"{label} SHA-256 非法")
        raise AssertionError from exc


def _anchor_parts(
        anchor: ObjectIdentity, *, label: str,
        ) -> tuple[SourceRef, str, int, tuple[tuple[int, int], ...]]:
    """严格解析并重建 Occurrence/Span anchor，返回其完整来源与原文范围。"""
    if anchor.object_kind not in {OBJECT_OCCURRENCE, OBJECT_SPAN}:
        _fail(f"{label} 必须是 Occurrence 或 Span")
    try:
        source = SourceRef.from_stable_key(tuple(anchor.components[:11]))
    except (TypeError, ValueError) as exc:
        _fail(f"{label} 来源前缀非法")
        raise AssertionError from exc
    try:
        if anchor.object_kind == OBJECT_OCCURRENCE:
            if len(anchor.components) != 14:
                raise ValueError("Occurrence 长度非法")
            start, end, ordinal = anchor.components[11:]
            rebuilt = occurrence_identity(
                source, start=start, end=end, ordinal=ordinal)
            members = ((start, end),)
            kind = "occurrence"
        else:
            if len(anchor.components) < 15:
                raise ValueError("Span 长度非法")
            ordinal = anchor.components[11]
            count = anchor.components[12]
            values = anchor.components[13:]
            if count <= 0 or len(values) != count * 2:
                raise ValueError("Span members 长度非法")
            members = tuple(
                (values[index], values[index + 1])
                for index in range(0, len(values), 2)
            )
            rebuilt = span_identity(source, members=members, ordinal=ordinal)
            kind = "span"
    except (TypeError, ValueError) as exc:
        _fail(f"{label} 结构非法")
        raise AssertionError from exc
    if rebuilt != anchor:
        _fail(f"{label} identity 与结构不一致")
    return source, kind, ordinal, members


def _anchor_to_document(anchor: ObjectIdentity) -> dict[str, Any]:
    """把来源化 anchor 展开为可重建结构，稳定键只作交叉核验。"""
    source, kind, ordinal, members = _anchor_parts(anchor, label="bound source anchor")
    document: dict[str, Any] = {
        "identity_key": list(anchor.stable_key()),
        "kind": kind,
        "ordinal": ordinal,
        "source_ref_key": list(source.stable_key()),
    }
    if kind == "occurrence":
        document["start"] = members[0][0]
        document["end"] = members[0][1]
    else:
        document["members"] = [list(member) for member in members]
    return document


def _anchor_from_document(value: Any, *, label: str) -> ObjectIdentity:
    """从结构化 anchor 重建身份，不允许只提交 opaque stable key。"""
    if not isinstance(value, dict):
        _fail(f"{label} 必须是 object")
    kind = value.get("kind")
    expected = (
        frozenset({"identity_key", "kind", "ordinal", "source_ref_key", "start", "end"})
        if kind == "occurrence"
        else frozenset({"identity_key", "kind", "ordinal", "source_ref_key", "members"})
        if kind == "span" else None
    )
    if expected is None:
        _fail(f"{label} kind 非法")
    _exact_fields(value, expected, label=label)
    source = _source_from_document(value["source_ref_key"], label=f"{label}.source")
    ordinal = value["ordinal"]
    if type(ordinal) is not int or ordinal < 0:
        _fail(f"{label}.ordinal 非法")
    try:
        if kind == "occurrence":
            start = value["start"]
            end = value["end"]
            if type(start) is not int or type(end) is not int:
                _fail(f"{label} Occurrence 范围必须是严格整数")
            rebuilt = occurrence_identity(
                source, start=start, end=end, ordinal=ordinal)
        else:
            raw_members = value["members"]
            if not isinstance(raw_members, list) or not raw_members:
                _fail(f"{label}.members 必须是非空范围列表")
            members = []
            for index, member in enumerate(raw_members):
                part = _strict_int_list(
                    member, label=f"{label}.members[{index}]")
                if len(part) != 2:
                    _fail(f"{label}.members[{index}] 必须恰有 start/end")
                members.append((part[0], part[1]))
            rebuilt = span_identity(source, members=tuple(members), ordinal=ordinal)
    except (TypeError, ValueError) as exc:
        _fail(f"{label} 结构非法")
        raise AssertionError from exc
    declared = _identity_from_document(value["identity_key"], label=f"{label}.identity")
    if rebuilt != declared:
        _fail(f"{label} identity 与结构不一致")
    return declared


def _bound_to_document(value: BoundProposition) -> dict[str, Any]:
    """完整展开递归 BoundProposition，保留 Role、Binder、Variable 与 anchor 结构。"""
    if not isinstance(value, BoundProposition):
        raise TypeError("bound proposition 类型错误")
    bindings = []
    for binding in value.bindings:
        if isinstance(binding.filler, ObjectIdentity):
            filler: dict[str, Any] = {
                "identity_key": list(binding.filler.stable_key()),
                "kind": "identity",
            }
        else:
            filler = {
                "bound": _bound_to_document(binding.filler),
                "kind": "bound_proposition",
            }
        bindings.append({
            "filler": filler,
            "ordinal": binding.ordinal,
            "role_key": list(binding.role.stable_key()),
        })
    return {
        "applied_variable_keys": [
            list(item.stable_key()) for item in value.applied_variables
        ],
        "bindings": bindings,
        "context_key": list(value.context.stable_key()),
        "instruction_key": list(value.instruction.stable_key()),
        "introduced_binder_keys": [
            list(item.stable_key()) for item in value.introduced_binders
        ],
        "predicate_key": list(value.predicate.stable_key()),
        "source_anchor": _anchor_to_document(value.source_anchor),
        "structure_key": list(value.structure.stable_key()),
        "template_key": list(value.template.stable_key()),
    }


def _bound_from_document(
        value: Any, *, label: str, depth: int = 0,
        active_templates: frozenset[ObjectIdentity] = frozenset(),
        ) -> BoundProposition:
    """递归重建 BoundProposition，并拒绝 stable-key-only、循环深度或字段漂移。"""
    if depth > 64:
        _fail(f"{label} 嵌套深度超过输入合同上限")
    document = _exact_fields(value, frozenset({
        "applied_variable_keys",
        "bindings",
        "context_key",
        "instruction_key",
        "introduced_binder_keys",
        "predicate_key",
        "source_anchor",
        "structure_key",
        "template_key",
    }), label=label)
    raw_binders = document["introduced_binder_keys"]
    raw_variables = document["applied_variable_keys"]
    raw_bindings = document["bindings"]
    if (not isinstance(raw_binders, list) or not isinstance(raw_variables, list)
            or not isinstance(raw_bindings, list)):
        _fail(f"{label} Binder/Variable/bindings 必须是列表")
    template = _identity_from_document(
        document["template_key"], label=f"{label}.template")
    if template in active_templates:
        _fail(f"{label} BoundProposition template 递归成环")
    next_active_templates = active_templates | {template}
    binders = tuple(_identity_from_document(
        item, label=f"{label}.binder") for item in raw_binders)
    if len(set(binders)) != len(binders):
        _fail(f"{label} introduced Binder 不得重复")
    if binders != tuple(sorted(binders, key=lambda item: item.stable_key())):
        _fail(f"{label} introduced Binder 必须按 stable key 排序")
    bindings = []
    for index, raw in enumerate(raw_bindings):
        binding = _exact_fields(raw, frozenset({
            "filler", "ordinal", "role_key",
        }), label=f"{label}.bindings[{index}]")
        ordinal = binding["ordinal"]
        if type(ordinal) is not int or ordinal < 0:
            _fail(f"{label}.bindings[{index}].ordinal 非法")
        filler_document = binding["filler"]
        if not isinstance(filler_document, dict):
            _fail(f"{label}.bindings[{index}].filler 必须是 object")
        filler_kind = filler_document.get("kind")
        if filler_kind == "identity":
            _exact_fields(filler_document, frozenset({
                "identity_key", "kind",
            }), label=f"{label}.bindings[{index}].filler")
            filler: ObjectIdentity | BoundProposition = _identity_from_document(
                filler_document["identity_key"],
                label=f"{label}.bindings[{index}].filler.identity",
            )
        elif filler_kind == "bound_proposition":
            _exact_fields(filler_document, frozenset({
                "bound", "kind",
            }), label=f"{label}.bindings[{index}].filler")
            filler = _bound_from_document(
                filler_document["bound"],
                label=f"{label}.bindings[{index}].filler.bound",
                depth=depth + 1,
                active_templates=next_active_templates,
            )
        else:
            _fail(f"{label}.bindings[{index}].filler kind 非法")
        try:
            role = _identity_from_document(
                binding["role_key"], label=f"{label}.bindings[{index}].role")
            validate_semantic_identity(role)
            bindings.append(BoundRoleBinding(
                role,
                filler,
                ordinal,
            ))
        except (TypeError, ValueError) as exc:
            _fail(f"{label}.bindings[{index}] typed 字段非法")
            raise AssertionError from exc
    try:
        result = BoundProposition(
            template,
            _identity_from_document(document["instruction_key"], label=f"{label}.instruction"),
            _identity_from_document(document["predicate_key"], label=f"{label}.predicate"),
            _identity_from_document(document["structure_key"], label=f"{label}.structure"),
            _anchor_from_document(document["source_anchor"], label=f"{label}.source_anchor"),
            _identity_from_document(document["context_key"], label=f"{label}.context"),
            binders,
            tuple(bindings),
            tuple(_identity_from_document(
                item, label=f"{label}.variable") for item in raw_variables),
        )
    except (TypeError, ValueError) as exc:
        _fail(f"{label} typed schema 非法")
        raise AssertionError from exc
    if _bound_to_document(result) != document:
        _fail(f"{label} 不是规范 BoundProposition 表达")
    return result


def _validate_bound_source(
        root: BoundProposition,
        record: ConversationHeldOutV4SourceRecord,
        *, label: str,
        ) -> None:
    """经共享无载荷内核回放 bound 树，保持本 capsule 的错误边界。"""
    try:
        validate_v4_bound_proposition_source(
            root, source=record.source, scalar_start=0,
            scalar_end=len(record.raw_text_scalars), label=label)
    except ConversationHeldOutV4RuntimeTaskStructureError as exc:
        _fail(str(exc))


def _representation_to_document(
        value: ConversationHeldOutV4Representation,
        ) -> dict[str, Any]:
    """导出完整输入 Representation，不把文本投影当权威字段。"""
    return {
        "ordinal": value.ordinal,
        "representation_key": list(value.representation.stable_key()),
        "scalars": list(value.scalars),
    }


def _representation_from_document(
        value: Any, *, label: str,
        ) -> ConversationHeldOutV4Representation:
    """恢复并核验 Representation 到原序 scalar 的可逆映射。"""
    document = _exact_fields(value, frozenset({
        "ordinal", "representation_key", "scalars",
    }), label=label)
    ordinal = document["ordinal"]
    if type(ordinal) is not int or ordinal < 0:
        _fail(f"{label}.ordinal 非法")
    try:
        result = ConversationHeldOutV4Representation(
            _identity_from_document(
                document["representation_key"], label=f"{label}.representation"),
            ordinal,
            _strict_int_list(document["scalars"], label=f"{label}.scalars"),
        )
    except (TypeError, ValueError) as exc:
        _fail(f"{label} Representation 非法")
        raise AssertionError from exc
    return result


def _source_record_to_document(
        value: ConversationHeldOutV4SourceRecord,
        ) -> dict[str, Any]:
    """导出完整原文、许可、归属和 URI，不能只留 SourceRef 摘要。"""
    return {
        "attribution_scalars": list(value.attribution_scalars),
        "batch_id": value.batch_id,
        "companion_assoc_id": value.companion_assoc_id,
        "companion_name_hash": value.companion_name_hash,
        "companion_type_hash": value.companion_type_hash,
        "content_sha256": digest_hex(value.content_sha256),
        "license_scalars": list(value.license_scalars),
        "raw_text_scalars": list(value.raw_text_scalars),
        "source_ref_key": list(value.source.stable_key()),
        "source_uri_scalars": list(value.source_uri_scalars),
    }


def _source_record_from_document(
        value: Any, *, label: str,
        ) -> ConversationHeldOutV4SourceRecord:
    """从无标签来源文档恢复 SourceRecord，并重新验证原文 hash。"""
    document = _exact_fields(value, frozenset({
        "attribution_scalars",
        "batch_id",
        "companion_assoc_id",
        "companion_name_hash",
        "companion_type_hash",
        "content_sha256",
        "license_scalars",
        "raw_text_scalars",
        "source_ref_key",
        "source_uri_scalars",
    }), label=label)
    metadata = tuple(document[name] for name in (
        "batch_id",
        "companion_type_hash",
        "companion_name_hash",
        "companion_assoc_id",
    ))
    if any(type(item) is not int or item < 0 for item in metadata):
        _fail(f"{label} source metadata 非法")
    try:
        result = ConversationHeldOutV4SourceRecord(
            _source_from_document(document["source_ref_key"], label=f"{label}.source"),
            _strict_int_list(document["raw_text_scalars"], label=f"{label}.raw", allow_empty=True),
            _digest_from_document(document["content_sha256"], label=f"{label}.content_sha256"),
            _strict_int_list(document["license_scalars"], label=f"{label}.license"),
            _strict_int_list(document["attribution_scalars"], label=f"{label}.attribution"),
            _strict_int_list(document["source_uri_scalars"], label=f"{label}.uri", allow_empty=True),
            *metadata,
        )
    except (TypeError, ValueError) as exc:
        _fail(f"{label} SourceRecord 非法")
        raise AssertionError from exc
    return result


def _request_to_document(value: QuestionRequest) -> dict[str, Any]:
    """导出完整 typed request；不携带 candidate、selection 或 expected 字段。"""
    return {
        "authorized_candidate_targets": [
            _bound_to_document(item) for item in value.authorized_candidate_targets
        ],
        "evidence_scope_key": list(value.evidence_scope.stable_key()),
        "goal_kind_key": list(value.goal_kind.stable_key()),
        "intent_key": list(value.intent.stable_key()),
        "query_kind_key": list(value.query_kind.stable_key()),
        "required": {
            "refute": value.required.refute,
            "support": value.required.support,
        },
        "response_scope_key": list(value.response_scope.stable_key()),
        "target": _bound_to_document(value.target),
        "target_branch_key": (
            None if value.target_branch is None
            else list(value.target_branch.stable_key())
        ),
        "trace": list(value.trace),
    }


def _request_from_document(value: Any, *, label: str) -> QuestionRequest:
    """恢复 QuestionRequest，并在构造层重跑 source/scope/授权目标不变量。"""
    document = _exact_fields(value, frozenset({
        "authorized_candidate_targets",
        "evidence_scope_key",
        "goal_kind_key",
        "intent_key",
        "query_kind_key",
        "required",
        "response_scope_key",
        "target",
        "target_branch_key",
        "trace",
    }), label=label)
    required = _exact_fields(
        document["required"], frozenset({"refute", "support"}),
        label=f"{label}.required")
    if type(required["support"]) is not bool or type(required["refute"]) is not bool:
        _fail(f"{label}.required 必须是严格 bool")
    raw_targets = document["authorized_candidate_targets"]
    if not isinstance(raw_targets, list):
        _fail(f"{label}.authorized_candidate_targets 必须是列表")
    target_branch_raw = document["target_branch_key"]
    if target_branch_raw is not None and not isinstance(target_branch_raw, list):
        _fail(f"{label}.target_branch_key 必须是整数列表或 null")
    try:
        result = QuestionRequest(
            _identity_from_document(document["query_kind_key"], label=f"{label}.query_kind"),
            _identity_from_document(document["intent_key"], label=f"{label}.intent"),
            _identity_from_document(document["goal_kind_key"], label=f"{label}.goal_kind"),
            _bound_from_document(document["target"], label=f"{label}.target"),
            LogicEvidenceState(required["support"], required["refute"]),
            _scope_from_document(document["evidence_scope_key"], label=f"{label}.evidence_scope"),
            _scope_from_document(document["response_scope_key"], label=f"{label}.response_scope"),
            _strict_key(document["trace"], label=f"{label}.trace"),
            (None if target_branch_raw is None else _identity_from_document(
                target_branch_raw, label=f"{label}.target_branch")),
            tuple(_bound_from_document(
                item, label=f"{label}.authorized_candidate_targets[{index}]")
                  for index, item in enumerate(raw_targets)),
        )
    except (TypeError, ValueError) as exc:
        _fail(f"{label} QuestionRequest 非法")
        raise AssertionError from exc
    if _request_to_document(result) != document:
        _fail(f"{label} 不是规范 QuestionRequest 表达")
    return result


def _evidence_plan_to_document(
        value: ConversationHeldOutV4RuntimeEvidencePlan,
        ) -> dict[str, Any]:
    """导出 runtime 真正消费的 Evidence plan 与其来源 scalar span。"""
    return {
        "competition_key": list(value.competition_key),
        "source_ref_key": list(value.source.stable_key()),
        "source_span": {
            "end": value.source_span_end,
            "start": value.source_span_start,
        },
        "stances": list(value.stances),
        "target": _bound_to_document(value.target),
    }


def _evidence_plan_from_document(
        value: Any, *, label: str,
        ) -> ConversationHeldOutV4RuntimeEvidencePlan:
    """恢复 Evidence plan；其 stance 和 span 会进入真实 H-00 ledger。"""
    document = _exact_fields(value, frozenset({
        "competition_key", "source_ref_key", "source_span", "stances", "target",
    }), label=label)
    span = _exact_fields(
        document["source_span"], frozenset({"end", "start"}),
        label=f"{label}.source_span")
    start = span["start"]
    end = span["end"]
    if type(start) is not int or type(end) is not int:
        _fail(f"{label}.source_span 必须是严格整数")
    try:
        result = ConversationHeldOutV4RuntimeEvidencePlan(
            _bound_from_document(document["target"], label=f"{label}.target"),
            _strict_key(document["competition_key"], label=f"{label}.competition"),
            _strict_int_list(document["stances"], label=f"{label}.stances"),
            _source_from_document(document["source_ref_key"], label=f"{label}.source"),
            start,
            end,
        )
    except (TypeError, ValueError) as exc:
        _fail(f"{label} Evidence plan 非法")
        raise AssertionError from exc
    return result


def _input_to_document(value: ConversationHeldOutV4RuntimeInput) -> dict[str, Any]:
    """导出一个 turn 的完整 typed 输入，source 只按 key 引用全局 source table。"""
    return {
        "case_key": list(value.case_key.components),
        "evidence_plans": [
            _evidence_plan_to_document(item) for item in value.evidence_plans
        ],
        "ordinal": value.ordinal,
        "representations": [
            _representation_to_document(item) for item in value.representations
        ],
        "request": _request_to_document(value.request),
        "source_ref_keys": [
            list(item.source.stable_key()) for item in value.source_records
        ],
        "turn_key": list(value.turn_key.components),
    }


def _input_from_document(
        value: Any,
        *,
        label: str,
        source_records: dict[SourceRef, ConversationHeldOutV4SourceRecord],
        ) -> ConversationHeldOutV4RuntimeInput:
    """从 turn 文档和全局 source table 重建 runtime 唯一可消费的输入对象。"""
    document = _exact_fields(value, frozenset({
        "case_key",
        "evidence_plans",
        "ordinal",
        "representations",
        "request",
        "source_ref_keys",
        "turn_key",
    }), label=label)
    ordinal = document["ordinal"]
    if type(ordinal) is not int or ordinal <= 0:
        _fail(f"{label}.ordinal 必须为正严格整数")
    raw_sources = document["source_ref_keys"]
    raw_representations = document["representations"]
    raw_plans = document["evidence_plans"]
    if (not isinstance(raw_sources, list) or not raw_sources
            or not isinstance(raw_representations, list)
            or not isinstance(raw_plans, list)):
        _fail(f"{label} sources/representations/evidence_plans 类型错误")
    references = tuple(_source_from_document(
        item, label=f"{label}.source_ref_keys[{index}]")
        for index, item in enumerate(raw_sources))
    if len(set(references)) != len(references):
        _fail(f"{label}.source_ref_keys 不得重复")
    try:
        records = tuple(source_records[item] for item in references)
    except KeyError as exc:
        _fail(f"{label} 引用了未登记 SourceRecord")
        raise AssertionError from exc
    request = _request_from_document(document["request"], label=f"{label}.request")
    representations = tuple(_representation_from_document(
        item, label=f"{label}.representations[{index}]")
        for index, item in enumerate(raw_representations))
    plans = tuple(_evidence_plan_from_document(
        item, label=f"{label}.evidence_plans[{index}]")
        for index, item in enumerate(raw_plans))
    try:
        result = ConversationHeldOutV4RuntimeInput(
            _protocol_key_from_document(document["case_key"], label=f"{label}.case_key"),
            _protocol_key_from_document(document["turn_key"], label=f"{label}.turn_key"),
            ordinal,
            request,
            representations,
            records,
            plans,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        _fail(f"{label} RuntimeInput 非法")
        raise AssertionError from exc
    if _input_to_document(result) != document:
        _fail(f"{label} 不是规范 RuntimeInput 表达")
    return result


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ExternalProducer:
    """外部输入方的声明性整数身份，不替代后续独立性物理审计。"""

    producer_key: ProtocolKey
    declaration: str = V4_EXTERNAL_PRODUCER_DECLARATION

    def __post_init__(self) -> None:
        """只接受指定的外部输入声明和完整 ProtocolKey。"""
        if not isinstance(self.producer_key, ProtocolKey):
            raise TypeError("external producer_key 类型错误")
        if self.declaration != V4_EXTERNAL_PRODUCER_DECLARATION:
            _fail("external producer declaration 未注册")

    def stable_key(self) -> tuple[int, ...]:
        """返回声明与 producer key 的完整整数身份。"""
        result = []
        _pack(result, self.producer_key.components)
        _pack(result, tuple(ord(item) for item in self.declaration))
        return tuple(result)

    def document(self) -> dict[str, Any]:
        """生成规范 producer 声明，不写入本机用户或路径。"""
        return {
            "declaration": self.declaration,
            "producer_key": list(self.producer_key.components),
        }

    @classmethod
    def from_document(cls, value: Any) -> "ConversationHeldOutV4ExternalProducer":
        """从严格字段集合恢复 producer 声明。"""
        document = _exact_fields(value, frozenset({
            "declaration", "producer_key",
        }), label="external producer")
        try:
            result = cls(
                _protocol_key_from_document(
                    document["producer_key"], label="external producer key"),
                document["declaration"],
            )
        except (TypeError, ValueError) as exc:
            _fail("external producer 非法")
            raise AssertionError from exc
        if result.document() != document:
            _fail("external producer 不是规范表达")
        return result


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ExternalInputCapsule:
    """未落盘的外部 typed 输入草案；只可发布到此前不存在的 K 盘根。"""

    family_key: ProtocolKey
    producer: ConversationHeldOutV4ExternalProducer
    dependencies: ConversationHeldOutV4DependencyBinding
    inputs: tuple[ConversationHeldOutV4RuntimeInput, ...]

    def __post_init__(self) -> None:
        """闭合 source table、来源范围、输入序和现阶段单来源 H-00 边界。"""
        if self.family_key != V4_RUNTIME_FAMILY_KEY:
            _fail("external capsule family_key 与 v4 runtime 不一致")
        if not isinstance(self.producer, ConversationHeldOutV4ExternalProducer):
            raise TypeError("external capsule producer 类型错误")
        if not isinstance(self.dependencies, ConversationHeldOutV4DependencyBinding):
            raise TypeError("external capsule dependencies 类型错误")
        if (not isinstance(self.inputs, tuple) or not self.inputs
                or any(not isinstance(item, ConversationHeldOutV4RuntimeInput)
                       for item in self.inputs)):
            raise TypeError("external capsule inputs 必须是非空 RuntimeInput tuple")
        if tuple(item.ordinal for item in self.inputs) != tuple(
                range(1, len(self.inputs) + 1)):
            _fail("external capsule input ordinal 必须连续")
        identities = tuple((item.case_key, item.turn_key) for item in self.inputs)
        if len(set(identities)) != len(identities):
            _fail("external capsule case/turn 不得重复")
        records: dict[SourceRef, ConversationHeldOutV4SourceRecord] = {}
        referenced: set[SourceRef] = set()
        for item in self.inputs:
            ordered_records = tuple(sorted(
                item.source_records, key=lambda record: record.source.stable_key()))
            if item.source_records != ordered_records:
                _fail("external capsule SourceRecord 必须按 SourceRef 排序")
            ordered_plans = tuple(sorted(
                item.evidence_plans, key=lambda plan: plan.target.stable_key()))
            if item.evidence_plans != ordered_plans:
                _fail("external capsule Evidence plan 必须按 target 排序")
            local_records = {record.source: record for record in item.source_records}
            if set(local_records) != {item.request.source, *(
                    plan.source for plan in item.evidence_plans)}:
                _fail("external capsule SourceRecord 存在未消费或缺失来源")
            for record in item.source_records:
                if not record.raw_text_scalars or not record.source_uri_scalars:
                    _fail("external capsule SourceRecord 必须含原文和 source URI")
                prior = records.get(record.source)
                if prior is not None and prior != record:
                    _fail("external capsule 跨 turn 同一 SourceRef 内容漂移")
                records[record.source] = record
                referenced.add(record.source)
            request_record = local_records[item.request.source]
            _validate_bound_source(
                item.request.target, request_record, label="external request target")
            for target in item.request.authorized_candidate_targets:
                _validate_bound_source(
                    target, request_record, label="external authorized target")
            for plan in item.evidence_plans:
                _validate_bound_source(
                    plan.target, request_record, label="external Evidence target")
        if set(records) != referenced:
            _fail("external capsule source table 未闭合")

    def source_records(self) -> tuple[ConversationHeldOutV4SourceRecord, ...]:
        """按 SourceRef 返回跨 turn 去重且不可漂移的完整 source table。"""
        table = {
            record.source: record
            for item in self.inputs for record in item.source_records
        }
        return tuple(sorted(table.values(), key=lambda item: item.source.stable_key()))

    def stable_key(self) -> tuple[int, ...]:
        """返回完整 typed input 身份；摘要永不替代 inputs 或原文。"""
        result = [1]
        for value in (
                self.family_key.components,
                self.producer.stable_key(),
                self.dependencies.stable_key()):
            _pack(result, value)
        result.append(len(self.inputs))
        for item in self.inputs:
            _pack(result, item.stable_key())
        return tuple(result)

    def document(self) -> dict[str, Any]:
        """生成完整无标签 JSON transport document。"""
        return {
            "artifact_kind": V4_EXTERNAL_INPUT_CAPSULE_KIND,
            "family_key": list(self.family_key.components),
            "format_version": 1,
            "producer": self.producer.document(),
            "provenance": {
                "artifact_sha256": digest_hex(self.dependencies.artifact_sha256),
                "document_sha256": digest_hex(self.dependencies.document_sha256),
                "inventory_sha256": digest_hex(self.dependencies.inventory_sha256),
            },
            "schema": V4_EXTERNAL_INPUT_CAPSULE_SCHEMA,
            "sources": [
                _source_record_to_document(item) for item in self.source_records()
            ],
            "turns": [_input_to_document(item) for item in self.inputs],
        }

    @classmethod
    def from_document(
            cls, value: Any,
            ) -> "ConversationHeldOutV4ExternalInputCapsule":
        """从完整 JSON transport 重建 draft，并要求重新导出逐字段一致。"""
        document = _exact_fields(value, frozenset({
            "artifact_kind",
            "family_key",
            "format_version",
            "producer",
            "provenance",
            "schema",
            "sources",
            "turns",
        }), label="external input capsule")
        if (type(document["format_version"]) is not int
                or document["artifact_kind"] != V4_EXTERNAL_INPUT_CAPSULE_KIND
                or document["schema"] != V4_EXTERNAL_INPUT_CAPSULE_SCHEMA
                or document["format_version"] != 1):
            _fail("external input capsule kind/schema/version 非法")
        provenance = _exact_fields(document["provenance"], frozenset({
            "artifact_sha256", "document_sha256", "inventory_sha256",
        }), label="external input capsule provenance")
        try:
            dependencies = ConversationHeldOutV4DependencyBinding(
                _digest_from_document(
                    provenance["artifact_sha256"], label="external provenance artifact"),
                _digest_from_document(
                    provenance["inventory_sha256"], label="external provenance inventory"),
                _digest_from_document(
                    provenance["document_sha256"], label="external provenance document"),
            )
        except (TypeError, ValueError) as exc:
            _fail("external input capsule provenance 非法")
            raise AssertionError from exc
        raw_sources = document["sources"]
        raw_turns = document["turns"]
        if (not isinstance(raw_sources, list) or not raw_sources
                or not isinstance(raw_turns, list) or not raw_turns):
            _fail("external input capsule sources/turns 必须为非空列表")
        sources = tuple(_source_record_from_document(
            item, label=f"external sources[{index}]")
            for index, item in enumerate(raw_sources))
        if tuple(sorted(sources, key=lambda item: item.source.stable_key())) != sources:
            _fail("external input capsule sources 必须按 SourceRef 排序")
        source_map = {item.source: item for item in sources}
        if len(source_map) != len(sources):
            _fail("external input capsule sources 不得重复")
        inputs = tuple(_input_from_document(
            item,
            label=f"external turns[{index}]",
            source_records=source_map,
        ) for index, item in enumerate(raw_turns))
        try:
            result = cls(
                _protocol_key_from_document(
                    document["family_key"], label="external capsule family_key"),
                ConversationHeldOutV4ExternalProducer.from_document(
                    document["producer"]),
                dependencies,
                inputs,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            _fail("external input capsule typed contract 非法")
            raise AssertionError from exc
        if result.document() != document:
            _fail("external input capsule 不是规范表达")
        return result


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4PreparedExternalInputCapsule:
    """已在内存完成预算与规范闭合、尚未写入任何 root 的 v1 capsule。"""

    capsule: ConversationHeldOutV4ExternalInputCapsule
    budget: ConversationHeldOutV4ExternalCapsuleBudget
    input_payload: bytes
    integer_payload: bytes
    manifest_payload: bytes

    def __post_init__(self) -> None:
        """重算三份 payload 与预算，阻止伪造 prepared object 绕过 prepare。"""
        if (not isinstance(self.capsule, ConversationHeldOutV4ExternalInputCapsule)
                or not isinstance(self.budget, ConversationHeldOutV4ExternalCapsuleBudget)
                or any(not isinstance(item, bytes) or not item for item in (
                    self.input_payload, self.integer_payload, self.manifest_payload))):
            raise TypeError("prepared external capsule types are invalid")
        expected_input = canonical_json_bytes(self.capsule.document())
        expected_integer = encode_integer_tuple(self.capsule.stable_key())
        expected_manifest = canonical_json_bytes(_manifest_document(
            self.capsule,
            input_payload=expected_input,
            ints_payload=expected_integer))
        if (self.input_payload != expected_input
                or self.integer_payload != expected_integer
                or self.manifest_payload != expected_manifest):
            _fail("prepared external capsule payload is not canonical")
        sizes = (
            len(self.input_payload),
            len(self.integer_payload),
            len(self.manifest_payload),
        )
        limits = (
            self.budget.max_input_payload_bytes,
            self.budget.max_integer_payload_bytes,
            self.budget.max_manifest_payload_bytes,
        )
        if any(size > limit for size, limit in zip(sizes, limits)):
            _fail("prepared external capsule file exceeds budget")
        if sum(sizes) > self.budget.max_total_payload_bytes:
            _fail("prepared external capsule total exceeds budget")


def prepare_v4_external_input_capsule(
        capsule: ConversationHeldOutV4ExternalInputCapsule,
        budget: ConversationHeldOutV4ExternalCapsuleBudget,
        ) -> ConversationHeldOutV4PreparedExternalInputCapsule:
    """在任何文件创建前编码并验证 v1 capsule 的三份受预算 payload。"""
    if not isinstance(capsule, ConversationHeldOutV4ExternalInputCapsule):
        raise TypeError("external input capsule type is invalid")
    if not isinstance(budget, ConversationHeldOutV4ExternalCapsuleBudget):
        raise TypeError("external capsule budget type is invalid")
    input_payload = canonical_json_bytes(capsule.document())
    integer_payload = encode_integer_tuple(capsule.stable_key())
    manifest_payload = canonical_json_bytes(_manifest_document(
        capsule, input_payload=input_payload, ints_payload=integer_payload))
    return ConversationHeldOutV4PreparedExternalInputCapsule(
        capsule, budget, input_payload, integer_payload, manifest_payload)


def _manifest_document(
        capsule: ConversationHeldOutV4ExternalInputCapsule,
        *,
        input_payload: bytes,
        ints_payload: bytes,
        ) -> dict[str, Any]:
    """生成 manifest-last 文件闭包；它不含自身 hash，避免循环依赖。"""
    return {
        "artifact_kind": V4_EXTERNAL_INPUT_CAPSULE_KIND,
        "files": {
            _INPUT_FILE.name: {
                "sha256": hashlib.sha256(input_payload).hexdigest(),
                "size": len(input_payload),
            },
            _INTS_FILE.name: {
                "sha256": hashlib.sha256(ints_payload).hexdigest(),
                "size": len(ints_payload),
            },
        },
        "format_version": 1,
        "input_integer_count": len(capsule.stable_key()),
        "input_stable_key_sha256": hashlib.sha256(
            ints_payload).hexdigest(),
        "provenance": {
            "artifact_sha256": digest_hex(capsule.dependencies.artifact_sha256),
            "document_sha256": digest_hex(capsule.dependencies.document_sha256),
            "inventory_sha256": digest_hex(capsule.dependencies.inventory_sha256),
        },
        "schema": V4_EXTERNAL_INPUT_MANIFEST_SCHEMA,
    }


def _is_reparse(path: Path) -> bool:
    """在 resolve 前检查 Windows reparse attribute，普通平台保持 0。"""
    try:
        stat = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return bool(getattr(stat, "st_file_attributes", 0) & _REPARSE_POINT)


def _require_existing_normal_directory(path: Path, *, label: str) -> Path:
    """逐级拒绝 link/reparse 后才 resolve，防止路径边界被重解析点掩盖。"""
    if not path.is_absolute():
        _fail(f"{label} 必须是绝对路径")
    chain = (path, *path.parents)
    for current in chain:
        if current.is_symlink() or _is_reparse(current):
            _fail(f"{label} 含链接或 reparse point")
        if not current.exists():
            _fail(f"{label} 不存在")
        if current == current.parent:
            break
    resolved = path.resolve()
    if (not resolved.is_dir() or resolved.is_symlink()
            or _is_reparse(resolved)):
        _fail(f"{label} 不是普通目录")
    return resolved


def _new_root(root: str | Path, *, require_k_drive: bool) -> Path:
    """只接受此前不存在、父目录已验证且生产位于 K 盘的 capsule 根。"""
    raw = Path(root)
    if not raw.is_absolute() or ".." in raw.parts:
        _fail("external capsule root 必须是不含 .. 的绝对路径")
    if os.path.lexists(raw) or raw.is_symlink() or _is_reparse(raw):
        _fail("external capsule root 必须此前不存在且不是链接")
    parent = _require_existing_normal_directory(raw.parent, label="external capsule parent")
    target = parent / raw.name
    if require_k_drive and target.drive.upper() != "K:":
        _fail("external capsule 生产根必须位于 K 盘")
    if target.exists() or os.path.lexists(target):
        _fail("external capsule root 已存在")
    return target


def _read_root(root: str | Path, *, require_k_drive: bool) -> Path:
    """在读任何 transport 文件前验证 root 本身及父级没有链接或 reparse。"""
    raw = Path(root)
    target = _require_existing_normal_directory(raw, label="external capsule root")
    if require_k_drive and target.drive.upper() != "K:":
        _fail("external capsule 生产根必须位于 K 盘")
    return target


def _plain_file(root: Path, relative: Path) -> Path:
    """返回 root 内唯一普通单硬链接文件，拒绝路径逃逸和特殊文件。"""
    path = root / relative
    if path.parent != root:
        _fail("external capsule 文件路径不允许子目录")
    if not path.is_file() or path.is_symlink() or _is_reparse(path):
        _fail(f"external capsule 文件不是普通文件: {relative}")
    try:
        stat = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        _fail(f"external capsule 文件不可读取: {relative}")
        raise AssertionError from exc
    if getattr(stat, "st_nlink", 1) != 1:
        _fail(f"external capsule 文件硬链接数必须为 1: {relative}")
    try:
        resolved = path.resolve()
    except OSError as exc:
        _fail(f"external capsule 文件无法解析: {relative}")
        raise AssertionError from exc
    if not resolved.is_relative_to(root):
        _fail(f"external capsule 文件路径越界: {relative}")
    return path


def _file_set(root: Path) -> frozenset[Path]:
    """要求 transport root 恰有三个普通文件，不允许额外缓存或投影。"""
    found = set()
    try:
        children = tuple(root.iterdir())
    except OSError as exc:
        _fail("external capsule root 不可枚举")
        raise AssertionError from exc
    for child in children:
        if child.is_dir() or child.is_symlink() or _is_reparse(child):
            _fail("external capsule root 不允许目录、链接或 reparse")
        if not child.is_file():
            _fail("external capsule root 含非普通文件")
        _plain_file(root, child.relative_to(root))
        found.add(child.relative_to(root))
    if frozenset(found) != _EXPECTED_FILES:
        _fail("external capsule 文件闭包必须精确为 input/json/ints/manifest")
    return frozenset(found)


def _read_plain_bytes(root: Path, relative: Path) -> bytes:
    """在读取前后均复核普通文件门，避免读取期间被替换为链接。"""
    path = _plain_file(root, relative)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        _fail(f"external capsule 文件读取失败: {relative}")
        raise AssertionError from exc
    _plain_file(root, relative)
    if not payload:
        _fail(f"external capsule 文件不能为空: {relative}")
    return payload


def _file_identity_from_stat(result: os.stat_result) -> tuple[int, ...]:
    """抽取跨 Windows 路径/句柄视图稳定的身份，供有界读取前后检测漂移。"""
    return (
        result.st_dev,
        result.st_ino,
        result.st_mode,
        result.st_nlink,
        result.st_size,
        result.st_mtime_ns,
    )


def _capture_plain_file_identity(
        root: Path, relative: Path,
        ) -> tuple[Path, tuple[int, ...]]:
    """在普通文件门之后记录可比较身份，不把读取动作混入身份采集。"""
    path = _plain_file(root, relative)
    try:
        result = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        _fail(f"external capsule 文件身份不可读取: {relative}")
        raise AssertionError from exc
    if not stat.S_ISREG(result.st_mode) or result.st_size <= 0:
        _fail(f"external capsule 文件不是非空普通文件: {relative}")
    return path, _file_identity_from_stat(result)


def _require_plain_file_identity(
        root: Path,
        relative: Path,
        expected: tuple[int, ...],
        ) -> tuple[Path, tuple[int, ...]]:
    """要求路径当前仍指向初始发现的同一普通文件身份。"""
    path, actual = _capture_plain_file_identity(root, relative)
    if actual != expected:
        _fail(f"external capsule 文件身份漂移: {relative}")
    return path, actual


def _read_bounded_plain_bytes(
        root: Path,
        relative: Path,
        *,
        maximum_bytes: int,
        expected_identity: tuple[int, ...],
        expected_size: int | None = None,
        ) -> bytes:
    """按已捕获身份读取精确字节数，整个路径不调用无上限 read_bytes。"""
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        _fail("external capsule bounded read budget 非法")
    if expected_size is not None and (
            type(expected_size) is not int or expected_size <= 0):
        _fail("external capsule declared file size 非法")
    path, identity = _require_plain_file_identity(
        root, relative, expected_identity)
    size = identity[4]
    if size > maximum_bytes:
        _fail(f"external capsule 文件超过读取预算: {relative}")
    if expected_size is not None and size != expected_size:
        _fail(f"external capsule manifest 文件大小漂移: {relative}")
    try:
        with path.open("rb") as handle:
            opened_identity = _file_identity_from_stat(os.fstat(handle.fileno()))
            if opened_identity != expected_identity:
                _fail(f"external capsule 文件身份漂移: {relative}")
            payload = handle.read(size)
    except OSError as exc:
        _fail(f"external capsule 文件读取失败: {relative}")
        raise AssertionError from exc
    if len(payload) != size or not payload:
        _fail(f"external capsule 文件读取长度漂移: {relative}")
    _require_plain_file_identity(root, relative, expected_identity)
    return payload


def _write_exclusive(path: Path, payload: bytes) -> None:
    """使用 xb 落盘一个非空不可覆盖文件，并立即复核普通文件边界。"""
    if not isinstance(payload, bytes) or not payload:
        _fail("external capsule 写入 payload 不能为空")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except OSError as exc:
        _fail(f"external capsule 不可覆盖写入失败: {path.name}")
        raise AssertionError from exc
    relative = path.relative_to(path.parent)
    _, identity = _capture_plain_file_identity(path.parent, relative)
    if identity[4] != len(payload):
        _fail(f"external capsule 写入后长度漂移: {path.name}")
    if _read_bounded_plain_bytes(
            path.parent,
            relative,
            maximum_bytes=len(payload),
            expected_identity=identity,
            expected_size=len(payload),
    ) != payload:
        _fail(f"external capsule 写入后内容漂移: {path.name}")


def _write_capsule_payloads(
        target: Path,
        *, input_payload: bytes, ints_payload: bytes, manifest_payload: bytes,
        ) -> None:
    """在已核验的新根按 manifest-last 顺序写入 v1 三文件闭包。"""
    try:
        target.mkdir()
    except OSError as exc:
        _fail("external capsule root 创建失败")
        raise AssertionError from exc
    if not target.is_dir() or target.is_symlink() or _is_reparse(target):
        _fail("external capsule root 创建后不是普通目录")
    _write_exclusive(target / _INPUT_FILE, input_payload)
    _write_exclusive(target / _INTS_FILE, ints_payload)
    _write_exclusive(target / _MANIFEST_FILE, manifest_payload)
    _file_set(target)


def write_prepared_v4_external_input_capsule(
        root: str | Path,
        prepared: ConversationHeldOutV4PreparedExternalInputCapsule,
        *, require_k_drive: bool = True,
        ) -> Path:
    """仅把已通过 prepare 的受预算 v1 capsule 排他发布到新 root。"""
    if not isinstance(prepared, ConversationHeldOutV4PreparedExternalInputCapsule):
        raise TypeError("prepared external capsule type is invalid")
    target = _new_root(root, require_k_drive=require_k_drive)
    _write_capsule_payloads(
        target,
        input_payload=prepared.input_payload,
        ints_payload=prepared.integer_payload,
        manifest_payload=prepared.manifest_payload)
    return target


def write_v4_external_input_capsule(
        root: str | Path,
        capsule: ConversationHeldOutV4ExternalInputCapsule,
        *,
        require_k_drive: bool = True,
        ) -> Path:
    """在新 K 盘 root 以 input、ints、manifest 的顺序发布外部无标签输入闭包。"""
    if not isinstance(capsule, ConversationHeldOutV4ExternalInputCapsule):
        raise TypeError("external input capsule 类型错误")
    target = _new_root(root, require_k_drive=require_k_drive)
    input_payload = canonical_json_bytes(capsule.document())
    ints_payload = encode_integer_tuple(capsule.stable_key())
    manifest_payload = canonical_json_bytes(_manifest_document(
        capsule, input_payload=input_payload, ints_payload=ints_payload))
    _write_capsule_payloads(
        target,
        input_payload=input_payload,
        ints_payload=ints_payload,
        manifest_payload=manifest_payload)
    return target


def _parse_external_input_manifest(manifest_payload: bytes) -> dict[str, Any]:
    """解析 v1 manifest 的无载荷头，供 legacy 与先 manifest 的 bounded reader 共用。"""
    try:
        manifest = parse_canonical_json_bytes(manifest_payload, require_object=True)
    except DatasetContractError as exc:
        _fail("external capsule manifest 不是规范 JSON")
        raise AssertionError from exc
    manifest = _exact_fields(manifest, frozenset({
        "artifact_kind",
        "files",
        "format_version",
        "input_integer_count",
        "input_stable_key_sha256",
        "provenance",
        "schema",
    }), label="external capsule manifest")
    if (type(manifest["format_version"]) is not int
            or type(manifest["input_integer_count"]) is not int
            or not isinstance(manifest["input_stable_key_sha256"], str)
            or manifest["artifact_kind"] != V4_EXTERNAL_INPUT_CAPSULE_KIND
            or manifest["schema"] != V4_EXTERNAL_INPUT_MANIFEST_SCHEMA
            or manifest["format_version"] != 1):
        _fail("external capsule manifest kind/schema/version 非法")
    return manifest


def _manifest_file_entries(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """验证 v1 manifest 的两份 data file 声明，尚不读取其内容。"""
    files = _exact_fields(manifest["files"], frozenset({
        _INPUT_FILE.name, _INTS_FILE.name,
    }), label="external capsule manifest files")
    entries = {}
    for name in (_INPUT_FILE.name, _INTS_FILE.name):
        entry = _exact_fields(files[name], frozenset({"sha256", "size"}),
                              label=f"external capsule manifest files.{name}")
        if type(entry["size"]) is not int or entry["size"] <= 0:
            _fail(f"external capsule manifest 文件大小非法: {name}")
        _digest_from_document(
            entry["sha256"], label=f"external capsule manifest files.{name}.sha256")
        entries[name] = entry
    return entries


def _runtime_source_capsule_from_payloads(
        manifest_payload: bytes,
        input_payload: bytes,
        ints_payload: bytes,
        *,
        manifest: dict[str, Any] | None = None,
        ) -> ConversationHeldOutV4RuntimeSourceCapsule:
    """以 legacy reader 的完整 typed 闭合规则从已读 payload 重建 runtime capsule。"""
    if manifest is None:
        manifest = _parse_external_input_manifest(manifest_payload)
    files = _manifest_file_entries(manifest)
    expected_files = {
        _INPUT_FILE.name: input_payload,
        _INTS_FILE.name: ints_payload,
    }
    for name, payload in expected_files.items():
        entry = _exact_fields(files[name], frozenset({"sha256", "size"}),
                              label=f"external capsule manifest files.{name}")
        if (type(entry["size"]) is not int or entry["size"] != len(payload)
                or entry["sha256"] != hashlib.sha256(payload).hexdigest()):
            _fail(f"external capsule manifest 文件身份漂移: {name}")
    try:
        document = parse_canonical_json_bytes(input_payload, require_object=True)
    except DatasetContractError as exc:
        _fail("external capsule input 不是规范 JSON")
        raise AssertionError from exc
    capsule = ConversationHeldOutV4ExternalInputCapsule.from_document(document)
    provenance = _exact_fields(manifest["provenance"], frozenset({
        "artifact_sha256", "document_sha256", "inventory_sha256",
    }), label="external capsule manifest provenance")
    expected_provenance = {
        "artifact_sha256": digest_hex(capsule.dependencies.artifact_sha256),
        "document_sha256": digest_hex(capsule.dependencies.document_sha256),
        "inventory_sha256": digest_hex(capsule.dependencies.inventory_sha256),
    }
    if provenance != expected_provenance:
        _fail("external capsule manifest provenance 漂移")
    try:
        decoded_ints = decode_integer_tuple(ints_payload)
    except (IntegerCodecError, TypeError, ValueError) as exc:
        _fail("external capsule canonical ints 非法")
        raise AssertionError from exc
    expected_key = capsule.stable_key()
    if (decoded_ints != expected_key or encode_integer_tuple(decoded_ints) != ints_payload
            or manifest["input_integer_count"] != len(expected_key)
            or manifest["input_stable_key_sha256"]
            != hashlib.sha256(ints_payload).hexdigest()):
        _fail("external capsule canonical ints 与 typed input 漂移")
    expected_manifest = canonical_json_bytes(_manifest_document(
        capsule, input_payload=input_payload, ints_payload=ints_payload))
    if manifest_payload != expected_manifest:
        _fail("external capsule manifest 不是当前 typed input 的规范闭包")
    return ConversationHeldOutV4RuntimeSourceCapsule(
        V4_RUNTIME_SOURCE_ORIGIN_EXTERNAL,
        _sha256(manifest_payload),
        capsule.dependencies,
        capsule.inputs,
        capsule.producer.producer_key,
        capsule.producer.declaration,
    )


def _validate_budgeted_external_capsule_sizes(
        manifest: dict[str, Any],
        *,
        identities: dict[Path, tuple[int, ...]],
        manifest_size: int,
        budget: ConversationHeldOutV4ExternalCapsuleBudget,
        ) -> dict[str, dict[str, Any]]:
    """在读取 data payload 前核验 manifest 声明、物理 size 与总预算。"""
    entries = _manifest_file_entries(manifest)
    input_size = entries[_INPUT_FILE.name]["size"]
    ints_size = entries[_INTS_FILE.name]["size"]
    if input_size > budget.max_input_payload_bytes:
        _fail("external capsule manifest input 声明超过读取预算")
    if ints_size > budget.max_integer_payload_bytes:
        _fail("external capsule manifest ints 声明超过读取预算")
    if manifest_size > budget.max_manifest_payload_bytes:
        _fail("external capsule manifest 声明超过读取预算")
    if (manifest_size + input_size + ints_size
            > budget.max_total_payload_bytes):
        _fail("external capsule manifest 声明总量超过读取预算")
    for relative, declared_size in (
            (_INPUT_FILE, input_size),
            (_INTS_FILE, ints_size),
            (_MANIFEST_FILE, manifest_size)):
        if identities[relative][4] != declared_size:
            _fail(f"external capsule manifest 文件大小漂移: {relative}")
    return entries


def read_budgeted_v4_external_input_capsule(
        root: str | Path,
        *,
        budget: ConversationHeldOutV4ExternalCapsuleBudget,
        require_k_drive: bool = True,
        ) -> ConversationHeldOutV4RuntimeSourceCapsule:
    """先有界读取 manifest，再按其受预算声明重建 v1 external typed capsule。"""
    if not isinstance(budget, ConversationHeldOutV4ExternalCapsuleBudget):
        raise TypeError("external capsule budget type is invalid")
    target = _read_root(root, require_k_drive=require_k_drive)
    _file_set(target)
    identities = {
        relative: _capture_plain_file_identity(target, relative)[1]
        for relative in _EXPECTED_FILES
    }
    manifest_payload = _read_bounded_plain_bytes(
        target,
        _MANIFEST_FILE,
        maximum_bytes=budget.max_manifest_payload_bytes,
        expected_identity=identities[_MANIFEST_FILE],
    )
    manifest = _parse_external_input_manifest(manifest_payload)
    entries = _validate_budgeted_external_capsule_sizes(
        manifest,
        identities=identities,
        manifest_size=len(manifest_payload),
        budget=budget,
    )
    _require_plain_file_identity(
        target, _INPUT_FILE, identities[_INPUT_FILE])
    _require_plain_file_identity(
        target, _INTS_FILE, identities[_INTS_FILE])
    input_payload = _read_bounded_plain_bytes(
        target,
        _INPUT_FILE,
        maximum_bytes=budget.max_input_payload_bytes,
        expected_identity=identities[_INPUT_FILE],
        expected_size=entries[_INPUT_FILE.name]["size"],
    )
    ints_payload = _read_bounded_plain_bytes(
        target,
        _INTS_FILE,
        maximum_bytes=budget.max_integer_payload_bytes,
        expected_identity=identities[_INTS_FILE],
        expected_size=entries[_INTS_FILE.name]["size"],
    )
    result = _runtime_source_capsule_from_payloads(
        manifest_payload, input_payload, ints_payload, manifest=manifest)
    for relative in _EXPECTED_FILES:
        _require_plain_file_identity(target, relative, identities[relative])
    return result


def read_v4_external_input_capsule(
        root: str | Path,
        *,
        require_k_drive: bool = True,
        ) -> ConversationHeldOutV4RuntimeSourceCapsule:
    """只读验证三文件闭包并返回 runtime 可消费的 external typed capsule。"""
    target = _read_root(root, require_k_drive=require_k_drive)
    _file_set(target)
    manifest_payload = _read_plain_bytes(target, _MANIFEST_FILE)
    input_payload = _read_plain_bytes(target, _INPUT_FILE)
    ints_payload = _read_plain_bytes(target, _INTS_FILE)
    return _runtime_source_capsule_from_payloads(
        manifest_payload, input_payload, ints_payload)


__all__ = [
    "ConversationHeldOutV4ExternalCapsuleBudget",
    "ConversationHeldOutV4ExternalCapsuleError",
    "ConversationHeldOutV4ExternalInputCapsule",
    "ConversationHeldOutV4ExternalProducer",
    "ConversationHeldOutV4PreparedExternalInputCapsule",
    "V4_EXTERNAL_INPUT_CAPSULE_KIND",
    "V4_EXTERNAL_INPUT_CAPSULE_SCHEMA",
    "V4_EXTERNAL_INPUT_MANIFEST_SCHEMA",
    "V4_EXTERNAL_PRODUCER_DECLARATION",
    "prepare_v4_external_input_capsule",
    "read_budgeted_v4_external_input_capsule",
    "read_v4_external_input_capsule",
    "write_prepared_v4_external_input_capsule",
    "write_v4_external_input_capsule",
]
