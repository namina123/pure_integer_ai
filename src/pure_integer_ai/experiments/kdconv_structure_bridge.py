"""KdConv 结构化属性桥。

课程正文只在构建期读取；sidecar 只保存码点整数、稳定身份和支持计数。
运行时把唯一的 (domain, subject, attribute, value) 结构物化到当前
TrainContext.graph_ontology，供查询输入投影和生成侧重填使用。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pure_integer_ai.cognition.shared.graph_ontology import relation_concept_identity
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE, ObjectIdentity, SourceRef, VersionBundle,
    concept_identity,
)
from pure_integer_ai.storage.assertion_identity import IDENTITY_GRAPH_OBJECT
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity, entity_identity, proposition_identity, role_identity,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationSurface, RelationSurfaceBinding, GraphRelationGeneration,
)
from pure_integer_ai.cognition.shared.response_plan import (
    ResponsePlan, ResponseRealization, ResponseSlot, response_act_identity,
)
from pure_integer_ai.cognition.shared.identity import OBJECT_CONCEPT, OBJECT_ENTITY
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import TYPE_INT, register_extension_table


FORMAT = "PURE_INTEGER_AI_KDCONV_STRUCTURE_COURSE_V3"
BRIDGE_VERSION = 3
NAMESPACE = 91630
SOURCE_KIND = 91631
PREDICATE_KIND = 1
ROLE_SUBJECT = 1
ROLE_VALUE = 2
FACT_TABLE = "kdconv_structure_fact_v3"
PART_TABLE = "kdconv_structure_part_v3"
_FIELD_SUBJECT = 1
_FIELD_ATTRIBUTE = 2
_FIELD_VALUE = 3
_FIELD_PREFIX = 4
_FIELD_MIDDLE = 5
_FIELD_SUFFIX = 6


def _positive(value: object) -> int:
    raw = repr(value).encode("utf-8")
    result = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    return (result & ((1 << 63) - 1)) or 1


def _codes(value: object) -> tuple[int, ...]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("KdConv structure text must be non-empty")
    return tuple(ord(item) for item in value.strip())


def _contains_integer(haystack: tuple[int, ...], needle: tuple[int, ...]) -> bool:
    """判断整数序列连续包含关系；不引入字符或语言相似度。"""
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(haystack[start:start + width] == needle
               for start in range(len(haystack) - width + 1))


def _source(domain: int, semantic_key: tuple[int, ...]) -> SourceRef:
    return SourceRef(SOURCE_KIND, domain, _positive(semantic_key),
                     GLOBAL_OWNER_SCOPE, VersionBundle())


def _row_key(domain: int, subject: tuple[int, ...], attribute: tuple[int, ...],
             value: tuple[int, ...]) -> tuple[int, ...]:
    return (domain, len(subject), *subject, len(attribute), *attribute,
            len(value), *value)


_BODY_ATTRIBUTE_NAMES = frozenset({
    "Information", "information", "简介", "摘要", "内容简介", "详细信息",
    "description", "Description", "summary", "Summary", "abstract", "Abstract",
})


def _is_body_attribute(value: str) -> bool:
    """Reject source-body fields; structured attributes remain untouched."""
    return value.strip() in _BODY_ATTRIBUTE_NAMES


@dataclass(frozen=True, slots=True)
class KdConvStructure:
    domain: int
    subject_values: tuple[int, ...]
    attribute_values: tuple[int, ...]
    value_values: tuple[int, ...]
    support_count: int
    frame_prefix: tuple[int, ...] = ()
    frame_middle: tuple[int, ...] = ()
    frame_suffix: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if (type(self.domain) is not int or self.domain <= 0
                or type(self.support_count) is not int or self.support_count <= 0):
            raise ValueError("KdConv structure integer header invalid")
        for name in ("subject_values", "attribute_values", "value_values",
                     "frame_prefix", "frame_middle", "frame_suffix"):
            value = getattr(self, name)
            if type(value) is not tuple or any(type(item) is not int or item < 0
                                               for item in value):
                raise ValueError(f"{name} must be integer tuple")
        if not self.subject_values or not self.attribute_values or not self.value_values:
            raise ValueError("KdConv structure slots must be non-empty")

    @property
    def semantic_key(self) -> tuple[int, ...]:
        return _row_key(self.domain, self.subject_values,
                        self.attribute_values, self.value_values)

    def to_dict(self) -> dict[str, object]:
        return {
            "v": BRIDGE_VERSION,
            "domain": self.domain,
            "subject": list(self.subject_values),
            "attribute": list(self.attribute_values),
            "value": list(self.value_values),
            "support": self.support_count,
            "prefix": list(self.frame_prefix),
            "middle": list(self.frame_middle),
            "suffix": list(self.frame_suffix),
        }

    @classmethod
    def from_dict(cls, value: object) -> "KdConvStructure":
        if not isinstance(value, dict) or value.get("v") != BRIDGE_VERSION:
            raise ValueError("KdConv structure sidecar version invalid")
        def ints(name: str, required: bool = True) -> tuple[int, ...]:
            raw = value.get(name)
            if not isinstance(raw, list) or (required and not raw) \
                    or any(type(item) is not int or item < 0 for item in raw):
                raise ValueError(f"KdConv sidecar field {name} invalid")
            return tuple(raw)
        return cls(value["domain"], ints("subject"), ints("attribute"),
                   ints("value"), value["support"], ints("prefix", False),
                   ints("middle", False), ints("suffix", False))


def _frame_for_response(subject: str, value: str, response: str) -> tuple[tuple[int, ...], ...]:
    """抽取带槽 literal 间隔；找不到 subject 时只保留 value 周围结构。"""
    def optional(text: str) -> tuple[int, ...]:
        return tuple(ord(item) for item in text)
    if value and value in response:
        before, after = response.split(value, 1)
        if subject and subject in before:
            prefix, middle = before.split(subject, 1)
            return optional(prefix), optional(middle), optional(after)
        return optional(before), (), optional(after)
    return (), (), ()


def build_kdconv_structure_course(source_root: str | Path,
                                  output: str | Path) -> dict[str, object]:
    """从冻结 KdConv train/dev/test 生成去重整数 sidecar。"""
    root = Path(source_root).resolve()
    target = Path(output).resolve()
    if not root.is_dir() or target.exists():
        raise ValueError("KdConv structure source/output invalid")
    domains = ("film", "music", "travel")
    domain_ids = {name: index for index, name in enumerate(domains, 1)}
    rows: dict[tuple[int, ...], KdConvStructure] = {}
    rejected = {"body_attribute": 0, "invalid_row": 0}
    for domain_name in domains:
        paths = [root / "data" / domain_name / f"{split}.json"
                 for split in ("train", "dev", "test")]
        # The KB is a structured source, not a response corpus.  It is read
        # only at build time and contributes typed subject/attribute/value
        # rows; body fields are rejected below before integer materialization.
        kb_path = root / "data" / domain_name / f"kb_{domain_name}.json"
        if kb_path.is_file():
            paths.append(kb_path)
        for path in paths:
            conversations = json.loads(path.read_text(encoding="utf-8"))
            records = []
            if path.name.startswith("kb_"):
                for subject_name, attrs in conversations.items():
                    structured = [
                        {"name": subject_name, "attrname": row[1],
                         "attrvalue": row[2]}
                        for row in attrs
                        if isinstance(row, list) and len(row) == 3
                    ]
                    records.append((subject_name, structured, ""))
            else:
                for conversation in conversations:
                    for message in conversation.get("messages", []):
                        records.append((None, message.get("attrs", []) or [],
                                        str(message.get("message", ""))))
            for subject_hint, attrs, message_surface in records:
                if not isinstance(attrs, list):
                    continue
                for attr in attrs:
                    if not isinstance(attr, dict):
                        continue
                    try:
                        subject_text = str(attr.get("name", subject_hint)).strip()
                        attribute_text = str(attr["attrname"]).strip()
                        value_text = str(attr["attrvalue"]).strip()
                    except KeyError:
                        continue
                    if not subject_text or not attribute_text or not value_text:
                        rejected["invalid_row"] += 1
                        continue
                    if _is_body_attribute(attribute_text):
                        rejected["body_attribute"] += 1
                        continue
                    subject = _codes(subject_text)
                    attribute = _codes(attribute_text)
                    value = _codes(value_text)
                    key = _row_key(domain_ids[domain_name], subject, attribute, value)
                    prior = rows.get(key)
                    # Response surfaces are not part of the structure sidecar.
                    # Keeping them here would create a source-text replay path.
                    prefix, middle, suffix = (), (), ()
                    if prior is None:
                        rows[key] = KdConvStructure(
                            domain_ids[domain_name], subject, attribute, value, 1,
                            prefix, middle, suffix)
                    else:
                        rows[key] = KdConvStructure(
                            prior.domain, prior.subject_values,
                            prior.attribute_values, prior.value_values,
                            prior.support_count + 1,
                            prior.frame_prefix or prefix,
                            prior.frame_middle or middle,
                            prior.frame_suffix or suffix)
    if not rows:
        raise ValueError("KdConv produced no structure rows")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join((json.dumps(rows[key].to_dict(), ensure_ascii=True,
                                   sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
                        for key in sorted(rows))
    target.write_bytes(payload)
    return {"format": FORMAT, "bridge_version": BRIDGE_VERSION,
            "row_count": len(rows), "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rejected": rejected,
            "path": target.as_posix()}


def read_kdconv_structure_course(path: str | Path) -> tuple[KdConvStructure, ...]:
    payload = Path(path).read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("KdConv structure sidecar must be newline terminated")
    rows = []
    seen = set()
    for line in payload.splitlines():
        item = KdConvStructure.from_dict(json.loads(line.decode("ascii")))
        if item.semantic_key in seen:
            raise ValueError("KdConv sidecar semantic duplicate")
        seen.add(item.semantic_key)
        rows.append(item)
    return tuple(sorted(rows, key=lambda item: item.semantic_key))


class KdConvStructureRuntime:
    """把 sidecar 结构接入当前 GraphOntology；不读取课程正文。"""

    def __init__(self, context, course_path: str | Path | None = None, *,
                 materialize: bool = True):
        self.context = context
        self._lazy = not materialize and course_path is None
        self.course_path = (None if course_path is None
                            else Path(course_path).resolve())
        _register_tables(context.backend)
        if self.course_path is not None:
            self.rows = read_kdconv_structure_course(self.course_path)
        elif self._lazy:
            # 发布模型只恢复整数 subject posting；完整事实在 QueryState
            # 确认输入命中后按 fact_id page-in。这样不把 12 万条事实复制成
            # Python 对象、结构候选和输入 posting 常驻内存。
            self.rows = ()
        else:
            self.rows = _read_rows(context.backend)
        self._fact_count = (len(self.rows) if not self._lazy
                            else context.backend.count(FACT_TABLE))
        self.facts: tuple[ActiveRelationSurface, ...] = ()
        self._fact_objects: dict[tuple[int, ...], ActiveRelationSurface] = {}
        self.structures = ()
        self._fact_rows: dict[tuple[int, ...], KdConvStructure] = {}
        # Query-time postings are keyed by the complete integer subject
        # sequence.  The first-codepoint index narrows lookup without ever
        # falling back to character-neighbour or substring similarity logic.
        self._facts_by_subject: dict[tuple[int, ...], tuple[ActiveRelationSurface, ...]] = {}
        self._subject_postings: dict[int, tuple[tuple[int, ...], ...]] = {}
        self._subject_fact_ids: dict[tuple[int, ...], tuple[int, ...]] = {}
        self._entity_fact_ids: dict[tuple[int, int], tuple[int, ...]] | None = None
        self._concept_fact_ids: dict[tuple[int, int], tuple[int, ...]] | None = None
        if self._lazy:
            self._build_lazy_postings()
        else:
            self._build(materialize=materialize)

    def _build_lazy_postings(self) -> None:
        """恢复完整 subject 索引，但不恢复事实对象或来源表层。"""
        grouped: dict[int, list[tuple[int, int]]] = {}
        connection = getattr(self.context.backend, "_conn", None)
        if connection is not None:
            cursor = connection.execute(
                f'SELECT "fact_id", "part_ordinal", "part_value" '
                f'FROM "{PART_TABLE}" WHERE "field_code"=? '
                'ORDER BY "fact_id", "part_ordinal"', (_FIELD_SUBJECT,))
            while True:
                batch = cursor.fetchmany(8192)
                if not batch:
                    break
                for fact_id, ordinal, value in batch:
                    grouped.setdefault(int(fact_id), []).append(
                        (int(ordinal), int(value)))
        else:
            for part in self.context.backend.select(
                    PART_TABLE, where={"field_code": _FIELD_SUBJECT},
                    order_by="fact_id"):
                grouped.setdefault(part["fact_id"], []).append(
                    (part["part_ordinal"], part["part_value"]))
        by_subject: dict[tuple[int, ...], list[int]] = {}
        for fact_id, values in grouped.items():
            ordered = tuple(value for _ordinal, value in sorted(values))
            if (not ordered
                    or tuple(item[0] for item in sorted(values))
                    != tuple(range(len(values)))):
                raise ValueError("KdConv lazy subject posting ordinal gap")
            by_subject.setdefault(ordered, []).append(fact_id)
        self._subject_fact_ids = {
            subject: tuple(sorted(ids))
            for subject, ids in by_subject.items()
        }
        postings: dict[int, list[tuple[int, ...]]] = {}
        for subject in self._subject_fact_ids:
            postings.setdefault(subject[0], []).append(subject)
        self._subject_postings = {
            codepoint: tuple(sorted(subjects, key=lambda item: (len(item), item)))
            for codepoint, subjects in postings.items()
        }

    def _build_lazy_graph_object_postings(self) -> None:
        """仅在图身份入口实际使用时构建 filler→fact 反向索引。"""
        if self._entity_fact_ids is not None:
            return
        connection = getattr(self.context.backend, "_conn", None)
        if connection is not None:
            domains = {
                int(fact_id): int(domain)
                for fact_id, domain in connection.execute(
                    f'SELECT "fact_id", "domain" FROM "{FACT_TABLE}" '
                    'ORDER BY "fact_id"')
            }
        else:
            domains = {
                item["fact_id"]: item["domain"]
                for item in self.context.backend.select(
                    FACT_TABLE, order_by="fact_id")
            }
        indexed_subject_ids = {
            fact_id for ids in self._subject_fact_ids.values() for fact_id in ids
        }
        if set(domains) != indexed_subject_ids:
            raise ValueError("KdConv lazy subject posting 与事实集合不一致")
        entity_ids: dict[tuple[int, int], list[int]] = {}
        for subject, fact_ids in self._subject_fact_ids.items():
            digest = _positive(subject)
            for fact_id in fact_ids:
                entity_ids.setdefault(
                    (domains[fact_id], digest), []).append(fact_id)

        concept_ids: dict[tuple[int, int], list[int]] = {}
        value_fact_ids: set[int] = set()
        if connection is not None:
            cursor = connection.execute(
                f'SELECT "fact_id", "part_ordinal", "part_value" '
                f'FROM "{PART_TABLE}" WHERE "field_code"=? '
                'ORDER BY "fact_id", "part_ordinal"', (_FIELD_VALUE,))
            current_id: int | None = None
            current_values: list[tuple[int, int]] = []

            def finish_value() -> None:
                if current_id is None:
                    return
                ordered = tuple(value for _ordinal, value in current_values)
                if (not ordered
                        or tuple(item[0] for item in current_values)
                        != tuple(range(len(current_values)))):
                    raise ValueError("KdConv lazy value posting ordinal gap")
                value_fact_ids.add(current_id)
                concept_ids.setdefault(
                    (domains[current_id], _positive(ordered)), []).append(
                        current_id)

            while True:
                batch = cursor.fetchmany(8192)
                if not batch:
                    break
                for fact_id, ordinal, value in batch:
                    fact_id = int(fact_id)
                    if current_id is not None and fact_id != current_id:
                        finish_value()
                        current_values = []
                    current_id = fact_id
                    current_values.append((int(ordinal), int(value)))
            finish_value()
        else:
            values_by_fact: dict[int, list[tuple[int, int]]] = {}
            for part in self.context.backend.select(
                    PART_TABLE, where={"field_code": _FIELD_VALUE},
                    order_by="fact_id"):
                values_by_fact.setdefault(part["fact_id"], []).append(
                    (part["part_ordinal"], part["part_value"]))
            if set(values_by_fact) != set(domains):
                raise ValueError("KdConv lazy value posting 与事实集合不一致")
            for fact_id, values in values_by_fact.items():
                ordered_values = sorted(values)
                if (not ordered_values
                        or tuple(item[0] for item in ordered_values)
                        != tuple(range(len(ordered_values)))):
                    raise ValueError("KdConv lazy value posting ordinal gap")
                value_fact_ids.add(fact_id)
                concept_ids.setdefault(
                    (domains[fact_id], _positive(tuple(
                        item[1] for item in ordered_values))), []).append(
                            fact_id)
        if value_fact_ids != set(domains):
            raise ValueError("KdConv lazy value posting 与事实集合不一致")
        self._entity_fact_ids = {
            key: tuple(sorted(set(ids))) for key, ids in entity_ids.items()
        }
        self._concept_fact_ids = {
            key: tuple(sorted(set(ids))) for key, ids in concept_ids.items()
        }

    def _read_selected_rows(
            self, fact_ids: tuple[int, ...],
            ) -> tuple[KdConvStructure, ...]:
        """按选中的 fact_id 恢复结构行，不扫描或复制其它事实。"""
        if not fact_ids:
            return ()
        backend = self.context.backend
        facts: list[dict[str, int]] = []
        parts: list[dict[str, int]] = []
        connection = getattr(backend, "_conn", None)
        if connection is None:
            selected = set(fact_ids)
            facts = [item for item in backend.select(FACT_TABLE, order_by="fact_id")
                     if item["fact_id"] in selected]
            parts = [item for item in backend.select(PART_TABLE, order_by="fact_id")
                     if item["fact_id"] in selected]
        else:
            for start in range(0, len(fact_ids), 900):
                batch = fact_ids[start:start + 900]
                placeholders = ",".join("?" for _ in batch)
                facts.extend({
                    "fact_id": int(row[0]), "domain": int(row[1]),
                    "support_count": int(row[2]),
                } for row in connection.execute(
                    f'SELECT "fact_id", "domain", "support_count" '
                    f'FROM "{FACT_TABLE}" WHERE "fact_id" IN ({placeholders}) '
                    'ORDER BY "fact_id"', batch))
                parts.extend({
                    "fact_id": int(row[0]), "field_code": int(row[1]),
                    "part_ordinal": int(row[2]), "part_value": int(row[3]),
                } for row in connection.execute(
                    f'SELECT "fact_id", "field_code", "part_ordinal", "part_value" '
                    f'FROM "{PART_TABLE}" WHERE "fact_id" IN ({placeholders}) '
                    'ORDER BY "fact_id", "field_code", "part_ordinal"', batch))
        parts_by_fact: dict[int, list[dict[str, int]]] = {}
        for part in parts:
            parts_by_fact.setdefault(part["fact_id"], []).append(part)
        rows: list[KdConvStructure] = []
        for fact in sorted(facts, key=lambda item: item["fact_id"]):
            grouped: dict[int, list[tuple[int, int]]] = {}
            for part in parts_by_fact.get(fact["fact_id"], ()):
                grouped.setdefault(part["field_code"], []).append(
                    (part["part_ordinal"], part["part_value"]))
            def field(code: int, required: bool = True) -> tuple[int, ...]:
                values = sorted(grouped.get(code, ()))
                if required and not values:
                    raise ValueError("KdConv trained graph missing structure parts")
                if tuple(item[0] for item in values) != tuple(range(len(values))):
                    raise ValueError("KdConv structure part ordinal gap")
                return tuple(item[1] for item in values)
            row = KdConvStructure(
                fact["domain"], field(_FIELD_SUBJECT), field(_FIELD_ATTRIBUTE),
                field(_FIELD_VALUE), fact["support_count"],
                field(_FIELD_PREFIX, False), field(_FIELD_MIDDLE, False),
                field(_FIELD_SUFFIX, False))
            if _positive(row.semantic_key) != fact["fact_id"]:
                raise ValueError("KdConv trained graph fact hash drift")
            rows.append(row)
        return tuple(rows)

    def _build_fact(self, row: KdConvStructure) -> ActiveRelationSurface:
        """从一条已 page-in 的纯整数结构构造活动关系事实。"""
        source = _source(row.domain, row.semantic_key)
        entity_source = SourceRef(
            SOURCE_KIND, row.domain, 1, GLOBAL_OWNER_SCOPE, VersionBundle())
        subject = entity_identity(entity_source,
                                  (1, _positive(row.subject_values)))
        value = concept_identity((NAMESPACE, 2, row.domain,
                                 _positive(row.value_values)))
        predicate_identity = relation_concept_identity((
            NAMESPACE, PREDICATE_KIND, row.domain,
            _positive(row.attribute_values)))
        proposition = proposition_identity(source, (3, _positive(row.semantic_key)))
        role_subject = role_identity((NAMESPACE, ROLE_SUBJECT))
        role_value = role_identity((NAMESPACE, ROLE_VALUE))
        fact = ActiveRelationSurface(
            proposition, predicate_identity,
            "".join(map(chr, row.attribute_values)),
            (
                RelationSurfaceBinding(role_subject, subject,
                    "".join(map(chr, row.subject_values)),
                    _positive(source.stable_key()), 0,
                    len(row.subject_values)),
                RelationSurfaceBinding(role_value, value,
                    "".join(map(chr, row.value_values)),
                    _positive(source.stable_key()),
                    len(row.subject_values) + len(row.attribute_values),
                    len(row.subject_values) + len(row.attribute_values) + len(row.value_values)),
            ),
            "".join(map(chr, (*row.subject_values, *row.attribute_values, *row.value_values))),
            _positive(source.stable_key()), 0, len(row.attribute_values))
        self._fact_rows[fact.proposition.stable_key()] = row
        self._fact_objects[fact.proposition.stable_key()] = fact
        return fact

    def ensure_fact_ids(self, fact_ids: tuple[int, ...]) -> tuple[ActiveRelationSurface, ...]:
        """按整数身份加载事实，并将其加入当前查询 hot set。"""
        if not self._lazy:
            return self.facts
        known = {
            _positive(row.semantic_key)
            for row in self._fact_rows.values()
        }
        missing = tuple(sorted(set(fact_ids) - known))
        for row in self._read_selected_rows(missing):
            self._build_fact(row)
        selected_ids = set(fact_ids)
        selected = tuple(sorted(
            (fact for fact in self._fact_objects.values()
             if _positive(self._fact_rows[fact.proposition.stable_key()].semantic_key)
             in selected_ids),
            key=lambda item: item.proposition.stable_key()))
        self.facts = tuple(sorted(self._fact_objects.values(),
                                  key=lambda item: item.proposition.stable_key()))
        by_subject: dict[tuple[int, ...], list[ActiveRelationSurface]] = {}
        for fact in self.facts:
            by_subject.setdefault(
                self._fact_rows[fact.proposition.stable_key()].subject_values,
                []).append(fact)
        self._facts_by_subject = {
            subject: tuple(sorted(items, key=lambda item: item.proposition.stable_key()))
            for subject, items in by_subject.items()
        }
        return selected

    def page_in_for_values(self, values: tuple[int, ...]) -> tuple[ActiveRelationSurface, ...]:
        """精确按 subject 整数序列选择 KdConv 事实。"""
        if not self._lazy:
            return self.facts
        ids: set[int] = set()
        for codepoint in set(values):
            for subject in self._subject_postings.get(codepoint, ()):
                if len(subject) >= 2 and _contains_integer(values, subject):
                    ids.update(self._subject_fact_ids[subject])
        return self.ensure_fact_ids(tuple(sorted(ids)))

    @staticmethod
    def _graph_object_lookup_key(
            identity: ObjectIdentity,
            ) -> tuple[int, int, int] | None:
        """把完整 KdConv filler 身份还原为 (kind, domain, digest)。"""
        if identity.object_kind == OBJECT_ENTITY:
            components = identity.components
            if len(components) != 15 or components[0] != 1:
                return None
            try:
                source = SourceRef.from_stable_key(components[1:12])
            except (TypeError, ValueError):
                return None
            if (components[12:14] != (2, 1)
                    or source.source_kind != SOURCE_KIND
                    or source.document_id != 1):
                return None
            digest = components[14]
            expected = entity_identity(
                SourceRef(
                    SOURCE_KIND, source.source_id, 1,
                    GLOBAL_OWNER_SCOPE, VersionBundle()),
                (1, digest),
            )
            if identity != expected:
                return None
            return OBJECT_ENTITY, source.source_id, digest
        if identity.object_kind == OBJECT_CONCEPT:
            components = identity.components
            if (len(components) != 4
                    or components[:2] != (NAMESPACE, 2)):
                return None
            domain, digest = components[2:]
            expected = concept_identity((NAMESPACE, 2, domain, digest))
            if identity != expected:
                return None
            return OBJECT_CONCEPT, domain, digest
        return None

    def page_in_for_graph_object_keys(
            self,
            graph_object_keys: tuple[tuple[int, ...], ...],
            ) -> tuple[ActiveRelationSurface, ...]:
        """按完整 Entity/Concept 图身份加载所有相邻结构事实。"""
        if (type(graph_object_keys) is not tuple
                or any(type(key) is not tuple or not key
                       or any(type(value) is not int or value < 0 for value in key)
                       for key in graph_object_keys)):
            raise ValueError("graph_object_keys 必须是非空非负整数 tuple")
        if len(set(graph_object_keys)) != len(graph_object_keys):
            raise ValueError("graph_object_keys 不得重复")
        requested = set(graph_object_keys)
        if not self._lazy:
            return tuple(sorted((
                fact for fact in self.facts
                if any(binding.filler.stable_key() in requested
                       for binding in fact.bindings)
            ), key=lambda item: item.proposition.stable_key()))
        ids: set[int] = set()
        lookups = []
        for key in graph_object_keys:
            try:
                identity = ObjectIdentity.from_stable_key(key)
            except (TypeError, ValueError):
                continue
            lookup = self._graph_object_lookup_key(identity)
            if lookup is None:
                continue
            lookups.append(lookup)
        if not lookups:
            return ()
        self._build_lazy_graph_object_postings()
        if self._entity_fact_ids is None or self._concept_fact_ids is None:
            raise RuntimeError("KdConv graph object posting 未闭合")
        for kind, domain, digest in lookups:
            index = (self._entity_fact_ids if kind == OBJECT_ENTITY
                     else self._concept_fact_ids)
            ids.update(index.get((domain, digest), ()))
        return self.ensure_fact_ids(tuple(sorted(ids)))

    def _build(self, *, materialize: bool) -> None:
        ontology = self.context.graph_ontology
        ontology.enable_physical_statement_projection()
        predicate_cache = {}
        existing_hashes: set[int] | None = None
        if not materialize:
            # Read-only release restore used to call ontology.resolve five
            # times per row.  A full graph_object scan is also too expensive
            # (the table is >1.5M rows), so compute the exact expected hashes
            # from the sidecar and probe them in bounded IN batches.
            connection = getattr(self.context.backend, "_conn", None)
            if connection is not None:
                registry = self.context.scoped_identity_store.registry
                expected: set[int] = set()
                role_subject = role_identity((NAMESPACE, ROLE_SUBJECT))
                role_value = role_identity((NAMESPACE, ROLE_VALUE))
                for row in self.rows:
                    source = _source(row.domain, row.semantic_key)
                    entity_source = SourceRef(
                        SOURCE_KIND, row.domain, 1,
                        GLOBAL_OWNER_SCOPE, VersionBundle())
                    subject = entity_identity(
                        entity_source, (1, _positive(row.subject_values)))
                    value = concept_identity((
                        NAMESPACE, 2, row.domain,
                        _positive(row.value_values)))
                    predicate = relation_concept_identity((
                        NAMESPACE, PREDICATE_KIND, row.domain,
                        _positive(row.attribute_values)))
                    proposition = proposition_identity(
                        source, (3, _positive(row.semantic_key)))
                    for identity in (subject, value, predicate, proposition,
                                     role_subject, role_value):
                        expected.add(registry.identity_hash(
                            IDENTITY_GRAPH_OBJECT, identity.stable_key()))
                existing_hashes = set()
                ordered_hashes = tuple(sorted(expected))
                for start in range(0, len(ordered_hashes), 900):
                    batch = ordered_hashes[start:start + 900]
                    placeholders = ",".join("?" for _ in batch)
                    existing_hashes.update(
                        int(item[0]) for item in connection.execute(
                            'SELECT "identity_hash" FROM "graph_object" '
                            f'WHERE "identity_hash" IN ({placeholders})',
                            batch))

        def require_existing(identity) -> None:
            if existing_hashes is None:
                if ontology.resolve(identity) is None:
                    raise ValueError("KdConv trained graph object missing")
                return
            identity_hash = self.context.scoped_identity_store.registry.identity_hash(
                IDENTITY_GRAPH_OBJECT, identity.stable_key())
            if identity_hash not in existing_hashes:
                raise ValueError("KdConv trained graph object missing")

        facts = []
        from pure_integer_ai.cognition.understanding.query_structure_adapter import RelationSurfaceStructureAdapter
        for row in self.rows:
            source = _source(row.domain, row.semantic_key)
            # Entity identity is domain/name scoped, never fact scoped: the
            # same movie/person occurring under several attributes resolves to
            # one semantic object instead of being re-entered per relation.
            entity_source = SourceRef(
                SOURCE_KIND, row.domain, 1, GLOBAL_OWNER_SCOPE, VersionBundle())
            subject = entity_identity(entity_source,
                                      (1, _positive(row.subject_values)))
            value = concept_identity((NAMESPACE, 2, row.domain,
                                     _positive(row.value_values)))
            predicate_key = (NAMESPACE, PREDICATE_KIND, row.domain,
                             _positive(row.attribute_values))
            predicate_identity = relation_concept_identity(predicate_key)
            if materialize:
                predicate = predicate_cache.setdefault(
                    predicate_key, ontology.materialize(predicate_identity))
                predicate_identity = ontology.identity_of(predicate)
            else:
                require_existing(predicate_identity)
            proposition = proposition_identity(source, (3, _positive(row.semantic_key)))
            role_subject = role_identity((NAMESPACE, ROLE_SUBJECT))
            role_value = role_identity((NAMESPACE, ROLE_VALUE))
            if materialize:
                for identity in (subject, value, proposition,
                                 role_subject, role_value):
                    ontology.materialize(identity)
                scope = document_scope(source)
                ontology.relate(predicate, ontology.resolve(subject),
                                ontology.resolve(value), scope=scope,
                                provenance_kind=EPI_STRUCTURED,
                                content_version=BRIDGE_VERSION)
                _write_row(self.context.backend, row)
            else:
                for identity in (subject, value, proposition,
                                 role_subject, role_value):
                    require_existing(identity)
            fact = ActiveRelationSurface(
                proposition, predicate_identity, "".join(map(chr, row.attribute_values)),
                (
                    RelationSurfaceBinding(role_subject, subject,
                        "".join(map(chr, row.subject_values)),
                        _positive(source.stable_key()), 0,
                        len(row.subject_values)),
                    RelationSurfaceBinding(role_value, value,
                        "".join(map(chr, row.value_values)),
                        _positive(source.stable_key()),
                        len(row.subject_values) + len(row.attribute_values),
                        len(row.subject_values) + len(row.attribute_values) + len(row.value_values)),
                ),
                "".join(map(chr, (*row.subject_values, *row.attribute_values, *row.value_values))),
                _positive(source.stable_key()), 0, len(row.attribute_values))
            facts.append(fact)
            self._fact_rows[fact.proposition.stable_key()] = row
        self.facts = tuple(sorted(facts, key=lambda item: item.proposition.stable_key()))
        self.structures = RelationSurfaceStructureAdapter.from_facts(self.facts)
        by_subject: dict[tuple[int, ...], list[ActiveRelationSurface]] = {}
        for fact in self.facts:
            row = self._fact_rows[fact.proposition.stable_key()]
            by_subject.setdefault(row.subject_values, []).append(fact)
        self._facts_by_subject = {
            subject: tuple(sorted(items, key=lambda item: item.proposition.stable_key()))
            for subject, items in by_subject.items()
        }
        postings: dict[int, list[tuple[int, ...]]] = {}
        for subject in self._facts_by_subject:
            postings.setdefault(subject[0], []).append(subject)
        self._subject_postings = {
            codepoint: tuple(sorted(subjects, key=lambda item: (len(item), item)))
            for codepoint, subjects in postings.items()
        }

    def generate(self, fact: ActiveRelationSurface) -> GraphRelationGeneration:
        row = self._fact_rows[fact.proposition.stable_key()]
        subject = next(item.surface for item in fact.bindings if item.role.components[-1] == ROLE_SUBJECT)
        value = next(item.surface for item in fact.bindings if item.role.components[-1] == ROLE_VALUE)
        prefix = "".join(map(chr, row.frame_prefix))
        middle = "".join(map(chr, row.frame_middle))
        suffix = "".join(map(chr, row.frame_suffix))
        if row.frame_prefix or row.frame_middle or row.frame_suffix:
            surface = prefix + subject + middle + value + suffix
        else:
            surface = subject + "".join(map(chr, row.attribute_values)) + value
        return GraphRelationGeneration(surface, fact.proposition, fact.source_hash, 2,
                                       trace=(NAMESPACE, row.support_count))

    def report(self) -> dict[str, int]:
        return {"bridge_version": BRIDGE_VERSION, "row_count": self._fact_count,
                "semantic_unique_count": self._fact_count,
                "graph_fact_count": len(self.facts), "source_body_read_for_binding": 0,
                "semantic_object_copy_count": 0,
                "frame_literal_count": sum(bool(item.frame_prefix or item.frame_middle or item.frame_suffix)
                                            for item in self.rows)}


def build_kdconv_structure_runtime(context, course_path: str | Path) -> KdConvStructureRuntime:
    return KdConvStructureRuntime(context, course_path)


def load_kdconv_structure_runtime(context) -> KdConvStructureRuntime:
    return KdConvStructureRuntime(context, None, materialize=False)


def _register_tables(backend) -> None:
    register_extension_table(backend, FACT_TABLE, [
        ("fact_id", TYPE_INT), ("domain", TYPE_INT),
        ("support_count", TYPE_INT),
    ], disc.DISC_NONE, indexes=[("domain",)], recovery_key=("fact_id",))
    register_extension_table(backend, PART_TABLE, [
        ("fact_id", TYPE_INT), ("field_code", TYPE_INT),
        ("part_ordinal", TYPE_INT), ("part_value", TYPE_INT),
    ], disc.DISC_NONE, indexes=[("fact_id",), ("field_code",)],
        recovery_key=("fact_id", "field_code", "part_ordinal"))


def _write_row(backend, row: KdConvStructure) -> None:
    fact_id = _positive(row.semantic_key)
    found = backend.select(FACT_TABLE, where={"fact_id": fact_id})
    expected = {"fact_id": fact_id, "domain": row.domain,
                "support_count": row.support_count}
    if found:
        if found != [expected]:
            raise ValueError("KdConv fact identity collision")
        return
    backend.insert(FACT_TABLE, expected)
    fields = (( _FIELD_SUBJECT, row.subject_values),
              (_FIELD_ATTRIBUTE, row.attribute_values),
              (_FIELD_VALUE, row.value_values),
              (_FIELD_PREFIX, row.frame_prefix),
              (_FIELD_MIDDLE, row.frame_middle),
              (_FIELD_SUFFIX, row.frame_suffix))
    backend.insert_many(PART_TABLE, (
        {"fact_id": fact_id, "field_code": code,
         "part_ordinal": ordinal, "part_value": value}
        for code, values in fields for ordinal, value in enumerate(values)))


def _read_rows(backend) -> tuple[KdConvStructure, ...]:
    facts = backend.select(FACT_TABLE, order_by="fact_id")
    # Restore the sidecar with one indexed scan.  The previous per-fact
    # select performed one SQLite round-trip for every structure (123k+ on
    # the V3 course), making a read-only release start take hours without
    # changing any graph semantics.
    fact_ids = {fact["fact_id"] for fact in facts}
    parts_by_fact: dict[int, list[dict[str, int]]] = {
        fact_id: [] for fact_id in fact_ids}
    for part in backend.select(PART_TABLE, order_by="fact_id"):
        fact_id = part["fact_id"]
        if fact_id not in parts_by_fact:
            raise ValueError("KdConv trained graph part references unknown fact")
        parts_by_fact[fact_id].append(part)
    rows = []
    for fact in facts:
        parts = parts_by_fact[fact["fact_id"]]
        grouped: dict[int, list[tuple[int, int]]] = {}
        for part in parts:
            grouped.setdefault(part["field_code"], []).append(
                (part["part_ordinal"], part["part_value"]))
        def field(code: int, required: bool = True) -> tuple[int, ...]:
            values = sorted(grouped.get(code, ()))
            if required and not values:
                raise ValueError("KdConv trained graph missing structure parts")
            if tuple(item[0] for item in values) != tuple(range(len(values))):
                raise ValueError("KdConv structure part ordinal gap")
            return tuple(item[1] for item in values)
        row = KdConvStructure(
            fact["domain"], field(_FIELD_SUBJECT), field(_FIELD_ATTRIBUTE),
            field(_FIELD_VALUE), fact["support_count"],
            field(_FIELD_PREFIX, False), field(_FIELD_MIDDLE, False),
            field(_FIELD_SUFFIX, False))
        if _positive(row.semantic_key) != fact["fact_id"]:
            raise ValueError("KdConv trained graph fact hash drift")
        rows.append(row)
    return tuple(rows)


__all__ = [
    "FORMAT", "KdConvStructure", "KdConvStructureRuntime",
    "build_kdconv_structure_course", "build_kdconv_structure_runtime",
    "load_kdconv_structure_runtime", "read_kdconv_structure_course",
]
