"""把既有来源化 response connector 观察编译为关系约束的多图槽课程。

输入观察只提供已审定的句法顺序和 literal 成员；动态成员必须与父图一个
active proposition 的全部 RoleBinding 精确双射。输出不保存整句表示、问答对或
父图 filler 表层，而以 ``(graph category, relation-role ordinal)`` 作为动态槽。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
)
from pure_integer_ai.experiments.response_generation_graph import (
    GENERIC_RESPONSE_GRAPH_CONCEPT,
    GENERIC_RESPONSE_GRAPH_ENTITY,
    GENERIC_RESPONSE_GRAPH_EVENT,
    GENERIC_RESPONSE_GRAPH_PROPOSITION,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    TrainedRelationGraphRuntime,
)
from pure_integer_ai.experiments.unknown_response_course import (
    GENERIC_RESPONSE_CONDITION_VERSION,
    RESPONSE_FEATURE_DISCOURSE_RELATION,
    RESPONSE_FEATURE_ENTITY,
    RESPONSE_FEATURE_EVENT,
    RESPONSE_FEATURE_PROPERTY,
    RESPONSE_FEATURE_UNKNOWN,
    RESPONSE_PART_GRAPH_ROLE,
    RESPONSE_PART_LITERAL,
    STRUCTURAL_RESPONSE_COURSE_VERSION,
    course_from_integer_payload,
)


SOURCE_FORMAT = "RESPONSE_STRUCTURE_SOURCE_LEDGER_V1"
OUTPUT_FORMAT = "STRUCTURAL_RESPONSE_COURSE_LEDGER_V2"
SOURCE_ENVELOPE = (91525, 1)
UNKNOWN_STANCE_ORDINAL = 3

_CATEGORY_BY_OBJECT_KIND = {
    OBJECT_ENTITY: GENERIC_RESPONSE_GRAPH_ENTITY,
    OBJECT_EVENT: GENERIC_RESPONSE_GRAPH_EVENT,
    OBJECT_PROPOSITION: GENERIC_RESPONSE_GRAPH_PROPOSITION,
    OBJECT_CONCEPT: GENERIC_RESPONSE_GRAPH_CONCEPT,
}
_FEATURE_BY_CATEGORY = {
    GENERIC_RESPONSE_GRAPH_ENTITY: RESPONSE_FEATURE_ENTITY,
    GENERIC_RESPONSE_GRAPH_EVENT: RESPONSE_FEATURE_EVENT,
    GENERIC_RESPONSE_GRAPH_PROPOSITION: RESPONSE_FEATURE_DISCOURSE_RELATION,
    GENERIC_RESPONSE_GRAPH_CONCEPT: RESPONSE_FEATURE_PROPERTY,
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _integers(value: object, *, where: str) -> object:
    """把 JSON array 恢复为严格整数 tuple，拒绝 bool/string 语义叶。"""
    if type(value) is int:
        if value < 0:
            raise ValueError(f"{where} 含负整数")
        return value
    if type(value) is list:
        return tuple(
            _integers(item, where=f"{where}[{index}]")
            for index, item in enumerate(value)
        )
    raise ValueError(f"{where} 只能含严格整数和有序数组")


def _role_mapping(dynamic: tuple[tuple, ...], fact: object) -> tuple[int, ...] | None:
    """按完整整数角色载体求双射；不解释文字，也不按相似度选择。"""
    bindings = fact.bindings
    if len(dynamic) != len(bindings):
        return None
    remaining = list(range(len(bindings)))
    result = []
    for part in dynamic:
        matches = tuple(
            index for index in remaining
            if tuple(map(ord, bindings[index].surface)) == part[-1]
        )
        if len(matches) != 1:
            return None
        result.append(matches[0])
        remaining.remove(matches[0])
    return tuple(result) if not remaining else None


def _frame_map(value: object) -> dict[int, tuple[int, ...]]:
    """恢复可选的旧 frame ordinal -> 完整 proposition identity。"""
    if value in (None, []):
        return {}
    records = _integers(value, where="parent_frame_map")
    if type(records) is not tuple:
        raise ValueError("parent_frame_map 不是有序记录")
    result: dict[int, tuple[int, ...]] = {}
    for record in records:
        if (type(record) is not tuple or len(record) != 2
                or type(record[0]) is not int or record[0] <= 0
                or type(record[1]) is not tuple or not record[1]
                or record[0] in result):
            raise ValueError("parent_frame_map 非法或重复")
        result[record[0]] = record[1]
    return result


def _coalesced_parts(parts: tuple[tuple, ...]) -> tuple[tuple, ...]:
    """只合并相邻 literal；动态槽边界和响应顺序保持不变。"""
    result: list[tuple] = []
    literal: list[int] = []

    def flush() -> None:
        if literal:
            result.append((RESPONSE_PART_LITERAL, tuple(literal)))
            literal.clear()

    for part in parts:
        if part[0] == RESPONSE_PART_LITERAL:
            literal.extend(part[1])
        else:
            flush()
            result.append(part)
    flush()
    return tuple(result)


def _feature_mask(categories: tuple[int, ...]) -> int:
    mask = RESPONSE_FEATURE_UNKNOWN
    for category in categories:
        feature = _FEATURE_BY_CATEGORY.get(category)
        if feature is None:
            raise ValueError("图槽 category 没有结构 feature")
        mask |= feature
    return mask


@dataclass(frozen=True, slots=True)
class CompositionalResponseVariant:
    """一个父图关系、其全角色图槽以及来源化句法成员。"""

    observation_ordinal: int
    frame_ordinal: int
    proposition_key: tuple[int, ...]
    condition: tuple[int, ...]
    parts: tuple[tuple, ...]

    def __post_init__(self) -> None:
        if (type(self.observation_ordinal) is not int
                or self.observation_ordinal <= 0
                or type(self.frame_ordinal) is not int
                or self.frame_ordinal <= 0
                or type(self.proposition_key) is not tuple
                or not self.proposition_key
                or type(self.condition) is not tuple
                or len(self.condition) != 4
                or self.condition[0] != GENERIC_RESPONSE_CONDITION_VERSION
                or type(self.parts) is not tuple
                or len(self.parts) < 3):
            raise ValueError("组合响应 variant 不完整")
        graph_parts = tuple(
            item for item in self.parts
            if item[0] == RESPONSE_PART_GRAPH_ROLE
        )
        if (len(graph_parts) < 2
                or len(set((item[1], item[2]) for item in graph_parts))
                != len(graph_parts)
                or not any(item[0] == RESPONSE_PART_LITERAL
                           for item in self.parts)):
            raise ValueError("组合响应必须含唯一的多图槽和 literal")

    def semantic_key(self) -> tuple:
        return self.condition, self.parts


def _compile_variants(
        database: Path,
        observation_payload: bytes,
        source_manifest: dict[str, object],
        ) -> tuple[CompositionalResponseVariant, ...]:
    """把 UNKNOWN 观察映射到当前父图完整角色，不接受部分命中。"""
    decoded = _integers(
        json.loads(observation_payload.decode("ascii")),
        where="response observations",
    )
    if (type(decoded) is not tuple or len(decoded) != 3
            or decoded[:2] != SOURCE_ENVELOPE
            or type(decoded[2]) is not tuple or not decoded[2]):
        raise ValueError("response observation envelope 漂移")
    observations = decoded[2]
    if (tuple(item[0] for item in observations)
            != tuple(range(1, len(observations) + 1))):
        raise ValueError("response observation ordinal 不连续")
    explicit = _frame_map(source_manifest.get("parent_frame_map"))
    result = []
    with TrainedRelationGraphRuntime(database) as runtime:
        facts = runtime.active_surface_facts()
        by_proposition = {
            item.proposition.stable_key(): item for item in facts
        }
        for observation in observations:
            if (type(observation) is not tuple or len(observation) != 4
                    or observation[2] != UNKNOWN_STANCE_ORDINAL
                    or type(observation[3]) is not tuple):
                continue
            dynamic = tuple(
                part for part in observation[3]
                if type(part) is tuple and len(part) == 3
                and part[0] == RESPONSE_PART_GRAPH_ROLE
            )
            if len(dynamic) < 2:
                continue
            fact = None
            mapping = None
            pinned = explicit.get(observation[1])
            if pinned is not None:
                fact = by_proposition.get(pinned)
                if fact is None:
                    raise ValueError("显式父命题不在当前 active graph")
                mapping = _role_mapping(dynamic, fact)
                if mapping is None:
                    raise ValueError("显式父命题与动态成员不成角色双射")
            else:
                matches = tuple(
                    (candidate, candidate_mapping)
                    for candidate in facts
                    for candidate_mapping in (_role_mapping(dynamic, candidate),)
                    if candidate_mapping is not None
                )
                if len(matches) != 1:
                    # 例如同一 Event 对的相反方向关系会共享两个表层载体。
                    # 它们仍保留在父图；本课程不能用 ordinal/排序替其裁决。
                    continue
                fact, mapping = matches[0]
            assert fact is not None and mapping is not None

            category_by_binding = tuple(
                _CATEGORY_BY_OBJECT_KIND.get(binding.filler.object_kind)
                for binding in fact.bindings
            )
            if any(category is None for category in category_by_binding):
                # Unsupported ontology kinds stay represented in the parent
                # connector; they are not silently coerced to Concept slots.
                continue
            ordinal_by_binding: dict[int, int] = {}
            next_ordinal: dict[int, int] = {}
            for binding_index, category in enumerate(category_by_binding):
                assert category is not None
                ordinal = next_ordinal.get(category, 0) + 1
                next_ordinal[category] = ordinal
                ordinal_by_binding[binding_index] = ordinal

            dynamic_cursor = 0
            structural_parts = []
            categories = []
            for part in observation[3]:
                if part[0] == RESPONSE_PART_LITERAL:
                    if (len(part) != 2 or type(part[1]) is not tuple
                            or not part[1]):
                        raise ValueError("response literal part 非法")
                    structural_parts.append(part)
                    continue
                if (len(part) != 3
                        or dynamic_cursor >= len(mapping)):
                    raise ValueError("response dynamic part 非法")
                binding_index = mapping[dynamic_cursor]
                dynamic_cursor += 1
                category = category_by_binding[binding_index]
                assert category is not None
                categories.append(category)
                structural_parts.append((
                    RESPONSE_PART_GRAPH_ROLE,
                    category,
                    ordinal_by_binding[binding_index],
                ))
            if dynamic_cursor != len(mapping):
                raise ValueError("response dynamic mapping 未消费完整")
            parts = _coalesced_parts(tuple(structural_parts))
            style = 920000 + observation[0]
            result.append(CompositionalResponseVariant(
                observation[0],
                observation[1],
                fact.proposition.stable_key(),
                (GENERIC_RESPONSE_CONDITION_VERSION,
                 _feature_mask(tuple(categories)), 0, style),
                parts,
            ))
    variants = tuple(sorted(result, key=lambda item: item.observation_ordinal))
    if len(variants) < 2:
        raise ValueError("可编译 UNKNOWN 多槽观察不足以形成 development/heldout")
    if len({item.semantic_key() for item in variants}) != len(variants):
        raise ValueError("组合响应 semantic variant 重复")
    return variants


def _course_payload(variants: tuple[CompositionalResponseVariant, ...]) -> bytes:
    records = []
    for ordinal, variant in enumerate(variants, 1):
        parts = []
        for part in variant.parts:
            if part[0] == RESPONSE_PART_LITERAL:
                parts.append([RESPONSE_PART_LITERAL, list(part[1])])
            else:
                parts.append([
                    RESPONSE_PART_GRAPH_ROLE, part[1], part[2]])
        records.append([ordinal, list(variant.condition), parts])
    return _canonical([STRUCTURAL_RESPONSE_COURSE_VERSION, 3, records])


def _manifest(
        payload: bytes,
        split: str,
        variants: tuple[CompositionalResponseVariant, ...],
        *,
        source_payload_sha256: str,
        source_manifest_sha256: str,
        compiled_database_sha256: str,
        source_parent_database_sha256: object,
        license_ids: tuple[str, ...],
        ) -> bytes:
    categories = sorted({
        part[1] for variant in variants for part in variant.parts
        if part[0] == RESPONSE_PART_GRAPH_ROLE
    })
    graph_slot_count = sum(
        part[0] == RESPONSE_PART_GRAPH_ROLE
        for variant in variants for part in variant.parts
    )
    proposition_commitment = _sha256(_canonical([
        list(item.proposition_key) for item in variants
    ]))
    literal_references = tuple(
        part[1] for variant in variants for part in variant.parts
        if part[0] == RESPONSE_PART_LITERAL
    )
    value = {
        "attribution": (
            "Project-owned response-structure annotations compiled against "
            "licensed parent graph roles; no source prose answer table."
        ),
        "course_sha256": _sha256(payload),
        "compiled_database_sha256": compiled_database_sha256,
        "format": OUTPUT_FORMAT,
        "free_dialogue_claim": 0,
        "graph_role_categories": categories,
        "graph_slot_count": graph_slot_count,
        "license_ids": list(license_ids),
        "literal_reference_count": len(literal_references),
        "literal_graph_object_count": len(set(literal_references)),
        "literal_graph_object_duplicate_count": 0,
        "minimum_graph_slots_per_variant": min(
            sum(part[0] == RESPONSE_PART_GRAPH_ROLE for part in item.parts)
            for item in variants
        ),
        "multi_graph_slot_variant_count": len(variants),
        "official_url": "https://opensource.org/license/mit",
        "observation_ordinals": [
            item.observation_ordinal for item in variants
        ],
        "parent_proposition_keys": [
            list(item.proposition_key) for item in variants
        ],
        "parent_proposition_commitment_sha256": proposition_commitment,
        "relation_qualified_graph_slots": 1,
        "rights_basis": (
            "Structural projection of already licensed response annotations; "
            "dynamic content remains supplied by the same QueryState graph."
        ),
        "schema_version": 2,
        "selection_rule": (
            "A completed three-graph QueryState must expose one closed relation "
            "whose ordered roles uniquely cover every graph slot."
        ),
        "source_id": "relation-compositional-response-course-20260919a-" + split,
        "source_manifest_sha256": source_manifest_sha256,
        "source_parent_database_sha256": source_parent_database_sha256,
        "source_observation_sha256": source_payload_sha256,
        "source_kind": "project_owned_structural_projection",
        "source_version": 1,
        "split": split,
        "unknown_stance_only": 1,
        "variant_count": len(variants),
        "whole_sentence_representation_count": 0,
    }
    return _canonical(value)


def compile_relation_compositional_course(
        *,
        source_database: str | Path,
        observations: str | Path,
        source_manifest: str | Path,
        output_root: str | Path,
        ) -> dict[str, object]:
    """编译并写出新的 K 盘 development/heldout 多槽课程。"""
    database = Path(source_database).resolve(strict=True)
    observation_path = Path(observations).resolve(strict=True)
    manifest_path = Path(source_manifest).resolve(strict=True)
    root = Path(output_root).resolve()
    if (database.drive.upper() != "K:"
            or root.drive.upper() != "K:"
            or root == Path(root.anchor)):
        raise ValueError("组合响应课程必须显式使用 K 盘模型和新 run root")
    if root.exists():
        raise FileExistsError(root)
    observation_payload = observation_path.read_bytes()
    source_manifest_payload = manifest_path.read_bytes()
    source = json.loads(source_manifest_payload.decode("utf-8-sig"))
    license_ids = source.get("license_ids")
    if (source.get("format") != SOURCE_FORMAT
            or source.get("source_sha256") != _sha256(observation_payload)
            or source.get("free_dialogue_claim") != 0
            or not isinstance(license_ids, list) or not license_ids
            or any(not isinstance(item, str) or not item
                   for item in license_ids)):
        raise ValueError("response observation 来源、许可或 SHA 未闭合")
    variants = _compile_variants(database, observation_payload, source)
    # 预先冻结的稳定次序交错分片；不读取运行结果或可答性重选。
    development = tuple(variants[::2])
    heldout = tuple(variants[1::2])
    if not development or not heldout:
        raise ValueError("组合响应 development/heldout 分片为空")
    literals = {
        split: tuple(
            part[1] for variant in items for part in variant.parts
            if part[0] == RESPONSE_PART_LITERAL
        )
        for split, items in (
            ("development", development), ("heldout", heldout))
    }
    if set(literals["development"]) & set(literals["heldout"]):
        raise ValueError("development/heldout literal chunk 泄漏")

    root.mkdir(parents=True)
    result: dict[str, object] = {
        "root": str(root),
        "integer_payload": 1,
        "source_variant_count": len(variants),
        "whole_sentence_representation_count": 0,
    }
    source_sha = _sha256(observation_payload)
    manifest_sha = _sha256(source_manifest_payload)
    database_sha = _sha256(database.read_bytes())
    for split, items in (
            ("development", development), ("heldout", heldout)):
        payload = _course_payload(items)
        # 用生产 decoder 反向核验输出协议；不建立旁路解释器。
        decoded = course_from_integer_payload(payload)
        if len(decoded.variants) != len(items):
            raise ValueError("组合响应输出不能由生产 decoder 完整恢复")
        ledger = _manifest(
            payload,
            split,
            items,
            source_payload_sha256=source_sha,
            source_manifest_sha256=manifest_sha,
            compiled_database_sha256=database_sha,
            source_parent_database_sha256=source.get("parent_database_sha256"),
            license_ids=tuple(license_ids),
        )
        (root / f"{split}_response_course.int.json").write_bytes(payload)
        (root / f"{split}_manifest.json").write_bytes(ledger)
        result[f"{split}_variant_count"] = len(items)
        result[f"{split}_course_sha256"] = _sha256(payload)
        result[f"{split}_manifest_sha256"] = _sha256(ledger)
        result[f"{split}_graph_slot_count"] = sum(
            part[0] == RESPONSE_PART_GRAPH_ROLE
            for item in items for part in item.parts
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-database", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(
        compile_relation_compositional_course(**vars(args)),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CompositionalResponseVariant",
    "compile_relation_compositional_course",
    "main",
]
