"""可复用单 claim 序列语法的一等整数规则及其物理成员图。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import ObjectIdentity, concept_identity
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record, unpack_record
from pure_integer_ai.experiments.operation_composition_course import OperationCompositionWitness
from pure_integer_ai.experiments.trained_relation_question_graph import _hash
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import TYPE_INT, SQLiteBackend, register_extension_table


SEQUENCE_KIND = 92500
MEMBERS = 92501
MEMBER = 92502
SEQUENCE_TRACE = 92503
SEQUENCE_STEP = 92504
TABLE = "native_claim_sequence"
PARTS = "native_claim_sequence_part"


def _integer_key(key):
    """规则身份只接受非空的非负整数序列。"""
    if type(key) is not tuple or not key or any(type(value) is not int or value < 0 for value in key):
        raise ValueError("native sequence requires complete integer identities")


@dataclass(frozen=True, slots=True)
class ClaimSequenceRule:
    """body* terminal 的有界许可；成员保存完整原生 witness，不以哈希代替。"""

    rule_id: int
    minimum_claims: int
    maximum_claims: int
    source_digest: tuple[int, ...]
    source_identity: tuple[int, ...]
    license_identity: tuple[int, ...]
    connector: ObjectIdentity
    # predicate identity, body native witness, terminal native witness
    members: tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...]

    def __post_init__(self):
        """所有边界、规则载体和完整成员在构造时即固定为规范整数记录。"""
        if (type(self.rule_id) is not int or self.rule_id <= 0
                or type(self.minimum_claims) is not int or type(self.maximum_claims) is not int
                or not 1 <= self.minimum_claims <= self.maximum_claims <= 8
                or type(self.source_digest) is not tuple or len(self.source_digest) != 32
                or any(type(value) is not int or not 0 <= value <= 255 for value in self.source_digest)
                or not isinstance(self.connector, ObjectIdentity)
                or type(self.members) is not tuple or not 1 <= len(self.members) <= 8):
            raise ValueError("native sequence has incomplete identity or bounded grammar")
        _integer_key(self.source_identity)
        _integer_key(self.license_identity)
        if self.connector != concept_identity((SEQUENCE_KIND, self.rule_id, *self.source_digest),
                owner=self.connector.owner, versions=self.connector.versions):
            raise ValueError("native sequence connector is not its complete rule carrier")
        predicates, variants = [], []
        for row in self.members:
            if type(row) is not tuple or len(row) != 3:
                raise ValueError("native sequence member is incomplete")
            predicate, body, terminal = row
            _integer_key(predicate)
            predicates.append(predicate)
            for key in (body, terminal):
                _integer_key(key)
                witness = OperationCompositionWitness.from_stable_key(key)
                variants.append(witness.variant_id)
        if (tuple(predicates) != tuple(sorted(set(predicates)))
                or len(set(variants)) != len(variants)):
            raise ValueError("native sequence members are duplicated or noncanonical")

    def stable_key(self):
        """完整编码规则、来源和每一个原生语法成员。"""
        return pack_record(SEQUENCE_KIND, (self.rule_id, self.minimum_claims, self.maximum_claims),
            self.source_digest, self.source_identity, self.license_identity, self.connector.stable_key(),
            pack_record(MEMBERS, *(pack_record(MEMBER, *row) for row in self.members)))

    @classmethod
    def from_stable_key(cls, key):
        """按冻结字段恢复，拒绝不规范记录或缺失成员。"""
        fields = unpack_record(key, SEQUENCE_KIND)
        if len(fields) != 6 or len(fields[0]) != 3:
            raise ValueError("native sequence record is incomplete")
        members = tuple(unpack_record(row, MEMBER) for row in unpack_record(fields[5], MEMBERS))
        result = cls(*fields[0], fields[1], fields[2], fields[3],
                     ObjectIdentity.from_stable_key(fields[4]), members)
        if result.stable_key() != key:
            raise ValueError("native sequence record is not canonical")
        return result

    def selected_members(self, predicates):
        """仅按当前有序谓词选择 body/terminal，不记忆完整谓词序列。"""
        by_predicate = {predicate: (body, terminal) for predicate, body, terminal in self.members}
        if (not self.minimum_claims <= len(predicates) <= self.maximum_claims
                or any(predicate not in by_predicate for predicate in predicates)):
            raise ValueError("current claims are outside the learned native sequence grammar")
        return tuple(by_predicate[predicate][int(index == len(predicates) - 1)]
                     for index, predicate in enumerate(predicates))


def _register(backend):
    """注册 append-only 的规则本体及完整整数成员序列。"""
    register_extension_table(backend, TABLE, [("rule_id", TYPE_INT), ("rule_hash", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY, indexes=[("rule_id",)], recovery_key=("rule_id",))
    register_extension_table(backend, PARTS,
        [("rule_id", TYPE_INT), ("part_ordinal", TYPE_INT), ("part_value", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY, indexes=[("rule_id", "part_ordinal")],
        recovery_key=("rule_id", "part_ordinal"))


def materialize_claim_sequence_rules(rules, database):
    """独占写入新 K 图；权威记录保留全部成员，索引可删除重建。"""
    target = Path(database).resolve()
    if (target.drive.upper() != "K:" or target.exists() or type(rules) is not tuple or not rules
            or any(not isinstance(rule, ClaimSequenceRule) for rule in rules)
            or len({rule.rule_id for rule in rules}) != len(rules)):
        raise ValueError("native sequence requires new K graph and unique complete rules")
    backend = SQLiteBackend(str(target))
    try:
        _register(backend)
        for rule in sorted(rules, key=lambda item: item.rule_id):
            key = rule.stable_key()
            backend.insert(TABLE, {"rule_id": rule.rule_id, "rule_hash": _hash(key)})
            backend.insert_many(PARTS, [{"rule_id": rule.rule_id, "part_ordinal": ordinal,
                "part_value": value} for ordinal, value in enumerate(key)])
        backend.commit()
    finally:
        backend.close()


def read_claim_sequence_rules(database):
    """冷恢复完整物理规则，孤儿、缺序和身份漂移均拒绝。"""
    backend = SQLiteBackend(str(Path(database).resolve(strict=True)), read_only=True)
    try:
        _register(backend)
        rows = backend.select(TABLE, order_by="rule_id")
        parts = backend.select(PARTS, order_by="rule_id, part_ordinal")
        if not rows or any(part["rule_id"] not in {row["rule_id"] for row in rows} for part in parts):
            raise ValueError("native sequence graph has missing or orphan rules")
        rules = []
        for row in rows:
            members = tuple(part for part in parts if part["rule_id"] == row["rule_id"])
            if tuple(part["part_ordinal"] for part in members) != tuple(range(len(members))):
                raise ValueError("native sequence record is incomplete")
            key = tuple(part["part_value"] for part in members)
            rule = ClaimSequenceRule.from_stable_key(key)
            if rule.rule_id != row["rule_id"] or _hash(key) != row["rule_hash"]:
                raise ValueError("native sequence identity drifted")
            rules.append(rule)
        return tuple(rules)
    finally:
        backend.close()
