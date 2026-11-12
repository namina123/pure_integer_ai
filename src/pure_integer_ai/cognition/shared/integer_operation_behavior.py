"""当前整数程序的输入/选择义务与语义编辑记录；不提供事实或语言词表。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.integer_operation_program import (
    IntegerOperationProgram, OperationProgramWitness, SEQUENCE, integer_key,
)
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record, unpack_record


OBLIGATION = 92340
SEMANTIC_EDIT = 92341
OUTPUT_BEHAVIOR_EDIT = 92342
EDIT_ATOMS = 1
EDIT_GUARDS = 2
EDIT_MATCH_BOUNDS = 3
EDIT_ORDER = 4
EDIT_PROJECTIONS = 5
EDIT_REALIZATION = 6
EDIT_OUTPUT_CONTRACT = 7


def semantic_selection_obligation(program: IntegerOperationProgram) -> tuple[int, ...]:
    """保存全部输入、匹配和顺序义务；标签、投影和表面组织不定义选择行为。"""
    if not isinstance(program, IntegerOperationProgram):
        raise TypeError("semantic obligation requires an integer program")
    return pack_record(OBLIGATION, program.external_variables,
        pack_record(SEQUENCE, *(atom.stable_key() for atom in program.atoms)),
        pack_record(SEQUENCE, *(pack_record(EDIT_GUARDS, guard) for guard in program.guards)),
        (program.minimum_matches, program.maximum_matches, program.order_kind),
        pack_record(SEQUENCE, *(role.stable_key() for role in program.order_roles)))


def semantic_edit_fields(before: IntegerOperationProgram,
                         after: IntegerOperationProgram) -> tuple[int, ...]:
    """精确列出实际改变的选择义务；本合同不授权同时改变输入或输出组织。"""
    if (before.operation_key != after.operation_key
            or before.external_variables != after.external_variables
            or before.projections != after.projections
            or before.realization_key != after.realization_key):
        raise ValueError("semantic edit changed frozen input/key/projection/realization")
    fields = tuple(code for code, left, right in (
        (EDIT_ATOMS, before.atoms, after.atoms),
        (EDIT_GUARDS, before.guards, after.guards),
        (EDIT_MATCH_BOUNDS, (before.minimum_matches, before.maximum_matches),
                           (after.minimum_matches, after.maximum_matches)),
        (EDIT_ORDER, (before.order_kind, before.order_roles),
                    (after.order_kind, after.order_roles)),
    ) if left != right)
    if not fields:
        raise ValueError("semantic edit has no actual selection obligation change")
    return fields


def _digest(value: tuple[int, ...]) -> None:
    """摘要只按固定32字节向量验证，不以宿主字符串或整数对象身份定义协议。"""
    if (type(value) is not tuple or len(value) != 32
            or any(type(item) is not int or not 0 <= item <= 255 for item in value)):
        raise ValueError("semantic edit requires a canonical SHA-256 byte vector")


@dataclass(frozen=True, slots=True)
class OperationSemanticEditReceipt:
    """完整前后本体及独立语境、输入和物理图身份；记录本身不自证能力。"""

    before: OperationProgramWitness
    after: OperationProgramWitness
    context_identity: tuple[int, ...]
    input_structure_key: tuple[int, ...]
    before_graph_sha256: tuple[int, ...]
    after_graph_sha256: tuple[int, ...]
    changed_fields: tuple[int, ...]

    def __post_init__(self) -> None:
        """只允许冻结范围内的完整语义编辑；来源和许可必须保持同域。"""
        if not isinstance(self.before, OperationProgramWitness) or not isinstance(self.after, OperationProgramWitness):
            raise TypeError("semantic edit requires full source-qualified programs")
        for value in (self.context_identity, self.before_graph_sha256, self.after_graph_sha256):
            _digest(value)
        integer_key(self.input_structure_key)
        if (self.before.source_identity != self.after.source_identity
                or self.before.license_identity != self.after.license_identity
                or self.before.authorized_forming_sources != self.after.authorized_forming_sources
                or self.before.source_extensions != self.after.source_extensions):
            raise ValueError("semantic edit changes current source/license/forming permission")
        if (type(self.changed_fields) is not tuple
                or any(type(code) is not int for code in self.changed_fields)
                or self.changed_fields != semantic_edit_fields(self.before.program, self.after.program)):
            raise ValueError("semantic edit fields do not describe the actual complete delta")
        if self.before_graph_sha256 == self.after_graph_sha256:
            raise ValueError("semantic edit physical graph identities are identical")

    def stable_key(self) -> tuple[int, ...]:
        """输出固定字段序和长度前缀，完整本体可独立冷解码。"""
        return pack_record(SEMANTIC_EDIT, self.before.stable_key(), self.after.stable_key(),
            self.context_identity, self.input_structure_key, self.before_graph_sha256,
            self.after_graph_sha256, self.changed_fields)

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> OperationSemanticEditReceipt:
        """拒绝截断、附加字段和不规范整数，不由摘要重建丢失本体。"""
        fields = unpack_record(integer_key(key), SEMANTIC_EDIT)
        if len(fields) != 7:
            raise ValueError("semantic edit receipt is incomplete")
        result = cls(OperationProgramWitness.from_stable_key(fields[0]),
            OperationProgramWitness.from_stable_key(fields[1]), *fields[2:])
        if result.stable_key() != key:
            raise ValueError("semantic edit receipt is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class OperationOutputBehaviorEditReceipt:
    """Physical projection-scope or preservation edit over unchanged selection semantics."""

    before: OperationProgramWitness
    after: OperationProgramWitness
    before_output: object
    after_output: object | None
    context_identity: tuple[int, ...]
    before_input_structure_key: tuple[int, ...]
    after_input_structure_key: tuple[int, ...]
    input_surface_sha256: tuple[int, ...]
    before_program_graph_sha256: tuple[int, ...]
    after_program_graph_sha256: tuple[int, ...]
    before_output_graph_sha256: tuple[int, ...]
    after_output_graph_sha256: tuple[int, ...]
    changed_fields: tuple[int, ...]

    def __post_init__(self) -> None:
        from pure_integer_ai.cognition.shared.integer_operation_output import OperationOutputWitness

        if (not isinstance(self.before, OperationProgramWitness)
                or not isinstance(self.after, OperationProgramWitness)
                or not isinstance(self.before_output, OperationOutputWitness)
                or (self.after_output is not None
                    and not isinstance(self.after_output, OperationOutputWitness))):
            raise TypeError("output behavior edit requires complete program and output witnesses")
        for value in (self.context_identity, self.input_surface_sha256,
                      self.before_program_graph_sha256, self.after_program_graph_sha256,
                      self.before_output_graph_sha256, self.after_output_graph_sha256):
            _digest(value)
        integer_key(self.before_input_structure_key)
        integer_key(self.after_input_structure_key)
        before, after = self.before.program, self.after.program
        before_contract = self.before_output.contract
        after_contract = None if self.after_output is None else self.after_output.contract
        if (self.before.source_identity != self.after.source_identity
                or self.before.license_identity != self.after.license_identity
                or self.before.authorized_forming_sources != self.after.authorized_forming_sources
                or self.before.source_extensions != self.after.source_extensions
                or semantic_selection_obligation(before) != semantic_selection_obligation(after)
                or before_contract.program_witness_key != self.before.stable_key()
                or (after_contract is not None
                    and after_contract.program_witness_key != self.after.stable_key())
                or self.before_program_graph_sha256 == self.after_program_graph_sha256
                or self.before_output_graph_sha256 == self.after_output_graph_sha256):
            raise ValueError("output behavior edit changes selection permission or lacks physical graph deltas")

        changed = tuple(code for code, left, right in (
            (EDIT_PROJECTIONS, before.projections, after.projections),
            (EDIT_REALIZATION, before.realization_key, after.realization_key),
            (EDIT_OUTPUT_CONTRACT, before_contract.stable_key(),
             () if after_contract is None else after_contract.stable_key()),
        ) if left != right)
        if type(self.changed_fields) is not tuple or self.changed_fields != changed:
            raise ValueError("output behavior edit fields do not describe the complete delta")

        before_projection_set = frozenset((row.atom, row.roles) for row in before.projections)
        after_projection_set = frozenset((row.atom, row.roles) for row in after.projections)
        if (before.operation_key == after.operation_key
                and before.realization_key == after.realization_key
                and before_projection_set > after_projection_set
                and self.before_input_structure_key == self.after_input_structure_key
                and before_contract.preserved_witness_key == ()
                and after_contract is None
                and before_contract.scope_variables
                and before_contract.projected_atoms == tuple(range(1, len(before.atoms) + 1))):
            return

        if after_contract is None:
            raise ValueError("only an incomplete output scope may remove output permission")
        before_contract_without_program = (
            before_contract.projected_atoms, before_contract.paths, before_contract.comparison_guards,
            before_contract.scope_variables, before_contract.current_request_required)
        after_contract_without_program = (
            after_contract.projected_atoms, after_contract.paths, after_contract.comparison_guards,
            after_contract.scope_variables, after_contract.current_request_required)
        if (before_projection_set == after_projection_set
                and before.projections != after.projections
                and before.realization_key != after.realization_key
                and self.input_surface_sha256
                and before_contract.preserved_witness_key == ()
                and after_contract.preserved_witness_key == self.before.stable_key()
                and before_contract_without_program == (
                    tuple(sorted(row.atom for row in before.projections)), before_contract.paths,
                    before_contract.comparison_guards, before_contract.scope_variables,
                    before_contract.current_request_required)
                and after_contract_without_program == (
                    tuple(sorted(row.atom for row in after.projections)), before_contract.paths,
                    before_contract.comparison_guards, before_contract.scope_variables,
                    before_contract.current_request_required)):
            return
        raise ValueError("output edit is neither a fail-closed scope reduction nor a preserved rewrite")

    def stable_key(self) -> tuple[int, ...]:
        return pack_record(OUTPUT_BEHAVIOR_EDIT, self.before.stable_key(), self.after.stable_key(),
            self.before_output.stable_key(),
            () if self.after_output is None else self.after_output.stable_key(), self.context_identity,
            self.before_input_structure_key, self.after_input_structure_key, self.input_surface_sha256,
            self.before_program_graph_sha256, self.after_program_graph_sha256,
            self.before_output_graph_sha256, self.after_output_graph_sha256, self.changed_fields)

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> OperationOutputBehaviorEditReceipt:
        from pure_integer_ai.cognition.shared.integer_operation_output import OperationOutputWitness

        fields = unpack_record(integer_key(key), OUTPUT_BEHAVIOR_EDIT)
        if len(fields) != 13:
            raise ValueError("output behavior edit receipt is incomplete")
        result = cls(OperationProgramWitness.from_stable_key(fields[0]),
            OperationProgramWitness.from_stable_key(fields[1]),
            OperationOutputWitness.from_stable_key(fields[2]),
            None if not fields[3] else OperationOutputWitness.from_stable_key(fields[3]), *fields[4:])
        if result.stable_key() != key:
            raise ValueError("output behavior edit receipt is not canonical")
        return result
