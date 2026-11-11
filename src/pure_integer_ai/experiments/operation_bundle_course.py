"""把有来源的共同课程编为请求、程序和输出义务；不提供语言或答案常量。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.cognition.shared.integer_operation_output import OperationOutputContract, OperationOutputPath
from pure_integer_ai.cognition.shared.integer_operation_program import (
    IntegerOperationProgram, OperationAtom, OperationPathQuery, OperationProjection,
    ORDER_DECLARED, ORDER_DIRECTED_CHAIN,
)
from pure_integer_ai.experiments.integer_relation_graph_course import integer_relation_graph_course_to_w06_adapter
from pure_integer_ai.experiments.natural_relation_preflight import write_preflight_json
from pure_integer_ai.experiments.operation_output_course import compile_train_operation_output, materialize_operation_output_graph
from pure_integer_ai.experiments.operation_program_course import compile_train_operation_program
from pure_integer_ai.experiments.operation_request_course import compile_train_operation_request, materialize_operation_request_graph
from pure_integer_ai.experiments.operation_sequence_graph import SEQUENCE_KIND
from pure_integer_ai.experiments.ph2_authored_relation_compile import authored_relation_identity, authored_relation_role_identity
from pure_integer_ai.experiments.ph2_dataset_contract import parse_canonical_json_bytes
from pure_integer_ai.experiments.ph2_w06_source_semantic import W06_RELATION_PROFILES
from pure_integer_ai.experiments.relation_composition_course import compile_train_relation_composition
from pure_integer_ai.experiments.trained_operation_program_graph import materialize_operation_program_graph


def compile_operation_bundle(root, facts, draft_path):
    """只在显式 K 根物化教师声明；实体只用于编译 reader，运行时不读取课程。"""
    root, source = Path(root).resolve(strict=True), Path(draft_path).resolve(strict=True)
    if root.drive.upper() != "K:" or not source.is_relative_to(root):
        raise ValueError("operation bundle requires an explicit K source root")
    raw = source.read_bytes()
    if len(raw) > 4 * 1024 * 1024 or not raw.endswith(b"\n"):
        raise ValueError("operation bundle needs one bounded physical record")
    draft = parse_canonical_json_bytes(raw[:-1], require_object=True)
    if (set(draft) != {"owner", "split", "license_id", "sequence_rule_id", "operations"}
            or (draft["owner"], draft["split"], draft["license_id"]) != ("teacher", "train", "CC0-1.0")
            or type(draft["operations"]) is not list or not 1 <= len(draft["operations"]) <= 32):
        raise ValueError("operation bundle lacks a licensed bounded teacher")
    permission = {name: draft[name] for name in ("owner", "split", "license_id")}
    adapter = integer_relation_graph_course_to_w06_adapter(facts.course, require_full_curriculum=False)
    schemas = {schema.relation.stable_key(): schema for schema in adapter.schemas}
    programs, frames, outputs, index, named = [], [], [], [], {}
    for number, entry in enumerate(draft["operations"], 1):
        stem = f"instructions/bundle-{number:02d}"
        atoms, role_orders = [], []
        for declaration in entry["atoms"]:
            profile = W06_RELATION_PROFILES[declaration["relation"]]
            schema = schemas[authored_relation_identity(profile.relation_kind).stable_key()]
            roles = tuple(authored_relation_role_identity(code) for code, _ in profile.role_object_kinds)
            if len(roles) != len(declaration["variables"]):
                raise ValueError("operation atom omits an actual relation role")
            atoms.append(OperationAtom(schema, tuple(sorted(zip(roles, declaration["variables"]),
                key=lambda pair: pair[0].stable_key()))))
            role_orders.append(roles)

        def edge(row):
            """课程坐标只引用当前 atom 的实际角色，不按 id 猜关系方向。"""
            coordinate, left, right = row
            return coordinate, role_orders[coordinate - 1][left - 1], role_orders[coordinate - 1][right - 1]

        path = (None if not entry.get("path_edges") else
            OperationPathQuery(*entry["external"], tuple(edge(row) for row in entry["path_edges"])))
        order = entry.get("order_roles", ())
        projections = tuple(OperationProjection(coordinate, tuple(slot.role for slot in atoms[coordinate - 1].schema.slots))
                            for coordinate in entry["projections"])
        program = IntegerOperationProgram(tuple(entry["operation_key"]), tuple(entry["external"]), tuple(atoms),
            tuple(tuple(row) for row in entry.get("guards", ())), projections,
            entry.get("minimum_matches", 1), entry.get("maximum_matches", 1),
            ORDER_DIRECTED_CHAIN if path is not None or order else ORDER_DECLARED,
            () if not order else (role_orders[order[0] - 1][order[1] - 1], role_orders[order[0] - 1][order[2] - 1]),
            (SEQUENCE_KIND, entry.get("sequence_rule_id", draft["sequence_rule_id"])), path)
        write_preflight_json(root, stem + ".program.train.json", {**permission, "program_key": list(program.stable_key())})
        witness = compile_train_operation_program(root / (stem + ".program.train.json"), facts)
        programs.append(witness)
        if entry["name"] in named:
            raise ValueError("bundle operation name is duplicated")
        named[entry["name"]] = witness
        rules = []
        output_paths = []
        for path_number, declaration in enumerate(entry.get("output_paths", ()), 1):
            rule_key = ()
            if "rule" in declaration:
                rule_source = stem + f".rule-{path_number:02d}.train.json"
                instruction = []
                for relation, left, right in declaration["rule"]["roles"]:
                    profile = W06_RELATION_PROFILES[relation]
                    instruction.extend((profile.relation_kind, profile.role_object_kinds[left - 1][0],
                                        profile.role_object_kinds[right - 1][0]))
                write_preflight_json(root, rule_source, {**permission,
                    "instruction": instruction, "rule_id": declaration["rule"]["rule_id"]})
                rule = compile_train_relation_composition(root / rule_source, facts)
                rules.append(rule)
                rule_key = rule.stable_key()
            output_paths.append(OperationOutputPath(tuple(edge(row) for row in declaration["edges"]),
                rule_key, declaration.get("result_atom", 0)))
        preserved = entry.get("preserved_operation")
        contract = OperationOutputContract(witness.stable_key(), tuple(sorted(entry["projections"])),
            tuple(output_paths), tuple(entry.get("comparison_guards", ())), tuple(entry.get("scope_variables", ())),
            () if not preserved else named[preserved].stable_key(), 1)
        write_preflight_json(root, stem + ".output.train.json", {**permission, "contract_key": list(contract.stable_key())})
        outputs.append(compile_train_operation_output(root / (stem + ".output.train.json"), tuple(programs), rules=tuple(rules)))
        request_sources = []
        for variant, question in enumerate(entry["questions"], 1):
            request_source = stem + f".request-{variant:02d}.train.json"
            readers = [{"variable": variable, "seed_id": seed,
                "role_key": list(authored_relation_role_identity(W06_RELATION_PROFILES[relation].role_object_kinds[role - 1][0]).stable_key())}
                for variable, seed, relation, role in entry["readers"]]
            write_preflight_json(root, request_source, {**permission, "question": question, "readers": readers})
            frames.append(compile_train_operation_request(root / request_source, facts, witness))
            request_sources.append(request_source)
        index.append({"name": entry["name"], "stem": stem, "operation_key": entry["operation_key"],
                      "request_sources": request_sources})
    materialize_operation_program_graph(tuple(programs), root / "programs.sqlite3")
    materialize_operation_request_graph(tuple(frames), root / "requests.sqlite3")
    materialize_operation_output_graph(tuple(outputs), tuple(programs), root / "outputs.sqlite3")
    write_preflight_json(root, "instruction-index.json", {"operations": index})
    return tuple(programs), tuple(frames), tuple(outputs)
