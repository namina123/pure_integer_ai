"""One development bundle for eight actual output duties; no family routing."""
from __future__ import annotations

from dataclasses import dataclass, replace

from pure_integer_ai.cognition.shared.integer_operation_output import OperationOutputContract, OperationOutputPath
from pure_integer_ai.cognition.shared.integer_operation_program import (
    UNEQUAL, ORDER_DIRECTED_CHAIN, IntegerOperationProgram, OperationAtom, OperationProjection,
)
from pure_integer_ai.experiments.integer_relation_graph_course import integer_relation_graph_course_to_w06_adapter
from pure_integer_ai.experiments.natural_relation_preflight import write_preflight_json
from pure_integer_ai.experiments.operation_output_course import compile_train_operation_output, materialize_operation_output_graph
from pure_integer_ai.experiments.operation_program_course import compile_train_operation_program
from pure_integer_ai.experiments.operation_request_course import compile_train_operation_request, materialize_operation_request_graph
from pure_integer_ai.experiments.ph2_authored_relation_compile import authored_relation_role_identity
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    RELATION_MEMBER, RELATION_SUBSET, ROLE_MEMBER_ELEMENT, ROLE_MEMBER_SET, ROLE_SUBSET_CHILD, ROLE_SUBSET_PARENT,
    ROLE_CAUSE, ROLE_EFFECT, ROLE_EVENT_BEFORE_SUBJECT, ROLE_EVENT_BEFORE_OBJECT,
    ROLE_PROPERTY_SUBJECT, ROLE_PROPERTY_ATTRIBUTE, ROLE_PROPERTY_VALUE, ROLE_PROPERTY_POLARITY,
    ROLE_PROPERTY_MODALITY, ROLE_PROPERTY_INTENSITY, ROLE_REFERS_FROM, ROLE_REFERS_TO,
    ROLE_SIMILAR_LEFT, ROLE_SIMILAR_RIGHT,
)
from pure_integer_ai.experiments.relation_composition_course import compile_train_relation_composition
from pure_integer_ai.experiments.trained_multi_claim_composer import GRAPH_KIND
from pure_integer_ai.experiments.trained_operation_program_graph import materialize_operation_program_graph


@dataclass(frozen=True, slots=True)
class DevelopmentOutputCase:
    family: str  # Audit-only title; never supplied to the production query.
    seed_ids: tuple[str, ...]
    witness: object
    output_witness: object
    frame: object
    question: str
    variant_id: int
    request_instruction_paths: tuple[str, ...]


def compile_development_output_course(root, facts, *, question_variants=None):
    adapter = integer_relation_graph_course_to_w06_adapter(facts.course, require_full_curriculum=False)
    question_variants = {} if question_variants is None else dict(question_variants)
    if any(type(name) is not str or not name or type(values) is not tuple or not values
           or any(type(value) is not str or not value for value in values)
           or len(set(values)) != len(values)
           for name, values in question_variants.items()):
        raise ValueError("operation request variants must be unique nonempty licensed surfaces")
    records = {"".join(map(chr, record.teacher_seed_values)): record for record in facts.course.records}
    schemas = {schema.relation.stable_key(): schema for schema in adapter.schemas}
    role = authored_relation_role_identity
    def atom(seed, pairs):
        return OperationAtom(schemas[records[seed].predicate_key],
            tuple(sorted(((role(code), variable) for code, variable in pairs), key=lambda row: row[0].stable_key())))
    member = lambda variables: atom("member-a", tuple(zip((ROLE_MEMBER_ELEMENT, ROLE_MEMBER_SET), variables)))
    sub = lambda variables: atom("subset", tuple(zip((ROLE_SUBSET_CHILD, ROLE_SUBSET_PARENT), variables)))
    cause = lambda variables: atom("cause-a", tuple(zip((ROLE_CAUSE, ROLE_EFFECT), variables)))
    time = lambda variables: atom("time-a", tuple(zip((ROLE_EVENT_BEFORE_SUBJECT, ROLE_EVENT_BEFORE_OBJECT), variables)))
    property_roles = (ROLE_PROPERTY_SUBJECT, ROLE_PROPERTY_ATTRIBUTE, ROLE_PROPERTY_VALUE,
                      ROLE_PROPERTY_POLARITY, ROLE_PROPERTY_MODALITY, ROLE_PROPERTY_INTENSITY)
    prop = lambda variables: atom("property-a", tuple(zip(property_roles, variables)))
    refer = lambda variables: atom("reference", tuple(zip((ROLE_REFERS_FROM, ROLE_REFERS_TO), variables)))
    similar = lambda variables: atom("similar", tuple(zip((ROLE_SIMILAR_LEFT, ROLE_SIMILAR_RIGHT), variables)))
    # Each case explicitly binds a current request; no answer or example filler
    # survives in the resulting input/program/output instruction graphs.
    specifications = (
        ("direct", ("member-a",), (member((1, 2)),), (1, 2), (), (1,),
         "请回答澄石乙是否属于红枝类，并列出完整关系。", ((1, "member-a", ROLE_MEMBER_ELEMENT), (2, "member-a", ROLE_MEMBER_SET))),
        ("explanation", ("member-a", "subset", "member-b"),
         (member((1, 2)), sub((2, 3)), member((1, 3))), (1, 3), (), (1, 2, 3),
         "解释澄石乙属于蓝叶类的依据，并列出前提和结果。", ((1, "member-a", ROLE_MEMBER_ELEMENT), (3, "member-b", ROLE_MEMBER_SET))),
        ("comparison", ("property-a", "property-b"),
         (prop((1, 3, 4, 5, 6, 7)), prop((2, 3, 8, 5, 6, 7))), (1, 2), ((UNEQUAL, 4, 8),), (1, 2),
         "比较甲灯与乙灯的颜色，列出两者的属性证据。", ((1, "property-a", ROLE_PROPERTY_SUBJECT), (2, "property-b", ROLE_PROPERTY_SUBJECT))),
        ("causal", ("cause-a", "cause-b"), (cause((1, 2)), cause((2, 3))), (1, 3), (), (1, 2),
         "从雨阀开启到浮标抬升，列出有证据的因果路径。", ((1, "cause-a", ROLE_CAUSE), (3, "cause-b", ROLE_EFFECT))),
        ("temporal", ("time-a", "time-b"), (time((1, 2)), time((2, 3))), (1, 3), (), (2, 1),
         "从开闸到合闸，按记录的时间先后列出过程。", ((1, "time-a", ROLE_EVENT_BEFORE_SUBJECT), (3, "time-b", ROLE_EVENT_BEFORE_OBJECT))),
        ("summary", ("member-a", "member-b", "member-c"),
         (member((1, 2)), member((1, 3)), member((1, 4))), (1, 2, 3, 4), (), (1, 2, 3),
         "汇总澄石乙在红枝类、蓝叶类和灰藤类中的成员关系。",
         ((1, "member-a", ROLE_MEMBER_ELEMENT), (2, "member-a", ROLE_MEMBER_SET), (3, "member-b", ROLE_MEMBER_SET), (4, "member-c", ROLE_MEMBER_SET))),
        ("rewrite", ("member-c", "member-a", "member-b"),
         (member((1, 2)), member((1, 3)), member((1, 4))), (1, 2, 3, 4), (), (3, 1, 2),
         "换个顺序重述澄石乙在红枝类、蓝叶类和灰藤类中的成员关系。",
         ((1, "member-a", ROLE_MEMBER_ELEMENT), (2, "member-a", ROLE_MEMBER_SET), (3, "member-b", ROLE_MEMBER_SET), (4, "member-c", ROLE_MEMBER_SET))),
        ("reference", ("reference", "similar"), (refer((1, 2)), similar((2, 3))), (1,), (), (1, 2),
         "湾标甲所指对象与哪个称呼相似？", ((1, "reference", ROLE_REFERS_FROM),)),
    )
    rows, programs, frames, obligations = [], [], [], []
    for number, (family, seeds, atoms, external, guards, projections, question, readers) in enumerate(specifications, 1):
        paths, comparison, scope, preservation = (), (), (), ()
        variant = 0 if number == 1 else number
        kwargs = (dict(order_kind=ORDER_DIRECTED_CHAIN,
                       order_roles=(role(ROLE_EVENT_BEFORE_SUBJECT), role(ROLE_EVENT_BEFORE_OBJECT))) if number == 5 else {})
        program = IntegerOperationProgram((92440, number), external, atoms, guards,
            tuple(OperationProjection(index, tuple(slot.role for slot in atoms[index - 1].schema.slots)) for index in projections),
            realization_key=() if not variant else (GRAPH_KIND, variant), **kwargs)
        stem = f"instructions/{number:02d}"
        write_preflight_json(root, stem + ".program.train.json", {"program_key": list(program.stable_key()),
            "owner": "teacher", "split": "train", "license_id": "CC0-1.0"})
        witness = compile_train_operation_program(root / (stem + ".program.train.json"), facts)
        programs.append(witness)
        rules = ()
        if number == 2:
            write_preflight_json(root, stem + ".rule.train.json", {"instruction": [
                RELATION_MEMBER, ROLE_MEMBER_ELEMENT, ROLE_MEMBER_SET,
                RELATION_SUBSET, ROLE_SUBSET_CHILD, ROLE_SUBSET_PARENT,
                RELATION_MEMBER, ROLE_MEMBER_ELEMENT, ROLE_MEMBER_SET],
                "rule_id": [92441, 1], "owner": "teacher", "split": "train", "license_id": "CC0-1.0"})
            rule = compile_train_relation_composition(root / (stem + ".rule.train.json"), facts)
            paths = (OperationOutputPath(((1, role(ROLE_MEMBER_ELEMENT), role(ROLE_MEMBER_SET)),
                (2, role(ROLE_SUBSET_CHILD), role(ROLE_SUBSET_PARENT))), rule.stable_key(), 3),)
            rules = (rule,)
        if number in {4, 5, 8}:
            roles = {4: (ROLE_CAUSE, ROLE_EFFECT, ROLE_CAUSE, ROLE_EFFECT),
                     5: (ROLE_EVENT_BEFORE_SUBJECT, ROLE_EVENT_BEFORE_OBJECT,
                         ROLE_EVENT_BEFORE_SUBJECT, ROLE_EVENT_BEFORE_OBJECT),
                     8: (ROLE_REFERS_FROM, ROLE_REFERS_TO, ROLE_SIMILAR_LEFT, ROLE_SIMILAR_RIGHT)}[number]
            paths = (OperationOutputPath(((1, role(roles[0]), role(roles[1])),
                                          (2, role(roles[2]), role(roles[3])))),)
        if number == 3:
            comparison = (1,)
        if number in {6, 7}:
            scope = external
        if number == 7:
            preservation = programs[5].stable_key()
        obligation = OperationOutputContract(witness.stable_key(), tuple(sorted(projections)),
            paths, comparison, scope, preservation, int(number == 8))
        write_preflight_json(root, stem + ".output.train.json", {"contract_key": list(obligation.stable_key()),
            "owner": "teacher", "split": "train", "license_id": "CC0-1.0"})
        output = compile_train_operation_output(root / (stem + ".output.train.json"), tuple(programs), rules=rules)
        obligations.append(output)
        surfaces = question_variants.get(family, (question,))
        if any(not isinstance(surface, str) or not surface for surface in surfaces):
            raise ValueError("operation request variants must be nonempty text")
        first_frame = None
        request_instruction_paths = []
        for variant_number, surface in enumerate(surfaces, 1):
            request_stem = (stem + ".request.train.json" if len(surfaces) == 1
                            else stem + f".request-{variant_number:02d}.train.json")
            write_preflight_json(root, request_stem, {"question": surface,
                "readers": [{"variable": variable, "seed_id": seed, "role_key": list(role(code).stable_key())}
                            for variable, seed, code in readers], "owner": "teacher", "split": "train", "license_id": "CC0-1.0"})
            frame = compile_train_operation_request(root / request_stem, facts, witness)
            frames.append(frame)
            request_instruction_paths.append(request_stem)
            if first_frame is None:
                first_frame = frame
        if variant:
            # Temporal output follows actual edge order even though its program
            # declared the projections in reverse; IDs cannot sort events.
            write_preflight_json(root, stem + ".composition.train.json", {
                "claim_keys": [list(records[seed].proposition_key) for seed in seeds],
                "separator_values": [65307], "variant_id": variant, "owner": "teacher", "split": "train", "license_id": "CC0-1.0"})
        rows.append(DevelopmentOutputCase(
            family, seeds, witness, output, first_frame, surfaces[0], variant,
            tuple(request_instruction_paths)))
    materialize_operation_program_graph(tuple(programs), root / "programs.sqlite3")
    materialize_operation_request_graph(tuple(frames), root / "requests.sqlite3")
    materialize_operation_output_graph(tuple(obligations), tuple(programs), root / "outputs.sqlite3")
    materialize_operation_output_graph((), tuple(programs), root / "outputs-removed.sqlite3")
    return tuple(rows)
