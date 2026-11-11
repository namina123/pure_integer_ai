"""把物理训练规则编译成单 claim body/terminal 原生语法及序列规则。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import concept_identity
from pure_integer_ai.cognition.shared.unicode_representation import validate_unicode_scalars
from pure_integer_ai.experiments.operation_composition_course import compile_train_operation_composition
from pure_integer_ai.experiments.operation_sequence_graph import ClaimSequenceRule, SEQUENCE_KIND
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes, parse_canonical_json_bytes


def compile_train_claim_sequence(source_path, derived_root, facts, core, generator):
    """只消费有来源训练样例；派生原生课程保存在同一显式 K 根内。"""
    source = Path(source_path).resolve(strict=True)
    target = Path(derived_root).resolve()
    if (source.drive.upper() != "K:" or target.drive.upper() != "K:"
            or target.exists() or not target.is_relative_to(source.parent)):
        raise ValueError("sequence compilation requires new derived courses inside the explicit K source root")
    raw = source.read_bytes()
    if len(raw) > 1024 * 1024 or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("sequence grammar requires one bounded physical train record")
    row = parse_canonical_json_bytes(raw[:-1], require_object=True)
    fields = {"rule_id", "minimum_claims", "maximum_claims", "members", "separator_values",
              "owner", "split", "license_id"}
    if (set(row) != fields or (row["owner"], row["split"], row["license_id"])
            != ("teacher", "train", "CC0-1.0")
            or type(row["rule_id"]) is not int or row["rule_id"] <= 0
            or type(row["minimum_claims"]) is not int or type(row["maximum_claims"]) is not int
            or not 1 <= row["minimum_claims"] <= row["maximum_claims"] <= 8
            or type(row["members"]) is not list or not 1 <= len(row["members"]) <= 8
            or type(row["separator_values"]) is not list or not 1 <= len(row["separator_values"]) <= 8):
        raise ValueError("sequence grammar lacks licensed bounded training obligations")
    validate_unicode_scalars(tuple(row["separator_values"]))
    records = {record.proposition_key: record for record in facts.course.records}
    selected, variants = [], []
    for member in row["members"]:
        if (type(member) is not dict or set(member)
                != {"claim_key", "output_gaps", "body_variant_id", "terminal_variant_id"}
                or type(member["claim_key"]) is not list or not member["claim_key"]
                or any(type(value) is not int or value < 0 for value in member["claim_key"])
                or any(type(member[name]) is not int or member[name] <= 0
                       for name in ("body_variant_id", "terminal_variant_id"))):
            raise ValueError("sequence member requires an exact training claim and native role grammar")
        record = records.get(tuple(member["claim_key"]))
        if record is None or record.sample_role_code != 1:
            raise ValueError("sequence member is outside licensed supported training facts")
        gaps = member["output_gaps"]
        if (type(gaps) is not list or not 2 <= len(gaps) <= 9
                or any(type(gap) is not list or len(gap) > 128 for gap in gaps)):
            raise ValueError("sequence member output gaps are incomplete")
        for gap in gaps:
            if gap:
                validate_unicode_scalars(tuple(gap))
        selected.append(record)
        variants.extend((member["body_variant_id"], member["terminal_variant_id"]))
    if (len(set(variants)) != len(variants)
            or len({record.predicate_key for record in selected}) != len(selected)):
        raise ValueError("sequence members require unique predicates and native variants")
    courses, members = [], []
    target.mkdir(parents=False)
    for index, (member, record) in enumerate(zip(row["members"], selected, strict=True), 1):
        keys = []
        for body, name in ((True, "body"), (False, "terminal")):
            gaps = member["output_gaps"]
            gaps = [list(gap) for gap in gaps]
            if body:
                gaps[-1].extend(row["separator_values"])
            instruction = {"claim_keys": [member["claim_key"]], "output_gaps": [gaps],
                "separator_values": [], "variant_id": member[f"{name}_variant_id"],
                "owner": "teacher", "split": "train", "license_id": "CC0-1.0"}
            path = target / f"{index:02d}-{name}.train.json"
            with path.open("xb") as stream:
                stream.write(canonical_json_bytes(instruction) + b"\n")
            course = compile_train_operation_composition(path, facts, core, generator)
            courses.append(course)
            keys.append(course.witness.stable_key())
        members.append((record.predicate_key, *keys))
    branch = courses[0].template.language_branch
    digest = tuple(hashlib.sha256(raw).digest())
    rule = ClaimSequenceRule(row["rule_id"], row["minimum_claims"], row["maximum_claims"], digest,
        facts.course.source_identity, facts.course.license_identity,
        concept_identity((SEQUENCE_KIND, row["rule_id"], *digest), owner=branch.owner, versions=branch.versions),
        tuple(sorted(members)))
    return rule, tuple(courses)
