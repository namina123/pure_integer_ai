"""角色词形入学保持完整来源、不学习整句并拒绝覆盖既有竞争。"""
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import language_branch_identity, representation_identity
from pure_integer_ai.cognition.shared.representation_rendering import representation_parts
from pure_integer_ai.experiments.generation_role_alias_course import (
    build_train_role_alias_manifest, compile_train_role_alias_record, missing_role_alias_request,
)
from pure_integer_ai.experiments.operation_fact_source import read_sqlite_operation_fact_source
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime


def _row(identity):
    return dict(branch_key=list(language_branch_identity((92292, 100)).stable_key()),
        family_key=[92292, 101], course_source_identity=list(identity),
        owner='teacher', split='train', license_id='CC0-1.0')


@pytest.mark.parametrize('field,value', (
    ('owner', 'evaluator'), ('split', 'held_out'), ('license_id', 'UNKNOWN'),
    ('branch_key', [11, True]), ('family_key', [1.0]), ('course_source_identity', []),
))
def test_invalid_teacher_instruction_fails_before_core_access(field, value):
    with pytest.raises((TypeError, ValueError)):
        compile_train_role_alias_record(dict(_row((1,)), **{field: value}), None, None)


@pytest.fixture(scope='module')
def learned_roles():
    """只读现役开发模型和明确教师来源，不接触任何评测器或重训。"""
    parent = os.environ.get('PIA_ROLE_ALIAS_PARENT')
    source = os.environ.get('PIA_ROLE_ALIAS_TEACHER')
    prototypes = os.environ.get('PIA_ROLE_ALIAS_PROTOTYPES')
    if not all((parent, source, prototypes)):
        pytest.skip('requires an explicitly named current development model and train source')
    paths = tuple(Path(value).resolve(strict=True) for value in (parent, source, prototypes))
    if any(path.drive.upper() != 'K:' for path in paths):
        raise ValueError('explicit K roots required')
    facts = read_sqlite_operation_fact_source(paths[1], paths[2])
    row = _row(facts.course.source_identity)
    with TrainedRelationGraphRuntime(paths[0] / 'training.sqlite3',
            surface_artifact_database=paths[0] / 'relation_surfaces.sqlite3') as core:
        request = compile_train_role_alias_record(row, facts, core)
    return facts, row, request


def test_role_aliases_are_exact_teacher_endpoints_with_current_forming_evidence(learned_roles):
    facts, row, request = learned_roles
    expected = set()
    for record in facts.course.records:
        endpoints = {tuple(endpoint.to_value()['endpoint_key']): endpoint.to_value()
                     for endpoint in record.endpoints}
        for binding in record.role_bindings:
            filler = tuple(binding.to_value()['filler_key'])
            endpoint = endpoints[filler]
            expected.add((filler, record.surface_values[endpoint['start']:endpoint['end']]))
    actual = {(binding.origin.stable_key(), representation_parts(binding.representation)[1])
              for binding in request.realizations}
    assert actual == expected
    assert request.references == ()
    assert all(binding.forming_evidence_keys for binding in request.realizations)
    assert all(representation_parts(binding.representation)[0] == tuple(row['family_key'])
               for binding in request.realizations)
    manifest = build_train_role_alias_manifest(row, request)
    assert len(manifest.entries) >= len(request.realizations)


def _runtime(request, mode):
    def preview(origin, branch, *, budget):
        assert branch == request.branch
        binding = next(binding for binding in request.realizations if binding.origin == origin)
        value = binding.representation
        if mode == 'different':
            value = representation_identity((92292, 102), representation_parts(value)[1],
                owner=branch.owner, versions=branch.versions)
        option = SimpleNamespace(value=value)
        options = () if mode == 'missing' else ((option, option) if mode == 'ambiguous' else (option,))
        selected = option if len(options) == 1 and mode != 'unselected' else None
        return SimpleNamespace(result=SimpleNamespace(options=options, selected=selected))
    return SimpleNamespace(preview_surface=preview)


def test_existing_exact_routes_are_inherited_without_retraining(learned_roles):
    _facts, _row_value, request = learned_roles
    assert missing_role_alias_request(request, _runtime(request, 'exact')) is None
    assert missing_role_alias_request(request, _runtime(request, 'missing')) == request


@pytest.mark.parametrize('mode', ('different', 'ambiguous', 'unselected'))
def test_existing_competition_or_other_family_cannot_be_overwritten(learned_roles, mode):
    _facts, _row_value, request = learned_roles
    with pytest.raises(ValueError):
        missing_role_alias_request(request, _runtime(request, mode))


def test_manifest_cannot_change_teacher_target(learned_roles):
    _facts, row, request = learned_roles
    with pytest.raises(ValueError):
        build_train_role_alias_manifest(dict(row, family_key=[92292, 102]), request)
    with pytest.raises(ValueError):
        build_train_role_alias_manifest(dict(row, owner='evaluator'), request)
    changed = replace(request, branch=language_branch_identity((92292, 103)))
    with pytest.raises(ValueError):
        build_train_role_alias_manifest(row, changed)
