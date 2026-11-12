"""显式教师课程把已学习角色词形接入指定R-01分支；不学习整句或知识。"""
from __future__ import annotations

from dataclasses import replace
import hashlib

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_BRANCH, ObjectIdentity, SourceRef, representation_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.representation_rendering import representation_parts
from pure_integer_ai.cognition.shared.alias_resolution import AliasRouteSearchBudget
from pure_integer_ai.experiments.operation_fact_source import verify_operation_fact_source
from pure_integer_ai.experiments.operation_training_surface import verify_operation_training_surface
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_generation_candidate_alias_contract import (
    GenerationCandidateAliasCourseRequest, GenerationCandidateRealizationBinding,
)
from pure_integer_ai.experiments.ph2_generation_candidate_alias_course import (
    AliasRelationManifestProfile, build_alias_relation_manifest,
)
from pure_integer_ai.experiments.ph2_generation_candidate_pack import RULE_LITERAL
from pure_integer_ai.storage.assertion_identity import IDENTITY_SOURCE_RECORD, IntegerIdentityRegistry


ROLE_ALIAS_PROFILE = 92292


def _key(value):
    """拒绝bool/float与整数相等造成的身份别名。"""
    if (type(value) is not list or not value
            or any(type(part) is not int or not 0 <= part < 1 << 63 for part in value)):
        raise ValueError('role alias instruction requires complete strict integer keys')
    return tuple(value)


def _instruction(row):
    """恢复完整教师指令，两个构造入口共享同一身份合同。"""
    if (type(row) is not dict or set(row) != {'branch_key', 'family_key',
            'course_source_identity', 'owner', 'split', 'license_id'}
            or (row['owner'], row['split'], row['license_id']) != ('teacher', 'train', 'CC0-1.0')):
        raise ValueError('role alias requires an explicit licensed train teacher instruction')
    branch = ObjectIdentity.from_stable_key(_key(row['branch_key']))
    family = _key(row['family_key'])
    identity = _key(row['course_source_identity'])
    if branch.object_kind != OBJECT_LANGUAGE_BRANCH:
        raise ValueError('role alias target must be a complete LanguageBranch')
    return branch, family, identity


def compile_train_role_alias_record(row, facts, core):
    """从核验过的教师角色Span生成纯原子请求，不写模型或调用生成器。

    目标分支与family是显式教师授权，不能由运行时自行迁移。所有课程成员
    都必须已经在当前Core中具有相同完整命题、物理来源和语言成员；不能只挑
    对当前问题有利的成员。实际形成证据由当前Core恢复，不接受外部证据键。
    """
    branch, family, identity = _instruction(row)
    adapter, _seeds = verify_operation_fact_source(facts)
    if (identity != facts.course.source_identity
            or facts.course.license_identity != tuple(map(ord, row['license_id']))):
        raise ValueError('role alias physical course identity or license differs')
    active = {fact.proposition.stable_key(): fact for fact in core.active_surface_facts()}
    frames = {frame.proposition.stable_key(): frame for frame in core.active_surface_frames()}
    candidates = {candidate.proposition.proposition.stable_key(): candidate for candidate in adapter.candidates}
    if len(candidates) != len(facts.course.records):
        raise ValueError('role alias requires every distinct physical teacher member')
    registry = IntegerIdentityRegistry(core.backend)
    aliases = {}
    for record in facts.course.records:
        key = record.proposition_key
        candidate = candidates[key]
        if (record.sample_role_code != 1 or record.perturbation_code != 0
                or key not in active or key not in frames):
            raise ValueError('role alias teacher lacks a current supported Core member')
        fact = active[key]
        source = core.generation_input(fact.proposition)
        if source.proposition.definition != candidate.proposition or not source.evidence:
            raise ValueError('role alias complete proposition or forming evidence differs')
        source_hash = registry.find(IDENTITY_SOURCE_RECORD, candidate.observation_source.stable_key())
        verify_operation_training_surface(record, fact, frames[key], source_hash)
        forming = tuple(sorted(evidence.stable_key() for evidence in source.evidence))
        for binding in fact.bindings:
            tokens = record.surface_values[binding.start:binding.end]
            if not tokens or tuple(map(ord, binding.surface)) != tokens:
                raise ValueError('role alias needs a nonempty exact teacher role span')
            alias = GenerationCandidateRealizationBinding(binding.filler,
                representation_identity(family, tokens, owner=branch.owner, versions=branch.versions),
                RULE_LITERAL, forming)
            route = (alias.origin, alias.representation)
            prior = aliases.get(route)
            if prior is not None:
                alias = replace(alias, forming_evidence_keys=tuple(sorted({
                    *prior.forming_evidence_keys, *alias.forming_evidence_keys})))
            aliases[route] = alias
    return GenerationCandidateAliasCourseRequest(branch, tuple(aliases.values()))


def missing_role_alias_request(request, runtime):
    """只追加不存在的路由；相同唯一路由继承，竞争不能被训练覆盖。"""
    missing = []
    for binding in request.realizations:
        proposal = runtime.preview_surface(binding.origin, request.branch,
            budget=AliasRouteSearchBudget(32, 32, 32))
        options = proposal.result.options
        if not options:
            missing.append(binding)
        elif (len(options) != 1 or proposal.result.selected is None
                or options[0].value != binding.representation
                or proposal.result.selected.value != binding.representation):
            raise ValueError('existing role realization is ambiguous or differs from the licensed target')
    if not missing:
        return None
    return GenerationCandidateAliasCourseRequest(request.branch, tuple(missing))


def build_train_role_alias_manifest(row, request):
    """绑定完整教师指令和新增原子请求；正文不进入来源元数据。"""
    branch, family, _identity = _instruction(row)
    if (request.branch != branch or request.references
            or any(binding.rule != RULE_LITERAL
                   or representation_parts(binding.representation)[0] != family
                   or binding.representation.owner != branch.owner
                   or binding.representation.versions != branch.versions
                   for binding in request.realizations)):
        raise ValueError('role alias manifest differs from its licensed target')
    raw = canonical_json_line(row)
    if len(raw) > 1024 * 1024:
        raise ValueError('role alias teacher instruction exceeds its bounded record')
    digest = tuple(hashlib.sha256(raw).digest())
    source = SourceRef(ROLE_ALIAS_PROFILE,
        (int.from_bytes(bytes(digest[:8]), 'big') & ((1 << 63) - 1)) or 1,
        0, branch.owner, branch.versions)
    profile = AliasRelationManifestProfile((ROLE_ALIAS_PROFILE, 1), digest, source, document_scope(source), 1)
    return build_alias_relation_manifest(profile, request)
