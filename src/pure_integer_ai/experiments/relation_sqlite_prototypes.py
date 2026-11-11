"""SQLite教师课程复用七类现役来源及专属编译协议，不定义自然语言规则。"""
from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_authored_alias_refers_w06_course import read_authored_alias_refers_w06_seeds
from pure_integer_ai.experiments.ph2_authored_causes_course import _compile_causes_seed, read_authored_causes_seeds
from pure_integer_ai.experiments.ph2_authored_mereology_course import _compile_mereology_seed, read_authored_mereology_seeds
from pure_integer_ai.experiments.ph2_authored_precedes_course import _compile_precedes_seed, read_authored_precedes_seeds
from pure_integer_ai.experiments.ph2_authored_property_course import read_authored_property_seeds
from pure_integer_ai.experiments.ph2_authored_relation_compile import compile_relation_seed
from pure_integer_ai.experiments.ph2_authored_relation_schema import AuthoredRelationSeed, ROLE_PROPERTY_INTENSITY
from pure_integer_ai.experiments.ph2_authored_semantic_pair_course import _compile_semantic_pair_seed, read_authored_semantic_pair_seeds
from pure_integer_ai.experiments.ph2_authored_subset_member_course import read_authored_subset_member_seeds
from pure_integer_ai.experiments.ph2_w06_registry import W06_RELATION_REGISTRY
from pure_integer_ai.experiments.ph2_w06_source_semantic import W06_RELATION_PROFILES


PROTOTYPE_LOADERS = {
    'authored_relation_alias_refers_w06_seed_v2.jsonl.sample': read_authored_alias_refers_w06_seeds,
    'authored_relation_subset_member_seed_v1.jsonl.sample': read_authored_subset_member_seeds,
    'authored_relation_property_seed_v1.jsonl.sample': read_authored_property_seeds,
    'authored_relation_mereology_seed_v1.jsonl.sample': read_authored_mereology_seeds,
    'authored_relation_similar_antonym_seed_v1.jsonl.sample': read_authored_semantic_pair_seeds,
    'authored_relation_precedes_seed_v1.jsonl.sample': read_authored_precedes_seeds,
    'authored_relation_causes_seed_v1.jsonl.sample': read_authored_causes_seeds,
}
_COMPILERS = {'MEREOLOGY': _compile_mereology_seed, 'SIMILAR_ANTONYM': _compile_semantic_pair_seed,
              'PRECEDES': _compile_precedes_seed, 'CAUSES': _compile_causes_seed}


@dataclass(frozen=True, slots=True)
class SqliteRelationPrototype:
    """原完整关系及显式Rational成员，不能由宿主默认值补齐属性强度。"""

    seed: AuthoredRelationSeed
    rational_role_values: tuple[tuple[int, int, int], ...] = ()


def read_sqlite_relation_prototypes(path):
    """只调用已登记的来源读取器，属性的整数强度与原关系共同保留。"""
    loader = PROTOTYPE_LOADERS.get(path.name)
    if loader is None:
        raise ValueError('教师关系原型未注册')
    result = []
    for original in loader(path):
        if loader is read_authored_property_seeds:
            result.append(SqliteRelationPrototype(original.relation,
                ((ROLE_PROPERTY_INTENSITY, original.intensity_num, original.intensity_den),)))
        else:
            result.append(SqliteRelationPrototype(original))
    return tuple(result)


def compile_sqlite_relation_prototype(record):
    """由冻结关系profile选择原编译器及证据种类，不跨关系借用标签。"""
    seed = record.seed
    profile = W06_RELATION_PROFILES.get(seed.relation_family)
    if profile is None:
        raise ValueError('教师关系没有现役W-06 profile')
    if seed.relation_family == 'PROPERTY':
        if len(record.rational_role_values) != 1:
            raise ValueError('属性课程缺少原Rational强度')
        compiled = compile_relation_seed(seed, semantic_identity=True,
            rational_role_values=record.rational_role_values)
    else:
        if record.rational_role_values:
            raise ValueError('非属性课程不得携带额外Rational角色')
        compiled = _COMPILERS.get(profile.substage_key, compile_relation_seed)(seed, semantic_identity=True)
    registry = W06_RELATION_REGISTRY[profile.substage_key]
    return compiled, (profile.substage_key, registry.teacher_evidence_kinds[0])
