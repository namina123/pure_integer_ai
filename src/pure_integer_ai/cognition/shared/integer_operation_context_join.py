"""按完整变量环境传播有界结构上下文，不读取真值、来源许可或输出。"""
from __future__ import annotations


def bind_operation_atom(atom, fact, environment):
    """完整 schema 与每个角色相等才扩展环境；重复变量必须指向同一对象。"""
    if fact.definition.predicate != atom.schema.relation:
        return None
    atom.schema.validate_definition(fact.definition)
    fillers = {binding.role: binding.filler for binding in fact.definition.bindings}
    result = dict(environment)
    for role, variable in atom.role_variables:
        filler = fillers[role]
        if variable in result and result[variable] != filler:
            return None
        result[variable] = filler
    return result


def _environment_key(environment):
    """规范有序完整变量值记录可跨语言重建，不用投影或摘要替代。"""
    return tuple((variable, filler.stable_key()) for variable, filler in sorted(environment.items()))


def join_operation_context(program, external, lookup, budget):
    """逐原子消费稀疏角色索引并保留所有命中前缀，包括相关反证和未闭合分支。

    lookup 接收 predicate、role、filler，返回含完整 definition 的有序记录。
    相同环境仅去重结构行走，原完整命题记录仍全部保留；超限不返回部分结果。
    """
    if program.path_query is not None or len(program.atoms) > budget.max_atoms:
        raise ValueError('operation context join requires a bounded non-path program')
    environments = (dict(external),)
    selected = {}
    attempts = 0
    for atom in program.atoms:
        following = {}
        for environment in environments:
            bindings = tuple((role, environment[variable]) for role, variable in atom.role_variables
                             if variable in environment)
            if not bindings:
                raise ValueError('operation context atom is disconnected from the current environment')
            rows = lookup(atom.schema.relation, *bindings[0])
            if len(rows) > budget.max_facts:
                raise ValueError('operation context candidate budget exceeded; no partial result')
            for row in rows:
                attempts += 1
                if attempts > budget.max_match_attempts:
                    raise ValueError('operation context join budget exceeded; no partial result')
                updated = bind_operation_atom(atom, row, environment)
                if updated is None:
                    continue
                key = row.definition.proposition.stable_key()
                if key in selected and selected[key] != row:
                    raise ValueError('operation context proposition identity differs')
                selected[key] = row
                if len(selected) > budget.max_facts:
                    raise ValueError('operation context candidate budget exceeded; no partial result')
                following[_environment_key(updated)] = updated
                if len(following) > budget.max_matches:
                    raise ValueError('operation context environment budget exceeded; no partial result')
        environments = tuple(following[key] for key in sorted(following))
        if not environments:
            break
    return tuple(selected[key] for key in sorted(selected))


__all__ = ['bind_operation_atom', 'join_operation_context']
