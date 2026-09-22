"""角色级 connector 的冻结整数身份协议，训练与只读运行时共用。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import ObjectIdentity, minimal_instruction_identity
from pure_integer_ai.experiments.language_generation_connector import LanguageConnectorOrdinalDefinition


RELATION_CONNECTOR_PROFILE = 91516
_ORDINAL_PREFIX = (20916, 1, 5)


def ordinal_definition(branch: ObjectIdentity, ordinal: int) -> LanguageConnectorOrdinalDefinition:
    """角色坐标来自图内声明，不以自然语言规则解释坐标。"""
    return LanguageConnectorOrdinalDefinition(minimal_instruction_identity(
        (*_ORDINAL_PREFIX, ordinal), owner=branch.owner, versions=branch.versions), ordinal)


def restore_ordinal(identity: ObjectIdentity, branch: ObjectIdentity) -> LanguageConnectorOrdinalDefinition:
    """只解码明确版本的图内坐标声明，未知布局不得猜数值。"""
    values = identity.components
    if (len(values) != 4 or values[:3] != _ORDINAL_PREFIX
            or identity != ordinal_definition(branch, values[3]).instruction):
        raise ValueError("connector 角色坐标声明不属于受支持整数协议")
    return ordinal_definition(branch, values[3])
