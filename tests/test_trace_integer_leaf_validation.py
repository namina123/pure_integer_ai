"""整数叶子快捷遍历保留全部类型、负值、映射和深度约束。"""
import pytest

from pure_integer_ai.experiments.graph_query_trace_contract import (
    GraphQueryTraceContractError, _integer_artifact_tree, _integer_key,
)


class IntegerSubclass(int):
    """仅用来验证宿主整数子类不能混入严格整数协议。"""


@pytest.mark.parametrize('bad', [True, False, IntegerSubclass(3), '7', None, -1])
@pytest.mark.parametrize('container', [list, tuple])
def test_full_key_rejects_bad_leaf_after_long_valid_prefix(bad, container):
    """长整数前缀不允许跳过尾部非法类型或负值。"""
    value = container([0] * 4096 + [bad])
    with pytest.raises(GraphQueryTraceContractError, match='nonnegative integer key'):
        _integer_key(value, 'test')


def test_key_preserves_all_ordered_members_and_empty_key():
    """保留所有原成员及通用空键合同，不裁剪或去重。"""
    value = [0, 1, 1, 2 ** 127]
    assert _integer_key(value, 'test') == tuple(value)
    assert _integer_key([], 'test') == ()


@pytest.mark.parametrize('bad', [True, False, IntegerSubclass(3), '7', object()])
def test_integer_tree_rejects_bad_nested_leaf_after_integer_run(bad):
    """父容器内快速核验仍对复合分支中的每个非整数叶子失败关闭。"""
    with pytest.raises(GraphQueryTraceContractError, match='noninteger value'):
        _integer_artifact_tree({'first': list(range(4096)), 'last': [None, {'bad': bad}]})


@pytest.mark.parametrize('bad_key', ['', 1, None])
def test_integer_tree_rejects_bad_mapping_keys(bad_key):
    """叶子优化不得忽略对象键的原字符串合同。"""
    with pytest.raises(GraphQueryTraceContractError, match='object key is invalid'):
        _integer_artifact_tree({'outer': [{bad_key: [1, 2, 3]}]})


def test_integer_tree_accepts_signed_values_none_and_deep_containers():
    """树协议仍允许有符号整数和None，并保留迭代式深度恢复。"""
    value = (None, -1, 0, 2 ** 127)
    for _ in range(1500):
        value = [value]
    _integer_artifact_tree({'deep': value, 'wide': tuple(range(4096))})
