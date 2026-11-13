"""整数续算优化须逐位等于独立的原canonical字节编码。"""
import pytest

from pure_integer_ai.crosscut.determinism.hasher import (
    Hasher, _encode_int, _fnv1a, _fnv1a_int,
)


@pytest.mark.parametrize('state', [0,1,(1<<63)-1,1<<63,(1<<64)-1,0xCBF29CE484222325])
def test_integer_update_matches_original_bytes_for_complete_byte_range_and_edges(state):
    """完整单字节区间及两侧边界对任意前缀FNV状态保持相同。"""
    values = (*range(-512,513),-(1<<80),1<<80,-(1<<4096),1<<4096)
    for value in values:
        assert _fnv1a_int(value,state) == _fnv1a(_encode_int(value),state)
    assert _encode_int(-128) == b'\x02\xff\x80'
    assert _encode_int(-127) == b'\x02\x81'
    assert _encode_int(127) == b'\x02\x7f'
    assert _encode_int(128) == b'\x02\x00\x80'


@pytest.mark.parametrize('seed', [0,'canonical-integer-test',(1,-1,255)])
def test_streamed_tuple_matches_generic_canonical_hash_without_changing_seed(seed):
    """跨快慢分支的完整身份元组与通用嵌套编码逐位相同。"""
    hasher = Hasher(seed)
    original = hasher._iv
    values = (*range(-256,257),-(1<<64),1<<64,-(1<<128),1<<128)
    for tag in (-128,-127,0,127,128,4096):
        assert hasher.h63_tagged_int_tuple(tag,values) == hasher.h63((tag,values))
    assert hasher._iv == original


@pytest.mark.parametrize('invalid', [True,False,1.0,'1'])
def test_streamed_identity_keeps_strict_integer_rejection(invalid):
    """数值相等不能让非整数进入身份哈希。"""
    with pytest.raises(TypeError):
        Hasher('canonical-integer-test').h63_tagged_int_tuple(1,(0,invalid))
