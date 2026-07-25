"""对话止血①（2026-07-18）：attach_role_seq/attach_token_seq 幂等守卫测试。

gate ATTACH_SEQ_IDEMPOTENT_MODE：first-write-wins per-row（backend.count 查全5列·已存在 skip）。
  - gate ON：同 struct_ref 调两次不翻倍 def_array 行数（解 ~16× 累积雪球·词瀑布根因①）。
  - gate OFF：裸 insert 累积翻倍（行为同今·bit-identical 守 CI baseline）。
  - per-row 全5列键：不同 order_index/ref 的行不误 skip。
详见 doc/重来_对话止血_词瀑布降级_设计_2026-07-18.md。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.config import gates
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.node_store import NODE_WORD, NODE_CONCEPT
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.cognition.understanding.role_precedes import (
    attach_role_seq, attach_token_seq,
)


@pytest.fixture(autouse=True)
def _gate_reset():
    """每测前后复位 ATTACH_SEQ_IDEMPOTENT_MODE（守测试隔离）。"""
    saved = gates.ATTACH_SEQ_IDEMPOTENT_MODE
    gates.ATTACH_SEQ_IDEMPOTENT_MODE = False
    yield
    gates.ATTACH_SEQ_IDEMPOTENT_MODE = saved


def _build_struct(ctx, struct_label, token_surfaces):
    """建 struct_ref(NODE_CONCEPT) + tokens(NODE_WORD)·不 attach（测试自调 attach_*）。"""
    sid = ctx.space_id
    struct_ref = ctx.concept_index.ensure(struct_label, space_id=sid,
                                          node_type=NODE_CONCEPT)
    tokens = [ctx.concept_index.ensure(t, space_id=sid, node_type=NODE_WORD)
              for t in token_surfaces]
    return struct_ref, tokens


def _role_rows(backend, struct_ref):
    """读 struct_ref def_array 中 role 标记行（ref_space_id==0）·按 order_index 序。"""
    sid, lid = struct_ref
    rows = backend.select("def_array",
                          where={"space_id": sid, "local_id": lid, "ref_space_id": 0})
    return sorted(rows, key=lambda r: r["order_index"])


def _token_rows(backend, struct_ref):
    """读 struct_ref def_array 中 token concept ref 行（ref_space_id!=0）·按 order_index 序。"""
    sid, lid = struct_ref
    all_rows = backend.select("def_array", where={"space_id": sid, "local_id": lid})
    return sorted([r for r in all_rows if r["ref_space_id"] != 0],
                  key=lambda r: r["order_index"])


# ---- attach_role_seq 幂等 ----

def test_attach_role_seq_gate_on_no_duplication():
    """gate ON：同 struct_ref 同 role_seq 调两次 -> def_array role 行不翻倍（first-write-wins）。"""
    b = DictBackend(); ctx = make_train_context(b)
    struct_ref, _ = _build_struct(ctx, "__seg_0_0", ["猫", "吃", "鱼"])
    gates.ATTACH_SEQ_IDEMPOTENT_MODE = True
    attach_role_seq(ctx.backend, struct_ref, [10, 20, 30], order_base=0)
    attach_role_seq(ctx.backend, struct_ref, [10, 20, 30], order_base=0)   # 重复 observe
    assert len(_role_rows(ctx.backend, struct_ref)) == 3   # 不翻倍


def test_attach_role_seq_gate_off_accumulates():
    """gate OFF：裸 insert -> 重复调两次翻倍（行为同今·bit-identical baseline）。"""
    b = DictBackend(); ctx = make_train_context(b)
    struct_ref, _ = _build_struct(ctx, "__seg_0_0", ["猫", "吃", "鱼"])
    gates.ATTACH_SEQ_IDEMPOTENT_MODE = False
    attach_role_seq(ctx.backend, struct_ref, [10, 20, 30], order_base=0)
    attach_role_seq(ctx.backend, struct_ref, [10, 20, 30], order_base=0)
    assert len(_role_rows(ctx.backend, struct_ref)) == 6   # 翻倍·累积（旧 bug 行为）


# ---- attach_token_seq 幂等 ----

def test_attach_token_seq_gate_on_no_duplication():
    """gate ON：同 struct_ref 同 token_seq 调两次 -> def_array token 行不翻倍。"""
    b = DictBackend(); ctx = make_train_context(b)
    struct_ref, tokens = _build_struct(ctx, "__seg_0_0", ["猫", "吃", "鱼"])
    gates.ATTACH_SEQ_IDEMPOTENT_MODE = True
    attach_token_seq(ctx.backend, struct_ref, tokens, order_base=0)
    attach_token_seq(ctx.backend, struct_ref, tokens, order_base=0)
    assert len(_token_rows(ctx.backend, struct_ref)) == 3


def test_attach_token_seq_gate_off_accumulates():
    """gate OFF：重复调两次翻倍（baseline）。"""
    b = DictBackend(); ctx = make_train_context(b)
    struct_ref, tokens = _build_struct(ctx, "__seg_0_0", ["猫", "吃", "鱼"])
    gates.ATTACH_SEQ_IDEMPOTENT_MODE = False
    attach_token_seq(ctx.backend, struct_ref, tokens, order_base=0)
    attach_token_seq(ctx.backend, struct_ref, tokens, order_base=0)
    assert len(_token_rows(ctx.backend, struct_ref)) == 6


# ---- per-row 全5列键（不误 skip 不同行）----

def test_attach_role_seq_gate_on_different_rows_not_skipped():
    """gate ON：同 struct_ref 不同 order_base 的两批 role 行不误 skip（per-row 全5列键·非整体去重）。"""
    b = DictBackend(); ctx = make_train_context(b)
    struct_ref, _ = _build_struct(ctx, "__seg_0_0", ["猫", "吃", "鱼"])
    gates.ATTACH_SEQ_IDEMPOTENT_MODE = True
    attach_role_seq(ctx.backend, struct_ref, [10, 20], order_base=0)
    attach_role_seq(ctx.backend, struct_ref, [30, 40], order_base=100)   # 不同 order_base·新行
    assert len(_role_rows(ctx.backend, struct_ref)) == 4   # 两批都不 skip


def test_attach_token_seq_repeat_safe_under_idempotent():
    """gate ON：重复 token（'的'跨 position 共享 concept ref）每 position 一行不误 skip。
    守 repeat-safe 语义（order_index 异·全5列键含 order_index·同 concept 多 position 多行保留）。"""
    b = DictBackend(); ctx = make_train_context(b)
    struct_ref, tokens = _build_struct(ctx, "__seg_0_0", ["的", "猫", "的", "鱼"])
    gates.ATTACH_SEQ_IDEMPOTENT_MODE = True
    attach_token_seq(ctx.backend, struct_ref, tokens, order_base=0)
    rows = _token_rows(ctx.backend, struct_ref)
    assert len(rows) == 4   # 4 position 全保留（含两个"的"·order_index 0/2 异·不误 skip）
    assert [r["ref_local_id"] for r in rows] == [tokens[0][1], tokens[1][1],
                                                  tokens[2][1], tokens[3][1]]


def test_attach_role_seq_gate_on_same_position_different_role_skipped():
    """gate ON：同 struct_ref 同 order_index 不同 role 值（re-observe emergent_role 累积场景）->
    first-write-wins per-position（4列键 sid,lid,order_index,ref_space_id·不含 ref_local_id）·第 2 次 skip。
    解 role_seq 2× token_seq 累积（冒烟实证：re-observe 同句 order_base 重置 + role 值变致全5列键不匹配）。"""
    b = DictBackend(); ctx = make_train_context(b)
    struct_ref, _ = _build_struct(ctx, "__seg_0_0", ["猫", "吃", "鱼"])
    gates.ATTACH_SEQ_IDEMPOTENT_MODE = True
    attach_role_seq(ctx.backend, struct_ref, [10, 20, 30], order_base=0)   # 第 1 次
    attach_role_seq(ctx.backend, struct_ref, [40, 50, 60], order_base=0)   # re-observe 同 position 不同 role
    rows = _role_rows(ctx.backend, struct_ref)
    assert len(rows) == 3   # per-position first-write-wins·第 2 次 skip（不因 role 值异累积 2×）
    assert [r["ref_local_id"] for r in rows] == [10, 20, 30]   # 保留第 1 次 role


def test_attach_token_seq_gate_on_same_position_different_token_skipped():
    """gate ON：同 struct_ref 同 order_index 不同 token concept（re-observe 换 tokenization 场景）->
    first-write-wins per-position（4列键·不含 ref_local_id）·第 2 次 skip·保留首个 token。
    镜像 role 测试 test_attach_role_seq_gate_on_same_position_different_role_skipped。"""
    b = DictBackend(); ctx = make_train_context(b)
    struct_ref, _ = _build_struct(ctx, "__seg_0_0", ["猫", "吃", "鱼"])
    gates.ATTACH_SEQ_IDEMPOTENT_MODE = True
    t0 = ctx.concept_index.ensure("猫", space_id=ctx.space_id, node_type=NODE_WORD)
    t1 = ctx.concept_index.ensure("吃", space_id=ctx.space_id, node_type=NODE_WORD)
    t2 = ctx.concept_index.ensure("鱼", space_id=ctx.space_id, node_type=NODE_WORD)
    attach_token_seq(ctx.backend, struct_ref, [t0, t1, t2], order_base=0)        # 第 1 次
    t0b = ctx.concept_index.ensure("狗", space_id=ctx.space_id, node_type=NODE_WORD)
    attach_token_seq(ctx.backend, struct_ref, [t0b, t1, t2], order_base=0)        # re-observe 同 position 换 token
    rows = _token_rows(ctx.backend, struct_ref)
    assert len(rows) == 3   # per-position first-write-wins·第 2 次 skip（不因 token 换累积）
    assert [r["ref_local_id"] for r in rows] == [t0[1], t1[1], t2[1]]   # 保留第 1 次 token
