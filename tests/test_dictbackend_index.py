"""test_dictbackend_index — DictBackend 内存索引正确性（2026-07-09·cProfile perf 真修）。

核心 invariant：**索引加速不改结果**——同 op 序下 indexed select/update/delete 结果 == 全表扫结果
（同集·同序·bit-identical）。桶内保插入序→_covering_candidates 候选序 == 全表扫过滤序。

设计：DictBackend._idx 桶存行 dict 引用·桶内序恒 == 插入序（insert 追加 / update 索引列变 +
delete + load_snapshot 从 _data rebuild）。本套测用纯扫参考实现 _scan_select 对照·随机 op 序差分。

铁律：纯整数（行值全 int）/ 确定性（fixed seed·bit-identical）/ 不走外挂。
"""
from __future__ import annotations

import random

from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import DictBackend, TYPE_INT

# 测试表 schema：a/b/c/v 多索引（mirror edge ix_from/ix_to/ix_endpoint/ix_type）·w 非索引列（strength-like）
_COLS = [("a", TYPE_INT), ("b", TYPE_INT), ("c", TYPE_INT),
         ("v", TYPE_INT), ("w", TYPE_INT)]
_INDEXES = [
    ("a", "b"),            # mirror ix_edge_from
    ("c",),                # mirror ix_edge_type
    ("a", "b", "c"),       # mirror ix_edge_endpoint
    ("v",),                # mirror ix_edge_tier（会 update 改）
]


def _new_backend(*, core: bool = False) -> DictBackend:
    # 默认 core=False：差分测含 delete 分支·核心表 append-only 禁 DELETE（铁律）·
    # 非核心扩展表 DELETE 协议允许（disc.check_write 放行）·正是 _do_delete 索引 rebuild 的合法域。
    b = DictBackend()
    b.register_table("t", _COLS, disc.DISC_MUTABLE_MONOTONE, _INDEXES, core=core)
    return b


def _scan_select(rows, where, where_gt, order_by, descending, limit):
    """纯扫参考实现（replicate 旧 _do_select 逻辑）—— indexed 结果须与之逐行相等。"""
    out = []
    for r in rows:
        if where and not all(r.get(k) == v for k, v in where.items()):
            continue
        if where_gt and not all(r.get(k, 0) > v for k, v in where_gt.items()):
            continue
        out.append(dict(r))
    if order_by is not None:
        out.sort(key=lambda r: (r.get(order_by) is None, r.get(order_by)),
                 reverse=descending)
    if limit is not None:
        out = out[:limit]
    return out


def _apply_update(rows, where, set_):
    """参考 update（同 backend 旧逻辑·就地改 ground truth rows）。"""
    for r in rows:
        if where and not all(r.get(k) == v for k, v in where.items()):
            continue
        for k, v in set_.items():
            if isinstance(v, tuple) and len(v) == 2 and v[0] == "+=":
                r[k] = (r.get(k) or 0) + v[1]
            else:
                r[k] = v


def _apply_delete(rows, where):
    return [r for r in rows if not all(r.get(k) == v for k, v in where.items())]


# 随机 where 候选（含覆盖索引命中 + 非覆盖退扫 + 空 where）
_WHERE_OPTIONS = [
    None,
    {"a": 1, "b": 1, "c": 1},   # exact endpoint（覆盖 ix_endpoint）
    {"a": 1, "b": 1},            # 覆盖 ix_from
    {"c": 1},                    # 覆盖 ix_type（exact）
    {"a": 1},                    # 无覆盖（ix_from 需 b）→ 退扫
    {"v": 1},                    # 覆盖 ix_tier
    {"w": 1},                    # 非索引列 → 退扫
    {"a": 1, "b": 1, "c": 1, "v": 1, "w": 1},  # 全列
    {},                          # 空 where（match all）
]


def test_select_matches_scan_random():
    """差分主测：200 轮随机 insert/update/delete/select·每次全 where 候选对照 indexed vs scan。"""
    rng = random.Random(20260709)
    b = _new_backend()
    ground: list[dict] = []

    for step in range(200):
        op = rng.choices(["insert", "update", "delete", "select"],
                         weights=[5, 3, 1, 6])[0]
        if op == "insert":
            row = {"a": rng.randint(1, 3), "b": rng.randint(1, 3),
                   "c": rng.randint(1, 3), "v": rng.randint(1, 2),
                   "w": rng.randint(1, 3)}
            b.insert("t", row)
            ground.append(dict(row))
        elif op == "update":
            where = rng.choice(_WHERE_OPTIONS[1:])  # 非 None
            # 混合：改索引列 v / 改复合索引 WHERE 列 a,b,c（审1 gap A·最危险路径·fuzz 证安全但补覆盖）
            #       / 改非索引列 w / 增量 +=
            roll = rng.random()
            if roll < 0.25:
                set_ = {"v": rng.randint(1, 2)}       # 改索引列 v→rebuild (v,)
            elif roll < 0.45:
                set_ = {"a": rng.randint(1, 3)}       # 改复合索引 WHERE 列 a→rebuild (a,b)/(a,b,c)
            elif roll < 0.6:
                set_ = {"b": rng.randint(1, 3)}       # 改复合索引 WHERE 列 b
            elif roll < 0.7:
                set_ = {"c": rng.randint(1, 3)}       # 改索引列 c→rebuild (c,)/(a,b,c)
            else:
                set_ = {"w": rng.randint(1, 3)}       # 改非索引列→dirty 空·零 rebuild
            if rng.random() < 0.3:
                set_ = {"w": ("+=", 1)}               # 增量（非索引列）
            b.update("t", dict(where), set_)
            _apply_update(ground, dict(where), set_)
        elif op == "delete":
            where = rng.choice([{"a": 1, "b": 1, "c": 1}, {"c": 2}, {"v": 1}])
            b.delete("t", dict(where))
            ground = _apply_delete(ground, dict(where))
        # 每次 op 后·全 where 候选 + order_by/limit 变体对照
        for where in _WHERE_OPTIONS:
            for order_by in [None, "w", "v"]:
                for limit in [None, 3]:
                    got = b.select("t", where=where, order_by=order_by, limit=limit)
                    exp = _scan_select(ground, where, None, order_by, False, limit)
                    assert got == exp, (
                        f"step={step} op={op} where={where} order_by={order_by} "
                        f"limit={limit}\n  got={got}\n  exp={exp}")


def test_select_where_gt_with_index():
    """覆盖索引 + where_gt 过滤：候选缩后 where_gt 过滤·结果 == 全表扫。"""
    b = _new_backend()
    ground = []
    rng = random.Random(7)
    for _ in range(40):
        row = {"a": rng.randint(1, 2), "b": rng.randint(1, 2),
               "c": rng.randint(1, 3), "v": rng.randint(0, 5), "w": rng.randint(0, 5)}
        b.insert("t", row)
        ground.append(dict(row))
    # where={c:X}（覆盖 ix_type）+ where_gt={v: 2}
    for c in (1, 2, 3):
        got = b.select("t", where={"c": c}, where_gt={"v": 2})
        exp = _scan_select(ground, {"c": c}, {"v": 2}, None, False, None)
        assert got == exp, f"c={c}\n  got={got}\n  exp={exp}"


def test_where_gt_orderby_limit_combo():
    """审1 gap B：覆盖索引 + where_gt + order_by + limit 三联组合（缩候选→过滤→排序→截断）。"""
    b = _new_backend()
    ground = []
    rng = random.Random(131)
    for _ in range(50):
        row = {"a": rng.randint(1, 2), "b": rng.randint(1, 2),
               "c": rng.randint(1, 3), "v": rng.randint(0, 5), "w": rng.randint(0, 9)}
        b.insert("t", row)
        ground.append(dict(row))
    for c in (1, 2, 3):
        got = b.select("t", where={"c": c}, where_gt={"v": 1}, order_by="w", limit=5)
        exp = _scan_select(ground, {"c": c}, {"v": 1}, "w", False, 5)
        assert got == exp, f"c={c} asc"
        got_d = b.select("t", where={"c": c}, where_gt={"v": 1},
                         order_by="w", descending=True, limit=3)
        exp_d = _scan_select(ground, {"c": c}, {"v": 1}, "w", True, 3)
        assert got_d == exp_d, f"c={c} desc"


def test_covering_picks_most_selective():
    """覆盖索引取列数最多者：where={a,b,c} 应命中 ix_endpoint(3列) 非 ix_from(2列)。"""
    b = _new_backend()
    # 插入多行同 (a,b) 不同 c·ix_from 桶大·ix_endpoint 桶小
    for c in range(10):
        b.insert("t", {"a": 1, "b": 1, "c": c, "v": 1, "w": 0})
    cands = b._covering_candidates("t", {"a": 1, "b": 1, "c": 5})
    assert cands is not None and len(cands) == 1   # ix_endpoint 命中单行
    # 若退到 ix_from 会得 10 行·确认选了最选择性
    assert cands[0]["c"] == 5


def test_insertion_order_in_bucket():
    """同索引值多行→桶内序 == 插入序（bit-identical 基础·无 order_by 时 select 同序）。"""
    b = _new_backend()
    seq = []
    rng = random.Random(11)
    for i in range(20):
        # 同 (a,b,c) 值不同 w·测桶内保插入序
        row = {"a": 1, "b": 1, "c": 1, "v": 1, "w": i}
        b.insert("t", row)
        seq.append(i)
    rows = b.select("t", where={"a": 1, "b": 1, "c": 1})   # 无 order_by
    assert [r["w"] for r in rows] == seq, "桶内序须 == 插入序"


def test_update_indexed_col_rebuilds():
    """set_tier-like：update 改索引列 v→(v,) 索引 rebuild·新分组正确。"""
    b = _new_backend()
    for i in range(5):
        b.insert("t", {"a": 1, "b": 1, "c": i, "v": 1, "w": i})
    b.update("t", where={"a": 1, "b": 1, "c": 0}, set_={"v": 2})  # 一行 v 1→2
    # (v,) 索引须反映：v=1 现 4 行·v=2 现 1 行
    assert len(b.select("t", where={"v": 1})) == 4
    assert len(b.select("t", where={"v": 2})) == 1
    assert b.select("t", where={"v": 2})[0]["c"] == 0


def test_update_increment_nonindexed_correct():
    """add_strength-like：增量 (+=) 改非索引列 w→值正确·索引不动。"""
    b = _new_backend()
    b.insert("t", {"a": 1, "b": 1, "c": 1, "v": 1, "w": 10})
    for _ in range(5):
        b.update("t", where={"a": 1, "b": 1, "c": 1}, set_={"w": ("+=", 1)})
    rows = b.select("t", where={"a": 1, "b": 1, "c": 1})
    assert len(rows) == 1 and rows[0]["w"] == 15


def test_delete_rebuilds_buckets():
    """delete repack _data→全索引 rebuild·删除后桶不含已删行。"""
    b = _new_backend()
    for c in range(5):
        b.insert("t", {"a": 1, "b": 1, "c": c, "v": 1, "w": 0})
    n = b.delete("t", {"c": 2})
    assert n == 1
    assert len(b.select("t", where={"a": 1, "b": 1, "c": 2})) == 0
    assert len(b.select("t", where={"c": 1})) == 1
    # 全表剩 4 行
    assert len(b.select("t")) == 4


def test_load_snapshot_rebuilds_indexes():
    """load_snapshot 换 _data→索引从新数据 rebuild·后续 indexed select 正确。"""
    b = _new_backend()
    b.insert("t", {"a": 1, "b": 1, "c": 1, "v": 1, "w": 0})
    snap = {
        "t": [
            {"a": 2, "b": 2, "c": 2, "v": 2, "w": 5},
            {"a": 2, "b": 2, "c": 3, "v": 2, "w": 6},
            {"a": 1, "b": 1, "c": 1, "v": 1, "w": 7},
        ],
    }
    b.load_snapshot(snap)
    # 索引须反映新数据（非旧行）
    assert len(b.select("t", where={"a": 2, "b": 2})) == 2     # ix_from
    assert len(b.select("t", where={"c": 1})) == 1             # ix_type
    assert len(b.select("t", where={"v": 2})) == 2             # ix_tier
    # 插入序保留（load_snapshot 行序）
    assert [r["w"] for r in b.select("t")] == [5, 6, 7]
    # 审1 gap C：load 后 indexed 查询无 order_by 返回序须 == 全表扫过滤序（桶内序==新 _data 序）
    loaded = [dict(r) for r in snap["t"]]
    for where in [{"a": 2, "b": 2}, {"c": 1}, {"v": 2}, {"a": 1, "b": 1, "c": 1}]:
        got = b.select("t", where=where)
        exp = _scan_select(loaded, where, None, None, False, None)
        assert got == exp, f"load 后序 where={where}\n  got={got}\n  exp={exp}"


def test_ensure_index_late_build_from_existing():
    """ensure_index 在已有数据后建→从已存行 build（镜像 SQLite CREATE INDEX）。"""
    b = DictBackend()
    b.register_table("t", _COLS, disc.DISC_MUTABLE_MONOTONE, [], core=True)  # 无初始索引
    for i in range(5):
        b.insert("t", {"a": 1, "b": 1, "c": i, "v": 1, "w": i})
    # 后建索引
    b.ensure_index("t", ("a", "b"))
    # 命中已存行（build from existing）
    assert len(b._covering_candidates("t", {"a": 1, "b": 1})) == 5
    assert len(b.select("t", where={"a": 1, "b": 1})) == 5


def test_defer_indexes_skips_build():
    """ensure_index(defer_indexes=True)→不建（镜像 SQLiteBackend·退扫正确）·defer=False 建。"""
    b = _new_backend()
    # defer=True 不建（_idx 不含）·select 退扫仍正确
    b.ensure_index("t", ("w",), defer_indexes=True)
    assert ("w",) not in b._idx.get("t", {})
    b.insert("t", {"a": 1, "b": 1, "c": 1, "v": 1, "w": 9})
    assert len(b.select("t", where={"w": 9})) == 1   # 退扫正确
    # defer=False 建
    b.ensure_index("t", ("w",), defer_indexes=False)
    assert ("w",) in b._idx["t"]


def test_bit_identical_two_runs():
    """同 op 序两跑→snapshot 完全相等（bit-identical·核心铁律）。"""
    def _run(seed):
        rng = random.Random(seed)
        b = _new_backend()
        for _ in range(60):
            b.insert("t", {"a": rng.randint(1, 3), "b": rng.randint(1, 3),
                           "c": rng.randint(1, 3), "v": rng.randint(1, 2),
                           "w": rng.randint(1, 5)})
        for c in (1, 2, 3):
            b.update("t", where={"c": c}, set_={"w": ("+=", 1)})
            b.update("t", where={"c": c, "v": 1}, set_={"v": 2})
        return b.snapshot()

    s1 = _run(42)
    s2 = _run(42)
    assert s1 == s2, "同 op 序两跑 snapshot 须 bit-identical"


def test_edge_hot_patterns_exact():
    """mirror 真实 edge 热模式：add_cooccurs_dedup exact 5-col + query_from + add_strength。"""
    b = DictBackend()
    # 模拟 edge schema（用真实列名子集）
    cols = [("space_id_from", TYPE_INT), ("local_id_from", TYPE_INT),
            ("space_id_to", TYPE_INT), ("local_id_to", TYPE_INT),
            ("edge_type", TYPE_INT), ("strength", TYPE_INT), ("tier", TYPE_INT)]
    idx = [("space_id_from", "local_id_from"),
           ("space_id_to", "local_id_to"),
           ("edge_type",),
           ("space_id_from", "local_id_from", "space_id_to", "local_id_to"),
           ("tier",)]
    b.register_table("edge", cols, disc.DISC_MUTABLE_MONOTONE, idx, core=True)
    ground = []
    rng = random.Random(99)
    # 模拟 4069 边规模（缩小到 800 验正确·COOCCURS 多重复 pair）
    for _ in range(800):
        fr, lr = rng.randint(1, 20), rng.randint(1, 20)
        et = rng.choice([6, 6, 6, 1, 2])  # 多 COOCCURS
        row = {"space_id_from": 1, "local_id_from": fr,
               "space_id_to": 1, "local_id_to": lr,
               "edge_type": et, "strength": 1, "tier": 1}
        b.insert("edge", row)
        ground.append(dict(row))
    # 热模式对照：add_cooccurs_dedup-like exact SELECT / query_from / query_type / add_strength UPDATE
    for _ in range(200):
        fr, lr, et = rng.randint(1, 20), rng.randint(1, 20), rng.choice([6, 1, 2])
        where = {"space_id_from": 1, "local_id_from": fr,
                 "space_id_to": 1, "local_id_to": lr, "edge_type": et}
        got = b.select("edge", where=where)
        exp = _scan_select(ground, where, None, None, False, None)
        assert got == exp
    # query_from
    for fr in range(1, 21):
        got = b.select("edge", where={"space_id_from": 1, "local_id_from": fr,
                                      "edge_type": 6})
        exp = _scan_select(ground, {"space_id_from": 1, "local_id_from": fr,
                                    "edge_type": 6}, None, None, False, None)
        assert got == exp, f"query_from fr={fr}"
    # query_type + add_strength 后再查
    before = len(b.select("edge", where={"edge_type": 6}))
    b.update("edge", where={"space_id_from": 1, "local_id_from": 1,
                            "space_id_to": 1, "local_id_to": 1, "edge_type": 6},
             set_={"strength": ("+=", 1)})
    after = len(b.select("edge", where={"edge_type": 6}))
    assert before == after   # strength update 不增行
