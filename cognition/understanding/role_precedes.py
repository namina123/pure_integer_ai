"""cognition.understanding.role_precedes — 模块2 role_seq + PRECEDES 建边（§7.1 地基）。

PRECEDES 是结构骨架边（A 类型 DAG·Kahn 分层载体·死路检测图基础）。
  - strength 恒 = 1（§7.1/A4·结构真值·reward 永不调·§十五C9-bis）
  - order_index 填 token 序（C4·同域不同段·句间序 = 句序 × TOKEN_CAP_OFFSET）
  - role 降字段不建独立边（§十一缺口#1）·role_seq 作结构概念点属性（def_array·§十五决策4）
  - 空间按 stage 传（M4·训练期 CORE / 训练后阅读 MEMORY·非硬编码 CORE）
  - 自环不建（PRECEDES 非自反）

attach_role_seq：role_seq 作结构概念点的 def_array 有序属性（变长"序列符号"范式·无 JSON）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_DERIVED
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.config import gates
from pure_integer_ai.cognition.shared.edge_types import EDGE_PRECEDES

PRECEDES_STRENGTH = 1            # §7.1 铁律·结构真值·reward 永不调
TOKEN_CAP_OFFSET = 1 << 20       # C4·句间序 × OFFSET 分段避免与 token 序撞值（全局纯整递增）


def _add_precedes(edge_store: EdgeStore, *, a: tuple[int, int], b: tuple[int, int],
                  source: int, order_index: int) -> bool:
    """PRECEDES add 单点（gate ON 走 add_precedes_dedup 跨 round 去重 / OFF 走 EdgeStore.add = bit-identical）。

    返 True=新建（计数）/ False=skip（dedup 已存在）。S2 dead-end 根因 §10.3·mirror COOCCURS dedup 接线范式。
    gate OFF 总返 True（append-only 总新建·等同旧 add）。strength 恒 1（§7.1）·role=None·epistemic_origin=None。
    """
    if getattr(gates, "PRECEDES_DEDUP_MODE", False):
        return edge_store.add_precedes_dedup(
            space_id_from=a[0], local_id_from=a[1],
            space_id_to=b[0], local_id_to=b[1],
            edge_type=EDGE_PRECEDES, source=source,
            order_index=order_index, tier=TIER_PRIMARY)
    edge_store.add(
        space_id_from=a[0], local_id_from=a[1],
        space_id_to=b[0], local_id_to=b[1],
        edge_type=EDGE_PRECEDES, strength=PRECEDES_STRENGTH,
        source=source, epistemic_origin=None,
        order_index=order_index, role=None, tier=TIER_PRIMARY)
    return True


def build_precedes_edges(edge_store: EdgeStore, refs: list[tuple[int, int]],
                         *, source: int, space_id: int,
                         order_base: int = 0) -> int:
    """句内 token 序 PRECEDES 建边（i 前于 i+1）。

    refs      已归一概念引用序列（段内）。
    order_base 该段 token 序起点（C4·跨段全局递增·句间序 = 句序 × TOKEN_CAP_OFFSET）。
    返回建边数。
    """
    assert PRECEDES_STRENGTH == 1
    n = 0
    for i in range(len(refs) - 1):
        a, b = refs[i], refs[i + 1]
        if a == b:
            continue  # 自环不建
        if _add_precedes(edge_store, a=a, b=b, source=source,
                         order_index=order_base + i):
            n += 1   # gate ON dedup 跳过已存在不计·OFF 总 True（bit-identical）
    return n


def build_struct_anchor(edge_store: EdgeStore,
                        struct_ref: tuple[int, int],
                        first_token: tuple[int, int],
                        *, source: int, space_id: int,
                        order_base: int = 0) -> int:
    """struct_ref → 段内首 token PRECEDES 锚边（item3 缺漏1·2026-07-02）。

    每段一条锚边·把 active 从 struct_ref 种子传到 token 概念（struct_ref↔token 原无边·
    CAUSES 边 from(token) 不 in active→BLOCKED→永不进 path.edges→反馈腿空转）。
    锚边是 PRECEDES（strength=1·§7.1 序边结构真值·reward 永不调）·propagate 只走 CAUSES
    自动排除锚边·不污染 reward。自环不建（struct_ref≠first_token）。
    """
    if struct_ref == first_token:
        return 0
    return 1 if _add_precedes(edge_store, a=struct_ref, b=first_token,
                              source=source, order_index=order_base) else 0


def build_inter_segment_precedes(edge_store: EdgeStore,
                                 struct_refs: list[tuple[int, int]],
                                 last_tokens: list[tuple[int, int]],
                                 *, source: int, space_id: int,
                                 seg_order_base: int) -> int:
    """句间序 PRECEDES（替旧 TYPE_SENTENCE_TRANSITION·§十五E·item3 缺漏5 d-1 修正）。

    段i末token → 段i+1 struct_ref（非旧 struct_ref[i]→struct_ref[i+1] 短路边）。
    去短路边致 sink 变浅：sink 只能经段内 token 链到达 → 落最深 → CAUSES 边在 sink 返回前被收集。
    order_index = 句序 × TOKEN_CAP_OFFSET（C4·与 token 序同域不同段）。
    last_tokens 与 struct_refs 等长·每段末 token（段内 resolved[-1]）。
    """
    n = 0
    for i in range(len(struct_refs) - 1):
        a = last_tokens[i]
        b = struct_refs[i + 1]
        if a == b:
            continue
        if _add_precedes(edge_store, a=a, b=b, source=source,
                         order_index=seg_order_base + i * TOKEN_CAP_OFFSET):
            n += 1   # gate ON dedup 跳过已存在不计·OFF 总 True（bit-identical）
    return n


def attach_role_seq(backend: StorageBackend,
                    struct_ref: tuple[int, int],
                    role_seq: list[int],
                    *, order_base: int = 0) -> None:
    """role_seq 作结构概念点属性（def_array 有序·§十五决策4 变长序列符号范式）。

    role 降字段不建边（§十一缺口#1）·role_seq 整体作 struct_concept 的 def_array。
    def_array: (space_id, local_id, order_index, ref_space_id, ref_local_id)——
    role 是整数标记·作 ref 存（ref_space_id=0 标记“role 标记非概念 ref”·消费方按此判）。

    **幂等守卫**（对话止血①·gate ATTACH_SEQ_IDEMPOTENT_MODE·兑现 gates.py:412-413 round8b defer）：
    跨 item 同 struct_ref（content hash 撞）反复 observe 致 def_array 累积 ~16× -> read_role_seq 返全长
    -> generate 千词瀑布。gate ON = first-write-wins per-row（backend.count 查全5列·已存在 skip）/
    OFF = 裸 insert（行为同今·bit-identical 守 CI）。镜像 _add_precedes:32 in-helper 范式·单线程 observe
    内调用无 TOCTOU。详见 doc/重来_对话止血_词瀑布降级_设计_2026-07-18.md。
    """
    sid, lid = struct_ref
    _dedup = getattr(gates, "ATTACH_SEQ_IDEMPOTENT_MODE", False)
    for i, role in enumerate(role_seq):
        assert_int(role, _where="attach_role_seq.role")
        row = {"space_id": sid, "local_id": lid, "order_index": order_base + i,
               "ref_space_id": 0, "ref_local_id": role}   # 0 = role 标记非概念 ref（消费方判）
        if _dedup and backend.count("def_array", {"space_id": sid, "local_id": lid,
                                                    "order_index": order_base + i,
                                                    "ref_space_id": 0}) > 0:
            continue   # first-write-wins per-position（4列键·不含 ref_local_id）·re-observe 同 position 不同 role 值亦 skip
        backend.insert("def_array", row)


def attach_token_seq(backend: StorageBackend,
                     struct_ref: tuple[int, int],
                     token_refs: list[tuple[int, int]],
                     *, order_base: int = 0) -> None:
    """段 token concept ref 序作 struct_ref def_array 属性（P0 #1040·生成侧 dispatch 沿此派发 token concept）。

    镜像 attach_role_seq 范式·存段内已归一 token concept ref（resolved·按 position·order_index=order_base+i）。
    与 role markers（ref_space_id==0）共存同 struct_ref def_array·消费方按 ref_space_id 区分：
    read_role_seq 读 ==0（role 标记）·read_token_seq 读 !=0（token concept ref）。

    **为何用存储而非 PRECEDES walk**：walk 按 concept ref dedup·但重复 token（功能词"的"跨 position
    共享同 concept ref）致 walk 丢 position（X→猫→X visited→break·漏第 2 个 X+后续）·真语料炸。
    存储每 position 一行（同 concept 多 position 多行）·按 order_index 读→完整序列·repeat-safe。

    **幂等守卫**（对话止血①·gate ATTACH_SEQ_IDEMPOTENT_MODE·同 attach_role_seq）：跨轮 re-observe 同
    struct_ref 累积 -> read_token_seq 返全长 -> fill loop 千词。gate ON first-write-wins per-row（含 order_index·
    同 concept 多 position 多行 order_index 异不误 skip）/ OFF 裸 insert（bit-identical）。observe call-site
    门控 DISPATCH_TOKEN_CHAIN_MODE（是否调本函数）·helper 内 ATTACH_SEQ_IDEMPOTENT_MODE 守幂等（独立 gate·正交）。
    """
    sid, lid = struct_ref
    _dedup = getattr(gates, "ATTACH_SEQ_IDEMPOTENT_MODE", False)
    for i, tref in enumerate(token_refs):
        assert_int(tref[0], tref[1], _where="attach_token_seq.tref")
        assert tref[0] != 0, "token concept space_id 不可为 0（撞 role 标记 sentinel·4列键误匹配）"
        row = {"space_id": sid, "local_id": lid, "order_index": order_base + i,
               "ref_space_id": tref[0], "ref_local_id": tref[1]}   # token concept 真实 space（!= 0 区别 role 标记）
        if _dedup and backend.count("def_array", {"space_id": sid, "local_id": lid,
                                                    "order_index": order_base + i,
                                                    "ref_space_id": tref[0]}) > 0:
            continue   # first-write-wins per-position（4列键）·同 position 第 2 次 skip·repeat-safe（order_index 异不误 skip）
        backend.insert("def_array", row)
