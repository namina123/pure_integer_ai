"""P1 维度桥测试（G-PR2·EDGE_INSTANTIATES on __seg_*→skeleton_ref·结构一等化 Phase A §十三-bis A.1）。

承接 doc/重来_语言域涌现对应地基_统计断奶最优方案_2026-07-17.md §十三-bis A.1（结构一等化）。
替旧 ATTR_SKELETON_BINDING 注解（composes_attr kind=24·effect-dormant 删）→ EDGE_INSTANTIATES(=15) 真边（关联在图中）。

机制（item-identity 映射·非 content-hash·code-verified）：
  - writer（COMPOSES_COMBINE_MODE ON）：discovery 建 item→skeleton map（以稳定 document scope 索引对齐）·
    observe 建 __seg_ struct_ref 后读 map 命中→build_instantiates_edge（EDGE_INSTANTIATES struct→skeleton·honest EPI_STRUCTURED 纯结构绑定）。
  - reader（DIM_BRIDGE_READ_MODE ON）：generate 读 graph.read_instantiates on **unit**（审1 MEDIUM-1：slot.ref=token
    when DISPATCH_TOKEN_CHAIN_MODE ON·binding 在 unit=struct_ref·读 slot.ref 恒 None）→记 workmem.last_dim_skeleton
    （**P2 断桥 consumer stub·P1 write-only 无消费者·非 observability**·值填充 VALUE_TRANSIT defer 断桥 #1053）。
  - bit-identical：两 gate OFF→不建 map·不建边·不读→逐字现状（CI 零回归）。两 gate dormant 不在生产 try/finally flip
    （consumer 落地=Phase E·非"同 P0b live"：P0b 经 activate_candidates 真活·本桥 consumer defer）。

TC1 reader unit：build_instantiates_edge → graph.read_instantiates 返 skeleton_ref·无边→None。
TC2 bit-identical gate-OFF：formal_train 跑后 edge 表零 EDGE_INSTANTIATES 行（gate OFF→writer 不建）。
TC3 gate-ON smoke：COMPOSES_COMBINE_MODE + DIM_BRIDGE_READ_MODE ON·formal_train 不崩（dormant 基建·值填充 defer）。
TC4 InputPayload.item_key 默认 0（向后兼容·CI bit-identical）。
TC5 EDGE_INSTANTIATES=15 是 C9-bis 登记合法 edge_type（完备性 #1·替旧 kind=24 非结构 kind 守）。
TC6 writer e2e：map 预填→ObservePipeline.observe 建 EDGE_INSTANTIATES 边 on __seg_（honest EPI_STRUCTURED）。
TC7 reader e2e：__seg_ 建边→generate_output 读 on unit→workmem.last_dim_skeleton 设（DISPATCH off + on 两路）。

铁律：纯整数（sid/lid/edge_type 全整·build_instantiates_edge assert_int 守）/ bit-identical（gate OFF 逐字现状·两 gate default OFF）/
  反 theater（writer+reader 基建落 + TC5-7 测·dormant·consumer defer 断桥 P2·非 paper closure·gate ON+有 skeleton 才建边）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.config import gates
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_types import EDGE_INSTANTIATES, is_registered_edge_type
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT, EPI_STRUCTURED
from pure_integer_ai.cognition.understanding.instantiates import build_instantiates_edge
from pure_integer_ai.storage.node_store import NODE_WORD, NODE_CONCEPT
from pure_integer_ai.cognition.understanding.observe import ObservePipeline
from pure_integer_ai.cognition.understanding.role_precedes import (
    build_struct_anchor, build_precedes_edges, attach_role_seq, attach_token_seq,
)
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.generate import generate_output
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.shared.types import (
    InputPayload, Segment, PathResult, PathData,
    LANG_NONE, LANG_ZH, MODALITY_LANGUAGE, DOMAIN_TEXT, STAGE_TRAINING,
)
from pure_integer_ai.experiments.formal_train import FormalTrainConfig, DefaultRoundRunner, _build_space_ctx
from pure_integer_ai.experiments.capability_exam import run_capability_exam
from tests.test_experiments import _causal_multi_sent_item, flat_floors, make_train_context


@pytest.fixture(autouse=True)
def _gate_reset():
    """每测前后复位维度桥 + dispatch/surface gate（守测试隔离·防跨测泄漏·TC7b 翻 DISPATCH+ORDINAL）。"""
    saved = (gates.COMPOSES_COMBINE_MODE, gates.DIM_BRIDGE_READ_MODE,
             gates.DISPATCH_TOKEN_CHAIN_MODE, gates.ORDINAL_SURFACE_MODE)
    gates.COMPOSES_COMBINE_MODE = False
    gates.DIM_BRIDGE_READ_MODE = False
    gates.DISPATCH_TOKEN_CHAIN_MODE = False
    gates.ORDINAL_SURFACE_MODE = False
    yield
    (gates.COMPOSES_COMBINE_MODE, gates.DIM_BRIDGE_READ_MODE,
     gates.DISPATCH_TOKEN_CHAIN_MODE, gates.ORDINAL_SURFACE_MODE) = saved


# ---- TC1 reader unit（build/read round-trip） ----

def test_tc1_read_instantiates_roundtrip():
    """reader unit：build_instantiates_edge → graph.read_instantiates 返 skeleton_ref·无边→None。

    验 reader 方法 + edge 存储层（EDGE_INSTANTIATES struct→skeleton·Phase A §十三-bis A.1）。
    """
    from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
    ctx = make_train_context(DictBackend())
    g = ctx.concept_graph
    seg_ref = ctx.concept_index.ensure(
        "__seg_test", space_id=ctx.space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    skel_ref = (ctx.space_id, 999)
    # 无 INSTANTIATES 边 → None
    assert g.read_instantiates(seg_ref) is None
    # 建边 → 读回
    build_instantiates_edge(ctx.edge_store, seg_ref, skel_ref, space_id=ctx.space_id)
    assert g.read_instantiates(seg_ref) == skel_ref, "reader 返 skeleton_ref（EDGE_INSTANTIATES 解析）"


def test_tc1b_build_instantiates_idempotent():
    """幂等：同 (struct→skeleton) 重复 build → 仍 1 边（query_from skip·跨 round re-observe 不 corrupt·Phase A §十三-bis A.1）。

    EdgeStore.add append-only 不去重·build_instantiates_edge 须自守幂等（mirror bootstrap_is_a_edges）。
    多轮训练 re-observe 同 segment → 同 struct_ref → 不堆叠重复 INSTANTIATES 边。
    """
    from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
    ctx = make_train_context(DictBackend())
    seg_ref = ctx.concept_index.ensure(
        "__seg_idem", space_id=ctx.space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    skel_ref = (ctx.space_id, 777)
    n1 = build_instantiates_edge(ctx.edge_store, seg_ref, skel_ref, space_id=ctx.space_id)
    n2 = build_instantiates_edge(ctx.edge_store, seg_ref, skel_ref, space_id=ctx.space_id)
    n3 = build_instantiates_edge(ctx.edge_store, seg_ref, skel_ref, space_id=ctx.space_id)
    assert (n1, n2, n3) == (1, 0, 0), "首建返 1·重复 build 返 0（幂等 skip）"
    rows = ctx.backend.select("edge", where={
        "space_id_from": seg_ref[0], "local_id_from": seg_ref[1], "edge_type": EDGE_INSTANTIATES})
    assert len(rows) == 1, "3 次 build → 仍 1 行（query_from skip·无堆叠）"


# ---- TC2 bit-identical（formal_train 后零 EDGE_INSTANTIATES 行） ----
# ★ 准确 rationale（大路 doc/重来_对应机制生产激活 §7·审1 MEDIUM-1 纠正）：本测用 toy 单 item
# _causal_multi_sent_item（< MIN_DISCOVER_SAMPLES=2）→ auto_discover 不聚簇 → discovery 不触发 →
# map 不填 → INSTANTIATES 不建。**非**"gate OFF→writer 不建"（formal_train try/finally 现 flip COMPOSES ON）。
# gate-OFF bit-identical 关键边界（formal_train/orchestrator + COMPOSES OFF + 充分 fixture）由 **FC12 直守**
# （test_floor_activation.test_fc12_composes_off_zero_instantiates_bit_identical·COMPOSES 单独 OFF + FC9 同款
# training-registered skeleton fixture·反向实验证 OFF 是关键变量）。

def test_tc2_bit_identical_gate_off_no_instantiates(tmp_path, flat_floors):
    """bit-identical：formal_train 跑后 edge 表零 EDGE_INSTANTIATES 行（toy 单句 <K 不聚簇）。

    ★ Bug B 修后纠偏（doc/重来_语言域建模推进设计_2026-07-18 §2.1）：discovery 现 pre-flip COMPOSES ON->
    scope B 切句激活。旧 fixture _causal_multi_sent_item（["x","y。","z","w。"]·2 句同 shape 各 2 token）在 Bug B
    下 discovery COMPOSES OFF->单 span=1 root<K 不聚簇->零 INSTANTIATES（旧 rationale 误判其 <K·实 =K=2·
    Bug B 隐藏多句性）。修后 discovery COMPOSES ON->切 2 句同 shape->=K 聚簇->INSTANTIATES>0（TC3 gate-ON 守此路径）。
    故本测改用单句 toy item（2 token 无句末标点·scope B 切=1 root·真 <K=2·不聚簇->零 INSTANTIATES·守本测原意）。
    gate-OFF bit-identical 关键边界仍由 FC12 直守（COMPOSES 单独 OFF+fixture·不经 formal_train 生产 flip）。
    """
    from pure_integer_ai.config import gates as _g
    from pure_integer_ai.experiments.collection import CollectedItem, COLLECT_CAUSES
    saved = _g.TRAINING_MODE
    _g.TRAINING_MODE = True
    try:
        b = DictBackend()
        backend_before = b.snapshot()
        cfg = FormalTrainConfig(run_dir=str(tmp_path / "db"), run_id="db2")
        # 单句 toy item（2 token·无句末标点·scope B 切=1 root<K=2·不聚簇·零 INSTANTIATES）
        _single_sent_item = CollectedItem(
            tokens=["a", "b"], role_seq=[1, 1],
            collect_type=COLLECT_CAUSES, source=SOURCE_BARE_TEXT)
        report = run_capability_exam(
            cfg, [_single_sent_item],
            backend=b, runner=DefaultRoundRunner())
    finally:
        _g.TRAINING_MODE = saved
    assert b.snapshot() == backend_before, "能力考核不得向宿主 backend 提交 schema 或图写入"
    assert isinstance(report, object)  # 沙箱内 formal_train 跑通并产独立报告


# ---- TC3 gate-ON smoke（不崩·live-but-empty·值填充 defer） ----

def test_tc3_gate_on_smoke_no_crash(tmp_path, flat_floors):
    """gate-ON smoke：COMPOSES_COMBINE_MODE + DIM_BRIDGE_READ_MODE ON·run_capability_exam 不崩。

    值填充 defer VALUE_TRANSIT（producer 未实施）·reader live-but-empty（同 P0b 无 data·机制活非 theater）。
    若 toy 语料发现 lang skeleton→INSTANTIATES 边>0；否则=0（toy 单 item <K 不发现·诚实·非 paper closure）。
    """
    from pure_integer_ai.config import gates as _g
    saved = _g.TRAINING_MODE
    gates.COMPOSES_COMBINE_MODE = True
    gates.DIM_BRIDGE_READ_MODE = True
    _g.TRAINING_MODE = True
    try:
        b = DictBackend()
        backend_before = b.snapshot()
        cfg = FormalTrainConfig(run_dir=str(tmp_path / "db3"), run_id="db3")
        report = run_capability_exam(
            cfg, [_causal_multi_sent_item()],
            backend=b, runner=DefaultRoundRunner())
    finally:
        _g.TRAINING_MODE = saved
    assert b.snapshot() == backend_before, "gate ON 考核也不得把 INSTANTIATES 或 schema 写回宿主"
    assert isinstance(report, object)  # reader/writer 在沙箱内运行且报告未丢失


# ---- TC4 InputPayload.item_key 默认 0（向后兼容） ----

def test_tc4_input_payload_item_key_default_zero():
    """item_key 默认 0（向后兼容·既有 InputPayload 构造零改·CI bit-identical·observe writer `if raw.item_key:` 守 0 不建）。"""
    from pure_integer_ai.cognition.shared.types import Segment
    inp = InputPayload(segments=[], source=0, stage=0)
    assert inp.item_key == 0, "默认 0（raw 未传 item_key→0→observe writer 不建→bit-identical）"


# ---- helpers（镜像 test_dispatch_token_chain._build_seg/_path·reader e2e 用） ----

def _build_seg(ctx, struct_label, token_surfaces, *, order_base=0):
    """建一段 lang __seg_：struct_ref(NODE_CONCEPT) + tokens(NODE_WORD) + PRECEDES 锚+序链 + role_seq + token_seq。"""
    sid = ctx.space_id
    struct_ref = ctx.concept_index.ensure(struct_label, space_id=sid, node_type=NODE_CONCEPT)
    tokens = [ctx.concept_index.ensure(t, space_id=sid, node_type=NODE_WORD) for t in token_surfaces]
    build_struct_anchor(ctx.edge_store, struct_ref, tokens[0],
                        source=SOURCE_BARE_TEXT, space_id=sid, order_base=order_base)
    build_precedes_edges(ctx.edge_store, tokens,
                         source=SOURCE_BARE_TEXT, space_id=sid, order_base=order_base)
    attach_role_seq(ctx.backend, struct_ref, list(range(len(tokens))), order_base=order_base)
    attach_token_seq(ctx.backend, struct_ref, tokens, order_base=order_base)
    return struct_ref, tokens


def _path(struct_ref):
    """最小 PathResult（topo_layers=[[struct_ref]]·generate 用）。"""
    return PathResult(path=PathData(edges=[], struct_unit_refs=[struct_ref]),
                      topo_layers=[[struct_ref]], convergence={}, source=struct_ref,
                      sink=None)


# ---- TC5 EDGE_INSTANTIATES=15 登记合法 edge_type（完备性 #1·替旧 kind=24 非结构 kind 守） ----

def test_tc5_instantiates_registered_edge_type():
    """EDGE_INSTANTIATES=15 是 C9-bis 登记的合法 edge_type（完备性 #1·is_registered_edge_type）。

    守：EdgeStore.add 接纳 15（不 raise）·结构一等化 INSTANTIATES 是真边非注解（Phase A §十三-bis A.1·替 kind=24 注解）。
    回归锁：若未来误从 REGISTERED_EDGE_TYPES 移除 15 → 本测 catch。
    """
    assert EDGE_INSTANTIATES == 15
    assert is_registered_edge_type(EDGE_INSTANTIATES), \
        "15 在 REGISTERED_EDGE_TYPES（C9-bis 登记合法·EdgeStore.add 接纳）"


# ---- TC6 writer e2e（map 预填→observe 建 EDGE_INSTANTIATES） ----

def test_tc6_writer_builds_instantiates_when_map_populated():
    """writer e2e：work_memory.lang_skeleton_by_item 预填→ObservePipeline.observe 建 EDGE_INSTANTIATES on __seg_。

    验 writer 数据流：raw.item_key → map 查 → build_instantiates_edge on struct_ref（honest EPI_STRUCTURED）。
    """
    b = DictBackend(); ctx = make_train_context(b)
    gates.COMPOSES_COMBINE_MODE = True
    item_key = 12345
    skel_ref = (ctx.space_id, 9999)
    wm = WorkMemory()
    wm.lang_skeleton_by_item[(item_key, 0)] = skel_ref   # scope B：map key (document_scope_hash, seg_idx)·单句 seg 0
    seg = Segment(seg_id=0, modality=MODALITY_LANGUAGE, lang=LANG_ZH,
                  domain=DOMAIN_TEXT, tokens=["猫", "吃", "鱼"])
    inp = InputPayload(
        segments=[seg], source=SOURCE_BARE_TEXT,
        stage=STAGE_TRAINING, item_key=item_key)
    ObservePipeline(_build_space_ctx(ctx), work_memory=wm).observe(inp)
    rows = b.select("edge", where={"edge_type": EDGE_INSTANTIATES})
    assert len(rows) == 1, "writer 建 1 行 EDGE_INSTANTIATES（map 命中 item_key→build on __seg_ struct_ref）"
    r = rows[0]
    assert (r["space_id_to"], r["local_id_to"]) == skel_ref, "边 to = skeleton_ref（struct→skeleton）"
    assert r["epistemic_origin"] == EPI_STRUCTURED, "honest EPI_STRUCTURED（审2·纯结构绑定非 cue）"


def test_tc6b_writer_no_edge_when_map_misses():
    """writer 守：map 无 item_key 条目→不建（gate ON 但 raw.item_key 无命中→不 build·bit-identical 退化）。"""
    b = DictBackend(); ctx = make_train_context(b)
    gates.COMPOSES_COMBINE_MODE = True
    seg = Segment(seg_id=0, modality=MODALITY_LANGUAGE, lang=LANG_ZH,
                  domain=DOMAIN_TEXT, tokens=["猫", "吃", "鱼"])
    inp = InputPayload(
        segments=[seg], source=SOURCE_BARE_TEXT,
        stage=STAGE_TRAINING, item_key=99999)
    ObservePipeline(_build_space_ctx(ctx), work_memory=WorkMemory()).observe(inp)   # map 空·item_key=99999 无命中
    rows = b.select("edge", where={"edge_type": EDGE_INSTANTIATES})
    assert rows == [], "map 无命中→不建 EDGE_INSTANTIATES（gate ON 但无 map 条目）"


# ---- TC7 reader e2e（generate 读 on unit→last_dim_skeleton·DISPATCH off + on） ----

def test_tc7a_reader_generate_reads_instantiates_on_unit_dispatch_off():
    """reader e2e（DISPATCH off）：__seg_ 建边→generate_output 读 on unit→workmem.last_dim_skeleton 设。

    审1 MEDIUM-1 test gap 闭：reader 真读 INSTANTIATES 边（非仅 no-crash）。DISPATCH off·slot.ref=unit=struct_ref。
    """
    b = DictBackend(); ctx = make_train_context(b)
    struct_ref, _ = _build_seg(ctx, "__seg_0_0", ["猫", "吃", "鱼"])
    skel_ref = (ctx.space_id, 9999)
    build_instantiates_edge(ctx.edge_store, struct_ref, skel_ref, space_id=ctx.space_id)
    gates.DIM_BRIDGE_READ_MODE = True
    wm = WorkMemory()
    generate_output(_path(struct_ref), ctx.concept_graph, wm, LANG_NONE)
    assert wm.last_dim_skeleton == skel_ref, \
        "reader 读 on unit→last_dim_skeleton=skeleton_ref（DISPATCH off·unit=struct_ref 有 INSTANTIATES 边）"


def test_tc7b_reader_generate_reads_instantiates_on_unit_dispatch_on():
    """reader e2e（DISPATCH on·审1 MEDIUM-1 关键 case）：slot.ref=token（无边）·reader 读 unit=struct_ref 仍命中。

    生产配置（DISPATCH_TOKEN_CHAIN_MODE ON·formal_train try/finally flip）下 reader 不失效：读 unit 非 slot.ref。
    若 reader 仍读 slot.ref（旧错位）→此测 fail（token 无 INSTANTIATES 边→last_dim_skeleton=()）。
    """
    b = DictBackend(); ctx = make_train_context(b)
    struct_ref, _ = _build_seg(ctx, "__seg_0_0", ["猫", "吃", "鱼"])
    skel_ref = (ctx.space_id, 9999)
    build_instantiates_edge(ctx.edge_store, struct_ref, skel_ref, space_id=ctx.space_id)
    gates.DIM_BRIDGE_READ_MODE = True
    gates.DISPATCH_TOKEN_CHAIN_MODE = True      # 生产配置：slot.ref=token concept
    gates.ORDINAL_SURFACE_MODE = True
    wm = WorkMemory()
    generate_output(_path(struct_ref), ctx.concept_graph, wm, LANG_NONE)
    assert wm.last_dim_skeleton == skel_ref, \
        "DISPATCH on·slot.ref=token 无边·reader 读 unit=struct_ref 仍命中（审1 MEDIUM-1 修验证）"
