"""#1134 测试：程度→属性器 intensity augment（degree 副词→命题 Rational intensity·平行 pol/mod·复用 EDGE_PROPERTY）。

权威设计 `doc/重来_ChineseSemanticKB能力映射_2026-07-16.md` §4.3 + 设计 `doc/重来_程度属性器intensity_2026-07-16.md`。
程度副词（很/非常/极其=2/1·较=3/2·稍=2/5·Rational·非 float·file-driven 来自 degree_cues_zh.txt）= 命题值强度缩放·
平行既有 polarity/modality·进 surface 后缀 `_i{num}_{den}` + ATTR_PROP_INTENSITY=30 结构存·gate DEGREE_MODE 默认 OFF（bit-identical）。

测：
  DI1 loader（resolve_degree_facts + load_degree_cues_file）parse num/den + E5 graceful
  DI2 populate_degree_cues + is_degree_cue/degree_intensity_of（gate OFF→False/None·ON→查表）
  DI3 extract_property_claims degree 窗口（degree cue at val_idx → value 后移+intensity·OFF→既有窗口不变）
  DI4 build_property_edges 8-tuple intensity(2/1) → surface `_i2_1` 后缀 + ATTR_PROP_INTENSITY=(2,1) 结构存
  DI5 build_property_edges 6-tuple（无 intensity·向后兼容）→ default 1/1 无后缀 + ATTR_PROP_INTENSITY=(1,1)·异 DI4 distinct 节点
  DI6 CI bit-identical（无 PURE_INTEGER_AI_LOCAL_DIR → resolve {} → cache 空 → is_degree_cue 恒 False → intensity 恒 1/1）

铁律：纯整数（num/den int·Rational）/ 确定性（sorted/gate）/ 不写死（degree cue+intensity 来自外部文件）/ bit-identical（gate OFF default）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.composes_attr import (
    register_composes_attr, read_composes_attrs, ATTR_PROP_INTENSITY, ATTR_PROPOSITION,
)
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.collection import resolve_degree_facts, load_degree_cues_file, SOURCE_BARE_TEXT
from pure_integer_ai.cognition.understanding.cue_words import (
    populate_degree_cues, degree_intensity_of, is_degree_cue, _DEGREE_CUES,
)
from pure_integer_ai.cognition.understanding.cue_extractor import extract_property_claims
from pure_integer_ai.cognition.understanding.property import build_property_edges
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_NONE
from pure_integer_ai.config import gates


@pytest.fixture
def degree_clean():
    """隔离 _DEGREE_CUES module cache + gates.DEGREE_MODE（save→clear→yield→restore）。"""
    saved_cache = {k: dict(v) for k, v in _DEGREE_CUES.items()}
    saved_gate = gates.DEGREE_MODE
    _DEGREE_CUES.clear()
    gates.DEGREE_MODE = False
    yield
    _DEGREE_CUES.clear()
    for k, v in saved_cache.items():
        _DEGREE_CUES[k] = dict(v)
    gates.DEGREE_MODE = saved_gate


# ---- DI1 loader ----

def test_di1_loader_parse_and_graceful(tmp_path):
    """load_degree_cues_file parse `cue\tnum/den`→(num,den)·resolve_degree_facts lang-keyed·E5 graceful。"""
    p = tmp_path / "degree_cues_zh.txt"
    p.write_text("# 注释\n\n非常\t2/1\n较\t3/2\n坏行\n稍 2/5\n负/-1\n零\t0/1\n", encoding="utf-8")
    m = load_degree_cues_file(str(p))
    assert m == {"非常": (2, 1), "较": (3, 2), "稍": (2, 5)}, "parse 正当 num/den·跳注释/空/坏行/非正"
    # resolve_degree_facts lang-keyed（degree_cues_{suffix}.txt）
    assert resolve_degree_facts(LANG_ZH, local_dir=str(tmp_path)) == m
    # 缺文件 / 未知 lang → {}
    assert resolve_degree_facts(LANG_ZH, local_dir=str(tmp_path / "nope")) == {}
    assert resolve_degree_facts(LANG_NONE, local_dir=str(tmp_path)) == {}


# ---- DI2 cue gate + cache ----

def test_di2_is_degree_cue_gated(degree_clean):
    """populate_degree_cues 喂 cache·gate DEGREE_MODE OFF→is_degree_cue False/degree_intensity_of None·ON→查表。"""
    populate_degree_cues(LANG_ZH, {"非常": (2, 1), "较": (3, 2)})
    # gate OFF（默认）→ 恒 False/None（bit-identical 守）
    assert gates.DEGREE_MODE is False
    assert is_degree_cue("非常", LANG_ZH) is False
    assert degree_intensity_of("非常", LANG_ZH) is None
    # gate ON → 查表
    gates.DEGREE_MODE = True
    assert is_degree_cue("非常", LANG_ZH) is True
    assert degree_intensity_of("非常", LANG_ZH) == (2, 1)
    assert degree_intensity_of("较", LANG_ZH) == (3, 2)
    assert degree_intensity_of("未知词", LANG_ZH) is None   # 非程度词
    assert is_degree_cue("未知词", LANG_ZH) is False


def test_di2b_populate_empty_is_noop(degree_clean):
    """空 mapping populate no-op（不污染 cache·生产 default 无文件→bit-identical）。"""
    populate_degree_cues(LANG_ZH, {})
    assert _DEGREE_CUES.get(LANG_ZH) is None or len(_DEGREE_CUES.get(LANG_ZH, {})) == 0
    gates.DEGREE_MODE = True
    assert degree_intensity_of("非常", LANG_ZH) is None   # 空 cache → None


# ---- DI3 extract degree 窗口 ----

def test_di3_extract_degree_window_shifts_value(degree_clean):
    """degree cue at val_idx(是 与 value 间) → value 后移一位+intensity·degree_on=False→既有窗口不变。

    tokens: 苹果(0) 的(1) 颜色(2) 是(3) 非常(4) 红(5)·是 at j=3·的 at j-2=1·val=j+1=4(非常)。
    degree_on=True + 非常→(2,1)：val 后移→5(红)·intensity (2,1)。
    degree_on=False：val 留 4(非常)·intensity (1,1)（既有窗口·degree 未识别）。
    """
    populate_degree_cues(LANG_ZH, {"非常": (2, 1)})
    gates.DEGREE_MODE = True
    toks = ["苹果", "的", "颜色", "是", "非常", "红"]
    on = extract_property_claims(toks, lang=LANG_ZH, degree_on=True)
    assert len(on) == 1
    # claim 8-tuple (subj,attr,val,0,pol,mod,inum,iden)·val=5(红)·intensity (2,1)
    assert on[0][2] == 5, "degree 占 val_idx(4=非常)→真 value 后移到 5(红)"
    assert (on[0][6], on[0][7]) == (2, 1), "intensity 从 非常 查表 (2,1)"
    # degree_on=False → 既有窗口（val=4=非常·intensity 1/1）
    off = extract_property_claims(toks, lang=LANG_ZH, degree_on=False)
    assert len(off) == 1
    assert off[0][2] == 4, "degree_off → val 留 4(非常)·既有窗口不变"
    assert (off[0][6], off[0][7]) == (1, 1), "degree_off → intensity 默认 1/1"


def test_di3b_extract_degree_boundary_skip(degree_clean):
    """degree 占 val_idx 但真 value(val_idx+1)越界 → skip（守反统计·不凑配）。"""
    populate_degree_cues(LANG_ZH, {"非常": (2, 1)})
    gates.DEGREE_MODE = True
    # 是 非常 <end>——degree 占末位·真 value 越界
    toks = ["苹果", "的", "颜色", "是", "非常"]
    on = extract_property_claims(toks, lang=LANG_ZH, degree_on=True)
    assert on == [], "degree 占 val_idx·真 value 越界 → skip（无凑配声明）"


# ---- DI4/DI5 build_property_edges intensity ----

def _ctx_with_prop():
    b = DictBackend()
    register_composes_attr(b)   # ATTR_PROP_INTENSITY 标记表（idempotent）
    ctx = make_train_context(b)
    return ctx


def test_di4_build_intensity_surface_and_attr():
    """8-tuple intensity(2/1) → 命题 surface `_i2_1` 后缀 + ATTR_PROP_INTENSITY=(2,1) 结构存。"""
    ctx = _ctx_with_prop()
    sid = ctx.space_id
    subj = ctx.concept_index.ensure("苹果", space_id=sid)
    attr = ctx.concept_index.ensure("颜色", space_id=sid)
    val = ctx.concept_index.ensure("红", space_id=sid)
    refs = [subj, attr, val]
    # 8-tuple (subj,attr,val,0,pol,mod,inum=2,iden=1)
    n = build_property_edges(ctx.edge_store, ctx.concept_index, ctx.backend, refs,
                             property_claims=[(0, 1, 2, 0, 0, 0, 2, 1)],
                             source=SOURCE_BARE_TEXT, space_id=sid)
    assert n == 1
    # 命题 surface 含 _i2_1 后缀（≠1/1）
    prop_surface = f"__prop_{subj[0]}_{subj[1]}_{attr[0]}_{attr[1]}_i2_1"
    prop_ref = ctx.concept_index.lookup(prop_surface, sid)
    assert prop_ref is not None, "intensity 命题节点 surface 含 _i2_1 后缀"
    attrs = read_composes_attrs(ctx.backend, prop_ref)
    assert attrs.get(ATTR_PROP_INTENSITY) == (2, 1), "ATTR_PROP_INTENSITY 结构存 (2,1)"
    assert attrs.get(ATTR_PROPOSITION) == (0, 0), "命题标记仍在"


def test_di5_build_default_no_suffix_distinct():
    """6-tuple（无 intensity·向后兼容）→ default 1/1 无 `_i` 后缀 + ATTR_PROP_INTENSITY=(1,1)·异 DI4 distinct 节点。"""
    ctx = _ctx_with_prop()
    sid = ctx.space_id
    subj = ctx.concept_index.ensure("苹果", space_id=sid)
    attr = ctx.concept_index.ensure("颜色", space_id=sid)
    val = ctx.concept_index.ensure("红", space_id=sid)
    refs = [subj, attr, val]
    # 同 (subj,attr,val) 两 claim：6-tuple（无 intensity·1/1）+ 8-tuple（intensity 2/1）→ 两个 distinct 命题节点
    build_property_edges(ctx.edge_store, ctx.concept_index, ctx.backend, refs,
                         property_claims=[(0, 1, 2, 0, 0, 0),          # 6-tuple·default 1/1·无后缀
                                          (0, 1, 2, 0, 0, 0, 2, 1)],    # 8-tuple·intensity 2/1·_i2_1
                         source=SOURCE_BARE_TEXT, space_id=sid)
    base_surface = f"__prop_{subj[0]}_{subj[1]}_{attr[0]}_{attr[1]}"           # default·无后缀
    inten_surface = f"__prop_{subj[0]}_{subj[1]}_{attr[0]}_{attr[1]}_i2_1"     # intensity 后缀
    base_ref = ctx.concept_index.lookup(base_surface, sid)
    inten_ref = ctx.concept_index.lookup(inten_surface, sid)
    assert base_ref is not None and inten_ref is not None, "两 distinct 命题节点（intensity 区分身份）"
    assert base_ref != inten_ref, "default(1/1) 与 intensity(2/1) 是不同命题节点（surface 后缀区分）"
    assert read_composes_attrs(ctx.backend, base_ref).get(ATTR_PROP_INTENSITY) == (1, 1), "default ATTR_PROP_INTENSITY=(1,1)"
    assert read_composes_attrs(ctx.backend, inten_ref).get(ATTR_PROP_INTENSITY) == (2, 1), "intensity ATTR_PROP_INTENSITY=(2,1)"


# ---- DI6 CI bit-identical ----

def test_di6_ci_no_file_no_degree_bit_identical(degree_clean):
    """无 PURE_INTEGER_AI_LOCAL_DIR → resolve {} → cache 空 → is_degree_cue 恒 False（纵 gate ON）→ intensity 恒 1/1。"""
    import os
    assert "PURE_INTEGER_AI_LOCAL_DIR" not in os.environ or not os.environ["PURE_INTEGER_AI_LOCAL_DIR"]
    assert resolve_degree_facts(LANG_ZH) == {}   # 无 local_dir → {}
    # 纵令 gate ON（cache 空）→ degree 不识别（bit-identical 守）
    gates.DEGREE_MODE = True
    assert is_degree_cue("非常", LANG_ZH) is False
    assert degree_intensity_of("非常", LANG_ZH) is None
    toks = ["苹果", "的", "颜色", "是", "非常", "红"]
    on = extract_property_claims(toks, lang=LANG_ZH, degree_on=True)
    assert len(on) == 1
    assert (on[0][6], on[0][7]) == (1, 1), "空 cache → intensity 默认 1/1（CI bit-identical）"
    assert on[0][2] == 4, "空 cache → degree 未识别·val 留 4(非常)·既有窗口不变"
