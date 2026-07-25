"""P0a 测试：concept_correspondence（码点对应）+ ordinal surface_of resolver。

纠偏回合 round2 地基首刀（doc/重来_设计偏离审查_2026-07-14.md A 偏离·plan velvet-juggling-garden）。
让系统 SPEAK + dump 留文本：observe 建 concept 时写码点 -> surface_of 读码点 -> chr -> 文本。

覆盖：
  CC1 record/load roundtrip（镜像 test_stage9 concept_identity :1973）+ 幂等
  CC2 surface_of 解析码点 -> 真字（gate ON）+ gate OFF 退 None（bit-identical）
  CC3 承重不变量：无对应（int surface / 无行 / 表未注册）-> None 非 ""（judge J2s truthiness 依赖）
  CC4 bit-identical local_id with correspondence（镜像 :2012 dedup-after-load·码点写不改 local_id 分配）
  CC5 unicode_codec BMP + supplementary（𝄞 U+1D11E）+ 空串

铁律：纯整数（码点 ord）/ 确定性 bit-identical / 反 theater（真产字 + load 回能产字 + reward 不读文本）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.config import gates
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.training.cursor import DUMP_TABLES, dump_run, load_run
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.storage.concept_identity import CONCEPT_IDENTITY_TABLE
from pure_integer_ai.storage.concept_correspondence import (
    register_concept_correspondence, record_correspondence, load_correspondence,
    CONCEPT_CORRESPONDENCE_TABLE, CORR_ORDINAL,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.result.graph_view import ConceptGraph, ordinal_surface_of
from pure_integer_ai.crosscut.integer.unicode_codec import encode, decode


@pytest.fixture(autouse=True)
def _gate_off():
    """每测后复位 ORDINAL_SURFACE_MODE（守测试隔离·防跨测泄漏）。"""
    saved = gates.ORDINAL_SURFACE_MODE
    gates.ORDINAL_SURFACE_MODE = False
    yield
    gates.ORDINAL_SURFACE_MODE = saved


# ---- CC5 unicode_codec ----

def test_unicode_codec_bmp_and_supplementary():
    """码点编解码：BMP（中文/英文）+ supplementary（𝄞 U+1D11E·PEP 393 一 ch 一码点）+ 空串 + 往返。"""
    assert encode("苹果") == (33529, 26524)
    assert encode("apple") == (97, 112, 112, 108, 101)
    assert encode("") == ()
    # supplementary plane：𝄞 = U+1D11E = 119070（非 surrogate pair·一 ch 一码点）
    assert encode("𝄞") == (119070,)
    # 往返
    for s in ["苹果", "apple", "", "𝄞", "a🍪b"]:
        assert decode(encode(s)) == s
    # decode 空序列 -> ""（合法·但 resolver 须把"无行"映射 None·非调 decode）
    assert decode(()) == ""
    # encode 非 str 拒
    with pytest.raises(TypeError):
        encode(42)  # type: ignore[arg-type]


# ---- CC1 record/load roundtrip + 幂等 ----

def test_concept_correspondence_record_load_roundtrip(tmp_path):
    """record 写码点 -> dump -> load -> load_correspondence 还原（镜像 concept_identity :1973）。"""
    b1 = DictBackend(); bootstrap(b1); register_concept_correspondence(b1)
    from pure_integer_ai.storage.spaces.registry import SpaceRegistry
    from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
    reg = SpaceRegistry(b1); sp = AbstractSpace.create(reg, "core"); sid = sp.space_id
    record_correspondence(b1, space_id=sid, local_id=42,
                          corr_kind=CORR_ORDINAL, codepoints=encode("苹果"))
    assert load_correspondence(b1, space_id=sid, local_id=42,
                               corr_kind=CORR_ORDINAL) == (33529, 26524)
    # 幂等：同 (concept, kind) 再写 -> skip（无重复行）
    record_correspondence(b1, space_id=sid, local_id=42,
                          corr_kind=CORR_ORDINAL, codepoints=encode("苹果"))
    assert load_correspondence(b1, space_id=sid, local_id=42,
                               corr_kind=CORR_ORDINAL) == (33529, 26524)
    # dump -> load roundtrip
    dump_run(b1, str(tmp_path), "rCC", spaces=[sid],
             tables=DUMP_TABLES + (CONCEPT_CORRESPONDENCE_TABLE,))
    b2 = DictBackend(); bootstrap(b2); register_concept_correspondence(b2)
    load_run(b2, str(tmp_path), "rCC")
    assert load_correspondence(b2, space_id=sid, local_id=42,
                               corr_kind=CORR_ORDINAL) == (33529, 26524), "跨 run roundtrip bit-identical"


def test_record_correspondence_bare_fixture_skip():
    """表未注册（bare fixture 未 register）-> KeyError 静默 skip（向后兼容·镜像 concept_identity）。"""
    b = DictBackend(); bootstrap(b)   # 未 register_concept_correspondence
    # 不抛·best-effort skip
    record_correspondence(b, space_id=1, local_id=1,
                          corr_kind=CORR_ORDINAL, codepoints=encode("x"))
    assert load_correspondence(b, space_id=1, local_id=1,
                               corr_kind=CORR_ORDINAL) == ()


# ---- CC2 surface_of 解析码点 -> 真字 ----

def test_surface_of_resolves_ordinal_codepoints():
    """ensure(str) 写对应 -> ordinal_surface_of / ConceptGraph.surface_of（gate ON）-> 真字。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    ci = ctx.concept_index
    ref = ci.ensure("苹果", space_id=sid)
    # module-level resolver
    assert ordinal_surface_of(b, ref) == "苹果"
    # gate OFF -> None（bit-identical·退占位）
    g = ConceptGraph(b)
    assert g.surface_of(ref) is None
    # gate ON -> 真字
    gates.ORDINAL_SURFACE_MODE = True
    g2 = ConceptGraph(b)
    assert g2.surface_of(ref) == "苹果"
    # cache 命中
    assert g2.surface_of(ref) == "苹果"


# ---- CC3 承重不变量：None 非 "" ----

def test_surface_of_returns_none_for_int_or_no_rows():
    """承重不变量：无对应（int surface / 无行）-> None 非 ""（judge J2s `if w:` truthiness 依赖）。

    "" falsy 会让 slot_fill_rate bound 不增 -> reward 变 -> 破 bit-identical。故 resolver 须返 None。
    """
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    ci = ctx.concept_index
    # int surface（QID/synset）-> 无对应（hook 仅 isinstance(str) 写）
    ref_int = ci.ensure(42, space_id=sid)
    assert ordinal_surface_of(b, ref_int) is None, "int surface 须 None 非 ''"
    # 无行 ref（不存在的 local_id）
    assert ordinal_surface_of(b, (sid, 999999)) is None, "无行须 None 非 ''"
    # gate ON 亦 None（非 ""）
    gates.ORDINAL_SURFACE_MODE = True
    g = ConceptGraph(b)
    assert g.surface_of(ref_int) is None
    assert g.surface_of((sid, 999999)) is None
    # 显式对照：None 非 ""（truthiness 皆是 falsy·但 None 是占位语义·"" 是空串语义·caller 退 #ref 占位）
    assert g.surface_of(ref_int) is None, "承重不变量：None 非 ''"


# ---- CC4 bit-identical local_id with correspondence ----

def test_concept_index_ensure_bit_identical_local_id_with_correspondence(tmp_path):
    """码点写不改 local_id 分配 + dedup-after-load 命中载入 local_id（镜像 :2012）+ load 回能产字。"""
    b1 = DictBackend(); ctx1 = make_train_context(b1)
    sid = ctx1.space_id
    ref = ctx1.concept_index.ensure("apple", space_id=sid)
    # dump（含 correspondence 表）
    dump_run(b1, str(tmp_path), "rB", spaces=[sid],
             tables=DUMP_TABLES + (CONCEPT_IDENTITY_TABLE, CONCEPT_CORRESPONDENCE_TABLE))
    # fresh backend + load + fresh ConceptIndex
    b2 = DictBackend(); ctx2 = make_train_context(b2)
    load_run(b2, str(tmp_path), "rB")
    ref2 = ctx2.concept_index.ensure("apple", space_id=sid)   # dedup 命中载入
    assert ref2 == ref, f"码点写不改 local_id 分配·dedup-after-load 须命中·{ref2} ≠ {ref}"
    # load 回能产字（A 核心目标）
    assert ordinal_surface_of(b2, ref2) == "apple", "load 回能产字"
