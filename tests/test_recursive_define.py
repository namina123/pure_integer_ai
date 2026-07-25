"""tests/test_recursive_define.py — G8 教师提问循环恢复（RECORD-time 递归 define·bit-identical）。

验 define_recursive（teacher/recursive_define.py）：DFS 递归 define + 三守（depth/budget/visited-self-ref）
+ atom 闭包 + 菱形 memo + RECORD↔REPLAY bit-identical + MODE_OFF no-op + provider KeyError 闭项守。

镜像 test_stage6 录放层 fixture 范式（DictBackend + register_recording_table + RecordableLLMTeacher）。
provider 契约：(ref) -> (text, [prereq_refs])·deterministic·caller-supplied。
"""
import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.teacher.recordable_teacher import (
    RecordableLLMTeacher, register_recording_table,
    MODE_OFF, MODE_RECORD, MODE_REPLAY,
)
from pure_integer_ai.teacher.teacher_boundary import KIND_DEFINE
from pure_integer_ai.teacher.recursive_define import (
    define_recursive, DEFAULT_MAX_DEPTH, DEFAULT_BUDGET,
)


@pytest.fixture
def core():
    b = DictBackend()
    bootstrap(b)
    register_recording_table(b)
    yield b
    b.close()


def _llm_factory():
    """假 LLM（MODE_RECORD 离线用·确定性·不调真 LLM·镜像 test_stage6._llm_factory）。"""
    def llm_call(kind, args):
        if kind == KIND_DEFINE:
            # args = ("define", sid, lid, text, content_type) → 用 caller 传的 text 回放
            return {"kind": KIND_DEFINE, "content_type": args[-1],
                    "text": args[3], "response_int": 0}
        return {"kind": kind, "content_type": 0, "text": "x", "response_int": 1}
    return llm_call


def _provider(table):
    """provider(ref) -> (text, prereq_refs)·dict-backed·deterministic。未知 ref → KeyError（闭项守测）。"""
    def p(ref):
        text, prereqs = table[ref]
        return text, list(prereqs)
    return p


def _recorded_texts(b):
    """读 recording 表所有 response_text（按 call_hash 排序·确定性·bit-identical 比对用）。"""
    rows = b.select("teacher_recording", where={}, limit=10000)
    return sorted(r["response_text"] for r in rows)


# ============ TC1 基本递归（A→B→C 链） ============

def test_recursive_define_chain(core):
    """A→[B]·B→[C]·C→[] → define A,B,C（count=3·全录）。"""
    b = core
    A, B, C = (1, 10), (1, 11), (1, 12)
    table = {A: ("tA", [B]), B: ("tB", [C]), C: ("tC", [])}
    t = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=_llm_factory())
    n = define_recursive(A, t, _provider(table))
    assert n == 3   # A, B, C 全 define
    texts = _recorded_texts(b)
    assert texts == ["tA", "tB", "tC"]


# ============ TC2 自指/环终止（A→B→A） ============

def test_recursive_define_cycle_terminates(core):
    """A→[B]·B→[A] → define A,B·重访 A 时 visited 命中→终止（count=2·不无限递归）。"""
    b = core
    A, B = (1, 10), (1, 11)
    table = {A: ("tA", [B]), B: ("tB", [A])}
    t = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=_llm_factory())
    n = define_recursive(A, t, _provider(table))
    assert n == 2   # A, B 各一次（A 第二次到访→ visited 跳过）
    assert _recorded_texts(b) == ["tA", "tB"]


# ============ TC3 深度上限（链长 > max_depth） ============

def test_recursive_define_depth_cap(core):
    """链 A→B→C→D→E→F（6 节点）·max_depth=5 → depth 0..4 定义 A..E·F depth=5 跳过（count=5）。"""
    b = core
    refs = [(1, 10 + i) for i in range(6)]   # A..F
    table = {refs[i]: (f"t{i}", [refs[i + 1]] if i + 1 < 6 else []) for i in range(6)}
    t = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=_llm_factory())
    n = define_recursive(refs[0], t, _provider(table), max_depth=5)
    assert n == 5   # A(depth0)..E(depth4)·F depth5>=max_depth 跳过
    assert _recorded_texts(b) == ["t0", "t1", "t2", "t3", "t4"]


# ============ TC4 预算上限（fan-out 爆炸防护） ============

def test_recursive_define_budget_cap(core):
    """A→[B,C,D,E,F]（5 prereq）·budget=3 → 只 define A + 前 2 prereq（count=3·防爆）。"""
    b = core
    A = (1, 10)
    prereqs = [(1, 20 + i) for i in range(5)]
    table = {A: ("tA", prereqs)}
    for p in prereqs:
        table[p] = (f"t{p[1]}", [])
    t = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=_llm_factory())
    n = define_recursive(A, t, _provider(table), budget=3)
    assert n == 3   # budget 耗尽后余 prereq 跳过


# ============ TC5 atom 闭包（已已知不递归） ============

def test_recursive_define_atom_stop(core):
    """A→[B]·B atom（已已知）→ B 不 define 不递归其 prereq（count=1·只 A）。"""
    b = core
    A, B, C = (1, 10), (1, 11), (1, 12)
    table = {A: ("tA", [B]), B: ("tB", [C]), C: ("tC", [])}
    t = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=_llm_factory())
    n = define_recursive(A, t, _provider(table), atom_fn=lambda r: r == B)
    assert n == 1   # B atom → 不定义不递归 → 只 A
    assert _recorded_texts(b) == ["tA"]


# ============ TC6 bit-identical（RECORD 链 == REPLAY 链） ============

def test_recursive_define_record_replay_bit_identical(core):
    """MODE_RECORD 录递归链 → MODE_REPLAY 同 provider 走同形链·每 define 命中（bit-identical）。

    G8 核心：driver 确定性 + teacher.define hash 录放 → 两模式递归树同形 → 跨 run 复现。
    """
    b = core
    A, B, C, D = (1, 10), (1, 11), (1, 12), (1, 13)
    # 菱形 + 链：A→[B,C]·B→[D]·C→[D]·D→[]
    table = {A: ("tA", [B, C]), B: ("tB", [D]), C: ("tC", [D]), D: ("tD", [])}
    # RECORD 阶段
    t_rec = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=_llm_factory())
    n_rec = define_recursive(A, t_rec, _provider(table))
    texts_after_record = _recorded_texts(b)
    # REPLAY 阶段（新 teacher·零 LLM·同 provider）
    t_rep = RecordableLLMTeacher(b, mode=MODE_REPLAY)
    n_rep = define_recursive(A, t_rep, _provider(table))
    texts_after_replay = _recorded_texts(b)
    # 两模式 count 同 + 录制表不变（REPLAY 不新增·全命中）
    assert n_rec == n_rep == 4   # A,B,D,C（菱形 D memo·只定义一次）
    assert texts_after_record == texts_after_replay == ["tA", "tB", "tC", "tD"]


# ============ TC7 菱形 memo（D 经 B 已定义→经 C 不重定义） ============

def test_recursive_define_diamond_memo(core):
    """A→[B,C]·B→[D]·C→[D] → D 只 define 一次（visited memo·count=4 非 5）。"""
    b = core
    A, B, C, D = (1, 10), (1, 11), (1, 12), (1, 13)
    table = {A: ("tA", [B, C]), B: ("tB", [D]), C: ("tC", [D]), D: ("tD", [])}
    t = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=_llm_factory())
    n = define_recursive(A, t, _provider(table))
    assert n == 4   # A,B,D,C（D 第二次到访→ visited 跳过·非 5）
    assert _recorded_texts(b) == ["tA", "tB", "tC", "tD"]


# ============ TC8 MODE_OFF no-op（driver 跑·define 全 None·零录制） ============

def test_recursive_define_mode_off_noop(core):
    """MODE_OFF → teacher.define 全返 None·driver 仍跑（count=3 递归形不变）·零录制。"""
    b = core
    A, B, C = (1, 10), (1, 11), (1, 12)
    table = {A: ("tA", [B]), B: ("tB", [C]), C: ("tC", [])}
    t = RecordableLLMTeacher(b, mode=MODE_OFF)
    n = define_recursive(A, t, _provider(table))
    assert n == 3   # driver 递归形不变（define 调用数）
    assert _recorded_texts(b) == []   # MODE_OFF 零录制（退场·D 墙）


# ============ TC9 provider KeyError 闭项守（未知 ref 当原子停） ============

def test_recursive_define_provider_unknown_ref_atom_stop(core):
    """A→[B]·B 未知（provider KeyError）→ B 当原子停·不抛崩（count=1·只 A）。"""
    b = core
    A, B = (1, 10), (1, 11)
    table = {A: ("tA", [B])}   # B 不在 table → provider(B) KeyError
    t = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=_llm_factory())
    n = define_recursive(A, t, _provider(table))
    assert n == 1   # A 定义·B 未知→KeyError 当原子停·不抛崩
    assert _recorded_texts(b) == ["tA"]


# ============ TC10 默认守值（镜像 legacy MAX_DEFINE_DEPTH=5 / budget=12） ============

def test_recursive_define_default_guards():
    """默认 max_depth=5·budget=12（镜像 legacy learner.py·防 3^5=243 爆炸）。"""
    assert DEFAULT_MAX_DEPTH == 5
    assert DEFAULT_BUDGET == 12


# ============ TC11 malformed prereq ref → 闭项守 skip 不抛崩（审 F2） ============

def test_recursive_define_malformed_ref_skip(core):
    """provider 返 malformed prereq（3-tuple 非 ConceptRef）→ _valid_concept_ref 守入口 skip·不抛崩。"""
    b = core
    A = (1, 10)
    bad = (1, 2, 3)   # 非 2-int-tuple·caller 违约
    table = {A: ("tA", [bad])}
    t = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=_llm_factory())
    n = define_recursive(A, t, _provider(table))   # 不抛 ValueError
    assert n == 1   # 只 A·bad ref 入口校验 skip


# ============ TC12 provider 返 None prereqs → 当叶子不抛崩（审 F3） ============

def test_recursive_define_none_prereqs_leaf(core):
    """provider 返 (text, None)（caller 违约）→ coerce 当叶子·不抛 TypeError。"""
    b = core
    A = (1, 10)
    def bad_provider(ref):
        return ("tA", None)   # None 非 list/tuple
    t = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=_llm_factory())
    n = define_recursive(A, t, bad_provider)   # 不抛 TypeError
    assert n == 1
    assert _recorded_texts(b) == ["tA"]
