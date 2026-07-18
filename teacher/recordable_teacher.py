"""teacher.recordable_teacher — RecordableLLMTeacher 录放层（§十一 E4/E10 + §8.1c-bis 来源③）。

录放层是断奶前教师元定义/define/QA/G5 ground-truth 的唯一可复现通道（录一次重跑零 LLM）。
  MODE_OFF    退场（断奶后·新遇未知靠 SHADOW+晋升闸·D 墙·所有调用返 None）
  MODE_RECORD 离线录制（断奶前·调注入式 llm_call→verify_teacher_boundary→写 recording 表）
  MODE_REPLAY 重跑/断奶后运行时（读 recording·零 LLM·bit-identical）

  miss→None 无 fallback（E4·最危险是静默 fallback 真 LLM 破 bit 级可复现）：
    REPLAY 模式 key 未命中 → 显式返 None（caller 降级标记该样本 reward 不落 strength 防脏信号）
    断奶前可显式触发补录（切 MODE_RECORD·新 run_id 非静默）·断奶后无补录路径=纯报错。

  recording 表（append-only·E2 切片一致性前提）：
    (call_hash, kind, content_type, response_text, response_int)
    call_hash = Hasher("TEACHER_REC").h63((kind, canonical_args))  # 确定性键·bit-identical
    幂等：同 key 重录跳过（录一次即足·守可复现）。

  rate_limiter AIMD（E10·护录制不触发 API 封·纯整数窗口）：
    录制是断奶前不可重建产物（错过窗口断奶后无法补录）·AIMD 加性增乘性减限流·
    护"录制完整"非"录制快"·宁可慢不可缺。重跑走录放层零 LLM 零限流=bit-identical。

接口（教师给事实判断非推理规则·§9 A2）：
  define(ref, text, *, content_type) -> dict | None       KIND_DEFINE（元定义/知识分流）
  judge_ground_truth(output, dag_path, graph) -> int      KIND_REWARD（G5/C6 Mode A·self_proof_fn 注入）
  confirm_causes(a, b) -> bool | None                     KIND_DEFINE（CAUSES 来源③·隐式因果确认·断奶前）
  label_error(ref, *, error_type) -> dict | None          KIND_ERROR_LABEL

铁律：纯整数（call_hash h63/response_int 0..1000/content_type 整数枚举）/ 确定性（key=Hasher·幂等）/ 不走外挂 LLM
  （REPLAY 零 LLM·RECORD 离线墙钟合法但重跑零限流）/ append-only（recording 表·E2 切片）/ 外部只启发
  （verify_teacher_boundary 拒越界·教师给事实判断非边关系真伪）/ 不写死（kind/content_type 元定义枚举）。
诚实边界：教师给事实判断非语义理解（接地墙）/ 录制时机墙钟非 bit-identical（重跑走录放层零 LLM 才 bit-identical）/
  miss→None 是诚实降级（该样本 reward 不落 strength）/ 断奶后退场是 D 墙。
"""
from __future__ import annotations

from typing import Any, Callable

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT, TYPE_TEXT
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import register_extension_table
from pure_integer_ai.cognition.shared.types import ConceptRef
from pure_integer_ai.teacher.teacher_boundary import (
    verify_teacher_boundary, is_acceptable,
    KIND_DEFINE, KIND_REWARD, KIND_ERROR_LABEL, KIND_NAME,
)

# ---- 模式 ----
MODE_OFF = 0       # 退场（断奶后）
MODE_RECORD = 1    # 离线录制（断奶前·调 llm_call）
MODE_REPLAY = 2    # 重跑/运行时（读 recording·零 LLM）

# content_type 枚举（教师给的内容类型分流·§十一 line733 元定义 PRIMARY 直落 / 知识 sign=0 检疫）
CONTENT_META_DEFINITION = 1   # 元定义层（出厂硬件·PRIMARY 直落核心·冷启动阶段1）
CONTENT_KNOWLEDGE = 2         # 知识定义（sign=0 检疫·有 reward 信号才过闸）

# G5/C6 ground-truth 返值（纯整·self_proof_fn 契约：1=pass / 0=fail）
GT_PASS = 1
GT_FAIL = 0

# rate_limiter AIMD 默认（E10·纯整数窗口·护录制不触发 API 封）
AIMD_WINDOW_MIN = 1
AIMD_WINDOW_MAX = 64
AIMD_ADD = 1        # 加性增（成功 +1）
AIMD_MULT_NUM = 1   # 乘性减 num/den（失败 ×1/2）
AIMD_MULT_DEN = 2


# ---- recording 表 ----

_RECORDING_COLUMNS = [
    ("call_hash", TYPE_INT),       # Hasher.h63((kind, canonical_args))·确定性键
    ("kind", TYPE_INT),            # 内容类型（KIND_*）
    ("content_type", TYPE_INT),    # 元定义/知识分流
    ("response_text", TYPE_TEXT),  # 文本响应（define/label·可空）
    ("response_int", TYPE_INT),    # 整数响应（ground-truth 0/1·confirm 0/1）
]
_RECORDING_INDEXES = [
    ("call_hash",),
    ("kind",),
]


def register_recording_table(backend: StorageBackend) -> None:
    """注册教师录放表（append-only·E2 切片一致性·非核心扩展表）。"""
    register_extension_table(backend, "teacher_recording",
                             _RECORDING_COLUMNS,
                             disc.DISC_APPEND_ONLY, _RECORDING_INDEXES)


# ---- AIMD rate_limiter（E10） ----

class _AIMDLimiter:
    """AIMD 加性增乘性减限流（纯整数窗口·护录制不触发 API 封·E10）。

    窗口调整基于反馈（成功增/失败减）确定性非墙钟随机。护"录制完整"非"录制快"。
    """

    def __init__(self) -> None:
        self._window = AIMD_WINDOW_MIN
        self._in_flight = 0

    def acquire(self) -> bool:
        """是否允许发起一次录制调用（窗口内放行）。"""
        if self._in_flight < self._window:
            self._in_flight += 1
            return True
        return False

    def release(self, *, success: bool) -> None:
        """反馈：成功加性增 / 失败乘性减（确定性·非墙钟随机）。"""
        if self._in_flight > 0:
            self._in_flight -= 1
        if success:
            self._window = min(AIMD_WINDOW_MAX, self._window + AIMD_ADD)
        else:
            self._window = max(AIMD_WINDOW_MIN,
                               (self._window * AIMD_MULT_NUM) // AIMD_MULT_DEN)

    @property
    def window(self) -> int:
        return self._window


# ---- RecordableLLMTeacher ----

# LLM 调用契约（注入式·MODE_RECORD 离线用·MODE_REPLAY 永不调）：
#   llm_call(kind: int, args: tuple) -> dict  （含 kind/text/response_int/content_type）
LLMCall = Callable[[int, tuple], dict]


class RecordableLLMTeacher:
    """录放层教师（断奶前在位 / 断奶后退场·§十一 E4/E10）。

    mode 决定行为：OFF→返 None / RECORD→调 llm_call 录制 / REPLAY→读 recording miss→None。
    所有调用经 verify_teacher_boundary 机械核查（白黑词汇表·§9 A2）·违例拒写。
    """

    def __init__(self, backend: StorageBackend, *,
                 mode: int = MODE_OFF,
                 llm_call: LLMCall | None = None,
                 rate_limiter: _AIMDLimiter | None = None,
                 source_id: int = 0) -> None:
        self._b = backend
        self._mode = mode
        self._llm_call = llm_call
        self._limiter = rate_limiter or _AIMDLimiter()
        self._hasher = Hasher("TEACHER_REC")
        self._source_id = source_id   # D3·录放层 source 标识（裁判源独立性比对用·§十一 #4-bis）
        # recording 表须已注册（caller bootstrap 后调 register_recording_table）
        # #1143 统计层断奶·教师干预计数（intervention_rate 测量源·fadeout 锚点 data）：
        # 每次 _call（REPLAY hit/miss·RECORD 录）+1·MODE_OFF 不计（退场无干预）。
        # additive 计数器·不改录放行为·bit-identical。formal_train 轮边界 snapshot delta → intervention_rate。
        self.call_count = 0

    # ---- 键与录制底层 ----

    def _key(self, kind: int, args: tuple) -> int:
        """确定性录制键 = Hasher.h63((kind, canonical_args))·bit-identical。"""
        return self._hasher.h63((kind, args))

    def _lookup(self, key: int) -> dict[str, Any] | None:
        rows = self._b.select("teacher_recording", where={"call_hash": key}, limit=1)
        return rows[0] if rows else None

    def _record(self, key: int, response: dict[str, Any]) -> None:
        """录制响应（append-only·幂等·verify_teacher_boundary 拒越界后写）。"""
        if not is_acceptable(response):
            return   # 越界拒写（§9 A2·不进核心·防注入边语义）
        existing = self._lookup(key)
        if existing is not None:
            return   # 幂等：同 key 已录·跳过（录一次即足·守可复现）
        self._b.insert("teacher_recording", {
            "call_hash": key,
            "kind": int(response.get("kind", 0)),
            "content_type": int(response.get("content_type", CONTENT_KNOWLEDGE)),
            "response_text": response.get("text"),
            "response_int": int(response.get("response_int", 0)),
        })

    def _call(self, kind: int, args: tuple) -> dict[str, Any] | None:
        """统一录放分发（mode 决定·miss→None 无 fallback·E4）。"""
        key = self._key(kind, args)
        if self._mode == MODE_OFF:
            return None   # 退场（断奶后·D 墙·新遇未知靠 SHADOW+晋升闸）
        self.call_count += 1   # #1143 教师干预计数（非 OFF = 系统问教师 = 依赖·fadeout 测量源）
        if self._mode == MODE_REPLAY:
            row = self._lookup(key)
            if row is None:
                return None   # E4·miss→None 无 fallback（不静默调真 LLM·破 bit 可复现）
            return {"kind": row["kind"], "content_type": row["content_type"],
                    "text": row["response_text"], "response_int": row["response_int"]}
        # MODE_RECORD：离线调 llm_call 录制（rate_limiter AIMD 护·E10）
        if self._llm_call is None:
            raise RuntimeError("MODE_RECORD 须注入 llm_call（断奶前离线录制）")
        if not self._limiter.acquire():
            # 窗口满·等待重试（首次录制墙钟非 bit-identical 部分·重跑走录放层零限流）
            return None
        try:
            response = self._llm_call(kind, args)
        finally:
            # 释放时反馈（成功增/失败减·确定性）——这里假定调用本身完成即 success
            self._limiter.release(success=True)
        self._record(key, response)
        return response

    # ---- 教师接口（事实判断·§9 A2 白词汇表） ----

    def define(self, ref: ConceptRef, text: str, *,
               content_type: int = CONTENT_KNOWLEDGE) -> dict[str, Any] | None:
        """教师定义（KIND_DEFINE·元定义 PRIMARY 直落 / 知识 sign=0 检疫按 content_type 分流）。

        ref 概念点 + text 定义文本。返 {kind, content_type, text} 或 None（miss/退场/越界）。
        断奶前教师走录放层确认因果/同指候选合法（§8.1c-bis 来源③）·断奶后退场。
        """
        sid, lid = ref
        assert_int(sid, lid, _where="teacher.define.ref")
        args = ("define", sid, lid, text, content_type)
        resp = self._call(KIND_DEFINE, args)
        if resp is None:
            return None
        resp = dict(resp)
        resp.setdefault("content_type", content_type)
        return resp

    def judge_ground_truth(self, output: Any, dag_path: Any,
                           graph: Any) -> int | None:
        """G5/C6 Mode A 教师 ground-truth（KIND_REWARD·self_proof_fn 注入契约）。

        返 GT_PASS(1)/GT_FAIL(0)·miss/退场 → None（E4 miss→None 无 fallback 红线·stub #3 修：
        caller 可区分"教师真判 PASS"与"miss 占位"·self_proof_check None→veto 防脏 reward 落 strength）。
        断奶前算术/代码域唯一承重正确性件·断奶后退场（Mode B self-consistency·defer）。
        """
        # ground-truth 键：用 output 的 produced 概念 ref 序 + dag_path sink 作 canonical args
        parts = getattr(output, "parts", None) or []
        produced = tuple((p.unit[0], p.unit[1]) for p in parts if getattr(p, "unit", None) is not None)
        sink = getattr(dag_path, "sink", None)
        sink_arg = (sink[0], sink[1]) if sink is not None else (-1, -1)
        args = ("gt", produced, sink_arg)
        resp = self._call(KIND_REWARD, args)
        if resp is None:
            return None   # E4·miss/退场 → None 无 fallback（caller 区分·防占位 pass 产脏 reward）
        return GT_PASS if int(resp.get("response_int", GT_PASS)) != GT_FAIL else GT_FAIL

    def confirm_causes(self, a: ConceptRef, b: ConceptRef) -> bool | None:
        """CAUSES 来源③ 隐式因果确认（KIND_DEFINE·断奶前·无指向词隐式因果候选）。

        教师判"A 是否导致 B"事实判断（非推理规则）·确认后产 CAUSES 边进核心填缺口。
        返 True/False / None（miss/退场·断奶后此来源消失回 D 墙）。
        """
        a_sid, a_lid = a
        b_sid, b_lid = b
        assert_int(a_sid, a_lid, b_sid, b_lid, _where="teacher.confirm_causes")
        args = ("confirm_causes", a_sid, a_lid, b_sid, b_lid)
        resp = self._call(KIND_DEFINE, args)
        if resp is None:
            return None
        return int(resp.get("response_int", 0)) != 0

    def confirm_is_a(self, child: ConceptRef, parent: ConceptRef) -> bool | None:
        """IS_A 来源③ 歧义确认（KIND_DEFINE·断奶前·对称 confirm_causes·补 IS_A 来源③ 不对称缺口）。

        教师判"child 是否是 parent 的 proper subset"事实判断（非推理规则）·确认后产/晋 IS_A 边
        （SHADOW 候选→PRIMARY·M9 测度走 initial_strength 非 sn/tn·§十五决策5）。
        返 True/False / None（miss/退场·断奶后此来源消失回 D 墙·TEACHER_MODE OFF→None）。
        """
        c_sid, c_lid = child
        p_sid, p_lid = parent
        assert_int(c_sid, c_lid, p_sid, p_lid, _where="teacher.confirm_is_a")
        args = ("confirm_is_a", c_sid, c_lid, p_sid, p_lid)
        resp = self._call(KIND_DEFINE, args)
        if resp is None:
            return None
        return int(resp.get("response_int", 0)) != 0

    def label_error(self, ref: ConceptRef, *, error_type: int) -> dict[str, Any] | None:
        """教师错误标签（KIND_ERROR_LABEL·训练信号）。"""
        sid, lid = ref
        assert_int(sid, lid, error_type, _where="teacher.label_error")
        args = ("label_error", sid, lid, error_type)
        return self._call(KIND_ERROR_LABEL, args)

    def name(self, ref: ConceptRef, *, surface: str) -> dict[str, Any] | None:
        """教师命名（KIND_NAME·同指别称/性质A 来源②③）。"""
        sid, lid = ref
        assert_int(sid, lid, _where="teacher.name")
        args = ("name", sid, lid, surface)
        return self._call(KIND_NAME, args)

    # ---- E4 replay 覆盖率（续训前置校验） ----

    def key_for(self, kind: int, args: tuple) -> int:
        """公开确定性键（供续训前置算 needed_keys 做 replay 覆盖率校验·E4）。"""
        return self._key(kind, args)

    def replay_coverage(self, needed: list[tuple[int, tuple]]) -> tuple[int, int]:
        """replay 覆盖率（E4·续训前置校验·未达标禁续训）。

        needed = [(kind, args), ...] 本次续训预期会 replay 的调用。
        返 (recorded, total)·recorded/total ≥ 阈值才允许续训（防 miss→None 静默降级破可复现）。
        """
        total = len(needed)
        if total == 0:
            return (0, 0)
        recorded = 0
        for kind, args in needed:
            if self._lookup(self._key(kind, args)) is not None:
                recorded += 1
        return (recorded, total)

    def recorded_keys(self, kind: int | None = None) -> list[int]:
        """列出已录制的 key（按 key 升序·确定性·覆盖率校验/补录用）。"""
        where = {"kind": kind} if kind is not None else None
        rows = self._b.select("teacher_recording", where=where)
        return sorted({r["call_hash"] for r in rows})

    @property
    def mode(self) -> int:
        return self._mode

    @property
    def source_id(self) -> int:
        """D3·录放层 source 标识（裁判源独立性比对用·§十一 #4-bis line710）。"""
        return self._source_id

    @property
    def rate_window(self) -> int:
        """当前 AIMD 窗口（E10·观测/调试用）。"""
        return self._limiter.window
