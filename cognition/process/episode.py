"""cognition.process.episode — 模块9 episode 主循环 orchestrator（M5 wiring 单点化）。

三大建模闭环的"环"靠主循环闭合：步进→生成→judge→反传→回步进。本模块把 wiring 单点化
（§十三D-E3 line1070 + 模块8 接口注释·散落两处接口注释无单点伪代码→实施易漏接
DEAD_END→reward<0 转换·防塌柱② greenfield·漏接则防塌缺柱② + 收敛判据恒假收敛）。

  episode_loop(input, subgraph_edges, seeds, workmem, intent, *, generate_fn, judge_fn, ...)
    -> (output, Episode)
  reward 符号契约（§十三D-E3）：judge 产 reward_judge≥0 / 步进死路产 reward_dead<0 /
    propagate 接收可负。两半边都进 propagate 都 tn++（REACHED_SINK 但 judge veto reward=0
    走 judge 分支产 0 / DEAD_END 走 if 分支产负）。
  wiring 是 greenfield production 落点（§十三D-E3）·非既有·实施按此契约落。

generate_fn / judge_fn 注入（卷三模块1/3 产出·Stage 5 接线·接口归系统）：
  generate_fn(path_result, workmem, input) -> output（卷三模块1·target_lang 偏好·C1 防跨语言）
  judge_fn(output, path_result, input, workmem) -> (reward, G_meta)（卷三模块3·ΠG·ΣwJ·reward≥0）
  首版未接卷三时传 None→output=None / REACHED_SINK reward=0（veto 语义）·DEAD_END reward<0 仍守。

gate：闭环核心默认 ON（H1）。确定性 bit-identical（无墙钟·同输入同输出）。
诚实边界：orchestrator 只接线不判语义（stable≠correct）。
"""
from __future__ import annotations

from typing import Any, Callable

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.storage.edge_store import EdgeStore
from pure_integer_ai.storage.experience_count import pack_ctx_code
from pure_integer_ai.cognition.shared.types import (
    InputPayload, IntentType, Episode, PathResult, GMeta, ConceptRef,
    TERMINAL_REACHED_SINK, TERMINAL_DEAD_END, REWARD_DEAD_END, G_META_DEAD_END,
)
from pure_integer_ai.cognition.process.dag_path import dag_path_step
from pure_integer_ai.cognition.process.reward_propagate import propagate_reward

GenerateFn = Callable[..., Any]
JudgeFn = Callable[..., tuple[int, GMeta]]


def _output_words(output: Any) -> list[str]:
    """从生成结果取词（落点④ 读 OutputResult.words·F6·词是生成侧产物 path 无词）。

    OutputResult.words 扁平 part.words（types 已落 property）·output=None/无 words → []。
    parts 是 OutputPart 对象非 str·旧版 isinstance(w,str) 恒返 []（stub #7 修：读 .words property）。
    """
    if output is None:
        return []
    words = getattr(output, "words", None)
    if words is None:
        return []
    return [w for w in words if isinstance(w, str)]


def _ctx_tag(input_payload: InputPayload, intent: IntentType) -> tuple:
    """context_tag 多维 (domain, modality, task, intent_type)·F7 治上下文串味。

    task 维首版 placeholder 0（InputPayload 尚无 task 字段·随 §十一6Q 扩·defer）。
    """
    task = 0   # defer（§十一6Q task 字段未落 InputPayload·占位 0）
    return (input_payload.domain, input_payload.modality, task, intent.type)


def episode_loop(input_payload: InputPayload,
                 subgraph_edges: list[dict[str, Any]],
                 seeds: list[ConceptRef], workmem: Any, intent: IntentType, *,
                 generate_fn: GenerateFn | None = None,
                 judge_fn: JudgeFn | None = None,
                 edge_store: EdgeStore, backend: Any,
                 current_seq: int = 0,
                 memory_active: bool = False,
                 coverage_threshold: int = 0,
                 memory_read: Any = None) -> tuple[Any, Episode]:
    """episode 主循环（wiring 单点化·M5）。

    返 (output, Episode)·Episode 聚合层供防塌/收敛验收消费（F5）。
    """
    assert_no_float(current_seq, _where="episode_loop")
    # —— 卷三 handoff：intent 写入 input_payload 供卷三 judge 读 input.intent（自锚于输入·§十四） ——
    input_payload.intent = intent
    # —— 复合 key ctx_code（第二刀·阶段6·ctx_tag 四维位打包·步进读桶==feed 写桶·防混淆频次） ——
    _ctx_tag_tuple = _ctx_tag(input_payload, intent)
    _ctx_code = pack_ctx_code(*_ctx_tag_tuple)
    # —— 过程建模（卷二模块4 步进产 DAG-path） ——
    path_result: PathResult = dag_path_step(
        subgraph_edges, seeds, workmem, intent,
        current_seq=current_seq, memory_active=memory_active,
        backend=backend, edge_store=edge_store, ctx_code=_ctx_code,
        key_skeleton=input_payload.key_skeleton,
        coverage_threshold=coverage_threshold)

    # —— 结果建模（卷三模块1 生成输出·target_lang 偏好·C1 防跨语言·F2） ——
    output = generate_fn(path_result, workmem, input_payload) \
        if generate_fn is not None else None

    # —— reward 生产（两生产者·M5 单点 wiring） ——
    if path_result.terminal == TERMINAL_DEAD_END:
        reward = REWARD_DEAD_END   # <0 纯整常量·步进死路产负（§十三D-E3·防塌 C3 greenfield）
        g_meta = G_META_DEAD_END   # 死路无 judge·meta 标记非 judge 产出（D1·5字段全 False）
    else:   # REACHED_SINK
        if judge_fn is not None:
            reward, g_meta = judge_fn(output, path_result, input_payload, workmem)
            assert reward >= 0, f"judge 输出约束 reward≥0·got {reward}"
        else:
            # 卷三未接·REACHED_SINK 默认 reward=0（veto 语义·诚实占位）
            reward, g_meta = 0, GMeta()

    # —— reward 反传（卷二模块8·两半边都进·R1 episode 级符号） ——
    propagate_reward(path_result, _output_words(output), reward,
                     _ctx_tag_tuple, intent.type, workmem,
                     edge_store=edge_store, backend=backend, memory_read=memory_read)
    # reward>0 sn++&tn++ / reward==0 tn++(veto) / reward<0 tn++(死路)·都 tn++ 破永正

    # —— F5 Episode 聚合层（防塌/收敛验收消费 Episode 非 OutputResult） ——
    vetoed = path_result.terminal == TERMINAL_REACHED_SINK and reward == 0
    episode = Episode(
        episode_id=0,
        run_id=getattr(workmem, "round_id", 0),
        input=input_payload,
        output=output,
        reward=reward,
        ref=path_result.sink,
        terminal=path_result.terminal,
        pr_vector=getattr(workmem, "pr_vector", {}),
        judge_G4_active=g_meta.G4_vetoed,
        judge_G2p_active=g_meta.G2p_vetoed,
        judge_G3a_active=g_meta.G3a_vetoed,
        judge_G3b_active=g_meta.G3b_vetoed,
        judge_G5_active=g_meta.G5_vetoed,
        judge_veto_count=1 if vetoed else 0,
        dead_end_count=1 if path_result.terminal == TERMINAL_DEAD_END else 0,
        vetoed=g_meta.vetoed,
        exploration_injected=getattr(path_result, "exploration_injected", False),
    )
    # #728 A 半：tri_space caller 接线（episode 末尾·propagate_reward 后·为下 episode 准备 workmem.replay/exclude）。
    # gate MEMORY_REPLAY_MODE OFF → tri_space early-return（workmem.replay 永空·dag_path local_seeds == seeds·bit-identical）。
    # gate ON + memory_read 传入 → tri_space query memory → 写 workmem.replay（info_ref concept ref·每 episode 清 fresh）。
    # lazy import 照 dag_path EXPLORATION_MODE lazy import anti_collapse 范式避 cognition.process ↔ cognition.result 顶层循环。
    from pure_integer_ai.cognition.result.tri_space import tri_space_coordination
    tri_space_coordination(episode, workmem=workmem, memory_space=memory_read)
    return output, episode
