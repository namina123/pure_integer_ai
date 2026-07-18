"""cognition.understanding.sense_lookup_hook — sense_lookup 注入式 hook（刀6 件7·摄入侧写真）。

make_sense_lookup(backend, space_id) -> SenseLookup | None
  生产 observe caller（formal_train）注入式 hook·查 sense_candidates 表返多义 sense 候选 ConceptRef list。
  gate SENSE_LOOKUP_MODE OFF → 返 None（observe sense_lookup=None·refers_to MultiRef 不产·退化 bit-identical）。
  gate ON → 返 callable：tok → [sense_ref, ...]（read_sense_candidates base_count>0 候选·不读 sc_tn 防循环）。

**防循环关键**：hook 只返 base_count>0 候选（boot 种先验·append-only·observe 不碰 base_count）·
  **不读 sc_tn 自身**（observe record_sense_token_seen 写 sc_tn·若 hook 读 sc_tn 则 observe 写→hook 读→
  MultiRef 产生→observe 又写·循环）。base_count 是 boot 写的稳定先验·hook 读 base_count 安全无环。

**时序**（反 theater 牙·plan 决断 5）：boot 段（formal_train boot）种 base_count 早于 stage loop observe →
  observe normalize 调 sense_lookup hook 时 sense_candidates 表已种 base_count → hook 返非空 → MultiRef 产生 →
  observe 各 sense record_sense_token_seen sc_tn++ → 摄入侧写真（非死列表）。

铁律：纯整数（ConceptRef list）/ 确定性（read_sense_candidates NodeRef 升序·bit-identical）/ 单向依赖
  （L4 understanding·import storage L0 + config crosscut·不环）/ gate 二分（SENSE_LOOKUP_MODE 默认 OFF·CI===生产）。
诚实边界：hook 返候选·不判真消歧（#479 墙·定义权归教师）·真消歧在理解侧 recognize（IS_A 共祖结构选优·
  非语义接地·stable≠correct·同 selection_pref_count 范式）。
"""
from __future__ import annotations

from pure_integer_ai.config import gates
from pure_integer_ai.storage.sense_candidates import read_sense_candidates, sense_surface_hash


def make_sense_lookup(backend, space_id: int):
    """造 sense_lookup hook（observe normalize_to_concept 注入式·refers_to.py SenseLookup 协议）。

    gate SENSE_LOOKUP_MODE OFF → 返 None（observe sense_lookup=None·MultiRef 不产·退化 bit-identical·
      退化链 5 步·plan 决断 5）。
    gate ON → 返 callable：tok surface → list[sense ConceptRef]（base_count>0 候选·NodeRef 升序·确定性）。

    caller（formal_train observe caller）每次 item 调（闭包轻量·无须 boot 缓存）。
    """
    if not getattr(gates, "SENSE_LOOKUP_MODE", False):
        return None   # gate OFF·observe 走原 OOV/lemma 路径·MultiRef 不产·退化 bit-identical

    def _sense_lookup(tok: str):
        """tok → 多义 sense 候选 ConceptRef list（base_count>0·boot 种先验·不读 sc_tn 防循环）。"""
        sh = sense_surface_hash(tok)
        candidates = read_sense_candidates(backend, space_id, sh)
        # 只返 base_count>0 候选（boot 种·observe 不碰 base_count·hook 读安全·不循环）
        # sc_tn 是 observe 自身频次·hook 不读 sc_tn（防 observe 写→hook 读→MultiRef→observe 又写 循环）
        return [sense_ref for sense_ref, base, _sc_sn, _sc_tn in candidates if base > 0]

    return _sense_lookup
