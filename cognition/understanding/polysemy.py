"""cognition.understanding.polysemy — 模块7 多义 sense 摄入（预处理 QID/synset → 1:N 挂概念）。

预处理离线（非 observe 热路径在线查·§H5）：训练前 Wikidata dump + WordNet synset →
QID/synset → 概念映射表。observe 消费预处理结果（QID 已解析·免网络延迟/限流）。

  - 同词多挂 1:N（§B12·REFERS_TO 边层）·消歧在理解侧 recognize 选（结构拟合·IS_A 共祖·刀6 修正·非生成侧）
  - 断奶后新遇多义词无预处理 QID → 回 #479 W2 truth 墙（sense 真义·**非 D 物理接地墙**）collide_score 拓扑兜底（§H5/§B12 消歧上限）
  - C2 不覆盖缺口补：C2 是学习侧（reward 标多义）·本模块是摄入侧 sense 消歧

首版：preprocess_sense_disambiguation 是离线管线骨架（外部 dump 注入·占位）·
observe 经 refers_to.sense_lookup hook 消费映射表。映射表 in-memory（run-scoped）·
持久化随 Stage 6 续训 defer。

**刀6 件7 修正**：本 SenseMapping 是 **in-memory 测试脚手架非生产路径**（preprocess_sense_disambiguation
零生产 caller·仅 test_stage3 用）。生产 sense 摄入侧 = `storage/sense_candidates.py` 表（boot 种 base_count +
observe 写 sc_tn·持久化跨 run）+ `sense_lookup_hook.make_sense_lookup`（生产 observe caller 注入）。消歧在
理解侧 recognize（formal_train clone aligning_root 逐 sense 试骨架对齐·ATTR_SLOT_ROLE IS_A 共祖选优·#479 墙·
结构选优 stable≠correct 非语义消歧·同 selection_pref_count 范式）。
"""
from __future__ import annotations

from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import ConceptRef


class SenseMapping:
    """QID/synset → 概念映射表（预处理离线产物·observe 读）。

    mapping: {token: [ConceptRef, ...]}（1:N·activate_candidates 返全部·禁取首）。
    首版 in-memory·由 preprocess_sense_disambiguation 填·observe 经 sense_lookup 读。
    """

    def __init__(self) -> None:
        self._mapping: dict[str, list[ConceptRef]] = {}

    def register(self, token: str, ref: ConceptRef) -> None:
        """预处理注册 token → concept（1:N·同 token 多 sense 各注册）。"""
        self._mapping.setdefault(token, []).append(ref)

    def lookup(self, token: str) -> list[ConceptRef]:
        """observe 消费（已解析·非在线查·1:N 返全部）。"""
        return list(self._mapping.get(token, []))

    def __contains__(self, token: str) -> bool:
        return token in self._mapping


def preprocess_sense_disambiguation(concept_index: ConceptIndex,
                                    *, space_id: int,
                                    wikidata_dump: list[tuple[str, int]] | None = None,
                                    wordnet_dump: list[tuple[str, int]] | None = None,
                                    ) -> SenseMapping:
    """预处理管线（一次性·训练前·非 observe 热路径）。

    wikidata_dump / wordnet_dump: [(canonical_token, qid_or_synset_id), ...]（外部注入）。
    per-QID 概念点（Q89 apple-fruit ↔ 中文苹果·Q3122 Apple-Inc ↔ 中文苹果公司·分开）。
    返回 SenseMapping 供 observe sense_lookup 读。
    """
    mapping = SenseMapping()
    for token, qid in (wikidata_dump or []):
        ref = concept_index.ensure(qid, space_id=space_id, tier=TIER_PRIMARY)
        mapping.register(token, ref)
    for lemma, synset_id in (wordnet_dump or []):
        ref = concept_index.ensure(synset_id, space_id=space_id, tier=TIER_PRIMARY)
        mapping.register(lemma, ref)
    return mapping
