"""cognition.understanding.pronoun_features — 代词特征元定义（出厂硬件·§十一#2-bis 性质B 软兜）。

代词 anaphora 的特征（人称/数/生命性/性别）= **元定义层固化**（同 cue_words·非语义规则·
reward 不调·断奶前后不变·§九铁律 enum 例外·同 space_routing META_DEFINITION）。

**非接地墙**（B5·元定义可自足）：人称/数/生命性/性别是语言出厂硬件（"他"=3 人称+男性+人类
是中文代词系统事实·非外部语义资源）·故可纯元定义表·不需 lemma/sense 那类离线预处理源。

消费：refers_occurrence.resolve_pronoun_occurrence 把特征概念作 PROPERTY 边进 PR 种子 e（软兜·
防 PR 软排序把"他"指向"苹果"非人称·§十四J4 低②）·硬过滤 defer。

每个特征组合 = 一个特征概念 int（refers_occurrence 作 content_hash ensure 进记忆空间 PROPERTY 边）。
中英 surface 不撞（"他"≠"he"）·单表 lang-agnostic（lookup 签名 (tok)->int|None·无 lang 参·
refers_to.PronounFeatureLookup 契约）。

诚实边界：
  - 特征是结构锚（防 PR 误排）非"词义→所指"映射（不判"他"指谁·只标人称/数/生命性/性别）。
  - exact token 匹配（caller 须代词切独立 token·首版纪律）·不命中返 None（守反统计契约）。
  - 首版 anaphora 人称代词 only·前指/指示代词（这/那/此）defer（§十一#2·同 is_pronoun 范围）。
"""
from __future__ import annotations

# 代词特征概念（人称×数×生命性×性别·每组合唯一 int·作记忆空间特征概念 content_hash）
# 100 段预留·101..= 代词特征概念（避开常见 concept hash 段·纯元定义常量·非写死语义规则）
FEAT_3P_SG_HUMAN_MALE = 101      # 他 / he / him（3 人称·单数·人类·男）
FEAT_3P_SG_HUMAN_FEMALE = 102    # 她 / she / her（3 人称·单数·人类·女）
FEAT_3P_SG_NONHUMAN = 103        # 它 / it（3 人称·单数·非人类）
FEAT_3P_PL_HUMAN = 104           # 他们（3 人称·复数·人类·男/混）
FEAT_3P_PL_HUMAN_FEMALE = 105    # 她们（3 人称·复数·人类·女）
FEAT_3P_PL_NONHUMAN = 106        # 它们（3 人称·复数·非人类）
FEAT_3P_PL_GENERIC = 107         # they / them（3 人称·复数·人/非人歧义·归 generic）

# 元定义 surface → 特征概念（出厂硬件·exact token 匹配·中英不撞单表）
_PRONOUN_FEATURES: dict[str, int] = {
    # 中文
    "他": FEAT_3P_SG_HUMAN_MALE,
    "她": FEAT_3P_SG_HUMAN_FEMALE,
    "它": FEAT_3P_SG_NONHUMAN,
    "他们": FEAT_3P_PL_HUMAN,
    "她们": FEAT_3P_PL_HUMAN_FEMALE,
    "它们": FEAT_3P_PL_NONHUMAN,
    # 英文
    "he": FEAT_3P_SG_HUMAN_MALE,
    "him": FEAT_3P_SG_HUMAN_MALE,
    "she": FEAT_3P_SG_HUMAN_FEMALE,
    "her": FEAT_3P_SG_HUMAN_FEMALE,
    "it": FEAT_3P_SG_NONHUMAN,
    "they": FEAT_3P_PL_GENERIC,
    "them": FEAT_3P_PL_GENERIC,
}


def lookup_pronoun_features(tok: str) -> int | None:
    """代词 surface → 特征概念 int（元定义出厂硬件·exact 匹配·不命中 None）。

    作 PROPERTY 边进 PR 种子 e 软兜（refers_occurrence 消费·防 PR 软排序误排人称）。
    非代词 / 未收录 → None（refers_to.is_pronoun 范围内但前指/指示代词 defer）。
    """
    return _PRONOUN_FEATURES.get(tok)
