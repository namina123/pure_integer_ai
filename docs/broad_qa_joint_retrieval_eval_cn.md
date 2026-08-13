# 来源约束检索与证据选择联合评测

本页记录联合检索评测及其 successor family。它把两条此前分开的能力串成一条真实链：问题先从随机 20k Wikipedia 页面与冻结来源页组成的联合索引中检索，再从命中终页选择最多四条可独立核验的证据 span。

这不是开放域通用问答、自由生成或断奶结果。题目、来源标题、答案和阈值均在系统运行前冻结；questions、labels、source target 和 alias ledger 分离。所有大数据、索引和运行产物只位于显式 K 盘 run root。

## V1 开发结果

`PH2_BROAD_QA_JOINT_RETRIEVAL_EVIDENCE_PACK_V1` 排除了旧外部上下文 family 的 454 个标题，冻结 200 dev 与 300 held-out。439 个新标题中，437 个可解析到冻结快照终页，2 个缺失。

- 未接 alias：Recall@20 `59.5%`，top1 `58.0%`，evidence hit `32.5%`，`FAIL`。
- 接通显式 alias：Recall@20 `88.0%`，top1 `85.5%`，evidence hit `44.0%`，`FAIL`。
- 后续通用问式与有限页内重选探针没有闭合 60% 证据门；最终 v4 evidence hit 为 `43.0%`。

V1 的 300 问 held-out 没有运行。停止在同一 dev 上继续塑形后，项目冻结了全新的 successor family。

## V2 successor 冻结合同

`PH2_BROAD_QA_JOINT_RETRIEVAL_EVIDENCE_PACK_V2` 同时排除：

- 旧 external family 的 454 个问题标题；
- joint V1 的 439 个 source target 标题。

共排除 893 个标题，新旧 joint 标题交集为 0。V2 冻结 200 dev、300 held-out 和 451 个 source target；dev/held-out 标题域仍隔离。manifest SHA-256 为 `47f19f8a33fd9992842efb744c93862437da9faa4eb775be12dd58cbdee373e9`。

来源设施按冻结快照独立构建：

- 初始标题选择命中 449/451，2 个缺失；
- terminal alias 为 449 `RESOLVED`、2 `MISSING`；
- 目标索引为 449 pages、4,330 passages、369,227 terms；
- 与随机 20k 合并后的联合索引为 20,439 pages、113,231 passages、3,766,159 terms，SHA-256=`01450bfb115532e19ef3cbe43f8e6cf92c5de3900969eeb58aa76d2407777fa7`。

successor 查询不扫描完整 alias 表，而以 `alias_term` 做有界精确锚定。回答可返回最多四条同页、同修订的真实 passage span；每条都携带原文、raw span/hash、来源和所选文本。评分器逐条回查引用，只在全部引用有效后使用证据联合文本判定答案命中。任一引用被篡改时整条 ANSWER fail closed。

## V2 开发结果

200 问 dev 的 successor 结果为：

- Recall@20：`199/200 = 99.5%`；
- top1 source hit：`198/200 = 99.0%`；
- ANSWER citation valid：`190/190 = 100%`；
- 全分母 evidence hit：`107/200 = 53.5%`；
- 查询 p50/p95：`174.356/299.486 ms`；
- 状态：`FAIL`，未降低预先冻结的 60% evidence 门；
- aggregate SHA-256：`f5455954b206dc8eb0e11af6800ba5b5f139411c94e91e8c084711c481ab016b`。

为了区分算法失败与来源版本漂移，aggregate V2 还核验了冻结终页是否实际包含旧数据集的任一金答案：

- 当前终页含金答案：`126/200 = 63.0%`；
- 在这 126 个可覆盖问题中，证据命中：`107/126 = 84.9206%`；
- CMRC2018：覆盖 72，命中 62，条件命中 `86.1111%`；
- DRCD：覆盖 54，命中 45，条件命中 `83.3333%`；
- 失败分账：`SOURCE_GOLD_ABSENT_FROM_SNAPSHOT=74`、`GOLD_NOT_IN_EVIDENCE=18`、`NON_ANSWER=1`。

CMRC2018/DRCD 保存的是较早 Wikipedia context，而检索目标是 2026-07 冻结终页。74 题的精确金答案不在当前终页，不能通过继续调检索、降低阈值、修改分母或添加个案规则解决。条件命中率用于诊断，不替代全分母 53.5% 和全局 `FAIL`。

V2 的 300 问 held-out 没有运行，也没有发布联合 PASS receipt。

## 下一合同

下一阶段先建立新的来源版本对齐合同，而不是继续在当前 200 问 dev 上调参：

1. 在运行问答算法前，对完整候选池冻结标题/alias 解析与“金答案是否存在于目标快照终页”的来源覆盖 census。
2. 覆盖率与算法准确率始终分账；若从可覆盖总体冻结新评测 family，必须同时公开原始总体覆盖率，不能把来源缺失从项目能力边界中抹去。
3. 新 family 继续排除所有已消费标题域，并在任何开发运行前冻结 split、阈值、问题、标签和来源身份。
4. 只有新的 source-aligned dev 达门且算法冻结后，才允许唯一一次 held-out formal run。

当前可准确表述的能力是：20k 来源约束抽取式预览、外部给定上下文证据选择 formal PASS，以及联合检索 successor dev `FAIL`。V2 已证明页面检索和引用核验接近闭合，同时暴露了必须单独处理的来源版本对齐问题。
