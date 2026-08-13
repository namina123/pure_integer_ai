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

## 来源对齐 census

下一合同已经按 V2 暴露的问题执行。完整候选冻结先排除 external、joint V1 和 joint V2 共 1,344 个已消费标题，再从 48,129 个合格外部问题中保留 10,061 个自然包含来源标题的问题。census 在运行新 family 问答前逐题检查 alias、当前终页、完整可见页面文本和实际索引前 12 个 passage：

- `SOURCE_ALIGNED=7,189`，原始候选总体覆盖率 `71.4541%`；
- `GOLD_ABSENT_FROM_TERMINAL_REVISION=1,854`；
- `GOLD_PRESENT_OUTSIDE_PROJECTED_PASSAGES=790`；
- `SOURCE_ALIAS_MISSING=159`；
- `GOLD_ONLY_IN_RAW_WIKITEXT=65`；
- `PASSAGE_PROJECTION_DIVERGES_FROM_FULL_PAGE=4`。

只有答案同时存在于完整可见页面文本和实际 passage 投影的题目进入新 family。这个选择规则在问答运行前冻结；原始总体的 28.5459% 未覆盖部分继续公开分账，不能用新 family 的准确率替代来源总体覆盖率。census SHA-256 为 `0809f96843c11bec6264065fb166498fc73e3df4a325833711d4a66bc7dc5823`。

## Source-aligned family 与开发门

新 family 冻结 200 dev、300 held-out，CMRC2018/DRCD 各半；dev 使用 182 个标题，held-out 使用 277 个标题，彼此和全部前代已消费标题域无重叠。family manifest SHA-256 为 `82f4d641cf44c553594c8a5610b071e1e3ec09197a6bcf562d9c838d6dfcd666`。

459 个来源页形成 4,514 passages、390,483 terms 的目标索引；与随机 20k 合并后的联合索引为 20,449 pages、113,431 passages、3,781,174 terms，SHA-256=`17bdea8850ca6afea3637fdab2bd4f58fa90fc0c3df5ea04ebf1a697a4c31cab`。开发结果为 Recall@20 `200/200`、top1 `200/200`、ANSWER citation valid `195/195`、evidence hit `165/200=82.5%`，状态 `PASS`，aggregate SHA-256=`bfdb15d6244ccd9a245598efb5987034a752397ceb5ed78cfcf73479bb92e9bf`。

## 交互开发切片（非 formal）

formal 结果之后，新建的交互开发集不读取或重跑任何已消费 held-out。它冻结 100 问、100 个不同标题，CMRC2018 60 问、DRCD 40 问，并以 CAUSE、COMPARISON、TIME、QUANTITY、RELATION 五个公开问式表面桶各 20 问做工程分账。表面桶不是已证明的语义理解类别，DRCD 的 CAUSE 库为空，因此该切片不声称来源平衡或因果泛化。

- Recall@20：`99/100 = 99%`；top1：`99/100 = 99%`；
- ANSWER citation valid：`97/97 = 100%`；evidence hit：`87/100 = 87%`；source-page gold coverage：`100/100`；
- 失败分账：`GOLD_NOT_IN_EVIDENCE=10`、`NON_ANSWER=2`、`RETRIEVAL_MISS_AT_20=1`；五桶 evidence hit 依次为 CAUSE `18/20`、COMPARISON `16/20`、TIME `17/20`、QUANTITY `18/20`、RELATION `18/20`；
- 独立生产 UNKNOWN/CLARIFY 回归：`4/4 PASS`，其中虚构实体优先 `UNKNOWN`，真实多义实体保持 `CLARIFY`。

该运行状态是 `DEVELOPMENT_NON_FORMAL`，不是 formal receipt、不是新能力断言，也不是语义类别证明。公开紧凑 receipt 为 [`broad_qa_interactive_development_receipt_v1.json`](../data/ph2/broad_qa_interactive_development_receipt_v1.json)，只含指标、边界和承诺哈希，不含题目、标签、正文、预测或本机路径。维度报告在 K 盘只读复算后重新核对来源覆盖与失败分账，SHA-256 为 `3b28edc134a0ccd09b32699f28ea100f51d91e8225f2d28812da8c285d7e826d`；旧报告未覆盖。

## 唯一 formal held-out

算法、family、census、索引、alias、selection、questions、labels、开发 aggregate 和 14 个算法文件绑定到公开提交 `7f3d87607eedc29c69eb17f40729be39e04f9045`。固定位置 intent 在预测前以 `OUTCOME_PENDING` 占用；普通 `predict` 拒绝 held-out；正式预测授权不读取 labels；`FORMAL_HELD_OUT` 评分在解析 labels 前重新验证完整冻结链。intent 已存在时禁止换运行目录重跑。

唯一一次 300 问正式结果：

- Recall@20：`300/300 = 100%`；
- top1 source hit：`300/300 = 100%`；
- ANSWER citation valid：`296/296 = 100%`；
- evidence hit：`253/300 = 84.3333%`；
- CMRC2018：`126/150 = 84.0%`；DRCD：`127/150 = 84.6666%`；
- `UNKNOWN=4`，`GOLD_NOT_IN_EVIDENCE=43`；
- 查询 p50/p95：`177.8824/350.2957 ms`；
- 状态：`PASS`，未降低 Recall 80%、top1 70%、引用 100%、证据 60% 的冻结门；
- aggregate SHA-256：`84bfeb9023ffa31386fb4dcd159af9d82d797c92393d5e83322210a3cf4d30f3`。

公开紧凑 receipt 为 [`broad_qa_source_aligned_formal_receipt_v1.json`](../data/ph2/broad_qa_source_aligned_formal_receipt_v1.json)。它不含第三方题目、标签、原文、预测或本机路径。

当前可以准确表述的新增能力是：在冻结中文 Wikipedia 来源、稀疏联合索引和来源版本对齐问题总体上，页面检索、来源约束抽取与逐引用核验通过了预声明的 300 问正式评测。它不是任意开放域来源覆盖、自由生成、成熟对话、语言断奶或通用问答。下一工程缺口是减少 43 个“正确页但证据窗口未包含金答案”和 4 个拒答，并另建未消费 family 验证关系、时间、数量、因果与比较约束；不得重跑或按本次 held-out 调参。
