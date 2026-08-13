# 来源约束检索与证据选择联合评测

本页记录新的独立评测 family `PH2_BROAD_QA_JOINT_RETRIEVAL_EVIDENCE_PACK_V1`。它把两条此前分开的能力串成一条真实链：问题先从一个含随机 20k Wikipedia 页面及题目来源页的联合索引中检索，再在命中的来源段落中选择可引用证据句。

这不是开放域通用问答、自由生成或断奶结果。题目必须天然包含来源标题，不向问题追加标题；来源标题和答案先于系统运行冻结。旧的外部上下文评测 family（300 held-out）及其标题域不复用。

## 冻结合同

- 来源：CMRC2018 与 DRCD 官方 checkout，保留 commit、文件 SHA、许可和 URL。
- 旧 family 的 454 个标题域全部排除；新 family 冻结 200 dev、300 held-out，439 个标题。
- dev/held-out 仍按标题域隔离；questions、labels、source target、alias ledger 分离。
- 首次标题扫描命中 437/439；其中 126 个是 Wikipedia 重定向，2 个不在快照。
- 重定向链按 page id 解析，最终 v4 alias ledger 为 437/439 `RESOLVED`，2 个为 `SOURCE_TITLE_NOT_IN_SNAPSHOT`。原始标题、链、终页 page/revision 均保留；alias 只作为显式离散关系进入索引。

运行入口：

```bash
pure-integer-broad-qa-joint freeze ...
pure-integer-broad-qa-joint select ...
pure-integer-broad-qa-joint resolve-aliases ...
pure-integer-broad-qa-joint build-target ...
pure-integer-broad-qa-joint augment ...
pure-integer-broad-qa-joint predict ...
pure-integer-broad-qa-joint score ... --scope DEVELOPMENT
```

大数据和运行产物必须位于显式 K 盘 run root；每个步骤独占发布，失败可从前一步恢复。旧 formal/private artifact 不会被读取、覆盖或重跑。

## 开发结果

### v1：未接 alias

- 联合索引：20,302 页。
- Recall@20：119/200 = 59.5%。
- top1 source hit：116/200 = 58.0%。
- ANSWER citation valid：101/101 = 100%。
- evidence hit：65/200 = 32.5%。
- 状态：`FAIL`。主要失败是 126 个重定向页未被绑定。

### v2：alias 链接通

- 联合索引：20,428 页；437 个题目标题终页中 428 个追加，9 个已在随机 20k 中。
- alias postings：3,177 个 term。
- Recall@20：176/200 = 88.0%。
- top1 source hit：171/200 = 85.5%。
- ANSWER citation valid：140/140 = 100%。
- evidence hit：88/200 = 44.0%。
- 查询 p50/p95：142.627/247.851 ms。
- 状态：`FAIL`。

### v3/v4：回答侧通用规则探针

只使用公开问式槽的整数形态偏置、明确来源限定门和单页有限段落重选，不读取 labels、不添加题目特例。

- v3：158 ANSWER，citation valid 158/158；evidence hit 89/200 = 44.5%。
- v4：159 ANSWER，citation valid 159/159；evidence hit 86/200 = 43.0%。
- Recall@20 与 top1 仍为 88.0% / 85.5%。
- v4 状态：`FAIL`，aggregate SHA-256=`917df05f60d8c8b8afa1f0f93ea5f5ccadf1044725430ca13293ed4c0f924f02`。

预先冻结的联合门是 Recall@20 80%、top1 70%、evidence hit 60%、ANSWER citation valid 100%。检索与来源归属已达到开发门附近或以上，但证据句选择仍明显未达 60%，因此 **没有运行 300 问 held-out，也没有发布联合 PASS receipt**。dev 结果仅用于定位：当前主要缺口是页面内事实定位、问式与段落关系的结构建模，而不是继续扩大索引页数。

## 下一步

停止在本 family dev 上继续调参。下一施工应设计新的、未消费题目 family，优先补通用的页面内事实定位/多句证据链表示，并保留现有检索、alias、citation 和资源预算合同；新 family 仍需先冻结再运行。当前能力应分别表述为：20k 来源约束检索预览、外部上下文证据选择 formal PASS、联合检索+证据选择 dev FAIL。
