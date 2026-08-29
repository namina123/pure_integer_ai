# 流畅对话与可持续记忆 v1：施工索引

本文件是当前长目标的短索引和恢复点，不替代代码契约、训练 manifest 或历史评测证据。目标是把已有的整数对话、后继图、广域来源检索、Runtime ledger 和 checkpoint 接成一个可独立运行的多轮交流能力，并以真实交流记录和独立发布评估为准，不以状态字段代替能力结果。

## 目标与硬约束

- 支持 60 轮以上、五个以上主题域的连续交流，包含追问、长句、重复回忆、跨进程恢复、资料导入、修订、删除和冲突隔离。
- 用户入口只输出自然语言和来源引用；内部的未知/澄清状态不得作为用户协议文本暴露。
- Runtime Memory 按会话和来源隔离，可回放、可修订，不自动晋升 Core；Core 学习必须有显式资格和证据。
- 不写死语言、简繁、词表、答案或代词表。语言与结构来自整数图、索引和课程数据。
- 数据能力以整数 tuple、SQLite 和可复现 digest 为边界，避免新增无必要第三方库，保持跨语言迁移路径。
- D: 只放代码和紧凑文档；K: 是训练、release、SQLite 和长任务产物盘。训练副本可用完即删，但源连接、许可和 SHA manifest 必须保留。
- 长训练启动后采用长间隔被动等待；只做主线回归，不反复运行全量 CI 或微型探针。

## 当前基线（2026-08-30）

- 发布候选：`K:\pure_integer_ai_work\model_releases\public-model-gc-v2-20260830`，独立 evaluator `PASS`。
- 评估证据：`K:\pure_integer_ai_work\dialogue_sessions\gc-v2-independent-eval-20260830-b\independent_release_evaluation.json`；held-out、unknown、negative、冲突澄清、跨来源引用、checkpoint 恢复均通过。
- 性能基线：warm p50 `615 us`、p95 `11389 us`，SQLite 语句 `34`，峰值工作集 `173248512` bytes。该数据是端到端发布评估基线，不代表所有长会话规模。
- 广域训练基线：`K:\pure_integer_ai_work\general_capability_campaign_20260829\gc-dialogue-stage1234-20260829e`；该 run 已封存，训练项 `2290`、occurrence `406545`、对话后继 `1813`。
- 当前增量训练：`K:\pure_integer_ai_work\fluent_dialogue_memory_campaign_20260830\gc-dialogue-stage1234-20260830a` 已完成 Stage 1；Stage 2--4 正在以 `gc-dialogue-stage1234-20260830b` 续接，使用同一 pack SHA `3af7e5689b1537ebba95bbba272bfd1b55f9c0230cc487458b29eca4d508c1a6`、K 盘 page-resume 和 bulk 可重建存储。训练完成前不得将其写成四阶段完成或发布候选。
- OASST1 回应组织 artifact 历史上仍为 `NE`，未接入当前 release；未确认与当前 run 绑定前不得直接使用。
- 训练输入扩展判断：在保留 authored/结构课程的前提下，K: 的 OASST1、OASST2 与受控 KDConv 公开切片合计可形成约 `4116` 个 case；当前 run 仅消费约 `3000` 个。已决定启动一次新的四阶段完整 run，优先扩大对话覆盖，不改变回答门或引入外部依赖。

## 阶段索引

1. **M1 长会话召回降本（进行中）**：恢复时一次建立 turn 特征缓存，查询直接消费缓存；保持 checkpoint 编码和身份不变。
2. **M2 Runtime 结构记忆接线**：核对 Runtime event 在语义/结构召回中的边界；原始 digest 不得冒充文本，资料内容继续通过来源 provider 读取。
3. **M3 组织能力增量**：只在已有课程不足时，从 K: 公开对话课程构建与当前 run 绑定的新 artifact；失败保留证据，不接入 release。
4. **M4 独立交流验收**：一次 60+ 轮真实交流，覆盖多域、追问、长句、重复回忆、重启、导入/修订/冲突和未知；只保留必要记录。
5. **M5 发布候选**：重新组装独立 release root，运行一次公开 validator 和端到端 evaluator，记录逐文件 SHA、来源/许可 manifest、性能和恢复点。

## 本轮施工

`PersistentBroadDialogueRecovery` 新增进程内 `cold_turn_features` 派生缓存。恢复阶段为每个冷轮次只生成一次 n-gram 特征；`query_relevant_turns()` 在候选排序时直接读取缓存，`with_turn()` 增量维护缓存，旧式外部构造仍有回退路径。缓存不写入 checkpoint、训练库或发布包，因此不会改变既有身份和跨语言数据契约。该切片已提交为当前分支最新变更的一部分，训练进程不依赖未提交的运行时产物。

### 2026-08-30 长目标审计补充

- `gc-dialogue-stage1234-20260830b` 已结束但未完成 Stage 2 门控：`stages_completed=[]`，CAUSES 覆盖率为 `34‰`，门槛为 `50‰`；Stage 3/4 未执行，`weaning_ready=false`。该 run 只能保留为失败证据，不能降低门槛、不能接入 release。2026-08-29 的四阶段 `PASS` 基线继续作为当前可用训练基线。
- `fluent-memory-60round-20260830-c` 提供了 60 轮、跨两次进程的真实记录：checkpoint 恢复、Runtime 资料导入、来源引用均实际运行。两段合计 `ANSWER=22`、`CLARIFY=15`、`UNKNOWN=23`；第一段 p50 `39,938 us`、p95 `160,868 us`，第二段 p50 `39,947 us`、p95 `93,965 us`。因此当前只能宣称“可恢复和可接线”，不能宣称流畅交流或 p95 目标达成。
- 记录暴露的承重缺口是：Runtime Memory 已保存用户轮次并可按相似度召回，但召回轮次尚未稳定进入自然回答生成；“记住/回忆”类问题仍频繁落到 UNKNOWN。修复必须复用已有语言无关召回与对话组织模型，不能在代码写入特定语言词表、固定答案或内部状态文本。
- 性能下一阶段聚焦首段冷启动、来源查询和长会话召回的算法缓存；目标为 warm p50 保持 10ms 级、p95 `<=100ms`，不牺牲未知、冲突和来源引用边界。

## 恢复点

- 已完成：长会话特征重复计算的代码切片。
- 已完成（本轮）：`tests/test_conversation_broad_dialogue_persistence.py` `2 passed`；`git diff --check` 通过。checkpoint 身份和旧式 recovery 构造兼容性保持不变。
- 进行中：`gc-dialogue-stage1234-20260830b` 续接 Stage 2--4 的全量公开课程训练（K:，PID 23916）；训练期间只被动等待，结束后读取 `training_summary.json`、pack manifest 和 SHA，再决定是否组装新 release。
- 已确认：`fluent-memory-60round-20260830` 首行 BOM 导致 59 个有效请求，`-b` 版本虽有 60 行但全部为 UNKNOWN；这两份记录都不能作为流畅交流证据，待训练完成后用无 BOM 原始 UTF-8 重新生成真实多主题交流记录。
- 下一步：将 Runtime Memory 的结构命中作为生成侧上下文证据，交给已有 learned/core dialogue runtime 组织回答；digest 仍只作身份校验，原始可读文本只从 checkpoint 中的 `DialogueTurn` 或来源 provider 读取。
- 本轮代码已接入 `memory_recall_response`：广域/来源路由明确未命中后，宿主用冷轮次整数特征的稀有多标量交集选择候选，并回放已有用户陈述；不读取 digest 作为表面、不新增语言词表或固定答句。独立集成回放确认“你还记得我刚才说的兴趣吗？”返回持久化陈述而非公开 UNKNOWN。
- 本轮同时加入 `append_broad_dialogue_checkpoint`。长会话正常追加只验证已持有的前驱 ordinal/identity 并排他写入后继；进程启动仍完整重放并核验链，避免每轮 O(n) 磁盘重读。针对性回归 `18 passed`，`compileall` 与 `git diff --check` 通过。
- 再下一步：基于当前训练 run 判断是否有可绑定的 response organization artifact；没有明确 PASS 证据时不启动新长训练。
- 完成条件：M1-M5 均有实际能力或明确失败证据，发布包可在 K: release root 独立启动，且新增差异不含密钥、绝对路径、私有评测或论文。
