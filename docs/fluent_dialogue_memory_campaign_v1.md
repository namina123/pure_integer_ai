# 流畅对话与可持续记忆 v1：施工索引

本文件是当前长目标的短索引和恢复点，不替代代码契约、训练 manifest 或历史评测证据。目标是把已有的整数对话、后继图、广域来源检索、Runtime ledger 和 checkpoint 接成一个可独立运行的多轮交流能力，并以真实交流记录和独立发布评估为准，不以状态字段代替能力结果。

## 目标与硬约束

- 支持 60 轮以上、五个以上主题域的连续交流，包含追问、长句、重复回忆、跨进程恢复、资料导入、修订、删除和冲突隔离。
- 用户入口只输出自然语言和来源引用；内部的未知/澄清状态不得作为用户协议文本暴露。
- Runtime Memory 按会话和来源隔离，可回放、可修订，不自动晋升 Core；Core 学习必须有显式资格和证据。
- 不写死语言、简繁、词表、答案或代词表。语言与结构来自整数图、索引和课程数据。
- 数据能力以整数 tuple、SQLite 和可复现 digest 为边界，避免新增无必要第三方库，保持跨语言迁移路径。
- D: 只放代码和紧凑文档；K: 是训练、release、SQLite 和长任务产物盘。训练副本可用完即删，但源连接、许可和 SHA manifest 必须保留。
- 长训练启动后只能采用被动等待，主动检查间隔至少 30 分钟、通常 60 分钟；禁止几分钟或
  十几分钟查看一次。只做主线回归，不反复运行全量 CI 或微型探针。

## 当前基线（2026-09-01）

- 发布候选：`K:\pure_integer_ai_work\model_releases\public-model-gc-v8-dialogue-stage34-20260901`，独立 evaluator `PASS`。
- 评估证据：`K:\pure_integer_ai_work\dialogue_sessions\gc-v8-post-memoryfix-eval-20260901`；held-out、unknown、negative、冲突澄清、跨来源引用、checkpoint 恢复均通过。
- 性能基线：warm p50 `651 us`、p95 `10077 us`；后续 route-gate 复测为 `606/9983 us`。该数据是端到端发布评估基线，不代表所有长会话规模。
- 广域训练基线：`K:\pure_integer_ai_work\general_capability_campaign_20260829\gc-dialogue-stage1234-20260829e`；该 run 已封存，训练项 `2290`、occurrence `406545`、对话后继 `1813`。
- 当前终次训练：`K:\pure_integer_ai_work\fluent_dialogue_memory_campaign_20260830\gc-dialogue-stage34-tx-20260831d` 四阶段均完成，并已组装为 v8 独立发布根；后续运行时修正未修改训练 payload，未重新训练。
- v8 已绑定 `oasst12-kdconv10k-llmcc0-v14-20260830` 回应组织 artifact；其
  capability/held-out 均为 `PASS`、generated coverage 为 `230‰`，但独立语义
  质量仍为 `NE`。因此只能把它作为受门控的表层候选，不能宣称组合式语义已
  达标，也不能绕过来源、记忆和未知门。
- 当前终次 run 实际消费 `15576` 个 case，其中 train/held-out/negative 为
  `12864/2530/182`，已包含 OASST1、OASST2、KDConv 10k、LLM CC0 和因果补片。
  不再启动较小的 `4116` case 训练；后续训练必须先有新的公开数据或算法差异，
  并直接针对真实组合式语义缺口。

### 2026-09-01 生成门修正

- v8 的 60 轮真实交流显示，单次出现的人工回答片段会因稀疏表面相似度
  跨主题抢答，出现代码、天气或闲聊片段与问题不相称的情况。生产端现要求
  一个回答片段至少有 `2` 次训练支持；该门只作用于最低优先级 learned
  response consumer，不改变 Core successor、广域来源、Runtime 资料、记忆
  回放或 JSONL 协议。模型和课程整数格式保持不变。
- 同一轮复测还发现 CSQ 来源段以默认 `220‰` 进入正式终端，低覆盖候选会被
  错误投影为答案。生产交互入口现在显式使用 `500‰` 来源正文覆盖门；离线
  `ScidbCsqPassageRuntime` 默认值保持不变，避免改变既有索引/专项合同。
- 来源追问解析器不再允许“标题前缀命中同一标题”自证。问式图必须先识别
  回答槽；极短省略问可继续核验同源，较长问题还须与上一问答共享至少两个
  索引特征。实现不写入代词、语言或主题词表。
- 最终复测位于 K 盘工作根下的
  `dialogue_sessions/gc-v8-dialogue-stage34-routefix-d-20260901`，
  两段合计 `ANSWER=13`、`CLARIFY=19`、`UNKNOWN=28`，p50 为
  `21850/74483 us`，p95 为 `117092/112999 us`。CSQ 的 UBI/改写误答与
  “互相矛盾”后四轮来源连锁均消失；跨进程“浙江卫视是什么？→它在哪里？”
  仍以 ordinal `0→1` 返回同一来源。两份响应 SHA-256 分别为
  `1f4a7022f25e6761e3f0a7a42326054c2f788fcb58545a5d27049c9d1274e77c`、
  `73e911fa49facce57e56cd3aa9c45bdf25b2859876cac21c9b90b6f5c09cafe4`。
- 记忆扩散修正后的生产 smoke 位于 K 盘工作根下的
  `dialogue_sessions/gc-v8-memory-diffusion-smoke-20260901`：两条未知陈述之间
  插入一条无关偏好后，第三轮和重启后的第四轮都返回原始兴趣陈述；checkpoint
  ordinal 从 `0..2` 延续到 `3`。该路径仍只回放 checkpoint 中已保存的用户文本，
  不读取 digest 生成表面，也不写入 Core 或训练库。
- 修正后的完整 60 轮复测位于 K 盘工作根下的
  `dialogue_sessions/gc-v8-dialogue-stage34-memoryfix-e-20260901`：内部状态统计为
  `ANSWER=18`、`CLARIFY=16`、`UNKNOWN=26`；r42 已回放兴趣陈述，r57--r60
  未再被旧来源连锁接管，来源命中只剩预期首问与冲突问题。两段 p50/p95 为
  `22541/125397 us`、`74897/120346 us`。
- 记忆修正后的独立发布 evaluator 仍为全项 `PASS`，证据位于
  `dialogue_sessions/gc-v8-post-memoryfix-eval-20260901`，aggregate SHA-256 为
  `67230fbafb2791cb515d6b090dc4caeac9f32f329cf213cbfd08e53d523a9fdf`；
  warm p50/p95 为 `651/10077 us`。该 SHA 仅在 evaluator 输出回读后登记，若文件
  被重新生成必须重新计算，不能手工复用。
- 发布级回归仍为 `PASS`，证据位于 K 盘工作根下的
  `dialogue_sessions/gc-v8-post-route-gate-eval-20260901`，
  aggregate SHA-256 为
  `263a1c2545b199dd58bfef9f63db1d704e6dfb51a6d4b4434a03368a96a05680`；
  warm p50/p95 为 `606/9983 us`。

## 阶段索引

1. **M1 长会话召回降本（已完成）**：恢复时一次建立 turn 特征缓存，查询直接消费缓存；保持 checkpoint 编码和身份不变。
2. **M2 Runtime 结构记忆接线（已完成）**：Runtime event、memory item、SourceRef、observation、结构、命题和 evidence 闭包已进入生成边界；原始 digest 不得冒充文本，资料内容继续通过来源 provider 读取。
3. **M3 组合式语义与组织增量（进行中）**：先消除长会话错误路由，再以真实失败族确定公开课程和算法差异；只有差异足够时才启动新的完整训练，失败保留证据且不接入 release。
4. **M4 独立交流验收**：一次 60+ 轮真实交流，覆盖多域、追问、长句、重复回忆、重启、导入/修订/冲突和未知；只保留必要记录。
5. **M5 发布候选**：重新组装独立 release root，运行一次公开 validator 和端到端 evaluator，记录逐文件 SHA、来源/许可 manifest、性能和恢复点。

## 本轮施工

`PersistentBroadDialogueRecovery` 新增进程内 `cold_turn_features` 派生缓存。恢复阶段为每个冷轮次只生成一次 n-gram 特征；`query_relevant_turns()` 在候选排序时直接读取缓存，`with_turn()` 增量维护缓存，旧式外部构造仍有回退路径。缓存不写入 checkpoint、训练库或发布包，因此不会改变既有身份和跨语言数据契约。该切片已提交为当前分支最新变更的一部分，训练进程不依赖未提交的运行时产物。

### 2026-09-01 Runtime provider 降本

`RuntimeMaterialResponseProvider` 现在在装配时一次建立进程内派生索引：精确问题、来源标题、问题特征到候选的倒排，以及每条绑定已经计算好的广域特征。精确命中、同源追问和自然改写都消费这些索引，不再在每轮遍历全部 Runtime binding 或重复生成问题特征。索引是 `runtime derived cache`，不进入 `bindings.int`、SQLite、manifest 或任何跨语言身份；重启后由相同的 binding/source ledger 确定性重建。资格、冲突、未知和 SourceRecord 回读顺序没有放宽。

该切片的 Runtime 语言、CLI 和 binding 持久化回归为 `11 passed`；没有修改训练 payload、release root 或论文。长会话性能仍需用真实 60+ 轮记录确认，不能把这次局部索引优化直接等同于 `p95 <= 100ms` 达标。

### 2026-09-01 Runtime 结构上下文实际消费

Runtime response provider 现在除了返回已资格化答案、来源和 citations，还会
返回由 event、memory item、SourceRef、observation、结构引用、命题记录和
evidence 身份组成的 `RuntimeMaterialGenerationContext`。该上下文只使用非负
整数 tuple，可由同一 Runtime ledger 确定性重建；digest 仅作身份校验，不能
投影为用户可见文字。

`answer_broad_dialogue_turn` 在接受 Runtime `ANSWER` 前验证 response-act 与
上下文闭合，并把上下文交给可选的 `generation_context_consumer`。发布终端已
把该 consumer 接到现有 `TrainedSurfaceRuntime.render`：结构证据参与表层候选
变体的确定性选择，未能安全组织时原样保留。普通 `surface_consumer` 在上下文
已消费时不再二次改写，避免重复组织；`answer`、来源、citations、
`DialogueTurn` 和 checkpoint 身份均不变。

本切片通过 Runtime/广域对话 `23 passed`、表层/学习对话 `13 passed`、CLI
集成 `7 passed` 和公开协议 `9 passed`，并通过 `compileall`、`git diff --check`。这证明结构上下文已经进入实际生成边界，仍
不等于组合式广域回答能力已经完成；下一步仍需在不降低回答门的前提下，以公开
因果和承接课程做一次完整训练对照。

### 2026-09-01 长会话记忆误召回修正

完整 60 轮记录进一步证明，旧 bootstrap 会把同一短尾串的重叠 n-gram 当成
多份独立证据，并可能把一次错误回放重新学习成“召回形状”。现在首次候选必须
同时满足独立共享码点数和查询码点覆盖门；索引只在回答写入前自身确实会选择
该回放时学习 replay pair，外部错误答案不能事后自证。操作符从“召回查询减去
被回放陈述”的整数特征差集中学习；跨样本重复操作符负责泛化，内容特征继续
负责选择具体记忆。

关机前的 `gc-v8-dialogue-stage34-memoryfix-e-20260901` checkpoint 已只读重建：
60 个轮次和 checkpoint 60 全部通过身份验证。修正后两种兴趣回忆仍返回原始
兴趣陈述；不存在对象、青石台颜色、夜间模式、Runtime 资料追问及缺少可靠
ordinal 证据的“最开始问了什么”均不再触发回放。专项回归为 `25 passed`，
持久化/终端协议组合回归为 `18 passed`；未修改旧 checkpoint 或 K: 证据。

独立发布回归位于
`dialogue_sessions/gc-v8-post-recall-index-eval-20260901`，aggregate SHA-256 为
`5a3740180648d7d63969bc8bf1c0d61bdc245b5aa8b86202bfdb6e2029c31a1c`。
held-out `5/5`、unknown `2/2`、negative `2/2`、冲突澄清、跨来源双引用和
checkpoint 恢复均为 `PASS`；warm p50/p95 为 `1565/9949 us`，读取 SQL
statement 共 `34`，峰值工作集 `514154496` bytes。该回归直接调用 v8 release
root 与当前仓库运行时，证明索引修复未退化发布承重边界；它不把 60 轮中的
组合式未知或不相称回答误记成语义质量通过。

### 2026-08-30 长目标审计补充

- `gc-dialogue-stage1234-20260830c` 已自然结束。它实际消费 `15464` 个 case、`12751` 个训练项、`103520` 个 turn，写入 `2615856` 个 occurrence、`12387` 个 dialogue successor projection 和 `1791295` 个 successor feature。Stage 1 达标；Stage 2 的 CAUSES 覆盖率为 `44‰`，低于未修改的 `50‰` 门槛，因此 Stage 3/4 未执行，`weaning_ready=false`。这不是四阶段完成，也不是断奶证据；保留 run 作为真实训练结果和恢复基座。
- 该 run 的训练图已组装为 `K:\pure_integer_ai_work\model_releases\public-model-gc-v7-dialogue-stage1-20260830`，并通过 `load_public_model_release(..., verify_payload_hashes=true)` 的逐文件 SHA 校验。它是“Stage 1 公开候选”，不是流畅交流发布版；v6 仍是当前能力基线，v7 不覆盖 v5/v6。
- v7 的 60 轮独立 JSONL 交流记录位于 `K:\pure_integer_ai_work\dialogue_sessions\gc-v7-dialogue-stage1-independent-20260830`：`ANSWER=10`、`CLARIFY=14`、`UNKNOWN=36`，p50 `36126 us`、p95 `147905 us`、峰值工作集 `514383872` bytes。结果没有超过 v6 的 12/14/34 基线，不能宣称新增训练改善了对话能力；记忆回放成功的边界仍以既有 h/i 记录为准。
- 本轮已经确认的真实承重边界：广域 fast route 会执行来源查询，`UNKNOWN/CLARIFY` 后的 Runtime Memory 可回放，但对未见过的组合式问题仍缺少稳定的结构化回答组织。继续堆 OASST/KdConv 近邻数据不能替代因果/意图/承接/改写等组合能力；不得通过降低相似度门槛制造答案。

- `fluent-memory-60round-20260830-c` 提供了 60 轮、跨两次进程的真实记录：checkpoint 恢复、Runtime 资料导入、来源引用均实际运行。两段合计 `ANSWER=22`、`CLARIFY=15`、`UNKNOWN=23`；第一段 p50 `39,938 us`、p95 `160,868 us`，第二段 p50 `39,947 us`、p95 `93,965 us`。因此当前只能宣称“可恢复和可接线”，不能宣称流畅交流或 p95 目标达成。
- 记录暴露的承重缺口是：Runtime Memory 已保存用户轮次并可按相似度召回，但召回轮次尚未稳定进入自然回答生成；“记住/回忆”类问题仍频繁落到 UNKNOWN。修复必须复用已有语言无关召回与对话组织模型，不能在代码写入特定语言词表、固定答案或内部状态文本。
- 性能下一阶段聚焦首段冷启动、来源查询和长会话召回的算法缓存；目标为 warm p50 保持 10ms 级、p95 `<=100ms`，不牺牲未知、冲突和来源引用边界。

## 恢复点

- 2026-09-01 意外关机后恢复检查：未发现残留训练进程或半写入的当前 run；K: 盘剩余约 89 GiB，v8 release 与终次训练根均可回读。当前没有需要续接的训练进程，因此不做短间隔轮询，也不重跑历史评测。
- 已完成：长会话特征重复计算的代码切片。
- 已完成（2026-09-01）：Runtime provider 的精确/来源/特征索引切片；受影响的 Runtime language、CLI、binding persistence 回归 `11 passed`。索引只存在进程内，不改变发布数据合同。
- 已完成（2026-09-01）：Runtime 结构上下文从 provider 经 `answer_broad_dialogue_turn` 接入发布终端的 `TrainedSurfaceRuntime`，并保持 citations/answer/checkpoint 身份守恒；专项回归 `23 + 13 + 7 + 9 passed`。上下文不写入 `DialogueTurn` 或 checkpoint，重启后由 Runtime ledger 重建。
- 已完成（2026-09-01）：长会话记忆索引改为回答前授权 replay pair，并加入独立码点证据、错误回放阻断和操作符差集学习；旧 60 轮 checkpoint 只读重建后，合法兴趣回忆保持，五类错误回放均为 `None`。专项/协议回归 `25 + 18 passed`。
- 已完成（2026-09-01）：修复后的独立 v8 release 回归全项 `PASS`，aggregate SHA-256 为 `5a3740180648d7d63969bc8bf1c0d61bdc245b5aa8b86202bfdb6e2029c31a1c`，warm p50/p95 为 `1565/9949 us`。
- 已完成（本轮）：`tests/test_conversation_broad_dialogue_persistence.py` `2 passed`；`git diff --check` 通过。checkpoint 身份和旧式 recovery 构造兼容性保持不变。
- 已完成（本轮）：`gc-dialogue-stage1234-20260830c` 的四阶段请求和独立 v7 组包/校验；结果只到 Stage 1，60 轮交流未改善，不能晋升为能力基线。
- 已确认：`fluent-memory-60round-20260830` 首行 BOM 导致 59 个有效请求，`-b` 版本虽有 60 行但全部为 UNKNOWN；这两份记录都不能作为流畅交流证据。训练后生成的 v7 记录同样未达到流畅门槛，详见上方统计。
- 下一步：从 v8 的真实未知、澄清与不相称回答中归并组合式失败族，核对现有
  `15576` case 对因果、承接、澄清、改写、比较、建议和总结的实际覆盖。只有
  形成新的公开数据/算法差异后才做一次完整训练对照；现有 v14 artifact 的
  `semantic_quality_status=NE` 必须保留，不得降低相似度或证据门槛。
- 本轮代码已接入 `memory_recall_response`：广域/来源路由明确未命中后，宿主用冷轮次整数特征的稀有多标量交集选择候选，并回放已有用户陈述；不读取 digest 作为表面、不新增语言词表或固定答句。独立集成回放确认“你还记得我刚才说的兴趣吗？”返回持久化陈述而非公开 UNKNOWN。
- 本轮同时加入 `append_broad_dialogue_checkpoint`。长会话正常追加只验证已持有的前驱 ordinal/identity 并排他写入后继；进程启动仍完整重放并核验链，避免每轮 O(n) 磁盘重读。针对性回归 `18 passed`，`compileall` 与 `git diff --check` 通过。
- 再下一步：补充能把 CAUSES 覆盖从 `44‰` 推过 `50‰` 的公开、可审计因果课程，并单独设计组合式对话行为课程（承接、澄清、改写、比较、建议、总结），先做一次完整训练切片再评估；不得把旧 response-organization `NE` artifact 接入 release。若新课程不可得，应记录授权/数据阻塞，不以降低门槛或重复近邻训练替代。
- 完成条件：M1-M5 均有实际能力或明确失败证据，发布包可在 K: release root 独立启动，且新增差异不含密钥、绝对路径、私有评测或论文。
