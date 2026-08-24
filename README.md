[English](README_EN.md) | **中文**

# PIDSLCA：纯整数确定性自学习认知架构

PIDSLCA（Pure-Integer Deterministic Self-Learning Cognitive Architecture）是一个完全公开的探索性研究项目。它研究一个具体问题：能否在不依赖浮点运算的前提下，构造一个可在普通个人电脑上运行、可逐位复现、可审计，并能从经验中更新的认知系统。

本仓库公开参考实现、当前测试、持续集成、格式样例、开发脚本和项目作者已完成的论文。项目中的“自学习”描述研究目标与机制，不表示系统已经获得通用智能、自主理解、成熟对话能力或生产可用性。

## 支持独立研究

PIDSLCA 由个人独立研究和维护，目前没有机构经费或商业赞助。捐赠将用于维持项目的公开开发，包括跨平台测试与 CI、实验所需的计算和存储、代码与论文的长期存档，以及持续维护。

**[通过微信、支付宝或 Ko-fi 支持 PIDSLCA 的公开研究](DONATE.md)**

每一份支持都在帮助代码、测试、研究记录和论文继续向所有人开放。捐赠完全自愿，不改变 MIT 许可，也不换取路线图优先级、私有版本或排他访问；无论是否捐赠，公开成果都按相同条件提供。

## 研究主题

- 用整数表示认知状态、关系强度、计数、证据和协议数据。
- 让固定输入与固定状态产生可复现的执行结果，便于审计和对照实验。
- 用图结构表达概念、关系、记忆、顺序、因果和可执行结构。
- 研究关系强化、结构归纳、记忆更新、构造性验证与恢复机制如何协同。
- 在普通硬件和标准 Python 环境中验证这些机制，而不是依赖大型专用基础设施。

## 项目特色

- **纯整数核心**：核心计算路径避免浮点状态，降低跨平台数值差异。
- **确定性执行**：固定输入与协议状态应得到逐位一致的结果。
- **关系计数强化**：关系通过可追踪的整数计数积累，并按显式条件提升。
- **结构归纳**：从多个可对齐样本中提取共享结构，而不是只保存表面文本。
- **构造性核验**：对可执行结果、逆变换、迁移和恢复路径进行独立检查。
- **可审计边界**：明确区分已实现机制、实验性能力和仍待验证的研究问题。
- **普通硬件可运行**：依赖保持有界，当前索引和探针可在普通个人电脑上构建、运行和审计。

## 项目作用

PIDSLCA 目前适合作为研究和工程验证基础：

- 复现实验：研究确定性认知架构、图推理、结构学习和整数化表示。
- 审计实现：检查一次状态变化来自什么输入、规则和证据。
- 构造原型：测试记忆、关系学习、生成、程序执行、恢复和评估机制。
- 教学与讨论：提供可运行代码、公开测试和论文材料，便于复核设计主张。

它目前不是聊天产品、通用智能系统或可直接部署的决策服务。受控工程测试通过，只能证明相应实现满足测试条件，不能替代真实世界中的语义、泛化和可靠性评估。

## 公开进度

项目当前处于持续开发的研究原型阶段，已经公开：

- 可安装的纯整数参考实现，以及确定性工具、图存储、记忆与恢复等基础模块。
- 关系机制、认知流程、训练编排、生成、程序执行和评估设施。
- 与当前实现对应的回归测试、跨平台持续集成、格式样例和开发辅助脚本。
- 论文 PDF、LaTeX 源码、参考文献和永久 DOI 存档信息。

当前研究重点包括运行效率、长文本与长期上下文、正式训练资料、用户交互，以及真实语义环境中的泛化和可靠性。公开测试反映工程实现的验证范围，不代表这些开放问题已经解决。

项目现在还公开了真实来源的中文广域事实问答纵切：从冻结的中文 Wikipedia 快照稳定选择 100,000 个候选页面，并在不按可答性重选的前提下完成 20,000 个 accepted 页面、109,006 个段落和 3,608,002 个稀疏特征的紧凑索引。每个 `ANSWER` 返回页面、修订、贡献者、原始证据 span/hash 和许可身份；固定的 24 问开发探针得到 `22 ANSWER / 1 UNKNOWN / 1 CLARIFY`，22 条引用从冻结原文重建后全部通过。20k 索引为 251,494,400 bytes，SHA-256 为 `e18db72b090dfdfd96aac23c74a5ad0751afe17c2dcfb02fc91f1213b0f7c4da`。有界多轮 posting merge 在保持该 SHA 逐字节不变的同时，把真实 publication 从 840.972 秒压缩到 542.421 秒，缩短 35.501%。

独立的外部上下文证据选择评测也已冻结并正式只运行一次：300 道 held-out 题的精确引用有效率为 100%，所选证据句包含金答案 `234/300=78%`。该结果通过预先声明的 70% 证据选择门（CMRC2018 为 70%，DRCD 为 86%），aggregate SHA-256 为 `82bc0c5083fe5c9ce4e8f1a3bfee756e3681fbd28ee0756e0e6bbefb9957c96d`。它测量的是给定上下文的回答侧证据选择，不是随机索引检索、自由生成或通用对话。详见[外部评测报告](docs/broad_qa_external_evidence_eval_cn.md)。

联合评测随后完成了来源版本对齐。完整候选 census 显示，10,061 道自然标题锚定问题中有 7,189 道同时在当前终页和实际索引 passage 预算内保留金答案，原始总体覆盖率为 `71.4541%`；未覆盖部分没有从项目边界中删除。排除所有已消费标题后冻结的新 family 在唯一一次 300 问 held-out 正式运行中达到 Recall@20 `300/300`、top1 `300/300`、ANSWER 引用有效 `296/296`，证据命中 `253/300=84.3333%`，通过预先冻结的 60% 门。aggregate SHA-256 为 `84bfeb9023ffa31386fb4dcd159af9d82d797c92393d5e83322210a3cf4d30f3`；公开紧凑 receipt 位于 [`data/ph2/broad_qa_source_aligned_formal_receipt_v1.json`](data/ph2/broad_qa_source_aligned_formal_receipt_v1.json)。详见[联合评测报告](docs/broad_qa_joint_retrieval_eval_cn.md)。

formal 纵切后又公开了一个不消费 held-out 的 100 问交互开发集：五个公开问式表面桶各 20 问，Recall@20 `99/100`、top1 `99/100`、ANSWER 引用有效 `97/97`、证据命中 `87/100=87%`，来源页金答案覆盖 `100/100`；四条原创 UNKNOWN/CLARIFY 生产回归 `4/4`。这些桶只是开发分账，不是已证明的语义理解类别；结果属于 `DEVELOPMENT_NON_FORMAL`，紧凑 receipt 位于 [`data/ph2/broad_qa_interactive_development_receipt_v1.json`](data/ph2/broad_qa_interactive_development_receipt_v1.json)。

这仍是来源约束的抽取式广域事实问答纵切，不是自由生成、通用问答或断奶结果。它证明了当前中文 Wikipedia 冻结快照内的稀疏页面检索、来源约束证据选择和逐引用核验可以在预声明评测上闭合；没有证明任意问题、任意来源更新、长对话或自主语言学习已经闭合。详细合同和诚实边界见[20k 开发预览](docs/broad_qa_20k_preview.md)、[外部评测报告](docs/broad_qa_external_evidence_eval_cn.md)与[联合评测报告](docs/broad_qa_joint_retrieval_eval_cn.md)。

2026-08-24 起，公开对话课程开始进入真实训练主线：清洗后的统一 pack 回读为 `731` 条（`365 train / 184 heldout / 182 negative`），pack SHA-256 为 `1c907caac90c6edb687ad45e0db490da9188028374d90757af8fc28b720ce03d`。其中保留完整句和 Markdown、HTML、源代码、表格、引用/嵌入等结构载体的原文；元数据不会被送入语言通道。该 pack 已在 `K:` 盘完成新的 stage-1 observe 训练，产生 SQLite 图、checkpoint/dump、cursor 和运行 manifest。训练结果目前只证明公开课程被真实消费并改变了图状态，不宣称生成能力或语言 mastered。可选的组合入口 `run_integrated_dialogue` 会保留有限对话热历史，先尝试公开完整命题句运行时，未命中再查询来源约束的中文 Wikipedia 索引；回答带来源标题和链接，无法确认时保持 UNKNOWN/CLARIFY/CONFLICT。

公开 split 对照显示，held-out 与 negative 均和训练集保持零身份重叠；这只是数据隔离和输入新颖性证据，不是问答正确率。当前 K 盘训练仍处于 stage-1 observe 阶段，广域入口仍是来源约束抽取式问答。可复跑的规模展示入口固定执行窄域完整句、长问句、来源绑定追问和广域问题，并对同一只读数据库重放：最近一次展示为 `14` 轮、`13 ANSWER / 1 UNKNOWN`、`5` 条长问句、`10` 个来源绑定回答，重放逐位一致；其中“它/该条目”等紧接追问会复用上一轮已确认的来源标题作为检索焦点。展示同时实际读取 v6 K 盘训练状态，并用 DLG-RAW-16 两个独立 family 学到的结构消费 3 条窄域回答和 4 条新值 typed probe，覆盖限定事实、UNKNOWN、CLARIFY、REPAIR（`trained_surface_consumer.bound=true`、`typed_probe_used_count=4`）。摘要同时绑定 `dialogue-pack-v6-clean-surface` 的训练 pack SHA 和只读 SQLite graph 状态。展示摘要会写入调用者指定的 K 盘路径；它不是通用问答或自由生成评测。

为验证“让人听懂的完整句子”不是只回放训练实体，项目又增加了独立的表层结构 held-out 开发评估：10 条新实体、新限定和新组合，覆盖 `ANSWER/UNKNOWN/CLARIFY/REPAIR`，每条生成结果均为长句（至少 48 UTF-8 bytes）。接入训练表层消费者后为 `10/10 PASS`，未接消费者的对照为 `10/10 NO_LEARNED_SURFACE`；报告 SHA-256 为 `7e77514e4b074eb46e2fa0e524f977c1fdafa28175430ccd4345f429f29479ec`。该结果证明的是已学表层结构对新 typed 输入的组合能力，不证明从原始文本自动理解、事实真值判断、自由生成或广域知识泛化；广域知识路径仍是带来源约束的检索与证据选择。

随后又加入真实六轮多轮开发切片：`ANSWER -> UNKNOWN -> CLARIFY -> ANSWER -> ANSWER -> ANSWER`。它在首轮确认矮寨大桥来源后故意进入“火星上的矮寨大桥”未知问题，紧接的“它”只得到 `CLARIFY`，实际检索问句没有注入旧来源；之后新的窄域完整句和黄山松来源追问恢复为可读 `ANSWER`。v6 运行结果为 `6` 轮、`4 ANSWER / 1 UNKNOWN / 1 CLARIFY`、`3` 条长回答、训练表层消费者实际使用 `1` 次，回放逐位一致；报告 SHA-256 为 `8543abacf0e5b8be72c1fb6cadfa3ab75d804062b28df8490ebf67b160a67e05`。这证明的是有限热历史中的来源焦点边界与回答恢复，不是无限长记忆、自由对话或通用知识理解。

为扩大来源面，v2 又把铁路、桥梁、知识图谱、机场和地理分布五个真实来源域组成 `6` 个场景、`19` 轮对话。每条 `ANSWER` 都要求主证据包含预声明的事实词，而不是只统计状态码；实际结果为 `17/17` 证据词命中、`16` 条长回答、`10` 次合法焦点注入，未知轮后的指代仍保持 `CLARIFY` 且不泄漏旧来源，回放逐位一致。报告 SHA-256 为 `45172ce30f021e524466af525fc3dbc9ee888c9517e097e1bc74f606b4e796c4`。这证明的是有限来源覆盖下的可读多轮问答和证据约束，不代表任意领域、任意长文本或开放域语义理解已经完成。

公开 24 问广域开发池现在也有独立的主证据审计：`23 ANSWER / 1 UNKNOWN`，`23/23` 预声明证据词命中，`23/23` 主证据不是 `Category:`、小写 `category:` 或表格残片，23 条回答均达到长回答阈值，重复运行逐位一致。审计过程中修正了证据去重顺序：不再让后出现的较大窗口删除先出现的正确短证据；奖牌、活动日期、人口、天体距离和发现者问题现在都优先显示完整事实句。报告 SHA-256 为 `68133dd1763a7d5a90ed72cb9d38434dd8c46bd80dff69de0ffa76297ca8f52a`。这仍是冻结中文 Wikipedia 20k 索引上的开发回归，不是 held-out 或通用问答结论。

```bash
# 训练大数据和运行产物必须放在 K:，以下命令不会把它们写入 Git 工作区
python -m pure_integer_ai.experiments.run_conversation_training \
  --project-root . \
  --run-root K:\pure_integer_ai_work\dialogue_training_week_v1 \
  --run-id dialogue-pack-v6-clean-surface --stages 1 --with-heldout-probe

python -m pure_integer_ai.experiments.run_integrated_dialogue \
  --database K:\pure_integer_ai_work\broad_qa_week_v1\indexes\broad-qa-20k-from-100k-target-v2.sqlite3 \
  --training-run-root K:\pure_integer_ai_work\dialogue_training_week_v1\dialogue-pack-v6-clean-surface \
  "保满铁路全长多少公里？"

python -m pure_integer_ai.experiments.run_dialogue_scale_showcase \
  --project-root . \
  --database K:\pure_integer_ai_work\broad_qa_week_v1\indexes\broad-qa-20k-from-100k-target-v2.sqlite3 \
  --training-run-root K:\pure_integer_ai_work\dialogue_training_week_v1\dialogue-pack-v6-clean-surface \
  --output K:\your_run_root\dialogue-scale-showcase-v12.json

python -m pure_integer_ai.experiments.run_conversation_surface_heldout \
  --project-root . \
  --training-run-root K:\pure_integer_ai_work\dialogue_training_week_v1\dialogue-pack-v6-clean-surface \
  --pack-sha256 1c907caac90c6edb687ad45e0db490da9188028374d90757af8fc28b720ce03d \
  --output K:\your_run_root\dialogue-surface-heldout-v1.json

python -m pure_integer_ai.experiments.run_conversation_multiturn_scale \
  --project-root . \
  --database K:\pure_integer_ai_work\broad_qa_week_v1\indexes\broad-qa-20k-from-100k-target-v2.sqlite3 \
  --training-run-root K:\pure_integer_ai_work\dialogue_training_week_v1\dialogue-pack-v6-clean-surface \
  --pack-sha256 1c907caac90c6edb687ad45e0db490da9188028374d90757af8fc28b720ce03d \
  --output K:\your_run_root\dialogue-multiturn-scale-v1.json

python -m pure_integer_ai.experiments.run_conversation_multiturn_scale_v2 \
  --project-root . \
  --database K:\pure_integer_ai_work\broad_qa_week_v1\indexes\broad-qa-20k-from-100k-target-v2.sqlite3 \
  --training-run-root K:\pure_integer_ai_work\dialogue_training_week_v1\dialogue-pack-v6-clean-surface \
  --pack-sha256 1c907caac90c6edb687ad45e0db490da9188028374d90757af8fc28b720ce03d \
  --output K:\your_run_root\dialogue-multiturn-scale-v2.json

python -m pure_integer_ai.experiments.run_broad_qa_dev_surface_audit \
  --project-root . \
  --database K:\pure_integer_ai_work\broad_qa_week_v1\indexes\broad-qa-20k-from-100k-target-v2.sqlite3 \
  --output K:\your_run_root\broad-qa-dev-surface-v1.json
```

## 快速开始

```bash
git clone https://github.com/namina123/pure_integer_ai.git pure_integer_ai
cd pure_integer_ai
python -m pip install -e ".[test]"
python -m pure_integer_ai.crosscut.guards.lint
python -m pytest -q
```

以上命令均从仓库根目录运行。项目支持 CPython 3.11 及以上版本，公开 CI 覆盖 Linux 和 Windows。`data/*.sample` 是可公开分发的格式示例，构建和测试不依赖私有资料或已归档项目。

### 实验性短回答探针

安装后可直接查询当前公开样例所形成的实验性学习结果：

```bash
pure-integer-qa "什么使得河水上涨？"
```

探针接收原始问题，也可用 `--source-ref 1,2,...` 限定来源。默认只输出稀疏短结果；需要完整审计轨迹时显式添加 `--audit`。`--repeat N` 会在同一个已构建运行时上重复查询，用于核对 warm query 的逐位一致性。这个入口只展示当前公开学习样例所覆盖的能力，不代表广域问答或成熟对话能力。

默认启动会校验并加载仓库中的类型化规范快照；快照缺失、损坏或与公开来源身份不一致时，会拒绝部分加载并完整重建运行时。

也可直接进入面向人的逐行短问答壳：

```bash
pure-integer-qa --interactive
你> 什么使得河水上涨？
系统> 暴雨
你> :quit
```

该壳只复用一次只读稀疏运行时；`ANSWER` 显示实际学习到的答案表面，其他情况保持类型化结果，不编造自然语言回复。它不保存终端输入，也不把终端历史冒充为长期记忆。输入 `:quit`、`:exit` 或 EOF 可退出。

需要查看同一公开学习链产生的完整命题句时，可使用独立展示壳：

```bash
pure-integer-qa --interactive-sentence
你> 暴雨使得什么？
系统> 暴雨使得河水上涨。
你> :quit
```

这里的句子直接来自同一条已选择学习证明中的完整生成结果，不是把短答案套进固定句式。它仍是有限公开样例的只读展示，不是自由对话、会话记忆或广域语言理解。

需要让终端输入也经过严格 raw-byte/UTF-8 整数边界时，使用单独的公开课程 demo：

```bash
# 完成上方安装后运行；未安装时先设置 PYTHONPATH=./src
py -3.11 -m pure_integer_ai.experiments.run_public_sentence_demo
# pure-integer-dialogue-demo
你> 暴雨使得什么？
系统> 暴雨使得河水上涨。
你> :quit
```

若只想立即验证第一句回答，也可以省略交互输入：

```bash
py -3.11 -m pure_integer_ai.experiments.run_public_sentence_demo --question "暴雨使得什么？"
# 系统> 暴雨使得河水上涨。
```

`--question="暴雨使得什么？"` 与上面的写法等价；两者都绕过易改写中文编码的文本管道。

这个入口从原始字节输入开始，严格匹配当前公开课程中的已学习路由；成功输出是同一条来源证明中的完整命题句，未学习、歧义或输入错误只显示类型化拒绝码。它不保存终端历史，也不写入长期记忆或数据库。这是有限公开课程演示，不等同于自由对话或广域理解。

当前公开的受限对话入口可使用：

```bash
# 完成上方安装后运行；未安装时先设置 PYTHONPATH=./src
# Windows PowerShell：chcp 65001 > $null，并把输入/输出编码设为 UTF-8
py -3.11 -m pure_integer_ai.experiments.run_public_frame_dialogue
# pure-integer-dialogue
你> 东岸入口何时启用？
系统> 此输入对应多个已学习路径，请重输其中一个完整问题：
澄川码头何时启用？
北川站东门何时启用？
你> 澄川码头何时启用？
系统> 澄川码头于2023年启用。
你> :quit
```

每一行都经过严格的原始字节与 UTF-8 输入边界。当前入口展示的是一个来源绑定的、可审计的两候选澄清流程：
系统先列出完整问题，随后只接受用户重输其中一个完整问题，并复制该问题实际得到的答案。输入必须在 UTF-8
交互终端中直接完成；旧式管道可能先改写中文字节。这个入口是公开窄域演示，不代表自由对话、长期记忆或广域问答。

同一入口也包含一个受限的来源绑定追问链：

```text
你> 寒潮导致什么？
系统> 寒潮使得路面结冰。
你> 它的原因是什么？
系统> 寒潮
```

这里的“它”只在已冻结的当前焦点和来源课程内解析；没有匹配证据时会返回类型化拒绝，不会猜测。

多个不同问题可通过长驻 JSONL 模式共享同一次运行时构建：

```bash
pure-integer-qa --jsonl
{"question":"什么使得河水上涨？"}
{"question":"河水上涨的原因是什么？","audit":false}
```

每个输入对象会立即得到一个结果记录；坏行返回类型化错误并继续处理后续行，输入结束时再输出一条 session probe。

### 来源约束广域问答入口

`pure-integer-broad-qa` 可以从冻结的中文 Wikipedia multistream snapshot 选择页面、构建紧凑整数索引并执行来源约束问答。`ask` 默认输出简洁答案、页面修订和来源链接：

```bash
pure-integer-broad-qa ask \
  --run-root <run-root> \
  --database <run-root>/indexes/broad-qa.sqlite3 \
  "矮寨大桥何时建成通车？"
```

不传问题参数时，`ask` 可从 UTF-8 stdin 逐行读取多个问题，并在同一个只读数据库连接上回答；Windows PowerShell 下建议把中文问题直接作为参数传入，避免旧管道编码改写字符。需要完整候选计数、证据链、raw span/hash 和贡献者信息时使用 `ask --audit`；原有 `query` 继续输出单题完整 JSON，供自动化调用。

构建需要从官方 Wikimedia 地址取得 snapshot manifest 中固定的 XML 和 index 文件；仓库不提交 3.5 GB 原始 dump 或 20k SQLite。公开的固定问题与路径无关探针位于 `data/ph2/broad_qa_dev_questions_v1.json` 和 `scripts/run_broad_qa_dev_probe.py`。此入口目前只做稀疏检索与来源约束抽取，不调用 LLM，也不代表自由生成、成熟对话或开放域语义学习已经完成。

### 公开来源词义探针

`pure-integer-sense` 查询由公开 Wiktionary 与 Wikidata 切片编译的词义候选：

```bash
pure-integer-sense "首页"
pure-integer-sense "金星" --context "距离太阳第二近的行星"
pure-integer-sense "金星" --primitive
pure-integer-sense "金星" --proposition
pure-integer-sense "什么是金星" --definition
pure-integer-sense "什么是金星" --context "{{lb|zh|astronomy}} [[太陽系]]的第二顆[[行星]]，為[[類地行星]]" --display-definition
pure-integer-sense "蘇維埃社會主義共和國聯盟" --artifact-version v2
pure-integer-sense "敗仗" --artifact-version v3
pure-integer-sense "亠" --artifact-version v4
```

结果保留可回溯的来源信息，并明确区分唯一、多义、未知和未合并的来源冲突。当前 artifact 只覆盖仓库中冻结的有界公开切片；候选存在不等于项目宣称它是最终事实，也不代表已经具备开放域词典或广域问答能力。

显式 `--primitive` 会把同一候选投影为类型化来源声明；`--proposition` 进一步给出带结构角色、来源和生命周期的命题投影。两者的含义始终是“该来源如此定义、标注或列为别名”，不是项目对内容真值的最终裁定。

显式 `--definition` 识别“什么是 X”和“X 是什么意思”两类通用中文问式。它只在词义和活动来源定义都唯一时返回来源中的定义原文；多义、跨来源冲突、未知、需要澄清、只有标签或别名、以及同一概念存在多个不同定义时都会拒绝选择。

`--display-definition` 只对上述已经选定的单一来源定义做确定性展示投影，同时保留原始文本、来源、许可、修订身份和完整承诺链。当前仅支持普通 wiki 链接及 `lb`/`label` 领域标签；未知模板、嵌套或不平衡结构、非法转义、歧义链接和非唯一来源会保留 raw 并明确拒绝渲染，不会猜测、删改或调用语言模型润色。所有新增模式均为显式选择，未加参数及原 `--definition` 的输出格式保持不变。

`--artifact-version v2` 显式选择基于同一公开 Wiktionary 快照、按标题长度分层和稳定哈希抽取的扩展 artifact。默认仍为冻结的 `v1`；`v2` 不按定义是否易于解析来重选词条，并保留无定义、非中文定义、redirect 和未知模板的公开审计记录。它扩大的是有来源约束的实验性覆盖，不代表已经成为完整词典或开放域问答系统。

`--artifact-version v3` 在相同公开规则下进一步选择 256 个未被 `v1`/`v2` 使用的标题。对应 census 保留全部页面与定义结果，并把中文真实定义中的未知模板按独立页面、修订和出现次数统计；频率达到门槛仍不会自动授权 renderer，必须另有公开规范证据。默认版本和已有 artifact 字节保持不变。

`--artifact-version v4` 再从同一 2026-07-01 Wiktionary 快照中选择 512 个标题，并在排序前排除 `v1`/`v2`/`v3` 已使用的全部 293 个标题。选择只由冻结快照、标题长度分层和稳定哈希决定；解析器随后只读取命中的 multistream block。公开 census 记录全部选中页面、定义、失败状态和模板频率，并继承公开规范审查结论：`place` 仍被阻塞，`zh-div` 仍未获 renderer 授权。`v4` 扩大的是可归属、可审计的实验性词义覆盖，不是训练、事实裁定或开放域能力声明。

项目同时公开了对 v4 中六类高频模板的规范审查和独立的确定性语义投影器。当前只支持审查证据闭合的中文“另一种写法”“同义词”“中文异体形式”和“姓氏”窄结构；原始模板文本、参数、来源和承诺链均被保留。`rfdef` 是请求补定义的维护标记，不能被伪装为词义；`†` 在冻结快照中没有可核验的模板身份，继续明确拒绝。该投影器尚未改写冻结的 v4 artifact，也不扩大默认命令的行为。

## 仓库结构

- `src/pure_integer_ai/`：可安装的主源码包
- `tests/`：与当前实现对应的公开回归测试
- `data/*.sample`：可公开分发的格式样例
- `.github/workflows/ci.yml`：跨平台测试与凭据扫描
- `scripts/`：可公开复用的开发辅助脚本
- `paper/`：论文 PDF、LaTeX 源码与参考文献

## 论文

本仓库公开并明确保留项目作者已经完成的论文，论文内容维持其发布版本；后续代码状态以本 README 和实际实现为准。

- [论文 PDF](paper/main.pdf)
- [LaTeX 源码](paper/)
- [Zenodo 存档与 DOI：10.5281/zenodo.21431532](https://doi.org/10.5281/zenodo.21431532)

## 参与贡献

欢迎通过 [Issues](https://github.com/namina123/pure_integer_ai/issues) 报告可复现问题、提出设计讨论，也欢迎提交 Pull Request。开始前请阅读 [贡献指南](CONTRIBUTING.md)，并在变更说明中明确行为影响、验证方式和仍未覆盖的边界。

## 开源许可

本仓库的原创代码和文档以 [MIT License](LICENSE) 公开发布。任何个人或组织都可以依照该许可证使用、复制、修改、合并、发布、分发、再许可或销售副本。项目不设置单独商业许可、营收门槛、用途限制、登记流程、事先批准、权利转让或附加协议；`LICENSE` 是本仓库原创内容的唯一许可文本。依赖和外部数据继续适用各自许可，其中广域问答使用的 WikiTextParser 为 GPLv3、OpenCC Python 实现为 Apache-2.0、Wikipedia 派生内容为 CC BY-SA 4.0；详见[第三方许可边界](docs/third_party_licenses.md)。

## 联系

邮箱：2698801855@qq.com
