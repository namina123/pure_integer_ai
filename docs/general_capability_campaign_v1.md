# 通用对话能力推进轮 v1

状态：已完成首个主训练、独立 release 组装与 30 轮交流验收；保留性能优化和公开 validator 为后续工作。

本阶段目标不是宣称通用智能，而是把当前的来源约束问答原型推进到一个能真实交流、能吸收更多公开语言材料、能在多轮中保持上下文并可独立发布的候选版本。既有已完成的四阶段 run 和历史评测保持只读，不覆盖、不重跑、不继承其能力结论。

## 验收成果

完成后必须存在一个新的 K 盘 campaign root，至少包含：

- 一条连续的 Stage 1-4 训练谱系，以及训练 SQLite、cursor、pack/source manifest、运行摘要和逐文件 SHA；
- 由该 run 构建的独立 release root，可脱离外部 QA SQLite 启动 JSONL/终端协议；
- 一次端到端交流记录：不少于 30 轮，覆盖 3 个以上知识域、自然追问、未知/澄清、来源引用、进程重启后的历史召回和运行时资料导入；
- held-out、negative、冲突/多义输入仍按既有边界处理，不以固定答案或回退路径掩盖失败；
- 一次性能记录，至少含 warm p50/p95、SQLite 读取数、峰值工作集，并与当前发布候选比较；
- 发布包不含密钥、私有标签、绝对本机路径或论文内容。

## 数据分层

### 语言与对话训练

优先消费已有公开课程：

1. 仓库内 authored 的完整句、关系、因果、指代、结构载体和回答组织课程；
2. K 盘 `oasst1_zh_human_dialogue_course_v2.jsonl` 与 `oasst2_zh_human_dialogue_course_v2.jsonl`；
3. K 盘 `kdconv_human_dialogue_course_v1.jsonl`，优先使用其人类多轮路径，不把来源 metadata 送入语言通道。

KdConv 的原始文件按上游 split 和领域排序，不能直接用文件前缀截断。已用
`select_dialogue_course_splits` 按内容哈希稳定抽取 `1,500 train + 300 heldout` 到
`K:\pure_integer_ai_work\general_capability_campaign_20260829\kdconv_train1500_heldout300.course.jsonl`，并保留选择 manifest。

本机只有约 34 GB 内存、K 盘剩余空间约 63 GB。首轮以 3,000 条左右 pack case 为上限，使用 SQLite `bulk` 写入；若实际峰值接近内存硬上限，降低同一 run 的 case 上限并保留原因，不删除既有 artifact。不得直接把 8 万条 KdConv 记录一次性送入严格 typed 阶段。

### 知识来源运行时接入

K 盘已有的 SciDB CSQ、WushuQA 和中文 Wikipedia 索引作为来源约束知识层接入，不冒充语言训练样本：

- CSQ：`K:\pure_integer_ai_work\scidb_training_courses_csq_20260826\csq_course_v1.jsonl`；
- WushuQA：`K:\pure_integer_ai_work\scidb_training_courses_wushuqa_20260826\wushu_qa_course.jsonl`；
- Wikipedia：沿用当前已发布的来源对齐索引。

若要让 CSQ/WushuQA 进入语言训练，必须另建带来源和许可的对话/命题适配器；在适配器完成前，只接入运行时来源查询和引用链。

## 执行顺序

### A. 一次主训练

使用现有 `run_conversation_training`，默认 authored 课程加 OASST1/2，并加入 KdConv 的受控前缀，建立新的 run id。先连续完成 Stage 1-4；不把每个阶段拆成大量临时 probe。训练只读 D 盘代码，所有 run、SQLite、cursor、课程副本和中间产物放 K 盘。

推荐首轮参数：

```powershell
New-Item -ItemType Directory -Force K:\pure_integer_ai_work\general_capability_campaign_20260829 | Out-Null
python -m pure_integer_ai.experiments.run_conversation_training `
  --project-root . `
  --run-root K:\pure_integer_ai_work\general_capability_campaign_20260829 `
  --run-id gc-dialogue-stage1234-20260829e `
  --stages 1,2,3,4 `
  --max-cases 3000 `
  --extra-course K:\pure_integer_ai_work\oasst1_training_course_zh_human_v2_20260826\oasst1_zh_human_dialogue_course_v2.jsonl `
  --extra-course K:\pure_integer_ai_work\oasst2_training_course_zh_human_20260827\oasst2_zh_human_dialogue_course_v2.jsonl `
  --extra-course K:\pure_integer_ai_work\general_capability_campaign_20260829\kdconv_train1500_heldout300.course.jsonl `
  --extra-course data\ph2\dialogue_relation_causes_scale_v1.course.jsonl.sample `
  --extra-course data\ph2\dialogue_relation_causes_scale_v2.course.jsonl.sample `
  --extra-course data\ph2\dialogue_postcheck_bridge_train_v1.course.jsonl.sample `
  --portable-source-identity `
  --storage-performance-mode bulk
```

如果首轮因内存不足停止，不能静默改写为成功；建立新 run id，降低 `--max-cases`，并在摘要中保留失败原因。成功后才进行 release 组装。

### B. 运行时知识与记忆接线

训练完成后，沿既有独立发布入口接入 Wikipedia/SciDB 来源 owner 和 Runtime material ledger。记忆层的施工目标是：

- 用户轮次继续写入 Runtime ledger 和 checkpoint；
- 在现有精确 item 读取、历史 n-gram 召回之外，增加语言无关的结构/命题索引；
- 查询顺序固定为当前热区、同会话冷历史、显式运行时资料、来源知识索引；
- 保持 source/scope 隔离，支持 revision/tombstone/conflict，不把普通 assertion 自动写入 Core；
- 后续再实现带资格、同意和回放凭据的 Runtime → Core 晋升，不在本轮用隐式晋升冒充学习。

### C. 独立发布与交流验收

从新的训练 run 生成 release root，使用现有独立协议入口运行一组完整交流脚本。验收只保留一份端到端记录和必要的性能记录，不执行全量 CI 或重复历史评测。必须确认：

- 用户输入可在同一 session-root 重启后被召回；
- 新增资料可被读取并逐来源引用；
- 新问法不会只依赖精确字符串命中；
- 未知、冲突和多义问题不会被旧来源泄漏或硬编码答案填充；
- release root 内身份、manifest 和 SHA 可复现。

## 训练后判断

- 若新 run 只增加了可消费语言材料，但生成仍主要是抽取/后继恢复，则标记为“广域语言覆盖扩大”，不宣称自由生成；
- 若多轮、来源和资料导入均可在独立进程稳定工作，则可发布新的交流候选；
- 只有当 Runtime 语义索引、自然语言查询、修订/删除和受控 Core 晋升全部闭合，才进入“持续学习”阶段；
- 任何单项失败都保留失败证据，不能用文档、状态码或小样本探针替代能力结果。

下一恢复点：首个主训练 run 结束后，读取其 `training_summary.json` 与 `run.manifest.json`，只做一次资源/能力判断，再决定是否组装 release；训练期间采用长等待，不进行分钟级轮询。

## 本轮实绩（2026-08-29）

主训练已经完成，状态从“待训练”更新为“已形成首个可独立运行候选”。训练 run 为
`K:\pure_integer_ai_work\general_capability_campaign_20260829\gc-dialogue-stage1234-20260829e`，四个阶段均完成，`weaning_ready=true`，训练 pack SHA 为
`31c0392a5b915124523d31d810bbf0d5f593cb29ba3593e32c8659ee35a5f7b9`。该 run 写入
2290 条训练项、406545 个 occurrence、1813 条对话后继和 251754 个后继特征，峰值工作集
为 11541639168 字节。

由 OASST1、OASST2 和 KdConv 稳定切片构建的整数对话 artifact 位于
`K:\pure_integer_ai_work\general_capability_campaign_20260829\learned_dialogue_response_gc_v1_20260829`，artifact/model SHA 为
`6264e6907877a60a2081a331a0c9ea7f7420b7b36994968435c820a0a7a44ba2`，held-out SHA 为
`fc676e58372a1017726d2c5f7b56870fb2889d4d18fa29592ff2d6463d4e1931`，能力状态为 `PASS`。

独立发布根为
`K:\pure_integer_ai_work\model_releases\public-model-gc-v1-20260829`。它内含训练 SQLite、广域 QA、SciDB passage、JSONL 协议、来源/许可 manifest、cursor 和逐文件 SHA，启动不依赖外部 QA 路径，不携带密钥、私有标签或绝对路径。

用户可见 JSONL 入口已完成 30 轮交流：14 轮答案、3 轮澄清、13 轮自然语言未知；覆盖对话代码/知识、Wikipedia、SciDB、未知问题和长句组织。记录位于
`K:\pure_integer_ai_work\dialogue_sessions\gc-v1-30round-20260829`，性能摘要为 warm p50=18925 微秒、p95=192249 微秒、总 SQLite 语句 38、峰值工作集 209182720 字节。首轮惰性窄域加载产生一次 1293555 微秒峰值，后续优化应优先消除该冷启动成本。

跨进程 checkpoint 已在
`K:\pure_integer_ai_work\dialogue_sessions\gc-v1-checkpoint-20260829` 验收，第二进程从 ordinal 1 继续；运行时资料导入已在
`K:\pure_integer_ai_work\dialogue_sessions\gc-v1-runtime-import-20260829-c` 验收，首问和特征重合追问均返回来源引用。

本轮发现并修复了 Runtime 追问调用已删除硬编码指代表导致的 `ImportError`：现在可注入对话层 resolver，未注入时仅基于同一绑定问题的确定性特征重合，缺少证据则保持澄清/未知。OASST1 回应组织 artifact 的 held-out 能力仍为 `NE`，因此没有把它放入首个 release，避免把未达标文件误当能力结果。下一阶段应在不改变当前发布根的前提下，优化冷启动与长会话性能，再补独立发布 validator 的公开运行证据。
