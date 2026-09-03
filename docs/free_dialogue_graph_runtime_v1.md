# 自由对话图运行时施工记录

## 目标

发布运行时只允许 Core 图、Interaction Memory 图、Dialogue/Companion 图三类
训练后结构提供回答。输入经过通用 Unicode 边界拆分和整数 token 片段化，随后
执行理解、过程候选选择、结果 token 组合三个阶段。运行时不读取课程、外部 QA、
OpenCC 或硬编码问答，不使用随机表层兜底。

## 已完成

- `SqliteDialogueSuccessorRuntime` 新增图路径运行方法 `respond_graph`。
- 从 `dialogue_successor_projection` 的 occurrence 端点恢复 current/response
  图路径；只消费已由训练图投影闭合的命题。
- 输入和候选均以纯整数 token/n-gram 特征建立确定性倒排；长片段、当前覆盖、
  历史热区共同参与整数置信度裁决。
- 精确问题使用图中完全相同输入的统计支持决胜；短输入另有整数多重集精确查询，
  避免高频码点 posting 截断使已训练短句不可达。
- 低置信自然语言只从当前命中的图候选回答中选择疑问式澄清；若没有可组合的
  语言证据，严格入口失败关闭。非法 Unicode 标量、替换字符或不可解释的混乱码型
  也只失败关闭；不能把普通自然语言标为 `UNKNOWN`。
- 发布入口启用 strict graph 模式，Dialogue 图不可用时直接报错；旧兼容调用仍保留
  给历史专项测试。发布模式每轮自动写入同级运行时 Interaction Memory SQLite，
  `--session` 可指定独立会话库。
- strict 模式的 Memory 召回只允许回答侧 `speaker_kind=2`；兼容非严格专项仍可
  显式选择旧的全轮次召回语义。每个有效用户输入先写入 Interaction Memory，随后
  才运行 Core/Memory/Dialogue 三图查询；回答侧无证据且 Dialogue 不是 occurrence 精确
  命中时，再以较低整数阈值查询同一会话用户侧 `speaker_kind=1`。因此用户事实可在重启
  后回忆，且不覆盖 Core 或置信度 1000 的精确 Dialogue 答案。
- 运行时内存边界已收紧：不预载全部投影、来源正文或 posting。查询特征触发 SQLite
  倒排分页，命题端点和来源正文仅对 shortlist 冷读，并以有限热缓存复用。
- 结果阶段核验所选 occurrence 的整数 token 后恢复已学习表层，保留空格、换行、
  Markdown 与其他结构，不再把 token 无间隔拼接成失真的句子。
- 删除按 SourceRecord 换行盲配问答的补回路径，避免把代码、Markdown 标记和普通
  文本行误当作 Dialogue successor。
- 三阶段 trace 已提升为 `cognition.shared.dialogue_pipeline.DialoguePipelineTrace` 公共
  值协议；理解 token、过程候选/命题 key、结果 token 和整数置信度均可由其他语言按
  `stable_key()` 重建，运行时 JSONL 同步公开这些整数 trace，不暴露内部状态或课程文本。
- 低证据自然语言不再按 proposition 首条顺序取澄清；运行时以输入的 Unicode 结构形状
  （token/码点数量、类别计数、疑问标点）和图内回答候选的同类形状确定性选取已学习的
  询问结构。仍然只恢复 Dialogue 图 occurrence，不调用语言词表或随机表层。
- 当前训练投影只具备整句 occurrence 恢复，尚无可证明的组合生成器；因此严格入口将
  所有 `input_exact=0` 的近似候选统一标为图内澄清，只有 `input_exact=1` 才发布为回答，
  防止整数重叠分数饱和导致近邻整句冒充自由回答。

## 真实回归（2026-09-02）

新发布根 `trained-graph-dialogue-20260902b` 和便携 JSONL 入口运行：

1. `你好`：`dialogue_graph`，精确命中训练图问候路径。
2. `中国首位宇航员是谁？`：`dialogue_graph`，恢复同源长句并保留标点。
3. 多轮承接问题：在已有图证据下返回图中已学习的承接句。
4. 可解释但未命中的自然语言：返回图中疑问式澄清，不暴露 `UNKNOWN`。
5. 混乱码型：严格入口失败关闭，不触发 boundary 或随机路径。

每轮输入和输出均写入 Interaction Memory 图；训练 SQLite 只读打开。未启动训练，
未读取或修改旧 private evaluator。

## 当前便携包

- 发布根：`K:\pure_integer_ai_work\model_releases\trained-graph-dialogue-20260902d`
- 最新便携目录：
  `K:\pure_integer_ai_work\portable_packages\pure-integer-ai-trained-graph-portable-20260903e`
- 包内 `run.py verify`：`PASS`（2026-09-03）
- 发布模型不含 `fallback_surfaces.txt`；清单只列训练 SQLite、cursor、状态、来源
  manifest 和协议配置。
- 受影响专项 `python -m pytest -q tests/test_conversation_typed_relation_bridge.py
  tests/test_trained_dialogue_memory_graph.py`：`4 passed`（仅专项，未跑全量）。
- 本包约 2.84 GB，主要是训练 SQLite 的完整图本体；不携带课程、外部 QA、论文、
  密钥或本机绝对路径。

Core typed connector 边界已修正：发布图恢复时，binding source 为命题本体的
Representation 槽保持 `emit`，其余训练模板常量/谓词/角色槽按图内声明转为
`silent`。因此完整 Core claim 不会再次被训练句式前后缀包裹；未修改训练
SQLite、课程或任何具体语言文字。

2026-09-03 便携包单进程整合回归覆盖 5 条请求：Core 关系输出、两条 Dialogue
精确后继、普通自然语言澄清和混合码型输入。Core 输出已为 `按开关导致灯亮`；
所有成功返回来源均为 `core_graph` 或 `dialogue_graph`，澄清标记仍来自
Dialogue 图，未触发 boundary。混合码型的替换 Unicode 标量判定由运行时代码
直接 fail-closed；该样例的首次外层命令转义错误未计入模型结果。

## 当前目标口径

自然语言是必须覆盖的对话域，不能以“未命中当前训练句”宣布语言未知；训练和图接线
需要继续扩展覆盖、承接、改写、澄清和语言组织能力。`UNKNOWN` 只保留为开发/评测
内部状态，正式用户接口只输出自然语言，或在无法解释的混乱码型上 fail-closed。

## 下一步

1. 在另一台机器上使用最新便携包验证 JSONL/终端入口及独立会话库；不得以
   字符近邻覆盖 Core/Dialogue 图答案。
2. 扩展广域对话训练覆盖后再做一次整合评估；评估只作为大能力检查，不替代
   训练结果。

## 恢复点（2026-09-02）

- 本轮修复了高频单码点 posting 截断、并列候选误澄清、结果表层空格丢失、
  strict/兼容 Memory 召回边界，以及 Core typed connector 重复发射模板
  literal 的边界错误。
- 旧便携包已移到 `F:\pure_integer_ai_archive\portable_packages_20260902`；K 盘仅
  保留当前模型、训练集和最新便携目录。
- 本轮已完成阶段 trace 公共接线、结构化澄清选择及 Memory 说话者变量隔离；旧
  `20260902h` 包不含本轮修正，后续包需在近似候选门修正后重新构建。字节级 JSONL
  整合回归已验证用户事实跨轮召回、已知问句完整恢复、低证据图内澄清和混乱码
  fail-closed；若进入长训练，按小时级被动等待，不频繁查看。
