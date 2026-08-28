# 独立对话版本 v0.1

发布日期：2026-08-28
版本标识：`public-dialogue-independent-v19-text-organization-20260828`

这是项目首个可独立回读的受限对话版本。它不是神经网络权重包，而是由公开课程训练出的确定性整数状态 artifact，以及复用同一理解、来源、证据、拒答和会话恢复链的运行入口。它能做来源约束的中文事实问答、有限多轮焦点保持和运行时资料导入；内部可以保留不确定性与澄清判定，但正式用户接口只呈现面向人的自然语言，不向调用者暴露内部状态枚举。不宣称自由生成、任意来源真值、无限记忆或通用智能。当前 release 仍是 Stage 1 候选，不是最终 readiness。

## 运行准备

完整 artifact 位于调用者自己的 K 盘 release root，例如：

```text
K:\pure_integer_ai_work\model_releases\public-dialogue-independent-v19-text-organization-20260828
```

release root 已包含广域 QA SQLite 索引、训练状态、公开课程、整数 sidecar、来源清单和协议
配置。仓库不携带这些 K 盘大文件；发布包本身可以脱离外部 QA SQLite 独立启动。

## 人类终端

```powershell
python -m pure_integer_ai.experiments.run_trained_dialogue_terminal `
    --release-root "K:\your_project\model_releases\public-dialogue-independent-v19-text-organization-20260828" `
  --session-root "K:\your_project\sessions\dialogue-v0.1"
```

## JSONL 交流协议

使用专用 `run_dialogue_protocol` 模块时，标准输入和标准输出均为无 BOM UTF-8 JSONL。每行请求必须是对象：

```json
{"id":"q1","op":"turn","text":"什么使得河水上涨？"}
```

返回值的核心字段形如：

```json
{"id":"q1","ordinal":0,"text":"暴雨导致河水上涨。","type":"response","citations":[],"turn_key":[...]}
```

正式响应对象包含 `id`、`type`、`ordinal`、`text`、`citations` 和 `turn_key`。
其中 `text` 是唯一的回答内容，始终是面向人的自然语言；来源存在时，`citations`
提供可核验的来源信息。内部理解/评测状态、原始答案槽和检索问式不属于正式用户协议，
不会出现在响应对象中。协议错误和会话结束同样通过自然语言 `text` 返回；`type`、`id`
和 `ordinal` 仅用于协议编排，不能被解释为回答状态。
结束会话：

```json
{"id":"q2","op":"quit"}
```

机器入口示例：

```powershell
python -m pure_integer_ai.experiments.run_dialogue_protocol `
    --release-root "K:\your_project\model_releases\public-dialogue-independent-v19-text-organization-20260828" `
  --session-root "K:\your_project\sessions\dialogue-v0.1"
```

`session-root` 可选但必须是 K 盘已存在目录；启用后会话 checkpoint 使用项目的整数格式，关闭进程再启动可以恢复最近有限热历史。协议错误返回 `type=error` 与自然语言 `text`，不会把错误请求当作问题消费；正式协议不含 `status=INVALID_REQUEST` 或其他内部状态字段。

release root 中的 `model/dialogue_protocol.json` 声明 JSONL、UTF-8、操作集合和 checkpoint
格式，并将响应字段固定为 `text`；启动器会在读取 release manifest 时校验该配置。

## 运行时资料导入

发布包保持不可变；新增资料写入独立的 K 盘 Runtime ledger，再由同一个入口按显式
SourceRef、许可、资格和问题绑定读取。示例：

```powershell
python -m pure_integer_ai.experiments.run_runtime_material_ingest `
  --material-file "K:\your_project\materials\manual.txt" `
  --output-root "K:\your_project\sessions\manual-runtime" `
  --source-kind 93 --source-id 1001 --scope-id 1001 `
  --license-id CC0-1.0 --batch-id 1 `
  --authority-key 7,1001 --version-key 1,1001 `
  --question "夜间模式如何影响屏幕？" `
  --qualification-state SUPPORTED --reason-id manual-authority

python -m pure_integer_ai.experiments.run_trained_dialogue_terminal `
  --release-root "K:\your_project\model_releases\public-dialogue-independent-v19-text-organization-20260828" `
  --runtime-material-ledger-root "K:\your_project\sessions\manual-runtime" `
  --runtime-material-sqlite "K:\your_project\sessions\manual-runtime\runtime.sqlite3"
```

导入器不会调用 LLM、不会修改 Core，也不会从正文猜答案；问题、资格理由和来源身份
必须由调用方显式提供。资料 ledger 可单独备份或撤回，不改变 release root 的文件身份。

## 发布训练身份

- pack SHA-256：`4775c4a1c88d075210e56ae8c2465ec6f34238691ae050f2c0be7447bd5238bc`
- 训练 run：`compact-20k-v2-shard-0002-stage1-20260826`
- 课程记录：5000（train 4308、held-out 475、negative 217；实际训练项 4308）
- 完成阶段：1（由 shard-0001 恢复后重放）
- typed 课程：本候选未启用 typed generation（`typed_items=0`）
- 整数索引：token vocabulary/sequence 与 aggregate occurrence 已绑定到训练 SQLite；
  当前候选输入仍为旧 token-only 分片，aggregate sidecar 的独立训练对照另行记录在 K 盘。
- artifact 清单：release root 内 `public_model_release.json` 及 `source_manifest.json`

发布 artifact 的大文件只在 K 盘保存；公开 Git 只保存协议、版本身份、来源/许可边界和复跑说明。模型能力必须与其来源索引、课程版本和运行 manifest 一起解释，不能只复制一个 SQLite 文件后声称可迁移。

## 已验证边界

真实独立进程已完成单问、边界问题的自然语言回复、跨进程焦点追问，以及独立 Runtime ledger 多来源 citation 回复；会话 checkpoint 可以跨进程回读。内部评测仍记录状态，但不会进入正式用户接口。候选 metrics 记录每轮 SQLite 读取数（25 条）和宿主峰值工作集（约 273 MiB），但尚未完成 held-out/negative/冲突全套正式 readiness 评测。热 SQLite 页缓存下协议 p95 与终端基线同量级；冷启动另行记录，不把冷页加载时间归因于协议编码。

## 对话域隔离

领域课程不会直接并入通用对话模型。发布包中的领域 artifact 由整数路由模型按已学习的区分特征延迟加载；每轮先查询通用模型，只有当前用户输入提供足够领域证据时才查询对应专家。助手上一轮生成的表面不参与领域激活，避免模型输出反过来污染后续路由。领域专家仍受相同的相似度、来源和未知边界约束，未满足证据门槛时保持通用回答或自然语言未知边界。
