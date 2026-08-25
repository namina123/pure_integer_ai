# 独立对话版本 v0.1

发布日期：2026-08-25
版本标识：`public-dialogue-v0.1-20260825`

这是项目首个可独立回读的受限对话版本。它不是神经网络权重包，而是由公开课程训练出的确定性整数状态 artifact，以及复用同一理解、来源、证据、拒答和会话恢复链的运行入口。它能做来源约束的中文事实问答、有限多轮焦点保持和显式 UNKNOWN/CLARIFY；不宣称自由生成、任意来源真值、无限记忆或通用智能。

## 运行准备

训练 artifact 位于调用者自己的 K 盘 release root，例如：

```text
K:\pure_integer_ai_work\model_releases\public-dialogue-v0.1-publish-20260825-r2
```

另需一个已经构建好的 K 盘广域 QA SQLite 索引。仓库不携带训练数据库、索引或大文件。

## 人类终端

```powershell
python -m pure_integer_ai.experiments.run_trained_dialogue_terminal `
  --project-root . `
  --qa-database "K:\your_project\indexes\broad-qa.sqlite3" `
  --training-run-root "K:\your_project\model_releases\public-dialogue-v0.1-publish-20260825-r2" `
  --session-root "K:\your_project\sessions\dialogue-v0.1"
```

## JSONL 交流协议

使用专用 `run_dialogue_protocol` 模块时，标准输入和标准输出均为无 BOM UTF-8 JSONL。每行请求必须是对象：

```json
{"id":"q1","op":"turn","text":"什么使得河水上涨？"}
```

响应对象至少包含 `id`、`type`、`ordinal`、`status`、`answer`、`display_answer`、`retrieval_question`、`citations` 和 `turn_key`。`status` 只有 `ANSWER`、`UNKNOWN`、`CLARIFY` 或 `REPAIR` 等既有对话状态；非 `ANSWER` 不携带事实答案或 citation。结束会话：

```json
{"id":"q2","op":"quit"}
```

机器入口示例：

```powershell
python -m pure_integer_ai.experiments.run_dialogue_protocol `
  --project-root . `
  --qa-database "K:\your_project\indexes\broad-qa.sqlite3" `
  --training-run-root "K:\your_project\model_releases\public-dialogue-v0.1-publish-20260825-r2" `
  --session-root "K:\your_project\sessions\dialogue-v0.1"
```

`session-root` 可选但必须是 K 盘已存在目录；启用后会话 checkpoint 使用项目的整数格式，关闭进程再启动可以恢复最近有限热历史。协议错误返回 `type=error` 和 `status=INVALID_REQUEST`，不会把错误请求当作问题消费。

## 发布训练身份

- pack SHA-256：`96c7d6abcf421ddd8130f9ed6ef74663fed69083f9d42923395903bf56b00615`
- 训练 run：`public-dialogue-v0.1-train-20260825`
- 课程记录：812（train 446、held-out 184、negative 182；实际训练项 445）
- 完成阶段：1、2、3、4
- typed 课程：`GenerationAdoptionPostcheckQuery` 13，`GenerationGeneralizationCandidateV1` 28
- artifact 清单：见 `data/ph2/public_dialogue_model_v0_1_release.json`

发布 artifact 的大文件只在 K 盘保存；公开 Git 只保存协议、版本身份、来源/许可边界和复跑说明。模型能力必须与其来源索引、课程版本和运行 manifest 一起解释，不能只复制一个 SQLite 文件后声称可迁移。

## 已验证边界

真实独立进程已完成六轮 JSONL：来源约束 ANSWER、未知问题 UNKNOWN、无焦点问题 UNKNOWN、来源问答 ANSWER、紧接追问 ANSWER、另一来源事实 ANSWER；会话 checkpoint 可以跨进程回读。metrics 还记录每轮 SQLite 语句读取数和宿主峰值工作集。热 SQLite 页缓存下协议 p95 与人类终端基线同量级；冷启动另行记录，不把冷页加载时间归因于协议编码。
