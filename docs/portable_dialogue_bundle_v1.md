# 便携对话测试包

这是一份供另一台电脑离线测试的代码与模型组合包，不代表正式公开发布或通用能力声明。
模型只读，运行时会话写入单独的 `runtime/`；代码和模型目录彼此分离，也可以通过参数
替换为外部模型目录。

## 环境

- CPython 3.11 或更高版本；
- 64 位系统，建议至少 8 GiB 内存；
- 不需要安装本项目，也不需要 `pip install` 第三方库；
- 完整包约 3.1 GiB，复制介质应有足够空间并支持单个约 2.7 GiB 的文件。

包内不携带 OpenCC 或其他宿主语言转换库。语言表面变体必须来自模型图；缺少图内
证据时不允许由 Python 代码补做简繁、词形或问式转换。Wikipedia/Wiktionary 构建
解析器等训练依赖也不进入运行包。

## 首次核验

在包根目录运行：

```powershell
python run.py verify
```

该命令严格核验运行代码、模型闭合文件集合及逐文件 SHA-256。搬运到另一台电脑后应先
执行一次；成功时输出 `"status":"PASS"`。

## 人类终端

Windows 可双击 `启动终端.cmd`，也可运行：

```powershell
python run.py terminal --session "runtime/session"
```

输入问题后按回车；输入 `/quit` 或 `/exit` 结束。`--session` 可省略；指定后会话 checkpoint
保存在模型之外，重启进程可恢复有限热历史。

## JSONL 接口

```powershell
python run.py jsonl --session "runtime/session"
```

标准输入每行一个 UTF-8 JSON 对象：

```json
{"id":"q1","op":"turn","text":"你能做什么？"}
{"id":"q2","op":"quit"}
```

每个请求对应一行 JSON 响应。用户可见回答只在 `text` 字段，不暴露内部
`UNKNOWN`/`CLARIFY` 状态。

## 指定分离模型

`--model` 必须指向含 `public_model_release.json` 的模型 release root：

```powershell
python run.py terminal --model "E:/models/public-model-gc-v8-dialogue-stage34-20260901"
```

代码包和模型可以放在不同磁盘；路径不要求是 `K:`。模型目录保持只读，`--session` 不得
放在模型内部。

## 性能档位

默认使用 `deferred-narrow-fast`，适合已执行 `verify` 后的交互测试。也可显式选择：

```powershell
python run.py terminal --performance-tier strict
```

`strict` 每次启动都重新执行模型内容 SHA 核验，启动更慢，但回答路径和模型身份不变。

## 当前边界与撤回说明

2026-09-01 的 v8 搬运包只证明代码、模型文件和 checkpoint 可以离线启动。真实人工
试用已确认它的回答主路径仍是来源知识索引，所谓 Core 对话只按字符相似度选回训练整句，
记忆也只是末级回放；它没有形成项目要求的概念图检索、关系组合生成或深层记忆。因此该
包是失败诊断样本，不是可发布模型，也不应继续用于证明对话、理解、记忆或广域问答能力。
