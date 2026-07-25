[English](README_EN.md) | **中文**

# PIDSLCA：纯整数确定性自学习认知架构

PIDSLCA（Pure-Integer Deterministic Self-Learning Cognitive Architecture）是一个公开的探索性研究项目。它研究一个具体问题：能否在不依赖浮点运算的前提下，构造可在普通个人电脑上运行、可逐位复现、可审计并能够从经验更新的认知系统。

本仓库公开参考实现、当前有效测试、持续集成配置、格式样例、必要开发脚本和论文材料。项目中的“自学习”是研究目标与机制描述，不表示系统已经获得通用智能、自主理解或成熟对话能力。

## 设计重点

- **纯整数核心**：核心状态、计数、强度和协议键使用整数表达。
- **确定性执行**：固定输入和协议状态应产生逐位一致的结果。
- **边计数强化**：关系通过可追踪的整数计数积累并按显式条件提升。
- **结构归纳**：从多个样本的可对齐结构中提取共享骨架。
- **构造性核验**：对可执行结果、逆变换和恢复路径进行独立检查。
- **可审计状态**：区分生产真活、可选设施、仅测试机制和未完成能力，避免把“代码存在”写成“能力达成”。

## 当前状态

状态快照：2026-07-25。

| 范围 | 状态 | 含义 |
|---|---|---|
| `PH1-CORE` | 已完成 | 第一阶段核心设施已完成总装，并形成 `J-F1`。 |
| `F-01` | 已通过受控装配验证 | 已覆盖来源准入、Memory 查询、问答与生成、Use/outcome 归因、回滚、重解析、迁移、恢复、克隆和并行确定性。 |
| `PH1-EXT` | 未完成 | `A-00` 效率与 surface、`A-04` 用户熟悉度/偏好、`A-07` 长文本与长期上下文仍待实现。 |
| `PH2` | 未进入 | 尚未开始正式课程、正式训练资料接入或第二阶段能力建设。 |

这里的 `J-F1` 只表示第一阶段设施可以承载后续资料与训练，不表示系统已经完成正式断奶、达到 `readiness=true`、掌握语言、成为可用聊天助手或具备生产可用性。当前成果仍以受控 fixture 和确定性工程验证为主，不应被解读为通用能力或语义正确性的实证结论。

## 验证记录

PH1 收口时的工程工作区记录如下：

- T0 专项与台账：`75 passed`
- T1 直接依赖：`523 passed`
- T2 汇合回归：`600 passed`
- T3 全量回归两次：每次 `3706 passed`
- `PYTHONHASHSEED=0/1` 的 F-01 报告 SHA-256 一致：`ed7f35522053e3dcb257ee48f49f06ec742d98b5df64a7e8c465e532ca1d0905`
- 守卫、源码编译和 `git diff --check` 通过

以上数字是 PH1 收口时的历史工程记录。当前有效测试已经随本仓库公开，并由本地命令和公开 CI 使用同一套入口执行；已归档实现、过期测试、私有设计记录、本地语料和实验产物不参与构建或验证。

公开仓库迁移后于 2026-07-26 使用 CPython 3.14.3 独立验证：可编辑安装、源码编译和四项内置守卫通过，完整测试结果为 `3708 passed`。

## 快速开始

运行时仅使用 Python 标准库。当前工程验证环境为 CPython 3.14.3。

```bash
git clone https://github.com/namina123/pure_integer_ai.git pure_integer_ai
cd pure_integer_ai
python -m pip install -e ".[test]"
python -m pure_integer_ai.crosscut.guards.lint
python -m pytest -q
```

以上命令均从仓库根目录运行。运行时仅依赖 Python 标准库；`.[test]` 只安装测试所需的 pytest。当前支持 CPython 3.11 及以上版本，公开 CI 覆盖 Linux 和 Windows。

仓库中的 `data/*.sample` 仅用于格式示例。完整语料、凭据、本地配置、日志、数据库和实验产物不包含在 Git 中，也不会被上述检查读取。构建、守卫和测试不依赖未公开文档或归档项目。

## 目录

- `src/pure_integer_ai/`：可安装的主源码包
- `src/pure_integer_ai/cognition/`：认知对象、理解、生成与过程机制
- `src/pure_integer_ai/storage/`：事件、Memory、恢复和持久化
- `src/pure_integer_ai/numeric/`、`src/pure_integer_ai/vm/`：纯整数数值对象与图程序执行
- `src/pure_integer_ai/experiments/`：训练编排、运行时装配和评估协议
- `src/pure_integer_ai/crosscut/`：确定性、整数约束和源码守卫
- `tests/`：当前公开回归测试
- `.github/workflows/ci.yml`：跨平台测试与凭据扫描
- `scripts/`：可公开复用的开发辅助脚本
- `paper/`：论文 PDF 与 LaTeX 源码

## 论文

- [论文 PDF](paper/main.pdf)
- [LaTeX 源码](paper/)
- [Zenodo 存档与 DOI：10.5281/zenodo.21431532](https://doi.org/10.5281/zenodo.21431532)

论文记录其发布时的架构与能力边界；代码后续状态以本 README 和实际实现为准。

## 参与贡献

欢迎通过 [Issues](https://github.com/namina123/pure_integer_ai/issues) 报告可复现问题、提出设计讨论，也欢迎提交 Pull Request。开始前请阅读 [贡献指南](CONTRIBUTING.md)，并在变更说明中明确行为影响、验证方式和仍未覆盖的边界。

## 开源许可

本仓库的原创代码和文档以 [MIT License](LICENSE) 公开发布。任何个人或组织都可以依照该许可证使用、复制、修改、合并、发布、分发、再许可或销售副本。项目不设置单独商业许可、营收门槛、用途限制、登记流程、事先批准、权利转让或附加协议；`LICENSE` 是唯一许可文本。

## 支持与联系

- [支持研究](DONATE.md)
- 邮箱：2698801855@qq.com
