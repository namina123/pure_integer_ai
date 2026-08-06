# 参与贡献 / Contributing

感谢你关注 PIDSLCA。项目欢迎可复现的问题报告、设计讨论、文档改进和代码贡献。

Thank you for your interest in PIDSLCA. Reproducible bug reports, design discussions, documentation improvements, and code contributions are welcome.

## 提交前 / Before contributing

1. 先搜索现有 Issues，避免重复报告；较大的行为或架构变更请先开 Issue 对齐范围。
2. 每个 Pull Request 只处理一个清晰问题，不混入无关重构或格式化。
3. 如实描述当前行为、预期行为、验证方式和已知未覆盖范围。
4. 不把受控 fixture 结果表述为通用能力、语义正确性或生产就绪证明。

1. Search existing Issues before opening a duplicate. For substantial behavior or architecture changes, open an Issue to align on scope first.
2. Keep each pull request focused on one clear problem and avoid unrelated refactoring or formatting.
3. State the current behavior, expected behavior, verification performed, and known coverage gaps.
4. Do not present controlled-fixture results as proof of general capability, semantic correctness, or production readiness.

贡献者不需要签署单独协议或转让权利。提交到本仓库的贡献按仓库现行 MIT License 提供；请只提交你有权公开的内容。

Contributors do not need to sign a separate agreement or assign rights. Contributions submitted to this repository are provided under its current MIT License; submit only material you have the right to publish.

## 代码约束 / Code requirements

- 保持纯整数核心、确定性执行、只追加审计和单向依赖约束。
- 运行时依赖限于 Python 标准库，除非社区先就变更达成明确共识。
- 解释性注释和文档字符串使用中文；公共标识符、协议常量和外部格式名可保留英文。
- 新增行为应提供可证伪的验证，失败路径不得静默降级。
- 新增生产类必须使用紧邻类或其首个装饰器的 `object-model` 注释归类；值对象使用 `@dataclass(frozen=True, slots=True)`，生命周期对象声明 `owner` 与 `cleanup` 边界。

- Preserve the pure-integer core, deterministic execution, append-only audit, and downward-only dependency constraints.
- Runtime dependencies remain limited to the Python standard library unless a change is explicitly agreed on first.
- Explanatory comments and docstrings are written in Chinese; public identifiers, protocol constants, and external format names may remain in English.
- New behavior should include falsifiable verification, and failure paths must not degrade silently.
- Every new production class must carry an adjacent `object-model` classification comment. Value objects use `@dataclass(frozen=True, slots=True)`; lifecycle objects declare their `owner` and `cleanup` boundaries.

支持的声明形式如下；既存未声明类由冻结 baseline 管理，不得把新类加入 baseline：

The supported declarations are shown below. A frozen baseline covers existing undeclared classes; new classes must never be added to it:

```python
# object-model: value
@dataclass(frozen=True, slots=True)
class SourceRef:
    source_id: int

# object-model: lifecycle; owner=request; cleanup=scope-end
class QueryRuntime:
    pass

# object-model: protocol
class SourceReaderProtocol(Protocol):
    pass

# object-model: exception
class SourceReadError(Exception):
    pass
```

## 本地检查 / Local checks

从仓库根目录安装测试依赖并运行守卫与完整测试：

Install test dependencies and run the guards and full suite from the repository root:

```bash
python -m pip install -e ".[test]"
python -m pure_integer_ai.crosscut.guards.lint
python scripts/object_model_lint.py
python -m pytest -q
```

提交 Pull Request 时，请在说明中列出实际执行的检查。若未能运行某项检查，请明确说明原因。

When opening a pull request, list the checks actually run. If a relevant check could not be run, state why.

## 社区协作 / Community conduct

请围绕技术事实讨论，尊重不同经验背景，不进行人身攻击、骚扰或歧视。维护者可以关闭破坏协作、泄露隐私或与项目无关的内容。

Keep discussion focused on technical facts and respect different levels of experience. Personal attacks, harassment, and discrimination are not acceptable. Maintainers may close content that disrupts collaboration, exposes private information, or is unrelated to the project.
