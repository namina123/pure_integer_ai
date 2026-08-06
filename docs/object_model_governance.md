# 对象模型治理 / Object Model Governance

生产代码以值结构为默认选择，只在对象拥有生命周期、资源、缓存、持久化边界或可替换协议时保留行为类。新增生产类必须在类或其首个装饰器的紧邻上一行声明对象类别。

Production code defaults to value structures. Behavioral classes are retained only for explicit lifecycles, resources, caches, persistence boundaries, or replaceable protocols. Every new production class must declare its object category on the line immediately before the class or its first decorator.

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

规则如下：

- `value` 必须使用 `@dataclass(frozen=True, slots=True)`。
- `lifecycle` 必须声明 `owner` 和 `cleanup`，明确状态归属与清理边界。
- `protocol` 和 `exception` 必须与对应继承家族一致。
- 冻结 baseline 只承认 guard 引入前已经存在的类；不得把新类加入 baseline。
- 既存类可以删除或增强为 frozen/slots。身份类别漂移、移除 frozen/slots 或新增未声明类都会使 guard 失败。
- 既存类补齐声明时，必须同时从 baseline 删除对应条目，避免以后退回旧豁免。

The rules are:

- `value` requires `@dataclass(frozen=True, slots=True)`.
- `lifecycle` requires `owner` and `cleanup` metadata.
- `protocol` and `exception` must match their inheritance families.
- The frozen baseline covers only classes that existed when the guard was introduced. New classes must not be added to it.
- Existing classes may be deleted or strengthened with frozen/slots semantics. Category drift, removing frozen/slots, or adding an undeclared class fails the guard.
- When an existing class gains a declaration, remove its legacy baseline entry so the exemption cannot return later.

从仓库根目录运行：

Run from the repository root:

```bash
python scripts/object_model_lint.py
python -m pytest -q tests/test_object_model_lint.py
```
