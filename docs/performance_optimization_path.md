# 性能优化路径 / Performance Optimization Path

## 目标与判定

项目的性能目标不是减少 `class` 关键字数量，而是在保持纯整数、确定性、
canonical 身份和证据合同的前提下，降低每个有效认知步骤的分配、驻留内存、
间接寻址和无关搜索成本。每个生产优化切片都必须有修改前基线、语义等价测试、
修改后复测和明确恢复点；没有测量收益的结构重写不得扩散。

纯整数不会在 CPython 中自动快于浮点。任意精度整数也是堆对象，较大的整数还会
跨多个 machine word。当前路线先消除 Python 对象模型开销；只有剖析证明解释器
本身成为主要瓶颈时，才评估紧凑整数缓冲区或原生后端。

## 最优顺序

1. **消除热路径重复工作。** 复用固定 Hasher、编译结果、解析表、resolver 和只读
   策略；避免在 token、edge、record 循环中重复编码 seed 或重建索引。该层通常
   风险最低、速度收益最高。固定 seed Hasher 首片已经完成并取得局部实测收益。
2. **压缩高基数值结构。** 按实际构造次数和常驻数量选择不可变 dataclass，分批加
   `slots=True`。主要目标是移除每实例 `__dict__`、改善局部性和降低 GC/allocator
   压力；不能预设构造必然更快。
3. **减少对象数量而不只减少对象体积。** 对嵌套引用、重复 owner/version、稳定键和
   小 tuple 做驻留、ID 化或拥有者级共享；必须保持 owner、版本和 canonical 边界。
4. **将批量图数据改为索引化紧凑存储。** 只有对象计数和访问剖析证明值得时，才把
   同构节点/边从对象数组迁为整数 ID、分栏数组和邻接索引。该层最可能决定长记忆
   容量，但也必须明确整数宽度、溢出、持久化版本和恢复格式。
5. **优化查询算法。** 对长期记忆和指代查询采用候选域、中心扩散、分层热区、停止
   条件与预算，避免全图扫描。算法复杂度的收益通常高于方法调用微优化。
6. **最后处理执行后端。** 若前五层完成后 CPython 装箱、调度或大整数运算仍占主导，
   再评估 `array`/buffer、专用扩展或其他语言实现。跨语言迁移只消费已审计的结构体
   标记和 canonical schema，不直接翻译任意 Python 类。

## 结构体静态标记

结构体使用零运行时成本、可由 AST 工具读取的紧邻注释：

```python
# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class Rational:
    num: int
    den: int
```

- `representation=struct`：字段定义身份；只允许校验、stable key 和 canonical 派生。
- `interop=pending`：尚未审计跨语言整数范围、容器、枚举、可选值和编码。
- `interop=portable`：字段 schema 与 canonical 编码已形成可复测合同。
- `interop=python-only`：已确认依赖 Python 语义，未来必须通过适配器迁移。

标记不等于 ABI，也不自动授权改变 pickle、manifest、receipt 或持久化字节。生产类
只有从 legacy baseline 删除后才算正式进入新治理；静态门禁止声明被悄悄移除。

## 分批准入门

每批最多处理一个紧密模块或一组共同 owner 的值结构，并依次满足：

1. 记录构造次数、常驻实例数、对象与 `__dict__` 体积、构造和访问基线。
2. 检索继承、弱引用、动态属性、`__dict__`、pickle、`dataclasses.replace` 和反射依赖。
3. 添加结构体标记和 `slots=True`，从 legacy baseline 删除并重算其摘要。
4. 验证相等性、hash、排序、异常、不变量、稳定键和 canonical 输出。
5. 运行受影响测试与对象模型门，并复测内存和时间；性能退化必须解释或回退。
6. 生产源码身份变化形成独立 successor 证据；不得继承旧 readiness 或覆盖历史 seal。

## 首批施工

首批对象为 `crosscut/integer/valtypes.py` 的 `Rational` 与 `FixedQuotient`。
`Rational` 的既有剖析注释记录过约 569 万次 `__post_init__` 调用；两者均为冻结、
无继承、无动态属性依赖的纯整数值结构。迁移前本机 100,000 个同时存活实例测得：

| 结构 | 当前实例+字典估计 | 100,000 实例追踪内存 | 构造中位数 |
| --- | ---: | ---: | ---: |
| `Rational` | 48 + 88 bytes | 9,601,264 bytes | 709 ns/item |
| `FixedQuotient` | 48 + 104 bytes | 11,201,392 bytes | 1,432 ns/item |

这些数值只作为本机同进程修改前后对照，不作跨 Python 版本或跨平台承诺。

首批迁移后，100,000 个同时存活实例的同口径结果如下：

| 结构 | 迁移后追踪内存 | 内存变化 | 交错构造中位数变化 |
| --- | ---: | ---: | ---: |
| `Rational` | 5,601,264 bytes | -41.7% | -2.6% |
| `FixedQuotient` | 7,201,392 bytes | -35.7% | -1.5% |

交错构造复测使用同一进程内等价的 legacy/slots 定义、300,000 次/轮、11 轮中位；
它只支持“没有观察到构造性能退化”的本机结论。两类实例均不再有 `__dict__`，
相等、hash、`asdict`、`replace` 与 pickle round-trip 保持；pickle 字节 SHA 发生变化，
因此不得把 Python pickle 当成 bit-identical canonical 合同。
