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

本批生产变更提交为 `c6799cb`，并由 append-only
`struct_layout_successor_receipt_v1.json` 绑定 parent/current 源码身份、布局变化、
本机方向性性能数据和有界验证。receipt 状态仅为
`STRUCT_LAYOUT_SUCCESSOR_EVIDENCED`，显式保持 readiness 未重发、PW-00A 未启动。

## 第二批施工

第二批依据既有 100,000-record SQLite 存储剖析场景选择
`storage/sealed_segment.py`。全局 pytest 调用剖析与 dataclass 包装采样分别超过
5 分钟、3 分钟预算且没有形成可用报告，已经停止，不应沿该高侵入路径重试。

三个候选均先做相同基线与交错复测，最终只接受高频 `SegmentBudget` 和
`SegmentRecord`：

| 结构 | 存活数量 | 迁移前 | 迁移后 | 内存变化 | 交错构造变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `SegmentBudget` | 100,000 | 9,601,264 bytes | 5,601,264 bytes | -41.7% | -4.9% |
| `SegmentRecord` | 100,000 | 9,601,648 bytes | 5,601,648 bytes | -41.7% | -4.7% |

`SealedSegment` 虽可使 20,000 实例由 3,693,712 降至 2,893,712 bytes，交错构造
却稳定慢约 4.1%，且 100,000-record 场景仅需约 100 个 segment。该迁移已撤回，
继续保留在 legacy baseline。嵌套 `SegmentRecord` 改为 slots 后，`SealedSegment`
canonical SHA-256 仍为
`f5a4f44527d919871d01e009763fbca049b363108ccc7e45d4f54804104a0312`。

第二批生产变更提交为 `6c1c70b`，receipt v2 严格串接 v1，并记录两个接受对象、
一个已撤回对象、两次废弃采样和有界验证。其状态仍只表示源码后继证据，
不发布 readiness，也不启动 PW-00A。

## 热缓存后续决策

第三个 slots 候选 `CachedSegmentRecord` 在 100,000 实例测量中可将内存由
12,001,608 降至 8,001,608 bytes，但同进程 5,000 条 page-in+get 交错周期稳定慢
约 3.1%，因此完整撤回且未形成公开代码提交。局部剖析证明真实热点是
`SegmentRecord.size_bytes()`：旧实现为了只取得长度，仍构造完整 framed integer
stream、编码 bytearray，再调用 `len()`。

新增的 `encoded_integer_tuple_size()` 与 `encoded_framed_integer_tuple_size()` 直接按
现有 outer-count、zigzag 和最短 unsigned-varint 规则累计位宽，不改变
`encode_integer_tuple()`。所有 7-bit 边界、负数、127/128 元素、511-bit 大整数和
非法输入异常类型均与实际编码交叉验证。`SegmentRecord.size_bytes()` 改用 framed
直算后，同进程 5,000 条 cache cycle、11 轮交错中位改善约 7.4%，canonical bytes
保持由原编码器产生。

该算法变更提交为 `0bb9a43`，并由独立
`performance_successor_receipt_v1.json` 串接结构体 receipt v2。性能 receipt 明确
限定为本机方向性证据，保持 readiness 未重发、PW-00A 未启动。

后续剖析发现上一版仍在每次 `SegmentRecord.size_bytes()` 调用中重复执行构造时已经
完成的 tuple/int 校验。公共 framed-size API 继续完整校验；新增内部 validated 核只由
`SegmentRecord` 调用，因为其字段已由 `__post_init__` 校验且 frozen+slots。相对上一
公开版，同进程交错测量中单独 size 调用约改善 38.1%，5,000 条 cache cycle 本轮约
改善 14.7%。这些仍是本机方向性数据，不能解释为跨平台固定比例。

已验证长度核提交为 `b9c8170`，性能 receipt v2 严格串接 v1。v1 在当前源码上默认
严格失效，只允许显式 historical 回读；v2 是当前源码证据，仍不转移 readiness。

`CachedSegmentRecord` 的 slots 方案虽被拒绝，但 cache owner 内部仍会在每次 get、
pin、unpin、flush 和 put 时用公开构造器重复验证已知字段。现在保留公开冻结记录和
全部外部校验，内部转换改用私有已验证构造器；同进程 5,000 条 page-in+get、11 轮
交错中位由 102,564,100 降至 70,977,400 ns，约改善 30.8%，实例布局不变。

该变更提交为 `02fff55`，性能 receipt v3 严格串接 v2，并绑定唯一生产源码
`storage/segment_cache.py` 的 parent/current 身份。receipt 固定六处内部构造点、
公开校验与缓存记录布局均未变化、内部 `__post_init__` 调用降为零；状态仍只表示
性能后继证据，不重发 readiness，也不启动 PW-00A。

后续 profile 表明完整 clean/unpinned cache 的 `clear()` 仍会经 `evict()` 对全部键
排序、逐字段校验并再次计算记录大小。`clear()` 已经先拒绝 pinned，并把 dirty 全部
flush 为 clean，因此成功分支可直接清空 owner 字典并把聚合字节计数归零，同时保留
逻辑访问序。固定 5,000 条 page-in+get+clear、11 轮交错中位由 118,593,100 降至
87,606,000 ns，约改善 26.2%；dirty flush、pinned 拒绝、返回计数和公开布局不变。

该变更提交为 `4930d57`，性能 receipt v4 严格串接 v3。v3 在后继源码上默认严格
失效，只允许显式 historical 回读；v4 仍仅为本机方向性性能证据，不转移 readiness。

`QuerySegmentHotSet.iter_range()` 的 record key 来自已经通过 `SegmentRecord` 不变量
和当前 reader epoch 的结果，继续调用公共 `SegmentPageCache.get()` 会对 descriptor/key
重复执行严格整数扫描。新增 cache owner 私有 `_get_validated()`；公共 `get()` 和
pin/unpin 等外部入口保持原校验，只有该 query consumer 使用已验证 lookup。5,000
records、11 轮 page-in+get+clear 交错中位由 85,376,300 降至 62,555,800 ns，约改善
26.8%，不改变记录布局或 public validation。

该变更提交为 `7cb4a3c`，性能 receipt v5 绑定 `query_hot_set.py` 与
`segment_cache.py` 两个源码 identity，严格串接 v4；仍不重发 readiness，也不启动
PW-00A。

对 v4 clean clear 分支做重入审计时发现，flush 回调可能在原 dirty 集之外新增 dirty
或 pinned 条目；直接清空会放宽旧 `evict()` 的拒绝语义。修正版在 flush 返回后重新检查
两类状态，异常时回到原有拒绝路径，clean/unpinned 快路径和 `-262‰` cycle 数据不变。
修正版提交为 `ec03b7a`，receipt v6 严格串接 v5；v5 仅作为 historical 性能证据，默认
不再宣称 current。
