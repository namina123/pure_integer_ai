# Java 17 G0 Portable Reference

当前状态：`REFERENCE_ONLY`。

这是 `GOV-CJSON-1`、G0b/G0c schema 与 G0b-1 chain-shape 的一个独立 Java 17
标准库参考实现。隔离的目标是让核心可以无缝迁移到任何能够表示有限 `u8` 字节序列、`u63`
整数、定长数组及确定性字节拼接的语言；它不是为了给 Python 减少依赖而存在。

本目录只使用 JDK 17 标准库：没有 Maven、Gradle、第三方 JSON/密码库、网络、子进程或
Python wrapper。核心使用自写的受限 GOV-CJSON-1 parser/encoder，并显式把 Java 的 signed
`byte` 转为 `0..255` 的无符号整数。

## 边界

- 不实现或调用 Ed25519、root pin、issuer authorization、revocation cutoff、capability 或任何
  训练/来源资格逻辑。
- 不读取 K 盘、source、metadata、环境、时钟、网络或业务路径。公开 fixture 的读取仅发生在
  `FixtureRunner` 测试适配层。
- 成功只说明当前公开 vector 与固定错误码在本 Java reference 中重现；不证明真实治理可信、
  来源合格、训练就绪，或“所有语言已经证明”。

## 目录

- `src/org/pidslca/g0/GovCjson.java`：有限字节/u63、GOV-CJSON-1、固定 domain message、SHA-256 identity。
- `src/org/pidslca/g0/GovernanceSchema.java`：root-registry、revocation-snapshot、source declaration 的精确 schema。
- `src/org/pidslca/g0/GovernanceChain.java`：三条无序有限链的 structural validation。
- `src/org/pidslca/g0/FixtureJson.java`：仅用于 legacy human-authored fixture 的 JSON 读取。
- `src/org/pidslca/g0/PublicFixtureSource.java`、`ManifestCatalog.java` 与 `FixtureRunner.java`：
  仅用于读取、hash 回读和执行公开 corpus；manifest index/page 先由核心 GOV-CJSON parser
  做 canonical readback，文件系统语义不会进入 core。

## 编译与运行

在 Git 根执行，输出目录放在系统临时目录，不会产生 Git build artifact：

```powershell
$out = Join-Path $env:TEMP "pidslca-g0-java17-classes"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$sources = Get-ChildItem ports/java17_g0_reference/src -Recurse -Filter *.java |
  ForEach-Object { $_.FullName }
javac --release 17 -encoding UTF-8 -Xlint:all -d $out $sources
java -cp $out org.pidslca.g0.FixtureRunner tests/fixtures
```

runner 先消费 pinned manifest index、其 sidecar、3 份 legacy authoring fixture、13 个
canonical page 与 8 个 raw binary input，并逐项核对 filename、byte count 和 SHA-256。当前
direct vector 覆盖为：19 个 wire parser、4 个 wire envelope、2 个 `int[] -> u8` adapter、
5 个跨层优先级、16 个 schema、10 个 chain case（包括两份 input order 相反且均预期 `101`
的 registry 双失败 page）。3 个 RFC 8032 public vector 仅核对公开 transport 长度和 `0|1`
预期值，runner 明确不做任何 Ed25519 调用。

这是一份 Java 单语言交叉验证，不是“全部语言已证明”。它也不构成真实治理可信、来源合格、
训练就绪或 production capability 的证明。
