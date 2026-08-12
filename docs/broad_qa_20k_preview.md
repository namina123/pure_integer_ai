# 20k 来源约束广域问答预览

这是项目在冻结中文 Wikipedia 快照上的第二个真实规模纵切。它把稳定选择的 100,000 个候选页面作为来源上限，并在不按可答性重选的前提下，构建出 20,000 个可投影页面的紧凑整数索引。

## 公开事实

- 来源：`ZHWIKIPEDIA_20260701`，快照 manifest SHA-256 为 `0e81569aaf6cf9cb688b41da27d5eff19707153ee5c74bf9bf362f34427869dd`。
- 候选上限：100,000；最终 accepted 页面：20,000；cutoff selection ordinal：64,236。
- 最终 SQLite：109,006 passages、3,608,002 terms、251,494,400 bytes；SHA-256 为 `e18db72b090dfdfd96aac23c74a5ad0751afe17c2dcfb02fc91f1213b0f7c4da`，SQLite `integrity_check=ok`。
- 每个回答保留页面、修订、贡献者、原始证据 span/hash、来源 URL 和 CC BY-SA 4.0 身份。未知实体不会因为关系词共现而被弱相关页面冒答；缺少标题锚点时查询会拒答或澄清。

## 探针边界

公开固定问题集仍是 24 问开发探针，永久范围为 `DEVELOPMENT_VERTICAL_PROBE_NOT_HELD_OUT`，不是独立 held-out。20k 运行结果为 `22 ANSWER / 1 UNKNOWN / 1 CLARIFY`，22 个引用的冻结源重建审计失败数为 0；与已公开 10k 运行逐题答案和状态无变化。

该结果证明的是来源约束的稀疏检索、证据抽取、拒答/澄清和可审计发布链在更大索引上的一次真实运行。它不证明通用问答、自由生成、长会话、开放域语义学习、永久记忆或断奶。20k 的 posting 外排合并和 SQLite 发布耗时仍是当前主要性能热点，后续会单独优化。

## 复核入口

机器可读 receipt 位于 [`data/ph2/broad_qa_20k_preview_receipt_v1.json`](../data/ph2/broad_qa_20k_preview_receipt_v1.json)。仓库不提交 3.5 GB 原始 dump、20k SQLite 或可再生分片；使用官方来源、冻结 manifest、公开构建 CLI 和 receipt 中的身份即可重建。构建时应使用显式的大数据工作根，不要把原始语料或大索引写入源码树。
