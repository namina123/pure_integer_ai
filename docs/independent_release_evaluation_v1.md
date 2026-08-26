# 独立发布版本评测

`scripts/run_independent_release_evaluation.py` 对 K 盘 release root 执行一次可复现的独立进程评测。评测不读取私有标签，也不修改发布包；所有输出写入新的 K 盘目录。

```powershell
python scripts/run_independent_release_evaluation.py `
  --release-root "K:\pure_integer_ai_work\model_releases\public-dialogue-independent-v10-shard2-stage1-20260826" `
  --output-root "K:\pure_integer_ai_work\independent_release_eval_<new-id>"
```

评测覆盖：

- release 内 held-out 标题问答；
- 不在知识索引中的开放集负例与未知问题；
- 显式 `CONFLICT` Runtime 资料必须返回 `CLARIFY`；
- 两个独立 Runtime 来源的 citation 完整性；
- 关闭并重启进程后的 checkpoint 与指代追问；
- 冷启动/热查询两档 p50、p95、SQLite 读取数和峰值工作集。

输出 `independent_release_evaluation.json` 及同名 `.sha256`。`status=PASS` 仅表示上述边界在该次冻结输入上全部通过，不等价于通用智能或无限记忆；`status=NE` 必须保留原始失败记录，不得改写成通过。

## Runtime ledger manifest

新的 Runtime 资料 run 会额外写出 `runtime_material_manifest.json`。其中列出 SQLite 和整数 event/observation/binding 文件的大小与 SHA-256、SourceRef 完整整数键、许可/批次/Companion 身份以及 scope/observation 摘要。加载时对文件集合、摘要和来源记录做闭包校验；缺失或漂移会 fail closed。旧的无 manifest run 仍可只读回放，以保持兼容。
