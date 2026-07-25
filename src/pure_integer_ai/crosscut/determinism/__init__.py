"""crosscut.determinism — 确定性/可复现/审计（依赖 integer + guards）。

模块：
  hasher       Hasher(seed) FNV-1a 63-bit 带种子哈希 + canonical 编码（跨宿主 bit 一致）
  drng         DRNG splitmix64（seed→整数流·唯一确定性随机入口·禁 random）
  golden       append-only golden 快照库（只增·序号单调）
  reproducible assert_reproducible（同 seed 两跑 hash 恒等→写 golden）
  audit_event  append-only 审计事件（timestamp_seq AUTOINCREMENT + event_hash 链式 prev_hash）
  cross_radix  longdiv 直接版 vs limb 版逐位比对（DiffReport·空=一致）

铁律：核心无墙钟（timestamp_seq 是唯一时间源）/ 确定性 bit-identical / append-only。
"""
