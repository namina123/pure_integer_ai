"""experiments — 实验入口层（依赖 training·可换·§一最上层）。

模块：
  metrics         度量 jsonl（同源 D2·oracle导通率/source_dist/断奶曲线）
  dump_to_sqlite  便携 SQLite 导出（portable artifact·非权威 dump）
  collection      五类收集框架（CollectedItem/CollectionSource·E10 local_dir 首选·E5 graceful）
  formal_train    正式训练驱动（§十二五阶段 + --resume 续训 + E7 pre-flight 放量门 + H2 标定 + 终 dump）

依赖方向：cognition ← teacher ← training ← experiments（单向·experiments 调 training+cognition+teacher·
  不反向）。experiments 是可换层（RoundRunner/CollectionSource 注入式·系统定义契约·用户填实现）。

铁律：纯整数 / 确定性 bit-identical（round_id 整序无墙钟·终 dump 同 cursor.dump_run）/
  gate 二分（TRAINING_MODE/TEACHER_MODE 默认 OFF·live-read）/ 几百G不重训（新 run_id·终 dump base·
  cursor stage-skip·度量门控合格才进下·E7 pre-flight 放量前守卫）。
诚实边界：experiments 是编排非认知（能力涌现非训练保证·地基建好≠能力必现·D 墙）/
  pre-flight 软守卫非硬保证 / 默认权重期 reward 不落 strength（H2） / stable≠correct。
"""
from __future__ import annotations
