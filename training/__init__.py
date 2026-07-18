"""training — 训练编排层（依赖 cognition + teacher·§十二五阶段）。

模块：
  stages   五阶段编排（结构骨架→因果+抽象→reward→promote 断奶→多模态 defer）+ G5/C6 harness 真接线
  oracle   oracle 标定（B1-B4 占位校验 + H2 小批量权重标定 + reward 导通率）
  promote  promote 三重（频次/reward/定义·SHADOW→PRIMARY tier flip·MUTABLE_MONOTONE）
  cursor   dump 续训（per-space dump·新 run_id·几百G不重训红线·E1/E4/E8）

铁律：纯整数 / 确定性 / 不写死（阶段配比涌现自§十二·阈值 oracle 标）/ 外部只启发（教师经录放层）/ 不走外挂 LLM
  （断奶后退场）/ 几百G不重训（每 run 新 run_id·终 dump base·度量门控合格才进下阶段）。
依赖方向：cognition ← teacher ← training（单向·training 调 cognition+teacher·不反向）。
"""
