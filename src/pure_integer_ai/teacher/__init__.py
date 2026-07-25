"""teacher — 录放层（依赖 cognition·断奶前合法 / 断奶后退场·§十二 + §十一录放层守卫）。

录放层是断奶前教师元定义/define/QA/权重标定/G5 ground-truth 的唯一可复现通道
（录一次重跑零 LLM·§十一 E4/E10）。断奶后退场（新遇未知靠 SHADOW + 晋升闸·D 墙）。

模块：
  teacher_boundary   verify_teacher_boundary 白黑词汇表机械核查（§9 A2 执行点·外部只启发绝不注入边语义）
  recordable_teacher RecordableLLMTeacher（MODE_RECORD/MODE_REPLAY/OFF·miss→None 无 fallback·E4）
  weaning            断奶判据（双曲线趋势 D1·window_rounds=4 runs·非布尔阈值）

铁律：纯整数（recording key=Hasher.h63·response_int 0/1·度量×1000）/ 确定性（录一次重跑 bit-identical）/
  不走外挂 LLM（MODE_REPLAY 零 LLM·MODE_RECORD 离线录制墙钟合法但重跑走录放层）/ 外部只启发
  （verify_teacher_boundary 拒越界·白黑词汇表机械核查·教师给事实判断非推理规则）/ append-only（recording 表）。
依赖方向：cognition ← teacher（单向·teacher 调 cognition 读图·cognition 不反向调 teacher）。
"""
