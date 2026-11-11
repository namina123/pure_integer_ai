# 整数关系图的增量学习

此入口在已有整数模型上学习新的关系和端点，不重新训练父图。课程通过有类型 SQLite
传入，编译、证据学习、图内生成和普通自然输入调用沿现役关系链执行。

课程适配支持以下七个子阶段：

| 子阶段 | 关系 |
| --- | --- |
| PURE_ALIAS_REFERS | PURE_ALIAS、REFERS |
| SUBSET_MEMBER | SUBSET、MEMBER |
| PROPERTY | PROPERTY |
| MEREOLOGY | PART_OF、HAS_PART |
| SIMILAR_ANTONYM | SIMILAR、ANTONYM |
| PRECEDES | EVENT_BEFORE、EVENT_AFTER、EVENT_SAME |
| CAUSES | CAUSES |

注册表中另有 EVENT_UNKNOWN，但现有来源只给出未知方向负例。这个入口不会把负例转成
正向知识。属性强度保留来源中的整数分子、分母，组成、相似/反义、时序和因果保留各自的
证明协议。端点身份替换仅作用于显式 local_id 字段，不会改变碰巧同值的预算或方向。

## 输入与运行

数据库结构由 `pure_integer_ai.experiments.authored_relation_sqlite_course.COURSE_SCHEMA`
定义。课程必须声明来源 URN、归属、CC0-1.0 许可、teacher owner、train 划分和完整分母。
原型以文件名、seed 身份和 SHA-256 绑定；只有唯一的 train/support/NONE 来源可以晋级。
每条关系按原始端点顺序给出整数码点、local_id 与逻辑序。

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -B scripts/train_relation_knowledge_extension.py `
  --course "K:\your_project\courses\relations.sqlite3" `
  --prototype-root data/ph2 `
  --parent-model "K:\your_project\models\parent" `
  --run-root "K:\your_project\runs\relation-increment-v1" `
  --limit 13
```

父模型必须已具备所需组件，并且来源正文从最初就存放在模型外。课程和运行产物使用
调用者显式指定的 K 盘目录；新运行目录不得存在。小批探针默认四条，扩量时显式指定
`--limit`。`--offset` 可以选择下一段课程。

训练依次复制父 Core、学习关系、建立生成关系、继承模型组件并验证新输出。各阶段使用
SQLite cursor 提交；只有原身份、来源和最后已提交模型均匹配时，才能以相同参数增加
`--resume`。未提交阶段留下的部分模型必须保留诊断，另建运行身份。

`training_receipt.json` 记录新增生成关系、父图身份、继承组件、正文计数及来源归档身份。
receipt 是运行摘要，持久学习和恢复状态由有类型数据库承载。训练例和开发探针分别记账；
图内生成通过不等于独立自然语言泛化通过。

## 模型与接口产物

普通输入的Core问式约束需要与当前输入中独立投影的filler逐角色精确对齐，核对来源、
起止和整数值。只命中问式间隔不能吞入连接词或切开已有端点。请求关系尚未闭合时，
对齐的问式仍会限制错误的反向回退；真实多谓词歧义继续保留。

调用 `TrainedGraphQueryBridge` 时须显式注入训练后生成器。完整主线依次连接
`TrainedGenerationConnectorRuntime`、`TrainedOperationCompositionRuntime` 和
`TrainedOperationSequenceRuntime`，并共享 bridge 的 `core_runtime`。未挂载生成器的
bridge 保持无生成结果，不会从来源正文构造回答。模型存在生成关系与当前调用实际
消费并交付这些关系，是两个需要分别验证的条件。

训练正文从首次写入即保存在 `run-root` 下、`model` 目录外的有类型来源归档。普通运行
只挂载训练后的模型组件，会话写入模型外的 companion SQLite。来源归档、课程、原型、
答案、回放和整句正文侧库都不得成为回答后备，也不得放进发布接口。

源码仓库中的课程示例用于开发和复现，JSONL 示例只是课程交换格式，不是模型或会话
存储。即使把完整句子改为整数码点或 BLOB，仍不能因此作为发布数据。本入口没有授予
自由对话或正式发布资格。发布只能携带训练后模型、必要程序和来源/许可/模型身份元数据。

## 测试

```powershell
$env:PURE_INTEGER_TEST_TMP_ROOT = "K:\your_project\tests\relation-increment"
python -B -m pytest -q -p no:cacheprovider `
  tests/test_relation_sqlite_prototypes.py tests/test_authored_relation_sqlite_course.py
```

这些专项检查七类适配、专属协议、属性 Rational 和负例隔离。它们不能替代真实增量学习
及自然输入的查询、生成、交付。可选历史多命题图回归通过 `PIA_TEST_CANONICAL_MODEL`、
`PIA_TEST_COMPOSITION_GRAPH`、`PIA_TEST_SURFACE_VARIANT_GRAPH` 显式提供图文件；不配置时
只跳过这些可选历史集成，不把跳过记为能力通过。

问式作用域的独立专项位于 `tests/test_question_syntax_scope.py`。它覆盖完整、WH和组合
问式的角色跨度、来源、开放谓词及歧义保留，不把纯语法解析当成事实成立。

## 已验证范围

2026-11-12的代表批在父图上实际新增13条有来源关系，均产生了图内生成结果；补齐
问式作用域和运行装配后，同一13条普通自然输入全部经过查询、生成及单命题交付。
运行未读取来源课程，模型组件保持不变，会话保存在模型外的有类型数据库。

这是训练表达的开发验证，没有独立语言held-out，也没有授予广域自由对话或正式发布
资格。属性调用仍明显较慢。相关专项通过与全仓质量状态分别记账：全仓源码和对象
模型守卫尚有失败，不能把此代表批通过解释为全部CI通过。
