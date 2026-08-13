"""把来源约束广域问答结果渲染为简洁、可核验的交互文本。"""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_broad_qa_contract import BroadQaResult


def _primary_answer_line(answer: str) -> str:
    """返回算法排名第一的非空证据回答，完整证据链留给 audit。"""
    for raw in answer.splitlines():
        value = raw.strip()
        if value:
            return value
    return ""


def render_broad_qa_text(result: BroadQaResult) -> str:
    """渲染一条不隐藏拒答状态且保留首要来源身份的中文回答。"""
    if not isinstance(result, BroadQaResult):
        raise TypeError("broad QA interactive result 类型非法")
    lines = [f"问题：{result.question}"]
    if result.status == "ANSWER":
        answer = _primary_answer_line(result.answer or "")
        if not answer or result.title is None or result.source_url is None:
            raise ValueError("broad QA ANSWER 缺少可显示内容")
        lines.append(f"回答：{answer}")
        lines.extend((
            f"来源：{result.title}，修订 {result.revision_id}，"
            f"{result.license_id}",
            result.source_url,
        ))
    elif result.status == "CLARIFY":
        lines.append("结果：存在多个接近候选，请补充更明确的实体或限定。")
    elif result.status == "CONFLICT":
        lines.append("结果：来源证据存在冲突，当前不作确定回答。")
    else:
        lines.append("结果：未找到足够且可核验的来源证据。")
    return "\n".join(lines)


__all__ = ["render_broad_qa_text"]
