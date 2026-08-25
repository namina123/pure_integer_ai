"""独立的 UTF-8 JSONL 对话协议入口。

该模块只固定协议模式并委托训练终端的完整初始化、来源查询、拒答和 checkpoint
路径；不会复制或旁路回答逻辑。
"""
from __future__ import annotations

import sys

from pure_integer_ai.experiments.run_trained_dialogue_terminal import (
    main as _terminal_main,
)


def main(argv: list[str] | None = None) -> int:
    """以 JSONL 协议启动独立对话进程。"""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--protocol" in arguments:
        raise SystemExit("run_dialogue_protocol 固定使用 JSONL，不要传 --protocol")
    arguments.extend(("--protocol", "jsonl"))
    return _terminal_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
