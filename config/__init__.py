"""config — 配置层（零依赖·gate二分·live-read模块属性）。

§六 gate 二分 + I2 落盘。gate 砍到个位数·live-read 用模块属性（env 覆盖·测试可翻）。
本层零业务依赖（仅 os 标准库）·任何上层可读 gate。
"""
from __future__ import annotations
