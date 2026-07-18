"""pure_integer_ai — 重来主线源码包（深化版·从零起）。

依据 doc/重来_落地规划与实施顺序.md 的目录结构与落地顺序。

分层（单向依赖·松耦合）::

    crosscut  ←  numeric  ←  storage  ←  vm  ←  algorithm  ←  cognition
                                              ←  teacher   ←  training  ←  experiments

旧代码 `_archive/legacy_v1/pure_integer_ai/` 全归档 reference-only，新包只参考原理非搬代码。
Stage 0 起手：crosscut（铁律地基）+ numeric。
"""
