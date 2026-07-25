"""文件化语言信号 seed 的 opt-in 装配入口。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.cognition.shared.graph_ontology import GraphOntology
from pure_integer_ai.cognition.shared.language_signal import (
    LanguageSignalRuntime,
    read_language_signal_catalog,
)


def build_language_signal_runtime(*, backend, concept_index,
                                  ontology: GraphOntology,
                                  seed_path: str | Path) -> LanguageSignalRuntime:
    """读取来源化 seed 并构造尚未写入的图 runtime。"""
    catalog = read_language_signal_catalog(seed_path)
    return LanguageSignalRuntime(
        backend=backend,
        concept_index=concept_index,
        ontology=ontology,
        catalog=catalog,
    )


__all__ = ["build_language_signal_runtime"]
