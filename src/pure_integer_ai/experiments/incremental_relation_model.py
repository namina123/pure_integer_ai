"""训练期续接已去除来源正文的关系模型，保留父图全部身份。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    TrainedRelationGraphError, TrainedRelationGraphRuntime,
)
from pure_integer_ai.experiments.relation_language_carrier import learn_carrier_frame
from pure_integer_ai.experiments.trained_relation_language_artifact import materialize_relation_language_members
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.source_record import SourceRecordRepository, register_source_record_table


class IncrementalRelationTrainingRuntime(TrainedRelationGraphRuntime):
    """训练期只读资源 owner；父表面来自已核验模型，新增表面来自本轮 Span。"""

    def __init__(self, database: Path, *, parent_core: Path, parent_surfaces: Path,
                 source_archive: Path | None = None):
        """限定 K 盘训练视图，不给生产查询新增后备路径或原文依赖。"""
        self._parent_core = parent_core.resolve(strict=True)
        self._parent_surfaces = parent_surfaces.resolve(strict=True)
        target = database.resolve(strict=True)
        if any(path.drive.upper() != "K:" for path in (
                self._parent_core, self._parent_surfaces, target)):
            raise ValueError("增量训练视图必须位于 K 盘")
        if target == self._parent_core:
            raise ValueError("增量训练不得写父模型")
        self._source_archive = source_archive.resolve(strict=True) if source_archive is not None else None
        if self._source_archive is not None and (
                self._source_archive.drive.upper() != "K:"
                or self._source_archive.is_relative_to(target.parent)
                or self._source_archive.is_relative_to(self._parent_core.parent)):
            raise ValueError("训练来源正文必须存放于模型目录外的显式 K 盘归档")
        super().__init__(target)

    def _restore_surface_facts(self):
        """逐项核验父命题及证据后，仅让共享 Span 恢复器处理新增命题。

        临时投影是只读训练视图，不改变任何 Core 记录。父知识不能以缺正文为由
        被重建或替换，父一等命题、角色、来源和证据必须在新库中逐字段相同。
        """
        with TrainedRelationGraphRuntime(self._parent_core,
                surface_artifact_database=self._parent_surfaces) as parent:
            old_facts, old_frames = parent.active_surface_facts(), parent.active_surface_frames()
            old_keys = {fact.proposition for fact in old_facts}
            current = {item.definition.proposition for item in self._snapshot.propositions}
            if not old_keys.issubset(current):
                raise TrainedRelationGraphError("增量 Core 丢失父 active 命题")
            for identity in sorted(old_keys, key=lambda item: item.stable_key()):
                if parent.generation_input(identity) != self.generation_input(identity):
                    raise TrainedRelationGraphError("增量 Core 的父命题、角色或证据漂移")
            self._parent_frame_contract = old_frames
        complete = self._snapshot
        try:
            self._snapshot = replace(complete, propositions=tuple(
                item for item in complete.propositions if item.definition.proposition not in old_keys))
            archive_path = getattr(self, "_source_archive", None)
            if archive_path is None:
                added = super()._restore_surface_facts()
            else:
                archive = SQLiteBackend(str(archive_path), read_only=True)
                try:
                    register_source_record_table(archive)
                    self._archived_sources = SourceRecordRepository(
                        archive, registry=self.context.scoped_identity_store.registry)
                    added = super()._restore_surface_facts()
                finally:
                    self._archived_sources = None
                    archive.close()
        finally:
            self._snapshot = complete
        return tuple(sorted((*old_facts, *added), key=lambda item: item.proposition.stable_key()))

    def _span_surface(self, span, spans, sources):
        """仅训练新增 Span 可读取显式外部归档；生产类没有该参数或路径。"""
        archived = getattr(self, "_archived_sources", None)
        return super()._span_surface(span, spans, archived if archived is not None else sources)

    def _restore_surface_frames(self):
        """原样继承父框架，只从新增事实的本次来源学习间隔和载体边缘。"""
        parent = self._parent_frame_contract
        old_keys = {item.proposition for item in parent}
        complete = self._surface_facts
        try:
            self._surface_facts = tuple(fact for fact in complete if fact.proposition not in old_keys)
            added = super()._restore_surface_frames()
            by_key = {fact.proposition: fact for fact in self._surface_facts}
            learned = tuple(learn_carrier_frame(by_key[frame.proposition], frame) for frame in added)
        finally:
            self._surface_facts = complete
        frames = tuple(sorted((*parent, *learned), key=lambda item: item.proposition.stable_key()))
        if (len(frames) != len(complete)
                or {frame.proposition for frame in frames} != {fact.proposition for fact in complete}):
            raise TrainedRelationGraphError("增量表面图的父/新框架未覆盖完整事实")
        return frames


def materialize_extended_relation_surfaces(database: Path, parent_core: Path,
                                         parent_surfaces: Path, output: Path, *,
                                         source_archive: Path | None = None):
    """保存训练后合并整数表面图，再由普通生产 runtime 冷恢复完整 active 集。

    该函数只用于候选训练，不组装发布包；父库只读，旧正文不回填，输出不得覆盖。
    """
    target = output.resolve()
    if target.drive.upper() != "K:" or target.exists() or not target.parent.is_dir():
        raise ValueError("增量表面产物必须使用独立全新 K 盘文件")
    with IncrementalRelationTrainingRuntime(database, parent_core=parent_core,
            parent_surfaces=parent_surfaces, source_archive=source_archive) as runtime:
        facts = tuple(replace(fact, evidence_surface='') for fact in runtime.active_surface_facts())
        frames, propositions = runtime.active_surface_frames(), runtime.active_propositions()
    result = materialize_relation_language_members(database, facts, frames, target,
        expected_propositions=propositions)
    return {**result, "source_body_restored": 0}
