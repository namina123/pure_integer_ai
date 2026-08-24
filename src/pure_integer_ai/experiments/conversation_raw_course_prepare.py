"""DLG-RAW-PERF-01：公开课程的内容锁定、非语义 preparation cache。

这里缓存的只是已验证 public course bytes 的确定性派生值。每次调用仍由
RAW-02 回读并校验当前课程 bytes；本模块不接触 backend、runtime、context 或任何
会话输入，因此删除 cache 不会改变回答、状态或整数结果码。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_generation_candidate_pack import (
    GenerationCandidatePack,
    build_generation_candidate_pack,
)
from pure_integer_ai.experiments.ph2_grounded_answer_compile import (
    GroundedAnswerTrainingBundle,
    compile_grounded_answer_training_records_from_payload,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    GroundedAnswerSurfaceModel,
    learn_grounded_answer_surface_model,
)


PREPARED_PUBLIC_COURSE_SCHEMA_V1 = 1
_PUBLIC_COURSE_PREPARATION_CACHE_MAX_ENTRIES = 8


# object-model: exception; interop=DLG-RAW-PERF-01
class ConversationRawCoursePreparationError(RuntimeError):
    """公开课程值不能在内容锁和确定性编译边界内准备。"""


def _sha256_u8(value: bytes, *, label: str) -> tuple[int, ...]:
    """以当前 SHA-256 implementation adapter 计算规范 32-byte 整数摘要。"""
    if type(value) is not bytes:
        raise TypeError(f"{label} 必须是 bytes")
    return tuple(hashlib.sha256(value).digest())


def _course_relative_path(value: str) -> str:
    """核验已由 RAW-02 路径门处理过的非空 logical source path。"""
    if not isinstance(value, str) or not value:
        raise ConversationRawCoursePreparationError(
            "DLG-RAW preparation course relative path 非法")
    return value


def _course_sha256(value: tuple[int, ...]) -> tuple[int, ...]:
    """核验来自 frame recipe 的固定 raw course SHA-256 u8 vector。"""
    if (not isinstance(value, tuple) or len(value) != 32
            or any(type(item) is not int or item < 0 or item > 255
                   for item in value)):
        raise ConversationRawCoursePreparationError(
            "DLG-RAW preparation course SHA-256 非法")
    return value


# object-model: value; representation=struct; interop=DLG-RAW-PERF-01
@dataclass(frozen=True, slots=True)
class PreparedPublicCourse:
    """一个内容锁课程的无 backend 派生值，不能替代原始 bytes 复核。"""

    course_relative_path: str
    course_raw_sha256: tuple[int, ...]
    bundle: GroundedAnswerTrainingBundle
    model: GroundedAnswerSurfaceModel
    pack: GenerationCandidatePack
    preparation_schema: int = PREPARED_PUBLIC_COURSE_SCHEMA_V1

    def __post_init__(self) -> None:
        """确认 cache entry 只由同一冻结课程和其确定性派生值组成。"""
        path = _course_relative_path(self.course_relative_path)
        digest = _course_sha256(self.course_raw_sha256)
        if self.preparation_schema != PREPARED_PUBLIC_COURSE_SCHEMA_V1:
            raise ConversationRawCoursePreparationError(
                "DLG-RAW preparation schema 未注册")
        if not isinstance(self.bundle, GroundedAnswerTrainingBundle):
            raise TypeError("DLG-RAW preparation bundle 类型错误")
        if not isinstance(self.model, GroundedAnswerSurfaceModel):
            raise TypeError("DLG-RAW preparation model 类型错误")
        if not isinstance(self.pack, GenerationCandidatePack):
            raise TypeError("DLG-RAW preparation pack 类型错误")
        if self.pack.model != self.model:
            raise ConversationRawCoursePreparationError(
                "DLG-RAW preparation pack/model 漂移")
        if self.pack.training_artifact_sha256 != bytes(digest).hex():
            raise ConversationRawCoursePreparationError(
                "DLG-RAW preparation pack/course SHA-256 漂移")
        object.__setattr__(self, "course_relative_path", path)
        object.__setattr__(self, "course_raw_sha256", digest)

    def matches(
            self,
            course_relative_path: str,
            course_raw_sha256: tuple[int, ...],
            ) -> bool:
        """只按路径、raw content lock 与冻结 schema 判断是否可复用。"""
        return (
            self.preparation_schema == PREPARED_PUBLIC_COURSE_SCHEMA_V1
            and self.course_relative_path == course_relative_path
            and self.course_raw_sha256 == course_raw_sha256
        )


# object-model: resource-owner; interop=DLG-RAW-PERF-01; semantic-state=none
class PublicCoursePreparationCache:
    """进程内可删除、有界 cache owner；不保存语义状态或 run-local object。"""

    __slots__ = ("_entries", "_hit_count", "_miss_count")

    def __init__(self) -> None:
        """创建空 cache；其内容不属于 RAW-02/RAW-04 canonical state。"""
        self._entries: tuple[PreparedPublicCourse, ...] = ()
        self._hit_count = 0
        self._miss_count = 0

    @property
    def entry_count(self) -> int:
        """返回当前物理 cache entry 数，仅用于有界性能诊断。"""
        return len(self._entries)

    @property
    def hit_count(self) -> int:
        """返回当前进程中的非语义 value-cache 命中数。"""
        return self._hit_count

    @property
    def miss_count(self) -> int:
        """返回当前进程中的确定性课程编译次数。"""
        return self._miss_count

    def get_or_prepare(
            self,
            payload: bytes,
            *,
            course_relative_path: str,
            course_raw_sha256: tuple[int, ...],
            ) -> PreparedPublicCourse:
        """先复核本轮 bytes，再命中或确定性重建仅含 value 的课程 entry。"""
        path = _course_relative_path(course_relative_path)
        digest = _course_sha256(course_raw_sha256)
        if _sha256_u8(payload, label="DLG-RAW preparation course payload") != digest:
            raise ConversationRawCoursePreparationError(
                "DLG-RAW preparation course SHA-256 漂移")
        for entry in self._entries:
            if entry.matches(path, digest):
                self._hit_count += 1
                return entry
        prepared = _prepare_public_course(payload, path, digest)
        self._entries = tuple(sorted(
            (*self._entries, prepared),
            key=lambda item: (item.course_relative_path, item.course_raw_sha256),
        ))[:_PUBLIC_COURSE_PREPARATION_CACHE_MAX_ENTRIES]
        self._miss_count += 1
        return prepared

    def clear(self) -> None:
        """删除全部可再生 value entry；历史诊断计数保留且不影响语义。"""
        self._entries = ()


def _prepare_public_course(
        payload: bytes,
        course_relative_path: str,
        course_raw_sha256: tuple[int, ...],
        ) -> PreparedPublicCourse:
    """从本轮已验证 source bytes 构造 immutable bundle、model 和 candidate pack。"""
    bundle = compile_grounded_answer_training_records_from_payload(
        payload,
        source_relative_path=course_relative_path,
    )
    model, _report = learn_grounded_answer_surface_model(bundle)
    pack = build_generation_candidate_pack(
        model,
        bytes(course_raw_sha256).hex(),
    )
    return PreparedPublicCourse(
        course_relative_path,
        course_raw_sha256,
        bundle,
        model,
        pack,
    )


def prepare_public_course(
        payload: bytes,
        *,
        course_relative_path: str,
        course_raw_sha256: tuple[int, ...],
        cache: PublicCoursePreparationCache | None = None,
        ) -> PreparedPublicCourse:
    """准备一轮 RAW-02 所需课程值；无 cache 时仍执行同一确定性路径。"""
    if cache is not None and not isinstance(cache, PublicCoursePreparationCache):
        raise TypeError("DLG-RAW preparation cache 类型错误")
    path = _course_relative_path(course_relative_path)
    digest = _course_sha256(course_raw_sha256)
    if _sha256_u8(payload, label="DLG-RAW preparation course payload") != digest:
        raise ConversationRawCoursePreparationError(
            "DLG-RAW preparation course SHA-256 漂移")
    if cache is not None:
        return cache.get_or_prepare(
            payload,
            course_relative_path=path,
            course_raw_sha256=digest,
        )
    return _prepare_public_course(payload, path, digest)


__all__ = [
    "PREPARED_PUBLIC_COURSE_SCHEMA_V1",
    "ConversationRawCoursePreparationError",
    "PreparedPublicCourse",
    "PublicCoursePreparationCache",
    "prepare_public_course",
]
