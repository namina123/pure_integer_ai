"""A-05 来源准入纯领域契约的专项测试。

PW-00 已覆盖 SourceAdmissionRuntime 在完整 dry-run 里的 happy-path 集成；本文件
补齐 ``SourceTrustRequest`` / ``SourceTrustAssessment`` 自身的入口校验、稳定键确定
性与内容敏感性，以及裁决不变量。纯读、无图、无 ctx，不依赖 M-05/M-10 装配。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONCEPT,
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
    SourceRef,
    TypedRef,
    VersionBundle,
    CorpusVersion,
    CurriculumVersion,
    ParserVersion,
    PrimitiveVersion,
)
from pure_integer_ai.cognition.shared.source_trust import (
    SOURCE_ADMISSION_ACCEPTED,
    SOURCE_ADMISSION_REJECTED,
    SOURCE_TRUST_PROTOCOL_VERSION,
    SourceTrustAssessment,
    SourceTrustRequest,
)
from pure_integer_ai.crosscut.guards.int_blocker import IntViolation


def _versions() -> VersionBundle:
    """固定版本束，保证稳定键不随测试版本漂移。"""
    return VersionBundle(
        CorpusVersion(3), ParserVersion(2), PrimitiveVersion(5), CurriculumVersion(4))


def _source(*, source_id: int = 9) -> SourceRef:
    """构造同一文档的来源引用。"""
    return SourceRef(71, source_id, 0, GLOBAL_OWNER_SCOPE, _versions())


def _route(value: int = 301) -> ObjectIdentity:
    """构造一等入口指令身份。"""
    return ObjectIdentity(
        OBJECT_MINIMAL_INSTRUCTION, (value,), GLOBAL_OWNER_SCOPE, _versions())


def _ref(local_id: int) -> TypedRef:
    """构造图内概念引用，按 local_id 区分以得到唯一稳定键。"""
    return TypedRef(OBJECT_CONCEPT, 1, local_id, GLOBAL_OWNER_SCOPE, _versions())


def _sorted_refs(*local_ids: int) -> tuple[TypedRef, ...]:
    """按稳定键唯一排序的引用元组，满足裁决不变量。"""
    refs = tuple(_ref(lid) for lid in local_ids)
    return tuple(sorted(refs, key=lambda item: item.stable_key()))


# ---------------------------------------------------------------------------
# SourceTrustRequest 入口校验
# ---------------------------------------------------------------------------

def test_request_valid_construction_and_stable_key_shape():
    """合法请求返回非空严格整数稳定键，且以协议版本打头。"""
    request = SourceTrustRequest(
        _route(), _source(), "正文", "license-A", 7, (401, 2))
    key = request.stable_key()
    assert isinstance(key, tuple) and key
    assert all(type(item) is int for item in key)
    assert key[0] == SOURCE_TRUST_PROTOCOL_VERSION


def test_request_rejects_non_identity_route():
    """route_kind 必须是一等 ObjectIdentity。"""
    with pytest.raises(TypeError, match="route_kind"):
        SourceTrustRequest("route", _source(), "正文", "lic", 1)  # type: ignore[arg-type]


def test_request_rejects_non_source_ref():
    """source 必须是 SourceRef，不接受概念引用或裸元组。"""
    with pytest.raises(TypeError, match="source"):
        SourceTrustRequest(_route(), _ref(1), "正文", "lic", 1)  # type: ignore[arg-type]


def test_request_rejects_empty_or_non_string_license():
    """许可声明必须是非空字符串。"""
    with pytest.raises(ValueError, match="license_id"):
        SourceTrustRequest(_route(), _source(), "正文", "", 1)
    with pytest.raises(ValueError, match="license_id"):
        SourceTrustRequest(_route(), _source(), "正文", 0, 1)  # type: ignore[arg-type]


def test_request_rejects_non_positive_or_weak_int_batch_id():
    """batch_id 必须是正严格整数；bool/float/零/负数全部拒绝。

    assert_int 放行 bool（逻辑标志），但 SourceTrustRequest 的 batch_id 另有
    ``type() is int`` 严格守卫挡掉 bool；float 由 int_blocker 抛 IntViolation。
    """
    for bad in (0, -1, True, 1.0):  # type: ignore[arg-type]
        with pytest.raises((ValueError, IntViolation)):
            SourceTrustRequest(_route(), _source(), "正文", "lic", bad)


def test_request_rejects_non_tuple_or_weak_int_trace():
    """trace 必须是严格整数元组；列表入参抛 TypeError，弱整数项抛 ValueError。"""
    with pytest.raises(TypeError, match="trace"):
        SourceTrustRequest(_route(), _source(), "正文", "lic", 1, [1, 2])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="trace"):
        SourceTrustRequest(_route(), _source(), "正文", "lic", 1, trace=(1, True))  # type: ignore[arg-type]


def test_request_stable_key_is_deterministic():
    """同一请求两次求键必须逐位相等。"""
    request = SourceTrustRequest(_route(), _source(), "正文", "lic", 3, (9,))
    assert request.stable_key() == request.stable_key()


def test_request_stable_key_is_content_sensitive():
    """原文、许可、批次、入口、来源任一改变，稳定键都改变。"""
    base = SourceTrustRequest(_route(), _source(), "正文", "lic", 3)
    altered = [
        SourceTrustRequest(_route(), _source(), "改文", "lic", 3),
        SourceTrustRequest(_route(), _source(), "正文", "lic-B", 3),
        SourceTrustRequest(_route(), _source(), "正文", "lic", 4),
        SourceTrustRequest(_route(302), _source(), "正文", "lic", 3),
        SourceTrustRequest(_route(), _source(source_id=10), "正文", "lic", 3),
    ]
    base_key = base.stable_key()
    assert all(other.stable_key() != base_key for other in altered)


def test_request_stable_key_does_not_grow_with_raw_text_length():
    """稳定键对原文走指纹，不应随原文变长而无界增长。"""
    short = SourceTrustRequest(_route(), _source(), "短", "lic", 1).stable_key()
    long = SourceTrustRequest(
        _route(), _source(), "正文" * 1000, "lic", 1).stable_key()
    assert len(short) == len(long)


# ---------------------------------------------------------------------------
# SourceTrustAssessment 不变量
# ---------------------------------------------------------------------------

def _assessment(
        *, decision: int = SOURCE_ADMISSION_ACCEPTED,
        reason_ids: tuple[int, ...] = (500,),
        anomaly_ids: tuple[int, ...] = (),
        ) -> SourceTrustAssessment:
    """构造合法裁决；reason_refs 必须非空且唯一排序。"""
    return SourceTrustAssessment(
        (1, 2, 3),
        (4, 5, 6),
        decision,
        (7, 8),
        (9, 10),
        _ref(100),
        _ref(200),
        _ref(300),
        _sorted_refs(*reason_ids),
        _sorted_refs(*anomaly_ids) if anomaly_ids else (),
        (11, 12),
    )


def test_assessment_accepted_property():
    """accepted 仅在 decision==ACCEPTED 时为真。"""
    assert _assessment(decision=SOURCE_ADMISSION_ACCEPTED).accepted is True
    assert _assessment(
        decision=SOURCE_ADMISSION_REJECTED,
        reason_ids=(500,),
        anomaly_ids=(700,),
    ).accepted is False


def test_assessment_rejects_unregistered_decision():
    """decision 必须是已注册的准入状态；未知值拒绝。"""
    with pytest.raises(ValueError, match="decision"):
        _assessment(decision=3)


def test_assessment_rejects_empty_or_weak_request_key():
    """request_key / policy_state_key 必须是非空严格整数元组。"""
    with pytest.raises(ValueError, match="request_key"):
        SourceTrustAssessment(
            (), (4, 5), SOURCE_ADMISSION_ACCEPTED, (7, 8), (9, 10),
            _ref(100), _ref(200), _ref(300), _sorted_refs(500), (), (11,))
    with pytest.raises((ValueError, TypeError), match="policy_state_key"):
        SourceTrustAssessment(
            (1, 2), (4, True), SOURCE_ADMISSION_ACCEPTED, (7, 8), (9, 10),
            _ref(100), _ref(200), _ref(300), _sorted_refs(500), (), (11,))


def test_assessment_requires_non_empty_reason_refs():
    """裁决必须至少提供一个图内 reason 依据。"""
    with pytest.raises(ValueError, match="reason"):
        SourceTrustAssessment(
            (1, 2), (4, 5), SOURCE_ADMISSION_ACCEPTED, (7, 8), (9, 10),
            _ref(100), _ref(200), _ref(300), (), (), (11,))


def test_assessment_rejects_accepted_with_blocking_anomaly():
    """已接受来源不得携带阻断异常。"""
    with pytest.raises(ValueError, match="阻断异常"):
        SourceTrustAssessment(
            (1, 2), (4, 5), SOURCE_ADMISSION_ACCEPTED, (7, 8), (9, 10),
            _ref(100), _ref(200), _ref(300),
            _sorted_refs(500), _sorted_refs(700), (11,))


def test_assessment_rejects_unsorted_or_duplicate_reason_refs():
    """reason_refs 必须按稳定键唯一排序。"""
    duplicate = (_ref(500), _ref(500))
    with pytest.raises(ValueError, match="reason_refs"):
        SourceTrustAssessment(
            (1, 2), (4, 5), SOURCE_ADMISSION_ACCEPTED, (7, 8), (9, 10),
            _ref(100), _ref(200), _ref(300), duplicate, (), (11,))
    unsorted = (_ref(600), _ref(500))
    with pytest.raises(ValueError, match="reason_refs"):
        SourceTrustAssessment(
            (1, 2), (4, 5), SOURCE_ADMISSION_ACCEPTED, (7, 8), (9, 10),
            _ref(100), _ref(200), _ref(300), unsorted, (), (11,))


def test_assessment_stable_key_is_deterministic_and_decision_sensitive():
    """稳定键确定，且接受/拒绝两种裁决给出不同键。"""
    accepted = _assessment(decision=SOURCE_ADMISSION_ACCEPTED)
    rejected = _assessment(
        decision=SOURCE_ADMISSION_REJECTED,
        reason_ids=(500,),
        anomaly_ids=(700,),
    )
    assert accepted.stable_key() == accepted.stable_key()
    assert accepted.stable_key() != rejected.stable_key()
    assert accepted.stable_key()[0] == SOURCE_TRUST_PROTOCOL_VERSION


def test_assessment_stable_key_round_trip_preserves_source_and_all_fields():
    """完整键须无损恢复裁决，request identity 仍能回读原 SourceRef。"""
    assessment = _assessment(decision=SOURCE_ADMISSION_ACCEPTED)
    restored = SourceTrustAssessment.from_stable_key(assessment.stable_key())
    assert restored == assessment

    request = SourceTrustRequest(_route(), _source(), "正文", "lic", 3, (9,))
    assert SourceTrustRequest.source_from_stable_key(
        request.stable_key()) == request.source
