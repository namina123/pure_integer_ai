"""由训练图成员派生 Memory 候选；内容关联与来源内结构关联分别编码。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import OBJECT_EVENT
from pure_integer_ai.cognition.understanding.artifact_query_input import (
    ARTIFACT_MEMORY_CATEGORY,
    artifact_memory_candidate_routes,
)
from pure_integer_ai.cognition.understanding.query_open_roles import OpenRelationCandidate
from pure_integer_ai.cognition.understanding.query_input_structure import (
    InputRelationCandidate,
    PROJECTION_FILLER,
    PROJECTION_PREDICATE,
    QueryInputStructure,
    STRUCTURE_CLOSED,
)

MEMORY_CANDIDATE_VERSION = 91502
MEMORY_GENERIC_CANDIDATE_VERSION = 91503
MEMORY_TOPIC_VERSION = 91509
MEMORY_TOPIC_PROOF_VERSION = 91510
MEMORY_CONTEXT_COMPETITION_VERSION = 91511
MEMORY_CATEGORY_TIME = 6
MEMORY_CATEGORY_DISCOURSE = 7
MEMORY_CATEGORY_OPEN_RELATION = 8
MEMORY_CATEGORY_RESPONSE = 9
MEMORY_CATEGORY_ARTIFACT = ARTIFACT_MEMORY_CATEGORY
MEMORY_CATEGORY_GRAPH_INPUT = 11


def frame_key(tag: int, *parts: tuple[int, ...]) -> tuple[int, ...]:
    """拼接带长度边界的严格整数记录，不以哈希替代成员身份。"""
    if type(tag) is not int or tag <= 0:
        raise ValueError("Memory key tag 必须是正严格整数")
    result = [tag]
    for part in parts:
        if (type(part) is not tuple
                or any(type(value) is not int or value < 0 for value in part)):
            raise ValueError("Memory key part 必须是非负严格整数 tuple")
        result.extend((len(part), *part))
    return tuple(result)


def _parts(key: tuple[int, ...], tag: int) -> tuple[tuple[int, ...], ...]:
    """严格解码完整记录，拒绝负长度、截断、布尔值和未消费尾部。"""
    if (type(key) is not tuple or not key or key[0] != tag
            or any(type(value) is not int or value < 0 for value in key)):
        raise ValueError("Memory 整数记录类型或版本非法")
    result = []
    cursor = 1
    while cursor < len(key):
        end = cursor + 1 + key[cursor]
        if end > len(key):
            raise ValueError("Memory 整数记录被截断")
        result.append(key[cursor + 1:end])
        cursor = end
    return tuple(result)


def memory_candidate_category(key: tuple[int, ...]) -> int:
    """恢复候选的开放结构类别，不把类别解释为事实成立。"""
    # A generic candidate is an integer-only observation envelope used when
    # no trained semantic topology covers the current input.  It is never a
    # Core fact route; its sole purpose is to keep the input in O/H/E and the
    # same-turn Dialogue frontier for later structural generation.
    if key and key[0] == MEMORY_GENERIC_CANDIDATE_VERSION:
        parts = _parts(key, MEMORY_GENERIC_CANDIDATE_VERSION)
        if len(parts) != 2 or len(parts[0]) != 1 or parts[0][0] not in range(1, 12):
            raise ValueError("Memory generic candidate category 非法")
        return parts[0][0]
    parts = _parts(key, MEMORY_CANDIDATE_VERSION)
    if not parts or len(parts[0]) != 1 or parts[0][0] not in range(1, 12):
        raise ValueError("Memory input candidate category 非法")
    return parts[0][0]


@dataclass(frozen=True, slots=True)
class MemoryTopicCandidate:
    """话题假设的完整图成员；同拓扑不合并内容，实际来源由 O/H/E 绑定。"""

    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    structure_key: tuple[int, ...]
    carrier_key: tuple[int, ...]
    members: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        for key in (self.proposition_key, self.predicate_key,
                    self.structure_key, self.carrier_key):
            if not key:
                raise ValueError("话题必须保留命题、谓词、结构和载体")
            frame_key(1, key)
        if type(self.members) is not tuple or self.members != tuple(sorted(set(self.members))):
            raise ValueError("话题成员必须排序去重")
        for member in self.members:
            fields = _parts(member, 1)
            if (len(fields) != 4 or len(fields[0]) != 2
                    or not fields[1] or fields[0][0] != fields[1][0]
                    or not fields[2] or not fields[3]):
                raise ValueError("话题成员缺少节点、角色、序或训练来源")

    def stable_key(self) -> tuple[int, ...]:
        """编码全部成员与图引用；空成员显式表示只有句法、尚无内容锚点。"""
        return frame_key(MEMORY_TOPIC_VERSION, self.proposition_key,
                         self.predicate_key, self.structure_key, self.carrier_key,
                         *self.members)

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> MemoryTopicCandidate:
        """从规范整数记录恢复话题对象，不依赖来源正文重新投影。"""
        fields = _parts(key, MEMORY_TOPIC_VERSION)
        if len(fields) < 4:
            raise ValueError("Memory 话题对象被截断")
        return cls(*fields[:4], fields[4:])

    def content_routes(self) -> tuple[tuple[int, ...], ...]:
        """只有实际出现的 filler 才能建立话题的共同节点/命题关联。"""
        if not self.members:
            return ()
        routes = {frame_key(3, self.proposition_key)}
        for member in self.members:
            routes.add(frame_key(2, _parts(member, 1)[1]))
        return tuple(sorted(routes))


def memory_topic_candidate(key: tuple[int, ...]) -> MemoryTopicCandidate | None:
    """识别有本体成员的新话题；旧拓扑仍可恢复但不是内容身份。"""
    if key and key[0] == MEMORY_GENERIC_CANDIDATE_VERSION:
        return None
    category = memory_candidate_category(key)
    parts = _parts(key, MEMORY_CANDIDATE_VERSION)
    if category in {MEMORY_CATEGORY_TIME, MEMORY_CATEGORY_DISCOURSE} and len(parts) == 2:
        return MemoryTopicCandidate.from_stable_key(parts[1])
    return None


def memory_candidate_routes(key: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """内容路由只由显式 filler/命题关联产生，句法相同不能激活旧话题。"""
    if key and key[0] == MEMORY_GENERIC_CANDIDATE_VERSION:
        return ()
    category = memory_candidate_category(key)
    parts = _parts(key, MEMORY_CANDIDATE_VERSION)
    routes = {frame_key(1, key)}
    if category == MEMORY_CATEGORY_RESPONSE:
        if len(parts) != 3 or not parts[1] or not parts[2]:
            raise ValueError("Memory 回应候选必须保留输出证明引用和实际动作")
        # 生成动作不是新的事实来源，不能借语言框架激活 Core 例值。
        return tuple(sorted(routes))
    if category == MEMORY_CATEGORY_ARTIFACT:
        # 载体输入按完整 projection、exact carrier 或显式 reference target 路由；
        # subtype 的格式由 artifact 合同独占解码，不能在通用 Memory 层重新把
        # category 10 缩窄回 projection-only。这里不读取 raw units、正文或字符
        # posting，也不把任意相同 carrier 当作同一概念。
        return artifact_memory_candidate_routes(key)
    if category == MEMORY_CATEGORY_GRAPH_INPUT:
        if len(parts) != 2 or not parts[1]:
            raise ValueError("Memory graph-input 候选缺少完整对象键")
        return tuple(sorted((*routes, frame_key(2, parts[1]))))
    if category == MEMORY_CATEGORY_OPEN_RELATION:
        if len(parts) != 2:
            raise ValueError("Memory 开放关系候选缺少完整解析")
        OpenRelationCandidate.from_stable_key(parts[1])
        # 仅同一完整角色解析可路由。框架相同或未知区间字面相同不等于同指，
        # 不借用训练示例 proposition/filler 召回其他话题。
        return tuple(sorted(routes))
    if category in {1, 2, 3, 4}:
        if len(parts) != 6 or len(parts[1]) != 3:
            raise ValueError("Memory 语义候选缺少节点/命题/角色结构")
        if parts[1][1] != PROJECTION_FILLER:
            return ()
        routes.update((frame_key(2, parts[2]), frame_key(3, parts[3])))
    elif category == 5:
        if len(parts) != 6 or len(parts[5]) != 7:
            raise ValueError("Memory 关系候选缺少命题结构")
        if parts[5][2] == 0:
            return ()
        routes.add(frame_key(3, parts[1]))
    else:
        topic = memory_topic_candidate(key)
        if topic is None:
            if len(parts) != 3:
                raise ValueError("Memory 旧时间/话题候选结构不完整")
            return ()
        content_routes = topic.content_routes()
        if not content_routes:
            return ()
        routes.update(content_routes)
    return tuple(sorted(routes))


def memory_competition_identity(candidate_key: tuple[int, ...],
                                competition_key: tuple[int, ...],
                                observation_key: tuple[int, ...]) -> tuple[int, ...]:
    """无内容锚点的结构竞争留在原 Observation，不能变成跨话题事实冲突。"""
    if memory_candidate_routes(candidate_key):
        return competition_key
    return frame_key(MEMORY_CONTEXT_COMPETITION_VERSION,
                     competition_key, observation_key)


def memory_input_candidate_keys(input_structure: QueryInputStructure,
                                *, session_id: int = 0,
                                speaker_kind: int = 1,
                                ) -> tuple[tuple[int, ...], ...]:
    """保留输入全部结构候选，并把话题/时间绑定到实际命中的图成员。"""
    if not isinstance(input_structure, QueryInputStructure):
        raise TypeError("input_structure 必须是 QueryInputStructure")
    if (type(session_id) is not int or session_id < 0
            or type(speaker_kind) is not int or speaker_kind <= 0):
        raise ValueError("generic Memory candidate session/speaker 必须是整数")
    result = set()
    result.update(
        memory_graph_input_candidate_key(key)
        for key in input_structure.graph_object_keys)
    # Open-role segmentations are ephemeral QueryState hypotheses. Persisting
    # every segmentation as an O/H/E row turns one unknown sentence into
    # hundreds of competing Memory records and replays no additional graph
    # knowledge on the next turn.  The source-local generic Observation below
    # preserves the interaction; closed relation candidates are handled by
    # the typed routes that follow.
    topic_members: dict[tuple[tuple[int, ...], tuple[int, ...]], set[tuple[int, ...]]] = {}
    # Standalone semantic members are open projections, not a closed Memory
    # concept route.  Routing every one-codepoint member here multiplies a
    # query into hundreds of SQLite route probes and lets accidental surface
    # overlap steer cross-turn recall.  Only members belonging to an exact
    # relation candidate may enter O/H/E; an input with no such relation still
    # gets the generic source-local Observation below.
    relation_propositions = {
        item.proposition_key for item in input_structure.relation_candidates
        if (item.state == STRUCTURE_CLOSED
            and item.required_count >= 2
            and item.required_count == item.support_count
            and item.predicate_coverage == 1)
    }
    # Open-role parses are first-class Memory hypotheses.  They retain the
    # complete frame, source-scoped spans and UNKNOWN stance so the same
    # QueryState can feed role generation and later cross-turn reference
    # resolution.  Do not collapse them into generic candidates or infer a
    # Core claim from their dynamic fillers.
    for candidate in input_structure.open_relations:
        result.add(frame_key(
            MEMORY_CANDIDATE_VERSION,
            (MEMORY_CATEGORY_OPEN_RELATION,),
            candidate.stable_key(),
        ))
    for item in input_structure.semantic_candidates:
        if item.proposition_key not in relation_propositions:
            continue
        semantic_parts = (
            (item.object_kind, item.projection_kind, item.member_ordinal),
            item.candidate_key, item.proposition_key, item.predicate_key, item.role_key)
        categories = {3, 4}
        if item.projection_kind == PROJECTION_FILLER:
            categories.add(1)
            topic_members.setdefault((item.proposition_key, item.source_ref), set()).add(
                frame_key(1, (item.object_kind, item.member_ordinal),
                          item.candidate_key, item.role_key, item.source_ref))
        if item.projection_kind == PROJECTION_PREDICATE or item.object_kind == OBJECT_EVENT:
            categories.add(2)
        for category in categories:
            result.add(frame_key(MEMORY_CANDIDATE_VERSION, (category,), *semantic_parts))
    for item in input_structure.relation_candidates:
        if not (item.state == STRUCTURE_CLOSED
                and item.required_count >= 2
                and item.required_count == item.support_count
                and item.predicate_coverage == 1):
            continue
        result.add(memory_relation_candidate_key(item))
        topic = MemoryTopicCandidate(
            item.proposition_key, item.predicate_key, item.structure_key, item.carrier_key,
            tuple(sorted(topic_members.get((item.proposition_key, item.source_ref), ()))))
        for category in (MEMORY_CATEGORY_TIME, MEMORY_CATEGORY_DISCOURSE):
            result.add(frame_key(MEMORY_CANDIDATE_VERSION, (category,), topic.stable_key()))
    # Preserve a source-local structural observation for every input, even
    # when a typed relation also matched.  Generic candidates deliberately
    # have no content routes, so they cannot assert or override the typed
    # relation; they keep the Companion/Dialogue O/H/E root available in the
    # same QueryState for graph-conditioned non-claim generation.
    for category in (MEMORY_CATEGORY_TIME, MEMORY_CATEGORY_DISCOURSE):
        result.add(frame_key(
            MEMORY_GENERIC_CANDIDATE_VERSION,
            (category,), input_structure.source_ref,
        ))
    return tuple(sorted(result))


def memory_graph_input_candidate_key(
        graph_object_key: tuple[int, ...],
        ) -> tuple[int, ...]:
    """Return the exact Memory hypothesis identity for one graph input."""
    if (type(graph_object_key) is not tuple or not graph_object_key
            or any(type(item) is not int or item < 0
                   for item in graph_object_key)):
        raise ValueError("graph_object_key 必须是非空非负严格整数 tuple")
    return frame_key(
        MEMORY_CANDIDATE_VERSION,
        (MEMORY_CATEGORY_GRAPH_INPUT,),
        graph_object_key,
    )


def memory_graph_input_object_key(
        candidate_key: tuple[int, ...],
        ) -> tuple[int, ...] | None:
    """Recover an exact graph object from a graph-input Memory candidate."""
    if (not candidate_key or candidate_key[0] != MEMORY_CANDIDATE_VERSION
            or memory_candidate_category(candidate_key)
            != MEMORY_CATEGORY_GRAPH_INPUT):
        return None
    parts = _parts(candidate_key, MEMORY_CANDIDATE_VERSION)
    if len(parts) != 2 or not parts[1]:
        raise ValueError("Memory graph-input candidate 被截断")
    return parts[1]


def memory_relation_candidate_key(
        candidate: InputRelationCandidate,
        ) -> tuple[int, ...]:
    """返回关系候选的唯一 Memory Hypothesis 路由键。"""
    if not isinstance(candidate, InputRelationCandidate):
        raise TypeError("candidate 必须是 InputRelationCandidate")
    return frame_key(
        MEMORY_CANDIDATE_VERSION, (5,), candidate.proposition_key,
        candidate.predicate_key, candidate.structure_key, candidate.carrier_key,
        (candidate.required_count, candidate.support_count,
         candidate.role_coverage, candidate.predicate_coverage,
         candidate.required_open, candidate.state, candidate.conflict_count))


def memory_topic_input_proof(candidate_key: tuple[int, ...],
                             input_structure: QueryInputStructure) -> tuple[int, ...]:
    """把话题成员对应的原始输入 Span/关联候选完整写入 Evidence，而非存正文答案。"""
    if (candidate_key and candidate_key[0] == MEMORY_CANDIDATE_VERSION
            and memory_candidate_category(candidate_key)
            == MEMORY_CATEGORY_GRAPH_INPUT):
        graph_key = memory_graph_input_object_key(candidate_key)
        if graph_key not in input_structure.graph_object_keys:
            raise ValueError("graph-input Evidence 不属于同一次输入")
        return frame_key(
            MEMORY_TOPIC_PROOF_VERSION,
            input_structure.source_ref,
            graph_key,
        )
    if (candidate_key and candidate_key[0] == MEMORY_CANDIDATE_VERSION
            and memory_candidate_category(candidate_key) == MEMORY_CATEGORY_OPEN_RELATION):
        fields = _parts(candidate_key, MEMORY_CANDIDATE_VERSION)
        candidate = OpenRelationCandidate.from_stable_key(fields[1])
        if candidate not in input_structure.open_relations:
            raise ValueError("开放关系证据必须来自同一次输入投影")
        return frame_key(MEMORY_TOPIC_PROOF_VERSION, input_structure.source_ref,
                         candidate.stable_key())
    topic = memory_topic_candidate(candidate_key)
    if topic is None:
        return ()
    projections = tuple(item.stable_key() for item in input_structure.semantic_candidates
                        if item.proposition_key == topic.proposition_key)
    relations = tuple(item.stable_key() for item in input_structure.relation_candidates
                      if item.proposition_key == topic.proposition_key
                      and item.structure_key == topic.structure_key
                      and item.carrier_key == topic.carrier_key)
    return frame_key(MEMORY_TOPIC_PROOF_VERSION, input_structure.source_ref,
                     topic.stable_key(), frame_key(1, *projections), frame_key(2, *relations))
