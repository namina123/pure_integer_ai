"""用完整成对训练见证构造WH请求前缀，不提供词表、事实或默认角色。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleFrame, pack_record, unpack_record
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    QUESTION_SOURCE_REF, SLOT_MODE_PLAIN, QuestionCompositionBudgetExceeded,
    TrainedRelationQuestionFrame,
)

QUESTION_PREFIX_RULE = 92048
PREFIXED_SLOT_FRAME = 92049
QUESTION_PREFIX_PAIR_LIST = 92050
QUESTION_PREFIX_PAIR = 92051


def _contract(frame):
    """前缀见证两端必须保留除首间隔外的全部原始语法合同。"""
    return (frame.operation_key, frame.predicate_key, frame.roles,
            frame.answer_role_key, frame.wh_values, frame.slot_mode, frame.gaps[1:])


def _pair_key(pair):
    """保留两份完整原帧身份，不用签名或来源尾号替代成员。"""
    return tuple(frame.stable_key() for frame in pair)


@dataclass(frozen=True, slots=True)
class QuestionPrefixRule:
    """同一谓词操作下完整及WH成对见证组成的一等整数语法构造。"""

    prefix_values: tuple[int, ...]
    complete_pairs: tuple[tuple[TrainedRelationQuestionFrame, TrainedRelationQuestionFrame], ...]
    slot_pairs: tuple[tuple[TrainedRelationQuestionFrame, TrainedRelationQuestionFrame], ...]

    def __post_init__(self):
        """同时要求两类见证，精确保留角色、模式及唯一首间隔差分。"""
        if (type(self.prefix_values) is not tuple or not self.prefix_values
                or any(type(value) is not int or not 0 <= value <= 0x10ffff for value in self.prefix_values)):
            raise ValueError('question prefix must be a strict integer sequence')
        expected = None
        for pairs, slot in ((self.complete_pairs, False), (self.slot_pairs, True)):
            if type(pairs) is not tuple or not pairs:
                raise ValueError('question prefix requires complete and WH witness pairs')
            if any(type(pair) is not tuple or len(pair) != 2
                   or any(type(frame) is not TrainedRelationQuestionFrame for frame in pair) for pair in pairs):
                raise ValueError('question prefix witnesses must be original trained frames')
            keys = tuple(_pair_key(pair) for pair in pairs)
            if keys != tuple(sorted(set(keys))):
                raise ValueError('question prefix pairs must be complete and canonical')
            for base, extended in pairs:
                scope = (base.operation_key, base.predicate_key)
                if expected is None:
                    expected = scope
                if (scope != expected or _contract(base) != _contract(extended)
                        or base.is_slot_question != slot or base.slot_mode != SLOT_MODE_PLAIN
                        or extended.gaps[0] != self.prefix_values + base.gaps[0]):
                    raise ValueError('question prefix witness contract drifted')

    @property
    def operation_key(self):
        """操作范围来自完整原帧，不由前缀表面推断。"""
        return self.complete_pairs[0][0].operation_key

    @property
    def predicate_key(self):
        """构造局限于同一受训谓词，不跨关系族借用语言授权。"""
        return self.complete_pairs[0][0].predicate_key

    @property
    def source_frames(self):
        """恢复所有成对成员供挂载模型资格逐项校验。"""
        return tuple(sorted({frame for pair in (*self.complete_pairs, *self.slot_pairs)
                             for frame in pair}, key=lambda frame: frame.stable_key()))

    def stable_key(self):
        """完整编码前缀及两类全部来源对，不省略成对关系。"""
        def group(pairs):
            """来源对使用显式有序整数记录，可跨语言逐项恢复。"""
            return pack_record(QUESTION_PREFIX_PAIR_LIST, *(
                pack_record(QUESTION_PREFIX_PAIR, *_pair_key(pair)) for pair in pairs))
        return pack_record(QUESTION_PREFIX_RULE, self.prefix_values,
                           group(self.complete_pairs), group(self.slot_pairs))

    @classmethod
    def from_stable_key(cls, key):
        """独立恢复全部原帧并重验双见证及唯一前缀差分。"""
        fields = unpack_record(key, QUESTION_PREFIX_RULE)
        if len(fields) != 3:
            raise ValueError('question prefix rule field count differs')
        def group(value):
            """完整恢复每对的两个训练对象，构造器继续核验所有字段。"""
            return tuple(tuple(TrainedRelationQuestionFrame.from_stable_key(part)
                               for part in unpack_record(pair, QUESTION_PREFIX_PAIR))
                         for pair in unpack_record(value, QUESTION_PREFIX_PAIR_LIST))
        return cls(fields[0], group(fields[1]), group(fields[2]))


@dataclass(frozen=True, slots=True)
class PrefixedSlotQuestionFrame:
    """原始WH帧与完整前缀见证的构造，不修改原答案、角色或后续间隔。"""

    base: TrainedRelationQuestionFrame
    rule: QuestionPrefixRule

    def __post_init__(self):
        """只允许同操作谓词的原始plain WH帧，拒绝指代及递归叠加。"""
        if (type(self.base) is not TrainedRelationQuestionFrame
                or type(self.rule) is not QuestionPrefixRule
                or not self.base.is_slot_question or self.base.slot_mode != SLOT_MODE_PLAIN
                or (self.base.operation_key, self.base.predicate_key)
                != (self.rule.operation_key, self.rule.predicate_key)):
            raise ValueError('prefixed WH base is outside the witnessed contract')

    @property
    def operation_key(self):
        """操作身份由base完整继承。"""
        return self.base.operation_key

    @property
    def predicate_key(self):
        """谓词身份由base完整继承。"""
        return self.base.predicate_key

    @property
    def roles(self):
        """观察角色及原顺序保持，不以其它见证角色替换。"""
        return self.base.roles

    @property
    def gaps(self):
        """唯一新操作是前置受训整数前缀，全部剩余间隔保持。"""
        return (self.rule.prefix_values + self.base.gaps[0], *self.base.gaps[1:])

    @property
    def answer_role_key(self):
        """答案角色仍由原始WH帧指定，前缀不能创造事实槽。"""
        return self.base.answer_role_key

    @property
    def wh_values(self):
        """问词整数成员从base继承，不提供新词或指代。"""
        return self.base.wh_values

    @property
    def slot_mode(self):
        """plain模式在构造时已经核验，不能扩大记忆权限。"""
        return self.base.slot_mode

    @property
    def is_slot_question(self):
        """本对象仅承载WH语法。"""
        return True

    @property
    def is_reference_slot(self):
        """前缀构造从不授权跨轮指代。"""
        return False

    @property
    def source_frames(self):
        """base及全部前缀见证共同作为挂载模型成员资格。"""
        return (self.base, *self.rule.source_frames)

    def stable_key(self):
        """完整编码base和双见证规则，不以派生摘要代替一等成员。"""
        return pack_record(PREFIXED_SLOT_FRAME, self.base.stable_key(), self.rule.stable_key())

    @classmethod
    def from_stable_key(cls, key):
        """从保存的整数成员重建相同语法及完整来源义务。"""
        fields = unpack_record(key, PREFIXED_SLOT_FRAME)
        if len(fields) != 2:
            raise ValueError('prefixed WH frame field count differs')
        return cls(TrainedRelationQuestionFrame.from_stable_key(fields[0]),
                   QuestionPrefixRule.from_stable_key(fields[1]))

    def open_frame(self):
        """把同一构造对象交给原开放角色解析器，不填充任何未知端点。"""
        key = self.stable_key()
        return OpenRoleFrame(key, self.predicate_key, (QUESTION_SOURCE_REF, *key),
                             self.roles, self.gaps, (0, len(self.roles) + sum(map(len, self.gaps))))


def prefixed_slot_question_frames(frames, values, *, covered_frames=(), max_combinations=4096, initial_combinations=0):
    """从当前训练图恢复双见证，只在原共享组合预算内构造当前前缀候选。"""
    if (type(max_combinations) is not int or max_combinations <= 0
            or type(initial_combinations) is not int or not 0 <= initial_combinations <= max_combinations):
        raise ValueError('question prefix composition budget must be positive and bounded')
    grouped = {}
    for frame in frames:
        if type(frame) is TrainedRelationQuestionFrame and frame.slot_mode == SLOT_MODE_PLAIN:
            grouped.setdefault(_contract(frame), []).append(frame)
    witnesses = {}
    for owners in grouped.values():
        for base in owners:
            for extended in owners:
                length = len(extended.gaps[0]) - len(base.gaps[0])
                if length <= 0 or extended.gaps[0][length:] != base.gaps[0]:
                    continue
                prefix = extended.gaps[0][:length]
                if values[:length] != prefix:
                    continue
                scope = (base.operation_key, base.predicate_key, prefix)
                witnesses.setdefault(scope, (set(), set()))[int(base.is_slot_question)].add((base, extended))
    covered = {(_contract(frame), frame.gaps[0]) for frame in (*frames, *covered_frames)}
    result = []
    searched = initial_combinations
    for (operation, predicate, prefix), (complete, slots) in sorted(witnesses.items()):
        if not complete or not slots:
            continue
        rule = QuestionPrefixRule(prefix, tuple(sorted(complete, key=_pair_key)), tuple(sorted(slots, key=_pair_key)))
        for base in sorted(frames, key=lambda frame: frame.stable_key()):
            if (type(base) is not TrainedRelationQuestionFrame or not base.is_slot_question
                    or base.slot_mode != SLOT_MODE_PLAIN
                    or (base.operation_key, base.predicate_key) != (operation, predicate)):
                continue
            gap = prefix + base.gaps[0]
            if (values[:len(gap)] != gap or any(part and not any(
                    values[start:start + len(part)] == part for start in range(len(values) - len(part) + 1))
                    for part in base.gaps[1:])):
                continue
            searched += 1
            if searched > max_combinations:
                raise QuestionCompositionBudgetExceeded('question prefix syntax exceeds shared request budget')
            shape = (_contract(base), gap)
            if shape not in covered:
                result.append(PrefixedSlotQuestionFrame(base, rule))
                covered.add(shape)
    return tuple(sorted(result, key=lambda frame: frame.stable_key()))
