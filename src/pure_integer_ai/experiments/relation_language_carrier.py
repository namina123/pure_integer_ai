"""无正文框架的来源化载体间隔，不拼接或回构完整来源句。"""
from dataclasses import replace


def validate_carrier_boundaries(frame):
    """仅核对框架自身边界，供事实视图已过滤的纯语言解析使用。"""
    if (type(frame.carrier_prefix) is not str or type(frame.carrier_suffix) is not str
            or type(frame.envelope_start) is not int or type(frame.envelope_end) is not int
            or not 0 <= frame.envelope_start <= frame.envelope_end
            or len(frame.carrier_prefix) != frame.envelope_start
            or not frame.roles or len(frame.gaps) != len(frame.roles) + 1):
        raise ValueError('language carrier identity or explicit boundary members differ')


def validate_carrier_frame(fact, frame):
    """逐段核对位置、角色和来源，None不能冒充已学习的空间隔。"""
    validate_carrier_boundaries(frame)
    if (frame.proposition != fact.proposition or frame.predicate != fact.predicate
            or frame.source_hash != fact.source_hash
            or frame.roles != tuple(binding.role for binding in fact.bindings)
            or len(frame.gaps) != len(frame.roles) + 1):
        raise ValueError('language carrier identity or explicit boundary members differ')
    cursor = frame.envelope_start
    intervals = (frame.carrier_prefix, *frame.gaps, frame.carrier_suffix)
    for binding, gap in zip(fact.bindings, frame.gaps[:-1], strict=True):
        cursor += len(gap)
        if (binding.source_hash != fact.source_hash or binding.start != cursor
                or binding.end != cursor + len(binding.surface)
                or any(binding.surface and binding.surface in text for text in intervals)):
            raise ValueError('language carrier role intervals or exemplar literals differ')
        cursor = binding.end
    if (cursor + len(frame.gaps[-1]) != frame.envelope_end
            or not frame.envelope_start <= fact.cue_start < fact.cue_end <= frame.envelope_end
            or fact.cue_end - fact.cue_start != len(fact.cue)):
        raise ValueError('language carrier gap/cue envelope is incomplete')


def learn_carrier_frame(fact, frame):
    """训练迁移从原来源切出边缘，已登记V3成员可原样继承；不猜补旧V2。"""
    if frame.carrier_prefix is not None or frame.carrier_suffix is not None:
        validate_carrier_frame(fact, frame)
        return frame
    if (not fact.evidence_surface
            or not 0 <= frame.envelope_start <= frame.envelope_end <= len(fact.evidence_surface)):
        raise ValueError('language migration lacks original licensed carrier evidence')
    learned = replace(frame, carrier_prefix=fact.evidence_surface[:frame.envelope_start],
                      carrier_suffix=fact.evidence_surface[frame.envelope_end:])
    validate_carrier_frame(fact, learned)
    return learned
