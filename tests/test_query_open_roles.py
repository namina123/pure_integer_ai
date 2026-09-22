"""开放角色整数机制专项；真实模型运行由独立施工入口产生。"""
from __future__ import annotations

import unittest

from pure_integer_ai.cognition.understanding.query_open_roles import (
    OpenRelationCandidate, OpenRoleBudgetExceeded, OpenRoleFrame, match_open_roles,
)


class OpenRolesTest(unittest.TestCase):
    """核验完整角色、歧义保留和预算失败的承重不变量。"""

    def test_complete_roles_roundtrip_and_no_example_identity(self):
        frame = OpenRoleFrame((18, 1), (7, 2), (3,), ((15, 1), (15, 2)),
                              ((), (8,), (9,)), (0, 5))
        values = (40, 41, 8, 50, 51, 52, 9)
        result = match_open_roles((frame,), values, (1, 7, len(values)))
        self.assertEqual(len(result), 1)
        candidate = result[0]
        self.assertEqual(tuple(item.values for item in candidate.bindings),
                         ((40, 41), (50, 51, 52)))
        self.assertEqual(OpenRelationCandidate.from_stable_key(candidate.stable_key()), candidate)
        self.assertEqual(candidate.values, values)
        self.assertNotEqual(candidate.stable_key(), frame.proposition)
        self.assertEqual(match_open_roles((frame,), (*values, 70), (1, 8, 8)), ())

    def test_ambiguous_delimiters_and_adjacent_roles_are_all_retained(self):
        frame = OpenRoleFrame((18, 1), (7, 2), (3,), ((15, 1), (15, 2)),
                              ((), (8,), ()), (0, 5))
        result = match_open_roles((frame,), (40, 8, 50, 8, 60), (1, 9, 5))
        self.assertEqual(len(result), 2)
        self.assertEqual({item.bindings[0].end for item in result}, {1, 3})
        adjacent = OpenRoleFrame((18, 2), (7, 2), (3,), ((15, 1), (15, 2)),
                                 ((9,), (), ()), (0, 5))
        self.assertEqual(len(match_open_roles((adjacent,), (9, 40, 50, 60), (1, 9, 4))), 2)
        with self.assertRaises(OpenRoleBudgetExceeded):
            match_open_roles((frame,), (40, 8, 50, 8, 60), (1, 9, 5), max_steps=2)

    def test_long_structure_does_not_depend_on_python_recursion(self):
        count = 1100
        frame = OpenRoleFrame((18, 1), (7, 2), (3,),
                              tuple((15, ordinal + 1) for ordinal in range(count)),
                              ((), *((8,) for _ in range(count))), (0, count * 2))
        values = tuple(value for _ in range(count) for value in (40, 8))
        result = match_open_roles((frame,), values, (1, 10, len(values)))
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].bindings), count)


if __name__ == "__main__":
    unittest.main()
