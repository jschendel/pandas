"""Tests for Interval-Interval operations, such as overlaps, contains, etc."""
import pytest

from pandas.tests.scalar.interval.base_ops import IntervalCases, PointCases


class TestOverlapsInterval(IntervalCases):

    method_name = "overlaps"

    def expected_proper(self, case, interval, query):
        expected = {
            'before_disjoint': False,
            'before_touching_left': interval.closed_left and query.closed_right,
            'before_touching_inside': True,
            'before_touching_right': True,
            'inside_touching_left': True,
            'inside_touching_right': True,
            'nested': True,
            'same': True,
            'after_touching_left': True,
            'after_touching_inside': True,
            'after_touching_right': interval.closed_right and query.closed_left,
            'after_disjoint': False,
            'containing': True,
            'degenerate_before': False,
            'degenerate_left': interval.closed_left and query.closed == "both",
            'degenerate_inside': query.closed == "both",
            'degenerate_right': interval.closed_right and query.closed == "both",
            'degenerate_after': False,
        }
        return expected[case]

    def expected_degenerate(self, case, interval, query):
        if interval.is_empty:
            return False

        # interval is a point
        expected = {
            'before_disjoint': False,
            'before_touching': query.closed_right,
            'after_touching': query.closed_left,
            'after_disjoint': False,
            'containing': True,
            'degenerate_before': False,
            'degenerate_same': query.closed == "both",
            'degenerate_after': False,
        }
        return expected[case]


class TestOverlapsPoint(PointCases):

    method_name = "overlaps"

    @pytest.mark.skip(reason='overlaps does not support point queries')
    def test_proper_cases(self):
        pass

    @pytest.mark.skip(reason='overlaps does not support point queries')
    def test_degenerate_cases(self):
        pass
