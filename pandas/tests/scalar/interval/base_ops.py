import numpy as np
import pytest

from pandas import Interval, Timedelta, isna, NaT


class BaseCases:
    @pytest.fixture
    def interval_proper(self, interval_start_shift, closed):
        """
        Fixture for generating Intervals of different dtypes and closed to use as
        the calling interval in test cases.
        """
        start, shift = interval_start_shift
        return Interval(start, start + 3 * shift, closed)

    @pytest.fixture
    def interval_degenerate(self, interval_start_shift, closed):
        """
        Fixture for generating Intervals of different dtypes and closed to use as
        the calling interval in test cases.
        """
        start = interval_start_shift[0]
        return Interval(start, start, closed)

    def test_proper_cases(self, interval_proper, proper_cases):
        """
        Test the method for proper (left != right) Intervals with Interval queries
        """
        case, query = proper_cases
        result = getattr(interval_proper, self.method_name)(query)
        expected = self.expected_proper(case, interval_proper, query)
        assert result is expected

    def test_degenerate_cases(self, interval_degenerate, degenerate_cases):
        """
        Test degenerate Intervals (left == right)
        """
        case, query = degenerate_cases
        result = getattr(interval_degenerate, self.method_name)(query)
        expected = self.expected_degenerate(case, interval_degenerate, query)
        assert result is expected


class IntervalCases(BaseCases):
    @pytest.fixture(
        params=[
            ('before_disjoint', -2, -1),
            ('before_touching_left', -1, 0),
            ('before_touching_inside', -1, 1),
            ('before_touching_right', -1, 3),
            ('inside_touching_left', 0, 1),
            ('inside_touching_right', 1, 3),
            ('nested', 1, 2),
            ('same', 0, 3),
            ('after_touching_left', 0, 4),
            ('after_touching_inside', 1, 4),
            ('after_touching_right', 3, 4),
            ('after_disjoint', 4, 5),
            ('containing', -1, 4),
            ('degenerate_before', -1, -1),
            ('degenerate_left', 0, 0),
            ('degenerate_inside', 1, 1),
            ('degenerate_right', 3, 3),
            ('degenerate_after', 4, 4)
        ],
        ids=lambda x: x[0],
    )
    def proper_cases(self, request, interval_start_shift, other_closed):
        """
        Fixture for generating Intervals of different dtypes and closed to use as
        test cases for the various methods.

        The fixture params denote the (left, right) `shift` multiplier to be added
        to `start` in order build the endpoints of an Interval.
        """
        start, shift = interval_start_shift
        left = start + request.param[1] * shift
        right = start + request.param[2] * shift
        interval = Interval(left, right, other_closed)
        return (request.param[0], interval)

    @pytest.fixture(
        params=[
            ('before_disjoint', -2, -1),
            ('before_touching', -1, 0),
            ('after_touching', 0, 1),
            ('after_disjoint', 1, 2),
            ('containing', -1, 1),
            ('degenerate_before', -1, -1),
            ('degenerate_same', 0, 0),
            ('degenerate_after', 1, 1)
        ],
        ids=lambda x: x[0],
    )
    def degenerate_cases(self, request, interval_start_shift, other_closed):
        """
        Fixture for generating Intervals of different dtypes and closed to use as
        test cases for the various methods.

        The fixture params denote the (left, right) `shift` multiplier to be added
        to `start` in order build the endpoints of an Interval.
        """
        start, shift = interval_start_shift
        left = start + request.param[1] * shift
        right = start + request.param[2] * shift
        interval = Interval(left, right, other_closed)
        return (request.param[0], interval)


class PointCases(BaseCases):
    def make_point(self, start, shift, multiplier):
        if isna(multiplier):
            return NaT if isinstance(shift, Timedelta) else np.nan
        return start + multiplier * shift

    @pytest.fixture(
        params=[('before', -1), ('left', 0), ('mid', 1.5), ('right', 3), ('after', 4), ('NA', np.nan)],
        ids=lambda x: x[0],
    )
    def proper_cases(self, request, interval_start_shift):
        """
        Fixture for generating Intervals of different dtypes and closed to use as
        test cases for the various methods.

        The fixture params denote the (left, right) `shift` multiplier to be added
        to `start` in order build the endpoints of an Interval.
        """
        start, shift = interval_start_shift
        multiplier = request.param[1]
        return (request.param[0], self.make_point(start, shift, multiplier))

    @pytest.fixture(
        params=[('before', -1), ('same', 0), ('after', 1), ('NA', np.nan)],
        ids=lambda x: x[0],
    )
    def degenerate_cases(self, request, interval_start_shift):
        """
        Fixture for generating Intervals of different dtypes and closed to use as
        test cases for the various methods.

        The fixture params denote the (left, right) `shift` multiplier to be added
        to `start` in order build the endpoints of an Interval.
        """
        start, shift = interval_start_shift
        multiplier = request.param[1]
        return (request.param[0], self.make_point(start, shift, multiplier))
