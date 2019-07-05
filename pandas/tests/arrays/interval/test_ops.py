"""Tests for Interval-Interval operations, such as overlaps, contains, etc."""
import numpy as np
import pytest

from pandas import Interval, IntervalIndex, Timedelta, Timestamp, isna
from pandas.core.arrays import IntervalArray
import pandas.util.testing as tm


@pytest.fixture(params=[IntervalArray, IntervalIndex])
def constructor(request):
    """
    Fixture for testing both interval container classes.
    """
    return request.param


@pytest.fixture
def intervals(request, constructor, interval_start_shift, closed):
    """
    Fixture for generating an IntervalArray/IntervalIndex of different
    dtypes and closed to be used as test cases.
    """
    start, shift = interval_start_shift
    tuples = [
        (start - 2 * shift, start - shift),      # before disjoint
        (start - shift, start),                  # before touching left
        (start - shift, start + shift),          # before touching inside
        (start - shift, start + 3 * shift),      # before touching right
        (start, start + shift),                  # inside touching left
        (start + shift, start + 3 * shift),      # inside touching right
        (start + shift, start + 2 * shift),      # nested
        (start, start + 3 * shift),              # same
        (start, start + 4 * shift),              # after touching left
        (start + shift, start + 4 * shift),      # after touching inside
        (start + 3 * shift, start + 4 * shift),  # after touching right
        (start + 4 * shift, start + 5 * shift),  # after disjoint
        (start - shift, start + 4 * shift),      # containing
        (start - shift, start - shift),          # degenerate before
        (start, start),                          # degenerate touching left
        (start + shift, start + shift),          # degenerate inside
        (start + 3 * shift, start + 3 * shift),  # degenerate touching right
        (start + 4 * shift, start + 4 * shift),  # degenerate after
        np.nan,                                  # missing value
    ]
    return constructor.from_tuples(tuples, closed=closed)


@pytest.fixture(params=[(0, 3), (0, 0), (10, 10)])
def interval(request, interval_start_shift, other_closed):
    """
    Fixture for generating Intervals of different dtypes and closed to use as
    test cases for the various methods.

    The fixture params denote the (left, right) `shift` multiplier to be added
    to `start` in order build the endpoints of an Interval.
    """
    start, shift = interval_start_shift
    left = start + request.param[0] * shift
    right = start + request.param[1] * shift
    return Interval(left, right, other_closed)


@pytest.fixture(params=[-10, 0, 1, 3, 10])
def point(request, interval_start_shift):
    """
    Fixture for generating points of different dtypes to use as test cases for
    the various methods.

    The fixture params denote the `shift` multiplier to be added to `start` in
    order to generate the point.
    """
    start, shift = interval_start_shift
    return start + request.param * shift


class QueryCases:

    # name of method being tests; to be defined in child class
    method_name = None

    def get_expected(self, intervals, query):
        # vectorized implementation should be consistent with the elementwise
        # operation on the individual Interval objects
        expected = []
        for interval in intervals:
            if isna(interval):
                interval_expected = False
            else:
                interval_expected = getattr(interval, self.method_name)(query)
            expected.append(interval_expected)
        return np.array(expected)

    def test_interval_queries(self, intervals, interval):
        # generically test interval queries for interval methods
        result = getattr(intervals, self.method_name)(interval)
        expected = self.get_expected(intervals, interval)
        tm.assert_numpy_array_equal(result, expected)

    def test_point_queries(self, intervals, point):
        # generically test point queries for interval methods
        result = getattr(intervals, self.method_name)(point)
        expected = self.get_expected(intervals, point)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("constructor2", [IntervalArray, IntervalIndex])
    def test_interval_iterable_notimplemented(self, constructor, constructor2):
        # interval iterable queries not currently supported
        intervals = constructor.from_breaks(range(5))
        intervals2 = constructor2.from_breaks(range(5))
        with pytest.raises(NotImplementedError):
            getattr(intervals, self.method_name)(intervals2)


class TestOverlaps(QueryCases):

    method_name = "overlaps"

    @pytest.mark.skip(reason="point queries are not supported by overlaps")
    def test_point_queries(self):
        pass

    @pytest.mark.parametrize(
        "other",
        [10, True, "foo", Timedelta("1 day"), Timestamp("2018-01-01")],
        ids=lambda x: type(x).__name__,
    )
    def test_overlaps_point_errors(self, constructor, other):
        interval_container = constructor.from_breaks(range(5))
        msg = "`other` must be an Interval, got {other}".format(
            other=type(other).__name__
        )
        with pytest.raises(TypeError, match=msg):
            interval_container.overlaps(other)
