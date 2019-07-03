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


@pytest.fixture(params=[
    (Timedelta('0 days'), Timedelta('1 day')),
    (Timestamp('2018-01-01'), Timedelta('1 day')),
    (0, 1)], ids=lambda x: type(x[0]).__name__)
def start_shift(request):
    """
    Fixture for generating intervals of different types from a start value
    and a shift value that can be added to start to generate an endpoint.
    """
    return request.param


def get_expected(intervals, query, method):
    # vectorized implementation should be consistent with the elementwise
    # operation on the individual Interval objects
    expected = []
    for interval in intervals:
        if isna(interval):
            interval_expected = False
        else:
            interval_expected = getattr(interval, method)(query)
        expected.append(interval_expected)
    return np.array(expected)


class TestOverlaps:

    def test_overlaps_interval(
            self, constructor, start_shift, closed, other_closed):
        start, shift = start_shift
        interval = Interval(start, start + 3 * shift, other_closed)

        # intervals: identical, nested, spanning, partial, adjacent, disjoint
        tuples = [(start, start + 3 * shift),
                  (start + shift, start + 2 * shift),
                  (start - shift, start + 4 * shift),
                  (start + 2 * shift, start + 4 * shift),
                  (start + 3 * shift, start + 4 * shift),
                  (start + 4 * shift, start + 5 * shift)]
        interval_container = constructor.from_tuples(tuples, closed)

        adjacent = (interval.closed_right and interval_container.closed_left)
        expected = np.array([True, True, True, True, adjacent, False])
        result = interval_container.overlaps(interval)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('other_constructor', [
        IntervalArray, IntervalIndex])
    def test_overlaps_interval_container(self, constructor, other_constructor):
        # TODO: modify this test when implemented
        interval_container = constructor.from_breaks(range(5))
        other_container = other_constructor.from_breaks(range(5))
        with pytest.raises(NotImplementedError):
            interval_container.overlaps(other_container)

    def test_overlaps_na(self, constructor, start_shift):
        """NA values are marked as False"""
        start, shift = start_shift
        interval = Interval(start, start + shift)

        tuples = [(start, start + shift),
                  np.nan,
                  (start + 2 * shift, start + 3 * shift)]
        interval_container = constructor.from_tuples(tuples)

        expected = np.array([True, False, False])
        result = interval_container.overlaps(interval)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('other', [
        10, True, 'foo', Timedelta('1 day'), Timestamp('2018-01-01')],
        ids=lambda x: type(x).__name__)
    def test_overlaps_invalid_type(self, constructor, other):
        interval_container = constructor.from_breaks(range(5))
        msg = '`other` must be Interval-like, got {other}'.format(
            other=type(other).__name__)
        with pytest.raises(TypeError, match=msg):
            interval_container.overlaps(other)


class TestContains:

    def get_tuples(self, start, shift):
        tuples = [
            (start, start),
            (start + shift, start + shift),
            (start, start + shift),
            (start, start + 2 * shift),
            (start + shift, start + 2 * shift),
            (start - shift, start + 3 * shift),
            np.nan]
        return tuples

    @pytest.fixture
    def interval_query(self, request, start_shift, other_closed):
        start, shift = start_shift
        left = start + request.param[0] * shift
        right = start + request.param[1] * shift
        return Interval(left, right, other_closed)

    @pytest.fixture
    def scalar_query(self, request, start_shift):
        start, shift = start_shift
        return start + request.param * shift

    @pytest.mark.parametrize('interval_query', [
        (0, 0), (0, 1), (1, 2), (-5, 5), (10, 10)], indirect=True)
    def test_contains_interval(
            self, constructor, closed, interval_query, start_shift):
        tuples = self.get_tuples(*start_shift)
        intervals = constructor.from_tuples(tuples, closed)

        result = intervals.contains(interval_query)
        expected = get_expected(intervals, interval_query, 'contains')
        print(result)
        print(expected)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('scalar_query', [0, 1, 10], indirect=True)
    def test_contains_scalar(
            self, constructor, closed, scalar_query, start_shift):
        tuples = self.get_tuples(*start_shift)
        intervals = constructor.from_tuples(tuples, closed)

        result = intervals.contains(scalar_query)
        expected = get_expected(intervals, scalar_query, 'contains')
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('other_constructor', [
        IntervalArray, IntervalIndex])
    def test_contains_interval_iterable(self, constructor, other_constructor):
        # TODO: modify this test when implemented
        interval_container = constructor.from_breaks(range(5))
        other_container = other_constructor.from_breaks(range(5))
        with pytest.raises(NotImplementedError):
            interval_container.contains(other_container)
