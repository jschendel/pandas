"""Tests for Interval-Interval operations, such as overlaps, contains, etc."""
import pytest

from pandas import Interval, Timedelta, Timestamp


@pytest.fixture(params=[
    (Timedelta('0 days'), Timedelta('1 day')),
    (Timestamp('2018-01-01'), Timedelta('1 day')),
    (0, 1)], ids=lambda x: type(x[0]).__name__)
def start_shift(request):
    """
    Fixture for generating intervals of types from a start value and a shift
    value that can be added to start to generate an endpoint
    """
    return request.param


class TestOverlaps:

    def test_overlaps_self(self, start_shift, closed):
        start, shift = start_shift
        interval = Interval(start, start + shift, closed)
        assert interval.overlaps(interval)

    def test_overlaps_nested(self, start_shift, closed, other_closed):
        start, shift = start_shift
        interval1 = Interval(start, start + 3 * shift, other_closed)
        interval2 = Interval(start + shift, start + 2 * shift, closed)

        # nested intervals should always overlap
        assert interval1.overlaps(interval2)

    def test_overlaps_disjoint(self, start_shift, closed, other_closed):
        start, shift = start_shift
        interval1 = Interval(start, start + shift, other_closed)
        interval2 = Interval(start + 2 * shift, start + 3 * shift, closed)

        # disjoint intervals should never overlap
        assert not interval1.overlaps(interval2)

    def test_overlaps_endpoint(self, start_shift, closed, other_closed):
        start, shift = start_shift
        interval1 = Interval(start, start + shift, other_closed)
        interval2 = Interval(start + shift, start + 2 * shift, closed)

        # overlap if shared endpoint is closed for both (overlap at a point)
        result = interval1.overlaps(interval2)
        expected = interval1.closed_right and interval2.closed_left
        assert result == expected

    @pytest.mark.parametrize('other', [
        10, True, 'foo', Timedelta('1 day'), Timestamp('2018-01-01')],
        ids=lambda x: type(x).__name__)
    def test_overlaps_invalid_type(self, other):
        interval = Interval(0, 1)
        msg = '`other` must be an Interval, got {other}'.format(
            other=type(other).__name__)
        with pytest.raises(TypeError, match=msg):
            interval.overlaps(other)


class TestContains:

    def test_contains_scalar(self, start_shift, closed):
        start, shift = start_shift
        interval = Interval(start, start + 2 * shift, closed)

        # midpoint is always contained
        assert interval.contains(start + shift)

        # endpoints are contained if the interval is closed on the given side
        expected = interval.closed_left
        result = interval.contains(start)
        assert result is expected

        expected = interval.closed_right
        result = interval.contains(start + 2 * shift)
        assert result is expected

    def test_contains_empty_scalar(self, start_shift, closed):
        start, shift = start_shift
        interval = Interval(start, start, closed)

        # endpoint is included if interval is a non-empty point
        expected = closed == 'both'
        result = interval.contains(start)
        assert result is expected

        # disjoint scalar is never contained
        assert not interval.contains(start + shift)

    def test_contains_self(self, start_shift, closed):
        # an interval always contains itself
        start, shift = start_shift
        interval = Interval(start, start + shift, closed)
        assert interval.contains(interval)

    def test_contains_disjoint(self, start_shift, closed, other_closed):
        start, shift = start_shift
        interval1 = Interval(start, start + shift, other_closed)
        interval2 = Interval(start + 2 * shift, start + 3 * shift, closed)

        # disjoint intervals should never contain each other
        assert not interval1.contains(interval2)
        assert not interval2.contains(interval1)

    def test_contains_nested_strict(self, start_shift, closed, other_closed):
        # nested intervals with no shared endpoints
        start, shift = start_shift
        interval1 = Interval(start, start + 3 * shift, other_closed)
        interval2 = Interval(start + shift, start + 2 * shift, closed)

        # The outer nested interval contains the inner nested interval
        assert interval1.contains(interval2)

        # The inner nested interval does not contain the outer nested interval
        assert not interval2.contains(interval1)

    def test_contains_nested_shared(self, start_shift, closed, other_closed):
        # nested intervals that share an endpoint
        start, shift = start_shift
        interval1 = Interval(start, start + 2 * shift, other_closed)
        interval2 = Interval(start + shift, start + 2 * shift, closed)

        # outer interval does not contain the inner interval if outers endpoint
        # is not included but the inner intervals endpoint is included
        expected = not (interval1.open_right and interval2.closed_right)
        result = interval1.contains(interval2)
        assert result is expected

        # The inner nested interval does not contain the outer nested interval
        assert not interval2.contains(interval1)

    def test_contains_overlapping(self, start_shift, closed, other_closed):
        start, shift = start_shift
        interval1 = Interval(start, start + 2 * shift, other_closed)
        interval2 = Interval(start + shift, start + 3 * shift, closed)

        # overlapping intervals should never contain each other
        assert not interval1.contains(interval2)
        assert not interval2.contains(interval1)

    @pytest.mark.parametrize('closed', ['left', 'right', 'neither'])
    @pytest.mark.parametrize('other_closed', ['left', 'right', 'neither'])
    def test_contains_empty_empty(self, start_shift, closed, other_closed):
        start, shift = start_shift
        interval1 = Interval(start, start, closed)
        interval2 = Interval(start + shift, start + shift, other_closed)

        # empty intervals are trivially contained, as all points are included
        assert interval1.contains(interval1)
        assert interval1.contains(interval2)
        assert interval2.contains(interval1)

    @pytest.mark.parametrize('closed', ['left', 'right', 'neither'])
    def test_contains_empty_nonempty(self, start_shift, closed, other_closed):
        start, shift = start_shift
        interval_empty = Interval(start - shift, start - shift, closed)
        interval_nonempty = Interval(start, start + shift, other_closed)

        # non-empty intervals always contain empty intervals
        assert interval_nonempty.contains(interval_empty)

        # empty intervals never contain a non-empty interval
        assert not interval_empty.contains(interval_nonempty)

    def test_contains_point(self, start_shift, closed):
        start, shift = start_shift
        interval_point = Interval(start, start, 'both')
        interval_empty = Interval(start, start, closed)
        interval_nonempty = Interval(start - shift, start + shift, closed)

        # point always contains empty interval (or an equal point interval)
        assert interval_point.contains(interval_empty)

        # empty interval never contains a point (unless empty = point interval)
        expected = closed == 'both'
        result = interval_empty.contains(interval_point)
        assert result is expected

        # point is contained in an enclosing interval and doesn't contain it
        assert interval_nonempty.contains(interval_point)
        assert not interval_point.contains(interval_nonempty)
