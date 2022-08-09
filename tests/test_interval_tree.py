from interval_tree import IntervalTree, gen_interval
from interval_tree import gen_intervals, gen_intervals_no_y
import numpy as np
import pytest


class TestIntervalTree:
    def test_constructor_passes_given_valid_intervals(self):
        intervals = gen_intervals(20)
        root = IntervalTree.from_ndarray(intervals)
        assert len(root) == 20

    def test_constructor_passes_given_valid_intervals_no_y(self):
        intervals = gen_intervals_no_y(20)
        root = IntervalTree.from_ndarray(intervals)
        assert len(root) == 20

    def test_constructor_raises_exception_when_given_none_value_in_constructor(self):
        with pytest.raises(TypeError):
            IntervalTree.from_ndarray(None)

    def test_constructor_raises_exception_when_given_wrongly_shaped_intervals(self):
        intervals = np.random.randint(0, 100, (2, 2, 1))
        with pytest.raises(ValueError):
            IntervalTree.from_ndarray(intervals)

    def test_inserting_with_tuple_of_tuples_passes(self):
        root = IntervalTree.from_ndarray(gen_interval()[:, np.newaxis])
        root.insert(((1, 2), (-1, -1)))

    def test_inserting_with_list_of_lists_passes(self):
        root = IntervalTree.from_ndarray(gen_interval()[:, np.newaxis])
        root.insert([[1, 2], [-1, -1]])

    def test_inserting_with_list_of_tuples_passes(self):
        root = IntervalTree.from_ndarray(gen_interval()[:, np.newaxis])
        root.insert([(1, 2), (-1, -1)])

    def test_conflicting_intervals(self):
        np.random.seed(0)
        intervals = gen_intervals(10)
        results = IntervalTree.find_conflicting_intervals(intervals)
        assert len(results) == 7
