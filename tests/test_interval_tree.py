from interval_tree import IntervalTree
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
