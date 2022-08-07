import pytest

from interval_tree import *
from utils import gen_intervals


class TestIntervalTree:
    def test_creating_offline_creation(self):
        intervals = gen_intervals(20)
        root = IntervalTree.from_ndarray(intervals)
        assert len(root) == 20
