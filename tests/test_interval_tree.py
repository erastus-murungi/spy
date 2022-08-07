from interval_tree import IntervalTree
from interval_tree import gen_intervals


class TestIntervalTree:
    def test_creating_offline_creation(self):
        intervals = gen_intervals(20)
        root = IntervalTree.from_ndarray(intervals)
        assert len(root) == 20
