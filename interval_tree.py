from dataclasses import dataclass
from itertools import takewhile
from typing import Callable, Any, Generator, Optional, Collection

import numpy as np


@dataclass
class IntervalTree:
    key: float | np.ndarray
    left: Optional["IntervalTree"]
    right: Optional["IntervalTree"]
    sorted_left: np.ndarray
    sorted_right: np.ndarray

    @staticmethod
    def recurse(
        base_case: Callable[[], Any], inductive_step: Callable[["IntervalTree"], Any]
    ) -> Callable[[Optional["IntervalTree"]], Any]:
        def f(root):
            match root:
                case None as __:
                    return base_case()
                case IntervalTree() as root:
                    return inductive_step(root)

        return f

    @staticmethod
    def sort_by_start_ascending(l_intervals: np.ndarray):
        return np.array(list(sorted(l_intervals, key=lambda a: a[0][0])))

    @staticmethod
    def sort_by_end_descending(l_intervals):
        return np.array(list(sorted(l_intervals, key=lambda a: a[0][1], reverse=True)))

    @staticmethod
    def from_ndarray(intervals: np.ndarray):
        return IntervalTree._construct(intervals)

    @staticmethod
    def _construct(
        intervals: np.ndarray,
    ) -> Optional["IntervalTree"]:
        if intervals.size == 0:
            return None

        x_mid = np.median(intervals[:, 0, 1])  # median endpoint
        mid = intervals[
            np.where(
                (intervals[:, 0, 0] <= x_mid) & (x_mid <= intervals[:, 0, 1])
            )  # right in the middle
        ]

        return IntervalTree(
            x_mid,
            IntervalTree._construct(
                intervals[intervals[:, 0, 1] < x_mid]
            ),  # completely to the left of the mid
            IntervalTree._construct(
                intervals[x_mid < intervals[:, 0, 0]]
            ),  # completely to the right of the mid
            IntervalTree.sort_by_start_ascending(mid),
            IntervalTree.sort_by_end_descending(mid),
        )

    def __iter__(self):
        yield from (
            self.key,
            self.left,
            self.right,
            self.sorted_left,
            self.sorted_right,
        )

    @property
    def intervals(self):
        return self.sorted_left

    def size(self):
        _size = IntervalTree.recurse(
            lambda: 0, lambda node: 1 + _size(node.left) + _size(node.right)
        )
        return _size(self)

    def depth(self):
        _depth = IntervalTree.recurse(
            lambda: 0, lambda node: 1 + max(_depth(node.left), _depth(node.right))
        )
        return _depth(self)

    def __len__(self):
        _n_intervals = IntervalTree.recurse(
            lambda: 0,
            lambda node: len(node.intervals)
            + _n_intervals(node.left)
            + _n_intervals(node.right),
        )
        return _n_intervals(self)

    def check_invariant(self):
        def inductive_step(root):
            left = f(root.left)
            right = f(root.right)
            x_mid = root.key

            if left is not None:
                assert np.all(left[:, 0, 1] < x_mid)
            if right is not None:
                assert np.all(x_mid < right[:, 0, 0])
            if root.intervals.size != 0:
                assert np.all(
                    (root.intervals[:, 0, 0] <= x_mid)
                    & (x_mid <= root.intervals[:, 0, 1])
                )
            return np.concatenate(
                IntervalTree._non_empty([left, right, root.intervals])
            )

        f = IntervalTree.recurse(lambda: None, inductive_step)
        f(self)

    @staticmethod
    def _non_empty(arrays: Collection[np.ndarray]):
        return list(filter(lambda arr: arr is not None and arr.size != 0, arrays))

    @staticmethod
    def _new_node(interval):
        return IntervalTree(
            interval[0, 1],
            None,
            None,
            interval[np.newaxis, :],
            interval[np.newaxis, :],
        )

    @staticmethod
    def _insert_impl(root: "IntervalTree", interval):
        match root:
            case None as __:
                return IntervalTree._new_node(
                    interval,
                )
            case IntervalTree() as root:
                if interval[0, 1] < root.key:
                    root.left = IntervalTree._insert_impl(root.left.i, interval)
                elif interval[0, 0] > root.key:
                    root.right = IntervalTree._insert_impl(root.right, interval)
                else:
                    intervals = np.concatenate(
                        IntervalTree._non_empty(
                            [root.intervals, interval[np.newaxis, :]]
                        )
                    )
                    root.sorted_left = IntervalTree.sort_by_start_ascending(intervals)
                    root.sorted_right = IntervalTree.sort_by_end_descending(intervals)
                return root

    def insert(self, interval):
        IntervalTree._insert_impl(self, interval)

    @staticmethod
    def all_intervals_impl(root):
        match root:
            case None as __:
                return
            case IntervalTree() as root:
                yield from root.intervals
                yield from IntervalTree.all_intervals(root.left)
                yield from IntervalTree.all_intervals(root.right)

    def all_intervals(self):
        return IntervalTree.all_intervals_impl(self)

    @staticmethod
    def overlapping_interval_search_impl(
        v: "IntervalTree", interval: tuple[float, float]
    ) -> Generator[np.ndarray, None, None]:
        match v:
            case IntervalTree() as root:
                l, r = interval
                if l <= root.key <= r:
                    yield from root.intervals  # O(|F|)
                    yield from IntervalTree.overlapping_interval_search_impl(
                        root.left, interval
                    )
                    yield from IntervalTree.overlapping_interval_search_impl(
                        root.right, interval
                    )
                if r < root.key:
                    yield from takewhile(lambda a: a[0, 0] <= r, root.intervals)
                    yield from IntervalTree.overlapping_interval_search_impl(
                        root.left, interval
                    )
                if l > root.key:
                    yield from takewhile(lambda a: a[0, 1] >= l, root.intervals)
                    yield from IntervalTree.overlapping_interval_search_impl(
                        root.right, interval
                    )

    @staticmethod
    def enclosing_interval_search_impl(
        v: "IntervalTree", interval: tuple[float, float]
    ):
        if v is None:
            return
        match v:
            case None as root:
                pass
            case IntervalTree() as root:
                l, r = interval
                pass

    def overlapping_interval_search(self, interval):
        l, r = interval
        if l > r:
            raise ValueError(
                f"invalid interval {interval}: {interval[0] > interval[1]}"
            )
        return IntervalTree.overlapping_interval_search_impl(self, interval)

    def enclosing_interval_search(self, point: float):
        return IntervalTree.overlapping_interval_search(self, (point, point))

    def __str__(self):
        return f"Size = {self.sorted_left.shape[0]}\nKey = {self.key:6.2f}\n" + (
            ""
            if self.intervals.size == 0
            else "\n".join(
                map(
                    lambda interval: f"[{interval[0]: 6.2f}, {interval[1]: 6.2f})",
                    self.intervals[:, 0],
                )
            )
        )
