from dataclasses import dataclass
from itertools import takewhile
from typing import Callable, Any, Generator, Optional, Collection

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class IntervalTree:
    key: float | np.ndarray
    left: Optional["IntervalTree"]
    right: Optional["IntervalTree"]
    sorted_left: np.ndarray
    sorted_right: np.ndarray

    @staticmethod
    def _get_recursive_function_on_tree(
        inductive_step: Callable[["IntervalTree"], Any],
        base_case: Callable[[], Any] = lambda: 0,
    ) -> Callable[[Optional["IntervalTree"]], Any]:
        def f(root):
            match root:
                case None as __:
                    return base_case()
                case IntervalTree() as root:
                    return inductive_step(root)

        return f

    @staticmethod
    def _sort_by_start_ascending(intervals: Collection[np.ndarray]) -> np.ndarray:
        return np.array(list(sorted(intervals, key=lambda a: a[0][0])))

    @staticmethod
    def _sort_by_end_descending(intervals: Collection[np.ndarray]) -> np.ndarray:
        return np.array(list(sorted(intervals, key=lambda a: a[0][1], reverse=True)))

    @staticmethod
    def from_ndarray(intervals: np.ndarray):
        if not isinstance(intervals, np.ndarray):
            raise TypeError(
                f"expected intervals to be of type np.ndarray not {type(intervals)}"
            )
        if len(intervals.shape) != 3 or (intervals.shape[1] != intervals.shape[2] != 2):
            raise ValueError(
                f"expected intervals shape to be (_, 2, 2) not {intervals.shape}"
            )
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
            IntervalTree._sort_by_start_ascending(mid),
            IntervalTree._sort_by_end_descending(mid),
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

    def node_count(self):
        node_count_function = IntervalTree._get_recursive_function_on_tree(
            lambda node: 1
            + node_count_function(node.left)
            + node_count_function(node.right),
        )
        return node_count_function(self)

    def depth(self):
        depth_function = IntervalTree._get_recursive_function_on_tree(
            lambda node: 1 + max(depth_function(node.left), depth_function(node.right)),
        )
        return depth_function(self)

    def __len__(self):
        interval_count_function = IntervalTree._get_recursive_function_on_tree(
            lambda node: len(node.intervals)
            + interval_count_function(node.left)
            + interval_count_function(node.right),
        )
        return interval_count_function(self)

    def sanity_check(self):
        def inductive_step(root):
            left = sanity_check_function(root.left)
            right = sanity_check_function(root.right)
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

        sanity_check_function = IntervalTree._get_recursive_function_on_tree(
            lambda: None, inductive_step
        )
        sanity_check_function(self)

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
                    root.sorted_left = IntervalTree._sort_by_start_ascending(intervals)
                    root.sorted_right = IntervalTree._sort_by_end_descending(intervals)
                return root

    def insert(self, interval: ArrayLike):
        interval = np.array(interval)
        if interval.shape != (2, 2):
            raise ValueError(
                f"incorrect interval shape {interval.shape}, expected {(2, 2)}"
            )
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
        raise NotImplementedError

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
