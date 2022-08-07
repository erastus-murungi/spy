import graphviz
import matplotlib.pyplot as plt
import numpy as np


# plt.style.use(plt.style.available[4])
np.random.seed(3)


def gen_interval(x_min=-100, x_max=100, y_min=-100, y_max=100):
    y = np.random.uniform(y_min, y_max)
    x1 = np.random.uniform(x_min, x_max)
    x2 = np.random.uniform(x1, x_max)
    return np.array([[x1, x2], [y, y]])


def gen_interval_no_y(x_min=-100, x_max=100):
    x1 = np.random.uniform(x_min, x_max)
    x2 = np.random.uniform(x1, x_max)
    return np.array([[x1, x2]])


def gen_intervals(n, x_min=-100, x_max=100, y_min=-100, y_max=100) -> np.ndarray:
    return np.array([gen_interval(x_min, x_max, y_min, y_max) for _ in range(n)])


def gen_intervals_no_y(n, x_min=-100, x_max=100) -> np.ndarray:
    return np.array([gen_interval_no_y(x_min, x_max) for _ in range(n)])


def plot_intervals(intervals):
    plt.figure(figsize=(10, 10), dpi=300)
    for interval in intervals:
        plt.plot(*interval)
    plt.show()


def plot_intervals_after_query(all_intervals, subset_intervals, query):
    plt.figure(figsize=(10, 6), dpi=300)
    for interval in all_intervals:
        if any(
            np.array_equal(interval, other_interval)
            for other_interval in subset_intervals
        ):
            plt.plot(*interval, color="green", marker="|", linestyle="dashed")
        else:
            plt.plot(*interval, color="black", marker="|", linestyle="dashed")
    plt.plot(list(query), [50, 50], color="red", linewidth=4)
    plt.show()


def draw_interval_tree_dot(root):
    dot = graphviz.Digraph()
    dot.body.append(
        """ 
    graph [fontname = "Courier"];
    node [fontname = "Courier"];
    edge [fontname = "Courier"];\n
    """
    )
    stack = [root]
    while stack:
        node = stack.pop()
        id_node = id(node)
        if node is None:
            continue
        dot.node(
            name=f"{id_node}",
            label=f"{node}",
        )
        dot.edge(f"{id_node}", f"{id(node.left)}")
        dot.edge(f"{id_node}", f"{id(node.right)}")

        stack.append(node.left)
        stack.append(node.right)
    dot.render(filename="interval_tree.dot")
