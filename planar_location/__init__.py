import random
from dataclasses import dataclass, astuple
from math import tau
from typing import Collection

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True, slots=True)
class Point2D:
    x: float
    y: float

    def __iter__(self):
        yield from astuple(self)


class Polygon(list[Point2D]):
    def __init__(self, points: Collection[Point2D] = None):
        super(Polygon, self).__init__()
        if points:
            super(Polygon, self).extend(points)


def plot_polygons(polygons: Collection[Polygon]) -> None:
    for polygon in polygons:
        xs, ys = [], []
        for x, y in polygon:
            xs.append(x)
            ys.append(y)
        xs.append(xs[0])
        ys.append(ys[0])
        plt.plot(xs, ys)
    plt.show()


def generate_random_polygon(
    centre_point: Point2D,
    average_radius: float,
    irregularity: float,
    spikey_ness: float,
    n_vertices: int,
) -> list[Point2D]:
    """
    Notes
    -----
    Start with the centre of the polygon at ctrX, ctrY,
    then creates the polygon by sampling points on a circle around the centre.
    Random noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Parameters
    ----------
    :param centre_point: coordinates of the "centre" of the polygon
    :param average_radius: in px, the average radius of this polygon, this roughly controls how
            large the polygon is, really only useful for order of magnitude.
    :param irregularity: [0,1] indicating how much variance there is in the angular spacing of vertices.
            [0,1] will map to [0, 2pi/num_vertices]
    :param spikey_ness: [0,1] indicating how much variance there is in each vertex from the circle of radius average_radius.
            [0,1] will map to [0, average_radius]
    :param n_vertices: self-explanatory
    :return: a list of vertices, in CCW order.
    """

    irregularity = np.clip(irregularity, 0, 1) * tau / n_vertices
    spikey_ness = np.clip(spikey_ness, 0, 1) * average_radius

    angle_steps = np.random.uniform(
        (tau / n_vertices) - irregularity,
        (tau / n_vertices) + irregularity,
        n_vertices,
    )

    # normalize the steps so that point 0 and point n+1 are the same
    angle_steps = angle_steps / (angle_steps.sum() / tau)

    coordinates = []
    angle = np.random.uniform(0, tau)
    for i in range(n_vertices):
        r_i = np.clip(
            np.random.uniform(average_radius, spikey_ness), 0, 2 * average_radius
        )
        coordinates.append(
            Point2D(
                centre_point.x + r_i * np.cos(angle),
                centre_point.y + r_i * np.sin(angle),
            )
        )
        angle += angle_steps[i]
    return coordinates


def point_in_polygon(point: Point2D, polygon: Polygon) -> bool:
    if len(polygon) < 3:
        return False
    else:
        points = polygon[:] + [polygon[0]]
        in_polygon = False
        for prev_point, curr_point in zip(points, points[1:]):
            if (point.x == curr_point.x) and (point.y == curr_point.y):
                # point is a corner
                return True
            if (curr_point.y > point.y) != (prev_point.y > point.y):
                slope = (point.x - curr_point.x) * (prev_point.y - curr_point.y) - (
                    prev_point.x - curr_point.x
                ) * (point.y - curr_point.y)
                if slope == 0:
                    # point is on boundary
                    return True
                if (slope < 0) != (prev_point.y < curr_point.y):
                    in_polygon = not in_polygon
        return in_polygon


if __name__ == "__main__":
    polys = [
        Polygon(
            generate_random_polygon(
                Point2D(random.randint(0, 100), random.randint(0, 100)),
                average_radius=100,
                irregularity=0,
                spikey_ness=0,
                n_vertices=5,
            )
        )
        for _ in range(4)
    ]
    # plot_polygons(polys)

    square = Polygon([Point2D(0, 0), Point2D(1, 0), Point2D(1, 1), Point2D(0, 1)])
    plot_polygons([square])
    ans = point_in_polygon(Point2D(0.5, 0.5), square)
    # print(square)
    print(ans)
