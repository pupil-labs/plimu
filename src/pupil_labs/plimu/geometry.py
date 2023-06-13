import numpy as np
import more_itertools


def angle_between(v1, v2, deg=False):
    if deg:
        return np.degrees(
            np.arccos(
                np.clip(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), 0, 1
                ),
            )
        )
    else:
        return np.arccos(
            np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))), 0, 1
        )


def get_unit_circle_polygon(n=100):
    return [[np.cos(i), np.sin(i)] for i in np.linspace(0, 2 * np.pi, n)]


def get_star_polygon():
    star_outer = [[np.cos(i), np.sin(i)] for i in np.linspace(0, 2 * np.pi, 6)[:-1]]
    star_inner = [
        [0.4 * np.cos(i + 2 * np.pi / 10), 0.4 * np.sin(i + 2 * np.pi / 10)]
        for i in np.linspace(0, 2 * np.pi, 6)[:-1]
    ]
    star_points = list(more_itertools.roundrobin(star_outer, star_inner))
    return star_points
