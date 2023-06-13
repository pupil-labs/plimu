import os
from pathlib import Path

import numpy as np
from svgpathtools import svg2paths


resource_path = Path(__file__).parent.parent / "resources"


def read_path_from_svg(file_path, n_samples=100, normalize=True):
    path_svg = svg2paths(file_path)[0][0]
    path = []
    for s in np.linspace(0, 1, n_samples):
        path.append([path_svg.point(s).real, path_svg.point(s).imag])
    path = np.asarray(path)
    if normalize:
        path = normalize_path(path)
    return path


def get_letter_from_svg(x):
    filename = resource_path / f"{x}.svg"
    if not os.path.isfile(filename):
        raise ValueError(f"SVG for letter {x} does not exist.")
    path = read_path_from_svg(filename)
    normalized_path = normalize_path(path)
    return normalized_path


def get_pl_logo_from_svg():
    paths_from_svg, _ = svg2paths(resource_path / "Primary-L-Black.svg")

    logo_paths = []
    for idx in range(len(paths_from_svg) - 1):
        path = []
        for s in np.linspace(0, 1, 200):
            path.append(
                [paths_from_svg[idx].point(s).real, paths_from_svg[idx].point(s).imag]
            )
        path = np.asarray(path)
        path = simplify_path_with_jump(path, 10)
        path -= np.asarray([68.0, 68.0])
        path /= 30
        logo_paths.append(path)

    return logo_paths


def normalize_path(points):
    points = points - np.mean(points, axis=0)
    points = points / np.max(np.abs(points), axis=0)
    return points


def simplify_path_with_jump(path, threshold=5):
    deltas = np.linalg.norm(path[1:, :] - path[:-1, :], axis=1)
    jump_size = np.max(deltas)
    if jump_size > threshold:  # path is not simple enough, but has a jump in it
        sep_idx = np.argmax(deltas) + 1  # index of the point where the jump occurs
        path_1 = path[:sep_idx, :]  # first part of the path
        path_2 = path[sep_idx:, :]  # second part of the path
        idx_1 = np.argmin(
            np.linalg.norm(path_1 - path_2[0], axis=1)
        )  # index of the point in the first part of the path that is closest to the first point of the second part of the path
        points_1_prime = np.concatenate(
            [path_1[idx_1:, :], path_1[:idx_1, :]],
            axis=0,  # reorder the first part of the path so that the last point of the first part is the first point
        )
        points_full = np.concatenate(
            [path_2, path_2[:1], points_1_prime, points_1_prime[:1], path_2[:1]],
            axis=0,
        )  # concatenate the two parts of the path
        return points_full
    else:
        return path
