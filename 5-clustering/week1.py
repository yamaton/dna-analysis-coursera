from typing import List
import numpy as np
import numpy.typing as npt


def farthest_first_traversal(data: npt.ArrayLike, k=int) -> npt.ArrayLike:
    """
    >>> data = [[0.0, 0.0], [5.0, 5.0], [0.0, 5.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [1.0, 2.0]]
    >>> computed = farthest_first_traversal(data, 3)
    >>> expected = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    >>> np.allclose(computed, expected)
    True
    """
    data = np.array(data)
    idx = 0
    centers = data[idx][None, :]
    while len(centers) < k:
        diff = data[:, :, None] - centers.T[None, :, :]
        idx = np.linalg.norm(diff, axis=1).min(axis=-1).argmax()
        centers = np.append(centers, [data[idx]], axis=0)

    return centers


def squared_error_distortion(data: npt.ArrayLike, centers: npt.ArrayLike) -> float:
    """
    >>> centers = np.array([[2.31, 4.55], [5.96, 9.08]])
    >>> data = np.array([[3.42, 6.03], [6.23, 8.25], [4.76, 1.64], [4.47, 4.33], [3.95, 7.61], [8.93, 2.97], [9.74, 4.03], [1.73, 1.28], [9.72, 5.01], [7.27, 3.77]])
    >>> computed = squared_error_distortion(data, centers)
    >>> np.isclose(computed, 18.246, atol=0.001)
    True
    """
    assert data.shape[1] == centers.shape[1]
    diff = data[:, :, None] - centers.T[None, :, :]
    res = (np.linalg.norm(diff, axis=1).min(axis=-1) ** 2).mean()
    return res


def lloyd(data: npt.ArrayLike, k: int, use_initializer=False) -> npt.ArrayLike:
    """Returns a set of centers consisting of k points

    >>> data = [[1.3, 1.1], [1.3, 0.2], [0.6, 2.8], [3.0, 3.2], [1.2, 0.7], [1.4, 1.6], [1.2, 1.0], [1.2, 1.1], [0.6, 1.5], [1.8, 2.6], [1.2, 1.3], [1.2, 1.0], [0.0, 1.9]]
    >>> expected = [[1.800, 2.867], [1.060, 1.140]]
    >>> data = np.array(data)
    >>> computed = lloyd(data, 2)
    >>> np.allclose(computed, expected, atol=0.001)
    True
    """
    data = np.array(data)

    # take first k points as tentative centers
    centers = data[:k, :]
    if use_initializer:
        centers = kmeans_initializer(data, k)
    centers_prev = np.zeros_like(centers)
    while not np.allclose(centers_prev, centers):
        # from centers to clusters
        clusters = _lloyd_clusters(data, centers)
        assert len(clusters) == k
        centers_prev = centers
        # from clusters to centers
        centers = np.array([cluster.mean(axis=0) for cluster in clusters])
    return centers


def _lloyd_clusters(data: npt.ArrayLike, centers: npt.ArrayLike) -> List[npt.ArrayLike]:
    """Returns clusters as list of points (_, dim)"""
    k, _ = centers.shape
    # distances has the shape (num_points, k)
    diff = data[:, :, None] - centers.T[None, :, :]
    distances: npt.ArrayLike = np.linalg.norm(diff, axis=1)
    # indices shows which cluster each point belongs to
    # it has the form [0, 1, ..., (k - 1)] of length `num_points`
    indices = distances.argmin(axis=-1)
    clusters = [data[indices == i, :] for i in range(k)]
    return clusters


def kmeans_initializer(data: npt.ArrayLike, k: int) -> npt.ArrayLike:
    num_points, _ = data.shape
    idx = np.random.choice(num_points)
    seen = {idx}
    centers = data[[idx], :]
    while len(centers) < k:
        diff = data[:, :, None] - centers.T[None, :, :]
        d2 = np.linalg.norm(diff, axis=1).min(axis=-1) ** 2
        p = d2 / d2.sum()
        idx = np.random.choice(num_points, p=p)
        while idx in seen:
            idx = np.random.choice(num_points, p=p)
        centers = np.append(centers, [data[idx]], axis=0)
        seen.add(idx)

    return np.array(centers)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()
    with open(args.input, "r") as f:
        k, m = map(int, f.readline().split())
        data = np.array([[float(s) for s in line.split()] for line in f.readlines()])
    res = lloyd(data, k, use_initializer=True)
    for xs in res:
        print(*xs)
