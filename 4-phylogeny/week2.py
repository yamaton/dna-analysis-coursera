import collections
import itertools as it
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple
import numpy as np
import numpy.typing as npt
from week1 import EdgeWeight, Node, distance_between_leaves

Cluster = Tuple[FrozenSet[Node], Node]
Matrix = npt.ArrayLike


def sum_squared_descrepancy(edgeweight: EdgeWeight, dist_mat: Matrix) -> int:
    """Sum squared distance between the dist_mat and the other inferred from a graph.
    Note that only non-diagonal upper triangle (row_i < col_j) is considered.
    """
    n_leaves = len(dist_mat)
    other_mat = distance_between_leaves(edgeweight, n_leaves)
    mat1 = np.triu(dist_mat, k=1)
    mat2 = np.triu(other_mat, k=1)
    return ((mat1 - mat2) ** 2).sum()


def upgma(dist_mat: Matrix) -> EdgeWeight:
    """UPGMA (Unweighted Pair Group Method with Arithmetic Mean)
    https://en.wikipedia.org/wiki/UPGMA

    >>> mat = [[ 0, 20, 17, 11], [20,  0, 20, 13], [17, 20,  0, 10], [11, 13, 10,  0]]
    >>> computed = upgma(mat)
    >>> expected = {(0, 5): 7.0, (5, 0): 7.0, (1, 6): 8.833, (6, 1): 8.833, (2, 4): 5.0, (4, 2): 5.0, (3, 4): 5.0, (4, 3): 5.0, (4, 5): 2.0, (5, 4): 2.0, (5, 6): 1.833, (6, 5): 1.833}
    >>> all(np.isclose(computed[k], expected[k], atol=0.001) for k in expected) and computed.keys() == expected.keys()
    True
    """
    dist_mat = np.array(dist_mat, dtype=float)

    n = len(dist_mat)  # leaves are represented by 0, 1, ..., (n - 1)
    assert dist_mat.shape == (n, n), f"{dist_mat}"

    # A cluster is represented by a set of leaves AND its root node
    clusters = [(frozenset([i]), i) for i in range(n)]
    edgeweight = dict()
    # age is the distance from the node to the leaves underneath
    # choice of leaves does not matter due to ultrametricity
    age = {i: 0.0 for i in range(n)}
    next_node = n

    while len(clusters) > 1:
        idx1, idx2 = _find_closest_pair(dist_mat)
        dist_mat, clusters, new_edgeweight, age = _merge(
            idx1, idx2, dist_mat, clusters, age, next_node
        )
        edgeweight.update(new_edgeweight)
        next_node += 1

    return edgeweight


def _find_closest_pair(dist_mat: Matrix) -> Tuple[int, int]:
    size = len(dist_mat)
    return min(
        ((i, j) for i in range(size) for j in range(i + 1, size)),
        key=lambda pair: dist_mat[pair],
    )


def _merge(
    i: int,
    j: int,
    dist_mat: Matrix,
    clusters: Iterable[Cluster],
    age: Dict[Node, float],
    next_node: Node,
) -> Tuple[Matrix, Iterable[Cluster], EdgeWeight, Dict[Node, float]]:

    nrows, ncols = dist_mat.shape
    assert nrows == ncols

    dist_between_clusters = dist_mat[i, j]
    xs, x_root = clusters[i]
    ys, y_root = clusters[j]
    dist_from_rest = [
        (dist_mat[k][i] * len(xs) + dist_mat[k][j] * len(ys)) / (len(xs) + len(ys))
        for k in range(nrows)
        if k != i and k != j
    ]

    clusters_new = [cluster for k, cluster in enumerate(clusters) if k != i and k != j]
    acluster = (xs | ys, next_node)
    clusters_new.append(acluster)
    assert len(clusters) == len(clusters_new) + 1

    age[next_node] = dist_between_clusters / 2
    ew = dict()
    ew[x_root, next_node] = ew[next_node, x_root] = age[next_node] - age[x_root]
    ew[y_root, next_node] = ew[next_node, y_root] = age[next_node] - age[y_root]

    mat = np.zeros((nrows - 1, ncols - 1), dtype=float)
    mat[:-1, :-1] = np.delete(np.delete(dist_mat, (i, j), axis=0), (i, j), axis=1)
    mat[-1, :-1] = dist_from_rest
    mat[:-1, -1] = dist_from_rest
    return mat, clusters_new, ew, age


def to_edges(edgeweight: EdgeWeight) -> List[str]:
    return [(a, b, f"{a}->{b}:{w:0.3f}") for (a, b), w in edgeweight.items()]


def neighbor_joining(dist_mat: Matrix) -> EdgeWeight:
    """
    >>> mat = [[0, 23, 27, 20], [23, 0, 30, 28], [27, 30, 0, 30], [20, 28, 30, 0]]
    >>> expected = {(0, 4): 8.000, (1, 5): 13.500, (2, 5): 16.500, (3, 4): 12.000, (4, 5): 2.000, (4, 0): 8.000, (4, 3): 12.000, (5, 1): 13.500, (5, 2): 16.500, (5, 4): 2.000}
    >>> computed = neighbor_joining(mat)
    >>> all(np.isclose(computed[k], expected[k], atol=0.001) for k in expected) and computed.keys() == expected.keys()
    True
    """
    ...


def q_matrix(mat: Matrix, set_diagonal: int=0) -> Matrix:
    """
    >>> mat = [[0, 5, 9, 9, 8], [5, 0, 10, 10, 9], [9, 10, 0, 8, 7], [9, 10, 8, 0, 3], [8, 9, 7, 3, 0]]
    >>> mat = np.asarray(mat)
    >>> expected = np.array([[0, -50, -38, -34, -34], [-50, 0, -38, -34, -34], [-38, -38, 0, -40, -40], [-34, -34, -40, 0, -48], [-34, -34, -40, -48, 0]])
    >>> computed = q_matrix(mat)
    >>> (expected == computed).all()
    True
    """
    n, m = mat.shape
    assert n == m
    res = (
        (n - 2) * mat - mat.sum(axis=0, keepdims=True) - mat.sum(axis=1, keepdims=True)
    )
    np.fill_diagonal(res, set_diagonal)
    return res


if __name__ == "__main__":
    n = int(input())
    mat = [[int(c) for c in input().split()] for line in range(n)]
    edgeweight = upgma(mat)
    edges_as_str = sorted(to_edges(edgeweight))
    for _, _, s in edges_as_str:
        print(s)
