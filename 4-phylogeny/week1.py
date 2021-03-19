from typing import Dict, Hashable, Iterable, List, Set, Tuple
import collections
import functools
import operator

Node = Hashable
Edge = Tuple[Node, Node]
EdgeWeight = Dict[Edge, int]
AdjList = Dict[Node, List[Node]]

Grid = List[List[int]]


def parse_weighted_edges(ss: Iterable[str]) -> EdgeWeight:
    """
    >>> parse_weighted_edges(["0->4:11", "1->4:2"])
    {(0, 4): 11, (1, 4): 2}
    """
    d = dict()
    for s in ss:
        edge_str, w_str = s.split(":")
        from_str, to_str = edge_str.split("->")
        from_, to_, w = map(int, (from_str, to_str, w_str))
        tup = (from_, to_)
        d[tup] = w
    return d


def distance_between_leaves(edgeweight: EdgeWeight, n_leaves: int) -> Grid:
    """Compute distances between leaves with DFS from each leaf

    >>> weight = {(0, 4): 11, (1, 4): 2, (2, 5): 6, (3, 5): 7, (4, 0): 11, (4, 1): 2, (4, 5): 4, (5, 4): 4, (5, 3): 7, (5, 2): 6}
    >>> distance_between_leaves(weight, 4)
    [[0, 13, 21, 22], [13, 0, 12, 13], [21, 12, 0, 13], [22, 13, 13, 0]]
    """
    g = get_adjlist(edgeweight)
    leaves = get_leaves(g)
    assert len(leaves) == n_leaves

    ds = []
    for leaf in leaves:
        d = dict()
        stack = [(leaf, 0)]
        seen = set()
        while stack:
            n, acc = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if n in leaves:
                d[leaf, n] = acc
            for n_next in g[n]:
                pair = n_next, acc + edgeweight[n, n_next]
                stack.append(pair)
        ds.append(d)

    dist_acc = {}
    for d in ds:
        dist_acc.update(d)
    dist_mat = [[dist_acc[x, y] for x in leaves] for y in leaves]
    return dist_mat


def get_adjlist(edgeweight: EdgeWeight) -> AdjList:
    g = collections.defaultdict(list)
    for (a, b) in edgeweight:
        g[a].append(b)
    return g


def get_leaves(g: AdjList) -> Set[Node]:
    return {k for k in g if len(g[k]) == 1}


def limb_length(mat: Grid, n: int, j: int) -> int:
    """Compute limb length of the node j using the limb length theorem

    The computational complexity is O(n^2)

    >>> mat = [[0, 13, 21, 22], [13, 0, 12, 13], [21, 12, 0, 13], [22, 13, 13, 0]]
    >>> limb_length(mat, 4, 1)
    2
    """
    assert len(mat) == len(mat[0]) == n
    d = min(
        (mat[i][j] + mat[j][k] - mat[i][k]) // 2
        for i in range(n)
        for k in range(i + 1, n)
        if i != j and k != j
    )
    return d


def limb_length_faster(mat: Grid, n: int, j: int) -> int:
    """Compute limb length using the limb length theorem

    The computational complexity is O(n)

    >>> mat = [[0, 13, 21, 22], [13, 0, 12, 13], [21, 12, 0, 13], [22, 13, 13, 0]]
    >>> limb_length_faster(mat, 4, 1)
    2
    """
    assert len(mat) == len(mat[0]) == n
    i = (j + 1) % n
    d = min(
        (mat[i][j] + mat[j][k] - mat[i][k]) // 2
        for k in range(n)
        if k != i and k != j
    )
    return d


def additive_phylogeny(mat: Grid) -> EdgeWeight:
    """
    >>> mat = [[0, 13, 21, 22], [13, 0, 12, 13], [21, 12, 0, 13], [22, 13, 13, 0]]
    >>> weight = {(0, 4): 11, (1, 4): 2, (2, 5): 6, (3, 5): 7, (4, 0): 11, (4, 1): 2, (4, 5): 4, (5, 4): 4, (5, 3): 7, (5, 2): 6}
    >>> additive_phylogeny(mat) == weight
    True
    """
    n = len(mat)
    if n == 2:
        return {(0, 1): mat[0][1], (1, 0): mat[1][0]}
    limb_len = limb_length_faster(mat, n, n - 1)
    for j in range(n - 1):
        mat[j][n - 1] -= limb_len
        mat[n - 1][j] = mat[j][n - 1]
    i, k = next((i, k) for i in range(n - 1) for k in range(i + 1, n - 1) if mat[i][k] == mat[i][n - 1] + mat[n - 1][k])
    x = mat[i][n - 1]
    mat = [line[:n] for line in mat[:n]]
    tree = additive_phylogeny(mat)






if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()
    with open(args.input, "r") as f:
        n = int(f.readline())
        j = int(f.readline())
        mat = [[int(s) for s in line.split()] for line in f.readlines()]
    res = limb_length_faster(mat, n, j)
    print(res)
