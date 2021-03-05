import collections
import functools
import operator
import itertools as it
from typing import Dict, Hashable, Iterable, List, Optional, Set, Tuple
from week1 import genome_path, to_debruijn_from_kmers
from utils import windowed

AdjacencySet = Dict[Hashable, Set[Hashable]]
AdjacencyList = Dict[Hashable, List[Hashable]]
Trajectory = List[Hashable]


def eulerian_cycle(g: AdjacencyList) -> Trajectory:
    """
    >>> g = {0: [3], 1: [0], 2: [1, 6], 3: [2], 4: [2], 5: [4], 6: [5, 8], 7: [9], 8: [7], 9: [6]}
    >>> res = eulerian_cycle(g)
    >>> res
    True
    """
    g = collections.defaultdict(set, g)

    n = sum(len(v) for v in g.values())  # number of remaining edges

    x = next(iter(g))
    path = [x]
    g, path, n = _get_path(g, path, n, x)

    while n > 0:
        # find new start point in cycle
        assert path[0] == path[-1]
        path.pop()
        x = next(x for x in path if g[x])
        idx = path.index(x)
        # reorder cycle such that the new start site comes to the last
        path = list(it.islice(it.chain(path[idx:], path), len(path)))
        path.append(x)

        g, path, n = _get_path(g, path, n, x)

    assert n == 0  # sanity checking!

    return path


def _get_path(
    g: AdjacencySet,
    traj: Trajectory,
    n: int,
    start: Hashable,
) -> Tuple[AdjacencySet, Trajectory, int]:
    x = start
    while (x in g) and g[x]:
        x = g[x].pop()
        traj.append(x)
        n -= 1
    return g, traj, n


def eulerian_path(g: AdjacencySet) -> List[Hashable]:
    """
    >>> g = {0: [2], 1: [3], 2: [1], 3: [0, 4], 6: [3, 7], 7: [8], 8: [9], 9: [6]}
    >>> eulerian_path(g)
    [6, 7, 8, 9, 6, 3, 0, 2, 1, 3, 4]
    """
    nodes = set(g.keys()) | functools.reduce(operator.or_, map(set, g.values()))
    out_degree = {node: len(g.get(node, frozenset())) for node in nodes}
    in_degree = collections.Counter(x for xs in g.values() for x in xs)

    candidates = [node for node in nodes if in_degree[node] != out_degree[node]]

    # process as eulerian cycle if the graph is all balanced
    if not candidates:
        return eulerian_cycle(g)

    g = collections.defaultdict(set, g)
    assert len(candidates) == 2  # assert the graph is nearly balanced
    start = next(node for node in candidates if in_degree[node] < out_degree[node])
    goal = next(node for node in candidates if in_degree[node] > out_degree[node])
    n = sum(out_degree.values())  # remaining edge count to traverse
    assert sum(len(ss) for ss in g.values()) == n

    path = [start]
    g, path, n = _get_path(g, path, n, start)
    assert path[-1] == goal
    assert sum(len(ss) for ss in g.values()) == n

    while n > 0:
        i, x = next((i, x) for i, x in enumerate(path) if g[x])
        g, cycle, n = _get_path(g, [x], n, x)

        assert sum(len(ss) for ss in g.values()) == n
        assert cycle[0] == cycle[-1]
        path = path[:i] + cycle + path[i + 1 :]

    return path


def string_reconstruction(patterns: Iterable[str], k: int) -> str:
    """
    >>> patterns = ["CTTA", "ACCA", "TACC", "GGCT", "GCTT", "TTAC"]
    >>> string_reconstruction(patterns, 4)
    'GGCTTACCA'
    """
    db = to_debruijn_from_kmers(patterns)
    path = eulerian_path(db)
    text = genome_path(path)
    return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        k = int(f.readline())
        patterns = [line.strip() for line in f.readlines()]

    res = string_reconstruction(patterns, k)
    print(res)
