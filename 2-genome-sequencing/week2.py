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
    """Pick a Eulerian path from graph `g`.

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
    assert len(patterns[0]) == k
    db = to_debruijn_from_kmers(patterns)
    path = eulerian_path(db)
    text = genome_path(path)
    return text


def k_universal_circular_string(k: int) -> str:
    """
    >>> k_universal_circular_string(4)
    '0000110010111101'
    """
    patterns = ["".join(cs) for cs in it.product("01", repeat=k)]
    res = string_reconstruction(patterns, k)
    # It's a matter of choice weather to include the origin/goal in a cycle representation
    return res[: -(k - 1)]


def to_kdmers(s: str, k: int, d: int) -> List[str]:
    """
    >>> to_kdmers("TAATGCCATGGGATGTT", 3, 2)
    ['AAT|CAT', 'ATG|ATG', 'ATG|ATG', 'CAT|GAT', 'CCA|GGA', 'GCC|GGG', 'GGG|GTT', 'TAA|CCA', 'TGC|TGG', 'TGG|TGT']
    """
    intervals1 = map(lambda cs: "".join(cs), windowed(s[: -(k + d)], k))
    intervals2 = map(lambda cs: "".join(cs), windowed(s[k + d :], k))
    return sorted(["|".join(pair) for pair in zip(intervals1, intervals2)])


# def string_reconstruction_from_paired_reads(
#     reads: Iterable[str], k: int, d: int
# ) -> str:
#     """
#     >>> reads = ["GAGA|TTGA", "TCGT|GATG", "CGTG|ATGT", "TGGT|TGAG", "GTGA|TGTT", "GTGG|GTGA", "TGAG|GTTG", "GGTC|GAGA", "GTCG|AGAT"]
#     >>> string_reconstruction_from_paired_reads(reads, 4, 2)
#     'GGCTTACCA'
#     """
#     assert len(reads[0]) == 2 * k + 1
#     db = to_paired_debruijn(reads)
#     paths = all_eulerian_paths(db)
#     for path in paths:
#         text = paired_genome_path(path)
#         if text is not None:
#             return text

#     raise ValueError("Something wrong?")


def to_paired_debruijn(patterns: Iterable[str]) -> AdjacencyList:
    d = collections.defaultdict(list)
    for pair in patterns:
        read1, read2 = pair.split("|")
        key = f"{read1[:-1]}|{read2[:-1]}"
        value = f"{read1[1:]}|{read2[1:]}"
        d[key].append(value)
    return {k: sorted(d[k]) for k in sorted(d.keys())}


def paired_genome_path(shards: List[str], d: int) -> Optional[str]:
    """Just merge sliding-window sequences from paired reads

    >>> paired_genome_path(["GACC|GCGC", "ACCG|CGCC",  "CCGA|GCCG",  "CGAG|CCGG", "GAGC|CGGA"], 2)
    'GACCGAGCGCCGGA'
    """
    k = (len(shards[0]) - 1) // 2
    reads1, reads2 = zip(*[pair.split("|") for pair in shards])
    gen = iter(reads1)
    res = [next(gen)]
    for s in gen:
        res.append(s[-1])
    s1 = "".join(res)

    gen = iter(reads2)
    res = [next(gen)]
    for s in gen:
        res.append(s[-1])
    s2 = "".join(res)

    if s1[k + d :] != s2[: -(k + d)]:
        return None
    return s1[: k + d] + s2


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        k, d = map(int, f.readline().split())
        shards = [line.strip() for line in f.readlines()]

    assert len(shards[0]) == 2 * k + 1
    res = paired_genome_path(shards, d)
    print(res)
