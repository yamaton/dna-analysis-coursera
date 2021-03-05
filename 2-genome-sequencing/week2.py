import collections
import itertools as it
from typing import Dict, Hashable, Iterable, List, Set
from utils import windowed

AdjacencySet = Dict[Hashable, Set[Hashable]]


def eulerian_cycle(g: AdjacencySet) -> List[Hashable]:
    """
    >>> g = {0: {3}, 1: {0}, 2: {1, 6}, 3: {2}, 4: {2}, 5: {4}, 6: {5, 8}, 7: {9}, 8: {7}, 9: {6}}
    >>> res = eulerian_cycle(g)
    >>> res
    True
    """
    n = sum(len(v) for v in g.values())  # number of remaining edges

    x = next(iter(g))
    traj = [x]
    while g[x]:
        x = g[x].pop()
        traj.append(x)
        n -= 1

    while n > 0:
        # find new start point in cycle
        traj.pop()
        x = next(x for x in traj if g[x])
        idx = traj.index(x)
        # reorder cycle such that the new start site comes to the last
        traj = list(it.islice(it.chain(traj[idx:], traj), len(traj)))
        traj.append(x)

        while g[x]:
            x = g[x].pop()
            traj.append(x)
            n -= 1

    assert n == 0  # sanity check!

    return traj


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        pairs = [line.strip().split(" -> ") for line in f.readlines()]
        g = dict()
        for k, s in pairs:
            g[k] = set(s.split(","))

    res = eulerian_cycle(g)
    print("->".join(res))
