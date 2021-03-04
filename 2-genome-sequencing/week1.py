import collections
from typing import Dict, Iterable, List, Set
import argparse
from utils import windowed

AdjacencyList = Dict[str, Set[str]]


def genome_path0(shards: Iterable[str]) -> str:
    """
    >>> genome_path0(["ACCGA", "CCGAA", "CGAAG", "GAAGC", "AAGCT"])
    'ACCGAAGCT'
    """
    gen = iter(shards)
    res = [next(gen)]
    for s in gen:
        res.append(s[-1])
    return "".join(res)


def graph_overlap(shards: Iterable[str]) -> AdjacencyList:
    """
    >>> expected = {"CATGC": {"ATGCG"}, "GCATG": {"CATGC"}, "GGCAT": {"GCATG"}, "AGGCA": {"GGCAC", "GGCAT"}}
    >>> computed = graph_overlap(["ATGCG", "GCATG", "CATGC", "AGGCA", "GGCAT", "GGCAC"])
    >>> computed == expected
    True
    """
    d = collections.defaultdict(set)
    for this in shards:
        for that in shards:
            if this == that:
                continue
            if this[1:] == that[:-1]:
                d[this].add(that)

    return dict(d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        shards = [s.strip() for s in f.readlines()]

    res = graph_overlap(shards)
    for k, v in res.items():
        print(f"{k} -> {','.join(v)}")
