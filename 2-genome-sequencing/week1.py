import collections
import itertools as it
from typing import Dict, Iterable, List, Set
import argparse
from utils import windowed

AdjacencySet = Dict[str, Set[str]]
AdjacencyList = Dict[str, Set[str]]


def genome_path(shards: Iterable[str]) -> str:
    """Just merge sliding-window sequences
    >>> genome_path(["ACCGA", "CCGAA", "CGAAG", "GAAGC", "AAGCT"])
    'ACCGAAGCT'
    """
    gen = iter(shards)
    res = [next(gen)]
    for s in gen:
        res.append(s[-1])
    return "".join(res)


def graph_overlap(shards: Iterable[str]) -> AdjacencySet:
    """Create a dictionary such that each key's suffix matches prefix of values.
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


def to_debruijn(text: str, k: int) -> AdjacencyList:
    """
    >>> computed = to_debruijn("AAGATTCTCTAAGA", 4)
    >>> expected = {"AAG":["AGA","AGA"], "AGA":["GAT"], "ATT":["TTC"], "CTA":["TAA"], "CTC":["TCT"], "GAT":["ATT"], "TAA":["AAG"], "TCT":["CTA", "CTC"], "TTC":["TCT"]}
    >>> computed == expected
    True
    """
    d = collections.defaultdict(list)
    w1, w2 = it.tee(map(lambda cs: "".join(cs), windowed(text, k - 1)))
    next(w2)
    for this, that in zip(w1, w2):
        d[this].append(that)

    return {k: sorted(d[k]) for k in sorted(d.keys())}


def to_debruijn_from_kmers(kmers: Iterable[str]) -> AdjacencyList:
    """Create de Bruijn graph from a set of k-mers

    >>> expected = {"AGG": ["GGG"],"CAG": ["AGG", "AGG"],"GAG": ["AGG"],"GGA": ["GAG"],"GGG": ["GGA","GGG"]}
    >>> res = to_debruijn_from_kmers(["GAGG", "CAGG", "GGGG", "GGGA", "CAGG", "AGGG", "GGAG"])
    >>> res == expected
    True
    """
    d = collections.defaultdict(list)
    for kmer in kmers:
        d[kmer[:-1]].append(kmer[1:])
    return {k: sorted(d[k]) for k in sorted(d.keys())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        kmers = [line.strip() for line in f.readlines()]

    res = to_debruijn_from_kmers(kmers)
    for k, v in res.items():
        print(f"{k} -> {','.join(v)}")
