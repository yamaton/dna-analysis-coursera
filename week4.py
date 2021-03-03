from typing import Iterable, Set, List, Tuple
import itertools as it
import sys
import functools
import operator
import collections
import random
from utils import hamming_distance, neighbors, sliding_window
from week3 import motifs2profile_laplace, most_probable_kmer, score


Profile = List[Tuple[float, float, float, float]]
Motifs = List[str]

to_profile = motifs2profile_laplace


def to_motifs(profile: Profile, dna: Iterable[str]) -> Motifs:
    k = len(profile[0])
    return [most_probable_kmer(seq, k, profile) for seq in dna]


def _randomized_motif_search_core(
    dna: Iterable[str], k: int, t: int
) -> Tuple[int, Motifs]:
    n = len(dna[0])
    motifs = []
    for seq in dna:
        i = random.randrange(0, n - k + 1)
        motif = seq[i : i + k]
        motifs.append(motif)
    best_motifs = motifs
    best_score = 10000000

    while True:
        profile = to_profile(motifs)
        motifs = to_motifs(profile, dna)
        if score(motifs) < best_score:
            best_score = score(motifs)
            best_motifs = motifs
        else:
            return best_score, best_motifs


def gibbs_sampler(
    dna: Iterable[str], k: int, t: int, repeat: int
) -> Tuple[int, Motifs]:
    """
    >>> dna = ["CGCCCCTCTCGGGGGTGTTCAGTAAACGGCCA", "GGGCGAGGTATGTGTAAGTGCCAAGGTGCCAG", "TAGTACCGAGACCGAAAGAAGTATACAGGCGT", "TAGATCAAGTTTCAGGTGCACGTCGGTGAACC", "AATCCACCAGCTCCACGTGCAATGTTGGCCTA"]
    >>> gibbs_sampler(dna, 8, 5, 100)
    ['TCTCGGGG', 'CCAAGGTG', 'TACAGGCG', 'TTCAGGTG', 'TCCACGTG']
    """
    n = len(dna[0])
    motifs = []
    for seq in dna:
        i = random.randrange(0, n - k + 1)
        motif = seq[i : i + k]
        motifs.append(motif)

    best_motifs = motifs
    for _ in range(repeat):
        idx = random.randrange(t)
        rest = [m for i, m in enumerate(motifs) if i != idx]
        profile = to_profile(rest)
        motifs[idx] = most_probable_kmer(dna[idx], k, profile)
        if score(motifs) < score(best_motifs):
            best_motifs = motifs
    return best_motifs


def randomized_motif_search(dna: Iterable[str], k: int, t: int) -> Motifs:
    """
    >>> dna = ["CGCCCCTCTCGGGGGTGTTCAGTAAACGGCCA", "GGGCGAGGTATGTGTAAGTGCCAAGGTGCCAG", "TAGTACCGAGACCGAAAGAAGTATACAGGCGT", "TAGATCAAGTTTCAGGTGCACGTCGGTGAACC", "AATCCACCAGCTCCACGTGCAATGTTGGCCTA"]
    >>> randomized_motif_search(dna, 8, 5)
    ['TCTCGGGG', 'CCAAGGTG', 'TACAGGCG', 'TTCAGGTG', 'TCCACGTG']
    """
    score, motifs = min(_randomized_motif_search_core(dna, k, t) for _ in range(1000))
    return motifs


if __name__ == "__main__":
    k, t = map(int, input().split())
    dna = [input().strip() for _ in range(t)]
    res = randomized_motif_search(dna, k, t)
    print(*res)
