from typing import Iterable, Set, List, Tuple
import itertools as it
import sys
import functools
import operator
import random
from utils import sliding_window
from week3 import motifs2profile_laplace, most_probable_kmer, score


Profile = List[Tuple[float, float, float, float]]
Motifs = List[str]

to_profile = motifs2profile_laplace


def to_motifs(profile: Profile, dna: Iterable[str]) -> Motifs:
    k = len(profile[0])
    return [most_probable_kmer(seq, k, profile) for seq in dna]


def sample_kmer_with_weighted_random(seq: str, k: int, profile: Profile) -> str:
    nuc2idx = dict(zip("ACGT", range(4)))
    assert len(profile) == 4
    assert len(profile[0]) == k

    # more likely to choose when a kmer appears multiple times. Is it ok?
    kmers = ["".join(cs) for cs in sliding_window(seq, k)]
    probs = []
    for kmer in kmers:
        indices = [nuc2idx[c] for c in kmer]
        iter = (ps[i] for i, ps in zip(indices, zip(*profile)))
        p = functools.reduce(operator.mul, iter)
        probs.append(p)

    return random.choices(kmers, weights=probs)[0]


def sample_kmer_with_weighted_random2(seq: str, k: int, profile: Profile) -> str:
    nuc2idx = dict(zip("ACGT", range(4)))
    assert len(profile) == 4
    assert len(profile[0]) == k

    # ignore duplicate appearences of kmer in seq
    kmer_prob_map = dict()
    iter = sliding_window(seq, k)
    iter = ("".join(cs) for cs in iter)
    for kmer in iter:
        if kmer in kmer_prob_map:
            continue
        indices = [nuc2idx[c] for c in kmer]
        iter = (ps[i] for i, ps in zip(indices, zip(*profile)))
        p = functools.reduce(operator.mul, iter)
        kmer_prob_map[kmer] = p

    kmers = list(kmer_prob_map.keys())
    return random.choices(kmers, weights=kmer_prob_map.values())[0]


def _randomized_motif_search_unit(
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
            best_motifs = motifs[:]
        else:
            return best_score, best_motifs


def randomized_motif_search(dna: Iterable[str], k: int, t: int) -> Motifs:
    """
    >>> dna = ["CGCCCCTCTCGGGGGTGTTCAGTAAACGGCCA", "GGGCGAGGTATGTGTAAGTGCCAAGGTGCCAG", "TAGTACCGAGACCGAAAGAAGTATACAGGCGT", "TAGATCAAGTTTCAGGTGCACGTCGGTGAACC", "AATCCACCAGCTCCACGTGCAATGTTGGCCTA"]
    >>> randomized_motif_search(dna, 8, 5)
    ['TCTCGGGG', 'CCAAGGTG', 'TACAGGCG', 'TTCAGGTG', 'TCCACGTG']
    """
    score, motifs = min(_randomized_motif_search_unit(dna, k, t) for _ in range(1000))
    return score, motifs


def gibbs_sampler(
    dna: Iterable[str], k: int, t: int, repeat: int
) -> Tuple[int, Motifs]:
    """
    >>> dna = ["CGCCCCTCTCGGGGGTGTTCAGTAAACGGCCA", "GGGCGAGGTATGTGTAAGTGCCAAGGTGCCAG", "TAGTACCGAGACCGAAAGAAGTATACAGGCGT", "TAGATCAAGTTTCAGGTGCACGTCGGTGAACC", "AATCCACCAGCTCCACGTGCAATGTTGGCCTA"]
    >>> gibbs_sampler(dna, 8, 5, 1000)
    ['TCTCGGGG', 'CCAAGGTG', 'TACAGGCG', 'TTCAGGTG', 'TCCACGTG']
    """
    n = len(dna[0])
    motifs = []
    for seq in dna:
        i = random.randint(0, n - k)
        motif = seq[i : i + k]
        motifs.append(motif)

    best_motifs = motifs
    for _ in range(repeat):
        idx = random.randrange(0, t)
        rest = [m for i, m in enumerate(motifs) if i != idx]
        profile = to_profile(rest)
        motifs[idx] = sample_kmer_with_weighted_random2(dna[idx], k, profile)
        if score(motifs) < score(best_motifs):
            best_motifs = motifs[:]
    return score(best_motifs), best_motifs


if __name__ == "__main__":
    k, t = map(int, input().split())
    dna = [input().strip() for _ in range(t)]
    res = randomized_motif_search(dna, k, t)
    print(*res)
