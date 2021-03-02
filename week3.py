import sys
from typing import Collection, Iterable, Set, List, Tuple
import itertools as it
import functools
import operator
import collections
import numpy as np
from utils import hamming_distance, neighbors, sliding_window


def hamming_distance_str(pattern: str, text: str):
    assert len(pattern) <= len(text)
    return min(
        hamming_distance(pattern, seg) for seg in sliding_window(text, len(pattern))
    )


def hamming_distance_set(pattern: str, dna: Iterable[str]):
    """
    >>> dna = ["TTACCTTAAC", "GATATCTGTC", "ACGGCGTTCG", "CCCTAAAGAG", "CGTCAGAGGT"]
    >>> hamming_distance_set("AAA", dna)
    5
    """
    return sum(hamming_distance_str(pattern, text) for text in dna)


def motif_enumeration(dna: List[str], k: int, d: int) -> Set[str]:
    """Search for all (k, d)-motifs in dna with brute force.
    O(t * n * X * t * n)
        where t = len(dna)
              n = len(dna[0])
              X = time complexity of neighbors()

    >>> motif_enumeration({"ATTTGGC", "TGCCTTA", "CGGTATC", "GAAAATT"}, 3, 1) == {"ATA", "ATT", "GTT", "TTT"}
    True
    """
    seen = set()
    res = set()
    for seq in dna:
        for seed in sliding_window(seq, k):
            if seed in seen:
                continue
            seen.add(seed)
            for pattern in neighbors(seed, d):
                if all(hamming_distance_str(pattern, s) <= d for s in dna):
                    res.add(pattern)
    return res


def median_string(dna: List[str], k: int) -> str:
    """Find median string with brute force
    O(4^k * t * n * k)

    >>> dna = ["AAATTGACGCAT", "GACGACCACGTT", "CGTCAGCGCCTG", "GCTGAGCACCGG", "AGTTCGGGACAG"]
    >>> median_string(dna, 3)
    'GAC'
    """
    iter = it.product("ATGC", repeat=k)
    iter = map(lambda tup: "".join(tup), iter)
    res = min(iter, key=lambda kmer: hamming_distance_set(kmer, dna))
    return res


# Tuple values are of the order (A, C, G, T)
Profile = List[Tuple[float, float, float, float]]


def most_probable_kmer(text: str, k: int, profile: Profile) -> str:
    """Find the most likelihood kmer given probability distribution profile such that

    argmax_{kmer} P(kmer | profile)

    >>> text = "ACCTGTTTATTGCCTAAGTTCCGAACAAACCCAATATAGCCCGAGGGCCT"
    >>> profile = [[0.2, 0.2, 0.3, 0.2, 0.3], [0.4, 0.3, 0.1, 0.5, 0.1], [0.3, 0.3, 0.5, 0.2, 0.4], [0.1, 0.2, 0.1, 0.1, 0.2]]
    >>> most_probable_kmer(text, 5, profile)
    'CCGAG'
    """
    nuc2idx = dict(zip("ACGT", range(4)))
    assert len(profile) == 4
    assert len(profile[0]) == k

    max_sofar = -1000
    ans = None
    for kmer in sliding_window(text, k):
        indices = [nuc2idx[c] for c in kmer]
        iter = (ps[i] for i, ps in zip(indices, zip(*profile)))
        prob = functools.reduce(operator.mul, iter)
        if prob > max_sofar:
            max_sofar = prob
            ans = kmer

    if ans is None:
        raise ValueError("Something is wrong!")
    return "".join(ans)


def motifs2profile(motifs: List[str]) -> List[List[float]]:
    """Form profile (= probability distribution) based on observations called `motifs`.

    Returns array of shape (4, n) where n is len(motifs[0]).
    Note that position [0, 1, 2, 3] corresponds to ACGT.
    """
    assert all(len(motif) == len(motifs[0]) for motif in motifs)
    motifs = np.asarray(motifs)
    res = []
    for col in zip(*motifs):
        counter = collections.Counter(col)
        total = len(col)
        probs = [counter[c] / total for c in "ACGT"]
        res.append(probs)

    # transpose
    res = [list(xs) for xs in zip(*res)]
    return res


def motifs2profile_laplace(motifs: List[str]) -> List[List[float]]:
    """Form profile (= probability distribution) based on observations called `motifs`
    with Laplace's rule of succession.

    Returns array of shape (4, n) where n is len(motifs[0]).
    Note that position [0, 1, 2, 3] corresponds to ACGT.
    """
    assert all(len(motif) == len(motifs[0]) for motif in motifs)
    motifs = np.asarray(motifs)
    res = []
    for col in zip(*motifs):
        counter = collections.Counter(col)
        total = len(col)
        probs = [(counter[c] + 1) / (total + 4) for c in "ACGT"]
        res.append(probs)

    # transpose
    res = [list(xs) for xs in zip(*res)]
    return res


def motifs2consensus(motifs: List[str]) -> str:
    nucs = []
    for col in zip(*motifs):
        counter = collections.Counter(col)
        nuc, _ = counter.most_common(1)[0]
        nucs.append(nuc)
    return "".join(nucs)


def score(motifs: List[str]) -> int:
    """Score motifs with row-wise calculation"""
    consensus = motifs2consensus(motifs)
    score = sum(hamming_distance(consensus, motif) for motif in motifs)
    return score


def score_col(motifs: List[str]) -> int:
    """Score motifs with column-wise calculation"""
    score = 0
    for col in zip(*motifs):
        counter = collections.Counter(col)
        total = len(col)
        _, x = counter.most_common(1)[0]
        score += total - x
    return score


def _greedy_motifs_search_template(
    dna: List[str], k: int, t: int, to_profile
) -> List[str]:
    assert len(dna) == t
    best_motifs = []
    best_score = 10000000
    iter = sliding_window(dna[0], k)
    iter = map(lambda cs: "".join(cs), iter)
    for kmer in iter:
        motifs = [kmer]
        for seq in dna[1:]:
            prof = to_profile(motifs)
            motif = most_probable_kmer(seq, k, prof)
            motifs.append(motif)

        if score(motifs) < best_score:
            best_score = score(motifs)
            best_motifs = motifs

    return best_motifs


def greedy_motifs_search(dna: List[str], k: int, t: int) -> List[str]:
    """Greedy motifs search
    >>> dna = ["GGCGTTCAGGCA", "AAGAATCAGTCA", "CAAGGAGTTCGC", "CACGTCAATCAC", "CAATAATATTCG"]
    >>> greedy_motifs_search(dna, 3, 5)
    ['CAG', 'CAG', 'CAA', 'CAA', 'CAA']
    """
    return _greedy_motifs_search_template(dna, k, t, motifs2profile)


def greedy_motifs_search_modified(dna: List[str], k: int, t: int) -> List[str]:
    """Modified greedy motifs search with Laplace's rule of sucession

    >>> dna = ["GGCGTTCAGGCA", "AAGAATCAGTCA", "CAAGGAGTTCGC", "CACGTCAATCAC", "CAATAATATTCG"]
    >>> greedy_motifs_search_modified(dna, 3, 5)
    ['TTC', 'ATC', 'TTC', 'ATC', 'TTC']
    """
    return _greedy_motifs_search_template(dna, k, t, motifs2profile_laplace)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()
    with open(args.input, "r") as f:
        pattern = f.readline().strip()
        dna = f.readline().strip().split()
    res = hamming_distance_set(pattern, dna)
    print(res)
