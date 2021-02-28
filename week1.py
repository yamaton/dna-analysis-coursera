import collections
import re
from collections import Counter
import itertools as it
from typing import Iterable, Tuple, List, Dict, Set
from utils import sliding_window

Patterns = Dict[str, int]


def frequent_words(text: str, k: int) -> Patterns:
    """
    >>> frequent_words("ACTGACTCCCACCCC", 3)
    {'CCC': 3}

    """
    counter = Counter("".join(chunk) for chunk in sliding_window(text, k))
    max_count = max(counter.values())
    res = {k: cnt for k, cnt in counter.items() if cnt == max_count}
    return res


def pattern_count(text: str, subst: str) -> int:
    """Count appearance of substring `subst` in `text`.

    Overlapping is counted.

    >>> pattern_count("CCC", "CC")
    2
    """
    m = re.compile(f"(?=({subst}))")
    return sum(1 for _ in m.finditer(text))


def pattern_match(text: str, subst: str) -> List[int]:
    """Count appearance of substring `subst` in `text`.

    Overlapping is counted.

    >>> pattern_match("GATATATGCATATACTT", "ATAT")
    [1, 3, 9]
    """
    m = re.compile(f"(?=({subst}))")
    return [x.start() for x in m.finditer(text)]



def find_clumps(genome: str, k: int, L: int, t: int) -> List[str]:
    """Find patterns forming (L, t)-clump

    [TODO] Improve the current computational complexity O(len(genome) * L * k)

    Clump (L, t) means that k-mers appear at least t times
    in a length-L window of a genome.

    >>> find_clumps("CGGACTCGACAGATGTGAAGAACGACAATGTGAAGACTCGACACGACAGAGTGAAGAGAAGAGGAAACATTGTAA", 5, 50, 4)
    ['CGACA', 'GAAGA']
    """
    intervals = sliding_window(genome, L)
    res: Set[str] = set()

    for interval in intervals:
        freqs = frequent_words(interval, k)
        for w, cnt in freqs.items():
            if cnt >= t:
                res.add(w)

    return sorted(res)


def find_clumps_faster(genome: str, k: int, L: int, t: int) -> List[str]:
    """Find pattens forming (L, t)-clump
    Computational complexity: O(L * len(genome))

    Clump (L, t) means that k-mers appear at least t times
    in a length-L window of a genome.

    >>> find_clumps_faster("CGGACTCGACAGATGTGAAGAACGACAATGTGAAGACTCGACACGACAGAGTGAAGAGAAGAGGAAACATTGTAA", 5, 50, 4)
    ['CGACA', 'GAAGA']
    """
    assert L <= len(genome)
    interval = genome[:L]
    kmer_counts = collections.Counter("".join(xs) for xs in sliding_window(interval, k))
    res = {w for w, cnt in kmer_counts.items() if cnt >= t}

    offset = 0
    while offset + L <= len(genome):
        w_out = genome[offset: offset + k]
        w_in = genome[offset + L - k + 1: offset + L + 1]
        kmer_counts[w_out] -= 1
        kmer_counts[w_in] += 1
        if kmer_counts[w_in] >= t:
            res.add(w_in)
        offset += 1

    return sorted(res)


if __name__ == "__main__":
    genome = input().strip()
    k, L, t = map(int, input().strip().split())
    res = find_clumps_faster(genome, k, L, t)
    print(*res)

