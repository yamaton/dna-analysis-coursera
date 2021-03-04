import collections
import itertools as it
from typing import List, Set
from utils import hamming_distance, neighbors, reverse_complement, sliding_window


def get_skew(genome: str) -> List[int]:
    """Compute (G - C) skew with scoring G = 1 and C = -1 and the rest = 0.

    >>> get_skew("CATGGGCATCGGCCATACGCC")
    [0, -1, -1, -1, 0, 1, 2, 1, 1, 1, 0, 1, 2, 1, 0, 0, 0, 0, -1, 0, -1, -2]
    """

    def to_score(c):
        if c == "G":
            res = 1
        elif c == "C":
            res = -1
        else:
            res = 0
        return res

    return [0] + list(it.accumulate(map(to_score, genome)))


def find_skew_minimum(genome: str) -> List[int]:
    """Return all indices giving minimum in skew

    >>> find_skew_minimum("TAAAGACTGCCGAGAGGCCAACACGAGTGCTAGAACGAGGGGCGTAAACGCGGGTCCGAT")
    [11, 24]
    """
    skew = get_skew(genome)
    minval = min(skew)
    return [i for i, x in enumerate(skew) if x == minval]


def approx_pattern_matching(pattern: str, genome: str, d: int) -> List[int]:
    """Find locations of k-mer (pattern') in the genome with
    hamming_distance(pattern, pattern') <= d

    >>> approx_pattern_matching("ATTCTGGA", "CGCCCGAATCCAGAACGCATTCCCATATTTCGGGACCACTGGCCTCCACGGTACGGACGTCAATCAAAT", 3)
    [6, 7, 26, 27]
    """
    intervals = sliding_window(genome, len(pattern))
    return [
        i
        for i, interval in enumerate(intervals)
        if hamming_distance(pattern, interval) <= d
    ]


def approx_pattern_count(pattern: str, genome: str, d: int) -> int:
    """
    >>> approx_pattern_count("GAGG", "TTTAGAGCCTTCAGAGG", 2)
    4
    """
    return len(approx_pattern_matching(pattern, genome, d))


def _frequent_words_helper(text: str, k: int, d: int) -> collections.Counter:
    freq_map = collections.Counter()
    n = len(text)
    for i in range(n - k + 1):
        pattern = text[i : i + k]
        neighborhood = neighbors(pattern, d)
        for neighbor in neighborhood:
            freq_map[neighbor] += 1
    return freq_map


def frequent_words_with_mismatches(text: str, k: int, d: int) -> Set[str]:
    """
    Find k-mer with Hamming distance at most `d` from a substring in `text`.

    >>> frequent_words_with_mismatches("ACGTTGCATGTCGCATGATGCATGAGAGCT", 4, 1) == {'ATGT', 'GATG', 'ATGC'}
    True
    """
    freq_map = _frequent_words_helper(text, k, d)
    _, maxval = freq_map.most_common(1)[0]
    res = {k for k, v in freq_map.items() if v == maxval}
    return res


def frequent_words_with_mismatches_with_revcomp(text: str, k: int, d: int) -> Set[str]:
    """
    >>> frequent_words_with_mismatches_with_revcomp("ACGTTGCATGTCGCATGATGCATGAGAGCT", 4, 1) == {'ATGT', 'ACAT'}
    True
    """
    freq_map = _frequent_words_helper(text, k, d)
    freq_map_rc = _frequent_words_helper(reverse_complement(text), k, d)
    freq_map += freq_map_rc
    _, maxval = freq_map.most_common(1)[0]
    res = {k for k, v in freq_map.items() if v == maxval}
    return res


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()
    with open(args.input, "r") as f:
        text = "".join(line.strip() for line in f.readlines())
    res = frequent_words_with_mismatches_with_revcomp(text, 9, 1)
    print(*res)
