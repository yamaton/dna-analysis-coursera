import re
from collections import Counter
import itertools as it
from typing import Tuple, List, Dict


Patterns = Dict[str, int]

def frequent_words(text: str, k: int) -> Patterns:
    """
    >>> frequent_words("ACTGACTCCCACCCC", 3)
    {'CCC': 3}

    """
    counter = Counter(
        ''.join(chunk) for chunk in sliding_window(text, k)
    )
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


def sliding_window(iterable, n=2):
    """
    >>> list(sliding_window([1, 2, 3, 4, 5], 3))
    [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    """
    iterables = it.tee(iterable, n)

    for num_skipped, iterable in enumerate(iterables):
        for _ in range(num_skipped):
            next(iterable, None)

    return zip(*iterables)


def reverse_complement(s: str):
    """
    >>> reverse_complement("AAAACCCGGT")
    'ACCGGGTTTT'

    """
    assert set(s).issubset(set('ATGC'))
    d = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(d[x] for x in reversed(s))


if __name__ == "__main__":
    kwd = input().strip()
    text = input().strip()
    res = pattern_match(text, kwd)
    print(*res)

