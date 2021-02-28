from typing import Iterable, Iterator, Set, Tuple
import itertools as it


def reverse_complement(s: str) -> str:
    """
    >>> reverse_complement("AAAACCCGGT")
    'ACCGGGTTTT'

    """
    assert set(s).issubset(set("ATCG"))
    d = dict(zip("ATCG", "TAGC"))
    return "".join(d[x] for x in reversed(s))


def chunks(iterable: Iterable, n: int) -> Iterator[Tuple]:
    """Get chunk of length n from iterables

    https://docs.python.org/3/library/itertools.html
    """
    args = [iter(iterable)] * n
    chunks = it.zip_longest(*args, fillvalue=None)
    for chunk in chunks:
        if chunk[-1] is None:
            chunk = chunk[:chunk.index(None)]
        yield chunk


def sliding_window(iterable: Iterable, n=2) -> Iterable[Tuple]:
    """
    >>> list(sliding_window([1, 2, 3, 4, 5], 3))
    [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    """
    iterables = it.tee(iterable, n)

    for num_skipped, iterable in enumerate(iterables):
        for _ in range(num_skipped):
            next(iterable, None)

    return zip(*iterables)


def hamming_distance(left: str, right: str) -> int:
    """Compute Hamming distance of two strings

    >>> hamming_distance("GGGCCGTTGGT", "GGACCGTTGAC")
    3
    """
    return sum(1 for x, y in zip(left, right) if x != y)


def neighbors(pattern: str, d: int) -> Set[str]:
    """Returns a set of k-mers whose Hamming distance from `pattern` does not exceed `d`.

    >>> neighbors("ACG", 1) == {'ACG', 'AGG', 'ACA', 'ACC', 'CCG', 'TCG', 'ACT', 'AAG', 'GCG', 'ATG'}
    True
    """
    if d == 0:
        return {pattern}

    if len(pattern) == 1:
        return set("ACGT")

    neighborhood = set()
    for text in neighbors(pattern[1:], d):
        if hamming_distance(pattern[1:], text) < d:
            for nuc in "ACGT":
                neighborhood.add(nuc + text)
        else:
            neighborhood.add(pattern[0] + text)

    return neighborhood


if __name__ == "__main__":
    pattern = input().strip()
    d = int(input())
    res = neighbors(pattern, d)
    print(*res)
