import collections
from functools import partial
import itertools as it
from typing import Generator, Iterable, List


def windowed(iterable: Iterable, n: int, step=1) -> Generator:
    """Returns sliding window as generator

    [NOTE] Borrowed from more-itertools

    >>> list(windowed([1,2,3,4], 3))
    [(1, 2, 3), (2, 3, 4)]
    """
    window = collections.deque(maxlen=n)
    i = n
    for _ in map(window.append, iterable):
        i -= 1
        if i == 0:
            i = step
            yield tuple(window)


def chunked(iterable: Iterable, n: int) -> Generator:
    """
    [NOTE] Borrowed from more-itertools
    >>> list(chunked([1, 2, 3, 4, 5, 6, 7], 3))
    [[1, 2, 3], [4, 5, 6], [7]]
    """
    iterator = iter(partial(take, n, iter(iterable)), [])
    return iterator


def take(n: int, iterable: Iterable) -> List:
    """
    >>> take(3, [1, 2, 3,4])
    [1, 2, 3]
    """
    return list(it.islice(iterable, n))



def reverse_complement(s: str) -> str:
    """
    >>> reverse_complement("AAAACCCGGT")
    'ACCGGGTTTT'

    """
    assert set(s).issubset(set("ATCG"))
    d = dict(zip("ATCG", "TAGC"))
    return "".join(d[x] for x in reversed(s))


if __name__ == "__main__":
    k = int(input())
    seq = input().strip()
    intervals = ["".join(cs) for cs in windowed(seq, k)]
    for s in intervals:
        print(s)
