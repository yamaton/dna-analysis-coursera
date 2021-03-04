import collections
from typing import Generator, Iterable


def windowed(seq: Iterable, n: int, step=1) -> Generator:
    """Returns sliding window as generator

    [NOTE] Borrowed from more-itertools

    >>> list(windowed([1,2,3,4], 3))
    [(1, 2, 3), (2, 3, 4)]
    """
    window = collections.deque(maxlen=n)
    i = n
    for _ in map(window.append, seq):
        i -= 1
        if i == 0:
            i = step
            yield tuple(window)


if __name__ == "__main__":
    k = int(input())
    seq = input().strip()
    intervals = ["".join(cs) for cs in windowed(seq, k)]
    for s in intervals:
        print(s)

