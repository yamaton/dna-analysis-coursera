from typing import Iterable, Tuple
import itertools as it


def reverse_complement(s: str) -> str:
    """
    >>> reverse_complement("AAAACCCGGT")
    'ACCGGGTTTT'

    """
    assert set(s).issubset(set("ATCG"))
    d = dict(zip("ATCG", "TAGC"))
    return "".join(d[x] for x in reversed(s))


def chunks(iterable: Iterable, n: int) -> Iterable[Tuple]:
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

