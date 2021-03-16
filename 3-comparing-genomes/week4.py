from typing import Hashable, Iterable, List


def greedy_sorting(seq: List[int]) -> List[List[int]]:
    """
    >>> greedy_sorting([-3, +4, +1, +5, -2])
    [[-1, -4, 3, 5, -2], [1, -4, 3, 5, -2], [1, 2, -5, -3, 4], [1, 2, 3, 5, 4], [1, 2, 3, -4, -5], [1, 2, 3, 4, -5], [1, 2, 3, 4, 5]]
    """
    history = []

    n = len(seq)
    for i in range(n):
        x = i + 1
        k = _index(seq, x)
        if k < 0:
            k = _index(seq, -x)

        if i < k:
            tmp = list(reversed(seq[i : k + 1]))
            for p in range(i, k + 1):
                seq[p] = -tmp[p - i]
            history.append(seq[:])

        assert abs(seq[i]) == x
        if seq[i] < 0:
            seq[i] = -seq[i]
            history.append(seq[:])

    return history


def _index(xs: Iterable[Hashable], y: Hashable) -> int:
    return next((i for i, x in enumerate(xs) if x == y), -1)


def count_breakpoints(seq: List[int]) -> int:
    """
    >>> count_breakpoints([+3, +4, +5, -12, -8, -7, -6, +1, +2, +10, +9, -11, +13, +14])
    8
    """
    n = len(seq)
    xs = [0] + seq + [n + 1]
    count = sum(1 for x, y in zip(xs, xs[1:]) if y != x + 1)
    return count


if __name__ == "__main__":
    xs = [int(s) for s in input().split()]
    res = count_breakpoints(xs)
    print(res)
    # for ns in res:
    #     s = " ".join(f"+{n}" if n > 0 else str(n) for n in ns)
    #     print(s)
