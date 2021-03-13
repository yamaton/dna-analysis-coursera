import collections
import itertools as it

from typing import Tuple
from utils import BLOSUM62_DICT, PAM250_DICT


def global_alignment(
    v: str, w: str, sigma=5, scoring_matrix=BLOSUM62_DICT
) -> Tuple[int, str, str]:
    """
    >>> global_alignment("PLEASANTLY", "MEANLY")
    (8, 'PLEASANTLY', '-MEA--N-LY')
    """
    nrow = len(v)
    ncol = len(w)
    dp = collections.defaultdict(lambda: -10000000)
    pred = dict()
    dp[0, 0] = 0
    for i in range(nrow + 1):
        for j in range(ncol + 1):
            if (i, j) == (0, 0):
                continue

            xs = (dp[i - 1, j] - sigma, dp[i, j - 1] - sigma)
            if i > 0 and j > 0:
                c1 = v[i - 1]
                c2 = w[j - 1]
                xs += (dp[i - 1, j - 1] + scoring_matrix[c1, c2],)

            dp[i, j] = max(xs)
            for x, dir_ in zip(xs, "↓→↘"):
                if dp[i, j] == x:
                    pred[i, j] = dir_
                    break
            else:
                raise ValueError("Something is wrong...")

    v_out = collections.deque([])
    w_out = collections.deque([])
    r, c = nrow, ncol
    while (r, c) != (0, 0):
        dir_ = pred[r, c]
        if dir_ == "↓":
            v_out.appendleft(v[r - 1])
            w_out.appendleft("-")
            r -= 1
        elif dir_ == "→":
            v_out.appendleft("-")
            w_out.appendleft(w[c - 1])
            c -= 1
        else:
            assert dir_ == "↘"
            v_out.appendleft(v[r - 1])
            w_out.appendleft(w[c - 1])
            r -= 1
            c -= 1

    return dp[nrow, ncol], "".join(v_out), "".join(w_out)


def local_alignment(
    v: str,
    w: str,
    sigma=5,
    scoring_matrix=PAM250_DICT,
) -> Tuple[int, str, str]:
    """
    >>> local_alignment("MEANLY", "PENALTY")
    (15, 'EANL-Y', 'ENALTY')
    """
    nrow = len(v)
    ncol = len(w)
    dp = collections.defaultdict(lambda: -10000000)
    pred = dict()
    dp[0, 0] = 0
    pred[0, 0] = "o"
    for i in range(nrow + 1):
        for j in range(ncol + 1):
            if (i, j) == (0, 0):
                continue

            xs = (0, dp[i - 1, j] - sigma, dp[i, j - 1] - sigma)
            if i > 0 and j > 0:
                c1 = v[i - 1]
                c2 = w[j - 1]
                xs += (dp[i - 1, j - 1] + scoring_matrix[c1, c2],)

            dp[i, j] = max(xs)
            for x, dir_ in zip(xs, ".↓→↘"):
                if dp[i, j] == x:
                    pred[i, j] = dir_

    goal = max(
        ((i, j) for i in range(nrow + 1) for j in range(ncol + 1)),
        key=lambda tup: dp[tup],
    )

    # print(f"{goal=}")
    # for i in range(nrow + 1):
    #     for j in range(ncol + 1):
    #         print(pred[i, j], end="")
    #     print()
    # print()

    v_out = collections.deque([])
    w_out = collections.deque([])
    r, c = goal
    while (r, c) != (0, 0):
        dir_ = pred[r, c]
        if dir_ == "↓":
            v_out.appendleft(v[r - 1])
            w_out.appendleft("-")
            r -= 1
        elif dir_ == "→":
            v_out.appendleft("-")
            w_out.appendleft(w[c - 1])
            c -= 1
        elif dir_ == "↘":
            v_out.appendleft(v[r - 1])
            w_out.appendleft(w[c - 1])
            r -= 1
            c -= 1
        else:
            assert dir_ == "."
            break

    return dp[goal], "".join(v_out), "".join(w_out)


if __name__ == "__main__":
    v = input().strip()
    w = input().strip()
    count, s1, s2 = global_alignment(v, w)
    print(count)
    print(s1)
    print(s2)
