import collections
import itertools as it

from typing import Callable, Coroutine, DefaultDict, Dict, Tuple
from utils import AAS, BLOSUM62_DICT, PAM250_DICT, SIMPLE_DICT, SIMPLE_DICT2

Coordinate = Tuple[int, int]


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
    pred[0, 0] = "."
    for i in range(nrow + 1):
        for j in range(ncol + 1):
            if (i, j) == (0, 0):
                continue

            xs = (dp[i - 1, j] - sigma, dp[i, j - 1] - sigma)
            if i > 0 and j > 0:
                c1 = v[i - 1]
                c2 = w[j - 1]
                xs += (dp[i - 1, j - 1] + scoring_matrix[c1, c2],)

            for x, dir_ in zip(xs, "↓→↘"):
                if dp[i, j] < x:
                    dp[i, j] = x
                    pred[i, j] = dir_

    goal = (nrow, ncol)
    v_out, w_out = _backtrack(pred, v, w, goal)

    return dp[nrow, ncol], v_out, w_out


def _backtrack(
    pred: Dict[Coordinate, str],
    v: str,
    w: str,
    goal: Coordinate,
) -> Tuple[str, str]:
    """backtracking"""
    v_out = collections.deque([])
    w_out = collections.deque([])
    r, c = goal
    # while is_not_done(r, c):
    while True:
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
            assert pred[r, c] == "."
            break

    return "".join(v_out), "".join(w_out)


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
    pred[0, 0] = "."
    for i in range(nrow + 1):
        for j in range(ncol + 1):
            if (i, j) == (0, 0):
                continue
            xs = (0, dp[i - 1, j] - sigma, dp[i, j - 1] - sigma)
            if i > 0 and j > 0:
                c1 = v[i - 1]
                c2 = w[j - 1]
                xs += (dp[i - 1, j - 1] + scoring_matrix[c1, c2],)

            for x, dir_ in zip(xs, ".↓→↘"):
                if dp[i, j] < x:
                    dp[i, j] = x
                    pred[i, j] = dir_

    goal = max(
        ((i, j) for i in range(nrow + 1) for j in range(ncol + 1)),
        key=lambda tup: dp[tup],
    )
    v_out, w_out = _backtrack(pred, v, w, goal)

    return dp[goal], v_out, w_out


def fit_alignment(
    v: str,
    w: str,
    sigma=1,
    scoring_matrix=SIMPLE_DICT,
) -> Tuple[int, str, str]:
    """fit w with respect to v, where sequence v is longer.

    >>> fit_alignment("GTAGGCTTAAGGTTA", "TAGATA")
    (2, 'TAGGCTTA', 'TAGA-T-A')
    """
    nrow = len(v)
    ncol = len(w)
    dp = collections.defaultdict(lambda: -10000000)
    pred = dict()
    for i in range(nrow + 1):
        dp[i, 0] = 0
        pred[i, 0] = "."

    for i in range(nrow + 1):
        for j in range(ncol + 1):
            if i == 0:
                continue
            xs = (dp[i - 1, j] - sigma, dp[i, j - 1] - sigma)
            if i > 0 and j > 0:
                c1 = v[i - 1]
                c2 = w[j - 1]
                xs += (dp[i - 1, j - 1] + scoring_matrix[c1, c2],)

            for x, dir_ in zip(xs, "↓→↘"):
                if dp[i, j] < x:
                    dp[i, j] = x
                    pred[i, j] = dir_

    goal = max(
        ((i, ncol) for i in range(nrow + 1)),
        key=lambda tup: dp[tup],
    )

    # print(f"{goal=}")
    # _print_grid(pred, nrow, ncol)
    v_out, w_out = _backtrack(pred, v, w, goal)

    return dp[goal], "".join(v_out), "".join(w_out)


def overlap_alignment(
    v: str,
    w: str,
    sigma=2,
    scoring_matrix=SIMPLE_DICT2,
) -> Tuple[int, str, str]:
    """Align overlap such that a suffix of v aligns a prefix of w

    >>> overlap_alignment("PAWHEAE", "HEAGAWGHEE")
    (1, 'HEAE', 'HEA-')
    """
    nrow = len(v)
    ncol = len(w)
    dp = collections.defaultdict(lambda: -10000000)
    pred = dict()
    for i in range(nrow + 1):
        dp[i, 0] = 0
        pred[i, 0] = "."

    for i in range(nrow + 1):
        for j in range(ncol + 1):
            if i == 0:
                continue
            xs = (dp[i - 1, j] - sigma, dp[i, j - 1] - sigma)
            if i > 0 and j > 0:
                c1 = v[i - 1]
                c2 = w[j - 1]
                xs += (dp[i - 1, j - 1] + scoring_matrix[c1, c2],)

            for x, dir_ in zip(xs, "↓→↘"):
                if dp[i, j] < x:
                    dp[i, j] = x
                    pred[i, j] = dir_

    goal = max(
        ((nrow, j) for j in range(ncol + 1)),
        key=lambda tup: dp[tup],
    )

    # print(f"{goal=}")
    # _print_grid(pred, nrow, ncol)
    v_out, w_out = _backtrack(pred, v, w, goal)

    return dp[goal], v_out, w_out


def _print_grid(grid: Dict[Coordinate, str], nrow: int, ncol: int):
    for i in range(nrow + 1):
        for j in range(ncol + 1):
            print(grid[i, j], end="")
        print()


def levenstein(v: str, w: str) -> int:
    """
    >>> levenstein("PLEASANTLY", "MEANLY")
    5
    """
    scoring_matrix = {(x, y): (0 if x == y else -1) for x in AAS for y in AAS}
    negscore, s1, s2 = global_alignment(v, w, sigma=1, scoring_matrix=scoring_matrix)
    return -negscore


if __name__ == "__main__":
    v = input().strip()
    w = input().strip()
    ans, v_out, w_out = overlap_alignment(v, w)
    print(ans)
    print(v_out)
    print(w_out)
