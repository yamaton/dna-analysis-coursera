from typing import Dict, List, Tuple
import collections
from utils import BLOSUM62_DICT

Coordinate = Tuple[int, int]
Edge = Tuple[Coordinate, Coordinate]


def alignment_with_affine_gap_penalties(
    v: str, w: str, sigma=11, epsilon=1, scoring_matrix=BLOSUM62_DICT
) -> Tuple[int, str, str]:
    """Alignment with Affine gap penalties
    sigma is a gap opening penality, and epsilon is a gap extension penalty

    >>> alignment_with_affine_gap_penalties("PRTEINS", "PRTWPSEIN")
    (8, 'PRT---EINS', 'PRTWPSEIN-')
    """
    dp_l = collections.defaultdict(lambda: -1000000)  # lower layer
    dp_m = collections.defaultdict(lambda: -1000000)  # middle layer
    dp_u = collections.defaultdict(lambda: -1000000)  # upper layer
    dp_m[0, 0] = 0

    # predecessor info
    pred = dict()
    pred[0, 0, "m"] = "."

    nrows = len(v)
    ncols = len(w)

    # [NOTE]
    # ↓ means single step down
    # | means more than one step down
    # → means single step right
    # - means more than one step right
    # o means transition from lower to middle in forward path
    # x means transition from upper to middle in forward path
    for i in range(nrows + 1):
        for j in range(ncols + 1):
            dp_l[i, j], pred[i, j, "l"] = max(
                (dp_m[i - 1, j] - sigma, "↓"), (dp_l[i - 1, j] - epsilon, "|")
            )
            dp_u[i, j], pred[i, j, "u"] = max(
                (dp_m[i, j - 1] - sigma, "→"), (dp_u[i, j - 1] - epsilon, "-")
            )
            if (i, j) != (0, 0):
                candidates = ((dp_l[i, j], "o"), (dp_u[i, j], "x"))
                if i > 0 and j > 0:
                    vi = v[i - 1]
                    wj = w[j - 1]
                    item = (dp_m[i - 1, j - 1] + scoring_matrix[vi, wj], "↘")
                    candidates += (item,)
                dp_m[i, j], pred[i, j, "m"] = max(candidates)

    # print("\nlower")
    # for i in range(nrow + 1):
    #     for j in range(ncol + 1):
    #         print(pred[i, j, "l"], end="")
    #     print()

    # print("\nmiddle")
    # for i in range(nrow + 1):
    #     for j in range(ncol + 1):
    #         print(pred[i, j, "m"], end="")
    #     print()

    # print("\nupper")
    # for i in range(nrow + 1):
    #     for j in range(ncol + 1):
    #         print(pred[i, j, "u"], end="")
    #     print()

    v_out = collections.deque([])
    w_out = collections.deque([])
    x = (nrows, ncols, "m")
    while True:
        if pred[x] == ".":
            break
        i, j, layer = x
        if layer == "m":
            if pred[x] == "o":
                x = (i, j, "l")
            elif pred[x] == "x":
                x = (i, j, "u")
            else:
                assert pred[x] == "↘"
                v_out.appendleft(v[i - 1])
                w_out.appendleft(w[j - 1])
                x = (i - 1, j - 1, "m")
        elif layer == "l":
            v_out.appendleft(v[i - 1])
            w_out.appendleft("-")
            if pred[x] == "↓":
                x = (i - 1, j, "m")
            else:
                assert pred[x] == "|"
                x = (i - 1, j, "l")
        else:
            assert layer == "u"
            v_out.appendleft("-")
            w_out.appendleft(w[j - 1])
            if pred[x] == "→":
                x = (i, j - 1, "m")
            else:
                assert pred[x] == "-"
                x = (i, j - 1, "u")

    return dp_m[nrows, ncols], "".join(v_out), "".join(w_out)


def alignment_with_middle_edge(
    v: str, w: str, sigma=5, scoring_matrix=BLOSUM62_DICT
) -> Edge:
    """
    >>> alignment_with_middle_edge("PLEASANTLY", "MEASNLY")
    ((4, 3), (5, 4))
    """
    nrows = len(v)
    ncols = len(w)

    # [0, 1, 2]    -->  (mid, mid+1) = (0, 1)
    # [0, 1, 2, 3]  --> (mid, mid+1) = (1, 2)
    assert ncols >= 2
    mid = ncols // 2 - 1
    dp = _middle_edge_helper(mid, v, w, sigma, scoring_matrix)

    for j in range(2):
        xs = [dp[i][j] for i in range(nrows + 1)]
        print(xs)

    mid_rev = ncols - mid - 2
    dp_bwd = _middle_edge_helper(
        mid_rev, "".join(reversed(v)), "".join(reversed(w)), sigma, scoring_matrix
    )

    for j in range(2):
        xs = [dp_bwd[i][j] for i in range(nrows + 1)]
        print(xs)

    # indices = [-1, -1]
    # maxval = [-1000000, -1000000]
    # for j in range(2):
    #     for i in range(nrows + 1):

    #         x = dp[i][j] + dp_bwd[nrows - i][1 - j]
    #         if maxval[j] < x:
    #             maxval[j] = x
    #             indices[j] = i

    # return (indices[0], mid), (indices[1], mid + 1)


def _get_mid(ncols: int) -> int:
    return ncols // 2 - 1


def _get_mid_bwd(ncols: int) -> int:
    return ncols - _get_mid(ncols) - 2


def _middle_edge_helper(
    mid: int, v: str, w: str, sigma: int, scoring_matrix: Dict[Tuple[str, str], int]
) -> List[List[int]]:
    nrows = len(v)
    ncols = len(w)

    # [0, 1, 2]    -->  (mid, mid+1) = (0, 1)
    # [0, 1, 2, 3]  --> (mid, mid+1) = (1, 2)
    assert ncols >= 2
    mid = ncols // 2 - 1

    dp = [[-10000000 for _ in range(2)] for _ in range(nrows + 1)]
    for j in range(mid + 2):
        for i in range(nrows + 1):
            if (i, j) == (0, 0):
                dp[i][j] = 0
                continue
            candidates = (dp[i][(j + 1) % 2] - sigma,)
            if i > 0:
                candidates += (dp[i - 1][j % 2] - sigma,)
            if i > 0 and j > 0:
                vi = v[i - 1]
                wj = w[j - 1]
                score = scoring_matrix[vi, wj]
                candidates += (dp[i - 1][(j + 1) % 2] + score,)
            dp[i][j % 2] = max(candidates)

    return dp


def _fix_dp_fwd(dp, nrows, ncols):
    mid = _get_mid(ncols)
    if mid % 2 == 1:
        for i in range(nrows + 1):
            dp[i, 0], dp[i, 1] = dp[i, 1], dp[i, 0]


def _fix_dp_bwd(dp, nrows, ncols):
    midcol = _get_mid(ncols)
    ...


def multiple_lcs(u: str, v: str, w: str) -> Tuple[int, str, str, str]:
    """longest common subsequences of three strings

    >>> multiple_lcs("ATATCCG", "TCCGA", "ATGTACTG")
    (3, 'AT-ATCCG-', '-T---CCGA', 'ATGTACTG-')
    """
    n0, n1, n2 = map(len, (u, v, w))
    dp = collections.defaultdict(lambda: -1000000)
    dp[0, 0, 0] = 0

    diff = dict()

    for i in range(n0 + 1):
        for j in range(n1 + 1):
            for k in range(n2 + 1):
                if (i, j, k) == (0, 0, 0):
                    continue
                candidates = [
                    (dp[i - 1, j, k], (1, 0, 0)),
                    (dp[i, j - 1, k], (0, 1, 0)),
                    (dp[i, j, k - 1], (0, 0, 1)),
                    (dp[i - 1, j - 1, k], (1, 1, 0)),
                    (dp[i - 1, j, k - 1], (1, 0, 1)),
                    (dp[i, j - 1, k - 1], (0, 1, 1)),
                ]
                if i > 0 and j > 0 and k > 0:
                    score = int(u[i - 1] == v[j - 1] == w[k - 1])
                    item = (dp[i - 1, j - 1, k - 1] + score, (1, 1, 1))
                    candidates.append(item)

                dp[i, j, k], diff[i, j, k] = max(candidates)

    goal = (n0, n1, n2)
    x = goal
    u_out = collections.deque([])
    v_out = collections.deque([])
    w_out = collections.deque([])

    while True:
        if x == (0, 0, 0):
            break
        i, j, k = x
        di, dj, dk = diff[x]
        c = u[i - 1] if di else "-"
        u_out.appendleft(c)
        c = v[j - 1] if dj else "-"
        v_out.appendleft(c)
        c = w[k - 1] if dk else "-"
        w_out.appendleft(c)
        x = (i - di, j - dj, k - dk)

    return dp[goal], *map(lambda ss: "".join(ss), (u_out, v_out, w_out))


if __name__ == "__main__":
    u = input().strip()
    v = input().strip()
    w = input().strip()
    ans, u_out, v_out, w_out = multiple_lcs(u, v, w)
    print(ans)
    print(u_out)
    print(v_out)
    print(w_out)
