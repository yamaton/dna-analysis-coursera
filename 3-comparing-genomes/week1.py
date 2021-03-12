"""
Comparing genomes (UCSD Bioinformatics III)


"""

import collections
from typing import Hashable, Iterable, List, Tuple

Path = List[Hashable]

def dp_change(money: int, coins: Iterable[int]) -> int:
    """

    >>> dp_change(40, [50, 25, 20, 10, 5, 1])
    2
    """
    maxval = 100000000
    dp = collections.defaultdict(lambda: maxval)
    dp[0] = 0
    for x in range(1, money + 1):
        dp[x] = 1 + min(dp[x - coin] for coin in coins)
    return dp[money]



def dp_change_with_backtracking(money: int, coins: Iterable[int]) -> Tuple[int, Path]:
    """Coin change problem with back tracking

    >>> dp_change_more(40, [50, 25, 20, 10, 5, 1])
    (2, [20, 20])
    """
    maxval = 100000000
    dp = collections.defaultdict(lambda: (maxval, None))
    dp[0] = (0, 0)   # (count, predecessor)
    for x in range(1, money + 1):
        count, pred = min((dp[x - coin][0], x - coin) for coin in coins)
        dp[x] = (1 + count, pred)

    path = []
    x = money
    while x > 0:
        path.append(x)
        _, x = dp[x]

    return dp[money][0], list(path)







if __name__ == "__main__":
    money = int(input())
    coins = [int(s) for s in input().strip().split(",")]
    res = dp_change_more(money, coins)
    print(res)
