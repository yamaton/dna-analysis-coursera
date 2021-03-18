from typing import Dict, Hashable, Iterable, List, Tuple

Node = Hashable
Edge = Tuple[Node, Node]
EdgeWeight = Dict[Edge, int]
AdjList = Dict[Node, List[Node]]

Grid = List[List[int]]

def parse_weighted_edges(ss: Iterable[str]) -> EdgeWeight:
    """
    >>> parse_weighted_edge(["0->4:11", "1->4:2"])
    {(0, 4): 11, (1, 4): 2}
    """
    d = dict()
    for s in ss:
        edge_str, w_str = s.split(":")
        from_str, to_str = edge_str.split("->")
        from_, to_, w = map(int, (from_str, to_str, w_str))
        tup = (from_, to_)
        d[tup] = w
    return d


def distance_between_leaves(edgeweight: EdgeWeight, n_leaves: int) -> Grid:
    """
    >>> weight = {(0, 4): 11, (1, 4): 2, (2, 5): 6, (3, 5): 7, (4, 0): 11, (4, 1): 2, (4, 5): 4, (5, 4): 4, (5, 3): 7, (5, 2): 6}
    >>> distance_between_leaves(weight, 4)
    [[0, 13, 21, 22], [13, 0, 12, 13], [21, 12, 0, 13], [22, 13, 13, 0]]
    """
    ...


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()
    with open(args.input, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    ew = parse_weighted_edges(lines)
    print(ew)
