"""
Bioinformatics Algorithms VI - DNA mutations


"""
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Set, Tuple, Union
import collections
import functools
import sys

sys.setrecursionlimit(1000000)

Node = int
Edge = Tuple[Node, Node]
Label = Hashable
LabeledEdges = Dict[Edge, Label]
NodeLabelPairToNode = Dict[Tuple[Node, Label], Node]

SuffixNode = Union[Tuple[int, int], int]
SuffixLabel = Hashable
SuffixTreeAdjList = Dict[SuffixNode, List[Tuple[SuffixNode, SuffixLabel]]]

CompactSuffixLabel = Tuple[int, int]
CompactSuffixTreeAdjList = Dict[SuffixNode, List[Tuple[SuffixNode, CompactSuffixLabel]]]


AdjList = Dict[Node, List[Node]]


@dataclass
class Trie:
    root: int
    tree: NodeLabelPairToNode


def trie_construction(patterns: List[str]) -> Trie:
    """
    >>> patterns = ["ATAGA", "ATC", "GAT"]
    >>> ss = ["0->1:A", "1->2:T", "2->3:A", "3->4:G", "4->5:A", "2->6:C", "0->7:G", "7->8:A", "8->9:T"]
    >>> expected = parse_labeled_edges(ss)
    >>> computed = trie_construction(patterns)
    >>> computed == expected
    True
    """
    trie = _real_trie_construction(patterns, root_node=0)
    out = to_labeled_edges(trie.tree)
    return out


def _real_trie_construction(
    patterns: List[str], root_node: int = -1
) -> NodeLabelPairToNode:
    unused_node = root_node + 1
    tree: NodeLabelPairToNode = dict()
    for pat in patterns:
        curr = root_node
        for c in pat:
            if (curr, c) in tree:
                curr = tree[curr, c]
            else:
                tree[curr, c] = unused_node
                curr = unused_node
                unused_node += 1
    return Trie(tree=tree, root=root_node)


def parse_labeled_edges(ss: Iterable[str]) -> LabeledEdges:
    d = dict()
    for s in ss:
        edge, label = s.strip().split(":")
        a, b = map(int, edge.split("->"))
        d[a, b] = label
    return d


def to_labeled_edges(d: NodeLabelPairToNode) -> LabeledEdges:
    return {(a, b): label for (a, label), b in d.items()}


def format_labeled_edges(d: LabeledEdges) -> List[str]:
    return [f"{a}->{b}:{l}" for (a, b), l in d.items()]


def trie_matching(text: str, patterns: Iterable[str]) -> List[int]:
    """
    >>> trie_matching("AATCGGGTTCAATCGGGGT", ["ATCG", "GGGT"])
    [1, 4, 11, 15]
    """
    trie = _real_trie_construction(patterns, root_node=-1)
    n = len(text)
    return [i for i in range(n) if prefix_trie_matching(text[i:], trie)]


def prefix_trie_matching(text: str, trie: Trie) -> bool:
    i = 0
    v = trie.root
    while True:
        if i < len(text) and (v, (c := text[i])) in trie.tree:
            v = trie.tree[v, c]
            i += 1
        elif _is_leaf(v, trie):
            return True
        else:
            break
    return False


def _is_leaf(node: Node, trie: Trie) -> bool:
    # Inefficient!
    return (node in trie.tree.values()) and all(n != node for n, _ in trie.tree.keys())


def suffix_tree_edge_labels(text: str) -> List[str]:
    """
    >>> s = "ATAAATG$"
    >>> computed = suffix_tree_edge_labels(s)
    >>> expected = ["AAATG$", "G$", "T", "ATG$", "TG$", "A", "A", "AAATG$", "G$", "T", "G$", "$"]
    >>> computed == sorted(expected)
    True
    """
    compact_suffix_tree = create_compact_suffix_tree(text)
    labels = [text[a:b] for xs in compact_suffix_tree.values() for (_, (a, b)) in xs]
    return sorted(labels)


def create_compact_suffix_tree(text: str) -> CompactSuffixTreeAdjList:
    return _compactify_suffix_tree(_construct_suffix_trie(text))


def _construct_suffix_trie(text: str) -> SuffixTreeAdjList:
    root_node = hash((-1, -1))
    tree: SuffixTreeAdjList = collections.defaultdict(list)
    tree[root_node] = []
    n = len(text)
    patterns = [text[i:] for i in range(n)]

    for i, pat in enumerate(patterns):
        curr = root_node
        for k, c in enumerate(pat):
            next_node, label = next(
                (
                    (next_node, label)
                    for (next_node, label) in tree[curr]
                    if text[label] == c
                ),
                (None, None),
            )
            if label is not None:
                curr = next_node
            else:
                new_node = hash((i, k))
                pair = new_node, i + k
                tree[curr].append(pair)
                curr = new_node
    return tree


def _compactify_suffix_tree(
    tree: SuffixTreeAdjList, root: Node = hash((-1, -1))
) -> CompactSuffixTreeAdjList:
    # handle when the root has only one edge outside
    x, length = _get_to_leaf_or_branch(root, tree)
    if root != x:
        # if root directly arrives at a leaf such as "AAA"
        i = tree[root][0]
        range_label = i, i + length + 1
        new_tree = {root: [x, range_label]}
        return new_tree

    # main part
    assert len(tree[root]) > 1
    new_tree = dict()
    for node, xs in tree.items():
        if len(xs) > 1:
            ys = []
            for (next_node, i) in xs:
                next_node, length = _get_to_leaf_or_branch(next_node, tree)
                range_label = (i, i + length + 1)
                pair = next_node, range_label
                ys.append(pair)
            new_tree[node] = ys
    return new_tree


def _get_to_leaf_or_branch(
    curr: SuffixNode, tree: SuffixTreeAdjList
) -> Tuple[Node, int]:
    if curr not in tree or len(tree[curr]) > 1:
        return curr, 0

    assert len(tree[curr]) == 1
    curr, _ = tree[curr][0]
    target, length = _get_to_leaf_or_branch(curr, tree)
    return target, length + 1


def longest_repeat(text: str) -> str:
    """Find longest-repeating substring by finding a longest path to a branch node from the root

    >>> longest_repeat("ATATCGTTTTATCGTT")
    'TATCGTT'
    """
    text = text if text[-1] == "$" else text + "$"
    g = create_compact_suffix_tree(text)

    # DFS
    candidates = []
    root = hash((-1, -1))
    stack = [(root, "")]
    while stack:
        v, s = stack.pop()
        if v not in g:
            continue

        if len(g[v]) > 1:
            candidates.append(s)

        for (u, (i, j)) in g[v]:
            item = (u, s + text[i:j])
            stack.append(item)

    res = max(candidates, key=len)
    return res


def parse_adjlist(lines: Iterable[str]) -> AdjList:
    d = dict()
    for line in lines:
        left, right = line.strip().split(" -> ")
        v = int(left)
        xs = [] if right == "{}" else [int(s) for s in right.split(",")]
        d[v] = xs
    return d


def parse_colors(lines: Iterable[str]) -> Dict[Node, str]:
    d = dict()
    for line in lines:
        l, r = line.strip().split(": ")
        n = int(l)
        d[n] = r
    return d


def colorize_nodes(g: AdjList, leaf_colors: Dict[Node, str], f=(lambda x: x)):
    """
    The argument f is how to extract node from an item in list in adjacency list.

    >>> g = parse_adjlist(["0 -> {}", "1 -> {}", "2 -> 0,1", "3 -> {}", "4 -> {}", "5 -> 3,2", "6 -> {}", "7 -> 4,5,6"])
    >>> leaf_colors = parse_colors(["0: red", "1: red", "3: blue", "4: blue", "6: red"])
    >>> computed = colorize_nodes(g, leaf_colors)
    >>> expected = {0: "red", 1: "red", 2: "red", 3: "blue", 4: "blue", 5: "purple", 6: "red", 7: "purple"}
    >>> computed == expected
    True
    """
    assert all((v not in g) or (g[v] == []) for v in leaf_colors)
    color = leaf_colors.copy()

    def _colorize_query(v: Node) -> str:
        if v in color:
            return color[v]

        children_colors = {_colorize_query(f(item)) for item in g[v]}
        if len(children_colors) > 1:
            ans = "purple"
        else:
            ans = children_colors.pop()
        color[v] = ans
        return ans

    vs = set(g.keys())
    vs |= functools.reduce(set.union, ({f(x) for x in xs} for xs in g.values()))
    for v in vs:
        _colorize_query(v)
    return color


def format_dict(d: Dict) -> List[str]:
    return [f"{k}: {v}" for k, v in d.items()]


def longest_shared_substring(text1: str, text2: str) -> str:
    """Find longest-repeating substring to purple node by finding a longest path to a branch node from the root

    >>> longest_shared_substring("TCGGTAGATTGCGCCCACTC", "AGGGGCTCGCAGTGTAAGAA")
    'AGA'
    """
    assert all((c not in t) for t in (text1, text2) for c in "#$")
    text = f"{text1}#{text2}$"
    g = create_compact_suffix_tree(text)

    leaf_color = dict()
    for xs in g.values():
        for (v, (i, j)) in xs:
            if v not in g or (not g[v]):
                s = text[i:j]
                col = "blue" if "#" in s else "red"
                leaf_color[v] = col
    color = colorize_nodes(g, leaf_color, lambda tup: tup[0])

    # DFS
    candidates = []
    root = hash((-1, -1))
    stack = [(root, "")]
    while stack:
        v, s = stack.pop()
        if (v not in g) or (not g[v]):
            continue

        if color[v] != "purple":
            continue

        candidates.append(s)
        for (u, (i, j)) in g[v]:
            item = (u, s + text[i:j])
            stack.append(item)

    res = max(candidates, key=len)
    return res


# def shorteset_nonshared_substring(text1: str, text2: str) -> str:
#     """Find shortest non-shared substring

#     >>> shorteset_nonshared_substring("CCAAGCTGCTAGAGG", "CATGCTGGGCTGGCT")
#     'AA'
#     """
#     ...


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("input")
    # args = parser.parse_args()
    # with open(args.input, "r") as f:
    #     lines = [line.strip() for line in f.readlines()]

    # idx = lines.index("-")
    # g = parse_adjlist(lines[:idx])
    # leaf_colors = parse_colors(lines[idx + 1:])
    # d = colorize_nodes(g, leaf_colors)
    # for k in sorted(d.keys()):
    #     print(f"{k}: {d[k]}")

    text1 = input().strip()
    text2 = input().strip()
    res = longest_shared_substring(text1, text2)
    print(res)
