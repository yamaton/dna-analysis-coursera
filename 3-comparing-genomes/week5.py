from typing import Hashable, Iterable, List, Set, Tuple
import itertools as it
import collections
from utils import AdjList, chunked

Ring = List  # a compromise!
SyntenyBlock = str
Chromosome = Ring[SyntenyBlock]
Genome = Set[Chromosome]

Node = int
Cycle = Ring[Node]
Edge = Tuple[Node, Node]


def chromosome_to_cycle(chromosome: Chromosome) -> Cycle:
    """
    >>> chromosome_to_cycle(['+1', '-2', '-3', '+4'])
    [1, 2, 4, 3, 6, 5, 7, 8]
    """
    nodes = []
    for x in chromosome:
        x = int(x)
        if x > 0:
            nodes.append(2 * x - 1)
            nodes.append(2 * x)
        else:
            assert x < 0
            x = -x
            nodes.append(2 * x)
            nodes.append(2 * x - 1)

    return nodes


def cycle_to_chromosome(cycle: Cycle) -> Chromosome:
    """
    >>> cycle_to_chromosome([1, 2, 4, 3, 6, 5, 7, 8])
    ['+1', '-2', '-3', '+4']
    """
    cycle = [int(s) for s in cycle]
    assert len(cycle) % 2 == 0
    xs = []
    for (x, y) in chunked(cycle, 2):
        assert max(x, y) % 2 == 0
        if x < y:
            xs.append(y // 2)
        else:
            xs.append(-x // 2)
    res = [to_synteny_block(x) for x in xs]
    return res


def to_synteny_block(x: int) -> SyntenyBlock:
    return f"+{x}" if x > 0 else str(x)


def format_chromosome(chromosome: Chromosome) -> str:
    """
    >>> format_chromosome(['+1', '-2', '-3', '+4'])
    '(+1 -2 -3 +4)'

    >>> format_chromosome([1, -2, -3, 4])
    '(+1 -2 -3 +4)'
    """
    xs = []
    for x in chromosome:
        if isinstance(x, int):
            x = to_synteny_block(x)
        xs.append(x)
    s = " ".join(xs)
    return f"({s})"


def format_genome(genome: Genome) -> str:
    """
    >>> format_genome([['+1', '-2', '-3', '+4'], ['-5', '-6', '+7']])
    '(+1 -2 -3 +4)(-5 -6 +7)'
    """
    res = "".join(format_chromosome(ch) for ch in genome)
    return res


def colored_edges(genome: Iterable[Chromosome]) -> List[Edge]:
    """
    >>> colored_edges([['+1', '-2', '-3'], ['+4', '+5', '-6']])
    [(2, 4), (3, 6), (5, 1), (8, 9), (10, 12), (11, 7)]
    """
    edges_list = [
        _cycle_to_colored_edges(chromosome_to_cycle(chromosome))
        for chromosome in genome
    ]
    edges = list(it.chain.from_iterable(edges_list))
    return edges


def _cycle_to_colored_edges(cycle: Cycle) -> List[Edge]:
    """
    >>> _cycle_to_colored_edges([1, 2, 4, 3, 6, 5, 7, 8])
    [(2, 4), (3, 6), (5, 7), (8, 1)]
    """
    assert len(cycle) % 2 == 0
    num_edges = len(cycle) // 2
    iterable = it.cycle(cycle)
    next(iterable)  # drop the first element
    iterable = it.islice(chunked(iterable, 2), num_edges)
    return list(iterable)


def parse_genome(s: str) -> Genome:
    """
    >>> parse_genome("(+1 -2 -3)(+4 +5 -6)")
    [['+1', '-2', '-3'], ['+4', '+5', '-6']]

    >>> parse_genome("(+1 -2 -3)")
    [['+1', '-2', '-3']]
    """
    s = s.strip().strip("()")
    genome_strs = s.split(")(")
    res = [genome_str.split() for genome_str in genome_strs]
    return res


def edges_to_genome(edges: Iterable[Edge]) -> Genome:
    """
    >>> edges = [(2, 4), (3, 6), (5, 1), (7, 9), (10, 12), (11, 8)]
    >>> edges_to_genome(edges)
    [['+1', '-2', '-3'], ['+4', '+6', '-5']]
    """
    g = dict()
    for (a, b) in edges:
        g[a] = b
        g[b] = a

    res = []
    chromosome = []
    x = 2
    while g:
        if chromosome and x == chromosome[0] * 2:
            res.append(chromosome)
            chromosome = []
            x = min(g) + 1
        else:
            synteny = (x + 1) // 2
            if x % 2 == 1:
                synteny = -synteny
            chromosome.append(synteny)
            x = g.pop(x)
            g.pop(x)
            x = (x - 1) if x % 2 == 0 else (x + 1)
    res.append(chromosome)

    return [[to_synteny_block(n) for n in ch] for ch in res]


def parse_edges(s: str) -> List[Edge]:
    """
    >>> s = "(2, 4), (3, 6), (5, 1), (7, 9), (10, 12), (11, 8)"
    >>> parse_edges(s)
    [(2, 4), (3, 6), (5, 1), (7, 9), (10, 12), (11, 8)]
    """
    tup_strs = s.split("), (")
    res = []
    for tup_str in tup_strs:
        a, b = map(int, tup_str.strip().strip("()").split(", "))
        res.append((a, b))
    return res


def two_break_on_genome_graph(
    edges: List[Edge], i1: int, i2: int, i3: int, i4: int
) -> List[Edge]:
    """Remove edges (i1, i2) and (i3, i4) from the edges and add (i1, i3) and (i2, i4).

    >>> edges = [(2, 4), (3, 8), (7, 5), (6, 1)]
    >>> computed = two_break_on_genome_graph(edges, 1, 6, 3, 8)
    >>> expected = [(2, 4), (3, 1), (7, 5), (6, 8)]
    >>> {frozenset(e) for e in computed} == {frozenset(e) for e in expected}
    True
    """
    removed = {frozenset((i1, i2)), frozenset((i3, i4))}
    xs = [tup for tup in edges if frozenset(tup) not in removed]
    xs.append((i1, i3))
    xs.append((i2, i4))
    return xs


def two_break_on_genome(genome: Genome, i1: int, i2: int, i3: int, i4: int) -> Genome:
    """
    >>> genome = parse_genome("(+1 -2 -4 +3)")
    >>> two_break_on_genome(genome, 1, 6, 3, 8)
    [['+1', '-2'], ['+3', '-4']]
    """
    edges = colored_edges(genome)
    edges = two_break_on_genome_graph(edges, i1, i2, i3, i4)
    res = edges_to_genome(edges)
    return res


def two_break_distance(p: Genome, q: Genome) -> int:
    """
    >>> p = parse_genome("(+1 +2 +3 +4 +5 +6)")
    >>> q = parse_genome("(+1 -3 -6 -5)(+2 -4)")
    >>> two_break_distance(p, q)
    3
    """
    p_edges = colored_edges(p)
    q_edges = colored_edges(q)
    g = collections.defaultdict(list)
    for (a, b) in it.chain(p_edges, q_edges):
        g[a].append(b)
        g[b].append(a)

    cycles = _trivial_cycles(g)
    for x in it.chain.from_iterable(cycles):
        g.pop(x)

    num_blocks = len(p_edges)

    cycle = []
    stack = []
    while stack or g:
        x = stack.pop() if stack else next(iter(g))
        if x in g:
            cycle.append(x)
            stack.extend(reversed(g[x]))
        elif len(cycle) > 2 and x == cycle[0]:
            cycles.add(frozenset(cycle))
            cycle = []
            stack = []
        g.pop(x, 0)
    return num_blocks - len(cycles)


def _trivial_cycles(g: AdjList) -> Set[Tuple[Hashable, Hashable]]:
    cycles = set()
    for x, ys in g.items():
        if len(set(ys)) == 1:
            y = next(iter(ys))
            tup = (x, y)
            cycles.add(frozenset(tup))
    return cycles


def two_break_sorting(p: Genome, q: Genome) -> List[Genome]:
    """
    >>> p = parse_genome("(+1 -2 -3 +4)")
    >>> q = parse_genome("(+1 +2 -4 -3)")
    >>> [format_genome(genome) for genome in two_break_sorting(p, q)]
    ['(+1 -2 -3 +4)', '(+1 -2 -3)(+4)', '(+1 -2 -4 -3)', '(-3 +1 +2 -4)']
    """
    ...


if __name__ == "__main__":
    genome = parse_genome(input())
    i1, i2, i3, i4 = map(int, input().split(","))
    genome = two_break_on_genome(genome, i1, i2, i3, i4)
    print(format_genome(genome))
