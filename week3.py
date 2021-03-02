
from typing import Set, List
from utils import hamming_distance, neighbors, sliding_window


def hamming_distance_str(pattern: str, text: str):
    assert len(pattern) <= len(text)
    return min(hamming_distance(pattern, seg) for seg in sliding_window(text, len(pattern)))


def motif_enumeration(dna: List[str], k: int, d: int) -> Set[str]:
    """Search for all (k, d)-motifs in dna with brute force.
    O(t * n * X * t * n)
        where t = len(dna)
              n = len(dna[0])
              X = time complexity of neighbors()
    >>> motif_enumeration({"ATTTGGC", "TGCCTTA", "CGGTATC", "GAAAATT"}, 3, 1) == {"ATA", "ATT", "GTT", "TTT"}
    True
    """
    seen = set()
    res = set()
    for seq in dna:
        for seed in sliding_window(seq, k):
            if seed in seen:
                continue
            seen.add(seed)
            for pattern in neighbors(seed, d):
                if all(hamming_distance_str(pattern, s) <= d for s in dna):
                    res.add(pattern)
    return res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()
    with open(args.input, "r") as f:
        k, d =  map(int, f.readline().strip().split())
        dna = {line.strip() for line in f.readlines()}
    res = motif_enumeration(dna, k, d)
    print(*res)
