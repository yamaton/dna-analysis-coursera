import collections
import functools
import operator
import itertools as it
from typing import Dict, Generator, List
from utils import chunked, reverse_complement, windowed

MassSpectrum = List[int]
Dna = str
Peptide = str


rc = reverse_complement

rna_table = {
    "UUU": "F",
    "UUC": "F",
    "UUA": "L",
    "UUG": "L",
    "UCU": "S",
    "UCC": "S",
    "UCA": "S",
    "UCG": "S",
    "UAU": "Y",
    "UAC": "Y",
    "UGU": "C",
    "UGC": "C",
    "UGG": "W",
    "CUU": "L",
    "CUC": "L",
    "CUA": "L",
    "CUG": "L",
    "CCU": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAU": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGU": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AUU": "I",
    "AUC": "I",
    "AUA": "I",
    "AUG": "M",
    "ACU": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAU": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGU": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GUU": "V",
    "GUC": "V",
    "GUA": "V",
    "GUG": "V",
    "GCU": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAU": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGU": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
    "UAA": "*",
    "UAG": "*",
    "UGA": "*",
}

dna_table = {k.translate(str.maketrans({"U": "T"})): v for k, v in rna_table.items()}

dna_rev_table = collections.defaultdict(set)
for k, v in dna_table.items():
    dna_rev_table[v].add(k)


three_to_one = {
    "Ala": "A",
    "Asx": "B",
    "Cys": "C",
    "Asp": "D",
    "Glu": "E",
    "Phe": "F",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Lys": "K",
    "Leu": "L",
    "Met": "M",
    "Asn": "N",
    "Pro": "P",
    "Gln": "Q",
    "Arg": "R",
    "Ser": "S",
    "Thr": "T",
    "Val": "V",
    "Trp": "W",
    "Tyr": "Y",
    "Glx": "Z",
    "X": "X",  # any codon
    "*": "*",
}


AA_MASS = {
    "G": 57,
    "A": 71,
    "S": 87,
    "P": 97,
    "V": 99,
    "T": 101,
    "C": 103,
    "I": 113,
    "L": 113,
    "N": 114,
    "D": 115,
    "K": 128,
    "Q": 128,
    "E": 129,
    "M": 131,
    "H": 137,
    "F": 147,
    "R": 156,
    "Y": 163,
    "W": 186,
}


def transcribe(dna: str) -> str:
    tb = str.maketrans({"T": "U"})
    return dna.translate(tb)


def translate(rna: str, check_stop_codon=False) -> str:
    """
    >>> translate("AUGGCCAUGGCGCCCAGAACUGAGAUCAAUAGUACCCGUAUUAACGGGUGA", check_stop_codon=True)
    'MAMAPRTEINSTRING'
    """
    assert len(rna) % 3 == 0
    iterator = ("".join(chunk) for chunk in chunked(rna, 3))
    s = "".join(rna_table[codon] for codon in iterator)

    if check_stop_codon:
        assert s[-1] == "*"  # stop codon at last
        assert "*" not in s[:-1]  # no stop codon otherwise
        s = s[:-1]
    return s


def count_distict_dna(protein: str) -> int:
    count_inv_map = collections.defaultdict(int)
    for _, amino in rna_table.items():
        count_inv_map[amino] += 1

    return functools.reduce(operator.mul, (count_inv_map[amino] for amino in protein))


def _dna_candidates(protein: str) -> Generator:
    """
    >>> sorted(list(_dna_candidates("MA")))
    ['ATGGCA', 'ATGGCC', 'ATGGCG', 'ATGGCT']
    """
    d = []
    for amino in protein:
        nuc = dna_rev_table[amino]
        d.append(nuc)
    g = map(lambda xs: "".join(xs), it.product(*d))
    return g


def peptide_encoding(dna: str, peptide: str) -> List[str]:
    """
    >>> dna = "ATGGCCATGGCCCCCAGAACTGAGATCAATAGTACCCGTATTAACGGGTGA"
    >>> peptide_encoding(dna, "MA")
    ['ATGGCC', 'GGCCAT', 'ATGGCC']
    """
    nuc_size = len(peptide) * 3
    iterator = ("".join(tup) for tup in windowed(dna, nuc_size))
    res = [
        s
        for s in iterator
        if peptide in (translate(transcribe(s)), translate(transcribe(rc(s))))
    ]
    return res


def peptide_encoding2(dna: Dna, peptide: Peptide) -> List[Dna]:
    """
    >>> dna = "ATGGCCATGGCCCCCAGAACTGAGATCAATAGTACCCGTATTAACGGGTGA"
    >>> peptide_encoding2(dna, "MA")
    ['ATGGCC', 'GGCCAT', 'ATGGCC']
    """
    candidates = frozenset(_dna_candidates(peptide))
    candidates |= frozenset(rc(s) for s in candidates)
    nuc_size = len(peptide) * 3
    assert len(next(iter(candidates))) == nuc_size
    iterator = ("".join(tup) for tup in windowed(dna, nuc_size))
    res = [s for s in iterator if s in candidates]
    return res


def format_3to1(protein: str) -> str:
    """
    >>> format_3to1("Val-Lys-Leu-Phe-Pro-Trp-Phe-Asn-Gln-Tyr")
    'VKLFPWFNQY'
    """
    aminos = [three_to_one[s] for s in protein.split("-")]
    return "".join(aminos)


def count_subpeptides(n: int) -> int:
    """Number of subpeptides that a cyclic peptide of length n has.

    >>> count_subpeptides(31315)
    980597910
    """
    # there are n amino acides to start from
    # and its length can vary from 1 to (n-1).
    # [NOTE] they don't call cyclic peptide itself as a subpeptide.
    return n * (n - 1)


def linear_spectrum(peptide: Peptide) -> MassSpectrum:
    """
    >>> linear_spectrum("NQEL")
    [0, 113, 114, 128, 129, 242, 242, 257, 370, 371, 484]
    """
    n = len(peptide)
    prefix_mass = [0] * (n + 1)
    for i, aa in enumerate(peptide, 1):
        prefix_mass[i] = prefix_mass[i - 1] + AA_MASS[aa]

    res = [0]
    for i in range(n):
        for j in range(i + 1, n + 1):
            item = prefix_mass[j] - prefix_mass[i]
            res.append(item)

    return sorted(res)


def cyclic_spectrum(peptide: Peptide) -> MassSpectrum:
    """
    >>> cyclic_spectrum("LEQN")
    [0, 113, 114, 128, 129, 227, 242, 242, 257, 355, 356, 370, 371, 484]
    """
    n = len(peptide)
    prefix_mass = [0] * (n + 1)
    for i, aa in enumerate(peptide, 1):
        prefix_mass[i] = prefix_mass[i - 1] + AA_MASS[aa]

    peptide_mass = prefix_mass[n]
    res = [0]
    for i in range(n):
        for j in range(i + 1, n + 1):
            item = prefix_mass[j] - prefix_mass[i]
            res.append(item)
            if 0 < i and j < n:
                item = peptide_mass - (prefix_mass[j] - prefix_mass[i])
                res.append(item)

    return sorted(res)

theoretical_spectrum = cyclic_spectrum



def bf_cyclopeptide_sequencing(spectrum: MassSpectrum) -> Peptide:
    """

    """
    for



if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("input")
    # args = parser.parse_args()
    # with open(args.input, "r") as f:
    #     s = "".join(line.strip() for line in f.readlines())
    s = input().strip()
    res = cyclic_spectrum(s)
    print(*res)
