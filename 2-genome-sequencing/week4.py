import collections
from types import FunctionType
from typing import Counter, Deque, Dict, List, Optional, Set, Tuple, Callable
import itertools as it

from week3 import (
    AA_MASS,
    AA_MASS_REDUCED,
    _expand,
    _parent_mass,
    cyclic_spectrum,
    linear_spectrum,
)

Peptide = str
MassSpectrum = List[int]
Leaderboard = Set[Peptide]
PeptideToSpectrum = Callable[[Peptide], MassSpectrum]

# Let latin-1 supplement [U+00A1, U+00A2, ...] correspond to [57, 58, ..., 200]
offset = 0x00A1
span = range(57, 201)
int2char = {(57 + i): chr(offset + i) for i, _ in enumerate(span)}
AA_MASS_EXTENDED = AA_MASS_REDUCED.copy()
masses = frozenset(AA_MASS_EXTENDED.values())
for i in span:
    if i not in masses:
        c = int2char[i]
        AA_MASS_EXTENDED[c] = i


def _scoring_helper(
    to_spectrum: PeptideToSpectrum,
    peptide: Peptide,
    spectrum: MassSpectrum,
    aa_set=AA_MASS_EXTENDED,
) -> int:
    estimated = collections.Counter(to_spectrum(peptide, aa_set))
    experiment = collections.Counter(spectrum)
    num_overlap = sum((estimated & experiment).values())
    return num_overlap


def cyclopeptide_score(
    peptide: Peptide, spectrum: MassSpectrum, aa_set=AA_MASS_EXTENDED
) -> int:
    """
    >>> spectrum = [0, 99, 113, 114, 128, 227, 257, 299, 355, 356, 370, 371, 484]
    >>> cyclopeptide_score("NQEL", spectrum, AA_MASS)
    11
    """
    return _scoring_helper(cyclic_spectrum, peptide, spectrum, aa_set)


def linearpeptide_score(
    peptide: Peptide, spectrum: MassSpectrum, aa_set=AA_MASS_EXTENDED
) -> int:
    """
    >>> spectrum = [0, 99, 113, 114, 128, 227, 257, 299, 355, 356, 370, 371, 484]
    >>> linearpeptide_score("NQEL", spectrum, AA_MASS)
    8
    """
    return _scoring_helper(linear_spectrum, peptide, spectrum, aa_set)


def leaderboard_cyclopeptide_sequencing(
    spectrum: MassSpectrum, n: int, aa_set: Optional[Dict[str, int]] = None
) -> Set[Peptide]:
    """cyclopeptide sequencing with scoring

    >>> spectrum = [0, 71, 113, 129, 147, 200, 218, 260, 313, 331, 347, 389, 460]
    >>> computed = leaderboard_cyclopeptide_sequencing(spectrum, 10)
    >>> expected = {'AEIF', 'AFIE', 'EAFI', 'IFAE', 'IEAF', 'FIEA', 'FAEI', 'EIFA'}
    >>> computed == expected
    True
    """
    leaderboard: Leaderboard = {""}
    leader_score = 0
    leader_peptide = set()

    while leaderboard:
        leaderboard = _expand(leaderboard, aa_set)
        to_remove = set()
        for peptide in leaderboard:
            if _mass(peptide) == _parent_mass(spectrum):
                x = cyclopeptide_score(peptide, spectrum)
                if x > leader_score:
                    leader_score = x
                    leader_peptide = {peptide}
                elif x == leader_score:
                    leader_peptide.add(peptide)

            elif _mass(peptide) > _parent_mass(spectrum):
                to_remove.add(peptide)
        leaderboard -= to_remove
        leaderboard = _trim(leaderboard, spectrum, n)
    return leader_peptide


def _mass(peptide: Peptide) -> int:
    return sum(AA_MASS_EXTENDED[aa] for aa in peptide)


def _trim(leaderboard: Leaderboard, spectrum: MassSpectrum, n: int) -> Leaderboard:
    """
    >>> leaderboard = {"LAST", "ALST", "TLLT", "TQAS"}
    >>> spectrum = [0, 71, 87, 101, 113, 158, 184, 188, 259, 271, 372]
    >>> computed = _trim(leaderboard, spectrum, 2)
    >>> expected = {"LAST", "ALST"}
    >>> computed == expected
    """
    if len(leaderboard) < n:
        return leaderboard
    pairs = [
        (linearpeptide_score(peptide, spectrum), peptide) for peptide in leaderboard
    ]
    pairs.sort(reverse=True)
    threshold, _ = pairs[n]
    res = set()
    for score, peptide in pairs:
        if score < threshold:
            break
        res.add(peptide)
    return res


def spectral_convolution(spectrum: MassSpectrum, k: int) -> List[Tuple[int, int]]:
    """Find convolution of a spectrum with multiplicity k or more

    >>> spectral_convolution([0, 137, 186, 323], 2)
    [(2, 137), (2, 186)]
    """
    counter = collections.Counter(
        b - a for (a, b) in it.combinations(spectrum, 2) if a < b
    )
    counter = [(cnt, diff) for diff, cnt in counter.items() if k <= cnt]
    return sorted(counter, key=lambda tup: (-tup[0], tup[1]))


def convolution_cyclopeptide_sequencing(
    spectrum: MassSpectrum, m: int, n: int
) -> Set[Peptide]:
    """Find cyclic peptide sequence from m most frequent spectral convolution elements (with tie),
    and the internal leaderboard is trimmed to n items (with ties as well).

    >>> spectrum = [57, 57, 71, 99, 129, 137, 170, 186, 194, 208, 228, 265, 285, 299, 307, 323, 356, 364, 394, 422, 493]
    >>> convolution_cyclopeptide_sequencing(spectrum, 20, 60)
    True
    """
    pairs = spectral_convolution(spectrum, 1)
    threshold, _ = pairs[m - 1]
    masses = {m for cnt, m in pairs if threshold <= cnt}
    aa_set = {c: mass for c, mass in AA_MASS_EXTENDED.items() if mass in masses}
    return leaderboard_cyclopeptide_sequencing(spectrum, n, aa_set)


if __name__ == "__main__":
    # m = int(input())
    # n = int(input())
    # spectrum = [int(s) for s in input().split()]
    # ps = convolution_cyclopeptide_sequencing(spectrum, m, n)
    # print(ps)
    # res = [peptide_as_hyphenated_mass(p, AA_MASS_EXTENDED) for p in ps]
    # print(res[0])

    peptide = input().strip()
    spectrum = [int(s) for s in input().split()]
    res = linearpeptide_score(peptide, spectrum, AA_MASS)
    print(res)
