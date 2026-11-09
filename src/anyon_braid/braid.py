"""Artin Braid Group $B_n$ representations and word algebra."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass(frozen=True)
class BraidGenerator:
    """Generator sigma_i or sigma_i^{-1} in Artin Braid Group B_n."""
    index: int  # 1-indexed generator index: 1 <= index < n
    power: int = 1  # +1 for clockwise braid, -1 for counter-clockwise

    def __post_init__(self):
        if self.index < 1:
            raise ValueError("Generator index must be >= 1")
        if self.power not in (1, -1):
            raise ValueError("Generator power must be +1 or -1")

    @property
    def inverse(self) -> "BraidGenerator":
        return BraidGenerator(self.index, -self.power)

    def __repr__(self) -> str:
        inv = "^-1" if self.power == -1 else ""
        return f"σ_{self.index}{inv}"


class BraidWord:
    """Sequence of braid generators representing an element w in B_n."""

    def __init__(self, n_strands: int, generators: List[BraidGenerator] = None):
        if n_strands < 2:
            raise ValueError("n_strands must be >= 2")
        self.n_strands = n_strands
        self.generators = list(generators) if generators else []
        for g in self.generators:
            if g.index >= n_strands:
                raise ValueError(f"Generator index {g.index} exceeds strand limit {n_strands-1}")

    @classmethod
    def from_indices(cls, n_strands: int, indices: List[int]) -> "BraidWord":
        """Construct braid word from list of signed indices, e.g., [1, -2, 1] means sigma_1 sigma_2^-1 sigma_1."""
        gens = []
        for idx in indices:
            if idx == 0:
                raise ValueError("Index cannot be 0")
            power = 1 if idx > 0 else -1
            gens.append(BraidGenerator(abs(idx), power))
        return cls(n_strands, gens)

    def compose(self, other: "BraidWord") -> "BraidWord":
        if self.n_strands != other.n_strands:
            raise ValueError("Cannot compose braids with different strand counts")
        return BraidWord(self.n_strands, self.generators + other.generators)

    def inverse(self) -> "BraidWord":
        return BraidWord(self.n_strands, [g.inverse for g in reversed(self.generators)])

    def simplify(self) -> "BraidWord":
        """Free group reduction: remove adjacent inverses sigma_i sigma_i^-1 = id."""
        reduced = []
        for g in self.generators:
            if reduced and reduced[-1].index == g.index and reduced[-1].power == -g.power:
                reduced.pop()
            else:
                reduced.append(g)
        return BraidWord(self.n_strands, reduced)

    def to_str(self) -> str:
        if not self.generators:
            return "e"  # identity
        return " ".join(str(g) for g in self.generators)

    def __repr__(self) -> str:
        return f"BraidWord(n={self.n_strands}, [{self.to_str()}])"


class BraidGroup:
    """Artin Braid Group B_n evaluator over unitary representations."""

    def __init__(self, n_strands: int, generator_matrices: Dict[int, np.ndarray]):
        """
        Args:
            n_strands: Number of physical anyon strands
            generator_matrices: Mapping of generator index i -> unitary matrix U(sigma_i)
        """
        self.n_strands = n_strands
        self.generator_matrices = {k: np.array(v, dtype=np.complex128) for k, v in generator_matrices.items()}
        self.dim = next(iter(generator_matrices.values())).shape[0]

    def evaluate(self, word: BraidWord) -> np.ndarray:
        """Computes unitary operator U(w) = prod_k U(g_k)."""
        u = np.eye(self.dim, dtype=np.complex128)
        for g in word.generators:
            base_mat = self.generator_matrices[g.index]
            if g.power == 1:
                mat = base_mat
            else:
                mat = base_mat.conj().T  # Unitary inverse
            u = mat @ u  # Operator acting sequentially on quantum state
        return u

    def verify_yang_baxter(self, i: int, j: int) -> float:
        """Verifies the Yang-Baxter relation: sigma_i sigma_{i+1} sigma_i = sigma_{i+1} sigma_i sigma_{i+1}.
        Returns Frobenius norm error ||lhs - rhs||_F.
        """
        s_i = self.generator_matrices[i]
        s_j = self.generator_matrices[j]
        lhs = s_i @ s_j @ s_i
        rhs = s_j @ s_i @ s_j
        return float(np.linalg.norm(lhs - rhs))

    def verify_far_commutation(self, i: int, j: int) -> float:
        """Verifies far-commutation relation: sigma_i sigma_j = sigma_j sigma_i for |i - j| >= 2."""
        s_i = self.generator_matrices[i]
        s_j = self.generator_matrices[j]
        lhs = s_i @ s_j
        rhs = s_j @ s_i
        return float(np.linalg.norm(lhs - rhs))
