"""Topological Hilbert Space, Anyon Fusion Trees, and Quantum State Evolution."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from .models import AnyonModel, FibonacciAnyonModel, IsingAnyonModel
from .braid import BraidWord, BraidGroup


@dataclass
class TopologicalState:
    """Pure topological quantum state vector in the degenerate ground space."""
    amplitudes: np.ndarray
    basis_labels: List[str]

    def __post_init__(self):
        self.amplitudes = np.array(self.amplitudes, dtype=np.complex128)
        norm = np.linalg.norm(self.amplitudes)
        if norm > 1e-12:
            self.amplitudes /= norm

    @property
    def dim(self) -> int:
        return len(self.amplitudes)

    def probabilities(self) -> np.ndarray:
        return np.abs(self.amplitudes) ** 2

    def expectation_value(self, operator: np.ndarray) -> complex:
        return complex(np.vdot(self.amplitudes, operator @ self.amplitudes))

    def measure(self, rng: Optional[np.random.Generator] = None) -> Tuple[int, str]:
        """Projective non-demolition fusion measurement on topological state."""
        probs = self.probabilities()
        probs = probs / np.sum(probs)
        if rng is None:
            idx = np.random.choice(len(probs), p=probs)
        else:
            idx = rng.choice(len(probs), p=probs)
        return idx, self.basis_labels[idx]


class FusionHilbertSpace:
    """Topological Hilbert Space encoded by N anyons of type anyon_type."""

    def __init__(self, model: AnyonModel, n_anyons: int = 4, anyon_type: str = "tau", total_charge: str = "tau"):
        self.model = model
        self.n_anyons = n_anyons
        self.anyon_type = anyon_type
        self.total_charge = total_charge
        self._init_basis()
        self._init_braid_group()

    def _init_basis(self):
        """Construct fusion tree basis.
        For 4 Fibonacci tau anyons with total charge tau:
        ((tau, tau) -> intermediate_1, (tau, tau) -> intermediate_2) or standard tree: |((tau, tau)_x, tau)->y, tau)->tau.
        For 4 tau anyons fusing to tau:
        Tree: ((tau_1, tau_2) -> x, tau_3) -> tau_4 where total charge is tau.
        Dimension = Fib(4 - 1) = Fib(3) = 2.
        Basis vectors: |x=1> and |x=tau>, labeled |0> and |1>.
        """
        if isinstance(self.model, FibonacciAnyonModel):
            if self.n_anyons == 4 and self.total_charge == "tau":
                self.basis_labels = ["|0⟩ (|x=1⟩)", "|1⟩ (|x=τ⟩)"]
                self.dim = 2
            else:
                self.basis_labels = [f"|basis_{i}⟩" for i in range(2)]
                self.dim = 2
        elif isinstance(self.model, IsingAnyonModel):
            if self.n_anyons == 4 and self.total_charge == "1":
                self.basis_labels = ["|0⟩ (|x=1⟩)", "|1⟩ (|x=ψ⟩)"]
                self.dim = 2
            else:
                self.basis_labels = ["|0⟩ (|x=1⟩)", "|1⟩ (|x=ψ⟩)"]
                self.dim = 2
        else:
            self.basis_labels = ["|0⟩", "|1⟩"]
            self.dim = 2

    def _init_braid_group(self):
        """Generates braid group representations for the encoded qubits."""
        s1, s2 = self.model.get_braid_generators()
        self.braid_group = BraidGroup(n_strands=4, generator_matrices={1: s1, 2: s2})

    def zero_state(self) -> TopologicalState:
        return TopologicalState(np.array([1.0, 0.0], dtype=np.complex128), self.basis_labels)

    def one_state(self) -> TopologicalState:
        return TopologicalState(np.array([0.0, 1.0], dtype=np.complex128), self.basis_labels)

    def plus_state(self) -> TopologicalState:
        return TopologicalState(np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=np.complex128), self.basis_labels)

    def evolve(self, state: TopologicalState, braid: BraidWord) -> TopologicalState:
        """Applies unitary evolution via braiding word: |psi_final> = U(braid) |psi_initial>."""
        U = self.braid_group.evaluate(braid)
        new_amplitudes = U @ state.amplitudes
        return TopologicalState(new_amplitudes, self.basis_labels)

    def calculate_fidelity(self, state_a: TopologicalState, state_b: TopologicalState) -> float:
        """Calculates quantum state fidelity F = |<psi_a|psi_b>|^2."""
        inner_prod = np.vdot(state_a.amplitudes, state_b.amplitudes)
        return float(np.abs(inner_prod) ** 2)
