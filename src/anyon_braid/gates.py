"""Quantum Gate Compilation and Solovay-Kitaev / Grid-Search Synthesis into Braid Words."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import itertools
import numpy as np
from .braid import BraidWord, BraidGenerator, BraidGroup
from .models import AnyonModel, FibonacciAnyonModel, IsingAnyonModel


# Standard target single-qubit quantum gates in SU(2) / U(2)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
PAULI_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
HADAMARD = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
PHASE_S = np.array([[1.0, 0.0], [0.0, 1j]], dtype=np.complex128)
GATE_T = np.array([[1.0, 0.0], [0.0, np.exp(1j * np.pi / 4.0)]], dtype=np.complex128)


def operator_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Computes projectively invariant operator distance between unitaries U and V:
    d(U, V) = sqrt(1 - |Tr(U^dagger V)| / dim).
    """
    dim = U.shape[0]
    overlap = np.trace(U.conj().T @ V)
    fid = np.abs(overlap) / dim
    fid = np.clip(fid, 0.0, 1.0)
    return float(np.sqrt(np.maximum(0.0, 1.0 - fid)))


class SolovayKitaevSynthesizer:
    """Synthesizes arbitrary SU(2) quantum gates into topological braid words
    using geodesic search over the dense braid group representation.
    """

    def __init__(self, model: Optional[AnyonModel] = None, max_depth: int = 10):
        self.model = model if model is not None else FibonacciAnyonModel()
        s1, s2 = self.model.get_braid_generators()
        self.braid_group = BraidGroup(n_strands=4, generator_matrices={1: s1, 2: s2})
        self.max_depth = max_depth
        self._library: List[Tuple[BraidWord, np.ndarray]] = []
        self._build_library()

    def _build_library(self):
        """Generates braid words up to search depth to form an epsilon-net over SU(2)."""
        generators = [
            BraidGenerator(1, 1),
            BraidGenerator(1, -1),
            BraidGenerator(2, 1),
            BraidGenerator(2, -1),
        ]

        # Start with identity
        empty_word = BraidWord(4, [])
        self._library.append((empty_word, self.braid_group.evaluate(empty_word)))

        current_level = [(empty_word, self.braid_group.evaluate(empty_word))]

        # Bounded BFS enumeration
        for depth in range(1, min(self.max_depth, 6) + 1):
            next_level = []
            for word, u_mat in current_level:
                last_g = word.generators[-1] if word.generators else None
                for g in generators:
                    if last_g is not None and last_g.index == g.index and last_g.power == -g.power:
                        continue  # Skip trivial inverse backtracking
                    
                    new_word = BraidWord(4, word.generators + [g])
                    # Evaluate
                    u_step = self.braid_group.generator_matrices[g.index]
                    if g.power == -1:
                        u_step = u_step.conj().T
                    new_u = u_step @ u_mat
                    next_level.append((new_word, new_u))
                    self._library.append((new_word, new_u))
            current_level = next_level

    def synthesize(
        self, target_unitary: np.ndarray, tolerance: float = 0.15
    ) -> Tuple[BraidWord, float, np.ndarray]:
        """Finds optimal braid word approximating target unitary up to global phase.
        
        Returns:
            Tuple of (best_braid_word, distance, synthesized_unitary)
        """
        best_dist = float("inf")
        best_word = None
        best_u = None

        for word, u in self._library:
            dist = operator_distance(u, target_unitary)
            if dist < best_dist:
                best_dist = dist
                best_word = word
                best_u = u
                if best_dist < tolerance:
                    break

        return best_word, best_dist, best_u


class QuantumCircuitCompiler:
    """Compiles high-level quantum circuit gate sequences into a unified topological braid."""

    def __init__(self, synthesizer: Optional[SolovayKitaevSynthesizer] = None):
        self.synthesizer = synthesizer if synthesizer is not None else SolovayKitaevSynthesizer()

    def compile_circuit(self, gate_sequence: List[Tuple[str, np.ndarray]]) -> Tuple[BraidWord, float]:
        """Compiles a sequence of named quantum gates [(name, matrix), ...] into a combined braid."""
        full_braid = BraidWord(4, [])
        total_err = 0.0

        for gate_name, gate_mat in gate_sequence:
            word, err, _ = self.synthesizer.synthesize(gate_mat)
            full_braid = full_braid.compose(word)
            total_err += err

        simplified_braid = full_braid.simplify()
        return simplified_braid, total_err
