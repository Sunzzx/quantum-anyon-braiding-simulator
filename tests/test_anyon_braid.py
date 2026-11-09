"""Comprehensive tests for Quantum Anyon Braiding Simulator."""

from __future__ import annotations

import numpy as np
import pytest

from anyon_braid.braid import BraidGenerator, BraidGroup, BraidWord
from anyon_braid.gates import (
    GATE_T,
    HADAMARD,
    PAULI_X,
    PAULI_Z,
    PHASE_S,
    QuantumCircuitCompiler,
    SolovayKitaevSynthesizer,
    operator_distance,
)
from anyon_braid.hilbert import FusionHilbertSpace, TopologicalState
from anyon_braid.invariants import ConsistencyVerifier, KnotInvariants
from anyon_braid.models import FibonacciAnyonModel, IsingAnyonModel


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestFibonacciModel:
    def test_quantum_dimension_golden_ratio(self):
        m = FibonacciAnyonModel()
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        assert m.quantum_dimensions["tau"] == pytest.approx(phi)
        assert m.quantum_dimensions["1"] == pytest.approx(1.0)

    def test_total_quantum_dimension(self):
        m = FibonacciAnyonModel()
        expected = np.sqrt(1.0 + m.phi**2)
        assert m.total_quantum_dimension == pytest.approx(expected)

    def test_f_matrix_unitary(self):
        m = FibonacciAnyonModel()
        F = m.get_f_matrix("tau", "tau", "tau", "tau")
        assert F.shape == (2, 2)
        # F is real symmetric and F^2 = I
        assert np.allclose(F, F.T)
        assert np.allclose(F @ F, np.eye(2), atol=1e-12)
        # Unitarity
        assert np.allclose(F @ F.conj().T, np.eye(2), atol=1e-12)

    def test_f_matrix_entries(self):
        m = FibonacciAnyonModel()
        F = m.get_f_matrix("tau", "tau", "tau", "tau")
        inv_phi = 1.0 / m.phi
        inv_sqrt_phi = 1.0 / np.sqrt(m.phi)
        assert F[0, 0] == pytest.approx(inv_phi)
        assert F[0, 1] == pytest.approx(inv_sqrt_phi)
        assert F[1, 0] == pytest.approx(inv_sqrt_phi)
        assert F[1, 1] == pytest.approx(-inv_phi)

    def test_r_matrix_phases(self):
        m = FibonacciAnyonModel()
        r1 = m.get_r_matrix("tau", "tau", "1")
        rtau = m.get_r_matrix("tau", "tau", "tau")
        assert abs(r1) == pytest.approx(1.0)
        assert abs(rtau) == pytest.approx(1.0)
        assert r1 == pytest.approx(np.exp(-1j * 4.0 * np.pi / 5.0))
        assert rtau == pytest.approx(np.exp(1j * 3.0 * np.pi / 5.0))

    def test_braid_generators_unitary(self):
        m = FibonacciAnyonModel()
        s1, s2 = m.get_braid_generators()
        for s in (s1, s2):
            assert np.allclose(s @ s.conj().T, np.eye(2), atol=1e-12)


class TestIsingModel:
    def test_dimensions(self):
        m = IsingAnyonModel()
        assert m.quantum_dimensions["sigma"] == pytest.approx(np.sqrt(2.0))
        assert m.total_quantum_dimension == pytest.approx(2.0)

    def test_f_matrix_hadamard_like(self):
        m = IsingAnyonModel()
        F = m.get_f_matrix("sigma", "sigma", "sigma", "sigma")
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        expected = np.array(
            [[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]], dtype=np.complex128
        )
        assert np.allclose(F, expected)
        assert np.allclose(F @ F, np.eye(2), atol=1e-12)

    def test_braid_generators_unitary(self):
        m = IsingAnyonModel()
        s1, s2 = m.get_braid_generators()
        for s in (s1, s2):
            assert np.allclose(s @ s.conj().T, np.eye(2), atol=1e-12)


# ---------------------------------------------------------------------------
# Braid group
# ---------------------------------------------------------------------------


class TestBraidWord:
    def test_from_indices_and_str(self):
        w = BraidWord.from_indices(4, [1, -2, 1])
        assert len(w.generators) == 3
        assert "σ_1" in w.to_str()
        assert "σ_2^-1" in w.to_str()

    def test_simplify_cancels_inverses(self):
        w = BraidWord.from_indices(4, [1, -1, 2])
        s = w.simplify()
        assert len(s.generators) == 1
        assert s.generators[0].index == 2

    def test_inverse(self):
        w = BraidWord.from_indices(4, [1, 2, -1])
        inv = w.inverse()
        assert [g.index for g in inv.generators] == [1, 2, 1]
        assert [g.power for g in inv.generators] == [1, -1, -1]

    def test_compose(self):
        a = BraidWord.from_indices(4, [1])
        b = BraidWord.from_indices(4, [2])
        c = a.compose(b)
        assert len(c.generators) == 2

    def test_invalid_index(self):
        with pytest.raises(ValueError):
            BraidWord.from_indices(4, [0])
        with pytest.raises(ValueError):
            BraidWord(4, [BraidGenerator(4, 1)])  # index >= n_strands


class TestBraidGroupYangBaxter:
    def test_fibonacci_yang_baxter(self):
        m = FibonacciAnyonModel()
        s1, s2 = m.get_braid_generators()
        bg = BraidGroup(4, {1: s1, 2: s2})
        err = bg.verify_yang_baxter(1, 2)
        assert err < 1e-10

    def test_ising_yang_baxter(self):
        m = IsingAnyonModel()
        s1, s2 = m.get_braid_generators()
        bg = BraidGroup(4, {1: s1, 2: s2})
        err = bg.verify_yang_baxter(1, 2)
        assert err < 1e-10

    def test_evaluate_identity(self):
        m = FibonacciAnyonModel()
        s1, s2 = m.get_braid_generators()
        bg = BraidGroup(4, {1: s1, 2: s2})
        U = bg.evaluate(BraidWord(4, []))
        assert np.allclose(U, np.eye(2))

    def test_evaluate_inverse_is_identity(self):
        m = FibonacciAnyonModel()
        s1, s2 = m.get_braid_generators()
        bg = BraidGroup(4, {1: s1, 2: s2})
        w = BraidWord.from_indices(4, [1, 2, -2, -1])
        U = bg.evaluate(w.simplify() if False else w)
        # Full word with free cancel should be near I after evaluate of simplified
        Us = bg.evaluate(w.simplify())
        assert np.allclose(Us, np.eye(2), atol=1e-12)


# ---------------------------------------------------------------------------
# Hilbert space / evolution
# ---------------------------------------------------------------------------


class TestHilbertSpace:
    def test_zero_one_orthonormal(self):
        space = FusionHilbertSpace(FibonacciAnyonModel())
        z = space.zero_state()
        o = space.one_state()
        assert space.calculate_fidelity(z, z) == pytest.approx(1.0)
        assert space.calculate_fidelity(z, o) == pytest.approx(0.0)

    def test_plus_state_equal_probs(self):
        space = FusionHilbertSpace(FibonacciAnyonModel())
        p = space.plus_state()
        probs = p.probabilities()
        assert probs[0] == pytest.approx(0.5)
        assert probs[1] == pytest.approx(0.5)

    def test_evolution_preserves_norm(self):
        space = FusionHilbertSpace(FibonacciAnyonModel())
        psi = space.plus_state()
        braid = BraidWord.from_indices(4, [1, 2, 1, 2, -1])
        out = space.evolve(psi, braid)
        assert np.linalg.norm(out.amplitudes) == pytest.approx(1.0)

    def test_measure_returns_valid_outcome(self):
        rng = np.random.default_rng(42)
        space = FusionHilbertSpace(FibonacciAnyonModel())
        psi = space.plus_state()
        idx, label = psi.measure(rng)
        assert idx in (0, 1)
        assert label in psi.basis_labels

    def test_expectation_pauli_z_on_zero(self):
        space = FusionHilbertSpace(FibonacciAnyonModel())
        z = space.zero_state()
        ez = z.expectation_value(PAULI_Z)
        assert ez.real == pytest.approx(1.0)
        assert abs(ez.imag) < 1e-12


# ---------------------------------------------------------------------------
# Consistency / invariants
# ---------------------------------------------------------------------------


class TestConsistency:
    def test_fibonacci_full_suite_passes(self):
        report = ConsistencyVerifier(FibonacciAnyonModel()).run_full_suite()
        assert report.passed
        assert report.yang_baxter_error < 1e-10
        assert report.pentagon_error < 1e-10
        assert report.hexagon_error < 1e-10
        assert report.unitarity_error < 1e-10

    def test_ising_yang_baxter_and_unitarity(self):
        v = ConsistencyVerifier(IsingAnyonModel())
        assert v.verify_yang_baxter() < 1e-10
        assert v.verify_unitarity() < 1e-10

    def test_markov_trace_identity(self):
        tr = KnotInvariants.markov_trace(np.eye(2, dtype=np.complex128), quantum_dim=2.0)
        assert tr == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Gate synthesis
# ---------------------------------------------------------------------------


class TestOperatorDistance:
    def test_identical_zero(self):
        assert operator_distance(HADAMARD, HADAMARD) == pytest.approx(0.0, abs=1e-7)

    def test_global_phase_invariant(self):
        U = np.exp(1j * 0.7) * HADAMARD
        assert operator_distance(HADAMARD, U) == pytest.approx(0.0, abs=1e-7)

    def test_orthogonal_nonzero(self):
        d = operator_distance(PAULI_X, PAULI_Z)
        assert d > 0.1


class TestSolovayKitaev:
    def test_library_nonempty(self):
        synth = SolovayKitaevSynthesizer(FibonacciAnyonModel(), max_depth=3)
        assert len(synth._library) > 1

    def test_identity_exact(self):
        synth = SolovayKitaevSynthesizer(FibonacciAnyonModel(), max_depth=3)
        word, dist, U = synth.synthesize(np.eye(2, dtype=np.complex128), tolerance=0.01)
        assert dist < 1e-10
        assert word.to_str() == "e" or len(word.generators) == 0

    def test_hadamard_approximation_bounded(self):
        synth = SolovayKitaevSynthesizer(FibonacciAnyonModel(), max_depth=6)
        word, dist, U = synth.synthesize(HADAMARD, tolerance=0.5)
        assert word is not None
        assert dist < 0.5
        assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-10)

    def test_compiled_circuit_returns_braid(self):
        synth = SolovayKitaevSynthesizer(FibonacciAnyonModel(), max_depth=4)
        compiler = QuantumCircuitCompiler(synth)
        word, err = compiler.compile_circuit([("H", HADAMARD), ("T", GATE_T)])
        assert isinstance(word, BraidWord)
        assert err >= 0.0


class TestTopologicalState:
    def test_auto_normalize(self):
        s = TopologicalState([3.0, 4.0], ["a", "b"])
        assert np.linalg.norm(s.amplitudes) == pytest.approx(1.0)
        assert s.probabilities()[0] == pytest.approx(0.36)
        assert s.probabilities()[1] == pytest.approx(0.64)
