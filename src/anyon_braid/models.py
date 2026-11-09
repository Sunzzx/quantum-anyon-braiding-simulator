"""Topological Anyon Models: Fusion categories, F-matrices, R-matrices, and topological invariants."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


class AnyonModel(ABC):
    """Abstract base class for Unitary Braided Fusion Categories (Anyon Models)."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def anyon_types(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def quantum_dimensions(self) -> Dict[str, float]:
        pass

    @property
    @abstractmethod
    def total_quantum_dimension(self) -> float:
        pass

    @abstractmethod
    def get_f_matrix(self, a: str, b: str, c: str, d: str) -> np.ndarray:
        """Returns F-matrix F^{abc}_d describing fusion basis re-coupling."""
        pass

    @abstractmethod
    def get_r_matrix(self, a: str, b: str, c: str) -> complex:
        """Returns topological twist/braiding phase R^{ab}_c."""
        pass


class FibonacciAnyonModel(AnyonModel):
    """Fibonacci Anyon Model ($G_2$ level 1 / $SU(2)_3$ quotient).

    Anyon types: '1' (vacuum), 'tau' (non-Abelian anyon).
    Fusion rules: tau x tau = 1 + tau
    Quantum dimension d_tau = phi = (1 + sqrt(5)) / 2 (Golden Ratio).
    Universal for topological quantum computation.
    """

    def __init__(self):
        self.phi = (1.0 + np.sqrt(5.0)) / 2.0
        self._types = ["1", "tau"]
        self._q_dim = {"1": 1.0, "tau": self.phi}
        self._total_dim = np.sqrt(1.0 + self.phi**2)

    @property
    def name(self) -> str:
        return "Fibonacci (Universal Non-Abelian Anyon)"

    @property
    def anyon_types(self) -> List[str]:
        return list(self._types)

    @property
    def quantum_dimensions(self) -> Dict[str, float]:
        return dict(self._q_dim)

    @property
    def total_quantum_dimension(self) -> float:
        return self._total_dim

    def get_f_matrix(self, a: str, b: str, c: str, d: str) -> np.ndarray:
        """F^{abc}_d basis transformation.

        For 4 tau anyons fusing to tau (a=b=c=d='tau'):
        Basis intermediate fusion states: ['1', 'tau']
        F^{tau,tau,tau}_tau = [[phi^-1, phi^-1/2],
                               [phi^-1/2, -phi^-1]]
        """
        inv_phi = 1.0 / self.phi
        inv_sqrt_phi = 1.0 / np.sqrt(self.phi)

        if a == "tau" and b == "tau" and c == "tau" and d == "tau":
            return np.array([[inv_phi, inv_sqrt_phi], [inv_sqrt_phi, -inv_phi]], dtype=np.complex128)
        elif a == "1" or b == "1" or c == "1":
            # Trivial identity re-coupling
            return np.array([[1.0]], dtype=np.complex128)
        else:
            return np.array([[1.0]], dtype=np.complex128)

    def get_r_matrix(self, a: str, b: str, c: str) -> complex:
        """R-matrix topological phase factor R^{ab}_c.

        R^{tau, tau}_1 = exp(-i 4pi / 5)
        R^{tau, tau}_tau = exp(i 3pi / 5)
        """
        if a == "tau" and b == "tau":
            if c == "1":
                return np.exp(-1j * 4.0 * np.pi / 5.0)
            elif c == "tau":
                return np.exp(1j * 3.0 * np.pi / 5.0)
            else:
                raise ValueError(f"Invalid fusion channel {c} for tau x tau")
        elif a == "1" or b == "1":
            return 1.0 + 0.0j
        return 1.0 + 0.0j

    def get_braid_generators(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates sigma_1 and sigma_2 braid matrices acting on the 2D qubit space
        encoded in 4 Fibonacci anyons with total charge tau: |0> = |(tau, tau)->1>, |1> = |(tau, tau)->tau>.

        sigma_1 acts directly in the (1, 2) fusion channel:
        sigma_1 = diag(R^{tau,tau}_1, R^{tau,tau}_tau)

        sigma_2 acts on (2, 3), obtained via F * R * F^{-1}:
        sigma_2 = F @ sigma_1 @ F
        """
        r1 = self.get_r_matrix("tau", "tau", "1")
        rtau = self.get_r_matrix("tau", "tau", "tau")
        sigma_1 = np.diag([r1, rtau])

        F = self.get_f_matrix("tau", "tau", "tau", "tau")
        sigma_2 = F @ sigma_1 @ F
        return sigma_1, sigma_2


class IsingAnyonModel(AnyonModel):
    """Ising Anyon Model (Majorana Zero Modes / fractional quantum Hall nu=5/2).

    Anyon types: '1' (vacuum), 'sigma' (Majorana vortex), 'psi' (chiral fermion).
    Fusion rules:
      sigma x sigma = 1 + psi
      sigma x psi = sigma
      psi x psi = 1
    Quantum dimensions: d_1=1, d_sigma=sqrt(2), d_psi=1.
    Total dimension D = 2.
    """

    def __init__(self):
        self._types = ["1", "sigma", "psi"]
        self._q_dim = {"1": 1.0, "sigma": np.sqrt(2.0), "psi": 1.0}
        self._total_dim = 2.0

    @property
    def name(self) -> str:
        return "Ising / Majorana Zero Mode (Clifford Topological Model)"

    @property
    def anyon_types(self) -> List[str]:
        return list(self._types)

    @property
    def quantum_dimensions(self) -> Dict[str, float]:
        return dict(self._q_dim)

    @property
    def total_quantum_dimension(self) -> float:
        return self._total_dim

    def get_f_matrix(self, a: str, b: str, c: str, d: str) -> np.ndarray:
        if a == "sigma" and b == "sigma" and c == "sigma" and d == "sigma":
            # F^{sigma, sigma, sigma}_sigma basis ['1', 'psi']
            inv_sqrt_2 = 1.0 / np.sqrt(2.0)
            return np.array([[inv_sqrt_2, inv_sqrt_2], [inv_sqrt_2, -inv_sqrt_2]], dtype=np.complex128)
        return np.array([[1.0]], dtype=np.complex128)

    def get_r_matrix(self, a: str, b: str, c: str) -> complex:
        if a == "sigma" and b == "sigma":
            if c == "1":
                return np.exp(-1j * np.pi / 8.0)
            elif c == "psi":
                return np.exp(1j * 3.0 * np.pi / 8.0)
        return 1.0 + 0.0j

    def get_braid_generators(self) -> Tuple[np.ndarray, np.ndarray]:
        r1 = self.get_r_matrix("sigma", "sigma", "1")
        rpsi = self.get_r_matrix("sigma", "sigma", "psi")
        sigma_1 = np.diag([r1, rpsi])

        F = self.get_f_matrix("sigma", "sigma", "sigma", "sigma")
        sigma_2 = F @ sigma_1 @ F
        return sigma_1, sigma_2
