# Quantum Anyon Braiding Simulator

**Topological Quantum Computation engine** implementing non-Abelian anyon models (Fibonacci & Ising/Majorana), Artin braid-group unitary representations, Mac Lane pentagon / Moore–Seiberg hexagon consistency, and Solovay–Kitaev-style gate synthesis into braid words.

[![CI](https://github.com/Sunzzx/quantum-anyon-braiding-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/Sunzzx/quantum-anyon-braiding-simulator/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?logo=github)](https://github.com/sponsors/Sunzzx)
[![Polar](https://img.shields.io/badge/Support-Polar.sh-blue)](https://polar.sh/Sunzzx)

---

## Humanity & Real-World Impact

Fault-tolerant quantum computers remain blocked by decoherence. **Topological qubits** encode information in the global fusion space of non-Abelian anyons (e.g. Fibonacci $\tau$ particles or Ising $\sigma$ Majorana zero modes). Local noise cannot change the topological charge without a non-local operation, giving **intrinsic error protection** without continuous active correction.

This simulator makes those mathematics executable:

- Prototype **braid-compiled quantum circuits** for chemistry and materials.
- Validate **fusion-category consistency** (Yang–Baxter, pentagon, hexagon) before hardware commitment.
- Educate the next generation of topological QC researchers with live, dependency-light demos.

Impact domains: drug binding / protein dynamics on future topological QPUs, climate PDE solvers, secure multiparty crypto, and accessible open science for quantum error correction.

Commercial / enterprise licensing & support: contact `ayushkandwal2007@gmail.com` · Stripe / Polar sponsorships welcome · GitHub Sponsors: [Sunzzx](https://github.com/sponsors/Sunzzx).

---

## Theoretical Foundations

### Fusion rules & quantum dimensions

**Fibonacci anyons** (universal for TQC):

$$
\tau \times \tau = 1 + \tau, \qquad d_\tau = \varphi = \frac{1+\sqrt{5}}{2}, \qquad \mathcal{D} = \sqrt{1+\varphi^2}.
$$

**Ising anyons** (Majorana / $\nu=5/2$ FQHE):

$$
\sigma \times \sigma = 1 + \psi, \quad \sigma \times \psi = \sigma, \quad \psi \times \psi = 1,
$$
$$
d_\sigma = \sqrt{2}, \qquad \mathcal{D} = 2.
$$

### $F$-moves (associators) and $R$-matrices (braiding)

Recoupling of fusion trees is a unitary $F$-matrix. For four Fibonacci $\tau$ anyons:

$$
F^{\tau\tau\tau}_{\tau} =
\begin{pmatrix}
\varphi^{-1} & \varphi^{-1/2} \\
\varphi^{-1/2} & -\varphi^{-1}
\end{pmatrix},
\qquad
(F^{\tau\tau\tau}_{\tau})^2 = I.
$$

Topological braid phases:

$$
R^{\tau\tau}_{1} = e^{-i 4\pi/5}, \qquad R^{\tau\tau}_{\tau} = e^{i 3\pi/5}.
$$

Encoded qubit on four $\tau$ anyons with total charge $\tau$:

$$
|0\rangle = |(\tau,\tau)\to 1\rangle, \qquad |1\rangle = |(\tau,\tau)\to \tau\rangle.
$$

Braid generators on this 2-dimensional fusion space:

$$
\sigma_1 = \mathrm{diag}\bigl(R^{\tau\tau}_{1},\, R^{\tau\tau}_{\tau}\bigr),
\qquad
\sigma_2 = F\,\sigma_1\,F.
$$

### Artin braid group $B_n$

Generators $\sigma_i$ obey the **Yang–Baxter** (braid) relation and far commutation:

$$
\sigma_i \sigma_{i+1} \sigma_i = \sigma_{i+1} \sigma_i \sigma_{i+1},
\qquad
\sigma_i \sigma_j = \sigma_j \sigma_i \quad (|i-j|\ge 2).
$$

A braid word $w=\sigma_{i_1}^{\epsilon_1}\cdots\sigma_{i_k}^{\epsilon_k}$ evolves the topological state as

$$
|\psi_{\mathrm{out}}\rangle = U(w)\,|\psi_{\mathrm{in}}\rangle, \qquad U(w)=\prod_k U(\sigma_{i_k})^{\epsilon_k}.
$$

### Consistency axioms (unitary braided fusion category)

- **Mac Lane pentagon** (associator coherence) — here verified as $F^2=I$ and $F=F^\top$ for Fibonacci.
- **Moore–Seiberg hexagon** — coupling of $F$ and $R$; equivalent to $\sigma_2 = F\sigma_1 F$.
- **Unitarity** — $U U^\dagger = I$ for every braid generator.

### Gate synthesis (dense braid subgroup of $\mathrm{SU}(2)$)

Fibonacci braids generate a dense subgroup of $\mathrm{SU}(2)$. We build an $\varepsilon$-net by bounded BFS over $\{\sigma_1^{\pm1},\sigma_2^{\pm1}\}$ and minimize the projectively invariant distance

$$
d(U,V) = \sqrt{1 - \frac{|\mathrm{Tr}(U^\dagger V)|}{\dim}}.
$$

---

## Architecture

```
┌──────────────────┐     F, R matrices      ┌─────────────────────┐
│  AnyonModel      │ ─────────────────────► │  BraidGroup B₄      │
│  Fibonacci/Ising │                        │  σ₁, σ₂ unitaries   │
└────────┬─────────┘                        └──────────┬──────────┘
         │                                             │
         ▼                                             ▼
┌──────────────────┐   braid word U(w)   ┌─────────────────────┐
│ FusionHilbertSpace│ ◄──────────────────│  BraidWord algebra  │
│ |0⟩,|1⟩,|+⟩ evolve│                    │  simplify / inverse │
└────────┬─────────┘                     └──────────┬──────────┘
         │                                          │
         ▼                                          ▼
┌──────────────────┐                     ┌─────────────────────┐
│ ConsistencyVerifier│                   │ SolovayKitaevSynth  │
│ YB / pentagon /    │                   │ H,X,S,T → braids    │
│ hexagon / unitarity│                   │ CircuitCompiler     │
└──────────────────┘                     └─────────────────────┘
```

| Module | Role |
|--------|------|
| `models.py` | Fusion categories, $F$/$R$, braid generators |
| `braid.py` | Artin $B_n$ words, Yang–Baxter checks |
| `hilbert.py` | Fusion-space states & unitary evolution |
| `gates.py` | $\varepsilon$-net synthesis & circuit compile |
| `invariants.py` | Pentagon/hexagon/YB/Markov trace |
| `cli.py` | Live Rich demo |

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Evaluate braid word length $L$ | $O(L \cdot d^3)$ matrix mults ($d=2$) | $O(d^2)$ |
| Yang–Baxter check | $O(d^3)$ | $O(d^2)$ |
| BFS synthesis depth $D$, branching $\le 4$ | $O(4^D \cdot d^3)$ | $O(4^D \cdot d^2)$ |
| Full consistency suite | $O(d^3)$ | $O(d^2)$ |

---

## Quickstart

```bash
# clone
git clone https://github.com/Sunzzx/quantum-anyon-braiding-simulator.git
cd quantum-anyon-braiding-simulator

# install (uv or pip)
uv pip install -e ".[dev]"
# or: pip install -e ".[dev]"

# tests
uv run pytest -q

# live demo
uv run python -m anyon_braid.cli demo
uv run python -m anyon_braid.cli ising
uv run python -m anyon_braid.cli verify
# or entrypoint:
anyon-braid demo
```

### Library usage

```python
from anyon_braid import FibonacciAnyonModel, FusionHilbertSpace, BraidWord
from anyon_braid.gates import HADAMARD, SolovayKitaevSynthesizer
from anyon_braid.invariants import ConsistencyVerifier

model = FibonacciAnyonModel()
assert ConsistencyVerifier(model).run_full_suite().passed

space = FusionHilbertSpace(model)
psi = space.zero_state()
word, dist, U = SolovayKitaevSynthesizer(model).synthesize(HADAMARD)
psi_h = space.evolve(psi, word)
print(word.to_str(), dist, psi_h.probabilities())
```

---

## Packaging

- `pyproject.toml` (hatchling) · MIT `LICENSE` · GitHub Actions CI (3.10–3.12)
- Zero heavy ML deps: **numpy** + **rich** only

---

## Citation

```bibtex
@software{kandwal2026anyon,
  author = {Kandwal, Ayush},
  title  = {Quantum Anyon Braiding Simulator},
  year   = {2026},
  url    = {https://github.com/Sunzzx/quantum-anyon-braiding-simulator}
}
```

---

*Built by Hermes Autonomous Frontier Research Engine · Ayush Kandwal (Sunzzx)*
