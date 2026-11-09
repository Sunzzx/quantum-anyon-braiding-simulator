"""Interactive CLI and visual demonstration of Quantum Anyon Braiding."""

from __future__ import annotations

import argparse
import sys

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from anyon_braid.braid import BraidWord
from anyon_braid.gates import (
    GATE_T,
    HADAMARD,
    PAULI_X,
    PHASE_S,
    QuantumCircuitCompiler,
    SolovayKitaevSynthesizer,
)
from anyon_braid.hilbert import FusionHilbertSpace
from anyon_braid.invariants import ConsistencyVerifier
from anyon_braid.models import FibonacciAnyonModel, IsingAnyonModel

console = Console()


def _print_banner() -> None:
    banner = Text()
    banner.append("QUANTUM ANYON BRAIDING SIMULATOR\n", style="bold cyan")
    banner.append(
        "Topological QC · Fibonacci / Ising · Yang–Baxter · Solovay–Kitaev\n",
        style="dim",
    )
    console.print(Panel(banner, border_style="cyan"))


def _matrix_table(title: str, mat: np.ndarray) -> Table:
    t = Table(title=title, show_header=True, header_style="bold magenta")
    n = mat.shape[0]
    t.add_column("row\\col", style="dim")
    for j in range(n):
        t.add_column(str(j))
    for i in range(n):
        row = [str(i)]
        for j in range(n):
            z = mat[i, j]
            row.append(f"{z.real:+.4f}{z.imag:+.4f}j")
        t.add_row(*row)
    return t


def demo_fibonacci() -> None:
    """Full live demo: model invariants, braid evolution, gate synthesis."""
    _print_banner()
    model = FibonacciAnyonModel()
    console.print(f"[bold green]Model:[/] {model.name}")
    console.print(f"  φ (golden ratio) = {model.phi:.10f}")
    console.print(f"  d_τ = {model.quantum_dimensions['tau']:.10f}")
    console.print(f"  Total quantum dimension 𝒟 = {model.total_quantum_dimension:.10f}")

    s1, s2 = model.get_braid_generators()
    console.print(_matrix_table("σ₁ (braid gen 1)", s1))
    console.print(_matrix_table("σ₂ (braid gen 2)", s2))

    verifier = ConsistencyVerifier(model)
    report = verifier.run_full_suite()
    status = "[bold green]PASSED[/]" if report.passed else "[bold red]FAILED[/]"
    console.print(
        Panel.fit(
            f"Yang–Baxter err: {report.yang_baxter_error:.2e}\n"
            f"Pentagon err:    {report.pentagon_error:.2e}\n"
            f"Hexagon err:     {report.hexagon_error:.2e}\n"
            f"Unitarity err:   {report.unitarity_error:.2e}\n"
            f"Suite: {status}",
            title="Topological Consistency",
            border_style="green" if report.passed else "red",
        )
    )

    space = FusionHilbertSpace(model, n_anyons=4, anyon_type="tau", total_charge="tau")
    psi0 = space.zero_state()
    console.print(f"\n[bold]Initial state |0⟩:[/] amps={psi0.amplitudes}")

    synth = SolovayKitaevSynthesizer(model, max_depth=6)
    word_h, dist_h, _u_h = synth.synthesize(HADAMARD, tolerance=0.2)
    console.print("\n[bold cyan]Solovay–Kitaev ≈ H[/]")
    console.print(f"  braid: {word_h.to_str()}")
    console.print(f"  operator distance d(U,H) = {dist_h:.6f}")

    psi_h = space.evolve(psi0, word_h)
    console.print(f"  |ψ⟩ after braid: {psi_h.amplitudes}")
    console.print(
        f"  P(|0⟩)={psi_h.probabilities()[0]:.4f}  P(|1⟩)={psi_h.probabilities()[1]:.4f}"
    )

    word_x, dist_x, _ = synth.synthesize(PAULI_X, tolerance=0.25)
    console.print("\n[bold cyan]Solovay–Kitaev ≈ X[/]")
    console.print(f"  braid: {word_x.to_str()}  d={dist_x:.6f}")

    yb = BraidWord.from_indices(4, [1, 2, 1])
    psi_yb = space.evolve(psi0, yb)
    console.print(f"\n[bold]Braid σ₁σ₂σ₁ on |0⟩ →[/] {psi_yb.amplitudes}")
    console.print(f"  probs: {psi_yb.probabilities()}")

    compiler = QuantumCircuitCompiler(synth)
    circuit = [("H", HADAMARD), ("S", PHASE_S), ("T", GATE_T)]
    full_word, total_err = compiler.compile_circuit(circuit)
    console.print(f"\n[bold magenta]Compiled H·S·T circuit braid:[/] {full_word.to_str()}")
    console.print(f"  cumulative approx error Σd = {total_err:.4f}")

    console.print(
        Panel.fit(
            "[bold]Humanity impact:[/] Topological qubits encoded in non-Abelian anyons "
            "are intrinsically protected against local decoherence — a path to fault-tolerant "
            "quantum computers for drug discovery, materials, and climate simulation.",
            border_style="blue",
        )
    )


def demo_ising() -> None:
    _print_banner()
    model = IsingAnyonModel()
    console.print(f"[bold green]Model:[/] {model.name}")
    console.print(f"  d_σ = √2 = {model.quantum_dimensions['sigma']:.10f}")
    console.print(f"  𝒟 = {model.total_quantum_dimension:.10f}")

    s1, s2 = model.get_braid_generators()
    console.print(_matrix_table("σ₁ (Ising)", s1))
    console.print(_matrix_table("σ₂ (Ising)", s2))

    yb_err = ConsistencyVerifier(model).verify_yang_baxter()
    unit_err = ConsistencyVerifier(model).verify_unitarity()
    console.print(f"Yang–Baxter error: {yb_err:.2e}  |  Unitarity error: {unit_err:.2e}")

    space = FusionHilbertSpace(model, n_anyons=4, anyon_type="sigma", total_charge="1")
    psi = space.plus_state()
    braid = BraidWord.from_indices(4, [1, 1, 2, -1])
    psi2 = space.evolve(psi, braid)
    console.print(f"|+⟩ → braid σ₁² σ₂ σ₁⁻¹ → amps={psi2.amplitudes}")
    console.print(f"fidelity(|+⟩, final) = {space.calculate_fidelity(psi, psi2):.6f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="anyon-braid",
        description="Topological Quantum Computation — Anyon Braiding Simulator",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="demo",
        choices=["demo", "fibonacci", "ising", "verify"],
        help="demo|fibonacci|ising|verify",
    )
    args = parser.parse_args(argv)

    if args.command in ("demo", "fibonacci"):
        demo_fibonacci()
    elif args.command == "ising":
        demo_ising()
    elif args.command == "verify":
        model = FibonacciAnyonModel()
        report = ConsistencyVerifier(model).run_full_suite()
        console.print(report)
        return 0 if report.passed else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
