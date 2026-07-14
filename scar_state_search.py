"""
Numerical scar-state search for a j_max = 1/2 Ising-limit plaquette chain.

WARNING:
    This file only studies the j_max = 1/2 Ising-limit/truncated model. It does
    not implement the full j_max > 1/2 SU(2) Hamiltonian with Wigner 6j symbols.
    Therefore, this can reproduce the scar-search logic of the paper in the
    truncated model, but not the full higher-representation SU(2) calculation.

The script applies the canonical Ising-limit Hamiltonian in
``ising_limit_model.py`` to an arbitrary periodic chain of N spins, diagonalises it, computes
half-chain entanglement entropy, and identifies highly excited low-entanglement
outliers as candidate quantum many-body scar states.

Example:
    python scar_state_search.py --N 10 --mode paper --diag dense
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ising_limit_model import (
    HamiltonianMode,
    IsingLimitParameters,
    build_sparse_hamiltonian as build_canonical_sparse_hamiltonian,
    diagonal_energy as canonical_diagonal_energy,
    flip_amplitude as canonical_flip_amplitude,
)


def bit_at(state: int, i: int, N: int) -> int:
    """
    Return bit i of a length-N basis state, using physical site ordering.

    Site i = 0 is the leftmost bit in state_to_bitstring(state, N). This matches
    the row-major reshape convention used for bipartite entanglement.
    """
    if i < 0 or i >= N:
        raise IndexError(f"site index i must lie in 0..{N - 1}")
    return (state >> (N - 1 - i)) & 1


def z_value(state: int, i: int, N: int) -> int:
    """Return the sigma_z eigenvalue at physical site i: bit 0 -> +1, bit 1 -> -1."""
    return 1 - 2 * bit_at(state, i, N)


def flip_bit(state: int, i: int) -> int:
    """
    Flip raw integer bit position i.

    Raw bit position i counts from the least significant bit. For a physical
    left-to-right site i_phys in an N-spin chain, use raw position N - 1 - i_phys.
    """
    if i < 0:
        raise IndexError("raw bit index i must be non-negative")
    return state ^ (1 << i)


def state_to_bitstring(state: int, N: int) -> str:
    """Return the computational-basis bitstring with exactly N bits."""
    return format(state, f"0{N}b")


def diagonal_energy(state: int, N: int, J: float) -> float:
    """Compute sum_i [J Z_i Z_{i+1} - 2J Z_i] with periodic boundaries."""
    params = IsingLimitParameters(
        n_sites=N,
        J=J,
        hx=0.0,
        periodic=True,
        mode=HamiltonianMode.TOY,
    )
    return canonical_diagonal_energy(state, params)


def offdiag_flip_coefficient(state: int, i: int, N: int, hx: float, mode: str) -> float:
    """
    Return the coefficient for flipping physical site i.

    Off-diagonal convention:
      - "toy": follows ``HamiltonianMode.TOY``, hx * nu_i X_i with
        nu_i = (Z_{i-1} + Z_{i+1}) / sqrt(2).
      - "paper": uses the appendix magnetic factor
        -2 * hx * ((1 - 3*Z_{i-1})/4) * ((1 - 3*Z_{i+1})/4).

    The neighbours are evaluated on the ket basis state. Since these factors
    depend only on neighbours and X_i flips only site i, this gives a Hermitian
    matrix for the real coefficients used here.
    """
    params = IsingLimitParameters(
        n_sites=N,
        J=0.0,
        hx=hx,
        periodic=True,
        mode=HamiltonianMode(mode),
    )
    return canonical_flip_amplitude(state, i, params)


def build_ising_hamiltonian_sparse(N: int, J: float, hx: float, mode: str = "paper") -> sp.csr_matrix:
    """
    Build the periodic j_max = 1/2 Ising-limit Hamiltonian as a sparse matrix.

    Basis states are integers 0..2**N-1, ordered as ordinary binary strings.
    The row/column index is therefore the computational basis index.
    """
    params = IsingLimitParameters(
        n_sites=N,
        J=J,
        hx=hx,
        periodic=True,
        mode=HamiltonianMode(mode),
    )
    return build_canonical_sparse_hamiltonian(params)


def diagonalize_hamiltonian(
    H: sp.csr_matrix,
    diag: str,
    num_eigs: int,
    sigma: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonalise H.

    Dense mode returns the full spectrum. Sparse mode uses eigsh and is intended
    for states around a target energy, usually the spectral middle.
    """
    dim = H.shape[0]
    if diag == "dense":
        dense = H.toarray()
        energies, vectors = np.linalg.eigh(dense)
    elif diag == "sparse":
        if dim <= 2:
            raise ValueError("Sparse diagonalisation requires dimension > 2.")
        k = min(num_eigs, dim - 2)
        if k < 1:
            raise ValueError("num_eigs must request at least one eigenstate.")
        if sigma is None:
            # Target the middle of the spectral range using a cheap extremal estimate.
            e_min = float(spla.eigsh(H, k=1, which="SA", return_eigenvectors=False)[0])
            e_max = float(spla.eigsh(H, k=1, which="LA", return_eigenvectors=False)[0])
            sigma = 0.5 * (e_min + e_max)
        energies, vectors = spla.eigsh(H, k=k, sigma=sigma, which="LM")
        order = np.argsort(energies)
        energies = energies[order]
        vectors = vectors[:, order]
    else:
        raise ValueError("diag must be 'dense' or 'sparse'.")

    return np.real_if_close(energies).astype(float), vectors


def entanglement_entropy_statevector(
    psi: np.ndarray,
    N: int,
    NA: Optional[int] = None,
    log_base: float = np.e,
) -> float:
    """
    Compute S_A = -Tr rho_A log(rho_A) for the first NA spins.

    psi is ordered in the computational basis. Reshaping into
    (2**NA, 2**(N-NA)) therefore partitions the first NA physical sites from
    the remaining sites.
    """
    if NA is None:
        NA = N // 2
    if NA <= 0 or NA >= N:
        raise ValueError("NA must satisfy 1 <= NA <= N - 1.")

    dim = 2**N
    psi = np.asarray(psi, dtype=complex)
    if psi.shape != (dim,):
        raise ValueError(f"psi must have shape {(dim,)}, got {psi.shape}.")

    norm = np.linalg.norm(psi)
    if norm == 0:
        raise ValueError("psi has zero norm.")
    psi = psi / norm

    dA = 2**NA
    dB = 2 ** (N - NA)
    Psi = psi.reshape(dA, dB)
    rho_A = Psi @ Psi.conj().T
    lambdas = np.linalg.eigvalsh(rho_A)
    lambdas = np.clip(np.real_if_close(lambdas), 0.0, 1.0)
    lambdas = lambdas[lambdas > 1e-14]
    if len(lambdas) == 0:
        return 0.0

    logs = np.log(lambdas)
    if log_base != np.e:
        logs = logs / np.log(log_base)
    return float(-np.sum(lambdas * logs))


def page_curve_for_state(psi: np.ndarray, N: int, log_base: float = np.e) -> List[Tuple[int, float]]:
    """Compute S_A for every contiguous left block NA = 1, ..., N-1."""
    return [(NA, entanglement_entropy_statevector(psi, N, NA=NA, log_base=log_base)) for NA in range(1, N)]


def find_scar_candidates(
    energies: np.ndarray,
    entropies: np.ndarray,
    window: int = 20,
    z_threshold: float = 2.5,
    middle_fraction: Tuple[float, float] = (0.25, 0.75),
) -> List[Dict[str, float]]:
    """
    Identify low-entanglement outliers compared with nearby states in energy order.

    The scar score is (local median entropy - entropy) / local std entropy. Large
    positive scores mean the state is much less entangled than nearby eigenstates.
    """
    if len(energies) != len(entropies):
        raise ValueError("energies and entropies must have the same length.")
    if len(energies) == 0:
        return []

    order = np.argsort(energies)
    sorted_energies = energies[order]
    sorted_entropies = entropies[order]
    E0 = float(np.min(energies))

    lo_frac, hi_frac = middle_fraction
    n = len(energies)
    lo = max(0, int(math.floor(lo_frac * n)))
    hi = min(n, int(math.ceil(hi_frac * n)))
    half_window = max(1, int(window) // 2)

    rows: List[Dict[str, float]] = []
    for sorted_pos in range(lo, hi):
        start = max(0, sorted_pos - half_window)
        stop = min(n, sorted_pos + half_window + 1)
        local = np.delete(sorted_entropies[start:stop], sorted_pos - start)
        if len(local) == 0:
            local = sorted_entropies[start:stop]

        local_median = float(np.median(local))
        local_std = float(np.std(local, ddof=1)) if len(local) > 1 else 0.0
        if local_std < 1e-12:
            score = 0.0
        else:
            score = (local_median - float(sorted_entropies[sorted_pos])) / local_std

        original_index = int(order[sorted_pos])
        rows.append(
            {
                "eigenstate_index": original_index,
                "energy": float(sorted_energies[sorted_pos]),
                "excitation_energy": float(sorted_energies[sorted_pos] - E0),
                "half_chain_entropy": float(sorted_entropies[sorted_pos]),
                "local_median_entropy": local_median,
                "anomaly_score": float(score),
                "is_candidate": bool(score > z_threshold),
            }
        )

    return sorted(rows, key=lambda row: row["anomaly_score"], reverse=True)


def top_wavefunction_components(psi: np.ndarray, N: int, top_k: int = 20) -> List[Dict[str, float]]:
    """Return the largest computational-basis probabilities in an eigenvector."""
    probs = np.abs(np.asarray(psi)) ** 2
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [
        {
            "basis_index": int(idx),
            "bitstring": state_to_bitstring(int(idx), N),
            "probability": float(probs[idx]),
        }
        for idx in top_indices
    ]


def save_spectrum_csv(
    output_dir: Path,
    energies: np.ndarray,
    entropies: np.ndarray,
    candidate_rows: Sequence[Dict[str, float]],
) -> Tuple[Path, Path]:
    """Save spectrum_entanglement.csv and scar_candidates.csv."""
    E0 = float(np.min(energies))
    spectrum = pd.DataFrame(
        {
            "eigenstate_index": np.arange(len(energies), dtype=int),
            "energy": energies,
            "excitation_energy": energies - E0,
            "half_chain_entropy": entropies,
        }
    )
    spectrum_path = output_dir / "spectrum_entanglement.csv"
    spectrum.to_csv(spectrum_path, index=False)

    candidate_df = pd.DataFrame(candidate_rows)
    if candidate_df.empty:
        candidate_df = pd.DataFrame(
            columns=[
                "eigenstate_index",
                "energy",
                "excitation_energy",
                "half_chain_entropy",
                "local_median_entropy",
                "anomaly_score",
                "is_candidate",
            ]
        )
    candidates_path = output_dir / "scar_candidates.csv"
    candidate_df.to_csv(candidates_path, index=False)
    return spectrum_path, candidates_path


def plot_entropy_vs_energy(
    output_dir: Path,
    energies: np.ndarray,
    entropies: np.ndarray,
    candidate_rows: Sequence[Dict[str, float]],
) -> Path:
    """Save entropy versus excitation energy, highlighting candidate scars."""
    E0 = float(np.min(energies))
    candidates = [row for row in candidate_rows if row["is_candidate"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(energies - E0, entropies, s=16, alpha=0.65, label="Eigenstates")
    if candidates:
        idx = np.array([int(row["eigenstate_index"]) for row in candidates], dtype=int)
        ax.scatter(energies[idx] - E0, entropies[idx], s=52, marker="x", color="crimson", label="Scar candidates")
        best = candidates[0]
        best_idx = int(best["eigenstate_index"])
        ax.annotate(
            f"#{best_idx}",
            xy=(energies[best_idx] - E0, entropies[best_idx]),
            xytext=(8, 8),
            textcoords="offset points",
            color="crimson",
        )
    ax.set_xlabel(r"Excitation energy $E - E_0$")
    ax.set_ylabel(r"Half-chain entropy $S_A$")
    ax.set_title("Entanglement entropy versus excitation energy")
    ax.legend()
    fig.tight_layout()

    path = output_dir / "entropy_vs_energy.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def choose_nearby_typical_index(
    energies: np.ndarray,
    entropies: np.ndarray,
    candidate_index: int,
    candidate_rows: Sequence[Dict[str, float]],
) -> Optional[int]:
    """Choose a nearby non-candidate state with entropy close to the local median."""
    candidate_flags = {int(row["eigenstate_index"]): bool(row["is_candidate"]) for row in candidate_rows}
    candidate_row = next((row for row in candidate_rows if int(row["eigenstate_index"]) == candidate_index), None)
    local_target = None if candidate_row is None else float(candidate_row["local_median_entropy"])

    order = np.argsort(np.abs(energies - energies[candidate_index]))
    for idx in order:
        idx = int(idx)
        if idx == candidate_index or candidate_flags.get(idx, False):
            continue
        if local_target is None:
            return idx
        if entropies[idx] >= local_target:
            return idx
    for idx in order:
        idx = int(idx)
        if idx != candidate_index:
            return idx
    return None


def plot_page_curve(
    output_dir: Path,
    vectors: np.ndarray,
    energies: np.ndarray,
    entropies: np.ndarray,
    candidate_rows: Sequence[Dict[str, float]],
    best_index: int,
    N: int,
    log_base: float,
) -> Path:
    """Save the Page-curve profile for the best candidate and a nearby typical state."""
    candidate_curve = page_curve_for_state(vectors[:, best_index], N, log_base=log_base)
    typical_index = choose_nearby_typical_index(energies, entropies, best_index, candidate_rows)
    typical_curve = None if typical_index is None else page_curve_for_state(vectors[:, typical_index], N, log_base=log_base)

    fig, ax = plt.subplots(figsize=(7, 5))
    x, y = zip(*candidate_curve)
    ax.plot(x, y, marker="o", label=f"Candidate #{best_index}")
    if typical_curve is not None and typical_index is not None:
        x_typ, y_typ = zip(*typical_curve)
        ax.plot(x_typ, y_typ, marker="s", label=f"Nearby typical #{typical_index}")
    ax.set_xlabel(r"Subsystem size $N_A$")
    ax.set_ylabel(r"Entanglement entropy $S_A$")
    ax.set_title("Entanglement page curve")
    ax.legend()
    fig.tight_layout()

    path = output_dir / f"page_curve_candidate_{best_index}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_wavefunction_components(
    output_dir: Path,
    components: Sequence[Dict[str, float]],
    candidate_index: int,
) -> Tuple[Path, Path]:
    """Save a bar plot and CSV of the largest basis probabilities."""
    df = pd.DataFrame(components)
    csv_path = output_dir / f"wavefunction_components_candidate_{candidate_index}.csv"
    df.to_csv(csv_path, index=False)

    labels = [str(row["bitstring"]) for row in components]
    values = [float(row["probability"]) for row in components]
    fig_width = max(8.0, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    ax.bar(np.arange(len(labels)), values, color="tab:blue")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel(r"Probability $|c_{basis}|^2$")
    ax.set_title(f"Largest wavefunction components for candidate #{candidate_index}")
    fig.tight_layout()

    png_path = output_dir / f"wavefunction_components_candidate_{candidate_index}.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return png_path, csv_path


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Scar-state finder for the j_max=1/2 Ising-limit SU(2) chain.")
    parser.add_argument("--N", type=int, default=10, help="Periodic chain length.")
    parser.add_argument("--J", type=float, default=-3.0 / 16.0, help="Diagonal electric/Ising coupling.")
    parser.add_argument("--hx", type=float, default=1.0, help="Magnetic/off-diagonal coupling.")
    parser.add_argument("--g2a", type=float, default=None, help="Optional placeholder coupling scale; not used by default.")
    parser.add_argument("--mode", choices=("toy", "paper"), default="paper", help="Off-diagonal convention.")
    parser.add_argument("--diag", choices=("dense", "sparse"), default="dense", help="Diagonalisation strategy.")
    parser.add_argument("--num-eigs", type=int, default=100, help="Number of sparse eigenstates to compute.")
    parser.add_argument("--sigma", type=float, default=None, help="Shift-invert target energy for sparse eigsh.")
    parser.add_argument("--window", type=int, default=20, help="Energy-index window for local entropy comparison.")
    parser.add_argument("--z-threshold", type=float, default=2.5, help="Scar anomaly score threshold.")
    parser.add_argument("--output-dir", type=Path, default=Path("scar_outputs"), help="Directory for CSV and plot outputs.")
    parser.add_argument("--log-base", choices=("e", "2"), default="e", help="Entropy logarithm base.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of wavefunction components to print and plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_base = np.e if args.log_base == "e" else 2.0
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dim = 2**args.N
    print(f"Building Hamiltonian for N={args.N}, dim={dim}, mode={args.mode}...")
    if args.g2a is not None:
        print("--g2a was supplied but is not used in this truncated Ising-limit implementation.")

    H = build_ising_hamiltonian_sparse(args.N, args.J, args.hx, mode=args.mode)
    print(f"Diagonalising with {args.diag} mode...")
    energies, vectors = diagonalize_hamiltonian(H, args.diag, args.num_eigs, args.sigma)

    E0 = float(np.min(energies))
    NA = args.N // 2
    print(f"Computing half-chain entanglement for {len(energies)} eigenstates...")
    entropies = np.array(
        [entanglement_entropy_statevector(vectors[:, i], args.N, NA=NA, log_base=log_base) for i in range(len(energies))]
    )

    candidate_rows = find_scar_candidates(
        energies,
        entropies,
        window=args.window,
        z_threshold=args.z_threshold,
        middle_fraction=(0.25, 0.75),
    )
    true_candidates = [row for row in candidate_rows if row["is_candidate"]]

    spectrum_path, candidates_path = save_spectrum_csv(output_dir, energies, entropies, candidate_rows)
    entropy_plot_path = plot_entropy_vs_energy(output_dir, energies, entropies, candidate_rows)

    page_plot_path: Optional[Path] = None
    wf_plot_path: Optional[Path] = None
    wf_csv_path: Optional[Path] = None
    best: Optional[Dict[str, float]] = true_candidates[0] if true_candidates else (candidate_rows[0] if candidate_rows else None)

    if best is not None:
        best_index = int(best["eigenstate_index"])
        page_plot_path = plot_page_curve(output_dir, vectors, energies, entropies, candidate_rows, best_index, args.N, log_base)
        components = top_wavefunction_components(vectors[:, best_index], args.N, top_k=args.top_k)
        wf_plot_path, wf_csv_path = plot_wavefunction_components(output_dir, components, best_index)

        print("\nTop scar candidates by anomaly score:")
        for row in candidate_rows[: min(10, len(candidate_rows))]:
            marker = "*" if row["is_candidate"] else " "
            print(
                f"{marker} idx={int(row['eigenstate_index']):4d} "
                f"E={row['energy']: .8f} "
                f"E-E0={row['excitation_energy']: .8f} "
                f"S={row['half_chain_entropy']: .8f} "
                f"score={row['anomaly_score']: .3f}"
            )

        print(f"\nLargest wavefunction components for best state #{best_index}:")
        for row in components:
            print(f"  {row['basis_index']:5d}  {row['bitstring']}  p={row['probability']:.8e}")

    print("\nSummary")
    print(f"  chain length N: {args.N}")
    print(f"  Hilbert-space dimension: {dim}")
    print(f"  Hamiltonian mode: {args.mode}")
    print(f"  eigenstates analysed: {len(energies)}")
    print(f"  scar candidates found: {len(true_candidates)}")
    if best is None:
        print("  best candidate: none")
    else:
        print(
            "  best candidate: "
            f"index={int(best['eigenstate_index'])}, "
            f"energy={best['energy']:.8f}, "
            f"entropy={best['half_chain_entropy']:.8f}, "
            f"anomaly_score={best['anomaly_score']:.3f}"
        )
    print("  saved outputs:")
    for path in [spectrum_path, candidates_path, entropy_plot_path, page_plot_path, wf_plot_path, wf_csv_path]:
        if path is not None:
            print(f"    {path}")


if __name__ == "__main__":
    main()
