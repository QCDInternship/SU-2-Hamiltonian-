"""Run QVAE seed-family detection on the periodic j_max=1/2 Ising limit.

This command-line program reuses the Hamiltonian, diagonalisation, entropy,
and entropy-anomaly routines from :mod:`scar_state_search`.  The QVAE score
measures learned structural similarity to one training seed; it is not a
probability that an eigenstate is a scar.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr

from qvae_scar_detector import QVAEConfig, QVAEScarDetector
from page_curve_k0 import (
    build_k0_hamiltonian,
    diagonalize_k0_hamiltonian,
    entanglement_entropy_full_state,
    find_scar_candidates as find_k0_scar_candidates,
    reconstruct_full_wavefunction,
)
from scar_state_search import (
    build_ising_hamiltonian_sparse,
    diagonalize_hamiltonian,
    entanglement_entropy_statevector,
    find_scar_candidates,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the periodic truncated-model workflow."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a QVAE seed-family detector on the periodic j_max=1/2 "
            "Ising-limit plaquette chain."
        )
    )
    spectrum = parser.add_argument_group("Hamiltonian and spectrum")
    spectrum.add_argument("--N", type=int, default=10, help="Periodic chain length.")
    spectrum.add_argument("--J", type=float, default=-3.0 / 16.0)
    spectrum.add_argument("--hx", type=float, default=1.0)
    spectrum.add_argument("--mode", choices=("toy", "paper"), default="paper")
    spectrum.add_argument(
        "--sector", choices=("full", "k0"), default="full",
        help="Translation sector to analyse (default: full).",
    )
    spectrum.add_argument("--diag", choices=("dense", "sparse"), default="dense")
    spectrum.add_argument("--num-eigs", type=int, default=100)
    spectrum.add_argument("--sigma", type=float, default=None)
    spectrum.add_argument("--window", type=int, default=20)
    spectrum.add_argument("--z-threshold", type=float, default=2.5)
    spectrum.add_argument("--log-base", choices=("e", "2"), default="e")

    qvae = parser.add_argument_group("QVAE")
    qvae.add_argument("--seed-index", type=int, default=None)
    qvae.add_argument("--trash-qubits", type=int, default=2)
    qvae.add_argument("--qvae-layers", type=int, default=4)
    qvae.add_argument("--qvae-steps", type=int, default=300)
    qvae.add_argument("--qvae-learning-rate", type=float, default=0.03)
    qvae.add_argument("--qvae-seed", type=int, default=0)
    qvae.add_argument("--qvae-patience", type=int, default=50)
    qvae.add_argument("--qvae-tolerance", type=float, default=1.0e-8)
    qvae.add_argument("--max-qvae-qubits", type=int, default=12)
    qvae.add_argument(
        "--comparison-top-k",
        type=int,
        default=10,
        help="Number of highest-ranked states used in ranking-overlap comparisons.",
    )
    qvae.add_argument("--output-dir", type=Path, default=Path("qvae_outputs"))
    return parser.parse_args(argv)


def _validate_cli(args: argparse.Namespace) -> None:
    """Reject unsafe or inconsistent arguments before allocating a statevector."""
    if args.N < 2:
        raise ValueError("--N must be at least 2.")
    if args.max_qvae_qubits < 2:
        raise ValueError("--max-qvae-qubits must be at least 2.")
    if args.N > args.max_qvae_qubits:
        raise ValueError(
            f"N={args.N} exceeds --max-qvae-qubits={args.max_qvae_qubits}. "
            "QVAE training uses a differentiable statevector whose memory and "
            "runtime grow exponentially; increase the limit explicitly only "
            "after confirming sufficient resources."
        )
    if args.num_eigs < 1:
        raise ValueError("--num-eigs must be positive.")
    if args.window < 1:
        raise ValueError("--window must be positive.")
    if args.comparison_top_k < 1:
        raise ValueError("--comparison-top-k must be positive.")
    for name in ("J", "hx", "z_threshold"):
        if not math.isfinite(float(getattr(args, name))):
            raise ValueError(f"--{name.replace('_', '-')} must be finite.")
    if args.sigma is not None and not math.isfinite(args.sigma):
        raise ValueError("--sigma must be finite when supplied.")


def _select_seed(
    requested_index: int | None,
    n_states: int,
    candidate_rows: Sequence[dict[str, Any]],
) -> tuple[int, bool]:
    """Select an explicit seed or the strongest entropy-anomaly candidate.

    Returns the selected eigenstate index and a flag indicating that the
    no-threshold-candidate fallback was needed.
    """
    if requested_index is not None:
        if requested_index < 0 or requested_index >= n_states:
            raise ValueError(
                f"--seed-index must refer to a returned eigenvector in "
                f"0..{n_states - 1}; got {requested_index}."
            )
        return requested_index, False

    passing = [row for row in candidate_rows if bool(row["is_candidate"])]
    if passing:
        best = max(passing, key=lambda row: float(row["anomaly_score"]))
        return int(best["eigenstate_index"]), False
    if not candidate_rows:
        raise RuntimeError("The entropy-anomaly detector returned no selectable rows.")
    best = max(candidate_rows, key=lambda row: float(row["anomaly_score"]))
    return int(best["eigenstate_index"]), True


def _library_versions() -> dict[str, str | None]:
    """Return relevant library versions without importing AI packages here."""
    versions: dict[str, str | None] = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }
    for label, distribution in (("pennylane", "PennyLane"), ("torch", "torch")):
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = None
    return versions


def deterministic_descending_ranks(
    values: Sequence[float], state_indices: Sequence[int]
) -> np.ndarray:
    """Return one-based ordinal ranks with deterministic tie handling.

    Larger finite values rank first. Equal values are ordered by increasing
    state index, and non-finite values rank last, also by increasing state
    index. The result contains a distinct integer rank for every state.
    """
    scores = np.asarray(values, dtype=float)
    indices = np.asarray(state_indices, dtype=int)
    if scores.ndim != 1 or indices.ndim != 1 or scores.shape != indices.shape:
        raise ValueError("values and state_indices must be one-dimensional and equally sized.")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("state_indices must be unique.")
    finite = np.isfinite(scores)
    sortable_scores = np.where(finite, scores, -np.inf)
    order = np.lexsort((indices, -sortable_scores, ~finite))
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=int)
    return ranks


def spearman_summary(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Calculate a finite-pair Spearman correlation with JSON-safe output."""
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1 or x_values.shape != y_values.shape:
        raise ValueError("Spearman inputs must be one-dimensional and equally sized.")
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    n_pairs = int(np.count_nonzero(mask))
    if n_pairs < 2:
        return {"coefficient": None, "pvalue": None, "n_pairs": n_pairs}
    if np.ptp(x_values[mask]) == 0.0 or np.ptp(y_values[mask]) == 0.0:
        return {"coefficient": None, "pvalue": None, "n_pairs": n_pairs}
    result = spearmanr(x_values[mask], y_values[mask])
    coefficient = float(result.statistic)
    pvalue = float(result.pvalue)
    return {
        "coefficient": coefficient if math.isfinite(coefficient) else None,
        "pvalue": pvalue if math.isfinite(pvalue) else None,
        "n_pairs": n_pairs,
    }


def top_k_overlap(
    first_ranks: Sequence[int],
    second_ranks: Sequence[int],
    state_indices: Sequence[int],
    k: int,
) -> dict[str, Any]:
    """Summarize the intersection of two deterministic top-K rankings."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) < 1:
        raise ValueError("k must be a positive integer.")
    first = np.asarray(first_ranks, dtype=int)
    second = np.asarray(second_ranks, dtype=int)
    indices = np.asarray(state_indices, dtype=int)
    if first.ndim != 1 or second.ndim != 1 or indices.ndim != 1:
        raise ValueError("ranks and state_indices must be one-dimensional.")
    if not (first.shape == second.shape == indices.shape):
        raise ValueError("ranks and state_indices must be equally sized.")
    effective_k = min(int(k), len(indices))
    first_top = set(indices[first <= effective_k].tolist())
    second_top = set(indices[second <= effective_k].tolist())
    shared = sorted(first_top & second_top)
    union = first_top | second_top
    return {
        "requested_k": int(k),
        "effective_k": effective_k,
        "overlap_count": len(shared),
        "overlap_fraction": len(shared) / effective_k if effective_k else 0.0,
        "jaccard_index": len(shared) / len(union) if union else 1.0,
        "shared_state_indices": shared,
    }


def _ranking_analysis(data: pd.DataFrame, top_k: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build per-state ranks, correlations, and pairwise top-K overlaps."""
    indices = data["eigenstate_index"].to_numpy(dtype=int)
    qvae_ranks = deterministic_descending_ranks(data["qvae_similarity"], indices)
    entropy_ranks = deterministic_descending_ranks(data["entropy_anomaly_score"], indices)
    overlap_ranks = deterministic_descending_ranks(data["seed_overlap_squared"], indices)
    ranking = pd.DataFrame(
        {
            "eigenstate_index": indices,
            "qvae_rank": qvae_ranks,
            "entropy_anomaly_rank": entropy_ranks,
            "seed_overlap_rank": overlap_ranks,
        }
    )
    summary = {
        "comparison_top_k": int(top_k),
        "correlations": {
            "qvae_similarity_vs_entropy_anomaly_score": spearman_summary(
                data["qvae_similarity"], data["entropy_anomaly_score"]
            ),
            "qvae_similarity_vs_negative_half_chain_entropy": spearman_summary(
                data["qvae_similarity"], -data["half_chain_entropy"]
            ),
            "qvae_similarity_vs_seed_overlap_squared": spearman_summary(
                data["qvae_similarity"], data["seed_overlap_squared"]
            ),
        },
        "top_k_overlaps": {
            "qvae_vs_entropy_anomaly": top_k_overlap(
                qvae_ranks, entropy_ranks, indices, top_k
            ),
            "qvae_vs_seed_overlap": top_k_overlap(
                qvae_ranks, overlap_ranks, indices, top_k
            ),
            "entropy_anomaly_vs_seed_overlap": top_k_overlap(
                entropy_ranks, overlap_ranks, indices, top_k
            ),
        },
    }
    return ranking, summary


def _save_plots(data: pd.DataFrame, history: pd.DataFrame, output_dir: Path) -> None:
    """Save the required QVAE diagnostic plots."""
    seed = data[data["is_training_seed"]]
    non_seed_lowest = data[~data["is_training_seed"]].nsmallest(10, "qvae_cost")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data["excitation_energy"], data["qvae_cost"], s=18, alpha=0.6, label="Eigenstates")
    ax.scatter(
        non_seed_lowest["excitation_energy"],
        non_seed_lowest["qvae_cost"],
        marker="x",
        s=55,
        color="tab:orange",
        label="Ten lowest-cost non-seed states",
    )
    ax.scatter(seed["excitation_energy"], seed["qvae_cost"], marker="*", s=150, color="crimson", label="Training seed")
    ax.set_xlabel(r"Excitation energy $E-E_0$")
    ax.set_ylabel("QVAE trash-qubit cost")
    ax.set_title("QVAE cost versus excitation energy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "qvae_cost_vs_energy.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(data["half_chain_entropy"], data["qvae_cost"], s=20, alpha=0.7)
    ax.scatter(seed["half_chain_entropy"], seed["qvae_cost"], marker="*", s=150, color="crimson")
    ax.set_xlabel("Half-chain entropy")
    ax.set_ylabel("QVAE trash-qubit cost")
    ax.set_title("QVAE cost versus half-chain entropy")
    fig.tight_layout()
    fig.savefig(output_dir / "qvae_cost_vs_entropy.png", dpi=200)
    plt.close(fig)

    finite_anomaly = data[np.isfinite(data["entropy_anomaly_score"])]
    seed_finite = seed[np.isfinite(seed["entropy_anomaly_score"])]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(finite_anomaly["entropy_anomaly_score"], finite_anomaly["qvae_cost"], s=20, alpha=0.7)
    if not seed_finite.empty:
        ax.scatter(seed_finite["entropy_anomaly_score"], seed_finite["qvae_cost"], marker="*", s=150, color="crimson")
    ax.set_xlabel("Entropy anomaly score")
    ax.set_ylabel("QVAE trash-qubit cost")
    ax.set_title("QVAE cost versus entropy anomaly score")
    fig.tight_layout()
    fig.savefig(output_dir / "qvae_cost_vs_anomaly_score.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["step"], history["cost"], color="tab:blue")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Training-seed QVAE cost")
    ax.set_title("QVAE training curve")
    fig.tight_layout()
    fig.savefig(output_dir / "qvae_training_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(data["seed_overlap_squared"], data["qvae_similarity"], s=20, alpha=0.7)
    ax.scatter(seed["seed_overlap_squared"], seed["qvae_similarity"], marker="*", s=150, color="crimson")
    ax.set_xlabel(r"Direct seed overlap $|\langle\psi_{seed}|\psi_i\rangle|^2$")
    ax.set_ylabel("QVAE similarity")
    ax.set_title("QVAE similarity versus direct seed overlap")
    fig.tight_layout()
    fig.savefig(output_dir / "qvae_similarity_vs_seed_overlap.png", dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> pd.DataFrame:
    """Execute the complete periodic truncated-model QVAE workflow."""
    _validate_cli(args)
    qvae_config = QVAEConfig(
        n_qubits=args.N,
        n_trash=args.trash_qubits,
        n_layers=args.qvae_layers,
        learning_rate=args.qvae_learning_rate,
        steps=args.qvae_steps,
        seed=args.qvae_seed,
        convergence_tolerance=args.qvae_tolerance,
        patience=args.qvae_patience,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    full_dimension = 1 << args.N
    if args.sector == "full":
        hamiltonian = build_ising_hamiltonian_sparse(args.N, args.J, args.hx, mode=args.mode)
        energies, vectors = diagonalize_hamiltonian(
            hamiltonian, args.diag, args.num_eigs, args.sigma
        )
        sector_dimension = full_dimension
        states = vectors
        entropy_function = lambda state: entanglement_entropy_statevector(
            state, args.N, NA=args.N // 2,
            log_base=np.e if args.log_base == "e" else 2.0,
        )
        candidate_function = find_scar_candidates
    else:
        hamiltonian, orbits, _ = build_k0_hamiltonian(
            args.N, args.J, args.hx, mode=args.mode
        )
        energies, vectors = diagonalize_k0_hamiltonian(
            hamiltonian, args.diag, args.num_eigs, args.sigma
        )
        sector_dimension = hamiltonian.shape[0]
        reconstructed = []
        for index in range(vectors.shape[1]):
            state = reconstruct_full_wavefunction(vectors[:, index], orbits, args.N)
            if state.shape != (full_dimension,):
                raise RuntimeError(
                    f"Reconstructed k=0 eigenstate {index} has shape {state.shape}; "
                    f"expected ({full_dimension},)."
                )
            norm = float(np.linalg.norm(state))
            if not np.isclose(norm, 1.0, rtol=1e-10, atol=1e-12):
                raise RuntimeError(
                    f"Reconstructed k=0 eigenstate {index} has norm {norm}; expected 1."
                )
            reconstructed.append(state)
        states = np.column_stack(reconstructed)
        entropy_function = lambda state: entanglement_entropy_full_state(
            state, args.N, subsystem_sites=tuple(range(args.N // 2)),
            log_base=np.e if args.log_base == "e" else 2.0,
        )
        candidate_function = find_k0_scar_candidates
    entropies = np.array(
        [entropy_function(states[:, index]) for index in range(len(energies))],
        dtype=float,
    )
    candidate_rows = candidate_function(
        energies,
        entropies,
        window=args.window,
        z_threshold=args.z_threshold,
        middle_fraction=(0.25, 0.75),
    )
    seed_index, used_fallback = _select_seed(
        args.seed_index, len(energies), candidate_rows
    )
    if used_fallback:
        print(
            "Warning: no entropy-anomaly row passed --z-threshold; using the "
            "returned row with the highest anomaly score as the QVAE seed."
        )

    seed_state = states[:, seed_index]
    if args.sector == "k0":
        print(f"N: {args.N}")
        print("sector: k0")
        print(f"dim(k0): {sector_dimension}")
        print(f"full statevector dimension: {full_dimension}")
        print(f"seed index: {seed_index}")
        print(f"reconstructed seed norm: {np.linalg.norm(seed_state):.1f}")
    detector = QVAEScarDetector(qvae_config)
    training = detector.fit(seed_state)
    qvae_rows = detector.scan(states[:, index] for index in range(len(energies)))

    anomaly_by_index = {
        int(row["eigenstate_index"]): row for row in candidate_rows
    }
    ground_energy = float(np.min(energies))
    records: list[dict[str, Any]] = []
    for index, qvae_row in enumerate(qvae_rows):
        anomaly = anomaly_by_index.get(index)
        overlap = float(abs(np.vdot(seed_state, states[:, index])) ** 2)
        records.append(
            {
                "eigenstate_index": index,
                "sector": args.sector,
                "sector_dimension": sector_dimension,
                "full_dimension": full_dimension,
                "energy": float(energies[index]),
                "excitation_energy": float(energies[index] - ground_energy),
                "half_chain_entropy": float(entropies[index]),
                "entropy_anomaly_score": (
                    float(anomaly["anomaly_score"]) if anomaly is not None else np.nan
                ),
                "entropy_candidate": bool(anomaly["is_candidate"]) if anomaly is not None else False,
                "seed_overlap_squared": overlap,
                "qvae_cost": float(qvae_row["qvae_cost"]),
                "qvae_similarity": float(qvae_row["qvae_similarity"]),
                "is_training_seed": index == seed_index,
            }
        )
    scan = pd.DataFrame.from_records(records)
    scan.to_csv(output_dir / "qvae_scan.csv", index=False)

    ranking, ranking_summary = _ranking_analysis(scan, args.comparison_top_k)
    ranking.to_csv(output_dir / "qvae_ranking_comparison.csv", index=False)
    with (output_dir / "qvae_ranking_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(ranking_summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

    history = pd.DataFrame(
        {"step": np.arange(len(training.cost_history)), "cost": training.cost_history}
    )
    history.to_csv(output_dir / "qvae_training_history.csv", index=False)
    detector.save_checkpoint(output_dir / "qvae_parameters.npz")

    configuration = {
        "model_scope": "periodic j_max=1/2 Ising-limit plaquette-chain model",
        "sector": args.sector,
        "full_dimension": full_dimension,
        "sector_dimension": sector_dimension,
        "reconstruction_used": args.sector == "k0",
        "seed_index_scope": (
            "k0_eigenstate_collection" if args.sector == "k0"
            else "full_eigenstate_collection"
        ),
        "hamiltonian": {
            "N": args.N,
            "J": args.J,
            "hx": args.hx,
            "mode": args.mode,
            "periodic": True,
        },
        "diagonalisation": {
            "method": args.diag,
            "num_eigs": args.num_eigs,
            "sigma": args.sigma,
            "returned_eigenstates": len(energies),
        },
        "entropy_detector": {
            "window": args.window,
            "z_threshold": args.z_threshold,
            "log_base": args.log_base,
        },
        "qvae": {
            "n_qubits": qvae_config.n_qubits,
            "n_latent": qvae_config.n_latent,
            "n_trash": qvae_config.n_trash,
            "n_layers": qvae_config.n_layers,
            "learning_rate": qvae_config.learning_rate,
            "steps": qvae_config.steps,
            "seed": qvae_config.seed,
            "device_name": qvae_config.device_name,
            "convergence_tolerance": qvae_config.convergence_tolerance,
            "patience": qvae_config.patience,
            "max_qvae_qubits": args.max_qvae_qubits,
            "comparison_top_k": args.comparison_top_k,
        },
        "seed_eigenstate_index": seed_index,
        "training": {
            "initial_cost": training.initial_cost,
            "final_cost": training.final_cost,
            "best_cost": training.best_cost,
            "best_step": training.best_step,
            "steps_completed": training.steps_completed,
            "converged": training.converged,
        },
        "library_versions": _library_versions(),
    }
    with (output_dir / "qvae_config.json").open("w", encoding="utf-8") as stream:
        json.dump(configuration, stream, indent=2, sort_keys=True)
        stream.write("\n")

    _save_plots(scan, history, output_dir)

    seed_anomaly = anomaly_by_index.get(seed_index)
    seed_anomaly_score = (
        float(seed_anomaly["anomaly_score"]) if seed_anomaly is not None else float("nan")
    )
    print("QVAE seed-family summary")
    print(f"  seed index: {seed_index}")
    print(f"  seed energy: {energies[seed_index]:.10g}")
    print(f"  seed entropy: {entropies[seed_index]:.10g}")
    print(f"  seed entropy anomaly score: {seed_anomaly_score:.10g}")
    print(f"  initial QVAE cost: {training.initial_cost:.10g}")
    print(f"  best QVAE cost: {training.best_cost:.10g}")
    print("  ranking correlations (Spearman rho):")
    for label, result in ranking_summary["correlations"].items():
        coefficient = result["coefficient"]
        display = "undefined" if coefficient is None else f"{coefficient:.6g}"
        print(f"    {label}: {display} (finite pairs={result['n_pairs']})")
    print(f"  top-{args.comparison_top_k} ranking overlaps:")
    for label, result in ranking_summary["top_k_overlaps"].items():
        print(
            f"    {label}: {result['overlap_count']}/{result['effective_k']} "
            f"(fraction={result['overlap_fraction']:.4g})"
        )
    print(
        "  Interpretation: high correlation with entropy means the QVAE may "
        "largely reproduce the low-entanglement ordering."
    )
    print(
        "  High correlation with seed overlap means the detector has learned "
        "features closely tied to the training state."
    )
    print(
        "  States with low QVAE cost but only moderate direct seed overlap are "
        "particularly useful candidates for further physical inspection."
    )
    print("  None of these metrics establishes scarring by itself.")
    print("  ten lowest-cost states after training:")
    for row in scan.nsmallest(10, "qvae_cost").itertuples(index=False):
        relation = (
            "training seed"
            if row.is_training_seed
            else "QVAE-related candidate; candidate member of the same learned family"
        )
        print(
            f"    index={row.eigenstate_index:4d} cost={row.qvae_cost:.8g} "
            f"similarity={row.qvae_similarity:.8g} -- {relation}, state "
            "structurally similar to the training seed"
        )
    print(f"  outputs: {output_dir}")
    return scan


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point with concise user-facing validation failures."""
    args = parse_args(argv)
    try:
        run(args)
    except (ImportError, ValueError, RuntimeError, FloatingPointError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
