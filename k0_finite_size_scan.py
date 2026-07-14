"""Finite-size scar-candidate scan in the periodic k=0 Ising-limit sector.

This program studies only the existing ``j_max=1/2`` effective SU(2)
Ising-limit model.  Hamiltonian construction, the definition of the k=0
sector, wavefunction reconstruction, entropy, and anomaly scoring are imported
from :mod:`page_curve_k0`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from numbers import Integral, Real
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

import numpy as np
import scipy
from scipy.sparse.linalg import eigsh

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from page_curve_k0 import (
    couplings_from_g2a,
    build_k0_hamiltonian,
    diagonalize_k0_hamiltonian,
    reconstruct_full_wavefunction,
    entanglement_entropy_full_state,
    find_scar_candidates,
    write_dict_csv,
)


SUMMARY_COLUMNS = (
    "N", "mode", "J", "hx", "g2a", "full_dimension", "k0_dimension",
    "diagonalisation_mode", "statistics_scope", "eigenstates_analysed",
    "global_energy_min", "global_energy_max", "spectral_width",
    "target_energy_density", "energy_density_window",
    "number_of_candidates", "candidate_found", "best_candidate_index",
    "candidate_energy", "candidate_excitation_energy", "candidate_energy_density",
    "distance_from_target_energy_density",
    "half_chain_entropy", "maximum_half_chain_entropy",
    "normalized_half_chain_entropy", "local_median_entropy", "entropy_deficit",
    "anomaly_score", "z_threshold", "anomaly_window", "runtime_seconds",
    "status", "error_message",
)

CANDIDATE_COLUMNS = (
    "N", "candidate_rank", "eigenstate_index", "mode", "J", "hx",
    "full_dimension", "k0_dimension", "diagonalisation_mode", "statistics_scope",
    "global_energy_min", "global_energy_max", "energy", "excitation_energy",
    "energy_density", "target_energy_density", "energy_density_window",
    "distance_from_target_energy_density", "half_chain_entropy", "normalized_half_chain_entropy",
    "local_median_entropy", "entropy_deficit", "anomaly_score", "is_candidate",
)

ALL_STATE_COLUMNS = (
    "N", "eigenstate_index", "mode", "J", "hx", "full_dimension",
    "k0_dimension", "diagonalisation_mode", "statistics_scope",
    "global_energy_min", "global_energy_max", "energy", "excitation_energy",
    "energy_density", "target_energy_density", "energy_density_window",
    "distance_from_target_energy_density", "half_chain_entropy", "normalized_half_chain_entropy",
    "local_median_entropy", "entropy_deficit", "anomaly_score", "is_candidate",
)


@dataclass(frozen=True)
class FiniteSizeScanConfig:
    """Validated, size-independent settings for one finite-size scan."""

    sizes: tuple[int, ...]
    J: float
    hx: float
    g2a: float
    mode: str
    diag: str
    max_dense_dim: int
    num_eigs: int
    sigma: float | None
    window: int
    z_threshold: float
    middle_fraction: tuple[float, float]
    log_base: float
    top_candidates: int
    output_dir: Path
    save_all_states: bool
    target_energy_density: float | None = None
    energy_density_window: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sizes, tuple):
            raise TypeError("sizes must be a tuple of integers.")
        if not self.sizes or any(
            isinstance(n, bool) or not isinstance(n, Integral) for n in self.sizes
        ):
            raise ValueError("sizes must contain integers.")
        if len(set(self.sizes)) != len(self.sizes):
            raise ValueError("sizes must not contain duplicates.")
        for name in ("J", "hx", "g2a", "z_threshold", "log_base"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if self.log_base <= 0.0 or math.isclose(self.log_base, 1.0):
            raise ValueError("log_base must be positive and different from one.")
        if self.mode not in {"toy", "paper"}:
            raise ValueError("mode must be 'toy' or 'paper'.")
        if self.diag not in {"auto", "dense", "sparse"}:
            raise ValueError("diag must be 'auto', 'dense', or 'sparse'.")
        for name in ("max_dense_dim", "num_eigs", "window", "top_candidates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
        if self.max_dense_dim < 1 or self.num_eigs < 1 or self.window < 1:
            raise ValueError("max_dense_dim, num_eigs, and window must be positive.")
        if self.top_candidates < 1:
            raise ValueError("top_candidates must be positive.")
        if not isinstance(self.middle_fraction, tuple) or len(self.middle_fraction) != 2:
            raise TypeError("middle_fraction must be a two-element tuple.")
        lo, hi = self.middle_fraction
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in (lo, hi)):
            raise TypeError("middle_fraction values must be real numbers.")
        if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 <= lo < hi <= 1.0):
            raise ValueError("middle_fraction must satisfy 0 <= low < high <= 1.")
        if self.sigma is not None:
            if isinstance(self.sigma, bool) or not isinstance(self.sigma, Real):
                raise TypeError("sigma must be a real number when supplied.")
            if not math.isfinite(self.sigma):
                raise ValueError("sigma must be finite when supplied.")
        if self.target_energy_density is None:
            if self.energy_density_window is not None:
                raise ValueError(
                    "energy_density_window requires target_energy_density."
                )
        else:
            if isinstance(self.target_energy_density, bool) or not isinstance(
                self.target_energy_density, Real
            ):
                raise TypeError("target_energy_density must be a real number.")
            if not math.isfinite(self.target_energy_density) or not (
                0.0 <= self.target_energy_density <= 1.0
            ):
                raise ValueError("target_energy_density must satisfy 0 <= target <= 1.")
            if self.energy_density_window is None:
                raise ValueError(
                    "energy_density_window must be resolved when a target is supplied."
                )
            if isinstance(self.energy_density_window, bool) or not isinstance(
                self.energy_density_window, Real
            ):
                raise TypeError("energy_density_window must be a real number.")
            if not math.isfinite(self.energy_density_window) or not (
                0.0 < self.energy_density_window <= 0.5
            ):
                raise ValueError("energy_density_window must satisfy 0 < window <= 0.5.")
        if not isinstance(self.output_dir, Path):
            raise TypeError("output_dir must be a pathlib.Path.")
        if not isinstance(self.save_all_states, bool):
            raise TypeError("save_all_states must be a bool.")


@dataclass
class FiniteSizeResult:
    """Summary values and per-state output rows for one system size."""

    N: int
    mode: str
    J: float
    hx: float
    g2a: float
    full_dimension: int = 0
    k0_dimension: int = 0
    diagonalisation_mode: str = ""
    statistics_scope: str = ""
    eigenstates_analysed: int = 0
    global_energy_min: float | str = ""
    global_energy_max: float | str = ""
    spectral_width: float | str = ""
    target_energy_density: float | str = ""
    energy_density_window: float | str = ""
    number_of_candidates: int = 0
    candidate_found: bool | str = ""
    best_candidate_index: int | str = ""
    candidate_energy: float | str = ""
    candidate_excitation_energy: float | str = ""
    candidate_energy_density: float | str = ""
    distance_from_target_energy_density: float | str = ""
    half_chain_entropy: float | str = ""
    maximum_half_chain_entropy: float | str = ""
    normalized_half_chain_entropy: float | str = ""
    local_median_entropy: float | str = ""
    entropy_deficit: float | str = ""
    anomaly_score: float | str = ""
    z_threshold: float = 0.0
    anomaly_window: int = 0
    runtime_seconds: float = 0.0
    status: str = "ok"
    error_message: str = ""
    candidate_rows: list[dict[str, Any]] = field(default_factory=list, repr=False)
    all_state_rows: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def summary_row(self) -> dict[str, Any]:
        """Return exactly the public summary CSV fields."""
        values = asdict(self)
        return {column: values[column] for column in SUMMARY_COLUMNS}


def _maximum_half_chain_entropy(n_sites: int, log_base: float) -> float:
    """Return the maximum entropy of the configured half chain."""
    half_sites = n_sites // 2
    if math.isclose(log_base, math.e, rel_tol=0.0, abs_tol=1e-14):
        return half_sites * math.log(2.0)
    if math.isclose(log_base, 2.0, rel_tol=0.0, abs_tol=1e-14):
        return float(half_sites)
    return half_sites * math.log(2.0) / math.log(log_base)


def _energy_coordinates(energy: float, e_min: float, e_max: float) -> tuple[float, float]:
    """Return excitation energy and globally normalized energy density."""
    excitation = float(energy - e_min)
    width = e_max - e_min
    density = 0.0 if abs(width) < 1e-14 else excitation / width
    return excitation, float(density)


def analyse_one_size(config: FiniteSizeScanConfig, n_sites: int) -> FiniteSizeResult:
    """Analyse one periodic k=0 sector and return its finite-size result."""
    if n_sites not in config.sizes:
        raise ValueError(f"N={n_sites} is not present in config.sizes.")
    started = time.perf_counter()
    H, orbits, state_to_orbit_index = build_k0_hamiltonian(
        n_sites, config.J, config.hx, mode=config.mode
    )
    full_dimension = 2**n_sites
    k0_dimension = int(H.shape[0])
    if len(orbits) != k0_dimension or len(state_to_orbit_index) != full_dimension:
        raise RuntimeError("The imported k=0 orbit data are dimensionally inconsistent.")

    backend = config.diag
    if backend == "auto":
        backend = "dense" if k0_dimension <= config.max_dense_dim else "sparse"

    if backend == "dense":
        energies, vectors = diagonalize_k0_hamiltonian(H, diag="dense")
        global_energy_min = float(np.min(energies))
        global_energy_max = float(np.max(energies))
        statistics_scope = "full_k0_spectrum"
    else:
        if k0_dimension <= 2:
            raise ValueError("Sparse diagonalisation requires k0_dimension > 2.")
        global_energy_min = float(eigsh(H, k=1, which="SA", return_eigenvectors=False)[0])
        global_energy_max = float(eigsh(H, k=1, which="LA", return_eigenvectors=False)[0])
        effective_sigma = config.sigma
        if effective_sigma is None and config.target_energy_density is not None:
            effective_sigma = global_energy_min + config.target_energy_density * (
                global_energy_max - global_energy_min
            )
        energies, vectors = diagonalize_k0_hamiltonian(
            H, diag="sparse", num_eigs=config.num_eigs, sigma=effective_sigma
        )
        statistics_scope = "targeted_sparse_window"

    half_chain = tuple(range(n_sites // 2))
    entropies: list[float] = []
    for column in range(vectors.shape[1]):
        psi_full = reconstruct_full_wavefunction(vectors[:, column], orbits, n_sites)
        norm = float(np.linalg.norm(psi_full))
        if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-10):
            raise RuntimeError(f"Reconstructed eigenvector {column} has norm {norm:.16g}.")
        entropies.append(
            entanglement_entropy_full_state(
                psi_full, n_sites, half_chain, log_base=config.log_base
            )
        )
    entropy_array = np.asarray(entropies, dtype=float)
    anomaly_rows = find_scar_candidates(
        energies,
        entropy_array,
        window=config.window,
        z_threshold=config.z_threshold,
        middle_fraction=config.middle_fraction,
    )
    maximum_entropy = _maximum_half_chain_entropy(n_sites, config.log_base)

    eligible_rows: list[dict[str, Any]] = []
    for row in anomaly_rows:
        enriched = dict(row)
        _, density = _energy_coordinates(
            float(row["energy"]), global_energy_min, global_energy_max
        )
        enriched["energy_density"] = density
        enriched["distance_from_target_energy_density"] = (
            ""
            if config.target_energy_density is None
            else abs(density - config.target_energy_density)
        )
        if config.target_energy_density is None or (
            float(enriched["distance_from_target_energy_density"])
            <= float(config.energy_density_window)
        ):
            eligible_rows.append(enriched)

    finite_rows = [
        row for row in eligible_rows if math.isfinite(float(row["anomaly_score"]))
    ]
    passing_rows = [row for row in finite_rows if bool(row["is_candidate"])]
    selected = finite_rows[0] if finite_rows else None

    common = {
        "N": n_sites, "mode": config.mode, "J": config.J, "hx": config.hx,
        "full_dimension": full_dimension, "k0_dimension": k0_dimension,
        "diagonalisation_mode": backend, "statistics_scope": statistics_scope,
        "global_energy_min": global_energy_min, "global_energy_max": global_energy_max,
        "target_energy_density": (
            "" if config.target_energy_density is None else config.target_energy_density
        ),
        "energy_density_window": (
            "" if config.energy_density_window is None else config.energy_density_window
        ),
    }
    candidate_output: list[dict[str, Any]] = []
    for rank, row in enumerate(eligible_rows[: config.top_candidates], start=1):
        energy = float(row["energy"])
        excitation, density = _energy_coordinates(energy, global_energy_min, global_energy_max)
        entropy = float(row["half_chain_entropy"])
        local_median = float(row["local_median_entropy"])
        candidate_output.append({
            **common, "candidate_rank": rank,
            "eigenstate_index": int(row["eigenstate_index"]), "energy": energy,
            "excitation_energy": excitation, "energy_density": density,
            "distance_from_target_energy_density": row[
                "distance_from_target_energy_density"
            ],
            "half_chain_entropy": entropy,
            "normalized_half_chain_entropy": entropy / maximum_entropy,
            "local_median_entropy": local_median,
            "entropy_deficit": local_median - entropy,
            "anomaly_score": float(row["anomaly_score"]),
            "is_candidate": bool(row["is_candidate"]),
        })

    anomaly_by_index = {int(row["eigenstate_index"]): row for row in anomaly_rows}
    all_state_output: list[dict[str, Any]] = []
    for index, (energy, entropy) in enumerate(zip(energies, entropy_array)):
        excitation, density = _energy_coordinates(
            float(energy), global_energy_min, global_energy_max
        )
        anomaly = anomaly_by_index.get(index)
        local_median: float | str = "" if anomaly is None else float(anomaly["local_median_entropy"])
        all_state_output.append({
            **common, "eigenstate_index": index, "energy": float(energy),
            "excitation_energy": excitation, "energy_density": density,
            "target_energy_density": (
                "" if config.target_energy_density is None else config.target_energy_density
            ),
            "energy_density_window": (
                "" if config.energy_density_window is None else config.energy_density_window
            ),
            "distance_from_target_energy_density": (
                ""
                if config.target_energy_density is None
                else abs(density - config.target_energy_density)
            ),
            "half_chain_entropy": float(entropy),
            "normalized_half_chain_entropy": float(entropy) / maximum_entropy,
            "local_median_entropy": local_median,
            "entropy_deficit": "" if anomaly is None else float(local_median) - float(entropy),
            "anomaly_score": "" if anomaly is None else float(anomaly["anomaly_score"]),
            "is_candidate": "" if anomaly is None else bool(anomaly["is_candidate"]),
        })

    result = FiniteSizeResult(
        N=n_sites, mode=config.mode, J=config.J, hx=config.hx, g2a=config.g2a,
        full_dimension=full_dimension, k0_dimension=k0_dimension,
        diagonalisation_mode=backend, statistics_scope=statistics_scope,
        eigenstates_analysed=len(energies), global_energy_min=global_energy_min,
        global_energy_max=global_energy_max,
        spectral_width=global_energy_max - global_energy_min,
        target_energy_density=(
            "" if config.target_energy_density is None else config.target_energy_density
        ),
        energy_density_window=(
            "" if config.energy_density_window is None else config.energy_density_window
        ),
        number_of_candidates=len(passing_rows), candidate_found=bool(passing_rows),
        maximum_half_chain_entropy=maximum_entropy, z_threshold=config.z_threshold,
        anomaly_window=config.window, candidate_rows=candidate_output,
        all_state_rows=all_state_output,
    )
    if config.target_energy_density is not None and selected is None:
        result.status = "no_state_in_target_window"
    if selected is not None:
        energy = float(selected["energy"])
        excitation, density = _energy_coordinates(energy, global_energy_min, global_energy_max)
        entropy = float(selected["half_chain_entropy"])
        local_median = float(selected["local_median_entropy"])
        result.best_candidate_index = int(selected["eigenstate_index"])
        result.candidate_energy = energy
        result.candidate_excitation_energy = excitation
        result.candidate_energy_density = density
        result.distance_from_target_energy_density = (
            ""
            if config.target_energy_density is None
            else abs(density - config.target_energy_density)
        )
        result.half_chain_entropy = entropy
        result.normalized_half_chain_entropy = entropy / maximum_entropy
        result.local_median_entropy = local_median
        result.entropy_deficit = local_median - entropy
        result.anomaly_score = float(selected["anomaly_score"])
    result.runtime_seconds = time.perf_counter() - started
    return result


def _failed_result(config: FiniteSizeScanConfig, n_sites: int, error: Exception, runtime: float) -> FiniteSizeResult:
    """Create a summary result for a size that could not be analysed."""
    return FiniteSizeResult(
        N=n_sites, mode=config.mode, J=config.J, hx=config.hx, g2a=config.g2a,
        full_dimension=2**n_sites, z_threshold=config.z_threshold,
        target_energy_density=(
            "" if config.target_energy_density is None else config.target_energy_density
        ),
        energy_density_window=(
            "" if config.energy_density_window is None else config.energy_density_window
        ),
        anomaly_window=config.window, runtime_seconds=runtime, status="failed",
        error_message=f"{type(error).__name__}: {error}",
    )


def _write_config(config: FiniteSizeScanConfig) -> Path:
    """Save resolved arguments and readily available package versions."""
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["model_scope"] = "periodic j_max=1/2 effective SU(2) Ising-limit model"
    payload["versions"] = {
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
    }
    path = config.output_dir / "k0_finite_size_config.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _print_table(results: Sequence[FiniteSizeResult]) -> None:
    """Print a compact finite-size summary table."""
    print("\nPeriodic k=0 j_max=1/2 effective Ising-limit scan")
    print(f"{'N':>3} {'dim(k0)':>8} {'eigs':>6} {'best':>6} {'e dens':>9} {'entropy':>10} {'deficit':>10} {'score':>8} {'cand':>5} {'sec':>8}")
    for row in results:
        def fmt(value: Any, width: int, precision: int = 4) -> str:
            return f"{value:{width}.{precision}f}" if isinstance(value, float) else f"{str(value or '-'):>{width}}"
        candidate = "yes" if row.candidate_found is True else ("no" if row.status == "ok" else "-")
        print(
            f"{row.N:3d} {row.k0_dimension:8d} {row.eigenstates_analysed:6d} "
            f"{str(row.best_candidate_index if row.best_candidate_index != '' else '-'):>6} {fmt(row.candidate_energy_density, 9)} "
            f"{fmt(row.half_chain_entropy, 10)} {fmt(row.entropy_deficit, 10)} "
            f"{fmt(row.anomaly_score, 8)} {candidate:>5} {row.runtime_seconds:8.3f}"
        )
        if row.status == "failed":
            print(f"    failed: {row.error_message}")


def _plot_candidate_sequence(
    results: Sequence[FiniteSizeResult], config: FiniteSizeScanConfig
) -> list[Path]:
    """Plot size-wise strongest candidates without joining missing sizes."""
    selected = sorted(
        (
            row
            for row in results
            if row.status == "ok" and row.best_candidate_index != ""
        ),
        key=lambda row: row.N,
    )
    sequence_description = (
        "Strongest candidate sequence in the chosen k=0 energy-density band"
        if config.target_energy_density is not None
        else "Strongest anomaly-ranked candidate sequence in the k=0 sector"
    )
    plots = [
        (
            "k0_candidate_energy_density_vs_N.png",
            "Strongest candidate energy density",
            [(row.N, float(row.candidate_energy_density)) for row in selected],
            "Normalized energy density",
        ),
        (
            "k0_entropy_deficit_vs_N.png",
            "Strongest candidate entropy deficit",
            [(row.N, float(row.entropy_deficit)) for row in selected],
            "Local median entropy - candidate entropy",
        ),
        (
            "k0_anomaly_score_vs_N.png",
            "Strongest candidate anomaly score",
            [(row.N, float(row.anomaly_score)) for row in selected],
            "Anomaly score",
        ),
    ]
    paths: list[Path] = []
    for filename, title, points, ylabel in plots:
        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        if points:
            axis.scatter(*zip(*points), marker="o", label="strongest candidate")
        if filename == "k0_candidate_energy_density_vs_N.png" and config.target_energy_density is not None:
            axis.axhline(
                config.target_energy_density, color="tab:red", linestyle="--",
                label="requested target energy density",
            )
        if filename == "k0_anomaly_score_vs_N.png":
            axis.axhline(
                config.z_threshold, color="tab:red", linestyle="--",
                label="candidate threshold",
            )
        axis.set_xlabel("N")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{title}\n{sequence_description}")
        axis.grid(alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend()
        figure.tight_layout()
        path = config.output_dir / filename
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    if selected:
        sizes = [row.N for row in selected]
        normalized = [float(row.normalized_half_chain_entropy) for row in selected]
        local_normalized = [
            float(row.local_median_entropy) / float(row.maximum_half_chain_entropy)
            for row in selected
        ]
        axis.scatter(sizes, normalized, marker="o", label="candidate entropy")
        axis.scatter(sizes, local_normalized, marker="s", label="local median entropy")
    axis.set_xlabel("N")
    axis.set_ylabel("Entropy / maximum half-chain entropy")
    axis.set_title(sequence_description)
    axis.grid(alpha=0.25)
    if axis.get_legend_handles_labels()[0]:
        axis.legend()
    figure.tight_layout()
    path = config.output_dir / "k0_normalized_entropy_vs_N.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Finite-size scar-candidate scan in the periodic k=0 sector of the j_max=1/2 effective Ising-limit model."
    )
    parser.add_argument(
        "--sizes", "--n-sites", type=int, nargs="+", required=True, dest="sizes",
        help="System sizes to scan; --n-sites is retained as an alias.",
    )
    parser.add_argument("--g2a", type=float, default=1.2)
    parser.add_argument("--J", type=float, default=None)
    parser.add_argument("--hx", type=float, default=None)
    parser.add_argument("--mode", choices=("toy", "paper"), default="paper")
    parser.add_argument("--diag", choices=("auto", "dense", "sparse"), default="auto")
    parser.add_argument("--max-dense-dim", type=int, default=3000)
    parser.add_argument("--num-eigs", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--z-threshold", type=float, default=2.5)
    parser.add_argument("--middle-fraction", type=float, nargs=2, default=(0.25, 0.75), metavar=("LOW", "HIGH"))
    parser.add_argument("--log-base", choices=("e", "2"), default="e")
    parser.add_argument("--top-candidates", type=int, default=10)
    parser.add_argument("--target-energy-density", type=float, default=None)
    parser.add_argument("--energy-density-window", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("k0_finite_size_outputs"))
    parser.add_argument("--save-all-states", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> FiniteSizeScanConfig:
    """Resolve coupling defaults and construct a validated scan config."""
    default_J, default_hx = couplings_from_g2a(args.g2a)
    energy_density_window = args.energy_density_window
    if args.target_energy_density is not None and energy_density_window is None:
        energy_density_window = 0.05
    return FiniteSizeScanConfig(
        sizes=tuple(args.sizes), J=default_J if args.J is None else args.J,
        hx=default_hx if args.hx is None else args.hx, g2a=args.g2a,
        mode=args.mode, diag=args.diag, max_dense_dim=args.max_dense_dim,
        num_eigs=args.num_eigs, sigma=args.sigma, window=args.window,
        z_threshold=args.z_threshold, middle_fraction=tuple(args.middle_fraction),
        log_base=math.e if args.log_base == "e" else 2.0,
        top_candidates=args.top_candidates, output_dir=args.output_dir,
        save_all_states=args.save_all_states,
        target_energy_density=args.target_energy_density,
        energy_density_window=energy_density_window,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run all requested sizes, write outputs, and return a process status."""
    try:
        config = config_from_args(parse_args(argv))
    except (TypeError, ValueError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_config(config)

    results: list[FiniteSizeResult] = []
    for n_sites in config.sizes:
        started = time.perf_counter()
        try:
            results.append(analyse_one_size(config, n_sites))
        except Exception as error:  # Continue the explicitly requested size scan.
            results.append(_failed_result(config, n_sites, error, time.perf_counter() - started))

    write_dict_csv(
        config.output_dir / "k0_finite_size_summary.csv",
        [result.summary_row() for result in results], SUMMARY_COLUMNS,
    )
    write_dict_csv(
        config.output_dir / "k0_finite_size_candidates.csv",
        [row for result in results for row in result.candidate_rows], CANDIDATE_COLUMNS,
    )
    tracking_rows = [
        result.candidate_rows[0]
        for result in sorted(results, key=lambda item: item.N)
        if result.status == "ok" and result.candidate_rows
    ]
    write_dict_csv(
        config.output_dir / "k0_candidate_tracking.csv",
        tracking_rows,
        CANDIDATE_COLUMNS,
    )
    if config.save_all_states:
        write_dict_csv(
            config.output_dir / "k0_finite_size_all_states.csv",
            [row for result in results for row in result.all_state_rows], ALL_STATE_COLUMNS,
        )
    _plot_candidate_sequence(results, config)
    _print_table(results)
    return 1 if all(result.status == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
