"""Periodic k=0 Page curves for the j_max = 1/2 Ising-limit plaquette chain."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ising_limit_model import (
    HamiltonianMode,
    IsingLimitParameters,
    build_dense_hamiltonian as build_canonical_dense_hamiltonian,
    diagonal_energy as canonical_diagonal_energy,
    flip_amplitude as canonical_flip_amplitude,
    flip_physical_site,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    from scipy.sparse import coo_matrix, csr_matrix
    from scipy.sparse.linalg import eigsh
except ImportError as exc:  # pragma: no cover - exercised only without scipy.
    raise ImportError(
        "page_curve_k0.py requires scipy. Install it with `python -m pip install scipy`."
    ) from exc


DEFAULT_TARGETS_DELTA = [0.0, 9.99, 13.14, 24.48]


def couplings_from_g2a(g2a: float) -> tuple[float, float]:
    """Paper-style lattice-unit approximation for the j_max = 1/2 Ising limit."""
    return -3.0 * g2a / 16.0, 1.0 / g2a


def bit_at(state: int, i: int) -> int:
    return (state >> i) & 1


def z_value(state: int, i: int) -> int:
    return 1 if bit_at(state, i) == 0 else -1


def flip_bit(state: int, i: int) -> int:
    return state ^ (1 << i)


def state_to_bitstring(state: int, n_sites: int) -> str:
    return format(state, f"0{n_sites}b")


def translate_state(state: int, n_sites: int, shift: int = 1) -> int:
    """Cyclically translate bit i to bit (i + shift) mod n_sites."""
    shift %= n_sites
    if shift == 0:
        return state

    translated = 0
    for i in range(n_sites):
        if bit_at(state, i):
            translated |= 1 << ((i + shift) % n_sites)
    return translated


def orbit_of_state(state: int, n_sites: int) -> tuple[int, ...]:
    return tuple(sorted({translate_state(state, n_sites, shift) for shift in range(n_sites)}))


def build_k0_orbits(n_sites: int) -> tuple[list[int], list[tuple[int, ...]], dict[int, int]]:
    if n_sites < 2:
        raise ValueError("n_sites must be at least 2.")

    representatives: list[int] = []
    orbits: list[tuple[int, ...]] = []
    state_to_orbit_index: dict[int, int] = {}
    visited: set[int] = set()

    for state in range(1 << n_sites):
        if state in visited:
            continue
        orbit = orbit_of_state(state, n_sites)
        rep = min(orbit)
        idx = len(orbits)
        representatives.append(rep)
        orbits.append(orbit)
        for orbit_state in orbit:
            visited.add(orbit_state)
            state_to_orbit_index[orbit_state] = idx

    return representatives, orbits, state_to_orbit_index


def diagonal_energy(
    state: int,
    n_sites: int,
    J: float,
    mode: str | HamiltonianMode = HamiltonianMode.TOY,
) -> float:
    params = IsingLimitParameters(
        n_sites=n_sites,
        J=J,
        hx=0.0,
        periodic=True,
        mode=HamiltonianMode(mode),
    )
    return canonical_diagonal_energy(state, params)


def offdiag_flip_amplitude(
    state: int,
    site: int,
    n_sites: int,
    hx: float,
    mode: str | HamiltonianMode = HamiltonianMode.TOY,
) -> float:
    # Backward-compatible wrapper: this script's historical helper treated
    # ``site`` as a least-significant-bit position.  The canonical model uses
    # physical site 0 as the most-significant bit, so translate here.
    canonical_site = n_sites - 1 - site
    params = IsingLimitParameters(
        n_sites=n_sites,
        J=0.0,
        hx=hx,
        periodic=True,
        mode=HamiltonianMode(mode),
    )
    return canonical_flip_amplitude(state, canonical_site, params)


def build_k0_hamiltonian(
    n_sites: int,
    J: float,
    hx: float,
    mode: str | HamiltonianMode = HamiltonianMode.TOY,
) -> tuple[csr_matrix, list[tuple[int, ...]], dict[int, int]]:
    mode = HamiltonianMode(mode)
    _, orbits, state_to_orbit_index = build_k0_orbits(n_sites)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    params = IsingLimitParameters(
        n_sites=n_sites,
        J=J,
        hx=hx,
        periodic=True,
        mode=mode,
    )

    for a, orbit_a in enumerate(orbits):
        len_a = len(orbit_a)
        for x in orbit_a:
            rows.append(a)
            cols.append(a)
            data.append(canonical_diagonal_energy(x, params) / len_a)

            for site in range(n_sites):
                y = flip_physical_site(x, site, n_sites)
                b = state_to_orbit_index[y]
                len_b = len(orbits[b])
                amp = canonical_flip_amplitude(x, site, params)
                rows.append(b)
                cols.append(a)
                data.append(amp / math.sqrt(len_a * len_b))

    dim = len(orbits)
    H = coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=float).tocsr()
    antihermitian = H - H.T.conjugate()
    if antihermitian.nnz:
        max_error = float(np.max(np.abs(antihermitian.data)))
        if max_error > 1e-12:
            raise RuntimeError(f"k=0 Hamiltonian is not Hermitian: max error {max_error}")
    return H.tocsr(), orbits, state_to_orbit_index


def _append_unique_pair(
    pairs: list[tuple[float, np.ndarray]],
    energy: float,
    vector: np.ndarray,
    energy_tol: float = 1e-8,
    overlap_tol: float = 1e-6,
) -> None:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return
    vector = vector / norm
    for old_energy, old_vector in pairs:
        same_energy = abs(energy - old_energy) <= energy_tol * max(1.0, abs(old_energy), abs(energy))
        same_vector = 1.0 - abs(np.vdot(old_vector, vector)) <= overlap_tol
        if same_energy and same_vector:
            return
    pairs.append((float(energy), vector))


def _is_same_eigenpair(
    energy_a: float,
    vector_a: np.ndarray,
    energy_b: float,
    vector_b: np.ndarray,
    energy_tol: float = 1e-8,
    overlap_tol: float = 1e-6,
) -> bool:
    same_energy = abs(energy_a - energy_b) <= energy_tol * max(
        1.0, abs(energy_a), abs(energy_b)
    )
    same_vector = 1.0 - abs(np.vdot(vector_a, vector_b)) <= overlap_tol
    return bool(same_energy and same_vector)


def _residual_norm(H: csr_matrix, energy: float, vector: np.ndarray) -> float:
    return float(np.linalg.norm(H @ vector - energy * vector))


def _normalise_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise RuntimeError("Encountered a zero-norm eigenvector.")
    return vector / norm


def _selection_row(
    target_delta: float,
    energy: float,
    vector: np.ndarray,
    e0: float,
    H: csr_matrix,
    solver_mode: str,
    duplicate_of: Optional[int] = None,
) -> dict[str, Any]:
    actual_delta = float(energy - e0)
    return {
        "target_delta_E": float(target_delta),
        "actual_energy": float(energy),
        "actual_delta_E": actual_delta,
        "target_error": abs(actual_delta - float(target_delta)),
        "residual_norm": _residual_norm(H, energy, vector),
        "solver_mode": solver_mode,
        "duplicate_of": duplicate_of,
    }


def solve_selected_eigenpairs(
    H: csr_matrix,
    mode: str = "auto",
    targets_delta: Optional[Sequence[float]] = None,
    n_states: Optional[Sequence[int]] = None,
    max_dense_dim: int = 3000,
    k_near: int = 4,
    selection_policy: str = "closest",
    max_k_near: int = 32,
) -> dict[str, Any]:
    dim = H.shape[0]
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be square.")

    use_dense = mode == "dense" or (mode == "auto" and dim <= max_dense_dim)
    if mode not in {"auto", "dense", "sparse"}:
        raise ValueError("mode must be 'auto', 'dense', or 'sparse'.")
    if selection_policy not in {"closest", "all-nearby"}:
        raise ValueError("selection_policy must be 'closest' or 'all-nearby'.")
    if isinstance(k_near, bool) or int(k_near) < 1:
        raise ValueError("k_near must be a positive integer.")
    if isinstance(max_k_near, bool) or int(max_k_near) < 1:
        raise ValueError("max_k_near must be a positive integer.")
    k_near = int(k_near)
    max_k_near = int(max_k_near)
    targets = list(DEFAULT_TARGETS_DELTA if targets_delta is None else targets_delta)

    if use_dense:
        dense = H.toarray()
        energies, vectors = np.linalg.eigh(dense)
        e0 = float(energies[0])
        if targets_delta is None and selection_policy == "all-nearby":
            indices = [0, dim // 4, dim // 2, 3 * dim // 4]
        else:
            indices = [int(np.argmin(np.abs(energies - (e0 + float(delta))))) for delta in targets]
        if n_states is not None:
            indices = [int(i) for i in n_states]

        cleaned_indices: list[int] = []
        target_rows: list[dict[str, Any]] = []
        selected_rows: list[dict[str, Any]] = []
        for idx in indices:
            if idx < 0 or idx >= dim:
                raise IndexError(f"Eigenstate index {idx} is outside valid range 0..{dim - 1}.")
            vector = _normalise_vector(vectors[:, idx])
            target_delta = float(energies[idx] - e0) if n_states is not None else float(targets[len(target_rows)])
            duplicate_of = cleaned_indices.index(idx) if idx in cleaned_indices else None
            row = _selection_row(
                target_delta,
                float(energies[idx]),
                vector,
                e0,
                H,
                "dense",
                duplicate_of=duplicate_of,
            )
            target_rows.append(row)
            if duplicate_of is None:
                cleaned_indices.append(idx)
                selected_rows.append(row)

        return {
            "mode": "dense",
            "energies": energies[cleaned_indices],
            "vectors": vectors[:, cleaned_indices],
            "all_energies": energies,
            "indices": cleaned_indices,
            "ground_energy": e0,
            "target_rows": target_rows,
            "selected_rows": selected_rows,
            "selection_policy": selection_policy,
        }

    if dim < 2:
        dense = H.toarray()
        energies, vectors = np.linalg.eigh(dense)
        return {
            "mode": "dense",
            "energies": energies,
            "vectors": vectors,
            "all_energies": energies,
            "indices": [0],
            "ground_energy": float(energies[0]),
            "target_rows": [
                _selection_row(0.0, float(energies[0]), vectors[:, 0], float(energies[0]), H, "dense")
            ],
            "selected_rows": [
                _selection_row(0.0, float(energies[0]), vectors[:, 0], float(energies[0]), H, "dense")
            ],
            "selection_policy": selection_policy,
        }

    pairs: list[tuple[float, np.ndarray]] = []
    target_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    ground_energy, ground_vector = eigsh(H, k=1, which="SA")
    e0 = float(ground_energy[0])
    ground_vec = _normalise_vector(ground_vector[:, 0])

    try:
        if selection_policy == "all-nearby":
            _append_unique_pair(pairs, e0, ground_vec)
            for delta in targets:
                if abs(delta) <= 1e-12:
                    continue
                target_absolute = e0 + float(delta)
                k = max(1, min(k_near, dim - 1))
                energies, vectors = eigsh(H, k=k, sigma=target_absolute, which="LM")
                order = np.argsort(energies)
                for idx in order:
                    _append_unique_pair(pairs, float(energies[idx]), vectors[:, idx])
            pairs.sort(key=lambda item: item[0])
            energies = np.array([item[0] for item in pairs], dtype=float)
            vectors = np.column_stack([item[1] for item in pairs])
            for energy, vector in pairs:
                selected_rows.append(
                    _selection_row(float(energy - e0), energy, vector, e0, H, "sparse")
                )
            return {
                "mode": "sparse",
                "energies": energies,
                "vectors": vectors,
                "all_energies": None,
                "indices": None,
                "ground_energy": e0,
                "target_rows": selected_rows,
                "selected_rows": selected_rows,
                "selection_policy": selection_policy,
            }

        selected_pairs: list[tuple[float, np.ndarray]] = []
        for delta in targets:
            target_delta = float(delta)
            target_absolute = e0 + target_delta
            if abs(target_delta) <= 1e-12:
                energy = e0
                vector = ground_vec
                duplicate_of = None
                for old_index, (old_energy, old_vector) in enumerate(selected_pairs):
                    if _is_same_eigenpair(energy, vector, old_energy, old_vector):
                        duplicate_of = old_index
                        break
                row = _selection_row(
                    target_delta, energy, vector, e0, H, "sparse", duplicate_of=duplicate_of
                )
                target_rows.append(row)
                if duplicate_of is None:
                    selected_pairs.append((energy, vector))
                    selected_rows.append(row)
                continue

            chosen: Optional[tuple[float, np.ndarray, Optional[int]]] = None
            k = max(1, min(k_near, dim - 1))
            k_limit = max(1, min(max_k_near, dim - 1))
            while k <= k_limit:
                energies, vectors = eigsh(H, k=k, sigma=target_absolute, which="LM")
                order = np.argsort(np.abs(energies - target_absolute))
                duplicate_candidate: Optional[tuple[float, np.ndarray, int]] = None
                for idx in order:
                    energy = float(energies[idx])
                    vector = _normalise_vector(vectors[:, idx])
                    duplicate_of = None
                    for old_index, (old_energy, old_vector) in enumerate(selected_pairs):
                        if _is_same_eigenpair(energy, vector, old_energy, old_vector):
                            duplicate_of = old_index
                            break
                    if duplicate_of is None:
                        chosen = (energy, vector, None)
                        break
                    if duplicate_candidate is None:
                        duplicate_candidate = (energy, vector, duplicate_of)
                if chosen is not None:
                    break
                if k >= k_limit:
                    if duplicate_candidate is not None:
                        energy, vector, duplicate_of = duplicate_candidate
                        chosen = (energy, vector, duplicate_of)
                    break
                k = min(k_limit, max(k + 1, 2 * k))

            if chosen is None:
                raise RuntimeError(
                    f"No eigenstate converged near target Delta E={target_delta:.12g}."
                )

            energy, vector, duplicate_of = chosen
            row = _selection_row(
                target_delta,
                energy,
                vector,
                e0,
                H,
                "sparse",
                duplicate_of=duplicate_of,
            )
            target_rows.append(row)
            if duplicate_of is None:
                selected_pairs.append((energy, vector))
                selected_rows.append(row)
    except Exception as exc:
        raise RuntimeError(
            "Sparse shift-invert eigensolve failed. Try a smaller N, increasing "
            "--max-dense-dim if memory allows, or adjusting "
            "--targets-delta/--k-near/--max-k-near."
        ) from exc

    energies = np.array([item[0] for item in selected_pairs], dtype=float)
    vectors = np.column_stack([item[1] for item in selected_pairs])
    return {
        "mode": "sparse",
        "energies": energies,
        "vectors": vectors,
        "all_energies": None,
        "indices": None,
        "ground_energy": e0,
        "target_rows": target_rows,
        "selected_rows": selected_rows,
        "selection_policy": selection_policy,
    }


def diagonalize_k0_hamiltonian(
    H: csr_matrix,
    diag: str = "dense",
    num_eigs: int = 100,
    sigma: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    dim = H.shape[0]
    if diag == "dense":
        energies, vectors = np.linalg.eigh(H.toarray())
    elif diag == "sparse":
        if dim <= 2:
            raise ValueError("Sparse diagonalisation requires dimension > 2.")
        k = min(int(num_eigs), dim - 2)
        if k < 1:
            raise ValueError("num_eigs must request at least one eigenstate.")
        if sigma is None:
            e_min = float(eigsh(H, k=1, which="SA", return_eigenvectors=False)[0])
            e_max = float(eigsh(H, k=1, which="LA", return_eigenvectors=False)[0])
            sigma = 0.5 * (e_min + e_max)
        energies, vectors = eigsh(H, k=k, sigma=sigma, which="LM")
        order = np.argsort(energies)
        energies = energies[order]
        vectors = vectors[:, order]
    else:
        raise ValueError("diag must be 'dense' or 'sparse'.")

    return np.real_if_close(energies).astype(float), vectors


def reconstruct_full_wavefunction(
    k0_vector: np.ndarray,
    orbits: Sequence[Sequence[int]],
    n_sites: int,
) -> np.ndarray:
    psi_full = np.zeros(1 << n_sites, dtype=complex)
    for coeff, orbit in zip(k0_vector, orbits):
        amp = coeff / math.sqrt(len(orbit))
        for state in orbit:
            psi_full[state] += amp

    norm = np.linalg.norm(psi_full)
    if norm == 0:
        raise RuntimeError("Reconstructed a zero-norm wavefunction.")
    return psi_full / norm


def entanglement_entropy_full_state(
    psi_full: np.ndarray,
    n_sites: int,
    subsystem_sites: Sequence[int],
    log_base: float = math.e,
) -> float:
    psi_full = np.asarray(psi_full, dtype=complex)
    expected_shape = (1 << n_sites,)
    if psi_full.shape != expected_shape:
        raise ValueError(f"psi_full must have shape {expected_shape}, got {psi_full.shape}.")

    subsystem = tuple(int(site) for site in subsystem_sites)
    if len(subsystem) == 0 or len(subsystem) == n_sites:
        return 0.0
    if len(set(subsystem)) != len(subsystem):
        raise ValueError("subsystem_sites must be unique.")
    if min(subsystem) < 0 or max(subsystem) >= n_sites:
        raise ValueError(f"subsystem_sites must lie in 0..{n_sites - 1}.")

    complement = tuple(site for site in range(n_sites) if site not in subsystem)
    site_to_axis = {site: n_sites - 1 - site for site in range(n_sites)}
    axes = tuple(site_to_axis[site] for site in subsystem + complement)

    psi = psi_full / np.linalg.norm(psi_full)
    psi_tensor = psi.reshape((2,) * n_sites)
    reordered = np.transpose(psi_tensor, axes=axes)
    matrix = reordered.reshape(2 ** len(subsystem), 2 ** len(complement))
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    probabilities = np.clip(singular_values**2, 0.0, 1.0)
    probabilities = probabilities[probabilities > 1e-14]
    if len(probabilities) == 0:
        return 0.0

    logs = np.log(probabilities)
    if log_base != math.e:
        logs = logs / math.log(log_base)
    return float(-np.sum(probabilities * logs))


def page_curve_for_state(
    psi_full: np.ndarray,
    n_sites: int,
    log_base: float = math.e,
) -> tuple[list[int], list[float]]:
    NA = list(range(n_sites + 1))
    SA = [0.0]
    for size in range(1, n_sites):
        SA.append(
            entanglement_entropy_full_state(
                psi_full,
                n_sites,
                subsystem_sites=tuple(range(size)),
                log_base=log_base,
            )
        )
    SA.append(0.0)
    return NA, SA


def find_scar_candidates(
    energies: np.ndarray,
    entropies: np.ndarray,
    window: int = 20,
    z_threshold: float = 2.5,
    middle_fraction: tuple[float, float] = (0.25, 0.75),
) -> list[dict[str, float]]:
    if len(energies) != len(entropies):
        raise ValueError("energies and entropies must have the same length.")
    if len(energies) == 0:
        return []

    order = np.argsort(energies)
    sorted_energies = energies[order]
    sorted_entropies = entropies[order]
    e0 = float(np.min(energies))

    lo_frac, hi_frac = middle_fraction
    n_states = len(energies)
    lo = max(0, int(math.floor(lo_frac * n_states)))
    hi = min(n_states, int(math.ceil(hi_frac * n_states)))
    half_window = max(1, int(window) // 2)

    rows: list[dict[str, float]] = []
    for sorted_pos in range(lo, hi):
        start = max(0, sorted_pos - half_window)
        stop = min(n_states, sorted_pos + half_window + 1)
        local = np.delete(sorted_entropies[start:stop], sorted_pos - start)
        if len(local) == 0:
            local = sorted_entropies[start:stop]

        local_median = float(np.median(local))
        local_std = float(np.std(local, ddof=1)) if len(local) > 1 else 0.0
        score = 0.0 if local_std < 1e-12 else (local_median - float(sorted_entropies[sorted_pos])) / local_std
        original_index = int(order[sorted_pos])
        rows.append(
            {
                "eigenstate_index": original_index,
                "energy": float(sorted_energies[sorted_pos]),
                "excitation_energy": float(sorted_energies[sorted_pos] - e0),
                "half_chain_entropy": float(sorted_entropies[sorted_pos]),
                "local_median_entropy": local_median,
                "anomaly_score": float(score),
                "is_candidate": bool(score > z_threshold),
            }
        )

    return sorted(rows, key=lambda row: row["anomaly_score"], reverse=True)


def top_wavefunction_components(psi_full: np.ndarray, n_sites: int, top_k: int = 20) -> list[dict[str, float]]:
    probs = np.abs(np.asarray(psi_full)) ** 2
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [
        {
            "basis_index": int(idx),
            "bitstring": state_to_bitstring(int(idx), n_sites),
            "probability": float(probs[idx]),
        }
        for idx in top_indices
    ]


def write_dict_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return path


def selected_targets_csv_path(output: Optional[str]) -> Optional[Path]:
    if not output:
        return None
    output_path = Path(output)
    return output_path.with_name(f"{output_path.stem}_selected_targets.csv")


def save_selected_targets_csv(
    output: Optional[str],
    target_rows: Sequence[dict[str, Any]],
    hamiltonian_mode: str | HamiltonianMode,
) -> Optional[Path]:
    path = selected_targets_csv_path(output)
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = HamiltonianMode(hamiltonian_mode)
    rows = [dict(row, hamiltonian_mode=mode.value) for row in target_rows]
    return write_dict_csv(
        path,
        rows,
        [
            "target_delta_E",
            "actual_energy",
            "actual_delta_E",
            "target_error",
            "residual_norm",
            "solver_mode",
            "hamiltonian_mode",
        ],
    )


def save_spectrum_csv(
    output_dir: Path,
    energies: np.ndarray,
    entropies: np.ndarray,
    candidate_rows: Sequence[dict[str, float]],
    mode: str | HamiltonianMode,
) -> tuple[Path, Path]:
    mode = HamiltonianMode(mode)
    e0 = float(np.min(energies))
    spectrum_rows = [
        {
            "eigenstate_index": idx,
            "mode": mode.value,
            "energy": float(energy),
            "excitation_energy": float(energy - e0),
            "half_chain_entropy": float(entropies[idx]),
        }
        for idx, energy in enumerate(energies)
    ]
    spectrum_path = write_dict_csv(
        output_dir / "k0_spectrum_entanglement.csv",
        spectrum_rows,
        ["eigenstate_index", "mode", "energy", "excitation_energy", "half_chain_entropy"],
    )
    candidate_rows_with_mode = [dict(row, mode=mode.value) for row in candidate_rows]
    candidates_path = write_dict_csv(
        output_dir / "k0_scar_candidates.csv",
        candidate_rows_with_mode,
        [
            "eigenstate_index",
            "mode",
            "energy",
            "excitation_energy",
            "half_chain_entropy",
            "local_median_entropy",
            "anomaly_score",
            "is_candidate",
        ],
    )
    return spectrum_path, candidates_path


def plot_entropy_vs_energy(
    output_dir: Path,
    energies: np.ndarray,
    entropies: np.ndarray,
    candidate_rows: Sequence[dict[str, float]],
    mode: str | HamiltonianMode,
) -> Path:
    import matplotlib.pyplot as plt
    mode = HamiltonianMode(mode)

    e0 = float(np.min(energies))
    candidates = [row for row in candidate_rows if row["is_candidate"]]

    fig, ax = plt.subplots(figsize=(9.0, 5.5), constrained_layout=True)
    ax.scatter(energies - e0, entropies, s=16, alpha=0.65, label="k=0 eigenstates")
    if candidates:
        idx = np.array([int(row["eigenstate_index"]) for row in candidates], dtype=int)
        ax.scatter(energies[idx] - e0, entropies[idx], s=52, marker="x", color="crimson", label="Scar candidates")
        best_idx = int(candidates[0]["eigenstate_index"])
        ax.annotate(
            f"#{best_idx}",
            xy=(energies[best_idx] - e0, entropies[best_idx]),
            xytext=(8, 8),
            textcoords="offset points",
            color="crimson",
        )
    ax.set_xlabel(r"Excitation energy $E - E_0$")
    ax.set_ylabel(r"Half-chain entropy $S_A$")
    ax.set_title(f"k=0 {mode.value} entanglement entropy versus excitation energy")
    ax.margins(x=0.04, y=0.08)
    ax.legend(fontsize=9)

    path = output_dir / "k0_entropy_vs_energy.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def choose_nearby_typical_index(
    energies: np.ndarray,
    entropies: np.ndarray,
    candidate_index: int,
    candidate_rows: Sequence[dict[str, float]],
) -> Optional[int]:
    candidate_flags = {int(row["eigenstate_index"]): bool(row["is_candidate"]) for row in candidate_rows}
    candidate_row = next((row for row in candidate_rows if int(row["eigenstate_index"]) == candidate_index), None)
    local_target = None if candidate_row is None else float(candidate_row["local_median_entropy"])

    order = np.argsort(np.abs(energies - energies[candidate_index]))
    for idx in order:
        idx = int(idx)
        if idx == candidate_index or candidate_flags.get(idx, False):
            continue
        if local_target is None or entropies[idx] >= local_target:
            return idx
    for idx in order:
        idx = int(idx)
        if idx != candidate_index:
            return idx
    return None


def plot_best_page_curve(
    output_dir: Path,
    full_vectors: Sequence[np.ndarray],
    energies: np.ndarray,
    entropies: np.ndarray,
    candidate_rows: Sequence[dict[str, float]],
    best_index: int,
    n_sites: int,
    log_base: float,
    mode: str | HamiltonianMode,
) -> Path:
    import matplotlib.pyplot as plt
    mode = HamiltonianMode(mode)

    candidate_curve = page_curve_for_state(full_vectors[best_index], n_sites, log_base=log_base)
    typical_index = choose_nearby_typical_index(energies, entropies, best_index, candidate_rows)
    typical_curve = None if typical_index is None else page_curve_for_state(full_vectors[typical_index], n_sites, log_base=log_base)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    ax.plot(*candidate_curve, marker="o", markersize=5, linewidth=1.8, label=f"Candidate #{best_index}")
    if typical_curve is not None and typical_index is not None:
        ax.plot(*typical_curve, marker="s", markersize=5, linewidth=1.8, label=f"Nearby typical #{typical_index}")
    ax.set_xlabel(r"Subsystem size $N_A$")
    ax.set_ylabel(r"Entanglement entropy $S_A$")
    ax.set_title(f"k=0 {mode.value} entanglement page curve")
    ax.margins(x=0.04, y=0.08)
    ax.legend(fontsize=9)

    path = output_dir / f"k0_page_curve_candidate_{best_index}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_wavefunction_components(
    output_dir: Path,
    components: Sequence[dict[str, float]],
    candidate_index: int,
    mode: str | HamiltonianMode,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    mode = HamiltonianMode(mode)

    csv_path = write_dict_csv(
        output_dir / f"k0_wavefunction_components_candidate_{candidate_index}.csv",
        components,
        ["basis_index", "bitstring", "probability"],
    )

    labels = [str(row["bitstring"]) for row in components]
    values = [float(row["probability"]) for row in components]
    fig_width = max(9.0, 0.45 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5), constrained_layout=True)
    ax.bar(np.arange(len(labels)), values, color="tab:blue")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel(r"Probability $|c_{basis}|^2$")
    ax.set_title(
        f"Largest full-basis components for k=0 {mode.value} candidate #{candidate_index}"
    )

    png_path = output_dir / f"k0_wavefunction_components_candidate_{candidate_index}.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return png_path, csv_path


def run_k0_scar_analysis(
    n_sites: int,
    g2a: float,
    J: Optional[float],
    hx: Optional[float],
    diag: str,
    num_eigs: int,
    sigma: Optional[float],
    window: int,
    z_threshold: float,
    output_dir: Path,
    log_base: float,
    top_k: int,
    mode: str | HamiltonianMode = HamiltonianMode.TOY,
) -> dict[str, Any]:
    mode = HamiltonianMode(mode)
    if J is None or hx is None:
        default_J, default_hx = couplings_from_g2a(g2a)
        if J is None:
            J = default_J
        if hx is None:
            hx = default_hx

    output_dir.mkdir(parents=True, exist_ok=True)
    H, orbits, _ = build_k0_hamiltonian(n_sites, J, hx, mode=mode)
    print(
        f"Building k=0 Hamiltonian for N={n_sites}, "
        f"dim(k0)={H.shape[0]}, mode={mode.value}..."
    )
    print(f"Diagonalising with {diag} mode...")
    energies, vectors = diagonalize_k0_hamiltonian(H, diag=diag, num_eigs=num_eigs, sigma=sigma)

    half_chain = tuple(range(n_sites // 2))
    full_vectors: list[np.ndarray] = []
    entropies: list[float] = []
    print(f"Computing half-chain entanglement for {len(energies)} k=0 eigenstates...")
    for idx in range(vectors.shape[1]):
        psi_full = reconstruct_full_wavefunction(vectors[:, idx], orbits, n_sites)
        full_vectors.append(psi_full)
        entropies.append(entanglement_entropy_full_state(psi_full, n_sites, half_chain, log_base=log_base))
    entropy_array = np.array(entropies, dtype=float)

    candidate_rows = find_scar_candidates(energies, entropy_array, window=window, z_threshold=z_threshold)
    true_candidates = [row for row in candidate_rows if row["is_candidate"]]
    spectrum_path, candidates_path = save_spectrum_csv(
        output_dir, energies, entropy_array, candidate_rows, mode
    )
    entropy_plot_path = plot_entropy_vs_energy(
        output_dir, energies, entropy_array, candidate_rows, mode
    )

    page_plot_path: Optional[Path] = None
    wf_plot_path: Optional[Path] = None
    wf_csv_path: Optional[Path] = None
    best = true_candidates[0] if true_candidates else (candidate_rows[0] if candidate_rows else None)

    if best is not None:
        best_index = int(best["eigenstate_index"])
        page_plot_path = plot_best_page_curve(
            output_dir,
            full_vectors,
            energies,
            entropy_array,
            candidate_rows,
            best_index,
            n_sites,
            log_base,
            mode,
        )
        components = top_wavefunction_components(full_vectors[best_index], n_sites, top_k=top_k)
        wf_plot_path, wf_csv_path = plot_wavefunction_components(
            output_dir, components, best_index, mode
        )

        print("\nTop k=0 scar candidates by anomaly score:")
        for row in candidate_rows[: min(10, len(candidate_rows))]:
            marker = "*" if row["is_candidate"] else " "
            print(
                f"{marker} idx={int(row['eigenstate_index']):4d} "
                f"E={row['energy']: .8f} "
                f"E-E0={row['excitation_energy']: .8f} "
                f"S={row['half_chain_entropy']: .8f} "
                f"score={row['anomaly_score']: .3f}"
            )

        print(f"\nLargest full-basis components for best k=0 state #{best_index}:")
        for row in components:
            print(f"  {row['basis_index']:5d}  {row['bitstring']}  p={row['probability']:.8e}")

    print("\nSummary")
    print(f"  chain length N: {n_sites}")
    print(f"  full Hilbert-space dimension: {1 << n_sites}")
    print(f"  k=0 Hilbert-space dimension: {H.shape[0]}")
    print(f"  Hamiltonian mode: {mode.value}")
    print(f"  couplings: J={J:.12g}, hx={hx:.12g}, g2a={g2a:.12g}")
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

    return {
        "energies": energies,
        "entropies": entropy_array,
        "candidate_rows": candidate_rows,
        "output_dir": output_dir,
        "mode": mode.value,
    }


def plot_page_curves_k0(
    n_sites: int,
    g2a: float = 1.2,
    J: Optional[float] = None,
    hx: Optional[float] = None,
    targets_delta: Optional[list[float]] = None,
    max_dense_dim: int = 3000,
    output: Optional[str] = None,
    show: bool = True,
    log_base: float = math.e,
    k_near: int = 4,
    mode: str | HamiltonianMode = HamiltonianMode.TOY,
    selection_policy: str = "closest",
    max_k_near: int = 32,
) -> dict[str, Any]:
    mode = HamiltonianMode(mode)
    if J is None or hx is None:
        default_J, default_hx = couplings_from_g2a(g2a)
        if J is None:
            J = default_J
        if hx is None:
            hx = default_hx

    H, orbits, state_to_orbit_index = build_k0_hamiltonian(n_sites, J, hx, mode=mode)
    print(f"k=0 Hilbert-space dimension: {H.shape[0]}")
    print(f"Hamiltonian mode: {mode.value}")
    solution = solve_selected_eigenpairs(
        H,
        mode="auto",
        targets_delta=targets_delta,
        max_dense_dim=max_dense_dim,
        k_near=k_near,
        selection_policy=selection_policy,
        max_k_near=max_k_near,
    )
    print(f"Eigensolver mode: {solution['mode']}")
    print(f"Selection policy: {solution['selection_policy']}")

    energies = np.asarray(solution["energies"], dtype=float)
    vectors = np.asarray(solution["vectors"])
    e0 = float(solution["ground_energy"])
    selected_rows = list(solution.get("selected_rows", []))
    target_rows = list(solution.get("target_rows", selected_rows))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.0, 6.25), constrained_layout=True)
    curves: list[dict[str, Any]] = []

    for idx in range(vectors.shape[1]):
        psi_full = reconstruct_full_wavefunction(vectors[:, idx], orbits, n_sites)
        NA, SA = page_curve_for_state(psi_full, n_sites, log_base=log_base)
        delta_e = float(energies[idx] - e0)
        row = selected_rows[idx] if idx < len(selected_rows) else {}
        target_delta = float(row.get("target_delta_E", delta_e))
        target_error = float(row.get("target_error", 0.0))
        label = (
            rf"$\Delta E_{{target}}={target_delta:.3g}$, "
            rf"$\Delta E={delta_e:.3g}$, "
            rf"$|\epsilon|={target_error:.1e}$"
        )
        ax.plot(NA, SA, marker="o", markersize=5, linewidth=1.8, label=label)
        curves.append(
            {
                "mode": mode.value,
                "target_delta_E": target_delta,
                "energy": float(energies[idx]),
                "delta_E": delta_e,
                "target_error": target_error,
                "residual_norm": float(row.get("residual_norm", np.nan)),
                "NA": NA,
                "SA": SA,
            }
        )

    ax.set_xlabel(r"Subsystem size $N_A$", fontsize=10)
    ax.set_ylabel(r"Entanglement entropy $S_A$", fontsize=10)
    ax.set_title(
        f"N={n_sites}, periodic k=0 {mode.value}, g2a={g2a:g}, J={J:.6g}, hx={hx:.6g}, "
        f"dim(k0)={H.shape[0]}",
        fontsize=11,
        pad=10,
    )
    ax.legend(
        title="Excitation energy",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=9,
    )
    ax.tick_params(axis="both", labelsize=9)
    ax.margins(x=0.04, y=0.08)
    ax.grid(True, alpha=0.3)

    if output:
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved Page-curve plot to: {output}")
    selected_targets_path = save_selected_targets_csv(output, target_rows, mode)
    if selected_targets_path is not None:
        print(f"Saved selected-target metadata to: {selected_targets_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    for row in target_rows:
        duplicate_of = row.get("duplicate_of")
        duplicate_note = "" if duplicate_of is None else f" duplicate_of_curve={duplicate_of}"
        print(
            "Target selection: "
            f"target Delta E={row['target_delta_E']:.12g}, "
            f"actual Delta E={row['actual_delta_E']:.12g}, "
            f"error={row['target_error']:.6e}, "
            f"residual={row['residual_norm']:.6e}"
            f"{duplicate_note}"
        )

    return {
        "n_sites": n_sites,
        "full_dim": 1 << n_sites,
        "k0_dim": H.shape[0],
        "g2a": g2a,
        "J": J,
        "hx": hx,
        "mode": mode.value,
        "energies": energies,
        "delta_E": energies - e0,
        "curves": curves,
        "output": output,
        "selected_targets_csv": selected_targets_path,
        "eigensolver_mode": solution["mode"],
        "selection_policy": solution["selection_policy"],
        "target_rows": target_rows,
        "selected_rows": selected_rows,
        "orbits": orbits,
        "state_to_orbit_index": state_to_orbit_index,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Periodic k=0 Page curves for the Ising-limit chain.")
    parser.add_argument("--n-sites", type=int, default=8)
    parser.add_argument("--g2a", type=float, default=1.2)
    parser.add_argument("--J", type=float, default=None)
    parser.add_argument("--hx", type=float, default=None)
    parser.add_argument("--targets-delta", type=float, nargs="*", default=None)
    parser.add_argument("--max-dense-dim", type=int, default=3000)
    parser.add_argument("--output", type=str, default="page_curve_k0.png")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--log-base", choices=["e", "2"], default="e")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in HamiltonianMode],
        default=HamiltonianMode.TOY.value,
        help="Hamiltonian convention to use (default: toy).",
    )
    parser.add_argument("--k-near", type=int, default=4)
    parser.add_argument(
        "--max-k-near",
        type=int,
        default=32,
        help="Maximum adaptive shift-invert subspace size for --selection-policy closest.",
    )
    parser.add_argument(
        "--selection-policy",
        choices=("closest", "all-nearby"),
        default="closest",
        help=(
            "closest selects one eigenstate per target; all-nearby preserves "
            "the old behavior and plots every nearby eigenstate returned."
        ),
    )
    parser.add_argument("--find-scars", action="store_true", help="Run k=0 scar-candidate analysis instead of selected Page curves.")
    parser.add_argument("--diag", choices=("dense", "sparse"), default="dense", help="Diagonalisation strategy for --find-scars.")
    parser.add_argument("--num-eigs", type=int, default=100, help="Number of sparse k=0 eigenstates to compute for --find-scars.")
    parser.add_argument("--sigma", type=float, default=None, help="Shift-invert target energy for sparse --find-scars.")
    parser.add_argument("--window", type=int, default=20, help="Energy-index window for local entropy comparison.")
    parser.add_argument("--z-threshold", type=float, default=2.5, help="Scar anomaly score threshold.")
    parser.add_argument("--output-dir", type=Path, default=Path("scar_outputs"), help="Directory for --find-scars outputs.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of wavefunction components to print and plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_base = math.e if args.log_base == "e" else 2.0
    if args.find_scars:
        run_k0_scar_analysis(
            n_sites=args.n_sites,
            g2a=args.g2a,
            J=args.J,
            hx=args.hx,
            diag=args.diag,
            num_eigs=args.num_eigs,
            sigma=args.sigma,
            window=args.window,
            z_threshold=args.z_threshold,
            output_dir=args.output_dir,
            log_base=log_base,
            top_k=args.top_k,
            mode=args.mode,
        )
        return

    data = plot_page_curves_k0(
        n_sites=args.n_sites,
        g2a=args.g2a,
        J=args.J,
        hx=args.hx,
        targets_delta=args.targets_delta,
        max_dense_dim=args.max_dense_dim,
        output=args.output,
        show=not args.no_show,
        log_base=log_base,
        k_near=args.k_near,
        mode=args.mode,
        selection_policy=args.selection_policy,
        max_k_near=args.max_k_near,
    )

    print(f"N: {data['n_sites']}")
    print(f"Full Hilbert dimension: {data['full_dim']}")
    print(f"k=0 Hilbert dimension: {data['k0_dim']}")
    print(f"Hamiltonian mode: {data['mode']}")
    print(f"Couplings: J = {data['J']:.12g}, hx = {data['hx']:.12g}, g2a = {data['g2a']:.12g}")
    print(f"Eigensolver mode: {data['eigensolver_mode']}")
    print(f"Selection policy: {data['selection_policy']}")
    print("Selected energies:")
    for energy, delta_e in zip(data["energies"], data["delta_E"]):
        print(f"  E = {energy:.12g}, Delta E = {delta_e:.12g}")
    print(f"Output filename: {data['output']}")
    if data["selected_targets_csv"] is not None:
        print(f"Selected-target CSV: {data['selected_targets_csv']}")


if __name__ == "__main__":
    main()
