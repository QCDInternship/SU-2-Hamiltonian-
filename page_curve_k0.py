"""Periodic k=0 Page curves for the j_max = 1/2 Ising-limit plaquette chain."""

from __future__ import annotations

import argparse
import math
from typing import Any, Optional, Sequence

import numpy as np

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


def diagonal_energy(state: int, n_sites: int, J: float) -> float:
    energy = 0.0
    for i in range(n_sites):
        zi = z_value(state, i)
        energy += J * zi * z_value(state, (i + 1) % n_sites)
        energy += -2.0 * J * zi
    return energy


def offdiag_flip_amplitude(state: int, site: int, n_sites: int, hx: float) -> float:
    left = z_value(state, (site - 1) % n_sites)
    right = z_value(state, (site + 1) % n_sites)
    return hx * (left + right) / math.sqrt(2.0)


def build_k0_hamiltonian(
    n_sites: int,
    J: float,
    hx: float,
) -> tuple[csr_matrix, list[tuple[int, ...]], dict[int, int]]:
    _, orbits, state_to_orbit_index = build_k0_orbits(n_sites)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for a, orbit_a in enumerate(orbits):
        len_a = len(orbit_a)
        for x in orbit_a:
            rows.append(a)
            cols.append(a)
            data.append(diagonal_energy(x, n_sites, J) / len_a)

            for site in range(n_sites):
                y = flip_bit(x, site)
                b = state_to_orbit_index[y]
                len_b = len(orbits[b])
                amp = offdiag_flip_amplitude(x, site, n_sites, hx)
                rows.append(b)
                cols.append(a)
                data.append(amp / math.sqrt(len_a * len_b))

    dim = len(orbits)
    H = coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=float).tocsr()
    H = 0.5 * (H + H.T.conjugate())
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


def solve_selected_eigenpairs(
    H: csr_matrix,
    mode: str = "auto",
    targets_delta: Optional[Sequence[float]] = None,
    n_states: Optional[Sequence[int]] = None,
    max_dense_dim: int = 3000,
    k_near: int = 4,
) -> dict[str, Any]:
    dim = H.shape[0]
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be square.")

    use_dense = mode == "dense" or (mode == "auto" and dim <= max_dense_dim)
    if mode not in {"auto", "dense", "sparse"}:
        raise ValueError("mode must be 'auto', 'dense', or 'sparse'.")

    if use_dense:
        dense = H.toarray()
        energies, vectors = np.linalg.eigh(dense)
        if targets_delta is None:
            indices = [0, dim // 4, dim // 2, 3 * dim // 4]
        else:
            e0 = energies[0]
            indices = [int(np.argmin(np.abs(energies - (e0 + delta)))) for delta in targets_delta]
        if n_states is not None:
            indices = [int(i) for i in n_states]

        cleaned_indices: list[int] = []
        for idx in indices:
            if idx < 0 or idx >= dim:
                raise IndexError(f"Eigenstate index {idx} is outside valid range 0..{dim - 1}.")
            if idx not in cleaned_indices:
                cleaned_indices.append(idx)

        return {
            "mode": "dense",
            "energies": energies[cleaned_indices],
            "vectors": vectors[:, cleaned_indices],
            "all_energies": energies,
            "indices": cleaned_indices,
            "ground_energy": float(energies[0]),
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
        }

    if targets_delta is None:
        targets_delta = DEFAULT_TARGETS_DELTA

    pairs: list[tuple[float, np.ndarray]] = []
    ground_energy, ground_vector = eigsh(H, k=1, which="SA")
    e0 = float(ground_energy[0])
    _append_unique_pair(pairs, e0, ground_vector[:, 0])

    try:
        for delta in targets_delta:
            if abs(delta) <= 1e-12:
                continue
            target_absolute = e0 + float(delta)
            k = max(1, min(int(k_near), dim - 1))
            energies, vectors = eigsh(H, k=k, sigma=target_absolute, which="LM")
            order = np.argsort(energies)
            for idx in order:
                _append_unique_pair(pairs, float(energies[idx]), vectors[:, idx])
    except Exception as exc:
        raise RuntimeError(
            "Sparse shift-invert eigensolve failed. Try a smaller N, increasing "
            "--max-dense-dim if memory allows, or adjusting --targets-delta/--k-near."
        ) from exc

    pairs.sort(key=lambda item: item[0])
    energies = np.array([item[0] for item in pairs], dtype=float)
    vectors = np.column_stack([item[1] for item in pairs])
    return {
        "mode": "sparse",
        "energies": energies,
        "vectors": vectors,
        "all_energies": None,
        "indices": None,
        "ground_energy": e0,
    }


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
) -> dict[str, Any]:
    if J is None or hx is None:
        default_J, default_hx = couplings_from_g2a(g2a)
        if J is None:
            J = default_J
        if hx is None:
            hx = default_hx

    H, orbits, state_to_orbit_index = build_k0_hamiltonian(n_sites, J, hx)
    print(f"k=0 Hilbert-space dimension: {H.shape[0]}")
    solution = solve_selected_eigenpairs(
        H,
        mode="auto",
        targets_delta=targets_delta,
        max_dense_dim=max_dense_dim,
        k_near=k_near,
    )
    print(f"Eigensolver mode: {solution['mode']}")

    energies = np.asarray(solution["energies"], dtype=float)
    vectors = np.asarray(solution["vectors"])
    e0 = float(solution["ground_energy"])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    curves: list[dict[str, Any]] = []

    for idx in range(vectors.shape[1]):
        psi_full = reconstruct_full_wavefunction(vectors[:, idx], orbits, n_sites)
        NA, SA = page_curve_for_state(psi_full, n_sites, log_base=log_base)
        delta_e = float(energies[idx] - e0)
        ax.plot(NA, SA, marker="o", label=fr"$\Delta E = {delta_e:.3f}$")
        curves.append({"energy": float(energies[idx]), "delta_E": delta_e, "NA": NA, "SA": SA})

    ax.set_xlabel(r"Subsystem size $N_A$")
    ax.set_ylabel(r"Entanglement entropy $S_A$")
    ax.set_title(
        f"N={n_sites}, periodic k=0, g2a={g2a:g}, J={J:.6g}, hx={hx:.6g}, "
        f"dim(k0)={H.shape[0]}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved Page-curve plot to: {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "n_sites": n_sites,
        "full_dim": 1 << n_sites,
        "k0_dim": H.shape[0],
        "g2a": g2a,
        "J": J,
        "hx": hx,
        "energies": energies,
        "delta_E": energies - e0,
        "curves": curves,
        "output": output,
        "eigensolver_mode": solution["mode"],
        "orbits": orbits,
        "state_to_orbit_index": state_to_orbit_index,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Periodic k=0 Page curves for the Ising-limit chain.")
    parser.add_argument("--n-sites", type=int, default=17)
    parser.add_argument("--g2a", type=float, default=1.2)
    parser.add_argument("--J", type=float, default=None)
    parser.add_argument("--hx", type=float, default=None)
    parser.add_argument("--targets-delta", type=float, nargs="*", default=None)
    parser.add_argument("--max-dense-dim", type=int, default=3000)
    parser.add_argument("--output", type=str, default="page_curve_k0.png")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--log-base", choices=["e", "2"], default="e")
    parser.add_argument("--k-near", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_base = math.e if args.log_base == "e" else 2.0
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
    )

    print(f"N: {data['n_sites']}")
    print(f"Full Hilbert dimension: {data['full_dim']}")
    print(f"k=0 Hilbert dimension: {data['k0_dim']}")
    print(f"Couplings: J = {data['J']:.12g}, hx = {data['hx']:.12g}, g2a = {data['g2a']:.12g}")
    print(f"Eigensolver mode: {data['eigensolver_mode']}")
    print("Selected energies:")
    for energy, delta_e in zip(data["energies"], data["delta_E"]):
        print(f"  E = {energy:.12g}, Delta E = {delta_e:.12g}")
    print(f"Output filename: {data['output']}")


if __name__ == "__main__":
    main()
