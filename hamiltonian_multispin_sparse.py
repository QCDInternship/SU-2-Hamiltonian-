"""Sparse version of the effective multi-spin Ising-limit Hamiltonian.

This implements the same j_max=1/2 effective spin-chain model as
``Hamiltonian_multispin.py``.  It is not a general SU(2) electric-basis
lattice-gauge-theory Hamiltonian.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from ising_limit_model import (
    HamiltonianMode,
    IsingLimitParameters,
    build_linear_operator,
    build_sparse_hamiltonian as build_canonical_sparse_hamiltonian,
)


DEFAULT_BENCHMARK_SIZES = (4, 6, 8, 10, 12)
# Keep benchmark dense allocations modest.  Larger dense entry counts are
# reported analytically, which is enough to expose the O(4^N) scaling.
MAX_BENCHMARK_DENSE_ENTRIES = 1_000_000


def _z_value(state: int, site: int, n_spins: int) -> int:
    """Return the Z eigenvalue at a physical site (site 0 is the MSB)."""
    bit = (state >> (n_spins - 1 - site)) & 1
    return 1 - 2 * bit


def build_sparse_hamiltonian(
    n_spins: int,
    J: float = 1.0,
    hx: float = 1.0,
    periodic: bool = True,
    dtype=np.float64,
    mode: str | HamiltonianMode = HamiltonianMode.TOY,
) -> sp.csr_matrix:
    """Build the effective Ising Hamiltonian directly in CSR format.

    Integer states use ordinary binary ordering: site 0 is the most-significant
    bit, and binary 0/1 corresponds to Z=+1/-1.  For open boundaries, a
    missing neighbor in nu_i contributes zero, matching the dense script.
    """
    if not isinstance(n_spins, (int, np.integer)) or isinstance(n_spins, bool):
        raise TypeError("n_spins must be an integer.")
    if n_spins < 2:
        raise ValueError("n_spins must be at least 2, matching Hamiltonian_multispin.py.")

    matrix_dtype = np.dtype(dtype)
    if matrix_dtype.kind not in "fc":
        raise TypeError("dtype must be a floating-point or complex dtype.")

    params = IsingLimitParameters(
        n_sites=n_spins,
        J=J,
        hx=hx,
        periodic=periodic,
        mode=HamiltonianMode(mode),
    )
    return build_canonical_sparse_hamiltonian(params).astype(matrix_dtype)


def diagonalize_sparse(
    H: sp.spmatrix,
    k: int = 6,
    which: str = "SA",
    tol: float = 0.0,
    maxiter: int | None = None,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted eigenvalues and eigenvectors, using dense fallback as needed."""
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square matrix.")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")
    if which not in {"SA", "LA"}:
        raise ValueError("which must be 'SA' or 'LA'.")
    v0 = _initial_vector(H.shape[0], seed)

    dim = H.shape[0]
    if k >= dim:
        eigenvalues, eigenvectors = np.linalg.eigh(H.toarray())
    else:
        eigenvalues, eigenvectors = eigsh(
            H, k=k, which=which, tol=tol, maxiter=maxiter, v0=v0
        )

    order = np.argsort(eigenvalues)
    return np.real_if_close(eigenvalues[order]), eigenvectors[:, order]


def _initial_vector(dim: int, seed: int | None) -> np.ndarray | None:
    if seed is None:
        return None
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return None
    return vector / norm


def _validate_lanczos_args(
    dim: int,
    k: int,
    which: str,
    tol: float,
    maxiter: int | None,
) -> None:
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")
    if k >= dim:
        raise ValueError("matrix-free eigsh requires k < Hilbert-space dimension.")
    if which not in {"SA", "LA"}:
        raise ValueError("which must be 'SA' or 'LA'.")
    if tol < 0:
        raise ValueError("tol must be non-negative.")
    if maxiter is not None and maxiter < 1:
        raise ValueError("maxiter must be positive when supplied.")


def residual_norms(
    operator,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Return ||H v_i - E_i v_i|| for each eigenpair."""
    return np.array(
        [
            np.linalg.norm(operator @ eigenvectors[:, index] - eigenvalue * eigenvectors[:, index])
            for index, eigenvalue in enumerate(eigenvalues)
        ],
        dtype=float,
    )


def diagonalize_matrix_free(
    params: IsingLimitParameters,
    k: int = 6,
    which: str = "SA",
    tol: float = 0.0,
    maxiter: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute extremal eigenpairs using a matrix-free LinearOperator."""
    dim = 1 << params.n_sites
    _validate_lanczos_args(dim, k, which, tol, maxiter)
    operator = build_linear_operator(params)
    start = perf_counter()
    eigenvalues, eigenvectors = eigsh(
        operator,
        k=k,
        which=which,
        tol=tol,
        maxiter=maxiter,
        v0=_initial_vector(dim, seed),
    )
    runtime = perf_counter() - start
    order = np.argsort(eigenvalues)
    eigenvalues = np.real_if_close(eigenvalues[order])
    eigenvectors = eigenvectors[:, order]
    residuals = residual_norms(operator, eigenvalues, eigenvectors)
    return eigenvalues, eigenvectors, residuals, runtime


def print_sparse_summary(H: sp.spmatrix) -> None:
    """Print SciPy's sparse representation, including all nonzero entries."""
    print(H)


def print_sparse_entries(
    H: sp.spmatrix, max_entries: int | None = None
) -> None:
    """Print nonzero matrix entries as a row/column/value table."""
    coo = H.tocoo()
    print("row   col   value")
    entries = zip(coo.row, coo.col, coo.data)
    for index, (row, col, value) in enumerate(entries):
        if max_entries is not None and index >= max_entries:
            print(f"... {coo.nnz - max_entries} more entries")
            break
        print(f"{row:<5} {col:<5} {value}")


def print_dense_matrix(H: sp.spmatrix, max_print_size: int = 16) -> None:
    """Print the full dense matrix when its dimension is within the limit."""
    dimension = H.shape[0]
    if dimension > max_print_size:
        print(
            f"Dense matrix has shape {H.shape}, which is larger than "
            f"max_print_size={max_print_size}."
        )
        print(
            f"Use --max-print-size {dimension} if you really want to print it."
        )
        return

    np.set_printoptions(precision=3, suppress=True, linewidth=160, threshold=np.inf)
    print(H.toarray())


def compare_dense_sparse(
    n_spins: int,
    J: float = 1.0,
    hx: float = 1.0,
    periodic: bool = True,
    mode: str | HamiltonianMode = HamiltonianMode.TOY,
) -> bool:
    """Compare this construction with the existing dense implementation."""
    from Hamiltonian_multispin import build_mapper

    mapper = build_mapper(n_sites=n_spins, J=J, hx=hx, periodic=periodic, mode=mode)
    if mapper.H_ising is None or mapper.evals is None:
        raise RuntimeError("Dense Hamiltonian construction did not complete.")

    sparse = build_sparse_hamiltonian(
        n_spins, J=J, hx=hx, periodic=periodic, mode=mode
    )
    dense = np.real_if_close(mapper.H_ising)
    matrix_difference = float(np.max(np.abs(dense - sparse.toarray())))
    dense_eigenvalues = np.real_if_close(mapper.evals)
    sparse_eigenvalues, _ = diagonalize_sparse(sparse, k=sparse.shape[0])
    eigenvalue_difference = float(
        np.max(np.abs(dense_eigenvalues - sparse_eigenvalues))
    )
    passed = bool(
        np.allclose(dense, sparse.toarray(), rtol=1e-12, atol=1e-12)
        and np.allclose(
            dense_eigenvalues, sparse_eigenvalues, rtol=1e-10, atol=1e-10
        )
    )

    print(f"Maximum absolute matrix difference: {matrix_difference:.6e}")
    print(f"Maximum absolute eigenvalue difference: {eigenvalue_difference:.6e}")
    print(f"Dense eigenvalues: {dense_eigenvalues}")
    print(f"Sparse eigenvalues: {sparse_eigenvalues}")
    print(f"Dense/sparse comparison: {'PASSED' if passed else 'FAILED'}")
    return passed


def print_page_curves(
    eigenvalues: np.ndarray, eigenvectors: np.ndarray, n_spins: int
) -> None:
    """Print contiguous-cut entropies for eigenvectors returned by ``eigsh``."""
    from Hamiltonian_multispin import GaugeToIsingMapper

    mapper = GaugeToIsingMapper(n_sites=n_spins)
    print("Page-curve-style contiguous-cut entropies S(L_A) in bits:")
    for eigen_index, eigenvalue in enumerate(eigenvalues):
        psi = eigenvectors[:, eigen_index]
        values = []
        for cut in range(1, n_spins):
            entropy = mapper.entanglement_entropy_of_state(
                psi, subsystem=tuple(range(cut)), log_base=2.0
            )
            values.append(f"L_A={cut}: {entropy:.6f}")
        print(f"  Eigenstate {eigen_index} (E={eigenvalue:.8g}): " + ", ".join(values))


def run_benchmark(
    sizes: tuple[int, ...] = DEFAULT_BENCHMARK_SIZES,
    J: float = 1.0,
    hx: float = 1.0,
    periodic: bool = True,
    k: int = 4,
    mode: str | HamiltonianMode = HamiltonianMode.TOY,
) -> None:
    """Print sparse sizes and low energies across a range of spin counts."""
    print("N | D=2^N | dense_entries=D^2 | sparse_nnz | density | lowest_eigenvalues")
    for n_spins in sizes:
        H = build_sparse_hamiltonian(
            n_spins, J=J, hx=hx, periodic=periodic, mode=mode
        )
        dim = H.shape[0]
        dense_entries = dim * dim

        # Materialize only small dense matrices.  This is deliberately not used
        # for diagonalisation; it merely ensures the benchmark's dense-storage
        # comparison never triggers a large allocation.
        if dense_entries <= MAX_BENCHMARK_DENSE_ENTRIES:
            dense_nbytes = H.toarray().nbytes
            assert dense_nbytes == dense_entries * H.dtype.itemsize

        eigenvalues, _ = diagonalize_sparse(H, k=min(k, dim - 1), which="SA")
        eigenvalue_text = np.array2string(eigenvalues, precision=6, separator=",")
        print(
            f"{n_spins} | {dim} | {dense_entries} | {H.nnz} | "
            f"{H.nnz / dense_entries:.6e} | {eigenvalue_text}"
        )

    print("Dense storage scales as O(4^N); dense entry counts above "
          f"{MAX_BENCHMARK_DENSE_ENTRIES:,} were not allocated.")
    print("Sparse nonzeros are bounded by roughly (N+1)2^N = O(N 2^N).")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sparse diagonalisation of the effective multi-spin Ising model."
    )
    parser.add_argument("--n-spins", type=int, default=6, help="Number of spins.")
    parser.add_argument("--J", type=float, default=1.0, help="Diagonal coupling J.")
    parser.add_argument("--hx", type=float, default=1.0, help="Flip coupling hx.")
    parser.add_argument("--k-eigs", type=int, default=6, help="Number of eigenpairs.")
    parser.add_argument(
        "--storage",
        choices=("csr", "matrix-free"),
        default="csr",
        help="Hamiltonian storage/backend (default: csr).",
    )
    parser.add_argument(
        "--which",
        choices=("SA", "LA"),
        default="SA",
        help="Extremal eigenvalues for eigsh: SA lowest, LA highest.",
    )
    parser.add_argument("--tol", type=float, default=0.0, help="eigsh convergence tolerance.")
    parser.add_argument("--maxiter", type=int, default=None, help="Maximum eigsh iterations.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic eigsh v0.")
    parser.add_argument("--open-boundary", action="store_true")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in HamiltonianMode],
        default=HamiltonianMode.TOY.value,
        help="Hamiltonian convention to use (default: toy).",
    )
    parser.add_argument("--compare-dense", action="store_true")
    parser.add_argument("--show-nnz", action="store_true")
    parser.add_argument(
        "--print-sparse",
        action="store_true",
        help="Print the SciPy sparse matrix representation.",
    )
    parser.add_argument(
        "--print-dense",
        action="store_true",
        help="Print the full dense matrix when it is within --max-print-size.",
    )
    parser.add_argument(
        "--print-coo",
        action="store_true",
        help="Print nonzero entries as a COO row/column/value table.",
    )
    parser.add_argument(
        "--max-print-size",
        type=int,
        default=16,
        metavar="D",
        help="Maximum Hilbert-space dimension allowed for dense printing (default: 16).",
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--page-curve",
        action="store_true",
        help="Print contiguous-cut entropies for the computed sparse eigenvectors.",
    )
    parser.add_argument(
        "--save-eigs-csv", type=Path, metavar="PATH", help="Save eigenvalues as CSV."
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_print_size < 1:
        raise SystemExit("--max-print-size must be a positive integer.")
    periodic = not args.open_boundary
    mode = HamiltonianMode(args.mode)
    if mode is HamiltonianMode.PAPER and not periodic:
        raise SystemExit("error: --mode paper is currently defined only for periodic boundaries.")
    if args.benchmark:
        run_benchmark(J=args.J, hx=args.hx, periodic=periodic, mode=mode)
        return 0
    if args.storage == "matrix-free":
        if args.print_sparse or args.print_coo or args.print_dense or args.show_nnz:
            raise SystemExit(
                "error: matrix-free storage does not construct a matrix to print or count."
            )
        if args.compare_dense:
            raise SystemExit("error: --compare-dense requires --storage csr.")
        params = IsingLimitParameters(
            n_sites=args.n_spins,
            J=args.J,
            hx=args.hx,
            periodic=periodic,
            mode=mode,
        )
        dim = 1 << args.n_spins
        eigenvalues, eigenvectors, residuals, runtime = diagonalize_matrix_free(
            params,
            k=args.k_eigs,
            which=args.which,
            tol=args.tol,
            maxiter=args.maxiter,
            seed=args.seed,
        )

        print(f"Number of spins: {args.n_spins}")
        print(f"Boundary conditions: {'periodic' if periodic else 'open'}")
        print(f"Hamiltonian mode: {mode.value}")
        print("Storage backend: matrix-free")
        print(f"Hilbert-space dimension: D=2^{args.n_spins}={dim}")
        print(f"Lanczos which: {args.which}")
        print(f"Lanczos runtime seconds: {runtime:.6f}")
        label = "Lowest" if args.which == "SA" else "Highest"
        print(f"{label} eigenvalues ({len(eigenvalues)}): {eigenvalues}")
        print(f"Residual norms: {residuals}")

        if args.page_curve:
            print_page_curves(eigenvalues, eigenvectors, args.n_spins)
        if args.save_eigs_csv is not None:
            np.savetxt(
                args.save_eigs_csv,
                np.asarray(eigenvalues).reshape(-1, 1),
                delimiter=",",
                header="eigenvalue",
                comments="",
            )
            print(f"Saved eigenvalues to: {args.save_eigs_csv}")
        return 0

    H = build_sparse_hamiltonian(
        args.n_spins, J=args.J, hx=args.hx, periodic=periodic, mode=mode
    )
    start = perf_counter()
    eigenvalues, eigenvectors = diagonalize_sparse(
        H,
        k=args.k_eigs,
        which=args.which,
        tol=args.tol,
        maxiter=args.maxiter,
        seed=args.seed,
    )
    runtime = perf_counter() - start
    residuals = residual_norms(H, eigenvalues, eigenvectors)

    print(f"Number of spins: {args.n_spins}")
    print(f"Boundary conditions: {'periodic' if periodic else 'open'}")
    print(f"Hamiltonian mode: {mode.value}")
    print("Storage backend: csr")
    print(f"Hilbert-space dimension: D=2^{args.n_spins}={H.shape[0]}")
    print(f"Sparse number of nonzero entries: {H.nnz}")
    if args.show_nnz:
        total = H.shape[0] ** 2
        print(f"Sparse density: {H.nnz / total:.6e} ({H.nnz}/{total})")
        print(f"Structural upper bound (N+1)D: {(args.n_spins + 1) * H.shape[0]}")
    print(f"Lanczos which: {args.which}")
    print(f"Lanczos runtime seconds: {runtime:.6f}")
    label = "Lowest" if args.which == "SA" else "Highest"
    print(f"{label} eigenvalues ({len(eigenvalues)}): {eigenvalues}")
    print(f"Residual norms: {residuals}")

    if args.print_sparse:
        print("Sparse Hamiltonian:")
        print_sparse_summary(H)
    if args.print_coo:
        print("COO Hamiltonian entries:")
        print_sparse_entries(H)
    if args.print_dense:
        print("Dense Hamiltonian:")
        print_dense_matrix(H, max_print_size=args.max_print_size)

    if args.page_curve:
        print_page_curves(eigenvalues, eigenvectors, args.n_spins)

    if args.save_eigs_csv is not None:
        np.savetxt(
            args.save_eigs_csv,
            np.asarray(eigenvalues).reshape(-1, 1),
            delimiter=",",
            header="eigenvalue",
            comments="",
        )
        print(f"Saved eigenvalues to: {args.save_eigs_csv}")

    if args.compare_dense:
        passed = compare_dense_sparse(
            args.n_spins, J=args.J, hx=args.hx, periodic=periodic, mode=mode
        )
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
