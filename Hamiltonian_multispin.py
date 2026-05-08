"""
Multi-spin Ising toy model, keeping the same overall structure as the original
GaugeToIsingMapper demo but removing the hard-coded N = 2 restriction.

Hamiltonian:

    H = sum_i [ J Z_i Z_{i+1} - 2 J Z_i + h_x * nu_i X_i ]
    nu_i = (Z_{i-1} + Z_{i+1}) / sqrt(2)

with periodic boundary conditions by default.

This script:
  1. enumerates the full computational basis {0,1}^N,
  2. builds the Hamiltonian matrix from matrix elements,
  3. diagonalizes it,
  4. computes bipartite entanglement entropy for arbitrary subsystems,
  5. optionally prints a contiguous-cut entanglement profile,
  6. optionally expands H in a Pauli-string basis.

Examples
--------
Run a 4-spin system and compute half-chain entanglement for each eigenstate:
    python Hamiltonian_multispin.py --n-sites 4

Run a 5-spin system and print the entropy for the first 2 qubits vs the rest:
    python Hamiltonian_multispin.py --n-sites 5 --subsystem 0 1

Show Page-curve-like entanglement profile S(L_A) for each eigenstate:
    python Hamiltonian_multispin.py --n-sites 4 --show-profile
"""

from __future__ import annotations

import argparse
import math
from itertools import product
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


PAULI_SINGLE = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_string_op(label: str) -> np.ndarray:
    """Build the N-qubit operator associated with a Pauli label like 'ZXI'."""
    if not label:
        raise ValueError("Pauli label must be non-empty.")

    # Start with the first one-qubit operator, then grow the full operator
    # one site at a time with Kronecker products.
    op = PAULI_SINGLE[label[0]]
    for char in label[1:]:
        op = np.kron(op, PAULI_SINGLE[char])
    return op


def ising_matrix_element(
    phi_a: np.ndarray,
    phi_b: np.ndarray,
    J: float,
    hx: float,
    periodic: bool = True,
) -> complex:
    """
    Compute <phi_a | H | phi_b> for the Ising Hamiltonian

        H = sum_i [ J Z_i Z_{i+1} - 2J Z_i + hx * nu_i X_i ],
        nu_i = (Z_{i-1} + Z_{i+1}) / sqrt(2),

    using periodic boundary conditions by default.

    Parameters
    ----------
    phi_a, phi_b
        Bitstrings of length N with entries in {0,1}.
    J, hx
        Couplings.
    periodic
        If True, use periodic boundary conditions. If False, use open boundaries.
    """
    N = len(phi_a)
    if len(phi_b) != N:
        raise ValueError("phi_a and phi_b must have the same length.")

    # In the computational basis, each bit is also a Z eigenstate.
    # Using spin values keeps the diagonal terms close to the formula.
    s_a = 1 - 2 * phi_a
    s_b = 1 - 2 * phi_b

    # The ZZ interaction and the linear Z field are diagonal, so they only
    # contribute when bra and ket are the same basis state.
    if np.array_equal(phi_a, phi_b):
        diag = 0.0
        max_i = N if periodic else N - 1
        for i in range(max_i):
            # With periodic boundaries, the final site couples back to site 0.
            # With open boundaries, the loop stops before that link.
            ip1 = (i + 1) % N if periodic else (i + 1)
            diag += J * s_a[i] * s_a[ip1]
        diag += -2.0 * J * float(np.sum(s_a))
        return diag

    # The X_i term flips exactly one spin. If two states differ at zero sites
    # or more than one site, this matrix element must vanish.
    diff_sites = np.nonzero(s_a != s_b)[0]
    if len(diff_sites) != 1:
        return 0.0

    i = int(diff_sites[0])
    # This check is redundant for ordinary 0/1 spin states, but it makes the
    # expected single-site flip explicit.
    if s_b[i] != -s_a[i]:
        return 0.0

    if periodic:
        im1 = (i - 1) % N
        ip1 = (i + 1) % N
        # The local flip strength depends on the neighboring Z eigenvalues.
        nu_eig = (s_a[im1] + s_a[ip1]) / math.sqrt(2.0)
        return hx * nu_eig

    # At an open boundary, the missing neighbor is treated as zero contribution
    # rather than wrapping around to the other end of the chain.
    left = s_a[i - 1] if i - 1 >= 0 else 0.0
    right = s_a[i + 1] if i + 1 < N else 0.0
    nu_eig = (left + right) / math.sqrt(2.0)
    return hx * nu_eig


class GaugeToIsingMapper:
    """
    Multi-spin toy Ising model version of the original mapper.

    - "Gauge basis" = all spin configurations in {0,1}^N.
    - "Ising basis" = the same states, together with their ±1 spin values.
    - Hamiltonian = the Ising Hamiltonian defined in ising_matrix_element().
    """

    def __init__(self, n_sites: int, periodic: bool = True):
        if n_sites < 2:
            raise ValueError("n_sites must be at least 2.")

        self.n = n_sites
        self.periodic = periodic

        self.gauge_basis: List[np.ndarray] = []
        self.ising_basis: List[np.ndarray] = []
        self.gauge_to_ising: Dict[int, int] = {}

        self.H_gauge: Optional[np.ndarray] = None
        self.H_ising: Optional[np.ndarray] = None

        self.pauli_coeffs: Dict[str, complex] = {}

        self.evals: Optional[np.ndarray] = None
        self.evecs: Optional[np.ndarray] = None

    def build_gauge_basis(self) -> None:
        """Enumerate the full computational basis {0,1}^N."""
        self.gauge_basis.clear()
        # product((0, 1), repeat=N) gives the standard binary ordering of all
        # basis states, which matches the reshape convention used later.
        for bits in product((0, 1), repeat=self.n):
            self.gauge_basis.append(np.array(bits, dtype=int))

        if not self.gauge_basis:
            raise RuntimeError("No basis states found.")

    def map_gauge_to_ising(self) -> None:
        """Map bitstrings 0/1 to Ising spins +1/-1."""
        if not self.gauge_basis:
            raise RuntimeError("Call build_gauge_basis() first.")

        self.ising_basis = []
        self.gauge_to_ising = {}

        for g_idx, cfg in enumerate(self.gauge_basis):
            # The mapping is currently one-to-one. Keeping the explicit map
            # makes it easy to swap in a different ordering later.
            self.ising_basis.append(1 - 2 * cfg)
            self.gauge_to_ising[g_idx] = g_idx

    def build_gauge_hamiltonian(
        self,
        matrix_element_fn: Callable[[np.ndarray, np.ndarray], complex],
    ) -> None:
        """Build the Hamiltonian in the computational basis."""
        if not self.gauge_basis:
            raise RuntimeError("Call build_gauge_basis() first.")

        dim = len(self.gauge_basis)
        H = np.zeros((dim, dim), dtype=complex)

        # Fill the full matrix from the supplied matrix element function.
        # For small toy systems this is clear and flexible enough.
        for a, phi_a in enumerate(self.gauge_basis):
            for b, phi_b in enumerate(self.gauge_basis):
                H[a, b] = matrix_element_fn(phi_a, phi_b)

        # Numerical roundoff or future custom matrix elements can introduce
        # tiny asymmetries, so we explicitly project back to a Hermitian matrix.
        self.H_gauge = 0.5 * (H + H.conj().T)

    def build_ising_hamiltonian(self) -> None:
        """Reorder the gauge Hamiltonian into the Ising basis."""
        if self.H_gauge is None:
            raise RuntimeError("Call build_gauge_hamiltonian() first.")
        if not self.gauge_to_ising:
            raise RuntimeError("Call map_gauge_to_ising() first.")

        dim = len(self.gauge_basis)
        H_ising = np.zeros_like(self.H_gauge)

        # This copy is trivial for the current one-to-one map, but the separate
        # step mirrors the original mapper structure.
        for g_i in range(dim):
            i = self.gauge_to_ising[g_i]
            for g_j in range(dim):
                j = self.gauge_to_ising[g_j]
                H_ising[i, j] = self.H_gauge[g_i, g_j]

        self.H_ising = H_ising

    def diagonalize_ising_hamiltonian(self) -> None:
        """Diagonalize H_ising and store eigenvalues/eigenvectors."""
        if self.H_ising is None:
            raise RuntimeError("Call build_ising_hamiltonian() first.")

        self.evals, self.evecs = np.linalg.eigh(self.H_ising)

    def get_eigenvector(self, eigen_index: int) -> np.ndarray:
        """Return normalized eigenvector number eigen_index."""
        if self.evecs is None or self.evals is None:
            raise RuntimeError("Call diagonalize_ising_hamiltonian() first.")

        dim_expected = 2 ** self.n
        if self.evecs.shape != (dim_expected, dim_expected):
            raise RuntimeError(
                f"Unexpected eigenvector shape {self.evecs.shape}, expected {(dim_expected, dim_expected)}"
            )
        if eigen_index < 0 or eigen_index >= dim_expected:
            raise IndexError(f"eigen_index must be between 0 and {dim_expected - 1}.")

        psi = self.evecs[:, eigen_index]
        norm = np.linalg.norm(psi)
        if norm == 0:
            raise RuntimeError("Encountered a zero-norm eigenvector.")
        # eigh already returns normalized columns, but normalizing here keeps
        # this method robust if the source of eigenvectors changes later.
        return psi / norm

    def entanglement_entropy_of_state(
        self,
        psi: np.ndarray,
        subsystem: Optional[Sequence[int]] = None,
        log_base: float = 2.0,
    ) -> float:
        """
        Compute the bipartite entanglement entropy S(A) for an arbitrary subsystem.

        Parameters
        ----------
        psi
            State vector of size 2^N.
        subsystem
            Sequence of qubit indices defining subsystem A.
            Default: first half of the chain.
        log_base
            Use 2.0 for bits, math.e for nats.
        """
        dim_expected = 2 ** self.n
        psi = np.asarray(psi, dtype=complex)
        if psi.shape != (dim_expected,):
            raise ValueError(f"psi must have shape {(dim_expected,)}, got {psi.shape}.")

        if subsystem is None:
            # By default, compare the first half of the chain with the rest.
            subsystem = tuple(range(self.n // 2))
        else:
            subsystem = tuple(int(i) for i in subsystem)

        if len(subsystem) == 0 or len(subsystem) == self.n:
            raise ValueError("Subsystem must contain between 1 and n-1 qubits.")
        if len(set(subsystem)) != len(subsystem):
            raise ValueError("Subsystem indices must be unique.")
        if min(subsystem) < 0 or max(subsystem) >= self.n:
            raise ValueError(f"Subsystem indices must lie in 0..{self.n - 1}.")

        complement = tuple(i for i in range(self.n) if i not in subsystem)

        psi = psi / np.linalg.norm(psi)
        # Reshape the flat vector into an N-index tensor, one index per qubit.
        psi_tensor = psi.reshape((2,) * self.n)
        # Move subsystem A to the front so the state can be viewed as a matrix
        # Psi[a, b] between subsystem A and its complement B.
        reordered = np.transpose(psi_tensor, axes=subsystem + complement)

        dA = 2 ** len(subsystem)
        dB = 2 ** len(complement)
        Psi = reordered.reshape(dA, dB)
        # Tracing out B gives rho_A. In matrix form, this is Psi Psi dagger.
        rho_A = Psi @ Psi.conj().T

        lambdas = np.linalg.eigvalsh(rho_A)
        # Small negative values can appear from floating point noise.
        lambdas = np.clip(lambdas, 0.0, 1.0)
        eps = 1e-14
        # Zero eigenvalues do not affect S, and removing them avoids log(0).
        lambdas = lambdas[lambdas > eps]
        if len(lambdas) == 0:
            return 0.0

        logs = np.log(lambdas)
        if log_base != math.e:
            logs = logs / math.log(log_base)

        return float(-np.sum(lambdas * logs))

    def entanglement_entropy_of_eigenstate(
        self,
        eigen_index: int,
        subsystem: Optional[Sequence[int]] = None,
        log_base: float = 2.0,
    ) -> float:
        """Convenience wrapper for the entropy of one eigenstate."""
        psi = self.get_eigenvector(eigen_index)
        return self.entanglement_entropy_of_state(psi, subsystem=subsystem, log_base=log_base)

    def entanglement_profile_of_eigenstate(
        self,
        eigen_index: int,
        log_base: float = 2.0,
    ) -> List[Tuple[int, float]]:
        """
        Compute S(L_A) for contiguous cuts A = {0,1,...,L_A-1}, L_A = 1..N-1.
        """
        psi = self.get_eigenvector(eigen_index)
        profile: List[Tuple[int, float]] = []
        for L_A in range(1, self.n):
            # Each profile point uses the first L_A sites as subsystem A.
            subsystem = tuple(range(L_A))
            entropy = self.entanglement_entropy_of_state(psi, subsystem=subsystem, log_base=log_base)
            profile.append((L_A, entropy))
        return profile

    def decompose_in_pauli_basis(
        self,
        use_operators: Tuple[str, ...] = ("I", "X", "Z"),
        cutoff: float = 1e-10,
    ) -> None:
        """Expand H_ising in the Pauli-string basis."""
        if self.H_ising is None:
            raise RuntimeError("Call build_ising_hamiltonian() first.")

        dim_expected = 2 ** self.n
        if self.H_ising.shape != (dim_expected, dim_expected):
            raise ValueError(
                f"H_ising has shape {self.H_ising.shape}, expected {(dim_expected, dim_expected)}"
            )

        self.pauli_coeffs.clear()
        # Pauli strings are orthogonal under the Hilbert-Schmidt inner product:
        # Tr(P_a P_b) = 2^N delta_ab. This gives the coefficient below.
        norm_factor = 2.0 ** self.n

        for label_tuple in product(use_operators, repeat=self.n):
            label = "".join(label_tuple)
            P = pauli_string_op(label)
            coeff = np.trace(self.H_ising @ P) / norm_factor
            # Skip coefficients that are numerically zero so the output stays
            # readable for larger systems.
            if abs(coeff) > cutoff:
                self.pauli_coeffs[label] = coeff

    def print_hamiltonian(self) -> None:
        """Print the Pauli decomposition in a readable form."""
        if not self.pauli_coeffs:
            print("No Pauli coefficients stored; call decompose_in_pauli_basis() first.")
            return

        terms = []
        for label, c in sorted(self.pauli_coeffs.items()):
            op_parts = []
            for i, ch in enumerate(label):
                if ch != "I":
                    op_parts.append(f"σ^{ch}_{i}")
            # If every factor is identity, print a plain identity term.
            op_str = " ".join(op_parts) if op_parts else "I"
            terms.append(f"{c.real:+.6g}{c.imag:+.6g}j * {op_str}")

        print("H_Ising ≈")
        for term in terms:
            print("  ", term)


def build_mapper(n_sites: int, J: float, hx: float, periodic: bool = True) -> GaugeToIsingMapper:
    """Helper to build and diagonalize the model in one call."""
    mapper = GaugeToIsingMapper(n_sites=n_sites, periodic=periodic)
    # Keep the setup steps separate so each method can still be tested or used
    # independently, while the CLI has a compact path for the common case.
    mapper.build_gauge_basis()
    mapper.map_gauge_to_ising()
    mapper.build_gauge_hamiltonian(
        lambda a, b: ising_matrix_element(a, b, J=J, hx=hx, periodic=periodic)
    )
    mapper.build_ising_hamiltonian()
    mapper.diagonalize_ising_hamiltonian()
    return mapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-spin toy Ising Hamiltonian and entanglement tool.")
    parser.add_argument("--n-sites", type=int, default=4, help="Number of spins/qubits.")
    parser.add_argument("--J", type=float, default=1.0, help="Diagonal coupling J.")
    parser.add_argument("--hx", type=float, default=1.0, help="Off-diagonal coupling hx.")
    parser.add_argument(
        "--open-boundary",
        action="store_true",
        help="Use open instead of periodic boundary conditions.",
    )
    parser.add_argument(
        "--subsystem",
        type=int,
        nargs="*",
        default=None,
        help="Subsystem qubit indices, e.g. --subsystem 0 1.",
    )
    parser.add_argument(
        "--show-profile",
        action="store_true",
        help="Print S(L_A) for contiguous cuts of each eigenstate.",
    )
    parser.add_argument(
        "--show-matrix",
        action="store_true",
        help="Print the Hamiltonian matrix explicitly.",
    )
    parser.add_argument(
        "--show-evecs",
        action="store_true",
        help="Print the full eigenvector matrix explicitly.",
    )
    parser.add_argument(
        "--decompose-pauli",
        action="store_true",
        help="Compute and print the Pauli-string decomposition.",
    )
    parser.add_argument(
        "--pauli-cutoff",
        type=float,
        default=1e-10,
        help="Cutoff used in the Pauli decomposition.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    periodic = not args.open_boundary

    # Build everything up front, then the rest of the script only formats the
    # requested pieces for display.
    mapper = build_mapper(
        n_sites=args.n_sites,
        J=args.J,
        hx=args.hx,
        periodic=periodic,
    )

    print(f"Number of sites: {mapper.n}")
    print(f"Boundary conditions: {'periodic' if mapper.periodic else 'open'}")
    print(f"Hilbert-space dimension: {2 ** mapper.n}")

    if mapper.n <= 5:
        print("\nGauge/Ising basis states (0/1 encoding):")
        for idx, cfg in enumerate(mapper.gauge_basis):
            print(f"  {idx}: {cfg}")

    if args.show_matrix or mapper.n <= 4:
        print("\nH_ising matrix in computational basis:")
        print(np.real_if_close(mapper.H_ising))

    print("\nEigenvalues of H_ising (energies):")
    print(np.real_if_close(mapper.evals))

    if args.show_evecs and mapper.evecs is not None:
        print("\nEigenvectors (columns) in computational basis:")
        print(np.real_if_close(mapper.evecs))

    subsystem = tuple(args.subsystem) if args.subsystem is not None and len(args.subsystem) > 0 else None
    if subsystem is None:
        # Match the default used by entanglement_entropy_of_state().
        default_size = mapper.n // 2
        subsystem_str = f"first {default_size} qubit(s): {tuple(range(default_size))}"
    else:
        subsystem_str = str(subsystem)

    print(f"\nEntanglement entropy S(A) in bits for subsystem A = {subsystem_str}:")
    for i, E in enumerate(mapper.evals):
        S = mapper.entanglement_entropy_of_eigenstate(i, subsystem=subsystem, log_base=2.0)
        print(f"  Eigenstate {i}: E = {E:.6f}, S_A = {S:.6f} bits")

    if args.show_profile:
        print("\nContiguous-cut entanglement profiles S(L_A):")
        for i, E in enumerate(mapper.evals):
            profile = mapper.entanglement_profile_of_eigenstate(i, log_base=2.0)
            profile_str = ", ".join(f"L_A={L}: {S:.6f}" for L, S in profile)
            print(f"  Eigenstate {i} (E = {E:.6f}): {profile_str}")

    if args.decompose_pauli:
        mapper.decompose_in_pauli_basis(use_operators=("I", "X", "Z"), cutoff=args.pauli_cutoff)
        print("\nPauli-string decomposition:")
        mapper.print_hamiltonian()
