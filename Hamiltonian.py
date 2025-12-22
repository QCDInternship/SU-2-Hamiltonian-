"""
2-spin toy Ising model, using the same structure as GaugeToIsingMapper.

Hamiltonian (N = 2, periodic, J = hx = 1):

    H = 2 Z0 Z1 - 2 (Z0 + Z1) + sqrt(2) (Z1 X0 + Z0 X1)

in the computational  basis
    |00>, |01>, |10>, |11>.

Mthods:
  1. Build the Hamiltonian matrix.
  2. Diagonalize it (get eigenvalues/eigenvectors).
  3. For each eigenvector, compute the entanglement entropy
     between qubit 0 (subsystem A) and qubit 1 (subsystem B).
"""

import numpy as np
from itertools import product
from typing import Callable, Dict, List, Tuple, Optional
import math

# Pauli matrices and Pauli-string builder

PAULI_SINGLE = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_string_op(label: str) -> np.ndarray:
    """
    Build the N-site Pauli operator corresponding to a string like 'ZX', 'IZ', etc.

    label: string of length N over {'I','X','Y','Z'}
    returns: 2^N x 2^N numpy array
    """
    op = PAULI_SINGLE[label[0]]
    for c in label[1:]:
        op = np.kron(op, PAULI_SINGLE[c])
    return op


# Matrix element for the Ising Hamiltonian 

def ising_matrix_element(phi_a: np.ndarray,
                         phi_b: np.ndarray,
                         J: float,
                         hx: float) -> complex:
    """
    Matrix element <phi_a | H | phi_b> for the Ising Hamiltonian

        H = sum_i [ J Z_i Z_{i+1} - 2J Z_i + hx * nu_i X_i ],
        nu_i = (Z_{i-1} + Z_{i+1}) / sqrt(2),

    with periodic boundary conditions.

    phi_a, phi_b : arrays of length N with entries 0 or 1
                   (0 -> |0>, 1 -> |1>)
    J, hx        : couplings
    """
    N = len(phi_a)

    # Map bits to σ^z eigenvalues: 0 -> +1, 1 -> -1
    s_a = 1 - 2 * phi_a
    s_b = 1 - 2 * phi_b

    # 1) Diagonal part: only if the configurations are identical
    if np.array_equal(phi_a, phi_b):
        diag = 0.0
        for i in range(N):
            ip1 = (i + 1) % N      # i+1 with periodic wrap
            diag += J * s_a[i] * s_a[ip1] - 2.0 * J * s_a[i]
        return diag

    # 2) Off-diagonal part: must differ by exactly one spin flip
    diff_sites = np.nonzero(s_a != s_b)[0]
    if len(diff_sites) != 1:
        return 0.0

    i = diff_sites[0]
    # Ensure it's a genuine flip: s_b[i] = - s_a[i]
    if s_b[i] != -s_a[i]:
        return 0.0

    # Now the hx * nu_i * X_i term:
    # nu_i = (1/√2) (Z_{i-1} + Z_{i+1})
    im1 = (i - 1) % N
    ip1 = (i + 1) % N
    nu_eig = (s_a[im1] + s_a[ip1]) / math.sqrt(2.0)

    # <s_a | nu_i X_i | s_b> = nu_eig * 1
    return hx * nu_eig


# Mapper class, simplified to the 2-spin toy model

class GaugeToIsingMapper:
    """
    2-spin toy Ising model version of the original mapper.

    - "Gauge basis" = all spin configurations in {0,1}^2.
    - "Ising basis" = same states, but we also keep ±1 version if desired.
    - Hamiltonian is the Ising Hamiltonian defined above.
    """

    def __init__(self, n_sites: int = 2):
        if n_sites != 2:
            raise ValueError("This toy model is hard-coded for n_sites = 2.")
        self.n = n_sites

        # List of basis configurations, each is an int array of shape (n,)
        self.gauge_basis: List[np.ndarray] = []
        self.ising_basis: List[np.ndarray] = []

        # Map gauge basis index -> ising basis index (here it's 1–1)
        self.gauge_to_ising: Dict[int, int] = {}

        # Hamiltonians
        self.H_gauge: Optional[np.ndarray] = None
        self.H_ising: Optional[np.ndarray] = None

        # Pauli expansion coefficients: label -> complex coefficient
        self.pauli_coeffs: Dict[str, complex] = {}

        # Spectrum
        self.evals: Optional[np.ndarray] = None
        self.evecs: Optional[np.ndarray] = None

    #  Basis construction 

    def build_gauge_basis(self) -> None:
        """
        Enumerate all basis states with spins in {0, 1} on each of the n sites.

        We no longer impose a Gauss-law constraint; this is a pure spin model.
        """
        self.gauge_basis.clear()
        for bits in product((0, 1), repeat=self.n):
            cfg = np.array(bits, dtype=int)
            self.gauge_basis.append(cfg)
        if not self.gauge_basis:
            raise RuntimeError("No basis states found!")

    def map_gauge_to_ising(self) -> None:
        """
        For each basis state (0/1), define an Ising spin configuration (±1):

            spin_i = +1  if bit_i = 0 (σ^z = +1)
            spin_i = -1  if bit_i = 1 (σ^z = -1)
        """
        if not self.gauge_basis:
            raise RuntimeError("Call build_gauge_basis() first.")

        self.ising_basis = []
        self.gauge_to_ising = {}

        for g_idx, cfg in enumerate(self.gauge_basis):
            # Map 0 -> +1, 1 -> -1
            ising_cfg = 1 - 2 * cfg
            self.ising_basis.append(ising_cfg)
            # Simple 1-to-1 mapping by index
            self.gauge_to_ising[g_idx] = g_idx

    #  Hamiltonian construction 

    def build_gauge_hamiltonian(
        self,
        matrix_element_fn: Callable[[np.ndarray, np.ndarray], complex],
    ) -> None:
        """
        Build the full Hamiltonian matrix H_gauge in the basis using a
        user given function for matrix elements.

        matrix_element_fn(phi_a, phi_b):
            - phi_a and phi_b are configurations (0/1 array)
            - should return <phi_a|H|phi_b> as a complex number
        """
        if not self.gauge_basis:
            raise RuntimeError("Call build_gauge_basis() first.")

        dim = len(self.gauge_basis)
        H = np.zeros((dim, dim), dtype=complex)

        for a, phi_a in enumerate(self.gauge_basis):
            for b, phi_b in enumerate(self.gauge_basis):
                H[a, b] = matrix_element_fn(phi_a, phi_b)

        # Hermitize to clean up numerical noise, if any
        H = 0.5 * (H + H.conj().T)
        self.H_gauge = H

    def build_ising_hamiltonian(self) -> None:
        """
        Reorder the gauge Hamiltonian into the Ising basis.

        Because our mapping is 1-to-1, this is just a permutation
        of rows and columns (here actually identity).
        """
        if self.H_gauge is None:
            raise RuntimeError("Call build_gauge_hamiltonian() first.")
        if not self.gauge_to_ising:
            raise RuntimeError("Call map_gauge_to_ising() first.")

        dim = len(self.gauge_basis)
        H_ising = np.zeros_like(self.H_gauge)

        for g_i in range(dim):
            i = self.gauge_to_ising[g_i]
            for g_j in range(dim):
                j = self.gauge_to_ising[g_j]
                H_ising[i, j] = self.H_gauge[g_i, g_j]

        self.H_ising = H_ising

    def diagonalize_ising_hamiltonian(self) -> None:
        """
        Diagonalize H_ising and store eigenvalues and eigenvectors.

        After calling this, you get:
            self.evals : array of shape (dim,)      (energies)
            self.evecs : array of shape (dim, dim)  (eigenvectors)
        Columns of self.evecs are eigenvectors in the |00>,|01>,|10>,|11> basis.
        """
        if self.H_ising is None:
            raise RuntimeError("Call build_ising_hamiltonian() first.")

        evals, evecs = np.linalg.eigh(self.H_ising)
        self.evals = evals
        self.evecs = evecs

    #  Entanglement entropy 

    def entanglement_entropy_of_eigenstate(
        self,
        eigen_index: int,
        log_base: float = 2.0,
    ) -> float:
        """
        Compute the entanglement entropy S(A) for eigenstate #eigen_index,
        where subsystem A = qubit 0 and subsystem B = qubit 1.

        For 2 spins:
          - eigenvector psi is length 4 (basis |00>,|01>,|10>,|11>)
          - we reshape psi into a 2x2 matrix Psi:
                rows   = state of A (0 or 1)
                cols   = state of B (0 or 1)
          - reduced density matrix of A is:
                rho_A = Psi @ Psi^\dagger
          - eigenvalues lambda_i of rho_A give:
                S_A = - sum_i lambda_i log_base(lambda_i)

        Returns
        -------
        S_A : float
            Entanglement entropy of A (in bits if log_base=2).
        """
        if self.evecs is None or self.evals is None:
            raise RuntimeError("Call diagonalize_ising_hamiltonian() first.")

        dim_expected = 2 ** self.n
        if self.evecs.shape != (dim_expected, dim_expected):
            raise RuntimeError(
                f"Unexpected eigenvector shape {self.evecs.shape}, expected {(dim_expected, dim_expected)}"
            )

        if eigen_index < 0 or eigen_index >= dim_expected:
            raise IndexError(f"eigen_index must be 0..{dim_expected-1}")

        # Take the eigenvector (column) corresponding to eigen_index
        psi = self.evecs[:, eigen_index]

        # Ensure normalization (should already be normalized, but we’re safe)
        norm = np.linalg.norm(psi)
        if norm == 0:
            raise RuntimeError("Eigenvector has zero norm, something went wrong.")
        psi = psi / norm

        # For 2 spins, n = 2:
        # - subsystem A = first qubit → dimension dA = 2
        # - subsystem B = second qubit → dimension dB = 2
        dA = 2
        dB = 2

        # Reshape into (dA, dB) matrix:
        #   Psi[0,0] = amplitude of |0_A 0_B> = |00>
        #   Psi[0,1] = amplitude of |0_A 1_B> = |01>
        #   Psi[1,0] = amplitude of |1_A 0_B> = |10>
        #   Psi[1,1] = amplitude of |1_A 1_B> = |11>
        Psi = psi.reshape(dA, dB)

        # Reduced density matrix of A: rho_A = Psi Psi^\dagger
        rho_A = Psi @ Psi.conj().T  # 2x2 matrix

        # Eigenvalues of rho_A
        lambdas = np.linalg.eigvalsh(rho_A)

        # Remove tiny numerical negatives / zeros
        eps = 1e-14
        lambdas = np.clip(lambdas, 0.0, 1.0)
        lambdas = lambdas[lambdas > eps]

        if len(lambdas) == 0:
            return 0.0

        # Compute entropy S
        logs = np.log(lambdas)
        if log_base != np.e:
            logs = logs / np.log(log_base)

        S = -np.sum(lambdas * logs)
        return float(S)

    #  Pauli decomposition 

    def decompose_in_pauli_basis(
        self,
        use_operators: Tuple[str, ...] = ("I", "X", "Z"),
        cutoff: float = 1e-10,
    ) -> None:
        """
        Expand H_ising in the operator basis tensor products of Pauli matrices.

        use_operators:
            which single-site ops to use; default ('I','X','Z') is
            enough for Ising-type Hamiltonians.

        cutoff:
            ignore coefficients with |c| < cutoff.
        """
        if self.H_ising is None:
            raise RuntimeError("Call build_ising_hamiltonian() first.")

        dim_expected = 2 ** self.n
        if self.H_ising.shape != (dim_expected, dim_expected):
            raise ValueError(
                f"H_ising has shape {self.H_ising.shape}, expected {(dim_expected, dim_expected)}"
            )

        self.pauli_coeffs.clear()
        # Normalization: for the n-qubit Pauli basis 
        norm_factor = 2.0 ** self.n

        # Iterate over all Pauli strings of length n
        for label_tuple in product(use_operators, repeat=self.n):
            label = "".join(label_tuple)
            P = pauli_string_op(label)
            coeff = np.trace(self.H_ising @ P) / norm_factor
            if abs(coeff) > cutoff:
                self.pauli_coeffs[label] = coeff

    def print_hamiltonian(self) -> None:
        """
        Print H_ising ≈ Σ c_label P_label in a human-readable way,
        using sigma^x_i, sigma^z_i notation.
        """
        if not self.pauli_coeffs:
            print("No Pauli coefficients stored; call decompose_in_pauli_basis() first.")
            return

        terms = []
        for label, c in sorted(self.pauli_coeffs.items(), key=lambda x: (len(x[0]), x[0])):
            # Build a string like "σ^Z_0 σ^Z_1"
            op_str_parts = []
            for i, ch in enumerate(label):
                if ch == "I":
                    continue
                op_str_parts.append(f"σ^{ch}_{i}")
            if not op_str_parts:
                op_str = "I"
            else:
                op_str = " ".join(op_str_parts)

            terms.append(f"{c.real:+.6g}{c.imag:+.6g}j * {op_str}")

        print("H_Ising ≈")
        for t in terms:
            print("  ", t)


# Example usage: build the 2-spin Hamiltonian and compute entanglement

if __name__ == "__main__":
    mapper = GaugeToIsingMapper(n_sites=2)

    # Step 1: build basis
    mapper.build_gauge_basis()
    print("Gauge/Ising basis states (0/1 encoding):")
    for idx, cfg in enumerate(mapper.gauge_basis):
        print(f"  {idx}: {cfg}")

    # Step 2: map to Ising ±1 basis (not strictly needed, but keeps structure)
    mapper.map_gauge_to_ising()

    # Step 3: build Hamiltonian with J = hx = 1
    J = 1.0
    hx = 1.0
    mapper.build_gauge_hamiltonian(
        lambda a, b: ising_matrix_element(a, b, J=J, hx=hx)
    )

    # Step 4: copy into Ising basis (identity permutation here)
    mapper.build_ising_hamiltonian()

    print("\nH_ising matrix in computational basis |00>,|01>,|10>,|11>:")
    print(np.real_if_close(mapper.H_ising))

    # Step 5: diagonalize H_ising (Step 1 for entanglement)
    mapper.diagonalize_ising_hamiltonian()
    print("\nEigenvalues of H_ising (energies):")
    print(np.real_if_close(mapper.evals))

    print("\nEigenvectors (columns) in basis |00>,|01>,|10>,|11>:")
    print(np.real_if_close(mapper.evecs))

    # Step 6: compute entanglement entropy for each eigenstate
    print("\nEntanglement entropy S(A) with A = qubit 0, B = qubit 1 (in bits):")
    for i, E in enumerate(mapper.evals):
        S = mapper.entanglement_entropy_of_eigenstate(i, log_base=2.0)
        print(f"  Eigenstate {i}: E = {E:.6f}, S_A = {S:.6f} bits")

    # decompose into Pauli strings
    mapper.decompose_in_pauli_basis(use_operators=("I", "X", "Z"), cutoff=1e-10)
    print("\nPauli-string decomposition:")
    mapper.print_hamiltonian()
