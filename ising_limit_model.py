"""Canonical j_max=1/2 Ising-limit Hamiltonians.

This module keeps the repository's two existing Ising-limit Hamiltonians
separate:

* ``HamiltonianMode.TOY`` matches ``Hamiltonian_multispin.py`` and
  ``hamiltonian_multispin_sparse.py``.
* ``HamiltonianMode.PAPER`` matches the periodic ``mode="paper"`` convention
  in ``scar_state_search.py``.

Basis convention
----------------
Basis indices use ordinary binary ordering.  Site 0 is the leftmost,
most-significant bit.  Bit 0 has Z=+1 and bit 1 has Z=-1.

``page_curve_k0.py`` historically used helper functions where site indices
refer to least-significant-bit positions, while the dense and sparse Ising
scripts used site 0 as the most-significant bit.  This module uses the latter
convention consistently.  That site-label convention is a basis-ordering
choice and must not change spectra or physical results.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from itertools import product
import math
from numbers import Real
from types import MappingProxyType
from typing import Iterator, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator


class HamiltonianMode(str, Enum):
    TOY = "toy"
    PAPER = "paper"


@dataclass(frozen=True)
class IsingLimitParameters:
    n_sites: int
    J: float
    hx: float
    periodic: bool
    mode: HamiltonianMode

    def __post_init__(self) -> None:
        if isinstance(self.n_sites, bool) or not isinstance(self.n_sites, (int, np.integer)):
            raise TypeError("n_sites must be an integer.")
        if int(self.n_sites) < 2:
            raise ValueError("n_sites must be at least 2.")
        if int(self.n_sites) != self.n_sites:
            raise ValueError("n_sites must be an integer value.")
        object.__setattr__(self, "n_sites", int(self.n_sites))

        for name in ("J", "hx"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, float(value))

        if not isinstance(self.periodic, bool):
            raise TypeError("periodic must be a bool.")

        try:
            mode = HamiltonianMode(self.mode)
        except ValueError as error:
            raise ValueError("mode must be HamiltonianMode.TOY or HamiltonianMode.PAPER.") from error
        object.__setattr__(self, "mode", mode)

        if mode is HamiltonianMode.PAPER and not self.periodic:
            raise ValueError("PAPER mode is currently defined only for periodic boundaries.")


@dataclass(frozen=True)
class ExactDiagonalizationResult:
    """Results of dense exact diagonalization.

    The columns of ``eigenvectors`` are the normalized eigenvectors associated
    with ``eigenvalues``.  ``hamiltonian`` uses the computational-basis order
    returned by :func:`computational_basis`.
    """

    params: IsingLimitParameters
    hamiltonian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


PauliOp = tuple[int, str]
PauliTerm = tuple[complex, tuple[PauliOp, ...]]


def _immutable_matrix(entries: Sequence[Sequence[complex]]) -> np.ndarray:
    """Create a read-only complex matrix for a module-level constant."""
    matrix = np.array(entries, dtype=complex)
    matrix.flags.writeable = False
    return matrix


PAULI_MATRICES: Mapping[str, np.ndarray] = MappingProxyType(
    {
        "I": _immutable_matrix(((1.0, 0.0), (0.0, 1.0))),
        "X": _immutable_matrix(((0.0, 1.0), (1.0, 0.0))),
        "Y": _immutable_matrix(((0.0, -1.0j), (1.0j, 0.0))),
        "Z": _immutable_matrix(((1.0, 0.0), (0.0, -1.0))),
    }
)

_PAULI_PRODUCT: dict[tuple[str, str], tuple[complex, str]] = {
    ("I", "I"): (1.0, "I"),
    ("I", "X"): (1.0, "X"),
    ("I", "Y"): (1.0, "Y"),
    ("I", "Z"): (1.0, "Z"),
    ("X", "I"): (1.0, "X"),
    ("Y", "I"): (1.0, "Y"),
    ("Z", "I"): (1.0, "Z"),
    ("X", "X"): (1.0, "I"),
    ("Y", "Y"): (1.0, "I"),
    ("Z", "Z"): (1.0, "I"),
    ("X", "Y"): (1.0j, "Z"),
    ("Y", "X"): (-1.0j, "Z"),
    ("Y", "Z"): (1.0j, "X"),
    ("Z", "Y"): (-1.0j, "X"),
    ("Z", "X"): (1.0j, "Y"),
    ("X", "Z"): (-1.0j, "Y"),
}


def _validate_n_sites(n_sites: int) -> int:
    """Validate and normalize a computational-basis site count."""
    if isinstance(n_sites, bool) or not isinstance(n_sites, (int, np.integer)):
        raise TypeError("n_sites must be an integer.")
    n_sites = int(n_sites)
    if n_sites < 1:
        raise ValueError("n_sites must be at least 1.")
    return n_sites


def computational_basis(n_sites: int) -> tuple[str, ...]:
    """Return all computational-basis labels in ordinary binary order.

    Site 0 is the leftmost, most-significant bit of every returned label.
    """
    n_sites = _validate_n_sites(n_sites)
    return tuple(format(state, f"0{n_sites}b") for state in range(1 << n_sites))


def basis_state_label(state: int, n_sites: int) -> str:
    """Return the fixed-width binary label for one computational-basis state.

    The first character is physical site 0, matching :func:`bit_at`.
    """
    n_sites = _validate_n_sites(n_sites)
    if isinstance(state, bool) or not isinstance(state, (int, np.integer)):
        raise TypeError("state must be an integer.")
    state = int(state)
    if state < 0 or state >= (1 << n_sites):
        raise ValueError(f"state must lie in 0..{(1 << n_sites) - 1}.")
    return format(state, f"0{n_sites}b")


def _validate_site(site: int, n_sites: int) -> int:
    if isinstance(site, bool) or not isinstance(site, (int, np.integer)):
        raise TypeError("site must be an integer.")
    site = int(site)
    if site < 0 or site >= n_sites:
        raise IndexError(f"site must lie in 0..{n_sites - 1}.")
    return site


def bit_at(state: int, site: int, n_sites: int) -> int:
    """Return the bit at physical site ``site`` with site 0 as the MSB."""
    n_sites = _validate_n_sites(n_sites)
    site = _validate_site(site, n_sites)
    if isinstance(state, bool) or not isinstance(state, (int, np.integer)):
        raise TypeError("state must be an integer.")
    state = int(state)
    if state < 0 or state >= (1 << n_sites):
        raise ValueError(f"state must lie in 0..{(1 << n_sites) - 1}.")
    return (state >> (n_sites - 1 - site)) & 1


def z_value(state: int, site: int, n_sites: int) -> int:
    """Return the Z eigenvalue at physical site ``site``."""
    return 1 - 2 * bit_at(state, site, n_sites)


def flip_physical_site(state: int, site: int, n_sites: int) -> int:
    """Flip a physical site using the canonical MSB site convention."""
    site = _validate_site(site, n_sites)
    if isinstance(state, bool) or not isinstance(state, (int, np.integer)):
        raise TypeError("state must be an integer.")
    state = int(state)
    if state < 0 or state >= (1 << n_sites):
        raise ValueError(f"state must lie in 0..{(1 << n_sites) - 1}.")
    return state ^ (1 << (n_sites - 1 - site))


def diagonal_energy(state: int, params: IsingLimitParameters) -> float:
    """Return the diagonal Z and ZZ energy for one basis state."""
    _ = bit_at(state, 0, params.n_sites)
    spins = [z_value(state, site, params.n_sites) for site in range(params.n_sites)]
    energy = -2.0 * params.J * sum(spins)
    n_links = params.n_sites if params.periodic else params.n_sites - 1
    for site in range(n_links):
        energy += params.J * spins[site] * spins[(site + 1) % params.n_sites]
    return float(energy)


def flip_amplitude(state: int, site: int, params: IsingLimitParameters) -> float:
    """Return the coefficient multiplying the single-site flip at ``site``."""
    site = _validate_site(site, params.n_sites)
    _ = bit_at(state, 0, params.n_sites)

    if params.mode is HamiltonianMode.TOY:
        if params.periodic:
            left = z_value(state, (site - 1) % params.n_sites, params.n_sites)
            right = z_value(state, (site + 1) % params.n_sites, params.n_sites)
        else:
            left = z_value(state, site - 1, params.n_sites) if site > 0 else 0.0
            right = (
                z_value(state, site + 1, params.n_sites)
                if site + 1 < params.n_sites
                else 0.0
            )
        return float(params.hx * (left + right) / math.sqrt(2.0))

    if params.mode is HamiltonianMode.PAPER:
        left = z_value(state, (site - 1) % params.n_sites, params.n_sites)
        right = z_value(state, (site + 1) % params.n_sites, params.n_sites)
        return float(
            -2.0
            * params.hx
            * ((1.0 - 3.0 * left) / 4.0)
            * ((1.0 - 3.0 * right) / 4.0)
        )

    raise ValueError(f"Unsupported Hamiltonian mode: {params.mode}")


def _simplify_pauli_ops(operators: tuple[PauliOp, ...]) -> tuple[complex, tuple[PauliOp, ...]]:
    per_site: dict[int, str] = {}
    phase: complex = 1.0
    for site, op in operators:
        if op not in {"I", "X", "Y", "Z"}:
            raise ValueError(f"Unknown Pauli operator: {op}")
        if op == "I":
            continue
        current = per_site.get(site, "I")
        factor, product = _PAULI_PRODUCT[(current, op)]
        phase *= factor
        if product == "I":
            per_site.pop(site, None)
        else:
            per_site[site] = product
    return phase, tuple(sorted(per_site.items()))


def _combine_terms(raw_terms: Iterator[PauliTerm], cutoff: float = 0.0) -> list[PauliTerm]:
    combined: dict[tuple[PauliOp, ...], complex] = {}
    for coeff, operators in raw_terms:
        phase, simplified = _simplify_pauli_ops(operators)
        value = complex(coeff) * phase
        if abs(value) <= cutoff:
            continue
        combined[simplified] = combined.get(simplified, 0.0) + value
    return [
        (coeff, operators)
        for operators, coeff in sorted(combined.items(), key=lambda item: item[0])
        if abs(coeff) > cutoff
    ]


def _validate_cutoff(cutoff: float) -> float:
    """Validate a non-negative finite coefficient cutoff."""
    if isinstance(cutoff, bool) or not isinstance(cutoff, Real):
        raise TypeError("cutoff must be a real number.")
    cutoff = float(cutoff)
    if not math.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("cutoff must be finite and non-negative.")
    return cutoff


def pauli_label_from_ops(
    n_sites: int,
    operators: Sequence[PauliOp],
) -> str:
    """Convert site-local Pauli operators to a full left-to-right label.

    Site 0 is the first character.  ``operators`` must already be simplified
    to at most one non-identity operator per site, as is guaranteed for terms
    yielded by :func:`iter_pauli_terms`.
    """
    n_sites = _validate_n_sites(n_sites)
    label = ["I"] * n_sites
    occupied: set[int] = set()
    for entry in operators:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise TypeError("each Pauli operator must be a (site, name) pair.")
        site, name = entry
        if isinstance(site, bool) or not isinstance(site, (int, np.integer)):
            raise TypeError("Pauli operator sites must be integers.")
        site = int(site)
        if site < 0 or site >= n_sites:
            raise ValueError(f"Pauli operator sites must lie in 0..{n_sites - 1}.")
        if name not in PAULI_MATRICES:
            raise ValueError(f"Unknown Pauli operator: {name!r}.")
        if name == "I":
            continue
        if site in occupied:
            raise ValueError(
                "Pauli operators must be simplified to one non-identity "
                "operator per site."
            )
        occupied.add(site)
        label[site] = name
    return "".join(label)


def analytical_pauli_decomposition(
    params: IsingLimitParameters,
    cutoff: float = 1e-12,
) -> dict[str, complex]:
    """Return the canonical analytical Hamiltonian Pauli decomposition.

    Terms come exclusively from :func:`iter_pauli_terms`; no dense matrix is
    constructed or traced.  Dictionary insertion order is lexicographic in
    the full left-to-right Pauli labels.
    """
    cutoff = _validate_cutoff(cutoff)
    coefficients = {
        pauli_label_from_ops(params.n_sites, operators): complex(coefficient)
        for coefficient, operators in iter_pauli_terms(params)
        if abs(coefficient) > cutoff
    }
    return dict(sorted(coefficients.items()))


def pauli_string_matrix(label: str) -> np.ndarray:
    """Return a Pauli-string matrix in left-to-right site order."""
    if not isinstance(label, str):
        raise TypeError("label must be a string.")
    if not label:
        raise ValueError("label must contain at least one Pauli operator.")
    unknown = set(label) - set(PAULI_MATRICES)
    if unknown:
        raise ValueError(f"label contains unknown Pauli operators: {sorted(unknown)}.")

    matrix = np.array([[1.0]], dtype=complex)
    for name in label:
        matrix = np.kron(matrix, PAULI_MATRICES[name])
    return matrix


def trace_pauli_decomposition(
    hamiltonian: np.ndarray,
    n_sites: int,
    cutoff: float = 1e-12,
    max_n_sites: int = 8,
) -> dict[str, complex]:
    """Decompose an arbitrary small matrix by tracing against Pauli strings.

    This generic ``4**N`` routine is intended for validation and arbitrary
    small matrices.  The canonical Ising-limit Hamiltonian decomposition must
    instead use :func:`analytical_pauli_decomposition` and, ultimately,
    :func:`iter_pauli_terms`.
    """
    n_sites = _validate_n_sites(n_sites)
    cutoff = _validate_cutoff(cutoff)
    if isinstance(max_n_sites, bool) or not isinstance(
        max_n_sites, (int, np.integer)
    ):
        raise TypeError("max_n_sites must be an integer.")
    max_n_sites = int(max_n_sites)
    if max_n_sites < 1:
        raise ValueError("max_n_sites must be at least 1.")
    if n_sites > max_n_sites:
        raise ValueError(
            f"Refusing trace Pauli decomposition for n_sites={n_sites}; "
            f"max_n_sites={max_n_sites}."
        )

    hamiltonian = np.asarray(hamiltonian)
    expected_shape = (1 << n_sites, 1 << n_sites)
    if hamiltonian.shape != expected_shape:
        raise ValueError(
            f"hamiltonian must have shape {expected_shape}, got {hamiltonian.shape}."
        )
    if not np.all(np.isfinite(hamiltonian)):
        raise ValueError("hamiltonian must contain only finite values.")

    coefficients: dict[str, complex] = {}
    dimension = 1 << n_sites
    for names in product(PAULI_MATRICES, repeat=n_sites):
        label = "".join(names)
        pauli_matrix = pauli_string_matrix(label)
        coefficient = np.trace(hamiltonian @ pauli_matrix) / dimension
        coefficient = complex(np.real_if_close(coefficient).item())
        if abs(coefficient) > cutoff:
            coefficients[label] = coefficient
    return coefficients


def reconstruct_from_pauli(
    coefficients: Mapping[str, complex],
) -> np.ndarray:
    """Reconstruct a matrix from equally sized Pauli-string coefficients."""
    if not isinstance(coefficients, Mapping):
        raise TypeError("coefficients must be a mapping of labels to coefficients.")
    if not coefficients:
        raise ValueError("coefficients must contain at least one Pauli string.")

    labels = tuple(coefficients)
    n_sites = len(labels[0])
    if n_sites < 1 or any(len(label) != n_sites for label in labels):
        raise ValueError("all Pauli labels must have the same positive length.")
    result = np.zeros((1 << n_sites, 1 << n_sites), dtype=complex)
    for label, coefficient in coefficients.items():
        if isinstance(coefficient, bool) or not np.isscalar(coefficient):
            raise TypeError("Pauli coefficients must be numeric scalars.")
        coefficient = complex(coefficient)
        if not np.isfinite(coefficient):
            raise ValueError("Pauli coefficients must be finite.")
        result += coefficient * pauli_string_matrix(label)
    return result


def iter_pauli_terms(params: IsingLimitParameters) -> Iterator[PauliTerm]:
    """Yield combined Pauli terms ``(coefficient, ((site, op), ...))``.

    Duplicate terms are combined, and repeated operators on the same site are
    simplified using the Pauli algebra.  For periodic N=2 paper mode, the two
    neighbouring Z factors coincide and the ``Z^2 = I`` simplification is
    therefore essential.
    """

    def raw_terms() -> Iterator[PauliTerm]:
        n_links = params.n_sites if params.periodic else params.n_sites - 1
        for site in range(n_links):
            yield params.J, ((site, "Z"), ((site + 1) % params.n_sites, "Z"))

        for site in range(params.n_sites):
            yield -2.0 * params.J, ((site, "Z"),)

        if params.mode is HamiltonianMode.TOY:
            coeff = params.hx / math.sqrt(2.0)
            if params.periodic:
                for site in range(params.n_sites):
                    yield coeff, (((site - 1) % params.n_sites, "Z"), (site, "X"))
                    yield coeff, ((site, "X"), ((site + 1) % params.n_sites, "Z"))
            else:
                for site in range(params.n_sites - 1):
                    yield coeff, ((site, "Z"), (site + 1, "X"))
                    yield coeff, ((site, "X"), (site + 1, "Z"))
            return

        if params.mode is HamiltonianMode.PAPER:
            for site in range(params.n_sites):
                left = (site - 1) % params.n_sites
                right = (site + 1) % params.n_sites
                yield -params.hx / 8.0, ((site, "X"),)
                yield 3.0 * params.hx / 8.0, ((left, "Z"), (site, "X"))
                yield 3.0 * params.hx / 8.0, ((site, "X"), (right, "Z"))
                yield -9.0 * params.hx / 8.0, (
                    (left, "Z"),
                    (site, "X"),
                    (right, "Z"),
                )
            return

        raise ValueError(f"Unsupported Hamiltonian mode: {params.mode}")

    yield from _combine_terms(raw_terms())


def build_sparse_hamiltonian(params: IsingLimitParameters) -> sp.csr_matrix:
    """Build the Hamiltonian directly in CSR format."""
    dim = 1 << params.n_sites
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for ket in range(dim):
        diag = diagonal_energy(ket, params)
        if diag != 0.0:
            rows.append(ket)
            cols.append(ket)
            data.append(diag)

        for site in range(params.n_sites):
            amplitude = flip_amplitude(ket, site, params)
            if amplitude == 0.0:
                continue
            bra = flip_physical_site(ket, site, params.n_sites)
            rows.append(bra)
            cols.append(ket)
            data.append(amplitude)

    H = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.float64).tocsr()
    antihermitian = H - H.getH()
    if antihermitian.nnz:
        max_error = float(np.max(np.abs(antihermitian.data)))
        if max_error > 1e-12:
            raise RuntimeError(f"Constructed Hamiltonian is not Hermitian: max error {max_error}")
    return H


def build_dense_hamiltonian(
    params: IsingLimitParameters, max_n_sites: int = 12
) -> np.ndarray:
    """Build a dense Hamiltonian for small systems."""
    if isinstance(max_n_sites, bool) or not isinstance(max_n_sites, (int, np.integer)):
        raise TypeError("max_n_sites must be an integer.")
    max_n_sites = int(max_n_sites)
    if max_n_sites < 2:
        raise ValueError("max_n_sites must be at least 2.")
    if params.n_sites > max_n_sites:
        raise ValueError(
            f"Refusing dense construction for n_sites={params.n_sites}; "
            f"max_n_sites={max_n_sites}."
        )
    return build_sparse_hamiltonian(params).toarray()


def exact_diagonalize(
    params: IsingLimitParameters,
    max_n_sites: int = 12,
) -> ExactDiagonalizationResult:
    """Construct and exactly diagonalize the dense small-system Hamiltonian.

    The Hamiltonian is constructed exclusively through
    :func:`build_dense_hamiltonian`.  The returned eigenvectors are stored in
    columns and retain the phases chosen by :func:`numpy.linalg.eigh`.

    Raises
    ------
    ValueError
        If the constructed matrix is not square, Hermitian, or finite, or if
        the eigensolver returns non-finite eigenvalues.
    RuntimeError
        If the returned eigenvectors fail orthonormality or residual checks.
    """
    hamiltonian = np.asarray(build_dense_hamiltonian(params, max_n_sites))
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError(
            f"Dense Hamiltonian must be square, got shape {hamiltonian.shape}."
        )
    if not np.all(np.isfinite(hamiltonian)):
        raise ValueError("Dense Hamiltonian must contain only finite values.")

    matrix_scale = max(1.0, float(np.linalg.norm(hamiltonian, ord=np.inf)))
    tolerance = 1e-10 * matrix_scale
    if not np.allclose(
        hamiltonian,
        hamiltonian.conj().T,
        rtol=1e-12,
        atol=tolerance,
    ):
        raise ValueError("Dense Hamiltonian must be Hermitian.")

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("Exact diagonalization produced non-finite eigenvalues.")
    if not np.all(np.isfinite(eigenvectors)):
        raise RuntimeError("Exact diagonalization produced non-finite eigenvectors.")

    identity = np.eye(eigenvectors.shape[1], dtype=eigenvectors.dtype)
    gram_matrix = eigenvectors.conj().T @ eigenvectors
    if not np.allclose(gram_matrix, identity, rtol=1e-10, atol=1e-10):
        raise RuntimeError("Exact eigenvectors are not orthonormal.")

    residuals = hamiltonian @ eigenvectors - eigenvectors * eigenvalues[np.newaxis, :]
    residual_norms = np.linalg.norm(residuals, axis=0)
    if np.any(residual_norms > tolerance):
        raise RuntimeError(
            "Exact eigenpair residual exceeds tolerance: "
            f"max residual {float(np.max(residual_norms)):.3e}, "
            f"tolerance {tolerance:.3e}."
        )

    return ExactDiagonalizationResult(
        params=params,
        hamiltonian=hamiltonian,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
    )


def reduced_density_matrix(
    statevector: np.ndarray,
    n_sites: int,
    subsystem: Sequence[int],
) -> np.ndarray:
    """Return the reduced density matrix of a pure-state subsystem.

    The computational-basis vector is reshaped into one tensor axis per site.
    With ordinary row-major ordering, tensor axis 0 is the leftmost,
    most-significant-bit site 0.  The selected subsystem axes are moved to the
    front in the order supplied, so arbitrary non-contiguous subsystems are
    supported before the complementary sites are traced out.

    The input state is normalized internally and must be nonzero.  Empty and
    full subsystems are valid and produce density matrices of shape ``(1, 1)``
    and ``(2**n_sites, 2**n_sites)``, respectively.
    """
    n_sites = _validate_n_sites(n_sites)
    statevector = np.asarray(statevector)
    expected_shape = (1 << n_sites,)
    if statevector.shape != expected_shape:
        raise ValueError(
            f"statevector must have shape {expected_shape}, got {statevector.shape}."
        )
    if not np.all(np.isfinite(statevector)):
        raise ValueError("statevector must contain only finite values.")

    try:
        subsystem_tuple = tuple(subsystem)
    except TypeError as error:
        raise TypeError("subsystem must be a sequence of integer site indices.") from error
    for site in subsystem_tuple:
        if isinstance(site, bool) or not isinstance(site, (int, np.integer)):
            raise TypeError("subsystem indices must be integers.")
    subsystem_tuple = tuple(int(site) for site in subsystem_tuple)
    if len(set(subsystem_tuple)) != len(subsystem_tuple):
        raise ValueError("subsystem indices must be unique.")
    if any(site < 0 or site >= n_sites for site in subsystem_tuple):
        raise ValueError(f"subsystem indices must lie in 0..{n_sites - 1}.")

    norm = float(np.linalg.norm(statevector))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("statevector must have nonzero finite norm.")
    normalized = np.asarray(statevector, dtype=complex) / norm

    complement = tuple(site for site in range(n_sites) if site not in subsystem_tuple)
    state_tensor = normalized.reshape((2,) * n_sites)
    reordered = np.transpose(state_tensor, axes=subsystem_tuple + complement)
    psi_matrix = reordered.reshape(
        1 << len(subsystem_tuple), 1 << len(complement)
    )
    density_matrix = psi_matrix @ psi_matrix.conj().T

    tolerance = 1e-12
    if not np.allclose(
        density_matrix, density_matrix.conj().T, rtol=1e-12, atol=tolerance
    ):
        raise RuntimeError("Reduced density matrix is not Hermitian.")
    trace = np.trace(density_matrix)
    if not np.isclose(trace, 1.0, rtol=1e-12, atol=tolerance):
        raise RuntimeError(
            f"Reduced density matrix must have unit trace, got {trace!r}."
        )
    return density_matrix


def von_neumann_entropy(
    density_matrix: np.ndarray,
    log_base: float = 2.0,
    tolerance: float = 1e-12,
) -> float:
    """Return ``-Tr(rho log(rho))`` for a normalized density matrix.

    Hermitian eigensolving is used.  Eigenvalues in ``[-tolerance, 0)`` are
    treated as floating-point noise and clipped to zero; more negative values
    are rejected.  Eigenvalues no larger than ``tolerance`` do not contribute
    to the logarithm.
    """
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise TypeError("tolerance must be a real number.")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    if isinstance(log_base, bool) or not isinstance(log_base, Real):
        raise TypeError("log_base must be a real number.")
    log_base = float(log_base)
    if not math.isfinite(log_base) or log_base <= 0.0 or log_base == 1.0:
        raise ValueError("log_base must be finite, positive, and different from 1.")

    density_matrix = np.asarray(density_matrix)
    if (
        density_matrix.ndim != 2
        or density_matrix.shape[0] != density_matrix.shape[1]
    ):
        raise ValueError(
            f"density_matrix must be square, got shape {density_matrix.shape}."
        )
    if not np.all(np.isfinite(density_matrix)):
        raise ValueError("density_matrix must contain only finite values.")
    if not np.allclose(
        density_matrix,
        density_matrix.conj().T,
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ValueError("density_matrix must be Hermitian.")
    trace = np.trace(density_matrix)
    if not np.isclose(trace, 1.0, rtol=tolerance, atol=tolerance):
        raise ValueError(f"density_matrix must have unit trace, got {trace!r}.")

    eigenvalues = np.linalg.eigvalsh(density_matrix)
    if np.any(eigenvalues < -tolerance):
        minimum = float(np.min(eigenvalues))
        raise ValueError(
            "density_matrix must be positive semidefinite; "
            f"minimum eigenvalue is {minimum:.3e}."
        )
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    eigenvalues = eigenvalues[eigenvalues > tolerance]
    if eigenvalues.size == 0:
        return 0.0

    entropy = -float(np.sum(eigenvalues * np.log(eigenvalues)) / math.log(log_base))
    if entropy < -tolerance:
        raise RuntimeError(f"Computed a negative von Neumann entropy: {entropy}.")
    return max(0.0, entropy)


def entanglement_entropy(
    statevector: np.ndarray,
    n_sites: int,
    subsystem: Sequence[int],
    log_base: float = 2.0,
) -> float:
    """Return pure-state entanglement entropy across a chosen subsystem cut."""
    density_matrix = reduced_density_matrix(statevector, n_sites, subsystem)
    return von_neumann_entropy(density_matrix, log_base=log_base)


def eigenstate_entropies(
    result: ExactDiagonalizationResult,
    subsystem: Sequence[int],
    log_base: float = 2.0,
) -> np.ndarray:
    """Return one entanglement entropy per eigenvector column in ``result``."""
    expected_rows = 1 << result.params.n_sites
    if result.eigenvectors.ndim != 2 or result.eigenvectors.shape[0] != expected_rows:
        raise ValueError(
            "result.eigenvectors must have one row per computational-basis state; "
            f"got shape {result.eigenvectors.shape}."
        )
    return np.array(
        [
            entanglement_entropy(
                result.eigenvectors[:, index],
                result.params.n_sites,
                subsystem,
                log_base=log_base,
            )
            for index in range(result.eigenvectors.shape[1])
        ],
        dtype=float,
    )


def apply_hamiltonian(vector: np.ndarray, params: IsingLimitParameters) -> np.ndarray:
    """Apply H to a vector without constructing a dense or sparse matrix."""
    vector = np.asarray(vector)
    dim = 1 << params.n_sites
    if vector.shape != (dim,):
        raise ValueError(f"vector must have shape {(dim,)}, got {vector.shape}.")
    result = np.zeros_like(vector, dtype=np.result_type(vector.dtype, np.float64))

    for ket in range(dim):
        value = vector[ket]
        result[ket] += diagonal_energy(ket, params) * value
        for site in range(params.n_sites):
            amplitude = flip_amplitude(ket, site, params)
            if amplitude == 0.0:
                continue
            bra = flip_physical_site(ket, site, params.n_sites)
            result[bra] += amplitude * value
    return result


def build_linear_operator(params: IsingLimitParameters) -> LinearOperator:
    """Return a matrix-free SciPy linear operator for the Hamiltonian."""
    dim = 1 << params.n_sites

    def matvec(vector: np.ndarray) -> np.ndarray:
        return apply_hamiltonian(vector, params)

    return LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec, dtype=np.float64)


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without performing model calculations."""
    parser = argparse.ArgumentParser(
        description="Small-system analysis for the canonical Ising-limit Hamiltonian."
    )
    parser.add_argument("--n-sites", type=int, default=2)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--hx", type=float, default=1.0)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in HamiltonianMode],
        default=HamiltonianMode.TOY.value,
    )
    parser.add_argument("--open-boundary", action="store_true")
    parser.add_argument("--max-dense-sites", type=int, default=12)
    parser.add_argument("--subsystem", type=int, nargs="+", default=None)
    parser.add_argument("--entropy-base", type=float, default=2.0)
    parser.add_argument("--show-basis", action="store_true")
    parser.add_argument("--show-matrix", action="store_true")
    parser.add_argument("--show-eigenvalues", action="store_true")
    parser.add_argument("--show-eigenvectors", action="store_true")
    parser.add_argument("--show-entropies", action="store_true")
    parser.add_argument("--show-pauli", action="store_true")
    parser.add_argument(
        "--pauli-method", choices=("analytical", "trace"), default="analytical"
    )
    parser.add_argument("--pauli-cutoff", type=float, default=1e-12)
    parser.add_argument("--legacy-two-site-output", action="store_true")
    return parser


def _entropy_unit_label(log_base: float) -> str:
    """Return a human-readable unit label for a logarithm base."""
    if math.isclose(log_base, 2.0, rel_tol=0.0, abs_tol=1e-15):
        return "bits"
    if math.isclose(log_base, math.e, rel_tol=0.0, abs_tol=1e-15):
        return "nats"
    return f"log-base-{log_base:g} units"


def _print_pauli_coefficients(coefficients: Mapping[str, complex]) -> None:
    """Print an already-computed Pauli decomposition deterministically."""
    for label, coefficient in coefficients.items():
        value = np.real_if_close(coefficient).item()
        print(f"  {value:+.12g} * {label}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the optional command-line small-system analysis tool."""
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    try:
        params = IsingLimitParameters(
            n_sites=args.n_sites,
            J=args.J,
            hx=args.hx,
            periodic=not args.open_boundary,
            mode=HamiltonianMode(args.mode),
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    if args.legacy_two_site_output and (
        params.n_sites != 2
        or params.mode is not HamiltonianMode.TOY
        or not params.periodic
    ):
        parser.error(
            "--legacy-two-site-output requires N=2, mode=toy, and periodic boundaries."
        )

    show_basis = args.show_basis or args.legacy_two_site_output
    show_matrix = args.show_matrix or args.legacy_two_site_output
    show_eigenvalues = args.show_eigenvalues or args.legacy_two_site_output
    show_eigenvectors = args.show_eigenvectors or args.legacy_two_site_output
    show_entropies = args.show_entropies or args.legacy_two_site_output
    show_pauli = args.show_pauli or args.legacy_two_site_output
    pauli_method = "analytical" if args.legacy_two_site_output else args.pauli_method

    if args.subsystem is None:
        subsystem = (0,) if params.n_sites == 2 else tuple(range(params.n_sites // 2))
    else:
        subsystem = tuple(args.subsystem)

    print(f"Number of sites (N): {params.n_sites}")
    print(f"Hilbert-space dimension: {1 << params.n_sites}")
    print(f"Boundary conditions: {'periodic' if params.periodic else 'open'}")
    print(f"Hamiltonian mode: {params.mode.value}")
    print(f"J: {params.J:g}")
    print(f"hx: {params.hx:g}")

    if show_basis:
        print("\nComputational/effective-Ising basis states:")
        for index, label in enumerate(computational_basis(params.n_sites)):
            print(f"  {index}: |{label}>")

    needs_eigensystem = show_eigenvalues or show_eigenvectors or show_entropies
    needs_dense_matrix = show_matrix or pauli_method == "trace" and show_pauli
    exact_result: ExactDiagonalizationResult | None = None
    hamiltonian: np.ndarray | None = None
    try:
        if needs_eigensystem:
            exact_result = exact_diagonalize(params, args.max_dense_sites)
            hamiltonian = exact_result.hamiltonian
        elif needs_dense_matrix:
            hamiltonian = build_dense_hamiltonian(params, args.max_dense_sites)
    except (TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    if show_matrix:
        assert hamiltonian is not None
        print("\nHamiltonian matrix in the computational basis:")
        print(np.real_if_close(hamiltonian))

    if show_eigenvalues:
        assert exact_result is not None
        print("\nEigenvalues:")
        print(np.real_if_close(exact_result.eigenvalues))

    if show_eigenvectors:
        assert exact_result is not None
        print("\nEigenvectors (columns) in the computational basis:")
        print(np.real_if_close(exact_result.eigenvectors))

    if show_entropies:
        assert exact_result is not None
        try:
            entropies = eigenstate_entropies(
                exact_result, subsystem, log_base=args.entropy_base
            )
        except (TypeError, ValueError, RuntimeError) as error:
            parser.error(str(error))
        units = _entropy_unit_label(args.entropy_base)
        print(
            f"\nEntanglement entropy for subsystem {list(subsystem)} "
            f"({units}):"
        )
        for index, (energy, entropy) in enumerate(
            zip(exact_result.eigenvalues, entropies)
        ):
            print(
                f"  Eigenstate {index}: E = {float(energy):.12g}, "
                f"S_A = {float(entropy):.12g} {units}"
            )

    if show_pauli:
        try:
            if pauli_method == "analytical":
                coefficients = analytical_pauli_decomposition(
                    params, cutoff=args.pauli_cutoff
                )
            else:
                assert hamiltonian is not None
                coefficients = trace_pauli_decomposition(
                    hamiltonian,
                    params.n_sites,
                    cutoff=args.pauli_cutoff,
                )
        except (TypeError, ValueError) as error:
            parser.error(str(error))
        print(f"\nPauli decomposition ({pauli_method}):")
        _print_pauli_coefficients(coefficients)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
