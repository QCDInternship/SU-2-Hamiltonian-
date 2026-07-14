"""TeNPy MPO construction for the canonical Ising-limit Hamiltonians.

This module builds MPOs from the already-coalesced Pauli terms produced by
``ising_limit_model.iter_pauli_terms``.  It deliberately does not retype the
toy or paper Hamiltonian formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ising_limit_model import (
    HamiltonianMode,
    IsingLimitParameters,
    iter_pauli_terms,
)


TENPY_IMPORT_ERROR = (
    "TeNPy is required for ising_limit_mpo.py. Install a stable TeNPy release "
    "to build and validate MPOs."
)


@dataclass(frozen=True)
class IsingLimitMPOResult:
    """Container for the TeNPy model and MPO."""

    params: IsingLimitParameters
    site: Any
    lattice: Any
    model: Any
    mpo: Any


def _import_tenpy() -> tuple[Any, Any, Any]:
    try:
        from tenpy.models.lattice import Chain
        from tenpy.models.model import CouplingModel
        from tenpy.networks.site import SpinHalfSite
    except ImportError as error:  # pragma: no cover - depends on optional dependency.
        raise ImportError(TENPY_IMPORT_ERROR) from error
    return Chain, CouplingModel, SpinHalfSite


def _operator_to_numpy(operator: Any) -> np.ndarray:
    if hasattr(operator, "to_ndarray"):
        return np.asarray(operator.to_ndarray())
    return np.asarray(operator)


def create_spin_half_site() -> Any:
    """Return a spin-1/2 TeNPy site with no conserved charge.

    The Hamiltonians contain X terms, so Z magnetization is not conserved.
    This function verifies that TeNPy's ``Sigmax`` and ``Sigmaz`` are Pauli
    matrices, not spin operators divided by two.
    """
    _, _, SpinHalfSite = _import_tenpy()
    site = SpinHalfSite(conserve=None)
    validate_spin_half_site(site)
    return site


def validate_spin_half_site(site: Any) -> None:
    """Verify the local Pauli operators used by the MPO construction."""
    sigma_x = _operator_to_numpy(site.get_op("Sigmax"))
    sigma_z = _operator_to_numpy(site.get_op("Sigmaz"))
    identity = np.eye(2)

    np.testing.assert_allclose(sigma_x @ sigma_x, identity, atol=1e-14)
    np.testing.assert_allclose(sigma_z @ sigma_z, identity, atol=1e-14)
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(sigma_x)), [-1.0, 1.0])
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(sigma_z)), [-1.0, 1.0])


def _tenpy_operator_name(pauli_name: str) -> str:
    if pauli_name == "X":
        return "Sigmax"
    if pauli_name == "Z":
        return "Sigmaz"
    if pauli_name == "Y":
        return "Sigmay"
    raise ValueError(f"Unsupported Pauli operator for TeNPy MPO: {pauli_name}")


def _add_pauli_term_to_model(model: Any, coefficient: complex, operators: tuple[tuple[int, str], ...]) -> None:
    if len(operators) == 0:
        raise ValueError("Identity-only Hamiltonian terms are not expected here.")

    indices = [site for site, _ in operators]
    names = [_tenpy_operator_name(op) for _, op in operators]
    strength = float(np.real_if_close(coefficient))

    if len(operators) == 1:
        model.add_onsite_term(strength, indices[0], names[0])
    elif len(operators) == 2:
        model.add_coupling_term(strength, indices[0], indices[1], names[0], names[1], op_string="Id")
    else:
        op_string = ["Id"] * (len(operators) - 1)
        model.add_multi_coupling_term(strength, indices, names, op_string)


def build_ising_limit_mpo(params: IsingLimitParameters, tol_zero: float = 1e-15) -> IsingLimitMPOResult:
    """Build a finite-chain TeNPy MPO for the canonical Ising-limit Hamiltonian."""
    params = IsingLimitParameters(
        n_sites=params.n_sites,
        J=params.J,
        hx=params.hx,
        periodic=params.periodic,
        mode=params.mode,
    )
    Chain, CouplingModel, _ = _import_tenpy()
    site = create_spin_half_site()
    lattice = Chain(params.n_sites, site, bc="open", bc_MPS="finite")
    model = CouplingModel(lattice)

    for coefficient, operators in iter_pauli_terms(params):
        _add_pauli_term_to_model(model, coefficient, operators)

    mpo = model.calc_H_MPO(tol_zero=tol_zero)
    return IsingLimitMPOResult(params=params, site=site, lattice=lattice, model=model, mpo=mpo)


def build_ising_limit_model(
    n_sites: int,
    J: float,
    hx: float,
    periodic: bool,
    mode: str | HamiltonianMode,
    tol_zero: float = 1e-15,
) -> IsingLimitMPOResult:
    """Backward-friendly helper accepting scalar parameters."""
    params = IsingLimitParameters(
        n_sites=n_sites,
        J=J,
        hx=hx,
        periodic=periodic,
        mode=HamiltonianMode(mode),
    )
    return build_ising_limit_mpo(params, tol_zero=tol_zero)
