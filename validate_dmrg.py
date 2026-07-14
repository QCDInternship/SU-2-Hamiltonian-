"""Validate finite toy/open DMRG against canonical exact diagonalisation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from dmrg_ground_state import TENPY_INSTALL_MESSAGE, run_dmrg
from ising_limit_model import HamiltonianMode, IsingLimitParameters, build_dense_hamiltonian


ENERGY_TOL = 1e-8
ENTROPY_TOL = 1e-7
OBSERVABLE_TOL = 1e-7


@dataclass
class ExactResult:
    energy: float
    entropy_nats: float
    z: np.ndarray
    x: np.ndarray
    zz: np.ndarray


def _operator(n_sites: int, entries: dict[int, np.ndarray]) -> np.ndarray:
    identity = np.eye(2)
    result = np.array([[1.0]])
    for site in range(n_sites):
        result = np.kron(result, entries.get(site, identity))
    return result


def exact_ground_state(n_sites: int) -> ExactResult:
    params = IsingLimitParameters(n_sites, 1.0, 1.0, False, HamiltonianMode.TOY)
    values, vectors = np.linalg.eigh(build_dense_hamiltonian(params))
    psi = vectors[:, 0]
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z = np.diag([1.0, -1.0])
    expectation = lambda op: float(np.real_if_close(np.vdot(psi, op @ psi)))
    singular = np.linalg.svd(psi.reshape(2 ** (n_sites // 2), -1), compute_uv=False)
    probabilities = singular * singular
    entropy = -float(np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])))
    return ExactResult(
        float(values[0]), entropy,
        np.array([expectation(_operator(n_sites, {i: Z})) for i in range(n_sites)]),
        np.array([expectation(_operator(n_sites, {i: X})) for i in range(n_sites)]),
        np.array([expectation(_operator(n_sites, {i: Z, i + 1: Z})) for i in range(n_sites - 1)]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare toy/open TeNPy DMRG with exact diagonalisation.")
    parser.add_argument("--sizes", type=int, nargs="+", default=(4, 6, 8))
    parser.add_argument("--chi-schedule", type=int, nargs="+", default=(16, 32, 64, 128))
    parser.add_argument("--max-sweeps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(" N |       E_exact |        E_DMRG |    |dE| | |d(E/N)| |    |dS| | max obs err | result")
    print("---+---------------+---------------+---------+----------+---------+-------------+-------")
    passed = True
    try:
        for n_sites in args.sizes:
            exact = exact_ground_state(n_sites)
            dmrg = run_dmrg(
                n_sites=n_sites, J=1.0, hx=1.0, mode="toy", boundary="open",
                chi_schedule=args.chi_schedule, max_sweeps=args.max_sweeps,
                energy_tol=1e-13, svd_min=1e-15, seed=args.seed,
                initial_states=("all-up", "all-down", "alternating", "random"),
            )
            de = abs(dmrg.energy - exact.energy)
            de_per_site = abs(dmrg.energy_per_site - exact.energy / n_sites)
            ds = abs(dmrg.half_chain_entropy_nats - exact.entropy_nats)
            obs = max(
                float(np.max(np.abs(np.asarray(dmrg.z) - exact.z))),
                float(np.max(np.abs(np.asarray(dmrg.x) - exact.x))),
                float(np.max(np.abs(np.asarray(dmrg.zz) - exact.zz))),
            )
            ok = de < ENERGY_TOL and ds < ENTROPY_TOL and obs < OBSERVABLE_TOL
            passed &= ok
            print(f"{n_sites:2d} | {exact.energy:13.9f} | {dmrg.energy:13.9f} | {de:7.1e} | {de_per_site:8.1e} | {ds:7.1e} | {obs:11.1e} | {'PASS' if ok else 'FAIL'}")
    except ImportError:
        print(f"error: {TENPY_INSTALL_MESSAGE}")
        return 2
    print("Validation PASSED" if passed else "Validation FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
