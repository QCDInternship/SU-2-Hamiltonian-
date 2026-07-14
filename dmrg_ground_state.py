"""Finite two-site DMRG ground states for the canonical Ising-limit MPO.

Only the validated toy Hamiltonian with finite open boundaries is exposed.
The Hamiltonian itself is constructed by :mod:`ising_limit_mpo`; this module
contains no independent Hamiltonian or MPO implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ising_limit_model import HamiltonianMode, IsingLimitParameters
from ising_limit_mpo import build_ising_limit_mpo


TENPY_INSTALL_MESSAGE = "TeNPy is required; install it with: pip install physics-tenpy"
INITIAL_STATES = ("all-up", "all-down", "alternating", "random")


@dataclass
class DMRGResult:
    energy: float
    energy_per_site: float
    converged: bool
    sweeps: int
    energy_history: list[float]
    truncation_error_history: list[float]
    maximum_bond_dimension: int
    z: list[float]
    x: list[float]
    zz: list[float]
    zx: list[float]
    xz: list[float]
    bond_entropy_nats: list[float]
    bond_entropy_bits: list[float]
    half_chain_entropy_nats: float
    half_chain_entropy_bits: float
    central_schmidt_values: list[float]
    energy_variance: float | None
    initial_state: str
    psi: Any


def _import_tenpy() -> tuple[Any, Any]:
    try:
        from tenpy.algorithms import dmrg
        from tenpy.networks.mps import MPS
    except ImportError as error:
        raise ImportError(TENPY_INSTALL_MESSAGE) from error
    return dmrg, MPS


def _product_state(name: str, n_sites: int, rng: np.random.Generator) -> list[str]:
    if name == "all-up":
        return ["up"] * n_sites
    if name == "all-down":
        return ["down"] * n_sites
    if name == "alternating":
        return ["up" if i % 2 == 0 else "down" for i in range(n_sites)]
    if name == "random":
        return ["up" if bit == 0 else "down" for bit in rng.integers(0, 2, n_sites)]
    raise ValueError(f"unknown initial state {name!r}")


def _real_list(values: Any) -> list[float]:
    return [float(np.real_if_close(value)) for value in np.asarray(values).reshape(-1)]


def _bond_expectations(psi: Any, left_op: str, right_op: str) -> list[float]:
    return [
        float(np.real_if_close(psi.expectation_value_term([(left_op, i), (right_op, i + 1)])))
        for i in range(psi.L - 1)
    ]


def _history(stats: dict[str, Any], names: Sequence[str]) -> list[float]:
    for name in names:
        if name in stats:
            return _real_list(stats[name])
    return []


def _energy_variance(mpo: Any, psi: Any, energy: float) -> float | None:
    """Use a TeNPy-provided variance method when the installed version has one."""
    for owner, name, args in (
        (mpo, "variance", (psi,)),
        (mpo, "expectation_value_power", (psi, 2)),
    ):
        method = getattr(owner, name, None)
        if method is None:
            continue
        try:
            value = float(np.real_if_close(method(*args)))
            return value if name == "variance" else value - energy * energy
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            pass
    return None


def _run_one(
    *, n_sites: int, J: float, hx: float, chi_schedule: Sequence[int],
    max_sweeps: int, energy_tol: float, svd_min: float, seed: int,
    initial_state: str,
) -> DMRGResult:
    dmrg, MPS = _import_tenpy()
    params = IsingLimitParameters(n_sites, J, hx, False, HamiltonianMode.TOY)
    built = build_ising_limit_mpo(params)
    # ``CouplingModel`` does not populate this attribute in every TeNPy
    # release when ``calc_H_MPO`` is called directly.  DMRG's engine consumes
    # the already-built canonical MPO through ``model.H_MPO``.
    built.model.H_MPO = built.mpo
    rng = np.random.default_rng(seed)
    psi = MPS.from_product_state(
        built.lattice.mps_sites(), _product_state(initial_state, n_sites, rng), bc="finite"
    )
    options = {
        "mixer": True,
        "max_sweeps": max_sweeps,
        "max_E_err": energy_tol,
        "chi_list": {i: int(chi) for i, chi in enumerate(chi_schedule)},
        "trunc_params": {"chi_max": int(max(chi_schedule)), "svd_min": svd_min},
        "verbose": 0,
    }
    engine = dmrg.TwoSiteDMRGEngine(psi, built.model, options)
    energy, psi = engine.run()
    energy = float(np.real_if_close(energy))
    stats = getattr(engine, "sweep_stats", {}) or {}
    energies = _history(stats, ("E", "energy"))
    if not energies or abs(energies[-1] - energy) > 1e-14:
        energies.append(energy)
    truncation = _history(stats, ("max_trunc_err", "trunc_err", "max_truncation_error"))
    sweeps = int(getattr(engine, "sweeps", len(energies)))
    converged = bool(
        len(energies) >= 2 and abs(energies[-1] - energies[-2]) <= energy_tol
    )
    entropy_nats = _real_list(psi.entanglement_entropy())
    entropy_bits = [value / math.log(2.0) for value in entropy_nats]
    central_cut = n_sites // 2
    schmidt = _real_list(psi.get_SL(central_cut))
    chis = list(getattr(psi, "chi", []))
    return DMRGResult(
        energy=energy,
        energy_per_site=energy / n_sites,
        converged=converged,
        sweeps=sweeps,
        energy_history=energies,
        truncation_error_history=truncation,
        maximum_bond_dimension=int(max(chis, default=1)),
        z=_real_list(psi.expectation_value("Sigmaz")),
        x=_real_list(psi.expectation_value("Sigmax")),
        zz=_bond_expectations(psi, "Sigmaz", "Sigmaz"),
        zx=_bond_expectations(psi, "Sigmaz", "Sigmax"),
        xz=_bond_expectations(psi, "Sigmax", "Sigmaz"),
        bond_entropy_nats=entropy_nats,
        bond_entropy_bits=entropy_bits,
        half_chain_entropy_nats=entropy_nats[central_cut - 1],
        half_chain_entropy_bits=entropy_bits[central_cut - 1],
        central_schmidt_values=schmidt,
        energy_variance=_energy_variance(built.mpo, psi, energy),
        initial_state=initial_state,
        psi=psi,
    )


def run_dmrg(
    *, n_sites: int, J: float = 1.0, hx: float = 1.0, mode: str = "toy",
    boundary: str = "open", chi_schedule: Sequence[int] = (16, 32, 64),
    max_sweeps: int = 12, energy_tol: float = 1e-12, svd_min: float = 1e-14,
    seed: int = 0, initial_states: Sequence[str] = ("all-up",),
) -> DMRGResult:
    """Run finite two-site DMRG and retain the lowest result over starts."""
    if mode != "toy" or boundary != "open":
        raise ValueError("DMRG currently supports only mode='toy' with boundary='open'.")
    if n_sites < 2 or max_sweeps < 1 or not chi_schedule or any(c < 1 for c in chi_schedule):
        raise ValueError("n_sites >= 2, positive sweeps, and positive chi values are required.")
    if energy_tol <= 0.0 or svd_min < 0.0:
        raise ValueError("energy_tol must be positive and svd_min non-negative.")
    if not initial_states or any(state not in INITIAL_STATES for state in initial_states):
        raise ValueError(f"initial_states must be drawn from {INITIAL_STATES}.")
    results = [
        _run_one(n_sites=n_sites, J=J, hx=hx, chi_schedule=chi_schedule,
                 max_sweeps=max_sweeps, energy_tol=energy_tol, svd_min=svd_min,
                 seed=seed + index, initial_state=state)
        for index, state in enumerate(initial_states)
    ]
    return min(results, key=lambda result: result.energy)


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def save_outputs(result: DMRGResult, output_dir: Path, parameters: dict[str, Any], save_mps: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = asdict(result)
    summary.pop("psi")
    summary["parameters"] = parameters
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "energy_history.csv", ("sweep", "energy", "energy_per_site"),
               [(i + 1, e, e / parameters["n_sites"]) for i, e in enumerate(result.energy_history)])
    _write_csv(output_dir / "local_observables.csv", ("site", "Z", "X"),
               [(i, result.z[i], result.x[i]) for i in range(parameters["n_sites"])])
    _write_csv(output_dir / "bond_observables.csv", ("left_site", "right_site", "ZZ", "ZX", "XZ"),
               [(i, i + 1, result.zz[i], result.zx[i], result.xz[i]) for i in range(parameters["n_sites"] - 1)])
    _write_csv(output_dir / "entanglement_profile.csv", ("cut", "entropy_nats", "entropy_bits"),
               [(i + 1, result.bond_entropy_nats[i], result.bond_entropy_bits[i]) for i in range(parameters["n_sites"] - 1)])
    _write_csv(output_dir / "schmidt_values.csv", ("index", "schmidt_value", "probability"),
               [(i, value, value * value) for i, value in enumerate(result.central_schmidt_values)])
    _write_csv(output_dir / "truncation_error_history.csv", ("sweep", "truncation_error"),
               [(i + 1, value) for i, value in enumerate(result.truncation_error_history)])
    _make_plots(result, output_dir)
    if save_mps:
        with (output_dir / "final_mps.pkl").open("wb") as handle:
            pickle.dump(result.psi, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _make_plots(result: DMRGResult, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = (
        ("energy_vs_sweep.png", range(1, len(result.energy_history) + 1), result.energy_history, "Sweep", "Energy"),
        ("bond_entropy.png", range(1, len(result.bond_entropy_nats) + 1), result.bond_entropy_nats, "Cut", "Entropy (nats)"),
        ("local_z.png", range(len(result.z)), result.z, "Site", r"$\langle Z_i\rangle$"),
        ("local_x.png", range(len(result.x)), result.x, "Site", r"$\langle X_i\rangle$"),
    )
    for filename, xs, ys, xlabel, ylabel in plots:
        fig, ax = plt.subplots()
        ax.plot(list(xs), ys, marker="o")
        ax.set(xlabel=xlabel, ylabel=ylabel)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finite two-site TeNPy DMRG ground-state solver.")
    parser.add_argument("--n-sites", type=int, required=True)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--hx", type=float, default=1.0)
    parser.add_argument("--mode", choices=("toy",), default="toy")
    parser.add_argument("--boundary", choices=("open",), default="open")
    parser.add_argument("--chi-schedule", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--max-sweeps", type=int, default=12)
    parser.add_argument("--energy-tol", type=float, default=1e-12)
    parser.add_argument("--svd-min", type=float, default=1e-14)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-state", choices=INITIAL_STATES, nargs="+", default=("all-up",))
    parser.add_argument("--output-dir", type=Path, default=Path("dmrg_outputs"))
    parser.add_argument("--save-mps", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n_sites < 2 or args.max_sweeps < 1 or any(c < 1 for c in args.chi_schedule):
        parser.error("--n-sites must be >= 2; sweeps and bond dimensions must be positive")
    parameters = vars(args).copy()
    parameters["output_dir"] = str(args.output_dir)
    try:
        result = run_dmrg(n_sites=args.n_sites, J=args.J, hx=args.hx, mode=args.mode,
                          boundary=args.boundary, chi_schedule=args.chi_schedule,
                          max_sweeps=args.max_sweeps, energy_tol=args.energy_tol,
                          svd_min=args.svd_min, seed=args.seed,
                          initial_states=args.initial_state)
    except ImportError as error:
        parser.exit(2, f"error: {error}\n")
    save_outputs(result, args.output_dir, parameters, args.save_mps)
    print(f"Ground-state energy: {result.energy:.15g}")
    print(f"Energy per site:     {result.energy_per_site:.15g}")
    print(f"Converged:           {result.converged}")
    print(f"Sweeps:              {result.sweeps}")
    print(f"Outputs:             {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
