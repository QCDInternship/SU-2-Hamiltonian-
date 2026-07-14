# SU-2 plaquette-chain effective Ising-limit tools

This repository studies the **effective `j_max = 1/2` Ising limit** of an
SU(2) plaquette-chain model. The working Hilbert space is the ordinary
computational basis of `N` effective spins, with dimension `2**N`. Site 0 is
the leftmost, most-significant bit.

This is not the unreduced SU(2) electric-link basis. Full `j_max > 1/2` SU(2)
dynamics, higher-representation link states, intertwiners, Wigner-symbol
matrix elements, and a full higher-cutoff gauge Hamiltonian are not
implemented.

## Canonical model and exact analysis

`ising_limit_model.py` is the canonical Hamiltonian implementation and the
small-system exact-analysis library. It provides:

- computational-basis enumeration and formatting;
- dense, sparse, and matrix-free Hamiltonian representations;
- dense exact diagonalisation with checked eigenpairs;
- reduced density matrices and arbitrary-subsystem entanglement entropy;
- entropy for every exact eigenstate;
- analytical Pauli terms and a bounded trace-based validation decomposition;
- a command-line interface for small-system inspection.

Other programs should reuse this module instead of restating the Hamiltonian.
In particular, MPO construction consumes `iter_pauli_terms()`, while exact
analysis consumes `build_dense_hamiltonian()` through `exact_diagonalize()`.

### Hamiltonian modes

`toy` mode implements

\[
H = \sum_i \left[J Z_i Z_{i+1} - 2J Z_i
  + h_x\frac{Z_{i-1}+Z_{i+1}}{\sqrt{2}}X_i\right].
\]

It supports periodic and open boundaries. Missing neighbours contribute zero
at open boundaries.

`paper` mode retains the repository's appendix-inspired magnetic factor

\[
-2h_x\left(\frac{1-3Z_{i-1}}{4}\right)
       \left(\frac{1-3Z_{i+1}}{4}\right)X_i,
\]

and currently supports periodic boundaries only. The two conventions are
deliberately separate and generally produce different spectra.

## Command-line examples

Show the complete historical two-spin calculation using only canonical code:

```bash
python ising_limit_model.py --n-sites 2 --J 1 --hx 1 --mode toy --legacy-two-site-output
```

The old command remains as a small compatibility entry point:

```bash
python Hamiltonian.py
```

It contains no Hamiltonian or analysis implementation and forwards to the
canonical legacy-output mode.

Inspect a four-site periodic TOY model:

```bash
python ising_limit_model.py --n-sites 4 --mode toy --show-basis --show-matrix --show-eigenvalues
```

Calculate exact eigenstate entropies for a non-contiguous subsystem in nats:

```bash
python ising_limit_model.py --n-sites 4 --subsystem 0 2 --entropy-base 2.718281828459045 --show-entropies
```

Print the primary analytical Pauli decomposition:

```bash
python ising_limit_model.py --n-sites 4 --mode paper --show-pauli --pauli-method analytical
```

Cross-check a small Hamiltonian with the generic trace decomposition:

```bash
python ising_limit_model.py --n-sites 3 --show-pauli --pauli-method trace
```

Run `python ising_limit_model.py --help` for all matrix, eigenvector, entropy,
boundary, cutoff, and dense-size options. Dense exact diagonalisation scales
exponentially and is intended only for small `N`; trace Pauli decomposition
scales as `4**N` and has a stricter safety limit.

## Specialized analyses

- `Hamiltonian_multispin.py` retains an older stateful multi-spin interface
  for compatibility. New exact-analysis code should use
  `ising_limit_model.py`.
- `hamiltonian_multispin_sparse.py` computes selected sparse or matrix-free
  eigenpairs of the same canonical effective model.
- `scar_state_search.py` searches the full effective-spin space for
  low-entanglement outliers.
- `page_curve_k0.py` and `k0_finite_size_scan.py` perform periodic zero-momentum
  sector analysis.
- `ising_limit_mpo.py` builds a TeNPy MPO from the analytical Pauli terms.
- `dmrg_ground_state.py` runs finite two-site DMRG for the open-boundary TOY
  model, and `validate_dmrg.py` compares it with canonical exact results.
- `qvae_scar_detector.py` and `run_qvae_scar_detector.py` provide an optional
  QVAE comparison for effective-model scar candidates.

Sparse, k=0, scar, DMRG, and QVAE calculations in this repository remain
calculations of the `j_max = 1/2` effective Ising model. A low-entanglement or
QVAE-selected state is a candidate requiring further physical validation, not
by itself proof of a quantum scar.

## Optional dependencies

SciPy is required by the sparse model. TeNPy is required for MPO/DMRG work.
PennyLane and PyTorch are optional and only needed for the QVAE workflow; its
additional requirements are listed in `requirements-ai.txt`.

## Testing

Run the suite with:

```bash
pytest -q
```

The regression tests freeze the historical two-spin matrix, spectrum,
eigenpair residuals, entropies, Pauli coefficients, and reconstruction. They
also verify the `Hamiltonian.py` compatibility entry point.
