# SU-2-Hamiltonian-

Entanglement entropy of 
(2+1)-dimensional SU(2) lattice gauge theory on plaquette chains.

This repository currently contains two closely related **toy-model / Ising-limit** scripts:

- `Hamiltonian.py` — the original **2-spin** version
- `Hamiltonian_multispin.py` — the generalized **multi-spin** version for `N >= 2`

These scripts are inspired by the `j_max = 1/2` truncation discussed in the accompanying paper, where the plaquette-chain problem maps to an effective Ising model. They are useful for exact diagonalization, entanglement-entropy calculations, and Pauli-string inspection in small systems.

## Implemented Hamiltonian

With default couplings `J = 1` and `hx = 1`, the code implements the Ising-form Hamiltonian

\[
H = \sum_i \left( J Z_i Z_{i+1} - 2J Z_i + h_x \, \nu_i X_i \right),
\qquad
\nu_i = \frac{Z_{i-1} + Z_{i+1}}{\sqrt{2}}.
\]

Periodic boundary conditions are used by default in the multi-spin code.

---

## File 1: `Hamiltonian.py`

The original 2-spin example works in the computational basis
`|00>, |01>, |10>, |11>`.

### What it does
1. enumerates the full basis of `{0,1}^2`,
2. constructs the Hamiltonian matrix `H` by evaluating matrix elements,
3. enforces Hermiticity via
   \[
   H \leftarrow \frac{H + H^\dagger}{2},
   \]
4. diagonalizes `H` using `numpy.linalg.eigh`,
5. for each eigenvector \(\psi\), computes bipartite entanglement entropy between qubit 0 and qubit 1:
   - reshape \(\psi\) into a `2×2` amplitude matrix \(\Psi\),
   - compute \(\rho_A = \Psi \Psi^\dagger\),
   - compute
     \[
     S(A) = -\mathrm{tr}(\rho_A \log \rho_A),
     \]
     with default log base 2, so entropy is reported in **bits**, 
6. expands `H` in the Pauli-string basis using
   \[
   c_P = \frac{1}{2^n}\,\mathrm{tr}(H P).
   \]

### Example output

![Entanglement entropy example](./final%20results.png)

---

## File 2: `Hamiltonian_multispin.py`

This is the generalized version of the original mapper, extended from 2 spins to an arbitrary number of spins/qubits.

### What it does
1. enumerates the full computational basis `{0,1}^N`,
2. builds the Hamiltonian matrix from matrix elements,
3. diagonalizes the Hamiltonian exactly,
4. computes bipartite entanglement entropy for **arbitrary subsystems**,
5. optionally prints a contiguous-cut entanglement profile `S(L_A)` for `L_A = 1, ..., N-1`,
6. optionally expands the Hamiltonian in a Pauli-string basis.

### Key features
- arbitrary system size `N >= 2`
- periodic boundary conditions by default
- optional open boundaries via CLI flag
- entropy for any subsystem, e.g. `--subsystem 0 1`
- optional Page-curve-style contiguous-cut profiles
- optional Hamiltonian / eigenvector / Pauli-decomposition printing

### Example commands

Run a 4-spin system and compute default half-chain entanglement:

```bash
python Hamiltonian_multispin.py --n-sites 4
```

Run a 5-spin system and compute the entropy of qubits 0 and 1 versus the rest:

```bash
python Hamiltonian_multispin.py --n-sites 5 --subsystem 0 1
```

Show contiguous-cut entanglement profiles for each eigenstate:

```bash
python Hamiltonian_multispin.py --n-sites 4 --show-profile
```

Use open instead of periodic boundary conditions:

```bash
python Hamiltonian_multispin.py --n-sites 4 --open-boundary
```

Print a Pauli-string decomposition:

```bash
python Hamiltonian_multispin.py --n-sites 4 --decompose-pauli
```

---

## Notes and current scope

- The current code is best viewed as a **small-system exact-diagonalization toy model**.
- The multi-spin script is still in the **Ising / `j_max = 1/2` limit**, not the full `j_max > 1/2` SU(2) gauge-theory Hilbert space.
- Because the Hilbert-space dimension grows as `2^N`, exact diagonalization becomes expensive quickly as `N` increases.

## Repository contents

- `Hamiltonian.py` — original 2-spin demo
- `Hamiltonian_multispin.py` — generalized multi-spin version
- `Page curves.pdf` — reference paper / background material
- `final results.png` — sample output from the 2-spin script

