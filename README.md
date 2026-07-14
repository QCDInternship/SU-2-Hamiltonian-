# Entanglement Entropy in the Truncated SU(2) Plaquette Chain

## Overview

This repository studies entanglement entropy in a truncated Hamiltonian formulation of SU(2) lattice gauge theory in two spatial dimensions and one time dimension.

The main focus is the electric flux truncation

```math
j_{\max}=\frac{1}{2}.
```

At this truncation, the physical plaquette chain can be represented by an effective spin chain.

The repository contains numerical tools for:

1. Exact diagonalisation

2. Sparse diagonalisation

3. Translation symmetry reduction

4. Page curve calculations

5. Quantum many body scar searches

6. Matrix product operator construction

7. Density matrix renormalisation group calculations

8. QVAE based anomaly detection

The project is inspired by the paper:

Lukas Ebner, Andreas Schäfer, Clemens Seidl, Berndt Müller, and Xiaojun Yao, **Entanglement entropy of (2 + 1) dimensional SU(2) lattice gauge theory on plaquette chains**, Physical Review D 110, 014505 (2024).

DOI: [10.1103/PhysRevD.110.014505](https://doi.org/10.1103/PhysRevD.110.014505)

## Physical model

The Kogut Susskind Hamiltonian contains electric and magnetic contributions.

```math
H=
\frac{g^2}{2}
\sum_L
\left(E_i^a\right)^2
-
\frac{2}{a^2g^2}
\sum_P
\mathrm{Tr}
\left[
\prod_{(n,\hat{i})\in P}
U(n,\hat{i})
\right].
```


## Effective Ising limit

This repository concentrates on the truncation

```math
j_{\max}=\frac{1}{2}.
```

At this cutoff, each effective plaquette degree of freedom can be represented by a binary spin variable.

The resulting Hamiltonian has the form

```math
H=
\sum_i
\left[
JZ_iZ_{i+1}
-
2JZ_i
+
h_x\nu_iX_i
\right].
```

The diagonal terms represent the effective electric contribution.

The terms containing the spin flip operator $X_i$ represent magnetic transitions between effective plaquette configurations.

The coupling parameters used in the reference paper are related to the lattice parameters by

```math
J=-\frac{3g^2}{16},
```

and

```math
h_x=\frac{1}{(ag)^2}.
```

## Basis convention

The canonical implementation uses ordinary binary ordering.

Site $0$ is the leftmost and most significant bit.

The bit to spin mapping is

```math
0\longleftrightarrow Z=+1,
\qquad
1\longleftrightarrow Z=-1.
```

For two sites, the ordered computational basis is

```math
|00\rangle,
\quad
|01\rangle,
\quad
|10\rangle,
\quad
|11\rangle.
```

For $N$ effective spins, the Hilbert space dimension is

```math
D=2^N.
```

The computational basis used in the programs is the basis of the effective Ising model. It should not be confused with the unreduced SU(2) link basis.

## Repository structure

### `ising_limit_model.py`

This is the canonical implementation of the effective Ising limit Hamiltonian.

It provides:

1. Parameter validation

2. Computational basis indexing

3. Diagonal energy evaluation

4. Neighbour dependent spin flip amplitudes

5. Dense Hamiltonian construction

6. Sparse Hamiltonian construction

7. Matrix free Hamiltonian application

8. Exact diagonalisation

9. Eigenvalue and eigenvector analysis

10. Reduced density matrix construction

11. Entanglement entropy calculations

12. Pauli term generation

13. Pauli decomposition and reconstruction

14. Command line inspection of the model

The other Ising limit programs use this module rather than defining independent Hamiltonian formulas.

Run the default model summary with

```bash
python ising_limit_model.py
```

Display all available options with

```bash
python ising_limit_model.py --help
```

Display the complete two site analysis with

```bash
python ising_limit_model.py \
  --n-sites 2 \
  --J 1 \
  --hx 1 \
  --mode toy \
  --legacy-two-site-output
```

### `Hamiltonian.py`

This is the original two site educational demonstration.

It performs the following operations:

1. Lists the computational basis

2. Constructs the four dimensional Hamiltonian

3. Diagonalises the Hamiltonian

4. Prints all eigenvalues and eigenvectors

5. Calculates the entanglement entropy of each eigenstate

6. Displays the Pauli decomposition

The canonical Hamiltonian physics is supplied by `ising_limit_model.py`.

Run it with

```bash
python Hamiltonian.py
```

### `Hamiltonian_multispin_sparse.py`

This program constructs the effective Hamiltonian in sparse form.

It supports selected eigenpair calculations using SciPy and provides a matrix free Lanczos workflow.

For $N$ spins, a dense Hamiltonian contains order

```math
4^N
```

matrix entries.

The sparse Hamiltonian contains approximately order

```math
N2^N
```

nonzero entries.

The sparse representation increases the accessible system size without changing the physical Hamiltonian.

View the available options with

```bash
python Hamiltonian_multispin_sparse.py --help
```

### `page_curve_k0.py`

This program constructs the periodic zero momentum sector using translation orbits.

It performs the following operations:

1. Generates translation orbits of computational basis states

2. Constructs the zero momentum Hamiltonian

3. Solves selected eigenstates using dense or sparse methods

4. Reconstructs complete computational basis statevectors

5. Calculates entanglement entropy for different subsystem sizes

6. Selects states near chosen excitation energies

7. Saves Page curve plots and numerical output

For a pure state, the Page curve is symmetric under

```math
N_A\longleftrightarrow N-N_A.
```

A representative result is shown below.

![Periodic zero momentum Page curve](page_curve_k0.png)

View the available options with

```bash
python page_curve_k0.py --help
```

### `k0_finite_size_scan.py`

This program repeats the zero momentum calculation for several chain lengths.

It is used to compare selected states across different system sizes and to investigate whether low entropy behaviour persists as $N$ increases.

The program collects quantities such as:

1. Chain length

2. Reduced Hilbert space dimension

3. Selected excitation energy

4. Half chain entropy

5. Entropy anomaly measures

6. Solver residuals

View the available options with

```bash
python k0_finite_size_scan.py --help
```

### `scar_state_search.py`

This program searches the truncated spectrum for highly excited states with unusually low entanglement entropy.

The analysis follows these steps:

1. Construct the periodic Hamiltonian

2. Diagonalise the full spectrum or a selected spectral region

3. Calculate half chain entanglement entropy

4. Compare each eigenstate with nearby states in energy

5. Assign an anomaly score

6. Rank low entropy candidates

7. Save candidate tables and plots

8. Inspect wavefunction components of selected states

A low entropy outlier is treated as a scar candidate. Low entropy alone is not sufficient to establish a quantum many body scar.

View the available options with

```bash
python scar_state_search.py --help
```

### `qvae_scar_detector.py`

This program applies a QVAE based anomaly detection workflow to scar candidate data.

The detector is designed to learn features of typical eigenstates and assign larger reconstruction based anomaly scores to unusual states.

The QVAE output is used for candidate prioritisation.

A final physical interpretation requires additional checks involving:

1. Entanglement entropy

2. Nearby eigenstates

3. Subsystem size scaling

4. Finite size scaling

5. Wavefunction components

6. Dynamical recurrence

### `run_qvae_scar_detector.py`

This is the command line runner for the QVAE workflow.

It connects stored spectral data to model training, evaluation, and candidate ranking.

View the available options with

```bash
python run_qvae_scar_detector.py --help
```

### `ising_limit_mpo.py`

This module converts the canonical Pauli terms into a TeNPy matrix product operator.

It does not define an independent Hamiltonian.

The operator content is obtained from `ising_limit_model.py`.

The program also verifies that the TeNPy local operators correspond to the Pauli matrices used by the canonical Hamiltonian.

### `dmrg_ground_state.py`

This program uses finite two site density matrix renormalisation group calculations to approximate the ground state of the open boundary toy Hamiltonian.

The program constructs an initial matrix product state and optimises it using repeated density matrix renormalisation group sweeps.

It reports:

1. Ground state energy

2. Energy per site

3. Convergence information

4. Number of sweeps

5. Maximum bond dimension

6. Truncation error history

7. Local $Z$ expectation values

8. Local $X$ expectation values

9. Nearest neighbour correlations

10. Bond entanglement entropies

11. Half chain entropy

12. Central Schmidt values

13. Energy variance when supported by the installed TeNPy version

View the available options with

```bash
python dmrg_ground_state.py --help
```

### `validate_dmrg.py`

This program compares density matrix renormalisation group results with exact diagonalisation for small open chains.

It checks:

1. Ground state energy

2. Energy per site

3. Half chain entanglement entropy

4. Local $Z$ expectation values

5. Local $X$ expectation values

6. Nearest neighbour $ZZ$ correlations

The validation compares

```math
E_{\mathrm{exact}}
```

with

```math
E_{\mathrm{DMRG}},
```

and compares

```math
S_{\mathrm{exact}}
```

with

```math
S_{\mathrm{DMRG}}.
```

Agreement between these results validates the matrix product operator and density matrix renormalisation group pipeline.

Run the validation with

```bash
python validate_dmrg.py
```

## Entanglement entropy

For a pure state $|\psi\rangle$, the density matrix is

```math
\rho=
|\psi\rangle\langle\psi|.
```

For a subsystem $A$, the reduced density matrix is

```math
\rho_A=
\mathrm{Tr}_{A^c}
\left(
|\psi\rangle\langle\psi|
\right).
```

The von Neumann entropy is

```math
S_A=
-\mathrm{Tr}
\left(
\rho_A\log\rho_A
\right).
```

If the eigenvalues of the reduced density matrix are $\lambda_\alpha$, the entropy can also be written as

```math
S_A=
-\sum_\alpha
\lambda_\alpha
\log\lambda_\alpha.
```

For a pure global state,

```math
S_A=S_{A^c}.
```

This equality produces a Page curve that is symmetric around half system size.

## Page curve behaviour

The expected qualitative behaviour depends on the energy of the state.

### Ground states

The ground state of a gapped system is expected to follow an area law.

Its entropy approaches a nearly constant value when the subsystem is much larger than the correlation length and much smaller than the complete system.

A representative area law fitting function is

```math
S_{\mathrm{area}}(N_A)
=
b_0
-
b_1
\left[
e^{-N_A/\ell_{\mathrm{corr}}}
+
e^{-(N-N_A)/\ell_{\mathrm{corr}}}
\right].
```

Here, $\ell_{\mathrm{corr}}$ is the correlation length.

### Highly excited states

Highly excited thermal states are expected to follow a volume law.

For subsystem sizes below half of the system,

```math
S_{\mathrm{vol}}(N_A)=sN_A,
\qquad
N_A\leq\frac{N}{2}.
```

For subsystem sizes above half of the system,

```math
S_{\mathrm{vol}}(N_A)=s(N-N_A),
\qquad
N_A\geq\frac{N}{2}.
```

Here, $s$ is the entropy density.

### Scar candidates

A scar candidate may appear as a highly excited state whose entropy is substantially lower than that of nearby states.

A convincing scar analysis should investigate several properties together:

1. The state is located away from the spectral boundaries

2. Its entropy is anomalously low compared with nearby states

3. Its behaviour persists across system sizes

4. Its Page curve grows more slowly than the volume law

5. Its wavefunction has enhanced overlap with special basis states

6. Its dynamical recurrence differs from that of nearby thermal states

The scar search and QVAE programs provide candidate rankings. They do not by themselves establish the existence of a quantum many body scar.

## Tensor network workflow

The tensor network calculation follows this sequence:

```text
canonical Hamiltonian
        |
        v
Pauli representation
        |
        v
matrix product operator
        |
        v
matrix product state
        |
        v
density matrix renormalisation group optimisation
```

The density matrix renormalisation group implementation is currently used for finite open chains.

For small chains, the result is checked directly against exact diagonalisation.

For larger chains, the matrix product state representation avoids storing the complete statevector of dimension

```math
2^N.
```

The efficiency of the method depends on the entanglement structure of the target state and the required bond dimension.


## Software dependencies

The core numerical programs use:

1. Python

2. NumPy

3. SciPy

4. Matplotlib

5. pandas

The matrix product operator and density matrix renormalisation group programs require TeNPy.

The QVAE programs require the machine learning and quantum computing libraries imported by those files.


The next major physics extension is the construction of a genuine higher representation SU(2) Hamiltonian with gauge invariant basis states and magnetic matrix elements beyond the effective Ising limit.

## Author

Subhashree Rameshwaram Kumaresan

