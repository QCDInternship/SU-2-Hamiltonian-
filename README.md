# SU-2-Hamiltonian-
Entanglement entropy of (2+1)-dimensional SU(2) lattice gauge theory  on plaquette chains


The Hamiltonian implemented (with default couplings `J = 1`, `hx = 1`) is:
- Constructed via matrix elements in the computational basis: `|00>, |01>, |10>, |11>`
- Diagonalized to obtain eigenvalues/eigenvectors
- Each eigenvector is reshaped into a 2×2 state matrix to compute reduced density matrices and entanglement entropy

THE CODE:
`Hamiltonian.py`:
1. enumerates the full basis of `{0,1}^2`,
2. constructs the Hamiltonian matrix `H` by evaluating matrix elements,
3. enforces Hermiticity via \(H \leftarrow (H + H^\dagger)/2\),
4. diagonalizes `H` using `numpy.linalg.eigh`,
5. for each eigenvector \(\psi\), computes bipartite entanglement entropy between qubit 0 and qubit 1:
   - reshape \(\psi\) into a `2×2` amplitude matrix \(\Psi\),
   - compute \(\rho_A = \Psi\Psi^\dagger\),
   - compute \(S(A) = -\mathrm{tr}(\rho_A \log \rho_A)\) (default log base 2 → “bits”),
6. expands `H` in the Pauli-string basis using
   \[
   c_P = \frac{1}{2^n}\,\mathrm{tr}(H P).
   \]


SAMPLE RESULTS:

![Entanglement entropy example](./final%20results.png)

