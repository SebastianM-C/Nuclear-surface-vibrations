# Fingerprints of global classical phase-space structure in quantum spectra

This code reproduces the results of the following article:
**Fingerprints of global classical phase-space structure in quantum spectra**,
*S. Micluta-Campeanu, M.C. Raportaru, A.I. Nicolin, V. Baran*,
Rom. Rep. Phys. **70**, 105 (2018)

The code is organized in two parts, corresponding to the simulations for
classical and quantum chaos.

- The classical mechanics code simulates the dynamics of the nuclear surface
modeled by the following Hamiltonian:

$$
H_{cl} = \frac{A}{2}(p_0^2+p_2^2)+\frac{A}{2}(q_0^2+q_2^2)
+\frac{B}{\sqrt{2}}q_0(3q_2^2-q_0^2)
+\frac{D}{4}(q_0^2+q_2^2)^2
$$

The equations of motion are numerically integrated in order to build Poincare maps.
In order to measure the degree of chaos for a given set of parameters at a given energy,
the maximal Lyapunov exponent is computed for each trajectory.

- The quantum mechanics code simulates the quantized version of the system by
diagonalizing the Hamiltonian expressed in the basis of a double harmonic oscillator

$$
\begin{split}
  H_B &= A \left( a_1^\dagger a_1 + a_2^\dagger a_2 \right)
    + \frac{B}{4} \bigg[ \left( 3 a_1^\dagger {a_2^\dagger}^2 + 3 a_1 a_2^2
                               - {a_1^\dagger}^3 - a_1^3 \right)   \\
  &\quad + 3 \left( a_1 {a_2^\dagger}^2 + a_1^\dagger a_2^2 - a_1^\dagger a_1^2 - {a_1^\dagger}^2 a_1
             + 2 a_1 a_2^\dagger a_2 + 2 a_1^\dagger a_2^\dagger a_2
          \right) \bigg]  \\
  &\quad + \frac{D}{16} \bigg[ 6 \left( {a_1^\dagger}^2 a_1^2 + {a_2^\dagger}^2 a_2^2 \right)
                        + 2 \left( a_1^2 {a_2^\dagger}^2 + {a_1^\dagger}^2 a_2^2 \right)
                        + 8 a_1^\dagger a_1 a_2^\dagger a_2  \\
  &\quad + 4 \left(a_1^\dagger a_1^3 + {a_1^\dagger}^3 a_1 + a_2^\dagger a_2^3 + {a_2^\dagger}^3 a_2
     + a_1^2 a_2^\dagger a_2 + {a_1^\dagger}^2 a_2^\dagger a_2 + a_1^\dagger a_1 a_2^2 + a_1^\dagger a_1 {a_2^\dagger}^2
        \right)  \\
  &\quad + \left( {a_1^\dagger}^4 + a_1^4 + {a_2^\dagger}^4 + a_2^4
     + 2 {a_1^\dagger}^2 {a_2^\dagger}^2 + 2 a_1^2 a_2^2
      \right)
                        \bigg].
\end{split}
$$

In order to verify the accuracy of the computed eigenvalues, two different dimensions
of the basis are taken and only the eigenvalues that are within $\delta$ of each other
are considered. The selected eigenvalues are then separated in the 3 irreducible
representations of the $C_{3v}$ symmetry group, corresponding to the symmetry of the
system. The parameter $\epsilon$ is used as the maximum numerical error for the
degenerated energy levels. The next step is to create the nearest neighbor
spacing distributions. The contributions of the 3 irreducible representations
are summed. The resulting histograms are fitted with a probability distribution
corresponding to a superposition of a Poisson distribution and a Wigner one

\[
P(s) = \alpha P_P(s) + (1-\alpha) P_W(s)
\]

and the $\alpha$ coefficient is obtained.  
