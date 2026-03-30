# Advanced methods in PhysicsNemo

## Custom PDEs

### Continuity Equation
We will simply try to solve the 1+1D continuity equation, to see the weird behaviour in vanilla PINNs, as described in [1], §6.1.

\begin{align}
\partial_t u + \beta \partial_x u &= 0 \,, \quad (t,x) \in [0,1] \times [-1, +1] \\
u(0, x) &= - \sin \frac\pi x .
\end{align}

### Allen-Cahan Equation

The Allen-Cahan equation in 1+1D is a _non-linear reaction-diffusion_ equation and describes the phase separation in multi-component alloy systems:
$$ \partial_t u + \rho u (u^2 - 1) - \nu \partial_x^2 u = 0 $$

we will try to solve
\begin{align}
\partial_t u + \rho u (u^2 - 1) - \nu \partial_x^2 u &= 0 \,, \quad (t,x) \in [0,1] \times [-1, +1] \\
u(0, x) &= x^2 \cos (\pi x) .
\end{align}
with $\nu=0.0001$ and $\rho = 5$.

See [1], §6.3.



## Optimisers

We will see how to use different optimisers in `PhysicsNeMo`, simply by changing a keyword in the `config.yaml` file in the `conf` directory.

## Fourier Network and its variation

## Custom Nodes & Mixture-of-Experts


------

#### References

[1] Simone Monaco, Daniele Apiletti, Training physics-informed neural networks: One learning to rule them all?, Results in Engineering, Volume 18, 2023, https://www.sciencedirect.com/science/article/pii/S2590123023001500?via%3Dihub#se0130