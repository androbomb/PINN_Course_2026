# Advanced methods in PhysicsNemo

## Use PhysicsNeMo in notebooks

Bulding on [this](https://github.com/NVIDIA/physicsnemo-sym/blob/2bb155bf0dab55adb45c624e2e0fd6979b57c6c6/docs/user_guide/notebook/notebook.ipynb), we will see how to use PhysicsNeMo also in notebooks for prototyping purposes. 



## Custom PDEs

### Continuity Equation
We will simply try to solve the 1+1D continuity equation, to see the weird behaviour in vanilla PINNs, as described in [1], §6.1.

\begin{align}
\partial_t u + \beta \partial_x u &= 0 \,, \quad (t,x) \in [0,1] \times [-1, +1] \\
u(0, x) &= - \sin \frac\pi x  \,,\\
u(t, x=+1) &= u(t, x=-1).
\end{align}

Notice that, w.r.t. the conventions used in the paper, we adimensionalised the problem via the change of variable $x\to (x - \pi)/(2\pi)$, and the redefinition $\beta \to \beta /(2\pi)$. 

### Allen-Cahan Equation

The Allen-Cahan equation in 1+1D is a _non-linear reaction-diffusion_ equation and describes the phase separation in multi-component alloy systems:
$$ \partial_t u + \rho u (u^2 - 1) - \nu \partial_x^2 u = 0 $$

we will try to solve
\begin{align}
\partial_t u + \rho u (u^2 - 1) - \nu \partial_x^2 u &= 0 \,, \quad (t,x) \in [0,1] \times [-1, +1] \\
u(0, x) &= x^2 \cos (\pi x) \\,
u(t, x=+1) &= u(t, x=-1). 
\end{align}
with $\nu=0.0001$ and $\rho = 5$.

See [1], §6.3.


## Optimisers

We will see how to use different optimisers in `PhysicsNeMo`, simply by changing a keyword in the `config.yaml` file in the `conf` directory.

The loss optimizer group contains the supported optimizers that can be used in PhysicsNeMo Sym which includes ones that are built into `PyTorch <https://pytorch.org/docs/stable/optim.html#algorithms>`_ as well as from `Torch Optimizer <https://github.com/jettify/pytorch-optimizer>`_ package.
Some of the most commonly used optimizers include:

- ``adam``: ADAM optimizer
- ``sgd``: Standard stochastic gradient descent
- ``rmsprop``: The RMSProp algorithm
- ``adahessian``: Second order stochastic optimization algorithm
- ``bfgs``: L-BFGS iterative optimization method

as well as these more unique optimizers:
``a2grad_exp``, ``a2grad_inc``, ``a2grad_uni``, ``accsgd``, ``adabelief``, ``adabound``, 
``adadelta``, ``adafactor``, ``adagrad``, ``adamax``, ``adamod``, ``adamp``, ``adamw``, ``aggmo``, 
``apollo``, ``asgd``, ``diffgrad``, ``lamb``, ``madgrad``, ``nadam``, ``novograd``, ``pid``, ``qhadam``, ``qhm``, ``radam``, 
``ranger``, ``ranger_qh``, ``ranger_va``, ``rmsprop``, ``rprop``, ``sgdp``, ``sgdw``, ``shampoo``, ``sparse_adam``,  ``swats``, ``yogi``.



## Balancing Losses

The `loss` config group is used to select different loss aggregations that are supported by PhysicsNeMo Sym.
A loss aggregation is the method used to combine the losses from different constraints.

Different methods can yield improved performance for some problems.

- ``sum``: Simple summation aggregation (default)
- ``grad_norm``: Gradient normalization for adaptive loss balancing
- ``homoscedastic``: Homoscedastic loss
- ``soft_adapt``: Adaptive loss weighting
- ``relobralo`` : Relative loss balancing with random lookback


## Fourier Network and its variation

Currently the architectures of the Fourier family that are shipped internally in PhysicsNeMo Sym that have a configuration group include:

- ``fourier_net``: Fourier neural network
- ``highway_fourier``: Fourier neural network with adaptive gating units 
- ``modified_fourier``:  Fourier neural network with two layers of Fourier features 
- ``multiplicative_fourier``: Fourier feature neural network with frequency connections
- ``multiscale_fourier``:  Multi-scale Four
- 

## Custom Nodes & Mixture-of-Experts


------

#### References

[1] Simone Monaco, Daniele Apiletti, Training physics-informed neural networks: One learning to rule them all?, Results in Engineering, Volume 18, 2023, https://www.sciencedirect.com/science/article/pii/S2590123023001500?via%3Dihub#se0130

[NvidiaPhysicsNemo_UserGuide] https://github.com/NVIDIA/physicsnemo-sym/tree/2bb155bf0dab55adb45c624e2e0fd6979b57c6c6/docs/user_guide 