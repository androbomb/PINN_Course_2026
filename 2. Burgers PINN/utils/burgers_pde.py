import torch

class Burgers_PDE:
    """
    Computes the PDE residual for the 1D Burgers equation:
        u_t + u u_x - D^2 u_xx = 0
    using the conservative form
        u_t + 1/2 (u**2)_x - D^2 u_xx = 0
    """
    def __init__(self, D: float = 0.5):
        self.D = D

    def compute_pde(self, coords: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        coords: (N, 2) tensor with columns [t, x]
        u:      (N, 1) predicted solution u(t,x)
        """
        # First derivatives wrt (t, x)
        grads = torch.autograd.grad(
            u, coords,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        u_t = grads[:, 0:1]   # shape (N,1)
        u_x = grads[:, 1:2]

        # Second derivative wrt x
        grads2 = torch.autograd.grad(
            u_x, coords,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0]

        u_xx = grads2[:, 1:2]

        # 
        u2_x = torch.autograd.grad(
            u.pow(2), coords,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0][:, 1:2] / 2

        # PDE residual
        return u_t + u2_x - (self.D ** 2) * u_xx
