import torch
import torch.nn as nn

class Heat1D_BC:
    """
    Boundary and initial conditions for the 1D heat equation:
        BC: u(t, x=±1) = 0
        IC: u(x, t=0) = cos(pi x / 2)
    """
    def __init__(self, cost_function=None):
        self.cost_function = cost_function or nn.MSELoss()
        self.pi_fact = float(torch.pi / 2)

    def boundary_cond(self, coords: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        Boundary condition:
            u(t, x=±1) = 0
        """
        true_val = torch.zeros_like(pred)
        return self.cost_function(pred, true_val)

    def initial_cond(self, coords: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        Initial condition:
            u(x, t=0) = cos(pi x / 2)
        """
        x = coords[:, 1:2]  # keep shape (N,1)
        true_val = torch.cos(self.pi_fact * x)

        return self.cost_function(pred, true_val)
