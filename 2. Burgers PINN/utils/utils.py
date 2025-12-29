import numpy as np
import torch

def exact_solution_func(coords: torch.Tensor, diffusion_param: float = 0.1) -> torch.Tensor:
    """
    Method to generate the exact solution function.
    """
    pi_fact = float(torch.pi/2)
    return torch.exp(- (pi_fact**2)*(diffusion_param**2)*coords[:, 0]) * torch.cos(pi_fact * coords[:, 1])

def exact_solution_func_np(T: np.array, X: np.array, diffusion_param: float = 0.1) -> np.array:
    """
    Method to generate the exact solution function.
    """
    pi_fact = float(np.pi/2)
    return np.exp(- (pi_fact**2)*(diffusion_param**2)*T) * np.cos(pi_fact * X)

class R2_extraction:
    def __init__(
        self,
        dimension: int = 2 ,
    ):
        self.d = dimension
        self.phi_dict = {}

    def r_d_extraction(self, N_points: int):
        """
        Method to perform the quasi-random R_d extraction;
        see https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
        """
        d = self.d
        g = self.phi(d)
        alpha = np.zeros(d)
        for j in range(d):
            alpha[j] = pow(1/g, j+1) %1
        z = np.zeros((N_points, d))
        # This number can be any real number.
        # Common default setting is typically seed=0
        # But seed = 0.5 is generally better.
        seed = 0.5 * np.random.rand(1)[0]
        for i in range(N_points):
            z[i] = (seed + alpha*(i+1))

        z = z %1
        return torch.tensor(z)

    def phi(self, d: int):
        """
        Compute the golden number in d dims;
        it then stores it into a dict for later use.
        """
        _key = f"{d}"
        if _key not in self.phi_dict:
            x=2.0000
            for i in range(10):
                x = pow(1+x,1/(d+1))

            self.phi_dict[_key] = x
        return self.phi_dict[_key]