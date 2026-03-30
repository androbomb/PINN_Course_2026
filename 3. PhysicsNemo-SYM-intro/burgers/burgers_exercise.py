import os
import warnings

import pandas as pd
import numpy as np
import torch

from sympy import Symbol, Eq, Abs, And, Or, Xor, Function, Number
from sympy import atan2, pi, sin, cos

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_1d import Line1D
from physicsnemo.sym.geometry.primitives_2d import Rectangle, Circle
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.key import Key
from physicsnemo.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)

from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.models.activation import Activation
from physicsnemo.sym.eq.pde import PDE

class Burgers(PDE):
    """
    Burgers equation 1D
        u_tt + u u_x - D^2 u_xx = 0

    Parameters
    ==========
    D : float, string
        Diffusion coefficient. If a string then the
        Diffusion  is input into the equation.
    """

    name = "Burgers"

    def __init__(self, u: str = 'u', D: float = 0.5):
        ...

        # set equations
        self.equations = {}
        self.equations["burgers"] = 
        
def get_model():
    flow_net = ...
    
    return flow_net

@physicsnemo.sym.main(config_path="conf", config_name="config_burgers")
def run(cfg: PhysicsNeMoConfig) -> None:
    # MACRO PARAMS
    _ell = 1.0
    _diffusion_coefficient = np.sqrt( 0.01/float(np.pi) )
    _t_f = 1.0
    # ====== PDE ===========================
    # make list of nodes to unroll graph on
    pde = Burgers(u="u",  D =_diffusion_coefficient)

    # ====== MODEL ===========================
    flow_net = get_model()
    # make nodes
    nodes = ...
    # ====== Geometry ===========================
    # vars
    x, t_symbol = Symbol("x"), Symbol("t")
    time_range = {t_symbol: (0, _t_f)}
    # geo
    geo_1D = ...
    # ====== Domain ===========================
    # make diamond domain
    domain = Domain()   # <====== DOMAIN instance =======
    # Interior
    interior = ...
    domain.add_constraint(interior, "interior")
    # BC
    BC = ...
    domain.add_constraint(BC, "BC")
    # initial condition
    IC = ...
    domain.add_constraint(IC, "IC")    
      
    # ====== inferencer ===========================
    # add inferencer data
    deltaT = 0.01
    deltaX = 0.01
    x = np.arange(-_ell, +_ell, deltaX)
    t = np.arange(0, _t_f, deltaT)
    X, T = np.meshgrid(x, t)
    X = np.expand_dims(X.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)
    invar_numpy  = {"x": X, "t": T}
    
    grid_inference = ...
    domain.add_inferencer(grid_inference, "inf_data")
    
    # ====== Solver ===========================
    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()
    
    
if __name__ == "__main__":
    run()