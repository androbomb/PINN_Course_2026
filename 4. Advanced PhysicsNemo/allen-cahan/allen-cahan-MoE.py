import os
import warnings

from typing import Optional, Dict, Tuple, Union, List

import pandas as pd
import numpy as np
import torch
from torch import nn as nn

from torch import Tensor
from typing import Dict
import numpy as np

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

from physicsnemo.sym.graph import Graph
from physicsnemo.sym.domain.constraint import Constraint

from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.models.activation import Activation

from physicsnemo.sym.eq.pde import PDE

from physicsnemo.sym.node import Node

import matplotlib.pyplot as plt
import scipy.interpolate

class AllenCahn(PDE):
    """
    Allen - Cahn equation 1D
        u_t + ρ u (u^2 - 1) - ν u_xx = 0

    Parameters
    ==========
    ρ  : float, string
        Reaction-Diffusion coeff
    ν  : float, string
        Diffusion coefficient. If a string then the
        Diffusion  is input into the equation.
    """

    name = "AllenCahn"

    def __init__(self, u: str = 'u', ν: float = 0.00001, ρ: float = 5.0):
        # coordinates
        x = Symbol("x")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "t": t}

        # make u function
        u = Function("u")(*input_variables)

        # wave speed coefficient
        if type(ν ) is str:
            ν  = Function(ν)(*input_variables)
        elif type(ν) in [float, int]:
            ν  = Number(ν)
            
        if type(ρ) is str:
            ρ = Function(ρ)(*input_variables)
        elif type(ρ) in [float, int]:
            ρ = Number(ρ)

        # set equations
        self.equations = {}
        self.equations["allencahn"] = u.diff(t, 1) + ρ * u*(u**2 - 1) - (ν * u.diff(x)).diff(x)

def get_model(
    input_keys  = [Key("x"), Key("t")],
    output_keys = [Key("u")],
    # Arch Specs
    layer_size = 512,
    nr_layers  = 4, 
    skip_connections     = True, 
    adaptive_activations = False, 
    activation_fn = Activation.SILU, 
    periodicity: Union[Dict[str, Tuple[float, float]], None] = None,
):
    flow_net = FullyConnectedArch(
        input_keys  = input_keys ,
        output_keys = output_keys ,
        # Arch Specs
        layer_size = layer_size,
        nr_layers  = nr_layers, 
        skip_connections     = skip_connections, 
        adaptive_activations = adaptive_activations, 
        activation_fn = activation_fn, 
        periodicity   = periodicity, 
    )
    
    return flow_net

class ComputeU(nn.Module):
    def __init__(self, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
    
    def forward(
        self, 
        in_vars: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        _sum_tot = 0
        for idx in range(1, self.n_experts+1):
            _sum_tot += in_vars[f"lambda_{idx}"] * in_vars[f"u_{idx}"]
        return {
            'u': _sum_tot
        }
    
class CustomInferencerPlotter(InferencerPlotter):
    def __init__(self, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        
    def __call__(self, invars, outvars):
        x,t = invars["x"][:,0], invars["t"][:,0]
        extent = (x.min(), x.max(), t.min(), t.max())
        
        # make plot
        fig, axs = plt.subplots(ncols = self.n_experts, nrows=2, figsize=(16,4), dpi=150,  constrained_layout=True)
        for idx in range(self.n_experts):
            # interpolate
            _u, _lambda = self.interpolate_output(
                x,t, 
                [ outvars[f"u_{idx+1}"][:,0] , outvars[f"lambda_{idx+1}"][:,0]], 
                extent
            )
            # U
            axs[0, idx].imshow(_u.T , origin="lower", extent=extent)
            axs[0, idx].set_title(f"Pred Exp u_{idx+1}")
            axs[0, idx].set_xlabel('x')
            axs[0, idx].set_ylabel('t')
            # Lambda
            axs[1, idx].imshow(_lambda.T , origin="lower", extent=extent)
            axs[1, idx].set_title(f"Import Exp l_{idx+1}")
            axs[1, idx].set_xlabel('x')
            axs[1, idx].set_ylabel('t')
        
        return [(fig, "custom_plot"),]
    
    @staticmethod
    def interpolate_output(x, y, us, extent):
        "Interpolates irregular points onto a mesh"

        # define mesh to interpolate onto
        xyi = np.meshgrid(
            np.linspace(extent[0], extent[1], 100),
            np.linspace(extent[2], extent[3], 100),
            indexing="ij",
        )

        # linearly interpolate points onto mesh
        us = [scipy.interpolate.griddata(
            (x, y), u, tuple(xyi)
            )
            for u in us]

        return us

    
@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # MACRO PARAMS
    _ell = 1.0
    _t_f = 1.0
    
    _periodicity = {"x": (-_ell, +_ell)}
    # ====== PDE ===========================
    # make list of nodes to unroll graph on
    pde = AllenCahn(u="u",  ρ = 5.0, ν = 0.0001)
    
    # ====== MoE MODEL ===========================
    input_keys  = [Key("x"), Key("t")]
    output_keys = [Key("u")]
    
    # Experts 
    # from https://towardsdatascience.com/mixture-of-experts-for-pinns-moe-pinns-6520adf32438
    # Expert 1: 2 layers with 64 nodes each and tanh activation
    # Expert 2: 2 layers with 64 nodes each and sine activation
    # Expert 3: 2 layers with 128 nodes each and tanh activation
    # Expert 4: 3 layers with 128 nodes each and swish activation
    # Expert 5: 2 layers with 256 nodes each and swish activation # there is no swish in physicsnemo, but SILU is a swish with parameter = 1
    experts = [
        # Expert 1
        get_model(
            input_keys  = input_keys, 
            output_keys = [Key("u_1")],
            #
            layer_size = 64,
            nr_layers  = 2, 
            skip_connections     = False, 
            adaptive_activations = False, 
            activation_fn = Activation.TANH,
            periodicity   = _periodicity,
        ),
        # Expert 2
        get_model(
            input_keys  = input_keys, 
            output_keys = [Key("u_2")] ,
            #
            layer_size = 64,
            nr_layers  = 2, 
            skip_connections     = False, 
            adaptive_activations = False, 
            activation_fn = Activation.SIN,
            periodicity   = _periodicity,      
        ),
        # Expert 3
        get_model(
            input_keys  = input_keys, 
            output_keys = [Key("u_3")] ,
            #
            layer_size = 128,
            nr_layers  = 2, 
            skip_connections     = False, 
            adaptive_activations = False, 
            activation_fn = Activation.TANH,
            periodicity   = _periodicity,      
        ),
        # Expert 4
        get_model(
            input_keys  = input_keys, 
            output_keys = [Key("u_4")] ,
            #
            layer_size = 128,
            nr_layers  = 3, 
            skip_connections     = False, 
            adaptive_activations = False, 
            activation_fn = Activation.SILU,
            periodicity   = _periodicity,      
        ),
        # Expert 5
        get_model(
            input_keys  = input_keys, 
            output_keys = [Key("u_5")] ,
            #
            layer_size = 256,
            nr_layers  = 2, 
            skip_connections     = False, 
            adaptive_activations = False, 
            activation_fn = Activation.SILU,
            periodicity   = _periodicity,      
        )
    ]
    # Gate
    u_keys = [Key(f"u_{idx+1}") for idx, expert in enumerate(experts)]
    lambda_keys = [Key(f"lambda_{idx+1}") for idx, expert in enumerate(experts)]
    gate_net = get_model(
        input_keys  = input_keys, 
        output_keys = lambda_keys,
        #
        layer_size = 64,
        nr_layers  = 2, 
        skip_connections     = False, 
        adaptive_activations = False, 
        activation_fn = Activation.SILU,
        periodicity   = _periodicity,   
    )
    
    # ====== Custom Node for computing MoE ===========================
    compute_u_node =  Node(
        inputs  = [*u_keys, *lambda_keys],  
        outputs = ['u'], 
        evaluate = ComputeU(n_experts = 5) ,
        name = 'gating_node'
    )
    # make nodes
    nodes = [compute_u_node] + pde.make_nodes() + [gate_net.make_node(name="gate_network")] + [*[expert.make_node(name=f"expert-{idx}") for idx, expert in enumerate(experts) ] ]
    # ====== Geometry ===========================
    # vars
    x, t_symbol = Symbol("x"), Symbol("t")
    time_range = {t_symbol: (0, _t_f)}
    # geo
    geo_1D = Line1D(point_1 = -_ell, point_2 = +_ell)

    # ====== Importance measure ===========================
    # make importance model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    importance_model_graph = Graph(
        nodes,
        invar=input_keys,
        req_names=[
            Key("u", derivatives=[Key("x")]),
            Key("u", derivatives=[Key("t")]),
        ],
    ).to(device)

    def importance_measure(invar):
        """
        Methdo to conpute the importance measere via the physicsnemo of Electric field
        """
        outvar = importance_model_graph(
            Constraint._set_device(invar, device=device, requires_grad=True)
        )
        importance = (
              outvar["u__x"] ** 2
            + outvar["u__t"] ** 2 
        ) ** 0.5 + 10
        return importance.cpu().detach().numpy()
    
    # Lambda weighting
    C_T = 15.0
    lambda_t = C_T * (1 - t_symbol/_t_f) + 1

    # Initial condition
    ic_dict = {
        "u": (x**2) * cos(pi*x/2)
    }
    # ====== Domain ===========================
    # make diamond domain
    domain = Domain()   # <====== DOMAIN instance =======
    # Interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_1D,
        outvar={"allencahn": 0},
        batch_size=cfg.batch_size.Interior,
        lambda_weighting={
            "allencahn": lambda_t * Symbol("sdf"),
        },
        parameterization = time_range,
        #fixed_dataset = False, 
        importance_measure = importance_measure, 
    )
    domain.add_constraint(interior, "interior")

    # initial condition
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_1D,
        outvar=ic_dict,
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"u": 10.0},
        parameterization={t_symbol: 0.0},
    )
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
    
    grid_inference = PointwiseInferencer(
        nodes = nodes,
        invar = invar_numpy,
        output_names = ["u"],
        batch_size = 1024,
        plotter    = InferencerPlotter(),
    )
    domain.add_inferencer(grid_inference, "inf_data")
    
    # ====== additional inferencers ===========================
    expert_inferencer = PointwiseInferencer(
        nodes = nodes,
        invar = invar_numpy,
        output_names = [f"u_{idx+1}" for idx, _ in enumerate(experts) ]  + [f"lambda_{idx+1}" for idx, _ in enumerate(experts) ] ,
        batch_size = 1024,
        plotter    = CustomInferencerPlotter(n_experts=5),
    )
    domain.add_inferencer(expert_inferencer, "expert_inf_data")
    # ====== Solver ===========================
    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()
    
    
if __name__ == "__main__":
    run()