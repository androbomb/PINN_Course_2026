import os
import warnings

from typing import Optional, Dict, Tuple, Union, List

import pandas as pd
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

from scipy import io

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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import axes_grid1
import scipy.interpolate

from physicsnemo.sym.models.fourier_net import FourierNetArch
from physicsnemo.sym.models.siren import SirenArch
from physicsnemo.sym.models.modified_fourier_net import ModifiedFourierNetArch
from physicsnemo.sym.models.dgm import DGMArch

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
    model_type: str = 'FullyConnectedArch' ,
    # Arch Specs
    layer_size = 512,
    nr_layers  = 4, 
    skip_connections     = True, 
    adaptive_activations = False, 
    activation_fn = Activation.SILU, 
    periodicity: Union[Dict[str, Tuple[float, float]], None] = None,
    # Fourier arch
    detach_keys: List[Key] = [], # default physicsnemo
    frequencies        = ("axis", [i for i in range(10)]), # default physicsnemo
    frequencies_params = ("axis", [i for i in range(10)]), # default physicsnemo
):
    if model_type == "FourierNetArch":
        flow_net = FourierNetArch(
            input_keys  = input_keys,  
            output_keys = output_keys,
            # Arch Specs
            layer_size = layer_size,
            nr_layers  = nr_layers, 
            skip_connections     = skip_connections, 
            adaptive_activations = adaptive_activations, 
            activation_fn = activation_fn, 
            # 
            frequencies        = frequencies ,  
            frequencies_params = frequencies_params ,
            detach_keys = detach_keys , 
        )
    elif model_type == "ModifiedFourierNetArch":
        flow_net = ModifiedFourierNetArch(
            input_keys  = input_keys,  
            output_keys = output_keys,
            # Arch Specs
            layer_size = layer_size,
            nr_layers  = nr_layers, 
            skip_connections     = skip_connections, 
            adaptive_activations = adaptive_activations, 
            activation_fn = activation_fn, 
            # 
            frequencies        = frequencies ,  
            frequencies_params = frequencies_params ,
            detach_keys = detach_keys , 
        )
    elif model_type == "SirenArch":
        flow_net = SirenArch(
            input_keys  = input_keys,
            output_keys = output_keys,
            # Arch Specs
            layer_size = layer_size,
            nr_layers = nr_layers, 
            detach_keys = detach_keys , 
        )
    else:
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
    def __init__(self, n_experts: int, file_path = "./data/allen_cahn.mat"):
        super().__init__()
        self.n_experts = n_experts
        self.file_path = file_path
        
    def __call__(self, invars, outvars):
        x, t   = invars["x"][:,0], invars["t"][:,0]
        extent = (x.min(), x.max(), t.min(), t.max())
        
        # make experts plot
        fig, axs = plt.subplots(ncols = self.n_experts, nrows=2, figsize=(20,4), dpi=150,  constrained_layout=True)
        for idx in range(self.n_experts):
            # interpolate
            _u, _lambda = self.interpolate_output(
                x,t, 
                [ outvars[f"u_{idx+1}"][:,0] , outvars[f"lambda_{idx+1}"][:,0]], 
                extent
            )
            # U
            im = axs[0, idx].imshow(_u.T , origin="lower", extent=extent, cmap='turbo')
            axs[0, idx].set_title(f"Pred Exp u_{idx+1}")
            axs[0, idx].set_xlabel('x')
            axs[0, idx].set_ylabel('t')
            #self.add_colorbar(im=im, current_ax = axs[0, idx])
            fig.colorbar(im, ax=axs[0, idx], shrink=0.6, pad=0.05)
            axs[0, idx].set_xticklabels([])
            axs[0, idx].set_yticklabels([])
            # Lambda
            iml = axs[1, idx].imshow(_lambda.T , origin="lower", extent=extent, cmap='turbo')
            axs[1, idx].set_title(f"Import Exp l_{idx+1}")
            axs[1, idx].set_xlabel('x')
            axs[1, idx].set_ylabel('t') 
            #self.add_colorbar(im=iml, current_ax = axs[1, idx])
            fig.colorbar(iml, ax=axs[1, idx], shrink=0.6, pad=0.05)
            axs[1, idx].set_xticklabels([])
            axs[1, idx].set_yticklabels([])
        # === add validator =====
        # u
        _u = self.interpolate_output(
            x,t, 
            [ outvars[f"u"][:,0] ], 
            extent
        )[0]
        
        
        file_path = to_absolute_path(self.file_path)
        if os.path.exists(file_path):
            fig_tot, axs_tot = plt.subplots(ncols = 3, nrows = 1, figsize=(25,8), dpi=150,  constrained_layout=True)
            fig_tot.set_constrained_layout_pads(
                w_pad=1., h_pad=5.,
                hspace=2., wspace=2.
            )
            
            data = io.loadmat(file_path) # <=== LOAD EXACT DATA ==========
            x_true, t_true = data['x'][0,:], data['t'][0,:]
            _u_true = data['usol']
            _u_true_reshaped = F.avg_pool1d(
                F.avg_pool1d(torch.from_numpy(_u_true[:200, 4:-4]), 5).T,
                2
            ).T
            _u_true_reshaped = _u_true_reshaped.detach().cpu().numpy()
            _u_true = _u_true_reshaped.T
            ax_tot  = axs_tot[0]
            ax_true = axs_tot[1]
            ax_diff = axs_tot[2]
            
            ax_tot.set_title('PINN')
            
            imt = ax_true.imshow(_u_true.T , origin="lower", extent=extent, cmap='turbo')
            fig_tot.colorbar(imt, ax=ax_true, shrink=0.6, pad=0.05)
            ax_true.set_xticklabels([])
            ax_true.set_yticklabels([])
            ax_true.set_title('True')
                        
            imd = ax_diff.imshow(_u_true.T - _u.T , origin="lower", extent=extent, cmap='turbo')
            fig_tot.colorbar(imd, ax=ax_diff, shrink=0.6, pad=0.05)
            ax_diff.set_xticklabels([])
            ax_diff.set_yticklabels([])
            _mean_err = np.mean(np.abs(_u_true - _u)**2)
            ax_diff.set_title(f'Diff - L2 avg: {_mean_err:.1e}')
        else:
            fig_tot, ax_tot = plt.subplots(ncols = 1, nrows = 1, figsize=(8,8), dpi=150,  constrained_layout=True)
        
        imf = ax_tot.imshow(_u.T , origin="lower", extent=extent, cmap='turbo')
        fig_tot.colorbar(imd, ax=ax_tot, shrink=0.6, pad=0.05)
        ax_tot.set_xticklabels([])
        ax_tot.set_yticklabels([])
        
        return [(fig, "experts_plot"), (fig_tot, "total_u"), ]
    
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
    
    @staticmethod
    def add_colorbar(im, current_ax = plt.gca(), aspect=20, pad_fraction=0.5, **kwargs):
        """
        Add a vertical color bar to an image plot.
        
        From https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        """
        divider = axes_grid1.make_axes_locatable(im.axes)
        width   = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
        pad     = axes_grid1.axes_size.Fraction(pad_fraction, width)
        #
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.sca(current_ax)
        return im.axes.figure.colorbar(im, cax=cax, **kwargs)

    
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
    input_keys     = [Key("x")    , Key("t")]
    per_input_keys = [Key("x_sin"), Key("t")]
    output_keys    = [Key("u")]
    
    # Experts 
    # from https://towardsdatascience.com/mixture-of-experts-for-pinns-moe-pinns-6520adf32438
    # Expert 1: 2 layers with 64 nodes each and tanh activation
    # Expert 2: 2 layers with 64 nodes each and sine activation
    # Expert 3: 2 layers with 128 nodes each and tanh activation
    # Expert 4: 3 layers with 128 nodes each and swish activation
    # Expert 5: 2 layers with 256 nodes each and swish activation # there is no swish in physicsnemo, but SILU is a swish with parameter = 1
    experts = [
        ####################################
        # FCC Models 
        ####################################
        # Expert 1
        get_model(
            input_keys  = input_keys, 
            output_keys = [Key("u_1")],
            #
            layer_size = 64,
            nr_layers  = 2, 
            skip_connections     = True, 
            adaptive_activations = True, 
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
            skip_connections     = True, 
            adaptive_activations = True, 
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
            skip_connections     = True, 
            adaptive_activations = True, 
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
            skip_connections     = True, 
            adaptive_activations = True, 
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
        ) ,
        ####################################
        # FourierNet Models 
        ####################################
        # Expert 6
        get_model(
            input_keys  = per_input_keys, 
            output_keys = [Key("u_6")] ,
            #
            model_type = "FourierNetArch", 
            layer_size = 64,
            nr_layers  = 3, 
            skip_connections     = True, 
            adaptive_activations = True, 
            activation_fn = Activation.SILU,
        ),
        # Expert 7
        get_model(
            input_keys  = per_input_keys, 
            output_keys = [Key("u_7")] ,
            #
            model_type = "ModifiedFourierNetArch", 
            layer_size = 64,
            nr_layers  = 3, 
            skip_connections     = True, 
            adaptive_activations = True, 
            activation_fn = Activation.SIN,
        ),
        # Expert 8
        get_model(
            input_keys  = per_input_keys, 
            output_keys = [Key("u_8")] ,
            #
            model_type = "SirenArch", 
            layer_size = 64,
            nr_layers  = 3, 
        )
    ]
    N_experts = len(experts)
    # Gate
    u_keys      = [Key(f"u_{idx+1}")      for idx, expert in enumerate(experts)]
    lambda_keys = [Key(f"lambda_{idx+1}") for idx, expert in enumerate(experts)]
    
    gate_net = get_model(
        input_keys  = input_keys, 
        output_keys = lambda_keys,
        #
        layer_size = 64,
        nr_layers  = 2, 
        skip_connections     = True, 
        adaptive_activations = True, 
        activation_fn = Activation.SILU,
        periodicity   = _periodicity,   
    )
    # ====== Custom Node for computing MoE ===========================
    compute_u_node =  Node(
        inputs  = [*u_keys, *lambda_keys],  
        outputs = ['u'], 
        evaluate = ComputeU(n_experts = N_experts) ,
        name = 'gating_node'
    )    
    # Cos node for ensuring periodicity
    node_cos = Node.from_sympy(
        sin(2 * np.pi * Symbol("x") ), # has to be periodic
        "x_sin",
    )
    # make nodes
    nodes = [compute_u_node] + [node_cos] + pde.make_nodes() + [gate_net.make_node(name="gate_network")] + [*[expert.make_node(name=f"expert-{idx}") for idx, expert in enumerate(experts) ] ]
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
        importance = 10*(
              outvar["u__x"] ** 2
            + outvar["u__t"] ** 2 
        ) ** 0.5 + 5
        return importance.cpu().detach().numpy()
    
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
            "allencahn": Symbol("sdf"),
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
        outvar={"u": (x**2) * cos(pi*x)},
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"u": 10.0},
        parameterization={t_symbol: 0.0},
        #
        importance_measure = importance_measure, 
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
        output_names = [f"u_{idx+1}" for idx, _ in enumerate(experts) ]  + [f"lambda_{idx+1}" for idx, _ in enumerate(experts) ] + ["u"] ,#+ ["allencahn"],
        batch_size = 1024,
        plotter    = CustomInferencerPlotter(n_experts=N_experts),
    )
    domain.add_inferencer(expert_inferencer, "expert_inf_data")
    # ====== Solver ===========================
    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()
    
    
if __name__ == "__main__":
    run()