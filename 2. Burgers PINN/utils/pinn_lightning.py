from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from .utils import exact_solution_func, R2_extraction
from .pinn_dnn import PINN_DNN
from .heat_pde import Heat1D_PDE
from .heat_bc  import Heat1D_BC

class HeatPINNLightning(pl.LightningModule):
    """
    PINN for 1D Heat Equation using PyTorch Lightning.
    Full PINN Class for the FORWARD problem.

    It incorporates:
        1. The PINN DNN (in self.DNN)
        2. The PDE      (in self._PDE)
        3. The BC       (in self._BC)

    The train_model() method is thus:
        1. ADAM loop
            1.1. call training_step() for adam
            1.2. Store best model
            1.3. Perform step for learning rate stepper
            1.4. Logs
            1.5. Check if patience reached;

    Args:
        For Args see init method.
    """
    def __init__(
        self,
        # Geometry
        time_interval : Tuple[float] = (0.0, 1.0),
        space_interval: Tuple[float] = (-1.0, 1.0),
        # Network
        n_inputs : int = 2,
        n_outputs: int = 1,
        hidden_layers: Tuple[int] = (32, 32, 32),
        activation_func=nn.Tanh,
        # PDE
        diffusion_coefficient: float = 0.5,
        # Loss components
        use_rec : bool = False,
        fun_batch_size: int = 4096,
        pde_batch_size: int = 4096,
        bc_batch_size : int = 1024,
        ic_batch_size : int = 1024,
        # SoftAdapt
        use_softadapt: bool = False,
        softadapt_start_epoch: int = 5,
        # Optimizer
        learning_rate: float = 1e-3,
        # sampler
        use_r2: bool = False, 
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Geometry
        self.t_min, self.t_max = time_interval
        self.x_min, self.x_max = space_interval

        # Network
        self.dnn = PINN_DNN(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            hidden_dims=list(hidden_layers),
            activation_func=activation_func,
        )

        # PDE + BC/IC
        self.pde   = Heat1D_PDE(D=diffusion_coefficient)
        self.bc_ic = Heat1D_BC(cost_function=nn.MSELoss())

        # Exact solution (for reconstruction)
        self.exact_solution_func = exact_solution_func
        self.use_rec = use_rec

        # Batch sizes
        self.fun_batch_size = fun_batch_size
        self.pde_batch_size = pde_batch_size
        self.bc_batch_size  = bc_batch_size
        self.ic_batch_size  = ic_batch_size

        # Loss weights (SoftAdapt will update these)
        self.weight_rec = 1.0
        self.weight_pde = 1.0
        self.weight_bc  = 1.0
        self.weight_ic  = 1.0

        # SoftAdapt
        self.use_softadapt = use_softadapt
        self.softadapt_start_epoch = max(2, softadapt_start_epoch)

        # Store previous epoch losses
        self.prev_losses = None

        # Loss function
        self.mse = nn.MSELoss()

        # sampler 
        self.use_r2 = use_r2
        if self.use_r2:
            self.r2_extractor = R2_extraction(dimension=2)

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(self, x):
        return self.dnn(x)

    # -------------------------------------------------------------------------
    # dataloader to trick lighnting
    # -------------------------------------------------------------------------
    def train_dataloader(self):
        # Lightning requires a dataloader, but we ignore the batch
        return torch.utils.data.DataLoader([0], batch_size=1)

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------
    def generate_coords(self):
        device = self.device
        # PDE points
        if self.use_r2:
            coords = self.r2_extractor.r_d_extraction(self.pde_batch_size)
            coords = coords.to(self.device)
            coords[:, 0] = self.t_min + (self.t_max - self.t_min) * coords[:, 0]
            coords[:, 0] = self.x_min + (self.x_max - self.x_min) * coords[:, 0]
        else:
            coords = torch.cat((
                self.t_min + (self.t_max - self.t_min) * torch.rand(self.pde_batch_size, 1, device=device),
                self.x_min + (self.x_max - self.x_min) * torch.rand(self.pde_batch_size, 1, device=device),
            ), dim=-1)
        coords.requires_grad_(True)

        # IC points (t = t_min)
        ic_coords = torch.cat((
            self.t_min * torch.ones(self.ic_batch_size, 1, device=device),
            self.x_min + (self.x_max - self.x_min) * torch.rand(self.ic_batch_size, 1, device=device),
        ), dim=-1)
        ic_coords.requires_grad_(True)

        # BC points (x = ±1)
        t_rand = self.t_min + (self.t_max - self.t_min) * torch.rand(self.bc_batch_size // 2, 1, device=device)
        bc_coords_p = torch.cat((t_rand, self.x_max * torch.ones_like(t_rand)), dim=-1)
        bc_coords_m = torch.cat((t_rand, self.x_min * torch.ones_like(t_rand)), dim=-1)
        bc_coords = torch.cat((bc_coords_m, bc_coords_p), dim=0)
        bc_coords.requires_grad_(True)

        return coords, ic_coords, bc_coords

    # -------------------------------------------------------------------------
    # Loss components
    # -------------------------------------------------------------------------
    def compute_pde_loss(self, coords, pred):
        heat_eq = self.pde.compute_heat(coords, pred)
        return self.mse(heat_eq, torch.zeros_like(heat_eq))

    def compute_ic_loss(self, ic_coords, pred_ic):
        return self.bc_ic.initial_cond(ic_coords, pred_ic)

    def compute_bc_loss(self, bc_coords, pred_bc):
        return self.bc_ic.boundary_cond(bc_coords, pred_bc)

    # -------------------------------------------------------------------------
    # SoftAdapt (epoch-wise)
    # -------------------------------------------------------------------------
    def soft_adapt(self, losses, prev_losses, eps=1e-8):
        """
        losses      = [rec, pde, ic, bc]
        prev_losses = [rec_prev, pde_prev, ic_prev, bc_prev]
        """
        Li = np.array(losses, dtype=np.float64)
        Lo = np.array(prev_losses, dtype=np.float64)

        ratio = Li / (Lo + eps)
        mu = np.max(ratio)

        ratio_t = torch.tensor(ratio - mu, dtype=torch.float32, device=self.device)
        w = torch.softmax(ratio_t, dim=0)

        return w.tolist()

    # -------------------------------------------------------------------------
    # Lightning: training_step
    # -------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        # sampling
        coords, ic_coords, bc_coords = self.generate_coords()

        # Predictions
        pred_pde = self(coords)
        pred_ic  = self(ic_coords)
        pred_bc  = self(bc_coords)

        # True solution for reconstruction
        true_pde = self.exact_solution_func(coords, diffusion_param=self.hparams.diffusion_coefficient)
        true_pde = true_pde.unsqueeze(-1)

        # Compute losses
        pde_loss = self.compute_pde_loss(coords, pred_pde)
        ic_loss  = self.compute_ic_loss(ic_coords, pred_ic)
        bc_loss  = self.compute_bc_loss(bc_coords, pred_bc)
        rec_loss = self.mse(pred_pde, true_pde) if self.use_rec else pred_pde.new_tensor(0.0)

        # Weighted sum
        total_weight = self.weight_pde + self.weight_ic + self.weight_bc
        if self.use_rec:
            total_weight += self.weight_rec

        loss = (
            self.weight_pde * pde_loss +
            self.weight_ic  * ic_loss  +
            self.weight_bc  * bc_loss
        )
        if self.use_rec:
            loss += self.weight_rec * rec_loss

        loss = loss / total_weight

        # Log losses
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/pde_loss", pde_loss)
        self.log("train/ic_loss", ic_loss)
        self.log("train/bc_loss", bc_loss)
        if self.use_rec:
            self.log("train/rec_loss", rec_loss)

        return {
            "loss": loss,
            "pde_loss": pde_loss.detach(),
            "ic_loss": ic_loss.detach(),
            "bc_loss": bc_loss.detach(),
            "rec_loss": rec_loss.detach(),
        }

    # -------------------------------------------------------------------------
    # Lightning: epoch end → SoftAdapt update
    # -------------------------------------------------------------------------
    def on_train_epoch_end(self):
        outputs = self.trainer.callback_metrics

        # Extract epoch-averaged losses
        rec = float(outputs.get("train/rec_loss", 0.0))
        pde = float(outputs["train/pde_loss"])
        ic  = float(outputs["train/ic_loss"])
        bc  = float(outputs["train/bc_loss"])

        current_losses = [rec, pde, ic, bc]

        # Apply SoftAdapt only after enough epochs
        if self.use_softadapt and self.current_epoch >= self.softadapt_start_epoch:
            if self.prev_losses is not None:
                w_rec, w_pde, w_ic, w_bc = self.soft_adapt(current_losses, self.prev_losses)

                self.weight_rec = float(w_rec)
                self.weight_pde = float(w_pde)
                self.weight_ic  = float(w_ic)
                self.weight_bc  = float(w_bc)

                self.log("weights/rec", self.weight_rec)
                self.log("weights/pde", self.weight_pde)
                self.log("weights/ic",  self.weight_ic)
                self.log("weights/bc",  self.weight_bc)

        # Store for next epoch
        self.prev_losses = current_losses

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=50,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
            },
        }
