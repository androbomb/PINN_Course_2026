import torch
import torch.nn as nn
import math

class PINN_DNN(nn.Module):
    """
    Fully-connected neural network with activation-aware initialization:
      - SiLU / ReLU  -> Kaiming initialization
      - Tanh         -> Xavier (Glorot) initialization
      - Other        -> fallback to Tanh + Xavier
    """
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hidden_dims: list,
        activation_func=None,   # user may pass nn.Tanh, nn.ReLU, nn.SiLU, etc.
        use_bias: bool = True,
    ):
        super().__init__()

        # -----------------------------
        # 1. Activation selection logic
        # -----------------------------
        if activation_func is None:
            # Default fallback
            activation_func = nn.Tanh

        # If user passed an *instance*, convert to class
        if isinstance(activation_func, nn.Module):
            activation_cls = activation_func.__class__
        else:
            activation_cls = activation_func

        # Normalize name for detection
        act_name = activation_cls.__name__.lower()

        # -----------------------------
        # 2. Select initialization mode
        # -----------------------------
        if "relu" in act_name or "silu" in act_name:
            self.init_mode = "kaiming"
        elif "tanh" in act_name:
            self.init_mode = "xavier"
        else:
            # Fallback to Tanh + Xavier
            activation_cls = nn.Tanh
            self.init_mode = "xavier"

        self.activation_cls = activation_cls

        # -----------------------------
        # 3. Build the network
        # -----------------------------
        layers = []
        dims = [n_inputs] + hidden_dims + [n_outputs]

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1], bias=use_bias))
            layers.append(self.activation_cls())

        # Output layer
        layers.append(nn.Linear(dims[-2], dims[-1], bias=True))
        self.network = nn.Sequential(*layers)

        # -----------------------------
        # 4. Initialize parameters
        # -----------------------------
        self.reset_parameters()

    # -----------------------------------------
    # Initialization depending on activation
    # -----------------------------------------
    def reset_parameters(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                if self.init_mode == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif self.init_mode == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                # Bias initialization
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        return self.network(x)
