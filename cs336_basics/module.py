import torch
from math import sqrt
from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import einsum


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        init_data = torch.empty(out_features, in_features, device=device, dtype=dtype)
        mean = 0
        std = sqrt(2 / (out_features + in_features))
        a = -3 * std
        b = 3 * std
        torch.nn.init.trunc_normal_(init_data, mean=mean, std=std, a=a, b=b)
        self.w = torch.nn.Parameter(init_data)

    def load_weights(self, weights: Float[Tensor, "d_out d_in"]):
        state_dict = {"w": weights}
        self.load_state_dict(state_dict)

    def forward(self, x: Float[Tensor, "... d_in"]) -> torch.Tensor:
        return einsum(self.w, x, "d_out d_in, ... d_in -> ... d_out")
