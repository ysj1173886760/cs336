import torch
from math import sqrt
from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import einsum, rearrange


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        init_data = torch.empty((out_features, in_features), device=device, dtype=dtype)
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


class Embedding(torch.nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, device=None, dtype=None
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim

        init_data = torch.empty((self.vocab_size, self.d_model), **factory_kwargs)
        mean = 0
        std = 1
        a, b = -3, 3
        torch.nn.init.trunc_normal_(init_data, mean, std, a, b)
        self.w = torch.nn.Parameter(init_data)

    def load_weights(self, weights: Float[Tensor, "vocab_size d_model"]):
        state_dict = {"w": weights}
        self.load_state_dict(state_dict)

    def forward(self, token_ids: Int[Tensor, "..."]):
        input_flat = token_ids.reshape(-1)
        # select embedding at dimension 0
        result = torch.index_select(self.w, 0, input_flat)
        return result.reshape((*token_ids.shape, -1))
