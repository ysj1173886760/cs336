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

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.eps = eps

        init_data = torch.ones((d_model, ), **factory_kwargs)
        self.gain = torch.nn.Parameter(init_data)
    
    def load_weights(self, weights: Float[Tensor, "d_model"]):
        state_dict = {"gain": weights}
        self.load_state_dict(state_dict)
    
    def forward(self, x: Float[Tensor, "... d_model"]):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # batch_size, d_model -> batch_size
        rms = torch.sqrt(torch.sum(x ** 2, dim=-1) / self.d_model + self.eps)
        # reshape to (..., 1). since we are doing vector-wise division
        rms = rms.reshape(*rms.shape, 1)

        result = x / rms * self.gain
        return result.to(dtype=in_dtype)

class FFNSwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
      super().__init__()
      self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
      self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
      self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
    
    def load_weights(self,
      w1_weight: Float[Tensor, " d_ff d_model"],
      w2_weight: Float[Tensor, " d_model d_ff"],
      w3_weight: Float[Tensor, " d_ff d_model"]):

      # nn.Module/nn.Parameter will get registered in self._parameters
      state_dict = {"w1.w": w1_weight, "w2.w": w2_weight, "w3.w": w3_weight}
      self.load_state_dict(state_dict)

    def silu(self, x: Float[Tensor, " ... d_ff"]):
      # element wise
      return torch.sigmoid(x) * x

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
      silu_res = self.silu(self.w1.forward(x))
      gate_res = silu_res * self.w3.forward(x)
      return self.w2.forward(gate_res)
  