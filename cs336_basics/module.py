import torch
from math import sqrt, cos, sin, pi
from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import einsum, rearrange
from typing import Iterable
import numpy.typing as npt
import numpy as np
from typing import IO, Any, BinaryIO
import os


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
        self.weight = torch.nn.Parameter(init_data)

    def load_weights(self, weights: Float[Tensor, "d_out d_in"]):
        state_dict = {"weight": weights}
        self.load_state_dict(state_dict)

    def forward(self, x: Float[Tensor, "... d_in"]) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


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
        self.weight = torch.nn.Parameter(init_data)

    def load_weights(self, weights: Float[Tensor, "vocab_size d_model"]):
        state_dict = {"weight": weights}
        self.load_state_dict(state_dict)

    def forward(self, token_ids: Int[Tensor, "..."]):
        input_flat = token_ids.reshape(-1)
        # select embedding at dimension 0
        result = torch.index_select(self.weight, 0, input_flat)
        return result.reshape((*token_ids.shape, -1))


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.eps = eps

        init_data = torch.ones((d_model,), **factory_kwargs)
        self.weight = torch.nn.Parameter(init_data)

    def load_weights(self, weights: Float[Tensor, "d_model"]):
        state_dict = {"weight": weights}
        self.load_state_dict(state_dict)

    def forward(self, x: Float[Tensor, "... d_model"]):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # batch_size, d_model -> batch_size
        rms = torch.sqrt(torch.sum(x**2, dim=-1) / self.d_model + self.eps)
        # reshape to (..., 1). since we are doing vector-wise division
        rms = rms.reshape(*rms.shape, 1)

        result = x / rms * self.weight
        return result.to(dtype=in_dtype)


class FFNSwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(
            in_features=d_model, out_features=d_ff, device=device, dtype=dtype
        )
        self.w2 = Linear(
            in_features=d_ff, out_features=d_model, device=device, dtype=dtype
        )
        self.w3 = Linear(
            in_features=d_model, out_features=d_ff, device=device, dtype=dtype
        )

    def load_weights(
        self,
        w1_weight: Float[Tensor, " d_ff d_model"],
        w2_weight: Float[Tensor, " d_model d_ff"],
        w3_weight: Float[Tensor, " d_ff d_model"],
    ):

        # nn.Module/nn.Parameter will get registered in self._parameters
        state_dict = {
            "w1.weight": w1_weight,
            "w2.weight": w2_weight,
            "w3.weight": w3_weight,
        }
        self.load_state_dict(state_dict)

    def silu(self, x: Float[Tensor, " ... d_ff"]):
        # element wise
        return torch.sigmoid(x) * x

    def forward(
        self, x: Float[Tensor, " ... d_model"]
    ) -> Float[Tensor, " ... d_model"]:
        silu_res = self.silu(self.w1.forward(x))
        gate_res = silu_res * self.w3.forward(x)
        return self.w2.forward(gate_res)


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.construct_cache()

    # TODO: optimize init process
    def construct_cache(self):
        cache = torch.empty((self.max_seq_len, self.d_k // 2, 2, 2))
        for i in range(self.max_seq_len):
            for k in range(self.d_k // 2):
                cache[i][k] = self.get_block_mat(i, k)
        self.register_buffer("rotate_mat", cache, persistent=False)

    def calc_angle(self, position, k):
        return position / (self.theta ** ((2 * k) / self.d_k))

    def get_block_mat(self, position: int, k: int) -> Tensor:
        return torch.tensor(
            [
                cos(self.calc_angle(position, k)),
                -sin(self.calc_angle(position, k)),
                sin(self.calc_angle(position, k)),
                cos(self.calc_angle(position, k)),
            ]
        ).reshape((2, 2))

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ) -> Float[Tensor, " ... sequence_length d_k"]:
        n1 = self.d_k // 2
        n2 = 2
        reshape_x = rearrange(
            x, "... sequence_length (n1 n2)-> ... sequence_length n1 n2", n1=n1, n2=n2
        )

        # batch_size, seq_len, d / k, 2, 2
        rotate_mat = self.rotate_mat[token_positions]

        result = einsum(reshape_x, rotate_mat, "... n1 j, ... n1 i j-> ... n1 i")
        return result.reshape(*x.shape)


def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    max = torch.max(x, dim=dim, keepdim=True).values
    x = torch.sub(x, max)
    exp = torch.exp(x)
    sum = torch.sum(exp, dim=dim, keepdim=True)
    return exp / sum


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    dot_product = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    # scale dot product
    dot_product = dot_product / sqrt(d_k)

    if mask is not None:
        mask_value = torch.where(mask, torch.tensor(0.0), torch.tensor(float("-inf")))
        dot_product += mask_value

    atten_score = softmax(dot_product, dim=-1)
    # use atten score to scale feature dimension
    return einsum(atten_score, V, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads

        if max_seq_len != 0:
            self.rope = RoPE(theta, self.d_k, max_seq_len)

        self.q_proj = Linear(self.d_model, self.d_model)
        self.k_proj = Linear(self.d_model, self.d_model)
        self.v_proj = Linear(self.d_model, self.d_model)
        self.output_proj = Linear(self.d_model, self.d_model)

    def load_weights(
        self,
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
    ):
        state_dict = {
            "q_proj.weight": q_proj_weight,
            "k_proj.weight": k_proj_weight,
            "v_proj.weight": v_proj_weight,
            "output_proj.weight": o_proj_weight,
        }
        self.load_state_dict(state_dict)

    def construct_causal_mask(self, sequence_length: int):
        mask = torch.triu(
            torch.ones(sequence_length, sequence_length), diagonal=1
        ).bool()
        return ~mask

    def split_heads(
        self, x: Float[Tensor, " ... sequence_length d_in"]
    ) -> Float[Tensor, " ... head sequence_length d_k"]:
        new_shape = x.shape[:-1] + (self.num_heads, self.d_k)
        return x.view(new_shape).transpose(-2, -3)

    def combine_heads(
        self, x: Float[Tensor, " ... head sequence_length d_k"]
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        return x.transpose(-2, -3).flatten(start_dim=-2)

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        # 先把每一个token从d_model投影到d_k上，然后再拼接成一个大的d_model
        # 等价于直接投影到d_model上，再去拆分。因为投影每一个d_k是独立的
        q_proj = self.split_heads(self.q_proj.forward(x))
        k_proj = self.split_heads(self.k_proj.forward(x))
        v_proj = self.split_heads(self.v_proj.forward(x))

        if token_positions is not None:
            q_proj = self.rope.forward(q_proj, token_positions.unsqueeze(1))
            k_proj = self.rope.forward(k_proj, token_positions.unsqueeze(1))

        sequence_length = x.shape[-2]
        causal_mask = self.construct_causal_mask(sequence_length)

        # 必须要先split head再去算attention，否则算的时候会把整个d_model都用来算atten score，就没有多头了
        attention = scaled_dot_product_attention(q_proj, k_proj, v_proj, causal_mask)
        attention = self.combine_heads(attention)
        return self.output_proj.forward(attention)


class TransformerBlock(torch.nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float
    ):
        super().__init__()

        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta)
        self.ln1 = RMSNorm(d_model)
        self.ffn = FFNSwiGLU(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)

    def load_weights(self, weights: dict):
        self.load_state_dict(weights)

    def forward(
        self,
        x: Float[Tensor, " batch sequence_length d_model"],
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        b, s, _ = x.shape
        token_position = torch.arange(s).expand(b, s)

        y = x + self.attn.forward(self.ln1.forward(x), token_position)

        y2 = y + self.ffn.forward(self.ln2.forward(y))

        return y2


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            )

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self, in_indices: Int[Tensor, " batch_size sequence_length"]
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.token_embeddings.forward(in_indices)
        for layer in self.layers:
            x = layer.forward(x)

        x = self.ln_final.forward(x)
        x = self.lm_head.forward(x)
        return x


# -log softmax(oi)[x i+1] ->
# log(sum(exp(oi))) - o[x i+1]
def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    # calc max logits
    max = torch.max(inputs, dim=-1, keepdim=True).values
    inputs = inputs - max
    exp_sum = torch.sum(torch.exp(inputs), dim=-1)

    target_logits = torch.gather(inputs, dim=1, index=targets.unsqueeze(1))
    loss = torch.log(exp_sum) - target_logits
    return torch.mean(loss)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8
    ):
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)
                state["t"] = t + 1

                grad = p.grad.data

                m = state.get("m", torch.zeros_like(grad))
                v = state.get("v", torch.zeros_like(grad))

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad**2)

                state["m"] = m
                state["v"] = v

                lr_adjust = lr * sqrt(1 - (beta2**t)) / (1 - (beta1**t))

                p.data = p.data - lr_adjust * m / (torch.sqrt(v) + eps)
                # apply weight decay
                p.data = p.data - lr * weight_decay * p.data

        return loss


def get_cos_lr_schedule(t, lr_max, lr_min, t_w, t_c) -> float:
    if t < t_w:
        return t / t_w * lr_max

    if t > t_c:
        return lr_min

    return 0.5 * (1 + cos((t - t_w) / (t_c - t_w) * pi)) * (lr_max - lr_min) + lr_min


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps = 1e-6
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.norm(torch.cat(grads), p=2)

    scale_factor = max_l2_norm / (total_norm + eps)
    if scale_factor < 1.0:
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.mul_(scale_factor)


def get_batch_data(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(dataset) >= context_length + 1
    indices = np.random.choice(
        len(dataset) - context_length, size=batch_size, replace=True
    )

    offsets = np.arange(context_length + 1).reshape(1, -1)
    idx = indices.reshape(batch_size, 1) + offsets

    window = dataset[idx]
    x = window[:, :-1]
    y = window[:, 1:]
    x = torch.from_numpy(x).to(device=device, dtype=torch.long)
    y = torch.from_numpy(y).to(device=device, dtype=torch.long)
    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(dict, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    dict = torch.load(src)
    model.load_state_dict(dict["model"])
    optimizer.load_state_dict(dict["optimizer"])
    return dict["iteration"]
