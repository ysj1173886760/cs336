from pydantic import BaseModel


class Params(BaseModel):
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    batch_size: int = 1


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"


def calc_memory(param: Params):
    embedding_layer = param.vocab_size * param.d_model
    lm_head = param.vocab_size * param.d_model

    single_transformer_block = (
        param.d_model * param.d_model * 4 + 3 * param.d_ff * param.d_model
    )

    total_transformer_block = param.num_layers * single_transformer_block

    total_size = embedding_layer + lm_head + total_transformer_block

    print(f"embedding_layer: {sizeof_fmt(embedding_layer * 4)}")
    print(f"lm_head: {sizeof_fmt(lm_head * 4)}")
    print(f"single_transformer_block: {sizeof_fmt(single_transformer_block * 4)}")
    print(f"total_transformer_block: {sizeof_fmt(total_transformer_block * 4)}")
    print(f"total_size: {sizeof_fmt(total_size * 4)}")
    print(f"param count: {total_size}")


def calc_flops(param: Params):
    ffn = 6 * param.context_length * param.d_model * param.d_ff
    attn_proj = 8 * param.context_length * (param.d_model**2)
    attn_dot_product = 4 * (param.context_length**2) * param.d_model

    ffn *= param.num_layers
    attn_dot_product *= param.num_layers
    attn_proj *= param.num_layers
    total = ffn + attn_proj + attn_dot_product

    print(f"ffn: {ffn:.3e} {ffn / total:.2%}")
    print(f"attn_dot_product: {attn_dot_product:.3e}. {attn_dot_product / total:.2%}")
    print(f"attn_proj: {attn_proj:.3e} {attn_proj / total:.2%}")
    print(f"total: {total:.3e}")

def calc_memory_training(param: Params):
    param_mem = 12 * param.d_model * param.d_model * param.num_layers + 2 * param.vocab_size * param.d_model

    grad_mem = param_mem

    d_type = 4

    optimizer_mem = 2 * param_mem

    activation_mem = 9 * param.d_model * param.context_length + 2 * param.context_length * param.context_length
    activation_mem *= param.num_layers
    activation_mem *= param.batch_size

    print(f"param_mem: {sizeof_fmt(param_mem * d_type)}")
    print(f"grad_mem: {sizeof_fmt(grad_mem * d_type)}")
    print(f"optimizer_mem: {sizeof_fmt(optimizer_mem * d_type)}")
    print(f"activation_mem: {sizeof_fmt(activation_mem * d_type)}")
    print(f"total: {sizeof_fmt((param_mem + grad_mem + optimizer_mem + activation_mem) * d_type)}")

if __name__ == "__main__":
    gpt_2_xl = Params(
        vocab_size=50257,
        context_length=1024,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=6400,
        batch_size=1,
    )
    print("gpt2_xl")
    calc_memory(gpt_2_xl)
    calc_flops(gpt_2_xl)

    # gpt2_small = Params(
    #     vocab_size=50257,
    #     context_length=1024,
    #     num_layers=12,
    #     d_model=768,
    #     num_heads=12,
    #     d_ff=3072,
    # )
    # print("gpt2_small")
    # calc_flops(gpt2_small)

    # gpt2_medium = Params(
    #     vocab_size=50257,
    #     context_length=1024,
    #     num_layers=24,
    #     d_model=1024,
    #     num_heads=16,
    #     d_ff=4096,
    # )
    # print("gpt2_medium")
    # calc_flops(gpt2_medium)

    # gpt2_large = Params(
    #     vocab_size=50257,
    #     context_length=1024,
    #     num_layers=36,
    #     d_model=1280,
    #     num_heads=12,
    #     d_ff=5120,
    # )
    # print("gpt2_large")
    # calc_flops(gpt2_large)

    # gpt2_xl_long = Params(
    #     vocab_size=50257,
    #     context_length=16384,
    #     num_layers=48,
    #     d_model=1600,
    #     num_heads=25,
    #     d_ff=6400,
    # )
    # print("gpt2_xl_long")
    # calc_flops(gpt2_xl_long)
