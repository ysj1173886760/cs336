from cs336_basics.module import (
    TransformerLM,
    AdamW,
    cross_entropy,
    gradient_clipping,
    get_batch_data,
    load_checkpoint,
    save_checkpoint,
    get_cos_lr_schedule,
    calc_gradient_norm,
    softmax,
)
from cs336_basics.bpe_tokenizer import BPETokenizer
import argparse
import numpy.typing as npt
import os
import numpy as np
from tqdm import tqdm
import torch
from math import sqrt
import wandb
import time
from datetime import datetime
import logging


def get_current_lr(optimizer, iteration):
    """获取当前的学习率"""
    if optimizer.lr_scheduling:
        return get_cos_lr_schedule(
            iteration, optimizer.lr_max, optimizer.lr_min, optimizer.t_w, optimizer.t_c
        )
    else:
        return optimizer.param_groups[0]["lr"]


def train(args, train_dataset: npt.NDArray, valid_dataset: npt.NDArray):
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "vocab_size": args.vocab_size,
                "context_length": args.context_length,
                "d_model": args.d_model,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "d_ff": args.d_ff,
                "rope_theta": args.rope_theta,
                "batch_size": args.batch_size,
                "lr_max": args.lr_max,
                "lr_min": args.lr_min,
                "weight_decay": args.weight_decay,
                "warmup_iterations": args.warmup_iterations,
                "cosine_annealing_iterations": args.cosine_annealing_iterations,
                "max_iterations": args.max_iterations,
            },
        )

    transformer = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
        device=args.device,
        dtype=torch.float32,
    )

    if args.device == "cpu":
        transformer = torch.compile(transformer)
    elif args.device.startswith("mps"):
        transformer = torch.compile(transformer, backend="aot_eager")

    optimizer = AdamW(
        transformer.parameters(),
        lr=args.lr_max,
        betas=args.betas,
        weight_decay=args.weight_decay,
    )

    enable_lr_scheduling = True

    iteration = try_load_checkpoint(args.checkpoint_path, transformer, optimizer)

    for iteration in tqdm(range(iteration, args.max_iterations + 1)):
        x, y = get_batch_data(
            train_dataset, args.batch_size, args.context_length, args.device
        )
        logits = transformer(x)

        # reshape to 2D for cross entropy
        logits = logits.reshape(-1, args.vocab_size)
        y = y.reshape(-1)

        loss = cross_entropy(logits, y)

        current_lr = args.lr_max
        if enable_lr_scheduling:
            current_lr = get_cos_lr_schedule(
                iteration,
                args.lr_max,
                args.lr_min,
                args.warmup_iterations,
                args.cosine_annealing_iterations,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        optimizer.zero_grad()
        loss.backward()

        grad_norm = calc_gradient_norm(transformer.parameters())
        gradient_clipping(transformer.parameters(), args.gradient_clipping, grad_norm)

        optimizer.step()

        if iteration % args.checkpoint_interval == 0:
            save_checkpoint(transformer, optimizer, iteration, args.checkpoint_path)
            logging.info(
                f"Checkpoint saved at iteration {iteration} current loss: {loss.item():.6f}"
            )

        # 使用 wandb 记录指标
        if args.use_wandb:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/gradient_norm": grad_norm,
                    "train/learning_rate": current_lr,
                    "iteration": iteration,
                }
            )

        if iteration % 100 == 0:
            logging.info(f"Iteration {iteration} | Loss: {loss.item():.6f}")
            logging.info(f"Gradient Norm: {grad_norm:.3e}")
            logging.info(f"Learning Rate: {current_lr:.3e}")

    # 训练完成后关闭 wandb
    if args.use_wandb:
        wandb.finish()


def try_load_checkpoint(checkpoint_path, transformer, optimizer) -> int:
    if os.path.exists(checkpoint_path):
        return load_checkpoint(checkpoint_path, transformer, optimizer)
    return 0


def load_data(data_path) -> npt.NDArray:
    return np.load(data_path, mmap_mode="r")


def nucleus_sampling(prob, p) -> int:
    sorted_prob, sorted_idx = torch.sort(prob, dim=-1, descending=True)
    acc_prob = torch.cumsum(sorted_prob, dim=-1)

    # 不减去sorted_prob的话，可能一个都选不上
    mask = acc_prob - sorted_prob <= p

    # 筛选出来新的集合
    filtered_prob = sorted_prob[mask]
    filtered_idx = sorted_idx[mask]

    # 重新归一化
    filtered_prob = filtered_prob / filtered_prob.sum()

    sample_idx = torch.multinomial(filtered_prob, num_samples=1)
    return filtered_idx[sample_idx]


@torch.no_grad()
def decode(args):
    transformer = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
        device=args.device,
        dtype=torch.float32,
    )
    if args.device == "cpu":
        transformer = torch.compile(transformer)
    elif args.device.startswith("mps"):
        transformer = torch.compile(transformer, backend="aot_eager")

    iteration = try_load_checkpoint(args.checkpoint_path, transformer, None)

    tokenizer = BPETokenizer.from_files(
        args.tokenizer_path, special_tokens=["<|endoftext|>"]
    )

    prompt = "Once upon a time"

    encoded_prompt = tokenizer.encode(prompt)
    prompt_length = len(encoded_prompt)
    # [1, context_length]
    encoded_data = torch.zeros(
        1, args.context_length, dtype=torch.long, device=args.device
    )
    encoded_data[0, :prompt_length] = torch.tensor(
        encoded_prompt, dtype=torch.long, device=args.device
    )

    current_idx = prompt_length
    end_token_id = tokenizer.encode("<|endoftext|>")[0]  # 获取end token的id

    while current_idx < args.context_length:
        logits = transformer(encoded_data)
        # 使用前一个位置的logits来预测当前位置的token
        logits = logits[0, current_idx - 1]

        prob = softmax(logits, dim=-1, temperature=args.temperature)

        if args.enable_nucleus_sampling:
            next_token = nucleus_sampling(prob, args.nucleus_sampling_p)
        else:
            next_token = torch.multinomial(prob, num_samples=1).item()

        encoded_data[0, current_idx] = next_token

        if next_token == end_token_id:
            break

        current_idx += 1

    print(tokenizer.decode(encoded_data[0, :current_idx].cpu().tolist()))
    print(f"final idx: {current_idx}")


def valid_dataset(dataset):
    special_token_count = (dataset == 256).sum()
    total_token_count = len(dataset)

    print(f"special token count: {special_token_count}")
    print(f"total token count: {total_token_count}")
    print(f"special token ratio: {special_token_count / total_token_count:.6f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    parser.add_argument("--train_data_path", type=str, default="tiny_stories_train.npy")
    parser.add_argument("--valid_data_path", type=str, default="tiny_stories_valid.npy")

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
    parser.add_argument("--gradient_clipping", type=float, default=1.0)

    parser.add_argument(
        "--checkpoint_path", type=str, default="checkpoints/checkpoint.pt"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=100)

    # lr schedule params
    parser.add_argument("--lr_max", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--warmup_iterations", type=int, default=200)
    parser.add_argument("--cosine_annealing_iterations", type=int, default=5000)

    # wandb params
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 wandb 记录训练指标")
    parser.add_argument("--wandb_project", type=str, default="llm", help="wandb 项目名称")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb 运行名称")

    # decode params
    parser.add_argument("--tokenizer_path", type=str, default="tiny_stories_result.pkl")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--enable_nucleus_sampling", action="store_true", help="是否启用 topk 采样"
    )
    parser.add_argument("--nucleus_sampling_p", type=float, default=0.9)

    args = parser.parse_args()

    # train_dataset = load_data(args.train_data_path)
    # valid_dataset = load_data(args.valid_data_path)

    # valid_dataset(train_dataset)

    # train_dataset = train_dataset[:10000]
    # train(args, train_dataset, valid_dataset)
    # train(args, valid_dataset, valid_dataset)
    decode(args)
