from cs336_basics.module import (
    TransformerLM,
    AdamW,
    cross_entropy,
    gradient_clipping,
    get_batch_data,
    load_checkpoint,
    save_checkpoint,
    get_cos_lr_schedule,
)
import argparse
import numpy.typing as npt
import os
import numpy as np
from tqdm import tqdm
import torch
from math import sqrt
import wandb
import weave
import time
from datetime import datetime


def calc_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += torch.sum(p.grad**2)
    return sqrt(total_norm)


def get_current_lr(optimizer, iteration):
    """获取当前的学习率"""
    if optimizer.lr_scheduling:
        return get_cos_lr_schedule(
            iteration, optimizer.lr_max, optimizer.lr_min, optimizer.t_w, optimizer.t_c
        )
    else:
        return optimizer.param_groups[0]["lr"]


def train(args, train_dataset: npt.NDArray, valid_dataset: npt.NDArray):
    # 初始化 wandb
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
    # optimizer.enable_lr_scheduling(args.lr_max, args.lr_min, args.warmup_iterations, args.cosine_annealing_iterations)

    iteration = try_load_checkpoint(args.checkpoint_path, transformer, optimizer)

    for iteration in tqdm(range(iteration, args.max_iterations)):
        x, y = get_batch_data(
            train_dataset, args.batch_size, args.context_length, args.device
        )
        logits = transformer(x)

        # reshape to 2D for cross entropy
        logits = logits.reshape(-1, args.vocab_size)
        y = y.reshape(-1)

        loss = cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()

        # 计算梯度范数
        grad_norm = calc_gradient_norm(transformer)

        # gradient_clipping(transformer.parameters(), args.gradient_clipping)
        optimizer.step()

        # 获取当前学习率
        current_lr = get_current_lr(optimizer, iteration)

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

        if iteration % 1 == 0:
            print(
                f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Iteration {iteration} | Loss: {loss.item():.6f}"
            )
            print(f"Gradient Norm: {grad_norm}")
            print(f"Learning Rate: {current_lr}")

    # 训练完成后关闭 wandb
    if args.use_wandb:
        wandb.finish()


def try_load_checkpoint(checkpoint_path, transformer, optimizer) -> int:
    if os.path.exists(checkpoint_path):
        return load_checkpoint(checkpoint_path, transformer, optimizer)
    return 0


def load_data(data_path) -> npt.NDArray:
    return np.load(data_path, mmap_mode="r")


if __name__ == "__main__":
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
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    # lr schedule params
    parser.add_argument("--lr_max", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--warmup_iterations", type=int, default=200)
    parser.add_argument("--cosine_annealing_iterations", type=int, default=5000)

    # wandb params
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 wandb 记录训练指标")
    parser.add_argument("--wandb_project", type=str, default="llm", help="wandb 项目名称")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb 运行名称")

    args = parser.parse_args()

    train_dataset = load_data(args.train_data_path)
    valid_dataset = load_data(args.valid_data_path)

    train(args, train_dataset, valid_dataset)
    # valid_dataset = valid_dataset[:50000]
    # train(args, valid_dataset, valid_dataset)
