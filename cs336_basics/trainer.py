from cs336_basics.module import (
    TransformerLM,
    AdamW,
    cross_entropy,
    gradient_clipping,
    get_batch_data,
    load_checkpoint,
    save_checkpoint,
)
import argparse
import numpy.typing as npt
import os
import numpy as np


def train(args, train_dataset: npt.NDArray, valid_dataset: npt.NDArray):
    transformer = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
    )

    optimizer = AdamW(
        transformer.parameters(),
        lr=args.lr,
        betas=args.betas,
        weight_decay=args.weight_decay,
    )

    iteration = try_load_checkpoint(args.checkpoint_path, transformer, optimizer)

    for iteration in range(iteration, args.max_iterations):
        x, y = get_batch_data(
            train_dataset, args.batch_size, args.context_length, args.device
        )
        logits = transformer(x)

        # reshape to 2D for cross entropy
        logits = logits.reshape(-1, args.vocab_size)
        y = y.reshape(-1)

        loss = cross_entropy(logits, y)
        loss.backward()

        print(f"Iteration {iteration} | Loss: {loss.item():.6f}")

        gradient_clipping(transformer.parameters(), args.gradient_clipping)

        optimizer.step()


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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--max_iterations", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    args = parser.parse_args()

    train_dataset = load_data(args.train_data_path)
    valid_dataset = load_data(args.valid_data_path)

    train(args, train_dataset, valid_dataset)
