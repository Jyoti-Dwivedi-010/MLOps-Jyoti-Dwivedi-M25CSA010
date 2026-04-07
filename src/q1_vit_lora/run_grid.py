import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_root", type=str, default="results/q1/lora_grid")
    parser.add_argument("--weights_root", type=str, default="weights/q1/grid")
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    ranks = [2, 4, 8]
    alphas = [2, 4, 8]
    dropout = 0.1

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    Path(args.weights_root).mkdir(parents=True, exist_ok=True)

    summaries = []

    for rank, alpha in itertools.product(ranks, alphas):
        exp_name = f"r{rank}_a{alpha}_d{dropout}"
        out_dir = Path(args.output_root) / exp_name
        cmd = [
            sys.executable,
            "-m",
            "src.q1_vit_lora.train_lora",
            "--data_root",
            args.data_root,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--rank",
            str(rank),
            "--alpha",
            str(alpha),
            "--dropout",
            str(dropout),
            "--output_dir",
            str(out_dir),
            "--save_weights_dir",
            args.weights_root,
        ]

        if args.use_wandb:
            cmd.append("--use_wandb")

        print(f"\n{'='*60}")
        print(f"Running experiment: {exp_name}")
        print(f"{'='*60}")
        print("Command:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            summaries.append(pd.read_json(summary_path, typ="series"))

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_df = summary_df[
            [
                "lora_layers",
                "rank",
                "alpha",
                "dropout",
                "overall_test_accuracy",
                "trainable_parameters",
                "best_weights",
            ]
        ]
        summary_df.to_csv(Path(args.output_root) / "q1_test_results_table.csv", index=False)
        print(f"\n{'='*60}")
        print("All experiments complete!")
        print(f"Results saved to: {Path(args.output_root) / 'q1_test_results_table.csv'}")
        print(f"{'='*60}")



if __name__ == "__main__":
    main()
