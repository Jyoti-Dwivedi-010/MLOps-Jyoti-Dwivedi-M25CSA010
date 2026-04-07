import argparse
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi

from src.q1_vit_lora.modeling import apply_lora_to_vit, create_vit_small_for_cifar100



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--alpha", type=int, required=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--message", type=str, default="Upload best LoRA weights for Assignment-5 Q1")
    args = parser.parse_args()

    print(f"Loading model with rank={args.rank}, alpha={args.alpha}, dropout={args.dropout}")
    model = create_vit_small_for_cifar100()
    for p in model.parameters():
        p.requires_grad = False
    model = apply_lora_to_vit(model, args.rank, args.alpha, args.dropout)
    state = torch.load(args.weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    out_dir = Path("tmp_hf_q1_export")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "pytorch_model.bin")

    readme = (
        "# Assignment-5 Q1 Best LoRA Model\n\n"
        "This repository contains LoRA adapted ViT-S weights for CIFAR-100.\n\n"
        f"## Model Configuration\n"
        f"- Rank: {args.rank}\n"
        f"- Alpha: {args.alpha}\n"
        f"- Dropout: {args.dropout}\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Uploading to HuggingFace: {args.repo_id}")
    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=str(out_dir),
        commit_message=args.message,
    )
    print(f"Successfully uploaded to {args.repo_id}")
    
    # Cleanup temp directory
    shutil.rmtree(out_dir, ignore_errors=True)
    print("Cleaned up temporary files.")



if __name__ == "__main__":
    main()
