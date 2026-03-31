#!/usr/bin/env python3
"""
Fine-tune F5-TTS Vietnamese model with collected dataset.
Usage: python3 finetune.py --version v2
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
BASE_MODEL_DIR = MODELS_DIR / "f5-tts-vietnamese"


def prepare_dataset():
    """Convert collected samples to F5-TTS training format."""
    meta_file = DATASET_DIR / "metadata.jsonl"
    if not meta_file.exists():
        print("No training data found. Use the app to collect audio samples first.")
        sys.exit(1)

    entries = []
    for line in meta_file.read_text().strip().split("\n"):
        entry = json.loads(line)
        audio_path = DATASET_DIR / entry["audio"]
        if audio_path.exists() and entry.get("transcript"):
            entries.append(entry)

    if len(entries) < 10:
        print(f"Only {len(entries)} samples with transcripts found.")
        print("Need at least 10 samples with transcripts for fine-tuning.")
        print("Upload more audio with transcripts via the web UI.")
        sys.exit(1)

    # Create training metadata in F5-TTS format
    train_file = DATASET_DIR / "train.txt"
    with open(train_file, "w") as f:
        for entry in entries:
            audio_path = DATASET_DIR / entry["audio"]
            f.write(f"{audio_path}|{entry['transcript']}\n")

    total_dur = sum(e["duration"] for e in entries)
    print(f"Prepared {len(entries)} samples ({total_dur/60:.1f} min) for training")
    return str(train_file)


def finetune(version, epochs=100, batch_size=3200, lr=1e-5):
    """Run fine-tuning using F5-TTS CLI."""
    train_file = prepare_dataset()
    output_dir = MODELS_DIR / version
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy vocab
    shutil.copy2(BASE_MODEL_DIR / "vocab.txt", output_dir / "vocab.txt")

    base_ckpt = str(BASE_MODEL_DIR / "model_last.pt")

    print(f"\nStarting fine-tune:")
    print(f"  Base model: {base_ckpt}")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size} frames")
    print(f"  Learning rate: {lr}")

    cmd = [
        sys.executable, "-m", "f5_tts.train.finetune_cli",
        "--exp_name", "F5TTS_Base",
        "--dataset_name", str(DATASET_DIR),
        "--pretrain", base_ckpt,
        "--finetune",
        "--tokenizer", "custom",
        "--tokenizer_path", str(BASE_MODEL_DIR / "vocab.txt"),
        "--epochs", str(epochs),
        "--batch_size_per_gpu", str(batch_size),
        "--learning_rate", str(lr),
        "--save_per_updates", "500",
        "--logger", "tensorboard",
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, cwd=str(BASE_DIR))
        if result.returncode == 0:
            # Find the latest checkpoint
            ckpts = sorted(output_dir.glob("model_*.pt"))
            if ckpts:
                latest = ckpts[-1]
                shutil.copy2(latest, output_dir / "model_last.pt")
                print(f"\nFine-tune complete! Model saved to {output_dir}")
                print(f"To activate: go to 'Training & Model' tab -> select '{version}' -> 'Chuyển model'")
            else:
                print("\nTraining completed but no checkpoint found.")
                print("Check the F5-TTS output directory for the checkpoint and copy it manually.")
        else:
            print(f"\nTraining failed with exit code {result.returncode}")
    except Exception as e:
        print(f"\nError during fine-tuning: {e}")
        print("\nAlternative: Use F5-TTS CLI directly:")
        print(f"  f5-tts_finetune-cli --pretrain {base_ckpt} --dataset_name {DATASET_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune F5-TTS Vietnamese")
    parser.add_argument("--version", required=True, help="Version name, e.g. v2, v3")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1600, help="Batch size in frames")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    args = parser.parse_args()

    finetune(args.version, args.epochs, args.batch_size, args.lr)
