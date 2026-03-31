#!/usr/bin/env python3
"""
Fine-tune Whisper model on collected STT corrections.
Usage: python3 finetune_whisper.py [--version v2] [--epochs 3] [--batch_size 8]

Corrections are collected automatically when users edit auto-transcribed text.
Data location: stt_dataset/corrections.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import soundfile as sf
import numpy as np
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union

BASE_DIR = Path(__file__).parent
STT_DATASET_DIR = BASE_DIR / "stt_dataset"
WHISPER_MODELS_DIR = BASE_DIR / "models" / "whisper"


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def load_corrections():
    """Load correction pairs from JSONL."""
    corrections_file = STT_DATASET_DIR / "corrections.jsonl"
    if not corrections_file.exists():
        print("No corrections found. Use the web UI to transcribe and correct audio.")
        sys.exit(1)

    entries = []
    for line in corrections_file.read_text().strip().split("\n"):
        if not line.strip():
            continue
        entry = json.loads(line)
        audio_path = STT_DATASET_DIR / entry["audio"]
        if audio_path.exists():
            entries.append({
                "audio": str(audio_path),
                "text": entry["corrected_text"],
            })

    if len(entries) < 10:
        print(f"Only {len(entries)} corrections found. Need at least 10 for fine-tuning.")
        print("Keep using Voice to Text and correcting errors to collect more data.")
        sys.exit(1)

    print(f"Loaded {len(entries)} correction pairs")
    return entries


def prepare_dataset(entries, processor):
    """Convert entries to HuggingFace dataset format."""
    dataset = Dataset.from_list(entries)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def prepare_example(example):
        audio = example["audio"]
        example["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        example["labels"] = processor.tokenizer(example["text"]).input_ids
        return example

    dataset = dataset.map(prepare_example, remove_columns=dataset.column_names)
    return dataset


def finetune(version="v2", epochs=3, batch_size=4, lr=1e-5):
    """Fine-tune Whisper on corrections."""
    entries = load_corrections()

    output_dir = WHISPER_MODELS_DIR / version
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Whisper large-v3 for fine-tuning...")
    model_name = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.language = "vi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    print("Preparing dataset...")
    dataset = prepare_dataset(entries, processor)

    # Split: 90% train, 10% eval
    split = dataset.train_test_split(test_size=0.1, seed=42)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        warmup_steps=50,
        num_train_epochs=epochs,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="tensorboard",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    print(f"\nStarting fine-tune:")
    print(f"  Version: {version}")
    print(f"  Train samples: {len(split['train'])}")
    print(f"  Eval samples: {len(split['test'])}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size} (x2 grad accum)")
    print(f"  Learning rate: {lr}")

    trainer.train()

    # Save final model
    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))

    # Convert to CTranslate2 format for faster-whisper
    print("\nConverting to CTranslate2 format for faster-whisper...")
    import subprocess as sp
    ct2_dir = output_dir / "ct2"
    result = sp.run([
        sys.executable, "-m", "ctranslate2.converters.transformers",
        "--model", str(output_dir / "final"),
        "--output_dir", str(ct2_dir),
        "--quantization", "float16",
    ], capture_output=True, text=True)

    if result.returncode == 0 and ct2_dir.exists():
        # Only set active if conversion succeeded
        active_file = WHISPER_MODELS_DIR / "active_version.txt"
        active_file.write_text(version)
        print(f"\nFine-tune complete! Model saved to {ct2_dir}")
        print(f"Active version set to: {version}")
        print("Restart the app to use the new model.")
    else:
        print(f"\nCT2 conversion failed: {result.stderr}")
        print(f"HuggingFace model saved to {output_dir / 'final'}")
        print("NOT setting as active. Convert manually and restart.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on STT corrections")
    parser.add_argument("--version", default="v2", help="Version name")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    args = parser.parse_args()

    finetune(args.version, args.epochs, args.batch_size, args.lr)
