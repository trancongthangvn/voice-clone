#!/usr/bin/env python3
"""
GPT-SoVITS inference script for Voice Clone app.
Loads model and generates speech from text.
Usage: python3 sovits_infer.py --gpt_path X --sovits_path X --ref_audio X --ref_text X --text X --output X
"""

import argparse
import os
import sys

# Add GPT-SoVITS to path
SOVITS_DIR = os.path.join(os.path.dirname(__file__), "GPT-SoVITS")
sys.path.insert(0, SOVITS_DIR)
os.chdir(SOVITS_DIR)

import soundfile as sf
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_path", required=True)
    parser.add_argument("--sovits_path", required=True)
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--ref_text", default="")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    # Build config
    config_path = os.path.join(SOVITS_DIR, "GPT_SoVITS", "configs", "tts_infer.yaml")
    tts_config = TTS_Config(config_path)
    tts_config.t2s_weights_path = args.gpt_path
    tts_config.vits_weights_path = args.sovits_path
    tts_config.device = "cuda"

    # Init TTS engine
    tts = TTS(tts_config)

    # Use "auto" for language detection (works for Vietnamese too)
    inputs = {
        "text": args.text,
        "text_lang": "auto",
        "ref_audio_path": args.ref_audio,
        "prompt_text": args.ref_text,
        "prompt_lang": "auto",
        "speed_factor": args.speed,
        "text_split_method": "cut5",
        "batch_size": 1,
        "seed": -1,
    }

    # Generate
    sr = None
    audio_chunks = []
    for chunk in tts.run(inputs):
        sr = chunk.get("sampling_rate", 32000)
        audio_data = chunk.get("data")
        if audio_data is not None:
            if hasattr(audio_data, 'numpy'):
                audio_data = audio_data.numpy()
            audio_chunks.append(audio_data.flatten())

    if audio_chunks:
        final_audio = np.concatenate(audio_chunks)
        sf.write(args.output, final_audio, sr)
        print(f"OK: saved to {args.output}")
    else:
        print("ERROR: no audio generated", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
