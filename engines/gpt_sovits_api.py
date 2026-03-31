"""
GPT-SoVITS API wrapper for Voice Clone app.
Runs as a separate FastAPI service on port 7861.
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import uvicorn

BASE_DIR = Path(__file__).parent.parent
SOVITS_DIR = Path(__file__).parent / "GPT-SoVITS"
VOICE_LIBRARY_DIR = BASE_DIR / "voice_library"
TRAIN_OUTPUT_DIR = BASE_DIR / "training_output"
TEMP_DIR = BASE_DIR / "temp"

for d in [VOICE_LIBRARY_DIR, TRAIN_OUTPUT_DIR, TEMP_DIR]:
    d.mkdir(exist_ok=True)

# Add GPT-SoVITS to path
sys.path.insert(0, str(SOVITS_DIR))

app = FastAPI(title="GPT-SoVITS API")

# Global model state
sovits_model = None


def get_voice_library():
    """List all trained voices in the library."""
    voices = []
    for voice_dir in sorted(VOICE_LIBRARY_DIR.iterdir()):
        if voice_dir.is_dir():
            meta_file = voice_dir / "meta.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                meta["id"] = voice_dir.name
                voices.append(meta)
    return voices


def get_voice_info(voice_id):
    """Get voice info by ID."""
    voice_dir = VOICE_LIBRARY_DIR / voice_id
    meta_file = voice_dir / "meta.json"
    if not meta_file.exists():
        return None
    meta = json.loads(meta_file.read_text())
    meta["id"] = voice_id
    meta["gpt_path"] = str(voice_dir / "gpt.ckpt")
    meta["sovits_path"] = str(voice_dir / "sovits.pth")
    meta["ref_audio"] = str(voice_dir / "ref.wav")
    return meta


@app.get("/voices")
def list_voices():
    return get_voice_library()


@app.post("/train")
async def train_voice(
    name: str = Form(...),
    description: str = Form(""),
    audio: UploadFile = File(...),
    transcript: str = Form(...),
):
    """Train a new voice from audio + transcript."""
    voice_id = name.strip().lower().replace(" ", "_")
    voice_dir = VOICE_LIBRARY_DIR / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded audio
    audio_path = voice_dir / "raw_audio.wav"
    with open(audio_path, "wb") as f:
        content = await audio.read()
        f.write(content)

    # Save reference audio (first 10s for inference)
    data, sr = sf.read(str(audio_path))
    ref_duration = min(len(data), sr * 15)  # Max 15s for ref
    sf.write(str(voice_dir / "ref.wav"), data[:ref_duration], sr)

    # Save metadata
    meta = {
        "name": name.strip(),
        "description": description,
        "transcript": transcript,
        "created_at": datetime.now().isoformat(),
        "status": "training",
        "duration": round(len(data) / sr, 1),
    }
    (voice_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    # Create training annotation file
    list_file = voice_dir / "train.list"
    list_file.write_text(f"{audio_path}|{name}|vi|{transcript}\n")

    # Start training in background
    train_script = BASE_DIR / "engines" / "train_voice.sh"
    subprocess.Popen(
        ["bash", str(train_script), voice_id, str(voice_dir)],
        stdout=open(voice_dir / "train.log", "w"),
        stderr=subprocess.STDOUT,
    )

    return {"status": "training_started", "voice_id": voice_id}


@app.post("/infer")
async def infer(
    voice_id: str = Form(...),
    text: str = Form(...),
    speed: float = Form(1.0),
):
    """Generate speech using a trained voice."""
    voice = get_voice_info(voice_id)
    if not voice:
        raise HTTPException(404, f"Voice '{voice_id}' not found")

    if voice.get("status") != "ready":
        raise HTTPException(400, f"Voice '{voice_id}' is not ready (status: {voice.get('status')})")

    output_path = TEMP_DIR / f"output_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"

    try:
        # Use GPT-SoVITS CLI for inference
        cmd = [
            sys.executable, str(SOVITS_DIR / "GPT_SoVITS" / "inference_cli.py"),
            "--gpt_model", voice["gpt_path"],
            "--sovits_model", voice["sovits_path"],
            "--ref_audio", voice["ref_audio"],
            "--ref_text", voice.get("transcript", ""),
            "--ref_language", "vi",
            "--target_text", text,
            "--target_language", "vi",
            "--output", str(output_path),
            "--speed", str(speed),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise Exception(result.stderr)

        return FileResponse(str(output_path), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/voices/{voice_id}")
def delete_voice(voice_id: str):
    voice_dir = VOICE_LIBRARY_DIR / voice_id
    if voice_dir.exists():
        shutil.rmtree(voice_dir)
        return {"status": "deleted"}
    raise HTTPException(404, "Voice not found")


@app.get("/voices/{voice_id}/status")
def voice_status(voice_id: str):
    voice = get_voice_info(voice_id)
    if not voice:
        raise HTTPException(404, "Voice not found")
    return {"status": voice.get("status"), "name": voice.get("name")}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7861)
