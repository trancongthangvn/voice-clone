import json
import shutil
import uuid
import subprocess
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torchaudio
import torch
import soundfile as sf

import psutil
from infrastructure import (
    models, gpu_queue, logger, health_check, get_gpu_info,
    run_gpu_inference, check_gpu_memory, preprocess_audio,
    validate_audio, start_memory_monitor,
)


# Monkey-patch torchaudio.load to use soundfile instead of torchcodec
def _load_soundfile(filepath, frame_offset=0, num_frames=-1, normalize=True,
                    channels_first=True, format=None, buffer_size=4096, backend=None):
    data, sr = sf.read(str(filepath), dtype="float32")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    else:
        data = data.T
    if frame_offset > 0:
        data = data[:, frame_offset:]
    if num_frames > 0:
        data = data[:, :num_frames]
    return torch.from_numpy(data), sr


torchaudio.load = _load_soundfile

import gradio as gr
from f5_tts.api import F5TTS

# Paths
BASE_DIR = Path(__file__).parent
VOICES_DIR = BASE_DIR / "voices"
OUTPUT_DIR = BASE_DIR / "outputs"
DATASET_DIR = BASE_DIR / "dataset"
HISTORY_FILE = BASE_DIR / "history.json"
MODELS_DIR = BASE_DIR / "models"
MODEL_DIR = MODELS_DIR / "f5-tts-vietnamese"
VOICE_LIBRARY_DIR = BASE_DIR / "voice_library"
STT_DATASET_DIR = BASE_DIR / "stt_dataset"
WHISPER_MODELS_DIR = MODELS_DIR / "whisper"

for d in [VOICES_DIR, OUTPUT_DIR, DATASET_DIR, VOICE_LIBRARY_DIR, STT_DATASET_DIR, WHISPER_MODELS_DIR]:
    d.mkdir(exist_ok=True)

# Model version tracking
CURRENT_MODEL_VERSION = "v1-base"


# ============================================================
# F5-TTS Engine
# ============================================================

def get_active_model_path():
    active_file = MODELS_DIR / "active_version.txt"
    if active_file.exists():
        version = active_file.read_text().strip()
        versioned = MODELS_DIR / version / "model_last.pt"
        if versioned.exists():
            return versioned, version
    return MODEL_DIR / "model_last.pt", "v1-base"


def _load_tts_impl():
    """Internal TTS loader - called by ModelManager."""
    ckpt_file, version = get_active_model_path()
    vocab_file = MODEL_DIR / "vocab.txt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if ckpt_file.exists():
        m = F5TTS(model="F5TTS_Base", ckpt_file=str(ckpt_file),
                  vocab_file=str(vocab_file), device=device)
    else:
        m = F5TTS(device=device)
        version = "default"
    return m, version


def load_model(force_reload=False):
    global CURRENT_MODEL_VERSION
    _, target_version = get_active_model_path()
    model, version = models.get_tts(_load_tts_impl, force_reload=force_reload,
                                     target_version=target_version)
    CURRENT_MODEL_VERSION = version
    return model


# ============================================================
# History
# ============================================================

def load_history():
    try:
        if HISTORY_FILE.exists():
            return json.loads(HISTORY_FILE.read_text())
    except Exception:
        pass
    return []


def save_history_entry(entry):
    try:
        history = load_history()
        history.insert(0, entry)
        history = history[:100]
        HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2))
    except Exception:
        pass


# ============================================================
# Dataset collection
# ============================================================

def collect_training_data(audio_path, transcript):
    if not audio_path:
        return
    try:
        sample_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        dest_audio = DATASET_DIR / f"{sample_id}.wav"
        data, sr = sf.read(str(audio_path))
        duration = len(data) / sr
        if duration < 1 or duration > 60:
            return
        sf.write(str(dest_audio), data, sr)
        meta = {
            "id": sample_id, "audio": dest_audio.name,
            "transcript": transcript or "", "duration": round(duration, 2),
            "collected_at": datetime.now().isoformat(),
        }
        with open(DATASET_DIR / "metadata.jsonl", "a") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ============================================================
# Voice management (F5-TTS quick clone)
# ============================================================

def save_voice(audio_path, voice_name):
    if not audio_path or not voice_name:
        return "Vui lòng upload audio và nhập tên giọng nói."
    voice_name = voice_name.strip().replace(" ", "_")
    dest = VOICES_DIR / f"{voice_name}.wav"
    data, sr = sf.read(audio_path)
    sf.write(str(dest), data, sr)
    collect_training_data(audio_path, "")
    return f"Đã lưu giọng nói: {voice_name}"


def get_saved_voices():
    return [v.stem for v in sorted(VOICES_DIR.glob("*.wav"))]


def delete_voice(voice_name):
    if not voice_name:
        return "Chọn giọng cần xóa.", gr.Dropdown(choices=get_saved_voices())
    path = VOICES_DIR / f"{voice_name}.wav"
    if path.exists():
        path.unlink()
    voices = get_saved_voices()
    return f"Đã xóa: {voice_name}", gr.Dropdown(choices=voices, value=voices[0] if voices else None)


# ============================================================
# F5-TTS Inference
# ============================================================

def clone_voice_f5(audio_input, ref_text, gen_text, saved_voice, speed):
    if not gen_text or not gen_text.strip():
        return None, "Vui lòng nhập văn bản cần đọc."

    ref_audio = None
    if audio_input:
        ref_audio = audio_input
    elif saved_voice:
        ref_path = VOICES_DIR / f"{saved_voice}.wav"
        if ref_path.exists():
            ref_audio = str(ref_path)

    if not ref_audio:
        return None, "Vui lòng upload audio mẫu hoặc chọn giọng đã lưu."

    collect_training_data(ref_audio, ref_text)
    output_path = OUTPUT_DIR / f"{uuid.uuid4().hex}.wav"

    def _do_inference():
        ok, msg = check_gpu_memory(min_free_mb=1500)
        if not ok:
            raise RuntimeError(msg)
        model = load_model()
        return model.infer(
            ref_file=ref_audio,
            ref_text=ref_text if ref_text and ref_text.strip() else "",
            gen_text=gen_text.strip(),
            speed=speed,
        )

    try:
        wav, sr, _ = gpu_queue.submit(_do_inference, timeout=200)
        sf.write(str(output_path), wav, sr)
        save_history_entry({
            "time": datetime.now().strftime("%d/%m %H:%M"),
            "text": gen_text.strip()[:80],
            "voice": saved_voice or "upload",
            "output": output_path.name,
            "engine": "F5-TTS",
            "model": CURRENT_MODEL_VERSION,
        })
        return str(output_path), f"Thành công! (F5-TTS {CURRENT_MODEL_VERSION})"
    except Exception as e:
        logger.error(f"F5-TTS inference failed: {e}")
        return None, f"Lỗi: {str(e)}"


def refresh_voices():
    voices = get_saved_voices()
    return gr.Dropdown(choices=voices, value=voices[0] if voices else None)


# ============================================================
# Voice Library (GPT-SoVITS fine-tuned voices)
# ============================================================

def get_library_voices():
    """List all trained voices in the library."""
    voices = []
    try:
        for voice_dir in sorted(VOICE_LIBRARY_DIR.iterdir()):
            try:
                if voice_dir.is_dir():
                    meta_file = voice_dir / "meta.json"
                    if meta_file.exists():
                        meta = json.loads(meta_file.read_text())
                        meta["id"] = voice_dir.name
                        voices.append(meta)
            except (json.JSONDecodeError, FileNotFoundError, OSError):
                continue
    except OSError:
        pass
    return voices


def get_library_voice_choices():
    voices = get_library_voices()
    return [f"{v['name']} ({v.get('status', '?')})" for v in voices]


def get_library_voice_ids():
    voices = get_library_voices()
    return [v['id'] for v in voices]


def library_tts(voice_selection, gen_text, speed):
    """Generate speech using a trained library voice via GPT-SoVITS."""
    if not gen_text or not gen_text.strip():
        return None, "Vui lòng nhập văn bản."
    if not voice_selection:
        return None, "Vui lòng chọn giọng."

    # Extract voice id from selection (single fetch to avoid race condition)
    voices = get_library_voices()
    choices = [f"{v['name']} ({v.get('status', '?')})" for v in voices]
    if voice_selection not in choices:
        return None, "Giọng không hợp lệ."

    voice = voices[choices.index(voice_selection)]
    voice_dir = VOICE_LIBRARY_DIR / voice["id"]

    if voice.get("status") != "ready":
        return None, f"Giọng '{voice['name']}' chưa sẵn sàng (trạng thái: {voice.get('status')})"

    gpt_path = voice_dir / "gpt.ckpt"
    sovits_path = voice_dir / "sovits.pth"
    ref_audio = voice_dir / "ref.wav"

    if not gpt_path.exists() or not sovits_path.exists():
        return None, "Thiếu model file. Cần huấn luyện lại."

    output_path = OUTPUT_DIR / f"lib_{uuid.uuid4().hex}.wav"

    try:
        sovits_venv = BASE_DIR / "engines" / "GPT-SoVITS" / "venv" / "bin" / "python3"
        sovits_dir = BASE_DIR / "engines" / "GPT-SoVITS"
        api_script = BASE_DIR / "engines" / "sovits_infer.py"

        cmd = [
            str(sovits_venv),
            str(api_script),
            "--gpt_path", str(gpt_path),
            "--sovits_path", str(sovits_path),
            "--ref_audio", str(ref_audio),
            "--ref_text", voice.get("transcript", ""),
            "--text", gen_text.strip(),
            "--output", str(output_path),
            "--speed", str(speed),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                                cwd=str(sovits_dir))
        if result.returncode != 0:
            return None, f"Lỗi GPT-SoVITS: {result.stderr[:300]}"

        if output_path.exists():
            save_history_entry({
                "time": datetime.now().strftime("%d/%m %H:%M"),
                "text": gen_text.strip()[:80],
                "voice": voice["name"],
                "output": output_path.name,
                "engine": "GPT-SoVITS",
            })
            return str(output_path), f"Thành công! (GPT-SoVITS - {voice['name']})"
        else:
            return None, "Không tạo được file audio."
    except subprocess.TimeoutExpired:
        return None, "Timeout - text quá dài hoặc model quá chậm."
    except Exception as e:
        return None, f"Lỗi: {str(e)}"


def train_new_voice(audio_file, voice_name, description, transcript, auto_transcript=""):
    """Start training a new voice for the library.
    Also saves correction data if transcript was edited."""
    try:
        return _train_new_voice_impl(audio_file, voice_name, description, transcript, auto_transcript)
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        return f"Lỗi: {str(e)}"


def _train_new_voice_impl(audio_file, voice_name, description, transcript, auto_transcript=""):
    if not audio_file or not voice_name or not transcript:
        return "Vui lòng điền đầy đủ: audio, tên, và transcript."

    # Auto-save STT correction if user edited the auto-generated transcript
    if auto_transcript and transcript.strip() != auto_transcript.strip():
        save_stt_correction(audio_file, auto_transcript, transcript)

    voice_id = voice_name.strip().lower().replace(" ", "_")
    voice_id = re.sub(r'[^a-z0-9_]', '', voice_id)
    if not voice_id:
        return "Tên giọng không hợp lệ (cần chứa ký tự a-z, 0-9)."

    voice_dir = VOICE_LIBRARY_DIR / voice_id
    if voice_dir.exists():
        return f"Giọng '{voice_id}' đã tồn tại. Chọn tên khác."

    info, msg = validate_audio(audio_file, min_duration=10, max_duration=0)
    if info is None:
        return msg

    voice_dir.mkdir(parents=True)

    # Preprocess audio (normalize, mono, trim silence)
    data, sr = preprocess_audio(audio_file)
    sf.write(str(voice_dir / "raw_audio.wav"), data, sr)

    # Save ref audio (first 15s for inference)
    ref_samples = min(len(data), sr * 15)
    sf.write(str(voice_dir / "ref.wav"), data[:ref_samples], sr)

    # Save metadata
    meta = {
        "name": voice_name.strip(),
        "description": description or "",
        "transcript": transcript.strip(),
        "created_at": datetime.now().isoformat(),
        "status": "training",
        "duration": round(info["duration"], 1),
    }
    (voice_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    # Create training list
    clean_transcript = transcript.strip().replace("\n", " ").replace("\r", " ").replace("|", ",")
    clean_transcript = re.sub(r'\s+', ' ', clean_transcript).strip()

    audio_path = voice_dir / "raw_audio.wav"
    duration = info["duration"]

    # Split long audio into ~10s segments (HuBERT OOM on >30s audio)
    MAX_SEGMENT_SEC = 10
    if duration > 30:
        logger.info(f"Splitting {duration:.0f}s audio into ~{MAX_SEGMENT_SEC}s segments...")
        segments_dir = voice_dir / "segments"
        segments_dir.mkdir(exist_ok=True)

        # Split audio into fixed-length chunks
        full_data, full_sr = sf.read(str(audio_path))
        if full_data.ndim > 1:
            full_data = full_data.mean(axis=1)

        segment_samples = int(MAX_SEGMENT_SEC * full_sr)
        n_segments = max(1, len(full_data) // segment_samples)

        # Split transcript proportionally by character count
        sentences = re.split(r'(?<=[.!?。！？,，;；])\s*', clean_transcript)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 3]

        train_lines = []
        for i in range(n_segments):
            start = i * segment_samples
            end = min((i + 1) * segment_samples, len(full_data))
            seg_data = full_data[start:end]

            if len(seg_data) < full_sr * 1:  # skip < 1s
                continue

            seg_path = segments_dir / f"seg_{i:04d}.wav"
            sf.write(str(seg_path), seg_data, full_sr)

            # Assign sentences to this segment proportionally
            seg_start_ratio = i / n_segments
            seg_end_ratio = (i + 1) / n_segments
            sent_start = int(seg_start_ratio * len(sentences))
            sent_end = int(seg_end_ratio * len(sentences))
            seg_text = " ".join(sentences[sent_start:sent_end]).strip()

            if not seg_text:
                seg_text = f"segment {i}"

            train_lines.append(f"{seg_path}|{voice_name}|zh|{seg_text}")

        (voice_dir / "train.list").write_text("\n".join(train_lines) + "\n")
        logger.info(f"Split into {len(train_lines)} segments")
    else:
        # Short audio: single entry
        (voice_dir / "train.list").write_text(
            f"{audio_path}|{voice_name}|zh|{clean_transcript}\n"
        )

    # Start training in background
    train_script = BASE_DIR / "engines" / "train_voice.sh"
    if not train_script.exists():
        return "Lỗi: không tìm thấy script huấn luyện."

    # Free ALL GPU memory for training (F5-TTS + Whisper)
    logger.info("Unloading ALL models to free VRAM for training...")
    models.unload_whisper()
    models.unload_tts()
    torch.cuda.empty_cache()
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # Start training subprocess (daemonized, log to file)
    subprocess.Popen(
        ["bash", str(train_script), voice_id, str(voice_dir)],
        stdout=open(voice_dir / "train.log", "w"),
        stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR / "engines" / "GPT-SoVITS"),
        start_new_session=True,
    )
    logger.info(f"Training started: {voice_id} (Whisper unloaded, will reload on next STT request)")

    return f"Đang huấn luyện giọng '{voice_name}' ({info['duration']:.0f}s audio)... Kiểm tra ở tab Thư viện."


TRAINING_STEPS = [
    ("Step 1/5: Text processing", 10),
    ("Step 2/5: HuBERT feature extraction", 25),
    ("Step 3/5: Semantic token extraction", 40),
    ("Step 4/5: Train SoVITS", 70),
    ("Step 5/5: Train GPT", 95),
    ("Copying final models", 98),
    ("Training COMPLETE", 100),
    ("FAILED", -1),
]


def get_training_progress(voice_id):
    """Parse training log and return progress percentage + current step."""
    log_file = VOICE_LIBRARY_DIR / voice_id / "train.log"
    meta_file = VOICE_LIBRARY_DIR / voice_id / "meta.json"

    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
            if meta.get("status") == "ready":
                return 100, "Hoàn tất"
            if meta.get("status") == "failed":
                return -1, f"Thất bại: {meta.get('error', 'unknown')}"
        except Exception:
            pass

    if not log_file.exists():
        return 0, "Chờ bắt đầu..."

    content = log_file.read_text()
    progress = 0
    current_step = "Đang khởi tạo..."

    for step_name, step_pct in TRAINING_STEPS:
        if step_name in content:
            if step_pct == -1:
                return -1, step_name
            progress = step_pct
            current_step = step_name

    # Try to extract epoch progress from training output
    import re as _re
    epoch_matches = _re.findall(r'Epoch\s+(\d+)/(\d+)', content)
    if epoch_matches:
        current_epoch, total_epochs = int(epoch_matches[-1][0]), int(epoch_matches[-1][1])
        # Determine which training phase we're in
        if progress >= 70:  # GPT training (step 5)
            step_progress = current_epoch / max(total_epochs, 1)
            progress = 70 + int(step_progress * 25)
        elif progress >= 40:  # SoVITS training (step 4)
            step_progress = current_epoch / max(total_epochs, 1)
            progress = 40 + int(step_progress * 30)

    return min(progress, 100), current_step


def get_training_log(voice_selection):
    """Get training log with progress info."""
    if not voice_selection:
        return "Chọn giọng để xem log.", ""
    voices = get_library_voices()
    # Match by name prefix (status in parentheses may have changed)
    voice = None
    for v in voices:
        label = f"{v['name']} ({v.get('status', '?')})"
        if voice_selection == label or voice_selection.startswith(v['name']):
            voice = v
            break
    if not voice:
        return "Giọng không hợp lệ. Bấm 'Refresh DS' để cập nhật.", ""
    voice_id = voice["id"]

    # Get progress
    progress, step = get_training_progress(voice_id)

    if progress == -1:
        progress_bar = (
            f'<div class="progress-bar failed"><div class="fill" style="width:100%"></div></div>'
            f'<div class="progress-info">Thất bại - {step}</div>'
        )
    elif progress >= 100:
        progress_bar = (
            f'<div class="progress-bar done"><div class="fill" style="width:100%"></div></div>'
            f'<div class="progress-info">Hoàn tất!</div>'
        )
    else:
        progress_bar = (
            f'<div class="progress-bar"><div class="fill" style="width:{progress}%"></div></div>'
            f'<div class="progress-info">{progress}% - {step}</div>'
        )

    # Get log tail
    log_file = VOICE_LIBRARY_DIR / voice_id / "train.log"
    if log_file.exists():
        content = log_file.read_text()
        log_tail = content[-3000:] if len(content) > 3000 else content
    else:
        log_tail = "Chưa có log."

    return progress_bar, log_tail


def delete_library_voice(voice_selection):
    if not voice_selection:
        return "Chọn giọng cần xóa.", gr.Dropdown(choices=get_library_voice_choices())
    voices = get_library_voices()
    choices = get_library_voice_choices()
    if voice_selection not in choices:
        return "Giọng không hợp lệ.", gr.Dropdown(choices=choices)
    voice = voices[choices.index(voice_selection)]
    voice_dir = VOICE_LIBRARY_DIR / voice["id"]
    if voice_dir.exists():
        shutil.rmtree(voice_dir)
    new_choices = get_library_voice_choices()
    return f"Đã xóa: {voice['name']}", gr.Dropdown(choices=new_choices, value=new_choices[0] if new_choices else None)


# ============================================================
# History & Stats
# ============================================================

def get_history_display():
    history = load_history()
    if not history:
        return "Chưa có lịch sử."
    lines = []
    for h in history[:20]:
        engine = h.get("engine", "F5-TTS")
        lines.append(f"**{h['time']}** | {engine} | {h.get('voice','?')} | {h['text']}")
    return "\n\n".join(lines)


def get_dataset_stats():
    meta_file = DATASET_DIR / "metadata.jsonl"
    if not meta_file.exists():
        return "Chưa có dữ liệu training."
    lines = [l for l in meta_file.read_text().strip().split("\n") if l.strip()]
    total = len(lines)
    total_duration = 0
    with_transcript = 0
    for l in lines:
        try:
            entry = json.loads(l)
            total_duration += entry.get("duration", 0)
            if entry.get("transcript"):
                with_transcript += 1
        except json.JSONDecodeError:
            continue
    lib_voices = get_library_voices()
    ready = sum(1 for v in lib_voices if v.get("status") == "ready")
    training = sum(1 for v in lib_voices if v.get("status") == "training")
    return (
        f"### F5-TTS Dataset\n"
        f"- Samples: **{total}** ({total_duration/60:.0f} phút)\n"
        f"- Có transcript: {with_transcript}/{total}\n"
        f"- Model: **{CURRENT_MODEL_VERSION}**\n\n"
        f"### Thư viện giọng (GPT-SoVITS)\n"
        f"- Tổng: **{len(lib_voices)}** giọng\n"
        f"- Sẵn dùng: **{ready}** | Đang train: **{training}**"
    )


def get_model_versions():
    versions = ["v1-base"]
    for d in sorted(MODELS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("v") and (d / "model_last.pt").exists():
            if d.name != "f5-tts-vietnamese":
                versions.append(d.name)
    return versions


def switch_model(version):
    if not version:
        return f"Model hiện tại: {CURRENT_MODEL_VERSION}"
    active_file = MODELS_DIR / "active_version.txt"
    if version == "v1-base":
        if active_file.exists():
            active_file.unlink()
    else:
        active_file.write_text(version)
    load_model(force_reload=True)
    return f"Đã chuyển sang: {CURRENT_MODEL_VERSION}"


# ============================================================
# Speech-to-Text (Whisper)
# ============================================================

WHISPER_TRANSCRIBE_OPTS = dict(
    beam_size=5,
    best_of=5,
    patience=1.5,
    vad_filter=True,
    vad_parameters=dict(
        min_silence_duration_ms=250,
        speech_pad_ms=200,
        threshold=0.4,
    ),
    word_timestamps=False,
    condition_on_previous_text=True,
)


def _load_whisper_impl():
    from faster_whisper import WhisperModel

    # Check available VRAM to pick best compute type
    gpu = get_gpu_info()
    free_mb = gpu["memory_free_mb"] if gpu else 4000

    # float16: best quality, ~4.5GB | int8_float16: faster, ~2.5GB
    if free_mb > 5000:
        compute = "float16"
    else:
        compute = "int8_float16"

    logger.info(f"Whisper compute_type={compute} (VRAM free: {free_mb}MB)")

    # Check for fine-tuned model
    active_file = WHISPER_MODELS_DIR / "active_version.txt"
    if active_file.exists():
        version = active_file.read_text().strip()
        ct2_path = WHISPER_MODELS_DIR / version / "ct2"
        if ct2_path.exists():
            return WhisperModel(str(ct2_path), device="cuda", compute_type=compute,
                                num_workers=4, cpu_threads=16)

    return WhisperModel("large-v3", device="cuda", compute_type=compute,
                        num_workers=4, cpu_threads=16)


def load_whisper():
    return models.get_whisper(_load_whisper_impl)


def transcribe_audio(audio_path, language):
    if not audio_path:
        return "", "Vui lòng upload audio."
    try:
        model = load_whisper()
        lang = language if language and language != "Auto" else None
        segments, info = model.transcribe(
            audio_path,
            language=lang,
            **WHISPER_TRANSCRIBE_OPTS,
        )
        full_text = ""
        segment_details = []
        for seg in segments:
            full_text += seg.text + " "
            start = f"{int(seg.start//60):02d}:{seg.start%60:05.2f}"
            end = f"{int(seg.end//60):02d}:{seg.end%60:05.2f}"
            segment_details.append(f"[{start} → {end}] {seg.text.strip()}")

        details = (
            f"**Ngôn ngữ:** {info.language} (confidence: {info.language_probability:.0%})\n"
            f"**Thời lượng:** {info.duration:.1f}s\n\n"
            + "\n".join(segment_details)
        )
        return full_text.strip(), details
    except Exception as e:
        return "", f"Lỗi: {str(e)}"


# ============================================================
# Training Dashboard
# ============================================================

def get_all_training_status():
    """Get status of all voices in library with progress bars."""
    voices = get_library_voices()
    if not voices:
        return "Chưa có giọng nào. Vào phần **Tạo giọng mới** ở trên để bắt đầu."
    parts = []
    for v in voices:
        vid = v.get("id", "")
        name = v.get("name", vid)
        status = v.get("status", "?")
        duration = v.get("duration", 0)

        if status == "ready":
            bar = f'<div class="progress-bar done"><div class="fill" style="width:100%"></div></div>'
            info = f"**{name}** - Sẵn dùng ({duration:.0f}s)"
        elif status == "failed":
            bar = f'<div class="progress-bar failed"><div class="fill" style="width:100%"></div></div>'
            info = f"**{name}** - Thất bại: {v.get('error','')}"
        elif status == "training":
            pct, step = get_training_progress(vid)
            pct = max(pct, 5)
            bar = f'<div class="progress-bar"><div class="fill" style="width:{pct}%"></div></div>'
            info = f"**{name}** - {pct}% {step}"
        else:
            bar = f'<div class="progress-bar"><div class="fill" style="width:0%"></div></div>'
            info = f"**{name}** - {status}"
        parts.append(f"{info}\n{bar}")
    return "\n\n".join(parts)


def get_gpu_status():
    """Get GPU usage info."""
    gpu = get_gpu_info()
    if gpu:
        q_info = f" | Queue: {gpu_queue.queue_size}" if gpu_queue.queue_size > 0 else ""
        return (
            f"**GPU:** {gpu['utilization_pct']}% | "
            f"**VRAM:** {gpu['memory_used_mb']}/{gpu['memory_total_mb']}MB "
            f"({gpu['memory_pct']}%)\n"
            f"**Temp:** {gpu['temperature_c']}°C{q_info}"
        )
    return "GPU: N/A"


def get_disk_status():
    """Get disk usage."""
    try:
        disk = psutil.disk_usage("/")
        return (
            f"**Disk:** {disk.used//(1024**3)}GB / {disk.total//(1024**3)}GB "
            f"(free: {disk.free//(1024**3)}GB)"
        )
    except Exception:
        return "Disk: N/A"


def refresh_dashboard():
    return get_all_training_status(), get_gpu_status(), get_disk_status(), get_dataset_stats()


def auto_transcribe(audio_path):
    """Auto-transcribe when audio is uploaded in Training Dashboard."""
    if not audio_path:
        return "", ""
    try:
        model = load_whisper()
        segments, info = model.transcribe(
            audio_path, language=None,
            **WHISPER_TRANSCRIBE_OPTS,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        status = f"Nhận dạng xong ({info.language}, {info.duration:.0f}s). Kiểm tra và sửa transcript trước khi train."
        return text.strip(), status
    except Exception as e:
        return "", f"Lỗi nhận dạng: {e}"


# ============================================================
# STT Self-learning: collect corrections for fine-tuning
# ============================================================

def save_stt_correction(audio_path, auto_text, corrected_text):
    """Save a correction pair when user edits auto-transcribed text.
    This builds training data for Whisper fine-tuning."""
    if not audio_path or not corrected_text or not corrected_text.strip():
        return "Cần audio và text đã sửa."
    corrected_text = corrected_text.strip()
    auto_text = (auto_text or "").strip()

    # Skip if no change was made
    if corrected_text == auto_text:
        return "Không có thay đổi - không cần lưu."

    try:
        sample_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        dest_audio = STT_DATASET_DIR / f"{sample_id}.wav"
        data, sr = sf.read(str(audio_path))
        sf.write(str(dest_audio), data, sr)
        duration = len(data) / sr

        entry = {
            "id": sample_id,
            "audio": dest_audio.name,
            "auto_text": auto_text,
            "corrected_text": corrected_text,
            "duration": round(duration, 2),
            "saved_at": datetime.now().isoformat(),
        }
        with open(STT_DATASET_DIR / "corrections.jsonl", "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        total = _count_stt_samples()
        return f"Đã lưu correction! Tổng: {total} samples. (Cần ~50+ để fine-tune)"
    except Exception as e:
        return f"Lỗi: {e}"


def _count_stt_samples():
    meta = STT_DATASET_DIR / "corrections.jsonl"
    if not meta.exists():
        return 0
    return sum(1 for l in meta.read_text().strip().split("\n") if l.strip())


def get_stt_stats():
    total = _count_stt_samples()
    if total == 0:
        return "Chưa có dữ liệu correction. Sửa transcript sau khi nhận dạng để hệ thống tự học."

    meta = STT_DATASET_DIR / "corrections.jsonl"
    lines = [l for l in meta.read_text().strip().split("\n") if l.strip()]
    total_dur = 0
    for l in lines:
        try:
            total_dur += json.loads(l).get("duration", 0)
        except Exception:
            pass

    # Check active whisper model
    active_file = WHISPER_MODELS_DIR / "active_version.txt"
    active = active_file.read_text().strip() if active_file.exists() else "base (large-v3)"

    status = "Sẵn sàng fine-tune!" if total >= 50 else f"Cần thêm {50 - total} samples"
    return (
        f"**Corrections:** {total} samples ({total_dur/60:.1f} phút)\n"
        f"**Whisper model:** {active}\n"
        f"**Trạng thái:** {status}"
    )


# ============================================================
# BUILD UI
# ============================================================

CUSTOM_CSS = """
/* ========================================
   Voice Clone - Light/Dark Theme
   Font: SVN Gilroy | Accent: Amber
   ======================================== */

@font-face { font-family: 'SVN Gilroy'; src: url('/static/fonts/SVN-Gilroy_Regular.otf') format('opentype'); font-weight: 400; }
@font-face { font-family: 'SVN Gilroy'; src: url('/static/fonts/SVN-Gilroy_Medium.otf') format('opentype'); font-weight: 500; }
@font-face { font-family: 'SVN Gilroy'; src: url('/static/fonts/SVN-Gilroy_SemiBold.otf') format('opentype'); font-weight: 600; }
@font-face { font-family: 'SVN Gilroy'; src: url('/static/fonts/SVN-Gilroy_Bold.otf') format('opentype'); font-weight: 700; }
@font-face { font-family: 'SVN Gilroy'; src: url('/static/fonts/SVN-Gilroy_Heavy.otf') format('opentype'); font-weight: 800; }

:root {
    --accent: #b45309; --accent-hover: #d97706; --accent-light: rgba(217,119,6,0.08);
    --accent-border: rgba(217,119,6,0.15); --accent-focus: rgba(217,119,6,0.3);
    --secondary: #6366f1; --secondary-light: rgba(99,102,241,0.1);
    --radius: 10px; --radius-sm: 8px;
}

/* Global */
.gradio-container, .gradio-container *:not(code):not(pre) {
    font-family: 'SVN Gilroy', -apple-system, sans-serif !important;
}

/* ── Theme toggle ── */
.theme-toggle {
    width: 34px; height: 34px; border-radius: 8px;
    border: 1px solid rgba(128,128,128,0.15); cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; transition: all 0.15s ease;
    background: transparent; color: inherit;
}
.theme-toggle:hover { background: rgba(128,128,128,0.1); }

/* ── Navbar ── */
.site-navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 24px; margin: -16px -16px 12px -16px;
    border-bottom: 1px solid rgba(128,128,128,0.12);
}
.site-navbar .nav-left {
    display: flex; align-items: center; gap: 10px;
}
.site-navbar .nav-logo {
    width: 36px; height: 36px; border-radius: 10px;
    background: var(--accent); display: flex; align-items: center;
    justify-content: center; color: #fff; font-weight: 800; font-size: 16px;
}
.site-navbar .nav-brand {
    font-size: 1.1em; font-weight: 700; letter-spacing: -0.02em;
}
.site-navbar .nav-brand span {
    font-weight: 400; opacity: 0.4; font-size: 0.75em; margin-left: 6px;
}
.site-navbar .nav-right {
    display: flex; align-items: center; gap: 8px;
}
.site-navbar .nav-badge {
    padding: 3px 10px; border-radius: 20px; font-size: 0.68em; font-weight: 500;
    background: var(--accent-light); color: var(--accent-hover);
    border: 1px solid var(--accent-border);
}
.site-navbar .nav-status {
    padding: 3px 10px; border-radius: 20px; font-size: 0.68em; font-weight: 500;
    background: rgba(34,197,94,0.08); color: #22c55e;
    border: 1px solid rgba(34,197,94,0.15);
    display: flex; align-items: center; gap: 4px;
}
.site-navbar .nav-status .dot {
    width: 6px; height: 6px; border-radius: 50%; background: #22c55e;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
}

/* ── Tabs ── */
.tabs > .tab-nav { border-bottom: 1px solid rgba(128,128,128,0.15) !important; }
.tabs > .tab-nav > button {
    font-weight: 500 !important; font-size: 0.88em !important;
    padding: 10px 16px !important; border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    transition: all 0.2s ease !important;
    border-bottom: 2px solid transparent !important; opacity: 0.55;
}
.tabs > .tab-nav > button.selected {
    color: var(--accent-hover) !important; opacity: 1;
    border-bottom: 2px solid var(--accent-hover) !important;
}
.tabs > .tab-nav > button:hover:not(.selected) { opacity: 0.8; }

/* ── Buttons ── */
.primary {
    background: var(--accent) !important; border: none !important;
    font-weight: 600 !important; color: #fff !important;
    border-radius: var(--radius-sm) !important; transition: all 0.15s ease !important;
}
.primary:hover {
    background: var(--accent-hover) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(217,119,6,0.18) !important;
}
.primary:active { transform: translateY(0) !important; }

.secondary {
    background: var(--secondary-light) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: var(--radius-sm) !important; transition: all 0.15s ease !important;
}
.secondary:hover { background: rgba(99,102,241,0.18) !important; }

button.stop { border-radius: var(--radius-sm) !important; transition: all 0.15s ease !important; }
button[disabled] { opacity: 0.5 !important; cursor: wait !important; }

/* ── Panels ── */
.panel, .block { border-radius: var(--radius) !important; }
.audio-container { border-radius: var(--radius) !important; }

/* ── Inputs ── */
textarea, input[type="text"] {
    border-radius: var(--radius-sm) !important; transition: border-color 0.15s ease !important;
}
textarea:focus, input[type="text"]:focus { border-color: var(--accent-focus) !important; }
.wrap .options { border-radius: var(--radius-sm) !important; }

/* ── Progress bar ── */
.progress-bar {
    width: 100%; height: 8px; border-radius: 4px;
    background: rgba(128,128,128,0.15); overflow: hidden; margin: 8px 0;
}
.progress-bar .fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, var(--accent), var(--accent-hover));
    transition: width 0.4s ease;
}
.progress-bar.done .fill { background: #22c55e; }
.progress-bar.failed .fill { background: #ef4444; }
.progress-info { font-size: 0.85em; opacity: 0.7; margin-top: 2px; }

/* ── Markdown ── */
.markdown-text h3 { font-weight: 600 !important; margin-bottom: 0.5em !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(128,128,128,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(128,128,128,0.35); }

/* ── Focus ── */
*:focus-visible { outline: 1.5px solid var(--accent-focus) !important; outline-offset: 2px !important; }

/* ── Accordion ── */
.accordion .label-wrap { font-weight: 500 !important; }

/* ═══════════════════════════════════
   Light Mode Override
   ═══════════════════════════════════ */
body.light-mode .gradio-container {
    background: #f8f9fb !important;
    color: #1e293b !important;
}
body.light-mode .site-navbar { border-bottom-color: #e2e8f0 !important; }
body.light-mode .site-navbar .nav-brand { color: #0f172a !important; }
body.light-mode .panel, body.light-mode .block {
    background: #fff !important;
    border: 1px solid #e2e8f0 !important;
}
body.light-mode textarea, body.light-mode input[type="text"] {
    background: #fff !important; color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
}
body.light-mode textarea:focus, body.light-mode input[type="text"]:focus {
    border-color: var(--accent) !important;
}
body.light-mode .tabs > .tab-nav {
    border-bottom: 1px solid #e2e8f0 !important;
}
body.light-mode .tabs > .tab-nav > button { color: #64748b !important; }
body.light-mode .tabs > .tab-nav > button.selected {
    color: var(--accent) !important;
    background: rgba(217,119,6,0.06) !important;
}
body.light-mode .markdown-text, body.light-mode .markdown-text h3 {
    color: #1e293b !important;
}
body.light-mode .secondary {
    background: rgba(99,102,241,0.08) !important;
    color: #4f46e5 !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
}
body.light-mode .progress-bar { background: #e2e8f0; }
body.light-mode ::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.12); }
body.light-mode .theme-toggle {
    background: #fff; border: 1px solid #e2e8f0;
    color: #1e293b;
}
"""

# JS for URL-based tab routing: /tts, /library, /stt, /train, /history
CUSTOM_JS = """
() => {
    const ROUTES = ['/tts', '/library', '/stt', '/train', '/history'];
    const LABELS = ['Text to Speech', 'Thư viện giọng', 'Nhận dạng giọng nói', 'Huấn luyện', 'Lịch sử'];

    function findTabButton(label) {
        var btns = document.querySelectorAll('button');
        for (var i = 0; i < btns.length; i++) {
            if (btns[i].textContent.trim() === label) return btns[i];
        }
        return null;
    }

    // Watch clicks on tab buttons -> update URL
    function watchTabs() {
        var found = 0;
        LABELS.forEach((label, i) => {
            var btn = findTabButton(label);
            if (btn) {
                found++;
                btn.addEventListener('click', () => {
                    if (window.location.pathname !== ROUTES[i]) {
                        window.history.pushState(null, '', ROUTES[i]);
                        document.title = LABELS[i] + ' - Voice Clone';
                    }
                });
            }
        });
        if (found < LABELS.length) setTimeout(watchTabs, 500);
    }
    setTimeout(watchTabs, 1000);

    window.addEventListener('popstate', () => {
        var i = ROUTES.indexOf(window.location.pathname);
        if (i >= 0) { var btn = findTabButton(LABELS[i]); if (btn) btn.click(); }
    });
}
"""

with gr.Blocks(title="Voice Clone - Overmind") as app:
    gr.HTML("""
        <nav class='site-navbar'>
            <div class='nav-left'>
                <div class='nav-logo'>VC</div>
                <div class='nav-brand'>Voice Clone <span>by Overmind</span></div>
            </div>
            <div class='nav-right'>
                <span class='nav-status'><span class='dot'></span> Online</span>
                <span class='nav-badge'>F5-TTS</span>
                <span class='nav-badge'>GPT-SoVITS</span>
                <span class='nav-badge'>Whisper</span>
                <button id='theme-toggle-btn' class='theme-toggle'
                    onclick="
                        const isDark = document.body.classList.toggle('light-mode');
                        this.textContent = isDark ? '☽' : '☀';
                        localStorage.setItem('vc-theme', isDark ? 'light' : 'dark');
                    ">☀</button>
            </div>
        </nav>
        <script>
            if (localStorage.getItem('vc-theme') === 'light') {
                document.body.classList.add('light-mode');
                var btn = document.getElementById('theme-toggle-btn');
                if (btn) btn.textContent = '☽';
            }
        </script>
    """)

    with gr.Tabs():
        # ═══════════════════════════════════════
        # Tab 1: Clone nhanh
        # ═══════════════════════════════════════
        with gr.Tab("Text to Speech"):
            gr.Markdown("Upload giọng mẫu hoặc chọn giọng có sẵn, nhập văn bản cần đọc.")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        label="Giọng mẫu (15-30s, nói rõ ràng, không nhạc nền)",
                        type="filepath", sources=["upload", "microphone"],
                    )
                    with gr.Row():
                        saved_voice = gr.Dropdown(
                            label="Hoặc chọn giọng đã lưu",
                            choices=get_saved_voices(), interactive=True, scale=4,
                        )
                        refresh_btn = gr.Button("Refresh", size="sm", scale=1)
                    with gr.Accordion("Transcript giọng mẫu (tùy chọn)", open=False):
                        ref_text = gr.Textbox(
                            label="Nội dung audio mẫu đang nói",
                            placeholder="Nhập chính xác → chất lượng clone tốt hơn",
                            lines=2,
                        )

                with gr.Column(scale=1):
                    gen_text = gr.Textbox(
                        label="Văn bản cần đọc", lines=6,
                        placeholder="Nhập nội dung bạn muốn giọng clone đọc...",
                    )
                    speed = gr.Slider(0.5, 2.0, 1.0, 0.1, label="Tốc độ đọc")
                    gen_btn = gr.Button("Tạo Audio", variant="primary", size="lg")
                    output_audio = gr.Audio(label="Kết quả", type="filepath")
                    status_text = gr.Textbox(label="Trạng thái", interactive=False, max_lines=1)

            gen_btn.click(clone_voice_f5,
                          [audio_input, ref_text, gen_text, saved_voice, speed],
                          [output_audio, status_text])
            refresh_btn.click(refresh_voices, outputs=[saved_voice])

        # ═══════════════════════════════════════
        # Tab 2: Thư viện giọng
        # ═══════════════════════════════════════
        with gr.Tab("Thư viện giọng"):
            lib_voices = get_library_voice_choices()
            if not lib_voices:
                gr.Markdown(
                    "**Chưa có giọng nào trong thư viện.**\n\n"
                    "Vào tab **Huấn luyện** để tạo giọng đầu tiên."
                )
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    lib_voice_dd = gr.Dropdown(
                        label="Chọn giọng đã huấn luyện",
                        choices=lib_voices, interactive=True,
                    )
                    lib_refresh = gr.Button("Refresh danh sách", size="sm")
                    lib_text = gr.Textbox(label="Văn bản cần đọc", lines=6,
                                          placeholder="Nhập nội dung cần đọc bằng giọng đã chọn...")
                    lib_speed = gr.Slider(0.5, 2.0, 1.0, 0.1, label="Tốc độ")
                    lib_gen_btn = gr.Button("Tạo Audio", variant="primary", size="lg")
                with gr.Column(scale=1):
                    lib_output = gr.Audio(label="Kết quả", type="filepath")
                    lib_status = gr.Textbox(label="Trạng thái", interactive=False, max_lines=2)

            lib_gen_btn.click(library_tts,
                              [lib_voice_dd, lib_text, lib_speed],
                              [lib_output, lib_status])
            def refresh_lib_voices():
                choices = get_library_voice_choices()
                return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
            lib_refresh.click(refresh_lib_voices, outputs=[lib_voice_dd])

        # ═══════════════════════════════════════
        # Tab 3: Nhận dạng giọng nói
        # ═══════════════════════════════════════
        with gr.Tab("Nhận dạng giọng nói"):
            gr.Markdown("Chuyển audio thành văn bản. Sửa kết quả sai giúp AI tự học, càng dùng càng chuẩn.")
            stt_auto_text = gr.State("")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    stt_audio = gr.Audio(
                        label="Upload audio",
                        type="filepath", sources=["upload", "microphone"],
                    )
                    stt_lang = gr.Dropdown(
                        label="Ngôn ngữ", value="Auto", interactive=True,
                        choices=["Auto", "vi", "en", "zh", "ja", "ko", "fr"],
                    )
                    stt_btn = gr.Button("Nhận dạng", variant="primary", size="lg")
                with gr.Column(scale=1):
                    stt_result = gr.Textbox(
                        label="Kết quả (sửa nếu sai)",
                        lines=10, interactive=True,
                    )
                    stt_details = gr.Markdown()

            with gr.Row():
                stt_save_btn = gr.Button("Lưu bản sửa (giúp AI học)", variant="secondary")
                stt_save_status = gr.Textbox(show_label=False, interactive=False, max_lines=1)
            stt_stats = gr.Markdown(value=get_stt_stats)

            def stt_and_store(audio, lang):
                text, details = transcribe_audio(audio, lang)
                return text, details, text

            stt_btn.click(stt_and_store, [stt_audio, stt_lang],
                          [stt_result, stt_details, stt_auto_text])
            stt_save_btn.click(save_stt_correction,
                               [stt_audio, stt_auto_text, stt_result],
                               [stt_save_status])

        # ═══════════════════════════════════════
        # Tab 4: Huấn luyện
        # ═══════════════════════════════════════
        with gr.Tab("Huấn luyện"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### Tạo giọng mới")
                    train_audio = gr.Audio(label="Audio giọng nói (sạch, không nhạc nền)",
                                           type="filepath", sources=["upload"])
                    with gr.Row():
                        train_name = gr.Textbox(label="Tên giọng", placeholder="vd: MC Thời sự", scale=2)
                        train_desc = gr.Textbox(label="Mô tả", placeholder="Nam, miền Bắc...", scale=2)
                    stt_train_btn = gr.Button("Nhận dạng audio thành text", variant="secondary")
                    train_transcript = gr.Textbox(
                        label="Transcript",
                        placeholder="Bấm nút trên để tự động nhận dạng, hoặc nhập thủ công.\nSửa chỗ sai trước khi huấn luyện.",
                        lines=6,
                    )
                    train_auto_text = gr.State("")
                    train_btn = gr.Button("Bắt đầu huấn luyện", variant="primary", size="lg")
                    train_status = gr.Textbox(label="Trạng thái", interactive=False, max_lines=2)

                    def transcribe_for_training(audio_path):
                        if not audio_path:
                            return "", "Upload audio trước.", ""
                        text, st = auto_transcribe(audio_path)
                        return text, st, text

                    stt_train_btn.click(transcribe_for_training, [train_audio],
                                        [train_transcript, train_status, train_auto_text])

                with gr.Column(scale=2):
                    gr.Markdown("### Hệ thống")
                    gpu_status = gr.Markdown(value=get_gpu_status)
                    disk_status = gr.Markdown(value=get_disk_status)
                    dash_stats = gr.Markdown(value=get_dataset_stats)
                    dash_refresh = gr.Button("Refresh", size="sm")

            gr.Markdown("---")
            gr.Markdown("### Tiến trình & Log")
            with gr.Row():
                with gr.Column(scale=1):
                    all_voices_status = gr.Markdown(value=get_all_training_status)
                with gr.Column(scale=2):
                    with gr.Row():
                        log_voice_dd = gr.Dropdown(
                            label="Chọn giọng", choices=get_library_voice_choices(), interactive=True, scale=3)
                        log_refresh = gr.Button("Cập nhật", size="sm", scale=1)
                        log_dd_refresh = gr.Button("Refresh DS", size="sm", scale=1)
                    train_progress = gr.Markdown(value="Chọn giọng rồi bấm Cập nhật")
                    train_log = gr.Textbox(label="Log", lines=10, interactive=False)

            with gr.Accordion("Quản lý giọng & model", open=False):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Lưu giọng nhanh (F5-TTS)**")
                        with gr.Row():
                            voice_upload = gr.Audio(label="Audio", type="filepath",
                                                    sources=["upload", "microphone"])
                            voice_name = gr.Textbox(label="Tên", placeholder="vd: giang_vien")
                        save_btn = gr.Button("Lưu", variant="primary", size="sm")
                        mgmt_status = gr.Textbox(show_label=False, interactive=False, max_lines=1)
                        save_btn.click(save_voice, [voice_upload, voice_name], [mgmt_status])
                    with gr.Column():
                        gr.Markdown("**Xóa giọng**")
                        del_voice_dd = gr.Dropdown(label="F5-TTS", choices=get_saved_voices(), interactive=True)
                        del_btn = gr.Button("Xóa", variant="stop", size="sm")
                        del_lib_dd = gr.Dropdown(label="GPT-SoVITS", choices=get_library_voice_choices(), interactive=True)
                        del_lib_btn = gr.Button("Xóa", variant="stop", size="sm")
                        del_status = gr.Textbox(show_label=False, interactive=False, max_lines=1)
                        del_btn.click(delete_voice, [del_voice_dd], [del_status, del_voice_dd])
                        del_lib_btn.click(delete_library_voice, [del_lib_dd], [del_status, del_lib_dd])
                    with gr.Column():
                        gr.Markdown("**F5-TTS Model**")
                        model_dd = gr.Dropdown(label="Version", choices=get_model_versions(),
                                               value=CURRENT_MODEL_VERSION, interactive=True)
                        switch_btn = gr.Button("Chuyển", variant="primary", size="sm")
                        model_status = gr.Textbox(show_label=False, interactive=False, max_lines=1)
                        switch_btn.click(switch_model, [model_dd], [model_status])

            # Wire training
            train_btn.click(train_new_voice,
                            [train_audio, train_name, train_desc, train_transcript, train_auto_text],
                            [train_status])
            log_refresh.click(get_training_log, [log_voice_dd], [train_progress, train_log])
            log_dd_refresh.click(
                lambda: gr.Dropdown(choices=get_library_voice_choices(),
                                    value=get_library_voice_choices()[0] if get_library_voice_choices() else None),
                outputs=[log_voice_dd])
            dash_refresh.click(refresh_dashboard,
                               outputs=[all_voices_status, gpu_status, disk_status, dash_stats])

        # ═══════════════════════════════════════
        # Tab 5: Lịch sử
        # ═══════════════════════════════════════
        with gr.Tab("Lịch sử"):
            history_md = gr.Markdown(value=get_history_display)
            history_refresh = gr.Button("Refresh")
            history_refresh.click(get_history_display, outputs=[history_md])


if __name__ == "__main__":
    # GPU optimization
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')

    # Validate critical files at startup
    vocab = MODEL_DIR / "vocab.txt"
    ckpt = MODEL_DIR / "model_last.pt"
    if not vocab.exists() or not ckpt.exists():
        logger.error(f"Missing model files: vocab={vocab.exists()}, ckpt={ckpt.exists()}")

    logger.info("Loading F5-TTS model...")
    load_model()
    logger.info("Loading Whisper large-v3 (sử dụng toàn bộ VRAM trống)...")
    load_whisper()

    # Start background memory monitor (unload Whisper only when training needs VRAM)
    start_memory_monitor(interval=30)

    gpu = get_gpu_info()
    if gpu:
        logger.info(f"All models loaded. VRAM: {gpu['memory_used_mb']}/{gpu['memory_total_mb']}MB. GPU queue active.")
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        root_path="https://voice.overmind.io.vn",
        ssr_mode=False,
        allowed_paths=[str(BASE_DIR / "static")],
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.amber,
            secondary_hue=gr.themes.colors.indigo,
            neutral_hue=gr.themes.colors.slate,
        ),
        css=CUSTOM_CSS,
        js=CUSTOM_JS,
        head="""
<script>
window.__VC_PATH = window.location.pathname;
window.__VC_ROUTED = false;

// Gradio 6 renders client-side - buttons don't exist at DOMContentLoaded
// Must poll continuously until Gradio finishes rendering
(function vcRoute() {
    var ROUTES = ['/tts', '/library', '/stt', '/train', '/history'];
    var TITLES = ['Text to Speech', 'Thư viện giọng', 'Nhận dạng giọng nói', 'Huấn luyện', 'Lịch sử'];
    var idx = ROUTES.indexOf(window.__VC_PATH);
    if (idx < 0) return;
    document.title = TITLES[idx] + ' - Voice Clone';

    var target = TITLES[idx];
    var tries = 0;
    var timer = setInterval(function() {
        if (window.__VC_ROUTED) { clearInterval(timer); return; }
        tries++;
        var btns = document.querySelectorAll('button');
        for (var i = 0; i < btns.length; i++) {
            if (btns[i].textContent.trim() === target) {
                btns[i].click();
                window.__VC_ROUTED = true;
                clearInterval(timer);
                return;
            }
        }
        if (tries > 300) clearInterval(timer); // 30s timeout
    }, 100);
})();

// Theme
if (localStorage.getItem('vc-theme') === 'light') {
    document.documentElement.classList.add('light-mode');
}
</script>
""",
    )
