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

    # Create training list (use "auto" for language detection)
    (voice_dir / "train.list").write_text(
        f"{voice_dir / 'raw_audio.wav'}|{voice_name}|auto|{transcript.strip()}\n"
    )

    # Start training in background (proper resource handling)
    train_script = BASE_DIR / "engines" / "train_voice.sh"
    if not train_script.exists():
        return "Lỗi: không tìm thấy script huấn luyện."

    # Start training subprocess (daemonized, log to file)
    subprocess.Popen(
        ["bash", str(train_script), voice_id, str(voice_dir)],
        stdout=open(voice_dir / "train.log", "w"),
        stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR / "engines" / "GPT-SoVITS"),
        start_new_session=True,  # Detach from parent
    )
    logger.info(f"Training started: {voice_id}")

    return f"Đang huấn luyện giọng '{voice_name}' ({info['duration']:.0f}s audio)... Kiểm tra ở tab Thư viện."


def get_training_log(voice_selection):
    """Get training log for a voice."""
    if not voice_selection:
        return "Chọn giọng để xem log."
    voices = get_library_voices()
    choices = get_library_voice_choices()
    if voice_selection not in choices:
        return "Giọng không hợp lệ."
    voice = voices[choices.index(voice_selection)]
    log_file = VOICE_LIBRARY_DIR / voice["id"] / "train.log"
    if log_file.exists():
        content = log_file.read_text()
        return content[-3000:] if len(content) > 3000 else content
    return "Chưa có log."


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
    # Check for fine-tuned model
    active_file = WHISPER_MODELS_DIR / "active_version.txt"
    if active_file.exists():
        version = active_file.read_text().strip()
        ct2_path = WHISPER_MODELS_DIR / version / "ct2"
        if ct2_path.exists():
            return WhisperModel(str(ct2_path), device="cuda", compute_type="float16",
                                num_workers=4, cpu_threads=16)
    return WhisperModel("large-v3", device="cuda", compute_type="float16",
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
    """Get status of all voices in library."""
    voices = get_library_voices()
    if not voices:
        return "Chưa có giọng nào trong thư viện."
    lines = []
    for v in voices:
        status = v.get("status", "?")
        icon = {"ready": "🟢", "training": "🟡", "failed": "🔴"}.get(status, "⚪")
        duration = v.get("duration", 0)
        created = v.get("created_at", "")[:10]
        lines.append(
            f"{icon} **{v['name']}** | {status} | {duration}s audio | {created}"
        )
    return "\n\n".join(lines)


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
   Voice Clone - Soft Dark Theme
   Font: SVN Gilroy
   Palette: Muted warm tones, no neon/gradient
   ======================================== */

@font-face {
    font-family: 'SVN Gilroy';
    src: url('/static/fonts/SVN-Gilroy_Regular.otf') format('opentype');
    font-weight: 400; font-style: normal;
}
@font-face {
    font-family: 'SVN Gilroy';
    src: url('/static/fonts/SVN-Gilroy_Medium.otf') format('opentype');
    font-weight: 500; font-style: normal;
}
@font-face {
    font-family: 'SVN Gilroy';
    src: url('/static/fonts/SVN-Gilroy_SemiBold.otf') format('opentype');
    font-weight: 600; font-style: normal;
}
@font-face {
    font-family: 'SVN Gilroy';
    src: url('/static/fonts/SVN-Gilroy_Bold.otf') format('opentype');
    font-weight: 700; font-style: normal;
}
@font-face {
    font-family: 'SVN Gilroy';
    src: url('/static/fonts/SVN-Gilroy_Heavy.otf') format('opentype');
    font-weight: 800; font-style: normal;
}

/* Global */
.gradio-container,
.gradio-container *:not(code):not(pre) {
    font-family: 'SVN Gilroy', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Header */
.main-header {
    text-align: center;
    padding: 1.8em 0 1.2em;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 0.5em;
}
.main-header h1 {
    font-size: 2em;
    margin: 0;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #e2e8f0;
}
.main-header p {
    color: #78819a;
    margin: 0.3em 0 0;
    font-size: 0.95em;
    font-weight: 400;
}
.main-header .version {
    display: inline-block;
    margin-top: 0.6em;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.72em;
    font-weight: 500;
    background: rgba(148,163,184,0.08);
    color: #8896ab;
    border: 1px solid rgba(148,163,184,0.12);
}

/* Tabs - clean, no glow */
.tabs > .tab-nav > button {
    font-weight: 500 !important;
    font-size: 0.9em !important;
    padding: 10px 18px !important;
    border-radius: 6px 6px 0 0 !important;
    transition: color 0.2s ease !important;
    color: #78819a !important;
}
.tabs > .tab-nav > button.selected {
    color: #cbd5e1 !important;
    border-bottom: 2px solid #7c8db5 !important;
    background: transparent !important;
}
.tabs > .tab-nav > button:hover {
    color: #a8b5c8 !important;
}

/* Buttons - muted, no glow */
.primary {
    background: #4a5568 !important;
    border: 1px solid #5a6a80 !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
    color: #e2e8f0 !important;
}
.primary:hover {
    background: #566884 !important;
    border-color: #6b7fa0 !important;
    box-shadow: none !important;
    transform: none !important;
}
button.stop {
    transition: all 0.15s ease !important;
}

/* Cards / Panels - subtle borders */
.panel, .block {
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.04) !important;
}

/* Audio */
.audio-container {
    border-radius: 10px !important;
}

/* Markdown */
.markdown-text h3 {
    font-weight: 600 !important;
    color: #c9d1dc !important;
    margin-bottom: 0.5em !important;
}

/* Scrollbar - thin, quiet */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.14); }

/* Focus - subtle */
*:focus-visible {
    outline: 1.5px solid #7c8db5 !important;
    outline-offset: 2px !important;
}

/* Inputs & textboxes - softer */
textarea, input[type="text"] {
    font-weight: 400 !important;
}

/* Slider */
input[type="range"]::-webkit-slider-thumb {
    transition: box-shadow 0.15s ease !important;
}
"""

with gr.Blocks(title="Voice Clone - Overmind") as app:
    gr.HTML("""
        <div class='main-header'>
            <h1>Voice Clone</h1>
            <p>Clone giọng nói & chuyển văn bản thành giọng nói</p>
            <span class='version'>F5-TTS + GPT-SoVITS</span>
        </div>
    """)

    with gr.Tabs():
        # === Tab 1: Clone nhanh (F5-TTS) ===
        with gr.Tab("Clone nhanh (F5-TTS)"):
            gr.Markdown("Zero-shot voice clone - upload audio mẫu, không cần huấn luyện")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Giọng mẫu")
                    audio_input = gr.Audio(
                        label="Upload audio mẫu (tối ưu 15-30s, nói rõ ràng)",
                        type="filepath", sources=["upload", "microphone"],
                    )
                    with gr.Row():
                        saved_voice = gr.Dropdown(
                            label="Hoặc chọn giọng đã lưu",
                            choices=get_saved_voices(), interactive=True, scale=3,
                        )
                        refresh_btn = gr.Button("🔄", size="sm", scale=1, min_width=40)
                    ref_text = gr.Textbox(
                        label="Transcript audio mẫu (nhập để tăng chất lượng)",
                        placeholder="Nhập chính xác nội dung audio mẫu đang nói...",
                        lines=2,
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### Văn bản cần đọc")
                    gen_text = gr.Textbox(
                        label="Nhập văn bản", lines=5,
                        placeholder="Nhập nội dung bạn muốn giọng clone đọc...",
                    )
                    speed = gr.Slider(0.5, 2.0, 1.0, 0.1, label="Tốc độ")
                    gen_btn = gr.Button("Tạo Audio", variant="primary", size="lg")
            with gr.Row():
                output_audio = gr.Audio(label="Kết quả", type="filepath")
                status_text = gr.Textbox(label="Trạng thái", interactive=False)

            gen_btn.click(clone_voice_f5,
                          [audio_input, ref_text, gen_text, saved_voice, speed],
                          [output_audio, status_text])
            refresh_btn.click(refresh_voices, outputs=[saved_voice])

        # === Tab 2: Thư viện giọng (GPT-SoVITS) ===
        with gr.Tab("Thư viện giọng"):
            gr.Markdown("Giọng nói chất lượng cao đã huấn luyện - chọn giọng và nhập text")
            with gr.Row():
                with gr.Column(scale=1):
                    lib_voice_dd = gr.Dropdown(
                        label="Chọn giọng",
                        choices=get_library_voice_choices(), interactive=True,
                    )
                    lib_refresh = gr.Button("Refresh danh sách", size="sm")
                    lib_text = gr.Textbox(label="Văn bản cần đọc", lines=5,
                                          placeholder="Nhập text...")
                    lib_speed = gr.Slider(0.5, 2.0, 1.0, 0.1, label="Tốc độ")
                    lib_gen_btn = gr.Button("Tạo Audio", variant="primary", size="lg")
                with gr.Column(scale=1):
                    lib_output = gr.Audio(label="Kết quả", type="filepath")
                    lib_status = gr.Textbox(label="Trạng thái", interactive=False)

            lib_gen_btn.click(library_tts,
                              [lib_voice_dd, lib_text, lib_speed],
                              [lib_output, lib_status])
            def refresh_lib_voices():
                choices = get_library_voice_choices()
                return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
            lib_refresh.click(refresh_lib_voices, outputs=[lib_voice_dd])

        # === Tab 3: Voice to Text (STT) ===
        with gr.Tab("Voice to Text"):
            gr.Markdown("Chuyển giọng nói thành văn bản (Whisper AI) - **càng dùng càng chuẩn**")
            stt_auto_text = gr.State("")  # store original auto text for comparison
            with gr.Row():
                with gr.Column(scale=1):
                    stt_audio = gr.Audio(
                        label="Upload audio cần chuyển thành text",
                        type="filepath", sources=["upload", "microphone"],
                    )
                    stt_lang = gr.Dropdown(
                        label="Ngôn ngữ",
                        choices=["Auto", "vi", "en", "zh", "ja", "ko", "fr"],
                        value="Auto", interactive=True,
                    )
                    stt_btn = gr.Button("Chuyển thành text", variant="primary", size="lg")
                with gr.Column(scale=1):
                    stt_result = gr.Textbox(
                        label="Kết quả text (sửa nếu sai → bấm Lưu để hệ thống tự học)",
                        lines=8, interactive=True,
                    )
                    stt_details = gr.Markdown(label="Chi tiết")

            def stt_and_store(audio, lang):
                text, details = transcribe_audio(audio, lang)
                return text, details, text  # 3rd output = store auto text

            stt_btn.click(stt_and_store, [stt_audio, stt_lang],
                          [stt_result, stt_details, stt_auto_text])

            with gr.Row():
                stt_save_btn = gr.Button("Lưu correction (giúp AI học)", variant="secondary")
                stt_save_status = gr.Textbox(label="", interactive=False)
            stt_stats = gr.Markdown(value=get_stt_stats)

            stt_save_btn.click(
                save_stt_correction,
                [stt_audio, stt_auto_text, stt_result],
                [stt_save_status],
            )

            gr.Markdown(
                "**Cách hoạt động:** Upload audio → Whisper nhận dạng → "
                "Bạn sửa chỗ sai → Bấm 'Lưu correction' → "
                "Hệ thống tích lũy dữ liệu → Fine-tune Whisper → Nhận dạng càng chuẩn."
            )

        # === Tab 4: Training Dashboard ===
        with gr.Tab("Training Dashboard"):
            with gr.Row():
                # Left: Train new voice
                with gr.Column(scale=1):
                    gr.Markdown("### Huấn luyện giọng mới")
                    train_audio = gr.Audio(label="Audio giọng nói (1-3 phút, sạch)",
                                           type="filepath", sources=["upload"])
                    train_name = gr.Textbox(label="Tên giọng", placeholder="vd: MC Thời sự")
                    train_desc = gr.Textbox(label="Mô tả (tùy chọn)",
                                            placeholder="Giọng nam miền Bắc, trầm ấm...")
                    stt_train_btn = gr.Button("Nhận dạng audio thành text", variant="secondary")
                    train_transcript = gr.Textbox(
                        label="Transcript (bấm nút trên để nhận dạng, sửa nếu sai)",
                        placeholder="Bấm 'Nhận dạng audio thành text' hoặc nhập thủ công...",
                        lines=5,
                    )
                    train_auto_text = gr.State("")
                    with gr.Row():
                        train_btn = gr.Button("Bắt đầu huấn luyện", variant="primary", size="lg")
                    train_status = gr.Textbox(label="Trạng thái", interactive=False)

                    def transcribe_for_training(audio_path):
                        if not audio_path:
                            return "", "Upload audio trước.", ""
                        text, status = auto_transcribe(audio_path)
                        return text, status, text

                    stt_train_btn.click(
                        fn=transcribe_for_training,
                        inputs=[train_audio],
                        outputs=[train_transcript, train_status, train_auto_text],
                    )

                # Right: Monitor & Status
                with gr.Column(scale=1):
                    gr.Markdown("### Bảng điều khiển")
                    with gr.Row():
                        gpu_status = gr.Markdown(value=get_gpu_status)
                        disk_status = gr.Markdown(value=get_disk_status)
                    dash_stats = gr.Markdown(value=get_dataset_stats)
                    dash_refresh = gr.Button("Refresh", size="sm")

                    gr.Markdown("### Trạng thái tất cả giọng")
                    all_voices_status = gr.Markdown(value=get_all_training_status)

            gr.Markdown("---")
            with gr.Row():
                # Training log
                with gr.Column():
                    gr.Markdown("### Training Log")
                    log_voice_dd = gr.Dropdown(
                        label="Chọn giọng xem log",
                        choices=get_library_voice_choices(), interactive=True)
                    log_refresh = gr.Button("Xem log", size="sm")
                    train_log = gr.Textbox(label="Log", lines=12, interactive=False)

                # Quick actions
                with gr.Column():
                    gr.Markdown("### Quản lý nhanh")
                    with gr.Accordion("Giọng nhanh (F5-TTS)", open=False):
                        with gr.Row():
                            voice_upload = gr.Audio(label="Audio", type="filepath",
                                                    sources=["upload", "microphone"])
                            voice_name = gr.Textbox(label="Tên", placeholder="vd: giang_vien")
                        save_btn = gr.Button("Lưu giọng nhanh", variant="primary", size="sm")
                        mgmt_status = gr.Textbox(label="Trạng thái", interactive=False)
                        save_btn.click(save_voice, [voice_upload, voice_name], [mgmt_status])

                    with gr.Accordion("Xóa giọng", open=False):
                        del_voice_dd = gr.Dropdown(label="Giọng nhanh (F5-TTS)",
                                                   choices=get_saved_voices(), interactive=True)
                        del_btn = gr.Button("Xóa", variant="stop", size="sm")
                        del_lib_dd = gr.Dropdown(label="Thư viện (GPT-SoVITS)",
                                                 choices=get_library_voice_choices(), interactive=True)
                        del_lib_btn = gr.Button("Xóa", variant="stop", size="sm")
                        del_status = gr.Textbox(label="Trạng thái", interactive=False)
                        del_btn.click(delete_voice, [del_voice_dd], [del_status, del_voice_dd])
                        del_lib_btn.click(delete_library_voice, [del_lib_dd], [del_status, del_lib_dd])

                    with gr.Accordion("F5-TTS Model Version", open=False):
                        model_dd = gr.Dropdown(label="Version", choices=get_model_versions(),
                                               value=CURRENT_MODEL_VERSION, interactive=True)
                        switch_btn = gr.Button("Chuyển model", variant="primary", size="sm")
                        model_status = gr.Textbox(label="Trạng thái", interactive=False)
                        switch_btn.click(switch_model, [model_dd], [model_status])

            # Wire up training (pass auto_text for correction tracking)
            train_btn.click(train_new_voice,
                            [train_audio, train_name, train_desc, train_transcript, train_auto_text],
                            [train_status])
            log_refresh.click(get_training_log, [log_voice_dd], [train_log])
            dash_refresh.click(refresh_dashboard,
                               outputs=[all_voices_status, gpu_status, disk_status, dash_stats])

        # === Tab 5: Lịch sử ===
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
    logger.info("Pre-loading Whisper model...")
    load_whisper()

    # Start background memory monitor
    start_memory_monitor(interval=30)

    logger.info("All models loaded. GPU queue active. Starting server...")
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        root_path="https://voice.overmind.io.vn",
        ssr_mode=False,
        allowed_paths=[str(BASE_DIR / "static")],
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.slate,
            secondary_hue=gr.themes.colors.gray,
            neutral_hue=gr.themes.colors.slate,
        ),
        css=CUSTOM_CSS,
    )
