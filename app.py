import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr
import soundfile as sf
import torch
from f5_tts.api import F5TTS

# Paths
BASE_DIR = Path(__file__).parent
VOICES_DIR = BASE_DIR / "voices"
OUTPUT_DIR = BASE_DIR / "outputs"
DATASET_DIR = BASE_DIR / "dataset"
HISTORY_FILE = BASE_DIR / "history.json"
MODELS_DIR = BASE_DIR / "models"
MODEL_DIR = MODELS_DIR / "f5-tts-vietnamese"

for d in [VOICES_DIR, OUTPUT_DIR, DATASET_DIR]:
    d.mkdir(exist_ok=True)

# Global state
tts_model = None
CURRENT_MODEL_VERSION = "v1-base"


def get_active_model_path():
    """Get the active model checkpoint path (latest version or base)."""
    active_file = MODELS_DIR / "active_version.txt"
    if active_file.exists():
        version = active_file.read_text().strip()
        versioned = MODELS_DIR / version / "model_last.pt"
        if versioned.exists():
            return versioned, version
    return MODEL_DIR / "model_last.pt", "v1-base"


def load_model(force_reload=False):
    global tts_model, CURRENT_MODEL_VERSION
    ckpt_file, version = get_active_model_path()
    if tts_model is not None and not force_reload and version == CURRENT_MODEL_VERSION:
        return tts_model

    vocab_file = MODEL_DIR / "vocab.txt"
    if ckpt_file.exists():
        print(f"Loading model {version} from {ckpt_file}")
        tts_model = F5TTS(
            model="F5TTS_Base",
            ckpt_file=str(ckpt_file),
            vocab_file=str(vocab_file),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        CURRENT_MODEL_VERSION = version
    else:
        print("Vietnamese model not found, using default F5-TTS model")
        tts_model = F5TTS(
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        CURRENT_MODEL_VERSION = "default"
    return tts_model


# --- History ---
def load_history():
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return []


def save_history_entry(entry):
    history = load_history()
    history.insert(0, entry)
    history = history[:100]  # Keep last 100
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2))


# --- Dataset collection ---
def collect_training_data(audio_path, transcript):
    """Auto-save uploaded audio + transcript for future fine-tuning."""
    if not audio_path:
        return
    sample_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    dest_audio = DATASET_DIR / f"{sample_id}.wav"
    data, sr = sf.read(audio_path)
    duration = len(data) / sr
    if duration < 1 or duration > 60:
        return
    sf.write(str(dest_audio), data, sr)
    meta = {
        "id": sample_id,
        "audio": dest_audio.name,
        "transcript": transcript or "",
        "duration": round(duration, 2),
        "collected_at": datetime.now().isoformat(),
    }
    meta_file = DATASET_DIR / "metadata.jsonl"
    with open(meta_file, "a") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")


# --- Voice management ---
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


# --- TTS ---
def clone_voice(audio_input, ref_text, gen_text, saved_voice, speed):
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

    # Auto-collect for training
    collect_training_data(ref_audio, ref_text)

    model = load_model()
    output_path = OUTPUT_DIR / f"{uuid.uuid4().hex}.wav"

    try:
        wav, sr, _ = model.infer(
            ref_file=ref_audio,
            ref_text=ref_text if ref_text and ref_text.strip() else "",
            gen_text=gen_text.strip(),
            speed=speed,
        )
        sf.write(str(output_path), wav, sr)

        # Save to history
        save_history_entry({
            "time": datetime.now().strftime("%d/%m %H:%M"),
            "text": gen_text.strip()[:80],
            "voice": saved_voice or "upload",
            "output": output_path.name,
            "model": CURRENT_MODEL_VERSION,
        })

        return str(output_path), f"Tạo thành công! (Model: {CURRENT_MODEL_VERSION})"
    except Exception as e:
        return None, f"Lỗi: {str(e)}"


def refresh_voices():
    voices = get_saved_voices()
    return gr.Dropdown(choices=voices, value=voices[0] if voices else None)


# --- History display ---
def get_history_display():
    history = load_history()
    if not history:
        return "Chưa có lịch sử."
    lines = []
    for h in history[:20]:
        lines.append(f"**{h['time']}** | {h.get('voice','?')} | {h['text']}")
    return "\n\n".join(lines)


def play_history_audio(evt: gr.SelectData):
    history = load_history()
    if evt.index < len(history):
        path = OUTPUT_DIR / history[evt.index]["output"]
        if path.exists():
            return str(path)
    return None


# --- Dataset stats ---
def get_dataset_stats():
    meta_file = DATASET_DIR / "metadata.jsonl"
    if not meta_file.exists():
        return "Chưa có dữ liệu training."
    lines = meta_file.read_text().strip().split("\n")
    total = len(lines)
    total_duration = 0
    with_transcript = 0
    for line in lines:
        entry = json.loads(line)
        total_duration += entry.get("duration", 0)
        if entry.get("transcript"):
            with_transcript += 1
    hours = total_duration / 3600
    mins = total_duration / 60
    return (
        f"**Tổng samples:** {total}\n\n"
        f"**Tổng thời lượng:** {hours:.1f}h ({mins:.0f} phút)\n\n"
        f"**Có transcript:** {with_transcript}/{total}\n\n"
        f"**Model hiện tại:** {CURRENT_MODEL_VERSION}"
    )


# --- Model version management ---
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
    return f"Đã chuyển sang model: {CURRENT_MODEL_VERSION}"


# === BUILD UI ===
CUSTOM_CSS = """
.main-header { text-align: center; padding: 1em 0 0.5em; }
.main-header h1 { font-size: 2em; margin: 0; }
.main-header p { color: #888; margin: 0.3em 0 0; }
.stat-card { padding: 0.8em; border-radius: 8px; }
"""

with gr.Blocks(title="Voice Clone - Overmind") as app:
    gr.HTML("""
        <div class='main-header'>
            <h1>Voice Clone</h1>
            <p>Clone giọng nói & Text-to-Speech tiếng Việt</p>
        </div>
    """)

    with gr.Tabs():
        # === Tab 1: Clone & TTS ===
        with gr.Tab("Clone & TTS"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Giọng mẫu")
                    audio_input = gr.Audio(
                        label="Upload audio mẫu (tối ưu 15-30s, nói rõ ràng)",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    with gr.Row():
                        saved_voice = gr.Dropdown(
                            label="Hoặc chọn giọng đã lưu",
                            choices=get_saved_voices(),
                            interactive=True,
                            scale=3,
                        )
                        refresh_btn = gr.Button("🔄", size="sm", scale=1, min_width=40)
                    ref_text = gr.Textbox(
                        label="Transcript audio mẫu (tùy chọn, nhập để tăng chất lượng)",
                        placeholder="Nhập chính xác nội dung audio mẫu đang nói...",
                        lines=2,
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### Văn bản cần đọc")
                    gen_text = gr.Textbox(
                        label="Nhập văn bản",
                        placeholder="Nhập nội dung bạn muốn giọng clone đọc...",
                        lines=5,
                    )
                    speed = gr.Slider(
                        minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                        label="Tốc độ đọc",
                    )
                    generate_btn = gr.Button(
                        "Tạo Audio", variant="primary", size="lg",
                    )

            with gr.Row():
                output_audio = gr.Audio(label="Kết quả", type="filepath")
                status_text = gr.Textbox(label="Trạng thái", interactive=False)

            generate_btn.click(
                fn=clone_voice,
                inputs=[audio_input, ref_text, gen_text, saved_voice, speed],
                outputs=[output_audio, status_text],
            )
            refresh_btn.click(fn=refresh_voices, outputs=[saved_voice])

        # === Tab 2: Quản lý giọng ===
        with gr.Tab("Quản lý giọng nói"):
            gr.Markdown("### Lưu giọng nói để dùng lại")
            with gr.Row():
                voice_upload = gr.Audio(
                    label="Upload audio giọng nói",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                with gr.Column():
                    voice_name = gr.Textbox(
                        label="Tên giọng nói",
                        placeholder="vd: giang_vien, mc_truyen_hinh...",
                    )
                    with gr.Row():
                        save_btn = gr.Button("Lưu giọng", variant="primary")
                        del_voice_dd = gr.Dropdown(
                            label="Xóa giọng",
                            choices=get_saved_voices(),
                            interactive=True,
                        )
                        del_btn = gr.Button("Xóa", variant="stop", size="sm")
            save_status = gr.Textbox(label="Trạng thái", interactive=False)

            save_btn.click(
                fn=save_voice,
                inputs=[voice_upload, voice_name],
                outputs=[save_status],
            )
            del_btn.click(
                fn=delete_voice,
                inputs=[del_voice_dd],
                outputs=[save_status, del_voice_dd],
            )

        # === Tab 3: Lịch sử ===
        with gr.Tab("Lịch sử"):
            history_md = gr.Markdown(value=get_history_display)
            history_audio = gr.Audio(label="Nghe lại", type="filepath", interactive=False)
            history_refresh = gr.Button("Refresh")
            history_refresh.click(fn=get_history_display, outputs=[history_md])

        # === Tab 4: Training Data & Model ===
        with gr.Tab("Training & Model"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Dataset tự thu thập")
                    gr.Markdown("Mỗi lần bạn upload audio mẫu, hệ thống tự động lưu lại "
                                "làm dữ liệu training cho fine-tune sau này.")
                    dataset_stats = gr.Markdown(value=get_dataset_stats)
                    stats_refresh = gr.Button("Refresh thống kê")
                    stats_refresh.click(fn=get_dataset_stats, outputs=[dataset_stats])

                with gr.Column():
                    gr.Markdown("### Quản lý Model")
                    model_dd = gr.Dropdown(
                        label="Chọn version model",
                        choices=get_model_versions(),
                        value=CURRENT_MODEL_VERSION,
                        interactive=True,
                    )
                    switch_btn = gr.Button("Chuyển model", variant="primary")
                    model_status = gr.Textbox(label="Trạng thái", interactive=False)
                    switch_btn.click(
                        fn=switch_model,
                        inputs=[model_dd],
                        outputs=[model_status],
                    )

                    gr.Markdown("### Hướng dẫn Fine-tune")
                    gr.Markdown(
                        "1. Thu thập data: Hệ thống tự động lưu audio upload vào `dataset/`\n"
                        "2. Chạy fine-tune:\n"
                        "```bash\ncd /home/thang/voice-clone\nsource venv/bin/activate\n"
                        "python3 finetune.py --version v2\n```\n"
                        "3. Sau khi train xong, chọn version mới ở dropdown trên và bấm 'Chuyển model'"
                    )


if __name__ == "__main__":
    print("Loading model...")
    load_model()
    print("Starting server...")
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        root_path="https://voice.overmind.io.vn",
        ssr_mode=False,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CUSTOM_CSS,
    )
