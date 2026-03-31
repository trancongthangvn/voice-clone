import os
import tempfile
import uuid
from pathlib import Path

import gradio as gr
import soundfile as sf
import torch
from f5_tts.api import F5TTS

# Paths
BASE_DIR = Path(__file__).parent
VOICES_DIR = BASE_DIR / "voices"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models" / "f5-tts-vietnamese"

VOICES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Global model instance
tts_model = None


def load_model():
    global tts_model
    if tts_model is not None:
        return tts_model

    ckpt_file = MODEL_DIR / "model_last.pt"
    vocab_file = MODEL_DIR / "vocab.txt"

    if ckpt_file.exists():
        print(f"Loading Vietnamese model from {ckpt_file}")
        tts_model = F5TTS(
            model="F5TTS_Base",
            ckpt_file=str(ckpt_file),
            vocab_file=str(vocab_file),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        print("Vietnamese model not found, using default F5-TTS model")
        tts_model = F5TTS(
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    return tts_model


def save_voice(audio_path, voice_name):
    """Save an uploaded voice sample for reuse."""
    if not audio_path or not voice_name:
        return "Vui lòng upload audio và nhập tên giọng nói."

    voice_name = voice_name.strip().replace(" ", "_")
    dest = VOICES_DIR / f"{voice_name}.wav"

    data, sr = sf.read(audio_path)
    sf.write(str(dest), data, sr)

    return f"Đã lưu giọng nói: {voice_name}"


def get_saved_voices():
    """List all saved voice profiles."""
    voices = sorted(VOICES_DIR.glob("*.wav"))
    return [v.stem for v in voices]


def clone_voice(audio_input, ref_text, gen_text, saved_voice, speed):
    """Generate speech using cloned voice."""
    if not gen_text or not gen_text.strip():
        return None, "Vui lòng nhập văn bản cần đọc."

    # Determine reference audio
    ref_audio = None
    if audio_input:
        ref_audio = audio_input
    elif saved_voice:
        ref_path = VOICES_DIR / f"{saved_voice}.wav"
        if ref_path.exists():
            ref_audio = str(ref_path)

    if not ref_audio:
        return None, "Vui lòng upload audio mẫu hoặc chọn giọng đã lưu."

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
        return str(output_path), "Tạo audio thành công!"
    except Exception as e:
        return None, f"Lỗi: {str(e)}"


def refresh_voices():
    voices = get_saved_voices()
    return gr.Dropdown(choices=voices, value=voices[0] if voices else None)


# Build Gradio UI
with gr.Blocks(
    title="Voice Clone - Overmind",
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
    .main-title { text-align: center; margin-bottom: 0.5em; }
    .subtitle { text-align: center; color: #666; margin-bottom: 1.5em; }
    """,
) as app:
    gr.HTML("<h1 class='main-title'>Voice Clone</h1>")
    gr.HTML("<p class='subtitle'>Clone giọng nói và tạo audio từ văn bản</p>")

    with gr.Tabs():
        # Tab 1: Voice Clone TTS
        with gr.Tab("Clone & TTS"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Giọng mẫu")
                    audio_input = gr.Audio(
                        label="Upload audio mẫu (10-30s)",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    saved_voice = gr.Dropdown(
                        label="Hoặc chọn giọng đã lưu",
                        choices=get_saved_voices(),
                        interactive=True,
                    )
                    refresh_btn = gr.Button("Refresh danh sách", size="sm")
                    ref_text = gr.Textbox(
                        label="Transcript audio mẫu (tùy chọn)",
                        placeholder="Nhập nội dung audio mẫu nói gì để tăng chất lượng...",
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
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Tốc độ đọc",
                    )
                    generate_btn = gr.Button(
                        "Tạo Audio", variant="primary", size="lg"
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

        # Tab 2: Manage Voices
        with gr.Tab("Quản lý giọng nói"):
            gr.Markdown("### Lưu giọng nói để dùng lại")
            with gr.Row():
                voice_upload = gr.Audio(
                    label="Upload audio giọng nói",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                voice_name = gr.Textbox(
                    label="Tên giọng nói",
                    placeholder="vd: giang_vien, mc_truyen_hinh...",
                )
            save_btn = gr.Button("Lưu giọng nói", variant="primary")
            save_status = gr.Textbox(label="Trạng thái", interactive=False)

            save_btn.click(
                fn=save_voice,
                inputs=[voice_upload, voice_name],
                outputs=[save_status],
            )

    # Load model on startup
    app.load(fn=lambda: None, outputs=None)


if __name__ == "__main__":
    print("Loading model...")
    load_model()
    print("Starting server...")
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_api=True,
    )
