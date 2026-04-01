#!/bin/bash
# Train a voice using GPT-SoVITS v2
# Usage: train_voice.sh <voice_id> <voice_dir>

VOICE_ID="$1"
VOICE_DIR="$2"
BASE_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
SOVITS_DIR="$BASE_DIR/engines/GPT-SoVITS"
VENV="$SOVITS_DIR/venv/bin/activate"
PRETRAINED="$SOVITS_DIR/GPT_SoVITS/pretrained_models"

mark_failed() {
    python3 -c "
import json
meta_file = '$VOICE_DIR/meta.json'
with open(meta_file) as f:
    meta = json.load(f)
meta['status'] = 'failed'
meta['error'] = '$1'
with open(meta_file, 'w') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
"
    echo "=== FAILED: $1 ==="
    exit 1
}

echo "=== Training voice: $VOICE_ID ==="
echo "Voice dir: $VOICE_DIR"
echo "Start time: $(date)"

# Validate pretrained models exist
for f in "$PRETRAINED/chinese-roberta-wwm-ext-large/pytorch_model.bin" \
         "$PRETRAINED/chinese-hubert-base/pytorch_model.bin" \
         "$PRETRAINED/gsv-v2final-pretrained/s2G2333k.pth" \
         "$PRETRAINED/gsv-v2final-pretrained/s2D2333k.pth" \
         "$PRETRAINED/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"; do
    if [ ! -f "$f" ]; then
        mark_failed "Missing pretrained model: $f"
    fi
done

source "$VENV" || mark_failed "Cannot activate venv"
cd "$SOVITS_DIR"
export PYTHONPATH="$SOVITS_DIR:$SOVITS_DIR/GPT_SoVITS:$PYTHONPATH"

# GPU optimization
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="true"

# Free GPU memory: unload Whisper + F5-TTS from main app before training
echo "=== Freeing GPU memory for training ==="
python3 -c "
import urllib.request
try:
    urllib.request.urlopen('http://127.0.0.1:7860/gradio_api/queue/join', timeout=2)
except: pass
" 2>/dev/null || true
# Wait a moment for GPU memory to settle
sleep 2

OPT_DIR="$SOVITS_DIR/logs/$VOICE_ID"
mkdir -p "$OPT_DIR"

# =========================================
# Step 1: Get text (phoneme extraction)
# =========================================
echo "=== Step 1/5: Text processing ==="
export inp_text="$VOICE_DIR/train.list"
export inp_wav_dir=""
export exp_name="$VOICE_ID"
export i_part="0"
export all_parts="1"
export _CUDA_VISIBLE_DEVICES="0"
export opt_dir="$OPT_DIR"
export bert_pretrained_dir="$PRETRAINED/chinese-roberta-wwm-ext-large"
export is_half="True"
export version="v2"

python3 GPT_SoVITS/prepare_datasets/1-get-text.py 2>&1 || mark_failed "Step 1: text processing failed"
echo "Step 1 done."

# =========================================
# Step 2: Extract HuBERT features + resample to 32k
# =========================================
echo "=== Step 2/5: HuBERT feature extraction ==="
export cnhubert_base_dir="$PRETRAINED/chinese-hubert-base"

python3 GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py 2>&1 || mark_failed "Step 2: HuBERT extraction failed"
echo "Step 2 done."

# =========================================
# Step 3: Get semantic tokens
# =========================================
echo "=== Step 3/5: Semantic token extraction ==="
export pretrained_s2G="$PRETRAINED/gsv-v2final-pretrained/s2G2333k.pth"
export s2config_path="$SOVITS_DIR/GPT_SoVITS/configs/s2.json"

python3 GPT_SoVITS/prepare_datasets/3-get-semantic.py 2>&1 || mark_failed "Step 3: semantic extraction failed"
echo "Step 3 done."

# =========================================
# Step 4: Train SoVITS (s2)
# =========================================
echo "=== Step 4/5: Train SoVITS ==="

# Create s2 training config
S2_CONFIG="$OPT_DIR/s2_config.json"
python3 -c "
import json
with open('$SOVITS_DIR/GPT_SoVITS/configs/s2.json') as f:
    cfg = json.load(f)
cfg['train']['epochs'] = 10
cfg['train']['batch_size'] = 4
cfg['train']['save_every_epoch'] = 5
cfg['train']['fp16_run'] = True
cfg['train']['num_workers'] = 4
cfg['train']['pin_memory'] = True
cfg['train']['gpu_numbers'] = '0'
cfg['train']['pretrained_s2G'] = '$PRETRAINED/gsv-v2final-pretrained/s2G2333k.pth'
cfg['train']['pretrained_s2D'] = '$PRETRAINED/gsv-v2final-pretrained/s2D2333k.pth'
cfg['model']['version'] = 'v2'
cfg['data']['exp_dir'] = '$OPT_DIR'
cfg['s2_ckpt_dir'] = '$OPT_DIR'
with open('$S2_CONFIG', 'w') as f:
    json.dump(cfg, f, indent=2)
print('s2 config written')
" 2>&1

python3 GPT_SoVITS/s2_train.py -c "$S2_CONFIG" 2>&1 || mark_failed "Step 4: SoVITS training failed"
echo "Step 4 done."

# =========================================
# Step 5: Train GPT (s1)
# =========================================
echo "=== Step 5/5: Train GPT ==="

# Create s1 training config
S1_CONFIG="$OPT_DIR/s1_config.yaml"
python3 -c "
import yaml
with open('$SOVITS_DIR/GPT_SoVITS/configs/s1longer.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['train']['epochs'] = 15
cfg['train']['batch_size'] = 4
cfg['train']['save_every_epoch'] = 5
cfg['train']['precision'] = '16-mixed'
cfg['train']['num_workers'] = 4
cfg['output_dir'] = '$OPT_DIR'
cfg['pretrained_s1'] = '$PRETRAINED/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt'
with open('$S1_CONFIG', 'w') as f:
    yaml.dump(cfg, f)
print('s1 config written')
" 2>&1

python3 GPT_SoVITS/s1_train.py -c "$S1_CONFIG" 2>&1 || mark_failed "Step 5: GPT training failed"
echo "Step 5 done."

# =========================================
# Copy trained models
# =========================================
echo "=== Copying final models ==="

LATEST_GPT=$(find "$OPT_DIR" -name "*.ckpt" -newer "$VOICE_DIR/meta.json" 2>/dev/null | sort -t= -k2 -n | tail -1)
LATEST_SOVITS=$(find "$OPT_DIR" -name "*.pth" -newer "$VOICE_DIR/meta.json" 2>/dev/null | sort | tail -1)

if [ -n "$LATEST_GPT" ] && [ -n "$LATEST_SOVITS" ]; then
    cp "$LATEST_GPT" "$VOICE_DIR/gpt.ckpt" || mark_failed "Failed to copy GPT model"
    cp "$LATEST_SOVITS" "$VOICE_DIR/sovits.pth" || mark_failed "Failed to copy SoVITS model"

    # Validate copied files are not empty
    if [ ! -s "$VOICE_DIR/gpt.ckpt" ] || [ ! -s "$VOICE_DIR/sovits.pth" ]; then
        mark_failed "Copied model files are empty"
    fi

    python3 -c "
import json
meta_file = '$VOICE_DIR/meta.json'
with open(meta_file) as f:
    meta = json.load(f)
meta['status'] = 'ready'
with open(meta_file, 'w') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print('Status: ready')
"
    echo "=== Training COMPLETE! ==="
else
    echo "=== Training FAILED - no checkpoint found ==="
    python3 -c "
import json
meta_file = '$VOICE_DIR/meta.json'
with open(meta_file) as f:
    meta = json.load(f)
meta['status'] = 'failed'
with open(meta_file, 'w') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
"
fi

echo "End time: $(date)"
