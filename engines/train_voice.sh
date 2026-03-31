#!/bin/bash
# Train a voice using GPT-SoVITS
# Usage: train_voice.sh <voice_id> <voice_dir>

VOICE_ID=$1
VOICE_DIR=$2
BASE_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
SOVITS_DIR="$BASE_DIR/engines/GPT-SoVITS"
VENV="$SOVITS_DIR/venv/bin/activate"

echo "=== Training voice: $VOICE_ID ==="
echo "Voice dir: $VOICE_DIR"
echo "Start time: $(date)"

source "$VENV"
cd "$SOVITS_DIR"

export PYTHONPATH="$SOVITS_DIR:$PYTHONPATH"

# Step 1: Audio preprocessing - extract semantic tokens
echo "=== Step 1: Preprocessing audio ==="
python3 GPT_SoVITS/prepare_datasets/1-get-text.py \
    --inp_text "$VOICE_DIR/train.list" \
    --exp_name "$VOICE_ID" \
    --gpu_numbers "0" \
    --pretrained_s2G_path "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth" 2>&1

# Step 2: Extract SSL features
echo "=== Step 2: Extract SSL features ==="
python3 GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py \
    --inp_text "$VOICE_DIR/train.list" \
    --exp_name "$VOICE_ID" 2>&1

# Step 3: Extract semantic tokens
echo "=== Step 3: Extract semantic tokens ==="
python3 GPT_SoVITS/prepare_datasets/3-get-semantic.py \
    --inp_text "$VOICE_DIR/train.list" \
    --exp_name "$VOICE_ID" 2>&1

# Step 4: Train SoVITS
echo "=== Step 4: Train SoVITS ==="
python3 GPT_SoVITS/s2_train.py \
    --config_file "GPT_SoVITS/configs/s2.json" \
    --exp_name "$VOICE_ID" \
    --gpu "0" \
    --pretrained_s2G "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth" \
    --pretrained_s2D "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth" \
    --batch_size 4 \
    --total_epoch 8 \
    --save_every_epoch 4 2>&1

# Step 5: Train GPT
echo "=== Step 5: Train GPT ==="
python3 GPT_SoVITS/s1_train.py \
    --config_file "GPT_SoVITS/configs/s1longer.yaml" \
    --exp_name "$VOICE_ID" \
    --gpu "0" \
    --pretrained_s1 "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt" \
    --batch_size 4 \
    --total_epoch 15 \
    --save_every_epoch 5 2>&1

# Step 6: Copy trained models to voice dir
echo "=== Step 6: Copying models ==="
LATEST_GPT=$(ls -t "logs/$VOICE_ID/"*.ckpt 2>/dev/null | head -1)
LATEST_SOVITS=$(ls -t "logs/$VOICE_ID/"*.pth 2>/dev/null | head -1)

if [ -n "$LATEST_GPT" ] && [ -n "$LATEST_SOVITS" ]; then
    cp "$LATEST_GPT" "$VOICE_DIR/gpt.ckpt"
    cp "$LATEST_SOVITS" "$VOICE_DIR/sovits.pth"

    # Update status to ready
    python3 -c "
import json
meta_file = '$VOICE_DIR/meta.json'
with open(meta_file) as f:
    meta = json.load(f)
meta['status'] = 'ready'
meta['gpt_model'] = 'gpt.ckpt'
meta['sovits_model'] = 'sovits.pth'
with open(meta_file, 'w') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print('Voice training complete!')
"
    echo "=== Training complete! ==="
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
