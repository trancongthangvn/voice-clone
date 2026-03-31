#!/bin/bash
# Backup trained models and datasets
# Run daily via cron: 0 3 * * * /home/thang/voice-clone/backup.sh

BACKUP_DIR="/home/thang/backups/voice-clone"
DATE=$(date +%Y%m%d)
mkdir -p "$BACKUP_DIR"

# Backup voice library (GPT-SoVITS trained models)
if [ -d "/home/thang/voice-clone/voice_library" ]; then
    tar czf "$BACKUP_DIR/voice_library_${DATE}.tar.gz" \
        -C /home/thang/voice-clone voice_library/ 2>/dev/null
fi

# Backup STT corrections dataset
if [ -d "/home/thang/voice-clone/stt_dataset" ]; then
    tar czf "$BACKUP_DIR/stt_dataset_${DATE}.tar.gz" \
        -C /home/thang/voice-clone stt_dataset/ 2>/dev/null
fi

# Backup F5-TTS dataset
if [ -d "/home/thang/voice-clone/dataset" ]; then
    tar czf "$BACKUP_DIR/dataset_${DATE}.tar.gz" \
        -C /home/thang/voice-clone dataset/ 2>/dev/null
fi

# Keep only last 7 days
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete

echo "$(date): Backup complete" >> "$BACKUP_DIR/backup.log"
