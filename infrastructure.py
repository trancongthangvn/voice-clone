"""
Infrastructure module: Thread-safe model management, GPU queue,
health monitoring, audio preprocessing, logging.
"""

import json
import logging
import logging.handlers
import os
import queue
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch

# ============================================================
# Structured Logging
# ============================================================

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            LOG_DIR / "app.log", maxBytes=10 * 1024 * 1024, backupCount=5,
        ),
    ],
)
logger = logging.getLogger("voice_clone")


# ============================================================
# C2: Thread-Safe Model Manager
# ============================================================

class ModelManager:
    """Thread-safe singleton model manager with VRAM cleanup."""

    def __init__(self):
        self._lock = threading.RLock()
        self._tts_model = None
        self._tts_version = "v1-base"
        self._whisper_model = None
        self._whisper_last_used = 0

    def get_tts(self, load_fn, force_reload=False, target_version=None):
        with self._lock:
            if (self._tts_model is not None
                    and not force_reload
                    and (target_version is None or target_version == self._tts_version)):
                return self._tts_model, self._tts_version

            # Free old model
            if self._tts_model is not None:
                logger.info(f"Unloading TTS model {self._tts_version}")
                del self._tts_model
                self._tts_model = None
                torch.cuda.empty_cache()

            model, version = load_fn()
            self._tts_model = model
            self._tts_version = version
            logger.info(f"TTS model loaded: {version}")
            return self._tts_model, self._tts_version

    @property
    def tts_version(self):
        return self._tts_version

    def get_whisper(self, load_fn):
        with self._lock:
            self._whisper_last_used = time.time()
            if self._whisper_model is None:
                logger.info("Loading Whisper model...")
                self._whisper_model = load_fn()
                logger.info("Whisper loaded")
            return self._whisper_model

    def unload_whisper(self):
        with self._lock:
            if self._whisper_model is not None:
                logger.info("Unloading Whisper to free VRAM")
                del self._whisper_model
                self._whisper_model = None
                torch.cuda.empty_cache()

    def maybe_unload_whisper(self, idle_seconds=300):
        """Unload Whisper if idle for too long. Disabled by default - keep models in VRAM."""
        # Disabled: user wants max GPU utilization, keep all models loaded
        pass


# Global instance
models = ModelManager()


# ============================================================
# C1: GPU OOM Recovery
# ============================================================

def get_gpu_info():
    """Get GPU memory and utilization info."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) >= 5:
            return {
                "memory_used_mb": int(parts[0]),
                "memory_total_mb": int(parts[1]),
                "memory_free_mb": int(parts[2]),
                "utilization_pct": int(parts[3]),
                "temperature_c": int(parts[4]),
                "memory_pct": round(int(parts[0]) / int(parts[1]) * 100, 1),
            }
    except Exception:
        pass
    return None


def check_gpu_memory(min_free_mb=1000):
    """Check if GPU has enough free memory for inference."""
    info = get_gpu_info()
    if info and info["memory_free_mb"] < min_free_mb:
        torch.cuda.empty_cache()
        time.sleep(0.3)
        info = get_gpu_info()
        if info and info["memory_free_mb"] < min_free_mb:
            return False, f"GPU memory thấp: {info['memory_free_mb']}MB free (cần {min_free_mb}MB)"
    return True, "OK"


def run_gpu_inference(fn, *args, max_retries=2, **kwargs):
    """Run inference with OOM detection and recovery."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.warning(f"GPU OOM attempt {attempt + 1}/{max_retries}")
                models.unload_whisper()
                torch.cuda.empty_cache()
                time.sleep(1)
                if attempt == max_retries - 1:
                    raise RuntimeError("GPU hết bộ nhớ. Thử lại sau.") from e
            else:
                raise


# ============================================================
# C3: GPU Inference Queue
# ============================================================

class InferenceQueue:
    """Single-threaded GPU queue to prevent concurrent OOM."""

    def __init__(self, max_size=10):
        self._queue = queue.Queue(maxsize=max_size)
        self._active = True
        self._current_task = None
        self._worker = threading.Thread(target=self._process, daemon=True)
        self._worker.start()
        self.stats = {"total": 0, "success": 0, "failed": 0, "queued": 0}

    def _process(self):
        while self._active:
            try:
                task = self._queue.get(timeout=1)
                self.stats["queued"] = self._queue.qsize()
                self._current_task = task
                try:
                    result = task["fn"](*task.get("args", ()), **task.get("kwargs", {}))
                    task["result_q"].put(("ok", result))
                    self.stats["success"] += 1
                except Exception as e:
                    task["result_q"].put(("error", str(e)))
                    self.stats["failed"] += 1
                    logger.error(f"Queue task failed: {e}")
                finally:
                    self.stats["total"] += 1
                    self._current_task = None
            except queue.Empty:
                continue

    def submit(self, fn, args=(), kwargs=None, timeout=200):
        """Submit task to GPU queue. Blocks until complete."""
        kwargs = kwargs or {}
        result_q = queue.Queue(maxsize=1)
        task = {"fn": fn, "args": args, "kwargs": kwargs, "result_q": result_q}

        try:
            self._queue.put(task, timeout=5)
        except queue.Full:
            raise RuntimeError("Hệ thống đang bận. Thử lại sau.")

        try:
            status, result = result_q.get(timeout=timeout)
            if status == "error":
                raise RuntimeError(result)
            return result
        except queue.Empty:
            raise TimeoutError("Inference timeout.")

    @property
    def queue_size(self):
        return self._queue.qsize()

    @property
    def is_busy(self):
        return self._current_task is not None


# Global queue
gpu_queue = InferenceQueue(max_size=10)


# ============================================================
# C4: Health Check
# ============================================================

def health_check():
    """Return system health status."""
    gpu = get_gpu_info()
    disk = psutil.disk_usage("/")
    ram = psutil.virtual_memory()

    status = "healthy"
    issues = []

    if gpu and gpu["memory_pct"] > 95:
        status = "degraded"
        issues.append("GPU memory > 95%")
    if disk.percent > 90:
        status = "degraded"
        issues.append("Disk > 90%")
    if ram.percent > 90:
        status = "degraded"
        issues.append("RAM > 90%")

    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "issues": issues,
        "gpu": gpu,
        "ram_pct": ram.percent,
        "disk_free_gb": round(disk.free / (1024 ** 3), 1),
        "queue_size": gpu_queue.queue_size,
        "queue_busy": gpu_queue.is_busy,
    }


# ============================================================
# M3: Audio Preprocessing
# ============================================================

def preprocess_audio(audio_path, target_sr=24000):
    """Normalize audio: mono, target sample rate, normalize volume."""
    data, sr = sf.read(str(audio_path), dtype="float32")

    # Convert to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        import torchaudio.functional as F
        tensor = torch.from_numpy(data).unsqueeze(0)
        tensor = F.resample(tensor, sr, target_sr)
        data = tensor.squeeze(0).numpy()
        sr = target_sr

    # Normalize volume (peak normalization to -3dB)
    peak = np.abs(data).max()
    if peak > 0:
        target_peak = 10 ** (-3 / 20)  # -3dB
        data = data * (target_peak / peak)

    # Remove leading/trailing silence
    threshold = 0.01
    non_silent = np.where(np.abs(data) > threshold)[0]
    if len(non_silent) > 0:
        start = max(0, non_silent[0] - int(sr * 0.1))
        end = min(len(data), non_silent[-1] + int(sr * 0.1))
        data = data[start:end]

    # Clip to prevent distortion
    data = np.clip(data, -1.0, 1.0)

    return data, sr


def validate_audio(audio_path, min_duration=1, max_duration=0):
    """Validate audio file. max_duration=0 means no limit."""
    try:
        data, sr = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        duration = len(data) / sr

        if duration < min_duration:
            return None, f"Audio quá ngắn ({duration:.1f}s). Cần ít nhất {min_duration}s."
        if max_duration > 0 and duration > max_duration:
            return None, f"Audio quá dài ({duration:.1f}s). Tối đa {max_duration}s."

        # Check for silence
        rms = np.sqrt(np.mean(data ** 2))
        if rms < 0.001:
            return None, "Audio gần như im lặng."

        return {"duration": duration, "sr": sr, "rms": rms}, "OK"
    except Exception as e:
        return None, f"Không đọc được audio: {e}"


# ============================================================
# M4: Memory Monitor (background thread)
# ============================================================

def start_memory_monitor(interval=30):
    """Background thread monitoring GPU memory and auto-unloading idle models."""

    def _monitor():
        while True:
            try:
                # Unload Whisper if idle > 5 min
                models.maybe_unload_whisper(idle_seconds=300)

                # Log GPU status periodically
                gpu = get_gpu_info()
                if gpu and gpu["memory_pct"] > 85:
                    logger.warning(f"GPU memory high: {gpu['memory_pct']}%")

            except Exception:
                pass
            time.sleep(interval)

    thread = threading.Thread(target=_monitor, daemon=True)
    thread.start()
    return thread
