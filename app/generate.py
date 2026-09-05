import re
import threading
import time
import uuid
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

from app.config import DEVICE, IDLE_UNLOAD_SECONDS, IMAGES_DIR, MODEL_ID

_pipeline = None
_last_used = 0.0
_lock = threading.Lock()

# Only safe, simple folder-name characters — conversation_id normally comes
# from Hermes' own UUIDs, but Iris is independently reachable on the LAN,
# so don't trust it blindly (no path traversal via "../" etc.).
_SAFE_FOLDER = re.compile(r"^[A-Za-z0-9_-]+$")


def _load_pipeline():
    """SDXL's pipeline has no safety_checker constructor arg (unlike
    SD1.5's pipeline) — omitted rather than guessed at."""
    return StableDiffusionXLPipeline.from_pretrained(MODEL_ID, dtype=torch.float16).to(DEVICE)


def generate_image(prompt: str, conversation_id: str | None = None) -> str:
    """Generates an image, saves it under IMAGES_DIR/<conversation_id>/,
    returns the relative path (e.g. "abc123/<uuid>.png"). Holds _lock for
    the whole load+generate — one GPU, one pipeline, requests are
    correctly serialized regardless of the idle-unload feature; this also
    prevents the idle-unload thread from freeing the pipeline mid-use."""
    global _pipeline, _last_used
    with _lock:
        if _pipeline is None:
            _pipeline = _load_pipeline()
        image = _pipeline(prompt).images[0]
        _last_used = time.time()

    folder = conversation_id if conversation_id and _SAFE_FOLDER.match(conversation_id) else "unsorted"
    output_dir = Path(IMAGES_DIR) / folder
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4()}.png"
    image.save(output_dir / filename)
    return f"{folder}/{filename}"


def _unload_if_idle() -> None:
    global _pipeline
    with _lock:
        if _pipeline is not None and (time.time() - _last_used) > IDLE_UNLOAD_SECONDS:
            del _pipeline
            _pipeline = None
            torch.cuda.empty_cache()


def _idle_unload_loop() -> None:
    while True:
        time.sleep(30)
        _unload_if_idle()


threading.Thread(target=_idle_unload_loop, daemon=True).start()
