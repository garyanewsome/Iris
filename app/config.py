import os

MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
DEVICE = os.environ.get("DEVICE", "cuda")

# Same idle-unload pattern as Ollama's own keep_alive — matters much more
# for SDXL (~11.5GB VRAM) than it did for SD1.5 (~3.3GB): confirmed by
# testing that leaving SDXL loaded indefinitely pushed Ollama's chat model
# to 96% CPU / 4% GPU, a real, measured performance hit, not a theoretical
# concern.
IDLE_UNLOAD_SECONDS = int(os.environ.get("IDLE_UNLOAD_SECONDS", "300"))
IMAGES_DIR = os.environ.get("IMAGES_DIR", "/mnt/storage/iris-images")

# Bare-metal on the homelab host (see README) — reachable on the LAN
# directly by IP:port, no Ingress/hostname since this never runs in K3s.
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://192.168.1.157:8100")
