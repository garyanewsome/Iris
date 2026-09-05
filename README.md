# Iris

Image generation service (Stable Diffusion XL).

## What it does

`POST /generate {"prompt": "...", "conversation_id": "..."}` → generates an image, saves it, returns `{"image_url": "..."}`. Images are served as static files from the same process (`/images/<conversation_id>/<filename>.png`). `conversation_id` is optional — omit it for an `unsorted/` folder; when present it's validated against a strict `[A-Za-z0-9_-]+` pattern (no path traversal).

`GET /health`

No chat logic, no persona — a single-purpose generation service meant to be called as a tool by other applications.

## Model

`stabilityai/stable-diffusion-xl-base-1.0` via `diffusers`/`StableDiffusionXLPipeline`, fp16, native 1024x1024 output. Loaded lazily on first request and automatically unloaded after `IDLE_UNLOAD_SECONDS` (default 300) of inactivity — frees the GPU for other workloads (Ollama) between uses, at the cost of a reload delay (~8s) on the next request after an idle period.

## Deployment shape — bare-metal, not K3s

Runs directly on the homelab host as a systemd service (`iris.service`), not in the K3s cluster — the cluster has no GPU device plugin configured, so no pod can access the GPU. Reachable on the LAN by IP:port (`http://192.168.1.157:8100`), no Ingress/hostname.

```bash
sudo systemctl status iris
sudo systemctl restart iris
```

## Setup

Needs Python 3.12 specifically (the OS default, 3.14, is too new for PyTorch's published wheels):

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update && sudo apt install -y python3.12 python3.12-venv
```

```bash
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8100
```

`torch` is deliberately not in `requirements.txt` — it needs the CUDA-specific index URL above; a plain `pip install torch` pulls a CPU-only build.

## Status

- [x] Generating real images on the RTX 3060, verified visually across many prompts
- [x] Idle-unload — verified both the pipeline reference clears and GPU memory is actually reclaimed (not just a superficial fix)
- [x] Per-conversation output folders
- [x] Wired up as a Hermes tool (`generate_image`)

## Not yet decided / open

- No SDXL refiner stage (base model only)
- `IDLE_UNLOAD_SECONDS` (300) is a first guess, not tuned against real usage patterns
