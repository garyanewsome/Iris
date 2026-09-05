from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import IMAGES_DIR, PUBLIC_BASE_URL
from app.generate import generate_image

app = FastAPI(title="Iris")

Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


class GenerateRequest(BaseModel):
    prompt: str
    conversation_id: str | None = None


class GenerateResponse(BaseModel):
    image_url: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    relative_path = generate_image(request.prompt, conversation_id=request.conversation_id)
    return {"image_url": f"{PUBLIC_BASE_URL}/images/{relative_path}"}
