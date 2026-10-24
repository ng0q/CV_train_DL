from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File

from app import model_loader
from io import BytesIO
from app.schemas import PredictionResponse
from PIL import Image


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.load_artifacts()
    yield


app = FastAPI(
    title="Image classifier API",
    description="Predict class of a picture",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Service"])
def health():
    if model_loader.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="Unsupported image format")
    
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    
    try:
        image = Image.open(BytesIO(content))
        image.load()
        image = image.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File is not a valid image")
    
    predicted_class, confidence = model_loader.predict(image)
    
    return PredictionResponse(predicted_class = predicted_class, confidence =confidence)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)