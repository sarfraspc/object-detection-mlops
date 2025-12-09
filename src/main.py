import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yolov8-api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP LOGIC
    model_path = os.getenv("MODEL_PATH", "yolov8n.pt")
    try:
        app.state.model = YOLO(model_path)
        logger.info(f"Model loaded: {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield  # The application runs while yielded
    
    # SHUTDOWN LOGIC
    logger.info("Model unloaded")

app = FastAPI(
    title="YOLOv8-Nano Object Detection API",
    version="1.0.0",
    description="Real-time object detection",
    lifespan=lifespan  
)

@app.get("/")
def health():
    return {"status": "healthy", "model": "yolov8n.pt"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="Invalid file: must be an image")
    
    contents = await file.read()
    
    if len(contents) > 10_000_000:  # 10 MB limit
        raise HTTPException(413, detail="File too large (max 10MB)")
    
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, detail="Cannot process image")

    # Access model securely
    model = app.state.model
    t0 = time.time()
    
    # Run inference
    results = model(image, verbose=False)[0]
    inference_time = time.time() - t0
    
    predictions = []
    boxes = results.boxes

    if boxes is not None and len(boxes) > 0:
        names = results.names
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            predictions.append({
                "bbox": [round(x, 1) for x in [x1, y1, x2, y2]],
                "confidence": round(conf, 3),
                "class_id": cls_id,
                "class_name": names.get(cls_id, "unknown"),
            })

    logger.info(f"Inference: {len(predictions)} objects in {inference_time:.3f}s")
    
    return JSONResponse({
        "predictions": predictions,
        "inference_time_s": round(inference_time, 3),
        "model": "yolov8n.pt",
    })