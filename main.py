from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from model_utils.segmentator import Segmentator
from model_utils.roya_detector import analizar_roya

app = FastAPI()

# Configura CORS para Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Carga el modelo YOLO al iniciar (ajusta la ruta)
segmentator = Segmentator("models/best.pt")


@app.post("/analizar")
async def analizar_imagen(file: UploadFile = File(...)):
    try:
        # 1. Leer imagen
        img_bytes = await file.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # 2. Segmentar hoja
        mask_hoja = segmentator.predict_mask(img)

        # 3. Detectar roya
        resultado = analizar_roya(img, mask_hoja)

        return {
            "success": True,
            "data": resultado
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }