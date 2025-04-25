from ultralytics import YOLO
import cv2
import numpy as np


class Segmentator:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict_mask(self, img):
        """Devuelve máscara binaria de la hoja segmentada"""
        results = self.model(img)
        mask = results[0].masks[0].data[0].cpu().numpy()  # Máscara YOLOv8
        return (mask * 255).astype(np.uint8)