import cv2
import numpy as np


def analizar_roya(img, mask_hoja):
    # 1. Aislar hoja
    hoja_aislada = cv2.bitwise_and(img, img, mask=mask_hoja)

    # 2. Detección de color (rangos ajustables)
    hsv = cv2.cvtColor(hoja_aislada, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 50, 50])
    upper = np.array([30, 255, 255])
    mask_roya = cv2.inRange(hsv, lower, upper)

    # 3. Cálculo de áreas
    area_total = cv2.countNonZero(mask_hoja)
    area_roya = cv2.countNonZero(mask_roya)
    porcentaje = (area_roya / area_total) * 100 if area_total > 0 else 0.0

    return {
        'porcentaje_roya': float(porcentaje),
        'area_afectada': int(area_roya),
        'area_total': int(area_total)
    }