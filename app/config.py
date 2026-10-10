"""Parámetros de la aplicación web.

IMG_SIZE y la normalización deben coincidir con las transformaciones de
validación/test usadas al entrenar el modelo (no con las de aumentación).
"""

from pathlib import Path

import torch

# Directorio base de la aplicación (las rutas se resuelven respecto a él)
BASE_DIR = Path(__file__).resolve().parent

# Rutas
MODEL_PATH = BASE_DIR / "models" / "model.pt"
CLASSES_PATH = BASE_DIR / "classes.json"

# Preprocesamiento
IMG_SIZE = 224
# None: solo se escala a [0, 1]. Con pesos de ImageNet usar:
# NORMALIZE_MEAN = [0.485, 0.456, 0.406]
# NORMALIZE_STD = [0.229, 0.224, 0.225]
NORMALIZE_MEAN = None
NORMALIZE_STD = None

# Salida del modelo: True si el modelo ya aplica softmax
MODEL_OUTPUTS_PROBS = False

# Presentación del resultado
CONFIDENCE_THRESHOLD = 0.50
TOP_K = 3

# Formatos de imagen permitidos
ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp"]

# Dispositivo de inferencia
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
