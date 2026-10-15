"""Parámetros de la aplicación web.

El tamaño de entrada, la normalización y el tipo de cada modelo NO se configuran aquí:
vienen en el modelo.json de cada carpeta de models/ (ver README.md), porque cada
modelo puede tener un preprocesamiento distinto.
"""

from pathlib import Path

import torch

# Directorio base de la aplicación (las rutas se resuelven respecto a él)
BASE_DIR = Path(__file__).resolve().parent

# Rutas
# Un subdirectorio por modelo: models/<id>/{model.pt, classes.json, modelo.json}
MODELS_DIR = BASE_DIR / "models"
# Clases del modo demo (se usa solo cuando no hay ningún modelo en MODELS_DIR)
DEMO_CLASSES_PATH = BASE_DIR / "classes.json"

# Tamaño de entrada del modo demo y de los modelos cuyo modelo.json no indica "imgsz"
DEFAULT_IMG_SIZE = 224

# Presentación del resultado
CONFIDENCE_THRESHOLD = 0.50
TOP_K = 3

# Formatos de imagen permitidos
ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp"]

# Dispositivo de inferencia
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
