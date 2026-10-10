"""Carga del modelo, preprocesamiento de imágenes e inferencia.

Este módulo no depende de Streamlit, por lo que puede reutilizarse
desde el notebook o desde otro servicio.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

import config


class PredictorError(Exception):
    """Error con un mensaje apto para mostrar al usuario."""


@dataclass
class Prediction:
    label: str
    confidence: float
    top_k: list  # [(nombre visible, probabilidad), ...]
    is_confident: bool
    demo: bool
    elapsed_ms: float


def display_name(class_id: str) -> str:
    """Convierte el nombre interno en nombre visible: homer_simpson -> Homer Simpson."""
    return " ".join(word.capitalize() for word in class_id.split("_"))


def load_classes(path: Path) -> list:
    """Lee classes.json y devuelve la lista de nombres visibles, en orden de índice."""
    if not path.exists():
        raise PredictorError(f"No se encontró el archivo de clases: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PredictorError(f"El archivo {path.name} no es un JSON válido: {exc}") from exc

    labels = []
    for item in data:
        if isinstance(item, str):
            labels.append(display_name(item))
        elif isinstance(item, dict) and "id" in item:
            labels.append(item.get("label") or display_name(item["id"]))
        else:
            raise PredictorError(f"Entrada no válida en {path.name}: {item!r}")
    if not labels:
        raise PredictorError(f"El archivo {path.name} no contiene clases.")
    return labels


def build_transform() -> transforms.Compose:
    """Redimensionamiento, escalado a [0, 1] y normalización opcional."""
    steps = [
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),  # escala los píxeles a [0, 1]
    ]
    if config.NORMALIZE_MEAN is not None and config.NORMALIZE_STD is not None:
        steps.append(transforms.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD))
    return transforms.Compose(steps)


def open_image(file) -> Image.Image:
    """Abre la imagen, corrige la orientación EXIF y la convierte a RGB."""
    try:
        image = Image.open(file)
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise PredictorError("No se pudo leer la imagen. Usa JPG, PNG o WEBP.") from exc


def _load_model(path: Path, device: str):
    """Carga un modelo completo guardado con torch.save o, en su defecto, uno TorchScript."""
    try:
        model = torch.load(str(path), map_location=device, weights_only=False)
    except Exception as exc:
        # TorchScript se intenta después porque no está soportado en Python 3.14+
        try:
            model = torch.jit.load(str(path), map_location=device)
        except Exception:
            raise PredictorError(f"No se pudo cargar el modelo ({path.name}): {exc}") from exc

    if isinstance(model, dict):
        raise PredictorError(
            "El archivo contiene solo pesos (state_dict). "
            "Exporte el modelo completo o en TorchScript."
        )
    if not callable(model):
        raise PredictorError(f"El archivo {path.name} no contiene un modelo válido.")
    model.eval()
    return model


class Predictor:
    """Clasificador de imágenes. Si no existe el modelo, funciona en modo demo."""

    def __init__(self, model_path: Path = config.MODEL_PATH,
                 classes_path: Path = config.CLASSES_PATH,
                 device: str = config.DEVICE):
        self.model_path = Path(model_path)
        self.device = device
        self.labels = load_classes(Path(classes_path))
        self.transform = build_transform()
        self.demo = not self.model_path.exists()
        self.model = None if self.demo else _load_model(self.model_path, device)
        if not self.demo:
            self._check_output_size()

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def _check_output_size(self) -> None:
        """Verifica que la salida del modelo coincida con la cantidad de clases."""
        dummy = torch.zeros(1, 3, config.IMG_SIZE, config.IMG_SIZE, device=self.device)
        with torch.inference_mode():
            outputs = self.model(dummy)
        if outputs.ndim != 2 or outputs.shape[1] != self.num_classes:
            raise PredictorError(
                f"El modelo tiene {outputs.shape[-1]} salidas pero classes.json "
                f"tiene {self.num_classes} clases."
            )

    def _demo_probs(self, image: Image.Image) -> torch.Tensor:
        """Probabilidades aleatorias, reproducibles para una misma imagen."""
        seed = int(hashlib.md5(image.tobytes()).hexdigest()[:8], 16)
        generator = torch.Generator().manual_seed(seed)
        logits = torch.randn(1, self.num_classes, generator=generator) * 3
        return torch.softmax(logits, dim=1)

    def predict(self, image: Image.Image) -> Prediction:
        start = time.perf_counter()
        tensor = self.transform(image).unsqueeze(0)  # [1, 3, H, W]

        if self.demo:
            probs = self._demo_probs(image)
        else:
            with torch.inference_mode():
                outputs = self.model(tensor.to(self.device))
            probs = outputs if config.MODEL_OUTPUTS_PROBS else torch.softmax(outputs, dim=1)

        probs = probs.squeeze(0).float().cpu()
        k = min(config.TOP_K, self.num_classes)
        values, indices = torch.topk(probs, k)
        top_k = [(self.labels[i], float(v)) for v, i in zip(values, indices)]
        label, confidence = top_k[0]

        return Prediction(
            label=label,
            confidence=confidence,
            top_k=top_k,
            is_confident=confidence >= config.CONFIDENCE_THRESHOLD,
            demo=self.demo,
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
