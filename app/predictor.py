"""Catálogo de modelos, preprocesamiento de imágenes e inferencia.

Cada modelo de la aplicación vive en su propia carpeta dentro de models/:

    models/<id>/
    ├── model.pt       peso o modelo completo
    ├── classes.json   clases en el mismo orden que los índices del modelo
    └── modelo.json    tipo de modelo, preprocesamiento y métricas de prueba

Tipos soportados (campo "tipo" de modelo.json):
- "ultralytics": checkpoint de Ultralytics (YOLO). Se carga con YOLO(...) y usa el
  mismo preprocesamiento de validación que se usó al entrenar.
- "torch": modelo completo de PyTorch (torch.save(model)) o TorchScript. El
  preprocesamiento se define en modelo.json ("redimension", "normalizacion", "salida").

Este módulo no depende de Streamlit, por lo que puede reutilizarse
desde el notebook o desde otro servicio.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms

import config

# Ultralytics instala paquetes por su cuenta cuando le falta alguno (p. ej. al abrir un archivo
# que no es imagen); en la app se desactiva para que nunca modifique el entorno en ejecución.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

MODEL_FILE = "model.pt"
CLASSES_FILE = "classes.json"
META_FILE = "modelo.json"
MODEL_TYPES = ("ultralytics", "torch")
RESIZE_MODES = ("recorte_central", "estirar")
OUTPUT_TYPES = ("logits", "probabilidades")


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


@dataclass
class ModelInfo:
    """Modelo disponible en models/, descrito por su modelo.json."""

    id: str  # nombre de la carpeta
    path: Path
    meta: dict

    @property
    def name(self) -> str:
        return self.meta.get("nombre") or self.id

    @property
    def kind(self) -> str:
        return self.meta.get("tipo", "")

    @property
    def img_size(self) -> int:
        return int(self.meta.get("imgsz") or config.DEFAULT_IMG_SIZE)

    @property
    def metrics(self) -> dict:
        return self.meta.get("metricas_test") or {}

    @property
    def mtime(self) -> float:
        """Fecha de modificación del peso: cambia si se vuelve a exportar el modelo."""
        return (self.path / MODEL_FILE).stat().st_mtime


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def display_name(class_id: str) -> str:
    """Convierte el nombre interno en nombre visible: homer_simpson -> Homer Simpson."""
    return " ".join(word.capitalize() for word in class_id.split("_"))


def load_classes(path: Path) -> list:
    """Lee classes.json y devuelve [(nombre interno, nombre visible), ...] en orden de índice."""
    if not path.exists():
        raise PredictorError(f"No se encontró el archivo de clases: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PredictorError(f"El archivo {path.name} no es un JSON válido: {exc}") from exc
    if not isinstance(data, list):
        raise PredictorError(f"El archivo {path.name} debe contener una lista de clases.")

    classes = []
    for item in data:
        if isinstance(item, str):
            classes.append((item, display_name(item)))
        elif isinstance(item, dict) and "id" in item:
            classes.append((item["id"], item.get("label") or display_name(item["id"])))
        else:
            raise PredictorError(f"Entrada no válida en {path.name}: {item!r}")
    if not classes:
        raise PredictorError(f"El archivo {path.name} no contiene clases.")
    return classes


def load_model_info(folder: Path) -> ModelInfo:
    """Lee y valida el modelo.json de una carpeta de models/."""
    for name in (META_FILE, MODEL_FILE, CLASSES_FILE):
        if not (folder / name).exists():
            raise PredictorError(f"{folder.name}: falta {name}.")
    try:
        meta = json.loads((folder / META_FILE).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PredictorError(f"{folder.name}: {META_FILE} no es un JSON válido: {exc}") from exc
    problem = _check_meta(meta)
    if problem:
        raise PredictorError(f"{folder.name}: {META_FILE} {problem}")
    return ModelInfo(id=folder.name, path=folder, meta=meta)


def _check_meta(meta) -> str:
    """Devuelve el primer problema de formato de modelo.json, o "" si es válido."""
    if not isinstance(meta, dict):
        return "debe ser un objeto JSON."
    if meta.get("tipo") not in MODEL_TYPES:
        return f"tiene \"tipo\" {meta.get('tipo')!r}; debe ser uno de {MODEL_TYPES}."
    imgsz = meta.get("imgsz")
    if imgsz is not None and not (isinstance(imgsz, int) and not isinstance(imgsz, bool) and imgsz > 0):
        return "tiene \"imgsz\" no válido; debe ser un entero positivo."
    for key in ("metricas_test", "origen"):
        if meta.get(key) is not None and not isinstance(meta[key], dict):
            return f"tiene \"{key}\" no válido; debe ser un objeto JSON o null."
    if meta.get("redimension", "recorte_central") not in RESIZE_MODES:
        return f"tiene \"redimension\" no válida; debe ser una de {RESIZE_MODES}."
    if meta.get("salida", "logits") not in OUTPUT_TYPES:
        return f"tiene \"salida\" no válida; debe ser una de {OUTPUT_TYPES}."
    norm = meta.get("normalizacion")
    if norm is not None and not (
        isinstance(norm, dict)
        and all(isinstance(norm.get(k), list) and len(norm[k]) == 3 and all(map(is_number, norm[k])) for k in ("media", "desviacion"))
    ):
        return "tiene \"normalizacion\" no válida; debe ser null o {\"media\": [3 números], \"desviacion\": [3 números]}."
    return ""


def _sort_key(model: ModelInfo):
    f1 = model.metrics.get("f1_macro")
    f1 = f1 if is_number(f1) else None
    return (f1 is None, -(f1 or 0.0), model.id)


def list_models(models_dir: Path = None) -> tuple:
    """Devuelve (modelos válidos, problemas), con los modelos del mayor al menor F1 macro de prueba.

    Las carpetas que no tienen modelo.json se ignoran; las que lo tienen pero están
    incompletas o mal formadas se informan en "problemas" para mostrarlas en la interfaz.
    """
    models_dir = models_dir or config.MODELS_DIR
    models, problems = [], []
    if not models_dir.is_dir():
        return models, problems
    for folder in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        if not (folder / META_FILE).exists():
            continue
        try:
            models.append(load_model_info(folder))
        except PredictorError as exc:
            problems.append(str(exc))
    models.sort(key=_sort_key)
    return models, problems


def build_transform(img_size: int, resize: str = "recorte_central", normalization=None) -> transforms.Compose:
    """Redimensionamiento, escalado a [0, 1] y normalización opcional."""
    if resize not in RESIZE_MODES:
        raise PredictorError(f"\"redimension\" debe ser uno de {RESIZE_MODES}, no {resize!r}.")
    if resize == "estirar":
        steps = [transforms.Resize((img_size, img_size))]
    else:  # lado corto a img_size y recorte central, como en validación
        steps = [transforms.Resize(img_size), transforms.CenterCrop(img_size)]
    steps.append(transforms.ToTensor())  # escala los píxeles a [0, 1]
    if normalization:
        steps.append(transforms.Normalize(normalization["media"], normalization["desviacion"]))
    return transforms.Compose(steps)


def open_image(file) -> Image.Image:
    """Abre la imagen, corrige la orientación EXIF y la convierte a RGB."""
    try:
        image = Image.open(file)
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except Exception as exc:  # Ultralytics reemplaza Image.open y puede lanzar otros tipos de error
        raise PredictorError("No se pudo leer la imagen. Usa JPG, PNG o WEBP.") from exc


def _load_torch_model(path: Path, device: str):
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
            "El archivo contiene solo pesos (state_dict) o un checkpoint. "
            "Exporte el modelo completo, en TorchScript, o use \"tipo\": \"ultralytics\" si es de YOLO."
        )
    if not callable(model):
        raise PredictorError(f"El archivo {path.name} no contiene un modelo válido.")
    if isinstance(model, torch.nn.Module):
        # Los pesos pueden venir en media precisión para que el archivo pese la mitad;
        # la inferencia se hace en float32
        model.float()
    model.eval()
    return model


class _UltralyticsModel:
    """Checkpoint de Ultralytics: YOLO aplica su propio preprocesamiento de validación."""

    def __init__(self, info: ModelInfo, device: str):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise PredictorError("Este modelo requiere el paquete ultralytics: pip install ultralytics") from exc
        try:
            self.model = YOLO(str(info.path / MODEL_FILE), task="classify")
        except Exception as exc:
            raise PredictorError(f"No se pudo cargar el modelo ({info.id}/{MODEL_FILE}): {exc}") from exc
        self.img_size = info.img_size
        self.device = 0 if device.startswith("cuda") else "cpu"
        self.class_ids = [self.model.names[i] for i in range(len(self.model.names))]

    @property
    def num_outputs(self) -> int:
        return len(self.class_ids)

    def describe(self) -> str:
        s = self.img_size
        return f"lado corto a {s} px, recorte central de {s}×{s} y escala a [0, 1] (transformación de validación de Ultralytics)"

    def probabilities(self, image: Image.Image) -> torch.Tensor:
        result = self.model.predict(image, imgsz=self.img_size, device=self.device, verbose=False)[0]
        return result.probs.data.float().cpu()


class _TorchModel:
    """Modelo completo de PyTorch o TorchScript, con el preprocesamiento descrito en modelo.json."""

    def __init__(self, info: ModelInfo, device: str):
        self.model = _load_torch_model(info.path / MODEL_FILE, device)
        self.device = device
        self.img_size = info.img_size
        self.resize = info.meta.get("redimension", "recorte_central")
        self.normalization = info.meta.get("normalizacion")
        self.outputs_probs = info.meta.get("salida", "logits") == "probabilidades"
        self.transform = build_transform(self.img_size, self.resize, self.normalization)
        self.class_ids = None  # el modelo no guarda nombres de clases: se usa classes.json
        self.num_outputs = self._count_outputs()

    def _count_outputs(self) -> int:
        dummy = torch.zeros(1, 3, self.img_size, self.img_size, device=self.device)
        try:
            with torch.inference_mode():
                outputs = self.model(dummy)
        except Exception as exc:
            raise PredictorError(f"El modelo falló con una entrada de {self.img_size}×{self.img_size}: {exc}") from exc
        if not isinstance(outputs, torch.Tensor) or outputs.ndim != 2:
            raise PredictorError("La salida del modelo debe ser un tensor de forma [lote, número de clases].")
        return outputs.shape[1]

    def describe(self) -> str:
        s = self.img_size
        size = f"estirada a {s}×{s}" if self.resize == "estirar" else f"lado corto a {s} px y recorte central de {s}×{s}"
        norm = (
            f"media {self.normalization['media']}, desviación {self.normalization['desviacion']}"
            if self.normalization
            else "escala a [0, 1]"
        )
        return f"{size}; {norm}"

    def probabilities(self, image: Image.Image) -> torch.Tensor:
        tensor = self.transform(image).unsqueeze(0).to(self.device)  # [1, 3, H, W]
        with torch.inference_mode():
            outputs = self.model(tensor)
        probs = outputs if self.outputs_probs else torch.softmax(outputs, dim=1)
        return probs.squeeze(0).float().cpu()


_BACKENDS = {"ultralytics": _UltralyticsModel, "torch": _TorchModel}


class Predictor:
    """Clasificador de imágenes. Sin modelo (model_info=None) funciona en modo demo."""

    def __init__(self, model_info: ModelInfo = None, device: str = config.DEVICE):
        self.info = model_info
        self.device = device
        self.demo = model_info is None
        if self.demo:
            classes = load_classes(config.DEMO_CLASSES_PATH)
            self.backend = None
            self.transform = build_transform(config.DEFAULT_IMG_SIZE)
        else:
            classes = load_classes(model_info.path / CLASSES_FILE)
            self.backend = _BACKENDS[model_info.kind](model_info, device)
        self.class_ids = [class_id for class_id, _ in classes]
        self.labels = [label for _, label in classes]
        if not self.demo:
            self._check_classes()

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def describe_preprocessing(self) -> str:
        if self.demo:
            s = config.DEFAULT_IMG_SIZE
            return f"lado corto a {s} px, recorte central de {s}×{s} y escala a [0, 1]"
        return self.backend.describe()

    def _check_classes(self) -> None:
        """Verifica que la salida del modelo coincida con classes.json (cantidad y, si se conoce, orden)."""
        if self.backend.num_outputs != self.num_classes:
            raise PredictorError(
                f"El modelo tiene {self.backend.num_outputs} salidas pero classes.json "
                f"tiene {self.num_classes} clases."
            )
        if self.backend.class_ids is not None and self.backend.class_ids != self.class_ids:
            raise PredictorError("El orden de las clases del modelo no coincide con classes.json.")

    def _demo_probs(self, image: Image.Image) -> torch.Tensor:
        """Probabilidades aleatorias, reproducibles para una misma imagen."""
        seed = int(hashlib.md5(image.tobytes()).hexdigest()[:8], 16)
        generator = torch.Generator().manual_seed(seed)
        logits = torch.randn(self.num_classes, generator=generator) * 3
        return torch.softmax(logits, dim=0)

    def predict(self, image: Image.Image) -> Prediction:
        start = time.perf_counter()
        if self.demo:
            self.transform(image)  # se ejecuta el preprocesamiento aunque no haya modelo
            probs = self._demo_probs(image)
        else:
            probs = self.backend.probabilities(image)

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
