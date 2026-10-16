# Aplicación web: clasificador de personajes de Los Simpson

Servicio web en Streamlit que recibe una imagen, la preprocesa automáticamente
(orientación, canales RGB, redimensionamiento y escalado a [0, 1]) y devuelve el
personaje predicho junto con su nivel de confianza y las tres clases más probables.

Admite **varios modelos a la vez**: cada uno vive en su carpeta dentro de `models/` y se
elige en el selector de la barra lateral, que muestra sus métricas en el conjunto de prueba.

## Requisitos

- Python 3.10 o superior.
- Dependencias de `requirements.txt` (incluye `ultralytics`, necesario para los modelos YOLO).

## Instalación y ejecución

Lo más simple es usar el mismo entorno del notebook (`DeepLearningModelos/.venv`):

```bash
cd DeepLearningModelos
source .venv/bin/activate
cd app
streamlit run app.py
```

O un entorno propio solo para la app:

```bash
cd DeepLearningModelos/app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

La aplicación queda disponible en `http://localhost:8501`.

> Si no se dispone de GPU, se puede instalar la versión de PyTorch solo para CPU (más liviana):
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

## Modelos

```
models/
└── <id>/                 p. ej. yolo11m-cls_test3
    ├── model.pt          peso o modelo completo
    ├── classes.json      clases en el orden de los índices del modelo
    └── modelo.json       tipo, preprocesamiento y métricas de prueba
```

- El selector lista todas las carpetas de `models/` que tienen `modelo.json`, del mayor al menor **F1 macro** de prueba.
- Para que aparezca un modelo nuevo basta con recargar la página. Si se reemplaza el `model.pt` de una carpeta, también se vuelve a cargar solo.
- Mientras no haya ningún modelo, la aplicación funciona en **modo demo**, con predicciones aleatorias sobre las clases de `classes.json`.

### Agregar una prueba

Desde `DeepLearningModelos/`:

```bash
python exportar_modelo.py test3                 # YOLO11
python exportar_modelo.py resnet50 test1        # ResNet-50
python exportar_modelo.py convnext_tiny test1   # ConvNeXt-Tiny
```

- **YOLO11:** copia `resultados/yolo11/test3/weights/best.pt` a `models/yolo11m-cls_test3/` junto con sus clases y métricas (`"tipo": "ultralytics"`).
- **ResNet-50:** guarda el modelo completo de `resultados/resnet50/test1/weights/best.pt` en `models/resnet50_test1/` con `"tipo": "torch"`, en media precisión (~47 MB en vez de ~94 MB). Su `modelo.json` incluye el recorte central y la normalización de ImageNet, y la app lo vuelve a float32 al cargarlo.
- **ConvNeXt-Tiny:** igual que ResNet-50: guarda el modelo completo de `resultados/convnext_tiny/test1/weights/best.pt` en `models/convnext_tiny_test1/` con `"tipo": "torch"`, en media precisión (~56 MB en vez de ~111 MB), con el mismo recorte central y normalización de ImageNet.

Los notebooks hacen lo mismo con `EXPORTAR_A_APP = True`.

### Formato de `modelo.json`

```json
{
  "nombre": "YOLO11m-cls · test3",
  "tipo": "ultralytics",
  "imgsz": 224,
  "metricas_test": {"accuracy": 0.95, "f1_macro": 0.93, "f1_ponderado": 0.95},
  "origen": {"notebook": "yolo11.ipynb", "prueba": "test3"}
}
```

| Campo | Obligatorio | Descripción |
|---|---|---|
| `tipo` | Sí | `"ultralytics"` (checkpoint de YOLO) o `"torch"` (modelo completo con `torch.save(model)` o TorchScript). |
| `nombre` | No | Texto del selector. Por defecto, el nombre de la carpeta. |
| `imgsz` | No | Lado de la imagen de entrada (por defecto 224). |
| `metricas_test` | No | Métricas en el conjunto de prueba. Se muestran `accuracy`, `f1_macro`, `f1_ponderado`, `precision_macro`, `recall_macro`, `accuracy_top5` e `imagenes_test` (las que estén). |
| `origen` | No | Datos libres sobre el entrenamiento (se muestran tal cual). |
| `redimension` | Solo `torch` | `"recorte_central"` (lado corto a `imgsz` y recorte central; por defecto) o `"estirar"` (directo a `imgsz`×`imgsz`). |
| `normalizacion` | Solo `torch` | `null` (solo escala a [0, 1]; por defecto) o `{"media": [...], "desviacion": [...]}`, p. ej. los valores de ImageNet. |
| `salida` | Solo `torch` | `"logits"` (por defecto; la app aplica softmax) o `"probabilidades"`. |

Con `"tipo": "ultralytics"`, el preprocesamiento lo hace Ultralytics con su transformación de validación, la misma con que se evaluó el modelo. Para un modelo propio de PyTorch (p. ej. una ResNet de torchvision), el preprocesamiento de `modelo.json` debe ser **idéntico** al de validación/test del notebook (no el de aumentación). Si es una arquitectura propia, `torch.load` busca su clase en el módulo donde estaba al guardar: definirla en un archivo `.py` (p. ej. `arquitectura.py`), importarla desde ahí en el notebook y copiar ese mismo archivo a `app/` (si se define dentro del notebook, queda como `__main__` y la app no la encuentra).

## Estructura

| Archivo | Descripción |
|---|---|
| `app.py` | Interfaz web (Streamlit): selector de modelos, carga de la imagen y resultado. |
| `predictor.py` | Catálogo de modelos, carga, preprocesamiento e inferencia (no depende de Streamlit). |
| `config.py` | Parámetros generales (carpeta de modelos, umbral, Top-K, dispositivo). |
| `classes.json` | Clases del modo demo (las 42 del dataset). |
| `models/` | Un subdirectorio por modelo. |
| `.streamlit/config.toml` | Tema visual: siempre claro, con fondo gris claro (se lee al ejecutar `streamlit run` desde `app/`). |
| `WEBAPP_SPEC.md` | Especificación técnica de la aplicación. |
