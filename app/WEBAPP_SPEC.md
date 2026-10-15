# Especificación — Página web de clasificación (Streamlit)

Documento de referencia para construir la página web del proyecto. **Alcance: solo la página web.**
El entrenamiento del modelo y la generación del `.pt` quedan fuera de este documento (se harán en el notebook).

---

## 1. Objetivo

Página web lo más básica posible, con ventanas y botones ordenados, que permita:

1. **Subir** una imagen desde el computador.
2. **Procesar** la imagen y clasificarla con el modelo elegido en un selector (`.pt` de YOLO/Ultralytics o de PyTorch).
3. **Mostrar el resultado** en la misma página, p. ej.: **"Es Homer Simpson (92%)"**.

Cubre el punto 6 de la pauta (Despliegue del Modelo): recibir una imagen externa cualquiera,
preprocesarla automáticamente (resolución, canales y escala) y retornar la clase predicha con su confianza.

---

## 2. Decisiones tomadas

| Tema | Decisión |
|---|---|
| Framework | **Streamlit** (todo en Python, un solo `app.py`) |
| Dataset | **The_Simpsons_Characters_Data**: 42 clases (`dataset/simpsons_dataset/`); cada modelo usa las suyas (YOLO11: 25) |
| Modelos | **Varios a la vez**: una carpeta por modelo en `models/` (`model.pt` + `classes.json` + `modelo.json`) y un **selector** en la barra lateral con sus métricas de prueba |
| Clases | `classes.json` de cada modelo (no hardcodeadas en el código) |
| Formato del modelo | `"tipo"` en `modelo.json`: **`ultralytics`** (checkpoint de YOLO, se carga con `YOLO(...)`) o **`torch`** (`torch.save(model)`; TorchScript como respaldo) |
| Sin modelos disponibles | **Modo demo**: predicciones aleatorias + aviso visible |
| Resultado | **Top-1 destacado + Top-3** con barras de probabilidad |
| Confianza baja | **Aviso con umbral** (por defecto 50 %, configurable) |
| Entrada | **Solo subir archivo** (JPG, JPEG, PNG, WEBP) |
| Ubicación | `DeepLearningModelos/app/` |
| Idioma y estilo | Castellano técnico, con tildes y mayúscula solo al inicio de oración (también en títulos y botones) |
| Ejecución | Local, en CPU por defecto (usa GPU si está disponible) |

---

## 3. Estructura de archivos

```
DeepLearningModelos/
└── app/
    ├── WEBAPP_SPEC.md      ← este documento
    ├── app.py              ← interfaz Streamlit (solo UI)
    ├── config.py           ← parámetros generales (carpeta de modelos, umbral, Top-K, dispositivo)
    ├── predictor.py        ← catálogo de modelos, carga, preprocesamiento, inferencia, modo demo
    ├── classes.json        ← clases del modo demo (las 42 del dataset)
    ├── requirements.txt    ← dependencias
    ├── README.md           ← instrucciones de ejecución (entregable de la pauta)
    └── models/
        ├── .gitkeep
        └── <id>/           ← un modelo por carpeta, p. ej. yolo11m-cls_test3/
            ├── model.pt
            ├── classes.json
            └── modelo.json
```

**Regla:** `app.py` no contiene lógica de modelo; `predictor.py` no importa Streamlit
(así se puede probar por separado y reutilizar desde el notebook).

---

## 4. Diccionario (glosario)

| Término | Significado en este proyecto |
|---|---|
| **Clase** | Personaje de Los Simpson que el modelo puede reconocer (p. ej. `homer_simpson`). |
| **Nombre interno** | Nombre de la carpeta del dataset (`homer_simpson`). Es lo que va en `classes.json`. |
| **Nombre visible** | Versión legible para mostrar: `homer_simpson` → `Homer Simpson`. Se genera automáticamente (reemplazar `_` por espacio y capitalizar) salvo que `classes.json` traiga un nombre explícito. |
| **Índice de clase** | Posición de la clase en la salida del modelo. Debe coincidir **exactamente** con el orden usado al entrenar. |
| **Logits** | Salida cruda del modelo (vector de tamaño `NUM_CLASSES`). |
| **Probabilidades** | `softmax(logits)`; suman 1. |
| **Confianza** | Probabilidad de la clase Top-1, mostrada en %. |
| **Top-1 / Top-3** | La clase más probable / las 3 más probables. |
| **Umbral** (`CONFIDENCE_THRESHOLD`) | Si la confianza Top-1 es menor, se muestra un aviso de incertidumbre. |
| **Modo demo** | Modo sin modelo real: genera probabilidades aleatorias para probar la interfaz. |
| **Preprocesamiento** | Transformaciones que convierten la imagen subida en el tensor que espera el modelo. |
| **TorchScript** | Formato de PyTorch (`torch.jit.save`) que guarda modelo + arquitectura; no soportado en Python 3.14+. |

---

## 5. Contrato con el modelo (lo que el notebook debe entregar)

La página asume lo siguiente. **Si el entrenamiento cambia algo, se ajusta solo el `modelo.json` / `classes.json` de ese modelo.**

### 5.1 Carpeta del modelo
- Cada modelo es una carpeta `app/models/<id>/` con `model.pt`, `classes.json` y `modelo.json`.
- El formato completo de `modelo.json` está en `app/README.md`. Campo obligatorio: `"tipo"`.
- **YOLO (`"tipo": "ultralytics"`):** `model.pt` es el `best.pt` de Ultralytics, tal cual. Se exporta desde
  `DeepLearningModelos/` con `python exportar_modelo.py testN` (o `EXPORTAR_A_APP = True` en `yolo11.ipynb`).
- **PyTorch (`"tipo": "torch"`):** modelo completo con `torch.save(model, "model.pt")`.
  - Si la arquitectura es de `torchvision` (ResNet, EfficientNet, VGG, etc.), se carga sin código adicional.
  - Si es una CNN propia, `torch.load` busca la clase en el módulo donde estaba al guardar: definirla en un archivo
    `.py` (p. ej. `arquitectura.py`), importarla desde ahí en el notebook y copiar ese mismo archivo a `app/`
    (si se define dentro del notebook, queda como `__main__` y la app no la encuentra).
  - **ResNet-50 (`resnet50.ipynb`):** se exporta con `python exportar_modelo.py resnet50 testN`, que crea
    `app/models/resnet50_testN/` con el modelo en media precisión (~47 MB); la app lo convierte a float32 al cargarlo.
  - Alternativa: TorchScript (`torch.jit.script(model).save("model.pt")`), pero ⚠️ **PyTorch no soporta
    TorchScript en Python 3.14+**. Orden de carga: primero `torch.load`, luego `torch.jit.load`.

### 5.2 Entrada y salida
- `ultralytics`: Ultralytics recibe la imagen y aplica su propia transformación de validación (lado corto a `imgsz`,
  recorte central, escala a [0, 1]); devuelve probabilidades.
- `torch`: entrada `float32` de forma `[1, 3, imgsz, imgsz]`, RGB, preprocesada según `redimension` y `normalizacion`
  de `modelo.json`. Salida `[1, NUM_CLASSES]`: logits (la app aplica softmax) o probabilidades (`"salida": "probabilidades"`).

### 5.3 Clases
- `classes.json` contiene la lista de clases **en el mismo orden que los índices del modelo**.
- Con `torchvision.datasets.ImageFolder` y con Ultralytics, el orden es **alfabético por nombre de carpeta**.
- Formato aceptado:
  ```json
  ["abraham_grampa_simpson", "apu_nahasapeemapetilon", "..."]
  ```
  o con nombre visible explícito:
  ```json
  [{"id": "homer_simpson", "label": "Homer Simpson"}, ...]
  ```
- La app valida que el número de salidas del modelo coincida con `classes.json` y, en los modelos de Ultralytics
  (que guardan los nombres de sus clases), también el orden. Si no coincide, muestra un error claro.

### 5.4 Clases del dataset (confirmadas)
El dataset `simpsons_dataset/` contiene 42 carpetas (una por personaje, 20.933 imágenes JPG; el directorio tiene 41.866 archivos porque incluye una copia anidada idéntica).
El `classes.json` de la raíz de `app/` (modo demo) las contiene en orden alfabético. Cada modelo trae su propio `classes.json` (p. ej. YOLO11 usa 25 clases, las que tienen más de 100 imágenes):

```
abraham_grampa_simpson, agnes_skinner, apu_nahasapeemapetilon, barney_gumble, bart_simpson,
carl_carlson, charles_montgomery_burns, chief_wiggum, cletus_spuckler, comic_book_guy,
disco_stu, edna_krabappel, fat_tony, gil, groundskeeper_willie, homer_simpson, kent_brockman,
krusty_the_clown, lenny_leonard, lionel_hutz, lisa_simpson, maggie_simpson, marge_simpson,
martin_prince, mayor_quimby, milhouse_van_houten, miss_hoover, moe_szyslak, ned_flanders,
nelson_muntz, otto_mann, patty_bouvier, principal_skinner, professor_john_frink,
rainier_wolfcastle, ralph_wiggum, selma_bouvier, sideshow_bob, sideshow_mel, snake_jailbird,
troy_mcclure, waylon_smithers
```

> **Nota para el notebook (fuera del alcance de la web):** existe una subcarpeta duplicada
> `simpsons_dataset/simpsons_dataset/` (20.933 imágenes). Si se usa `ImageFolder` sobre `simpsons_dataset/`,
> aparecería como una clase 43 ficticia: hay que excluirla o eliminarla antes de entrenar.
> Además, hay clases con muy pocas imágenes (`lionel_hutz`: 3, `disco_stu`: 8, `troy_mcclure`: 8).
> Aun así, se recomienda regenerar `classes.json` desde el notebook para garantizar el mismo orden.

---

## 6. Configuración (`config.py`)

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `MODELS_DIR` | `models/` | Carpeta con un subdirectorio por modelo. |
| `DEMO_CLASSES_PATH` | `classes.json` | Clases del modo demo. |
| `DEFAULT_IMG_SIZE` | `224` | Tamaño de entrada del modo demo y de los modelos cuyo `modelo.json` no indica `imgsz`. |
| `CONFIDENCE_THRESHOLD` | `0.50` | Bajo este valor se muestra aviso de incertidumbre. |
| `TOP_K` | `3` | Cantidad de clases en el ranking. |
| `ALLOWED_TYPES` | `["jpg", "jpeg", "png", "webp"]` | Extensiones permitidas. |
| `DEVICE` | `"cuda"` si disponible, si no `"cpu"` | Dispositivo de inferencia. |

⚠️ **Importante:** el tamaño de entrada y el preprocesamiento de cada modelo van en su `modelo.json`, y deben ser
**idénticos** a los usados en validación/test del notebook (no los de *data augmentation*). Si no coinciden, las
predicciones serán malas. Los modelos de Ultralytics aplican su propia transformación de validación.

---

## 7. Lógica de procesamiento (`predictor.py`)

### 7.1 Catálogo y carga del modelo
1. En cada recarga se listan las carpetas de `models/` que tienen `modelo.json` (ordenadas por F1 macro de prueba).
   Las que están incompletas o mal formadas se muestran como aviso, sin detener la app.
2. Si no hay ningún modelo → **modo demo**.
3. El modelo elegido se carga una sola vez (`@st.cache_resource`, clave = carpeta + fecha de `model.pt`):
   - `ultralytics`: `YOLO("models/<id>/model.pt", task="classify")`.
   - `torch`: `torch.load(..., weights_only=False)`; si falla, `torch.jit.load(...)`. Si el resultado es un `dict`
     (state_dict o checkpoint) → error claro. Luego `model.eval()`.
4. Se valida `classes.json` contra la salida del modelo (número de clases y, en Ultralytics, el orden).
5. Al cambiar de modelo en el selector se borra el resultado anterior.

### 7.2 Preprocesamiento de la imagen subida
1. Abrir con Pillow: `Image.open(archivo)`.
2. Corregir orientación EXIF (`ImageOps.exif_transpose`) — fotos de celular.
3. Convertir a **RGB** (`.convert("RGB")`): resuelve PNG con transparencia (RGBA), escala de grises (L) y paletas (P).
4. **Redimensionar** según el modelo: lado corto a `imgsz` + recorte central (Ultralytics y `"redimension": "recorte_central"`)
   o directo a `imgsz × imgsz` (`"redimension": "estirar"`).
5. Convertir a tensor y **escalar a [0, 1]** (`transforms.ToTensor()`).
6. Si `modelo.json` trae `normalizacion` → aplicar `transforms.Normalize`.
7. Agregar dimensión de lote: `[3, H, W] → [1, 3, H, W]`.

### 7.3 Inferencia
1. `with torch.inference_mode(): logits = model(x.to(DEVICE))`.
2. `probs = softmax(logits, dim=1)` (salvo `"salida": "probabilidades"`; Ultralytics ya entrega probabilidades).
3. Obtener `top_k` (índices + probabilidades) y mapear índices a nombres visibles.
4. Devolver un resultado simple, p. ej.:
   ```python
   {
     "label": "Homer Simpson",
     "confidence": 0.92,
     "top_k": [("Homer Simpson", 0.92), ("Bart Simpson", 0.05), ("Lisa Simpson", 0.01)],
     "is_confident": True,       # confidence >= CONFIDENCE_THRESHOLD
     "demo": False,
     "elapsed_ms": 38.2,
   }
   ```

### 7.4 Modo demo
- Se activa automáticamente si no hay ningún modelo en `models/`.
- Genera probabilidades aleatorias (softmax de un vector aleatorio) sobre las clases de `classes.json`.
- La semilla se deriva del contenido de la imagen → la misma imagen da siempre el mismo resultado.
- Ejecuta igualmente el preprocesamiento real (para validar que funciona).

---

## 8. Interfaz (`app.py`)

### 8.1 Layout

```
┌────────────────────────────────────────────────────────────┐
│  🍩 Clasificador de personajes de Los Simpson              │
│  Sube una imagen y el modelo dirá qué personaje es.        │
│  [⚠️ Modo demo: no hay modelos en models/]       (si aplica)│
├─────────────────────────────┬──────────────────────────────┤
│  1. Imagen                  │  2. Resultado                │
│                             │                              │
│  [ Subir imagen (arrastrar) ]│  Es Homer Simpson            │
│  [ 🔍 Clasificar ] [Limpiar]│  Confianza: 92 %             │
│  ┌───────────────────────┐  │                              │
│  │   vista previa        │  │  Top-3                       │
│  │                       │  │  Homer Simpson   ▓▓▓▓▓▓▓▓ 92%│
│  │                       │  │  Bart Simpson    ▓        5% │
│  └───────────────────────┘  │  Lisa Simpson    ░        1% │
├─────────────────────────────┴──────────────────────────────┤
│ ▸ Detalles del modelo (colapsable): ruta, nº de clases,    │
│   preprocesamiento, dispositivo, tiempo de inferencia      │
└────────────────────────────────────────────────────────────┘

Barra lateral: **selector de modelo** + métricas de prueba del modelo elegido (accuracy, F1 macro,
F1 ponderado, precision y recall macro, accuracy top-5) y su origen (notebook, prueba, fecha, épocas).
```

### 8.2 Componentes
| Elemento | Componente Streamlit |
|---|---|
| Configuración de página | `st.set_page_config(page_title=..., page_icon="🍩", layout="wide")` |
| Título / subtítulo | `st.title`, `st.caption` |
| Selector de modelo | `st.sidebar` + `st.selectbox` y `st.metric` para las métricas de prueba |
| Aviso modo demo | `st.warning` |
| Dos columnas | `st.columns(2)` dentro de `st.container(border=True)` |
| Subir imagen | `st.file_uploader(type=ALLOWED_TYPES)` |
| Vista previa | `st.image(..., width="stretch")` |
| Botones | `st.button("🔍 Clasificar", type="primary")` y `st.button("Limpiar")` lado a lado |
| Resultado confiable | `st.success("Es **Homer Simpson**")` + `st.metric("Confianza", "92 %")` |
| Resultado poco confiable | `st.warning("No estoy seguro… podría ser **Homer Simpson** (41 %)")` |
| Top-3 | `st.progress(valor, text="Homer Simpson — 92 %")` por cada clase |
| Detalles | `st.expander("Detalles del modelo")` |
| Errores | `st.error(...)` con mensaje en español |

### 8.3 Comportamiento
- Los botones van justo bajo el selector (antes de la vista previa) para que sigan visibles con imágenes altas.
- El botón **Clasificar** está deshabilitado si no hay imagen subida.
- Al subir una imagen **nueva**, se borra el resultado anterior.
- **Limpiar** borra la imagen y el resultado (usar `st.session_state` y cambiar la `key` del uploader).
- Mientras clasifica: `st.spinner("Analizando imagen...")`.
- El modelo se carga una sola vez (`@st.cache_resource`), no en cada clic.
- Antes de la primera clasificación, la columna derecha muestra un texto guía:
  *"Sube una imagen y presiona Clasificar."*

---

## 9. Manejo de errores

| Caso | Mensaje / acción |
|---|---|
| Archivo no es imagen válida o está corrupto | `st.error("No se pudo leer la imagen. Usa JPG, PNG o WEBP.")` |
| No hay modelos en `models/` | Modo demo + `st.warning` permanente arriba |
| Carpeta de modelo incompleta o `modelo.json` inválido | Aviso en la barra lateral; el resto de los modelos sigue disponible |
| `model.pt` de tipo `torch` es un `state_dict` | `st.error` explicando que se requiere modelo completo / TorchScript |
| Modelo `ultralytics` sin el paquete instalado | `st.error` con `pip install ultralytics` |
| Error al cargar el modelo (otro) | `st.error` con el detalle; la app no se cae |
| `classes.json` no coincide con la salida del modelo | `st.error("El modelo tiene N salidas pero classes.json tiene M clases.")` |
| Falta `classes.json` | Aviso en la barra lateral (modelo omitido) |

---

## 10. Dependencias (`requirements.txt`)

```
streamlit>=1.50
torch
torchvision
ultralytics
pillow
numpy
```

## 11. Ejecución

```bash
cd DeepLearningModelos/app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Abrir `http://localhost:8501`.

Para agregar un modelo real: crear su carpeta en `app/models/<id>/` (para YOLO: `python exportar_modelo.py testN`
desde `DeepLearningModelos/`) y recargar la página.

---

## 12. Criterios de aceptación

- [ ] `streamlit run app.py` levanta la página sin errores, **sin** modelos (modo demo visible).
- [ ] Se puede subir JPG, PNG (incluido con transparencia) y WEBP; se ve la vista previa.
- [ ] Al presionar **Clasificar** se muestra "Es <personaje>", la confianza y el Top-3.
- [ ] Con confianza < umbral se muestra el aviso de incertidumbre.
- [ ] **Limpiar** deja la página en su estado inicial.
- [ ] Un archivo no válido muestra un error y la app sigue funcionando.
- [ ] Al agregar una carpeta de modelo (YOLO o PyTorch) en `models/`, aparece en el selector sin tocar el código.
- [ ] Al cambiar de modelo en el selector, se muestran sus métricas y se borra el resultado anterior.
- [ ] Interfaz completamente en español, ordenada en dos columnas.

## 13. Fuera de alcance (por ahora)

- Entrenamiento y generación del `.pt` (notebook). La app solo muestra las métricas de prueba que trae cada `modelo.json`.
- Webcam, carga por URL, múltiples imágenes a la vez.
- Despliegue en la nube / autenticación.
- Git / GitHub: no se hace commit ni push sin autorización explícita.

## 14. Pendientes por confirmar

1. ~~Clases reales del dataset~~ → confirmadas (42).
2. ~~`IMG_SIZE` y normalización finales~~ → van en el `modelo.json` de cada modelo.
3. ~~Formato definitivo del `.pt`~~ → `ultralytics` (YOLO) o `torch` (modelo completo); entrenamiento en Python 3.14.
