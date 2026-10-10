# Especificación — Página web de clasificación (Streamlit)

Documento de referencia para construir la página web del proyecto. **Alcance: solo la página web.**
El entrenamiento del modelo y la generación del `.pt` quedan fuera de este documento (se harán en el notebook).

---

## 1. Objetivo

Página web lo más básica posible, con ventanas y botones ordenados, que permita:

1. **Subir** una imagen desde el computador.
2. **Procesar** la imagen y clasificarla con el modelo PyTorch (`.pt`).
3. **Mostrar el resultado** en la misma página, p. ej.: **"Es Homer Simpson (92%)"**.

Cubre el punto 6 de la pauta (Despliegue del Modelo): recibir una imagen externa cualquiera,
preprocesarla automáticamente (resolución, canales y escala) y retornar la clase predicha con su confianza.

---

## 2. Decisiones tomadas

| Tema | Decisión |
|---|---|
| Framework | **Streamlit** (todo en Python, un solo `app.py`) |
| Dataset | **The_Simpsons_Characters_Data**: 42 clases confirmadas (`dataset/The_Simpsons_Characters_Data/simpsons_dataset/`) |
| Clases | Archivo editable `classes.json` (no hardcodeadas en el código) |
| Formato del modelo | **`.pt` con modelo completo** (`torch.save(model)`; TorchScript como respaldo) |
| Sin `.pt` disponible | **Modo demo**: predicciones aleatorias + aviso visible |
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
    ├── config.py           ← parámetros (rutas, tamaño de imagen, umbral, normalización)
    ├── predictor.py        ← carga del modelo, preprocesamiento, inferencia, modo demo
    ├── classes.json        ← lista ordenada de clases (índice → nombre)
    ├── requirements.txt    ← dependencias
    ├── README.md           ← instrucciones de ejecución (entregable de la pauta)
    └── models/
        └── .gitkeep        ← aquí se copiará el modelo: models/model.pt
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

La página asume lo siguiente. **Si el entrenamiento cambia algo, se ajusta solo `config.py` / `classes.json`.**

### 5.1 Archivo
- Ruta: `app/models/model.pt` (configurable en `config.py` → `MODEL_PATH`).
- **Formato recomendado: modelo completo** con `torch.save(model, "model.pt")`.
  - Si la arquitectura es de `torchvision` (ResNet, EfficientNet, VGG, etc.), se carga sin código adicional.
  - Si es una CNN propia, la clase debe ser importable al cargar: copiar su definición a `app/`
    (p. ej. `app/arquitectura.py`) e importarla en `predictor.py`.
- Alternativa: TorchScript (`torch.jit.script(model).save("model.pt")`). La app lo soporta como respaldo,
  pero ⚠️ **PyTorch no soporta TorchScript en Python 3.14+** (versión instalada localmente), por lo que
  puede fallar. Orden de carga en la app: primero `torch.load`, luego `torch.jit.load`.

### 5.2 Entrada y salida
- Entrada: tensor `float32` de forma `[1, 3, IMG_SIZE, IMG_SIZE]`, RGB.
- Salida: logits `[1, NUM_CLASSES]` (sin softmax; la app aplica softmax).
  Si el modelo ya incluye softmax, poner `MODEL_OUTPUTS_PROBS = True` en `config.py`.

### 5.3 Clases
- `classes.json` contiene la lista de clases **en el mismo orden que los índices del modelo**.
- Con `torchvision.datasets.ImageFolder`, el orden es **alfabético por nombre de carpeta**
  (`dataset.classes`). Se recomienda generarlo desde el notebook:
  ```python
  import json
  json.dump(train_dataset.classes, open("classes.json", "w"), indent=2)
  ```
- Formato aceptado:
  ```json
  ["abraham_grampa_simpson", "agnes_skinner", "..."]
  ```
  o con nombre visible explícito:
  ```json
  [{"id": "homer_simpson", "label": "Homer Simpson"}, ...]
  ```
- La app valida que `len(classes) == NUM_CLASSES` de la salida del modelo; si no coincide, muestra un error claro.

### 5.4 Clases del dataset (confirmadas)
El dataset `simpsons_dataset/` contiene 42 carpetas (una por personaje, 41.866 imágenes JPG en total).
`classes.json` se inicializa con ellas en orden alfabético (el mismo que usa `ImageFolder`):

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
| `MODEL_PATH` | `models/model.pt` | Ruta al modelo (relativa a `app/`). |
| `CLASSES_PATH` | `classes.json` | Ruta a la lista de clases. |
| `IMG_SIZE` | `224` | Lado de la imagen cuadrada de entrada. |
| `NORMALIZE_MEAN` | `None` | Si es `None`: solo escala a [0, 1] (lo que pide la pauta). Si se usa transfer learning con pesos ImageNet: `[0.485, 0.456, 0.406]`. |
| `NORMALIZE_STD` | `None` | Ídem; ImageNet: `[0.229, 0.224, 0.225]`. |
| `MODEL_OUTPUTS_PROBS` | `False` | `True` si el modelo ya aplica softmax. |
| `CONFIDENCE_THRESHOLD` | `0.50` | Bajo este valor se muestra aviso de incertidumbre. |
| `TOP_K` | `3` | Cantidad de clases en el ranking. |
| `ALLOWED_TYPES` | `["jpg", "jpeg", "png", "webp"]` | Extensiones permitidas. |
| `DEVICE` | `"cuda"` si disponible, si no `"cpu"` | Dispositivo de inferencia. |

⚠️ **Importante:** `IMG_SIZE`, `NORMALIZE_MEAN` y `NORMALIZE_STD` deben ser **idénticos** a los usados en
validación/test del notebook (no los de *data augmentation*). Si no coinciden, las predicciones serán malas.

---

## 7. Lógica de procesamiento (`predictor.py`)

### 7.1 Carga del modelo (una sola vez, cacheada con `@st.cache_resource` desde `app.py`)
1. Si `MODEL_PATH` **no existe** → activar **modo demo**.
2. Si existe:
   1. Intentar `torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)`.
   2. Si falla, intentar `torch.jit.load(MODEL_PATH, map_location=DEVICE)`.
   3. Si el resultado es un `dict` (es un `state_dict`, no un modelo completo) → error claro:
      *"El archivo contiene solo pesos (state_dict). Exporte el modelo completo o en TorchScript."*
   4. Llamar `model.eval()`.
3. Cargar `classes.json` y validar contra la salida del modelo (pasada de prueba con un tensor de ceros).

### 7.2 Preprocesamiento de la imagen subida
1. Abrir con Pillow: `Image.open(archivo)`.
2. Corregir orientación EXIF (`ImageOps.exif_transpose`) — fotos de celular.
3. Convertir a **RGB** (`.convert("RGB")`): resuelve PNG con transparencia (RGBA), escala de grises (L) y paletas (P).
4. **Redimensionar** a `IMG_SIZE × IMG_SIZE`.
5. Convertir a tensor y **escalar a [0, 1]** (`transforms.ToTensor()`).
6. Si hay `NORMALIZE_MEAN/STD` → aplicar `transforms.Normalize`.
7. Agregar dimensión de lote: `[3, H, W] → [1, 3, H, W]`.

### 7.3 Inferencia
1. `with torch.inference_mode(): logits = model(x.to(DEVICE))`.
2. `probs = softmax(logits, dim=1)` (salvo `MODEL_OUTPUTS_PROBS = True`).
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
- Se activa automáticamente si no hay `model.pt`.
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
│  [⚠️ Modo demo: no se encontró models/model.pt]  (si aplica)│
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
│   tamaño de entrada, dispositivo, tiempo de inferencia     │
└────────────────────────────────────────────────────────────┘
```

### 8.2 Componentes
| Elemento | Componente Streamlit |
|---|---|
| Configuración de página | `st.set_page_config(page_title=..., page_icon="🍩", layout="wide")` |
| Título / subtítulo | `st.title`, `st.caption` |
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
| No existe `model.pt` | Modo demo + `st.warning` permanente arriba |
| `model.pt` es un `state_dict` | `st.error` explicando que se requiere modelo completo / TorchScript |
| Error al cargar el modelo (otro) | `st.error` con el detalle; la app no se cae |
| `classes.json` no coincide con la salida del modelo | `st.error("El modelo tiene N salidas pero classes.json tiene M clases.")` |
| Falta `classes.json` | `st.error` indicando la ruta esperada |

---

## 10. Dependencias (`requirements.txt`)

```
streamlit>=1.50
torch
torchvision
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

Para usar el modelo real: copiar el `.pt` a `app/models/model.pt`, reemplazar `classes.json`
por el generado en el notebook, revisar `IMG_SIZE`/normalización en `config.py` y reiniciar la app.

---

## 12. Criterios de aceptación

- [ ] `streamlit run app.py` levanta la página sin errores, **sin** `model.pt` (modo demo visible).
- [ ] Se puede subir JPG, PNG (incluido con transparencia) y WEBP; se ve la vista previa.
- [ ] Al presionar **Clasificar** se muestra "Es <personaje>", la confianza y el Top-3.
- [ ] Con confianza < umbral se muestra el aviso de incertidumbre.
- [ ] **Limpiar** deja la página en su estado inicial.
- [ ] Un archivo no válido muestra un error y la app sigue funcionando.
- [ ] Al poner un `model.pt` real (TorchScript o modelo completo), la app lo usa sin tocar el código.
- [ ] Interfaz completamente en español, ordenada en dos columnas.

## 13. Fuera de alcance (por ahora)

- Entrenamiento, comparación de modelos y generación del `.pt` (notebook).
- Webcam, carga por URL, múltiples imágenes a la vez.
- Despliegue en la nube / autenticación.
- Git / GitHub: no se hace commit ni push sin autorización explícita.

## 14. Pendientes por confirmar

1. ~~Clases reales del dataset~~ → confirmadas (42).
2. `IMG_SIZE` y normalización finales usados en el entrenamiento → actualizar `config.py`.
3. Formato definitivo del `.pt` (modelo completo recomendado) y versión de Python donde se entrenará.
