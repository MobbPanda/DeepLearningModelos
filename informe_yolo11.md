# Informe del modelo YOLO11m-cls

Documento de apoyo del notebook `yolo11.ipynb`. Contiene:

- La **guía paso a paso para ejecutar el notebook en otro computador** (sección 2).
- Las decisiones sobre los datos y la **receta de la división**, para que el resto del grupo use exactamente la misma (secciones 3 a 5).
- El **historial de pruebas** de entrenamiento (sección 8), que el notebook actualiza automáticamente al final de cada ejecución. El resto del documento se puede editar libremente.

---

## 1. Entregables del modelo

| Entregable | Ubicación | ¿Está en git? |
|---|---|---|
| Notebook del modelo (todo el flujo en un solo archivo) | `yolo11.ipynb` | Sí |
| Dataset original y dataset dividido 70/15/15 (25 clases) | `dataset/simpsons_dataset/` y `dataset/simpsons_split/` | No (1,2 GB y unos 550 MB): se copian aparte, ver paso 2 |
| Resultados de cada prueba (métricas, gráficos, configuración) | `resultados/yolo11/testN/` | Sí, sin los pesos |
| Pesos de cada prueba (`best.pt`, `last.pt`) | `resultados/yolo11/testN/weights/` | No (se regeneran al entrenar) |
| Modelos exportados a la aplicación web (uno por prueba elegida) | `app/models/yolo11m-cls_testN/` | Sí (excepción en `.gitignore` para su `model.pt`) |
| Script de exportación a la web | `exportar_modelo.py` | Sí |
| Esta documentación | `informe_yolo11.md` | Sí |

## 2. Cómo ejecutar el notebook en otro computador

Incluye los comandos para Linux y Windows. Todos se ejecutan desde la carpeta `DeepLearningModelos/`.

El flujo del notebook (dataset dividido, entrenamiento, evaluación y exportación) se probó en Linux con Python 3.14, tanto con el dataset original como solo con `simpsons_split/`. Los comandos de instalación de PyTorch siguen el índice oficial de PyTorch (versiones verificadas el 14 de octubre de 2026).

### 2.1 Requisitos

| Componente | Requisito |
|---|---|
| Sistema operativo | Linux o Windows 10/11 de 64 bits |
| Python | 3.10 a 3.14 (probado con 3.14) |
| GPU | NVIDIA con al menos 4 GB de VRAM y driver actualizado (probado en NVIDIA T400 de 4 GB; recomendado RTX 4060 de 8 GB). **El notebook exige GPU con CUDA**: sin ella se detiene en la sección 0.2 |
| Editor | VS Code con las extensiones **Python** (`ms-python.python`) y **Jupyter** (`ms-toolsai.jupyter`) |
| Otros | Git e internet en la primera ejecución (se descarga `yolo11m-cls.pt`, unos 21 MB) |
| Espacio en disco | ~6 GB para el entorno (PyTorch con CUDA), 0,6 GB para el dataset dividido y ~50 MB por prueba |

### 2.2 Paso 1: obtener el código

```bash
git clone https://github.com/MobbPanda/DeepLearningModelos.git
cd DeepLearningModelos
git checkout <rama>   # la rama que contiene yolo11.ipynb (o main, una vez integrada)
```

### 2.3 Paso 2: copiar el dataset dividido

El dataset no está en git. Hay que copiar la carpeta `dataset/simpsons_split/` (unos 550 MB) desde el computador donde se creó. Por ejemplo, se puede comprimir y pasar por un pendrive o Google Drive:

```bash
# En el computador de origen (Linux), desde DeepLearningModelos/
zip -rq simpsons_split.zip dataset/simpsons_split
# En el computador de destino, desde DeepLearningModelos/
unzip -q simpsons_split.zip
```

En Windows: clic derecho sobre la carpeta → *Comprimir en archivo ZIP* y luego *Extraer todo* en el destino.

Al terminar, la estructura debe quedar así:

```
DeepLearningModelos/
├── yolo11.ipynb
├── informe_yolo11.md
└── dataset/
    └── simpsons_split/
        ├── division.csv
        ├── train/   (25 carpetas)
        ├── val/     (25 carpetas)
        └── test/    (25 carpetas)
```

- **El dataset original no es necesario** para entrenar: el notebook usa el manifiesto `division.csv` para la exploración inicial.
- Solo hay que copiar también el original (`dataset/simpsons_dataset/` o `dataset/The_Simpsons_Characters_Data/simpsons_dataset/`, 1,2 GB) si se quiere **cambiar la división**, por ejemplo con otro umbral de clases.
- Si no se tiene `simpsons_split/`, otra opción es descargar el dataset desde Kaggle y dejarlo en `dataset/simpsons_dataset/`. El notebook creará la división con la misma receta.

### 2.4 Paso 3: crear el entorno virtual

**Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Windows (PowerShell):**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Si PowerShell bloquea la activación, ejecutar una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y volver a intentar.

### 2.5 Paso 4: instalar PyTorch con soporte para GPU

**Importante:** PyTorch se instala **antes** que Ultralytics. En el orden inverso, en Windows pip instala la versión de PyTorch que solo usa CPU.

1. Ejecutar `nvidia-smi` y mirar el valor **CUDA Version** (arriba a la derecha). Indica la versión máxima de CUDA que soporta el driver.
2. Instalar según ese valor:

| CUDA Version en `nvidia-smi` | Comando |
|---|---|
| 13.0 o superior | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130` |
| 12.6 a 12.9 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` |
| Menor a 12.6 | Actualizar el driver de NVIDIA y volver al punto 1 |
| Sin GPU NVIDIA | No sirve para entrenar (el notebook exige GPU). Solo para la aplicación web: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |

Ambas variantes (cu126 y cu130) instalan PyTorch 2.14 y sirven para las RTX 4060 y la T400.

### 2.6 Paso 5: instalar el resto de las dependencias

```bash
pip install ultralytics scikit-learn seaborn pandas ipykernel
pip install streamlit   # solo si también se va a ejecutar la aplicación web
```

### 2.7 Paso 6: verificar la instalación

```bash
python -c "import torch, ultralytics; print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO DETECTADA'); print('ultralytics', ultralytics.__version__)"
```

Resultado esperado, por ejemplo: `torch 2.14.1+cu130 | GPU: NVIDIA GeForce RTX 4060`. Si dice `NO DETECTADA` teniendo GPU, revisar la sección 2.12.

### 2.8 Paso 7: configurar VS Code

1. Abrir la carpeta `DeepLearningModelos` (*File → Open Folder*).
2. Instalar las extensiones **Python** y **Jupyter** si no están.
3. Abrir `yolo11.ipynb`.
4. Elegir el kernel: botón **Select Kernel** (arriba a la derecha) → *Python Environments* → `.venv` (o *Jupyter Kernel* → **Python 3.14 (YOLO11 .venv)**, si se registró con `python -m ipykernel install --user --name yolo11-simpsons --display-name "Python 3.14 (YOLO11 .venv)"`).
   - Si no aparece: `Ctrl+Shift+P` → *Python: Select Interpreter* → *Enter interpreter path*.
   - La ruta es `.venv/bin/python` en Linux o `.venv\Scripts\python.exe` en Windows. Luego `Ctrl+Shift+P` → *Developer: Reload Window*.
5. Verificar que el ajuste **Jupyter: Notebook File Root** tenga su valor por defecto, `${fileDirname}`. El notebook usa como base la carpeta donde está guardado.

### 2.9 Paso 8: revisar los parámetros (sección 0.1 del notebook)

| Parámetro | Valor | Cuándo cambiarlo |
|---|---|---|
| `HIPERPARAMETROS["epochs"]` / `["patience"]` | 50 / 10 | Más épocas si la validación sigue mejorando al final |
| `HIPERPARAMETROS["batch"]` | -1 (AutoBatch) | Si aparece *CUDA out of memory*, poner 32 o 16 |
| `HIPERPARAMETROS["workers"]` | 8 | Bajar a 2 o 4 si el equipo tiene poca RAM o pocos núcleos |
| `TEST_A_EVALUAR` | `None` | `"testN"` para reevaluar una prueba sin volver a entrenar |
| `EXPORTAR_A_APP` | `False` | `True` para exportar la prueba a `app/models/yolo11m-cls_testN/` (también se puede después con `python exportar_modelo.py testN`) |
| `DATOS["min_imagenes_por_clase"]` | 100 | Cambiarlo regenera la división (requiere el dataset original) |

La primera celda (`# %pip install ...`) no es necesaria si se siguieron los pasos 4 y 5.

### 2.10 Paso 9: ejecutar

1. Botón **Run All**. El progreso del entrenamiento se ve en la salida de la sección 5.
2. Duración aproximada para 50 épocas: del orden de 1 hora en una RTX 4060 y de 1,5 a 3 horas en una T400. La detención anticipada puede acortarla.
3. Mientras entrena, no cerrar VS Code ni dejar que el equipo se suspenda.
4. Al terminar, todo queda en `resultados/yolo11/testN/`: pesos, métricas, gráficos y `resumen.md`. La tabla de la sección 8 de este informe se actualiza sola.

### 2.11 Paso 10: exportar el modelo a la aplicación web y guardar en git

La aplicación web admite varios modelos a la vez (un selector en la barra lateral, con las métricas de prueba de cada uno). Cada prueba exportada queda en su propia carpeta `app/models/yolo11m-cls_testN/` con `model.pt`, `classes.json` y `modelo.json`.

1. Elegir la prueba según el historial (sección 8).
2. Exportarla, desde `DeepLearningModelos/` y con el entorno activado (no se vuelve a entrenar ni a evaluar):

   ```bash
   python exportar_modelo.py testN
   ```

   Alternativa: en la sección 0.1 poner `TEST_A_EVALUAR = "testN"` y `EXPORTAR_A_APP = True`, y *Run All* (reevalúa sin entrenar y exporta).
3. Probar la aplicación: `cd app` y luego `streamlit run app.py`. Elegir el modelo en la barra lateral.
4. Guardar en git:

   ```bash
   git add yolo11.ipynb informe_yolo11.md exportar_modelo.py resultados/ app/
   git commit -m "Entrenamiento YOLO11m-cls: testN"
   git push
   ```

   El `.gitignore` excluye automáticamente el dataset, los entornos virtuales y los pesos intermedios; los `model.pt` de `app/models/*/` sí se versionan (unos 20 MB cada uno).

### 2.12 Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| La verificación dice `GPU: NO DETECTADA` | Se instaló PyTorch sin CUDA, o el driver es antiguo | `pip uninstall -y torch torchvision`, revisar `nvidia-smi` y repetir el paso 4 |
| `AssertionError: No se detectó una GPU con CUDA` (sección 0.2) | El notebook entrena solo en GPU y el kernel no la ve | Elegir el kernel de `.venv` y revisar la fila anterior |
| `CUDA out of memory` | Lote demasiado grande para la GPU | `HIPERPARAMETROS["batch"] = 16` (o 8) |
| El kernel `.venv` no aparece en VS Code | Falta `ipykernel` en el entorno o VS Code no lo detectó | `pip install ipykernel` con el entorno activado y *Developer: Reload Window* |
| `AssertionError: No se encontró el dataset original ... ni el dataset dividido` | La carpeta no está en `DeepLearningModelos/dataset/simpsons_split/` | Revisar la estructura del paso 2 |
| `RuntimeError: La división calculada no coincide ...` | Se cambió `DATOS` (umbral, proporciones o semilla) sin el dataset original | Restaurar los valores de `DATOS` o copiar también el dataset original |
| Error al descargar `yolo11m-cls.pt` | Sin internet | Conectarse, o descargar el archivo desde https://github.com/ultralytics/assets/releases y dejarlo en `DeepLearningModelos/` |
| `Bus error` / `Error del bus` al importar `torch` | Entorno virtual dañado (por ejemplo, por un apagado durante la instalación) | Borrar `.venv` y repetir los pasos 3 a 5 |
| PowerShell no permite activar el entorno | Política de ejecución de scripts | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

---

## 3. Dataset

- **Nombre:** The Simpsons Characters Data (original en `dataset/simpsons_dataset/`; fuente pública: Kaggle, *alexattia/the-simpsons-characters-dataset*).
- **Tarea:** clasificación multiclase de etiqueta única (un personaje por imagen).
- **Formato:** una carpeta por personaje, imágenes JPG en RGB de tamaños variables.

### 3.1 Copia anidada (41.866 → 20.933 imágenes)

El directorio contiene **41.866 archivos**, pero la subcarpeta `simpsons_dataset/simpsons_dataset/` es una **copia exacta** de las 42 carpetas de personajes.

| Ubicación | Imágenes |
|---|---|
| Carpetas de personajes (`simpsons_dataset/<personaje>/`) | 20.933 |
| Copia anidada (`simpsons_dataset/simpsons_dataset/<personaje>/`) | 20.933 |
| **Total de archivos** | **41.866** |

- **Verificación:** se comparó la huella MD5 (contenido byte a byte) de cada archivo. Las 20.933 imágenes de la copia coinciden con las originales.
- **Decisión:** se excluye la copia anidada.
- **Motivo:** usar ambas duplicaría cada imagen. Si una copia cae en entrenamiento y su gemela en prueba, el modelo "ya conoce" la imagen evaluada, una fuga de información (*data leakage*) que infla artificialmente las métricas.

### 3.2 Imágenes repetidas con distinto nombre

Dentro de las 20.933 imágenes hay **6 pares de archivos idénticos**, siempre del mismo personaje:

| Personaje | Archivos idénticos |
|---|---|
| Waylon Smithers | `pic_0038.jpg` = `pic_0051.jpg` |
| Professor John Frink | `pic_0012.jpg` = `pic_0014.jpg` |
| Lenny Leonard | `pic_0257.jpg` = `pic_0262.jpg` |
| Mayor Quimby | `pic_0054.jpg` = `pic_0176.jpg` |
| Cletus Spuckler | `pic_0010.jpg` = `pic_0012.jpg` |
| Rainier Wolfcastle | `pic_0011.jpg` = `pic_0016.jpg` |

**Decisión:** se conserva una sola copia de cada par antes de dividir (por el mismo riesgo de fuga de información). Quedan **20.927 imágenes únicas**.

---

## 4. Filtro de clases: más de 100 imágenes por personaje

**Decisión:** se entrenan solo los personajes con **más de 100 imágenes** (`min_imagenes_por_clase = 100`; como ningún personaje tiene exactamente 100, "≥ 100" y "> 100" seleccionan las mismas clases). Quedan **25 de las 42 clases (20.184 imágenes)**. Se excluyen 17 personajes con 743 imágenes en total (3,6 % de las imágenes únicas). Caso límite: entra Selma Bouvier (103) y queda fuera Carl Carlson (98).

**Justificación:**

1. **Evaluación confiable.** Con la división 70/15/15, la clase más pequeña (Selma Bouvier) deja 16 imágenes en validación y 15 en prueba. Con umbrales menores, varias clases quedarían con muy pocas imágenes de prueba (Lionel Hutz tiene 3 imágenes en total). Con tan pocos ejemplos, la *precision*, el *recall*, el F1 y la fila de la matriz de confusión de esa clase dependen de 1 o 2 imágenes, y el promedio macro se vuelve inestable.
2. **Aprendizaje suficiente.** Cada clase conserva al menos 72 imágenes de entrenamiento, lo que, junto con la aumentación de datos, permite aprender rasgos generalizables. Las clases excluidas tendrían entre 2 y 69.
3. **Cumplimiento de la pauta.** La evaluación exige al menos 10 clases; 25 clases supera ampliamente ese mínimo.
4. **Impacto acotado.** Se descarta solo el 3,6 % de las imágenes.

**Limitación que hay que declarar en la presentación:** el modelo no reconoce a los 17 personajes excluidos. Si recibe una imagen de alguno de ellos, la asignará a uno de los 25 conocidos (idealmente con baja confianza, lo que la aplicación web advierte con su umbral de 50 %).

### 4.1 Clases excluidas

| Personaje | Imágenes únicas |
|---|---|
| Carl Carlson | 98 |
| Ralph Wiggum | 89 |
| Patty Bouvier | 72 |
| Martin Prince | 71 |
| Professor John Frink | 64 |
| Snake Jailbird | 55 |
| Cletus Spuckler | 46 |
| Rainier Wolfcastle | 44 |
| Agnes Skinner | 42 |
| Sideshow Mel | 40 |
| Otto Mann | 32 |
| Fat Tony | 27 |
| Gil | 27 |
| Miss Hoover | 17 |
| Disco Stu | 8 |
| Troy McClure | 8 |
| Lionel Hutz | 3 |
| **Total** | **743** |

### 4.2 Desbalance que persiste

Aun con el filtro, el desbalance es alto: entre **103 imágenes** (Selma Bouvier) y **2.246** (Homer Simpson), una razón de unas 22 veces. Por eso:

- La división es **estratificada**.
- Se reportan métricas **macro** (todas las clases pesan igual) además de las ponderadas.
- Se analiza la relación entre el F1 de cada clase y su número de imágenes.
- `cls_pw` (ponderación por clase de Ultralytics) queda disponible como hiperparámetro para probarlo en otras pruebas.

---

## 5. Dataset dividido y receta de la división

Esta rama usa su propia división; no necesita coincidir con la de otras ramas del grupo. La receta queda documentada para poder reproducirla.

### 5.1 Dataset ya dividido: `dataset/simpsons_split/`

La división está en `dataset/simpsons_split/`, con la estructura que exige Ultralytics (y que también lee `torchvision.datasets.ImageFolder`):

```
dataset/
├── simpsons_dataset/              ← original: 42 carpetas + copia anidada (excluida), 1,2 GB
└── simpsons_split/
    ├── division.csv               ← manifiesto: las 20.933 imágenes originales y su destino
    ├── train/<personaje>/*.jpg    ← 14.128 imágenes (25 carpetas)
    ├── val/<personaje>/*.jpg      ←  3.028 imágenes (25 carpetas)
    └── test/<personaje>/*.jpg     ←  3.028 imágenes (25 carpetas)
```

- Son **copias** de las imágenes originales (unos 550 MB); el dataset original no se modificó.
- Ninguna de las dos carpetas se sube a git (`.gitignore`: `*dataset/`).
- **Verificación realizada (15 de octubre de 2026):**
  - Las 20.184 copias son idénticas byte a byte a sus originales (MD5).
  - Ninguna imagen está repetida ni aparece en más de un subconjunto.
  - Las 25 clases están en los tres subconjuntos, y la proporción de *train* por clase va de 69,8 % a 70,3 %.
  - La función `division_vigente()` del notebook devuelve `True`: el notebook reutiliza la carpeta sin regenerarla.
- **Manifiesto `division.csv`.** Columnas: `archivo` (ruta original, p. ej. `homer_simpson/pic_0001.jpg`), `clase`, `md5`, `subconjunto` y `ruta_division` (ruta dentro de `simpsons_split/`).
  - Valores de `subconjunto`: `train` (14.128), `val` (3.028), `test` (3.028), `excluida` (743, clases con 100 imágenes o menos) y `duplicada` (6, copias repetidas eliminadas).

### 5.2 Receta (cómo se generó)

La división la crea el propio notebook (sección 2) con este procedimiento:

1. Listar las imágenes de las 42 carpetas de personajes, excluyendo `simpsons_dataset/simpsons_dataset/`, ordenadas por clase y nombre de archivo.
2. Eliminar duplicados por MD5 conservando la primera aparición (`drop_duplicates("md5", keep="first")`).
3. Conservar las clases con ≥ 100 imágenes (equivale a más de 100).
4. `train_test_split(df, test_size=0.30, stratify=clase, random_state=42)` → `train` y `temp`.
5. `train_test_split(temp, test_size=0.50, stratify=clase, random_state=42)` → `val` y `test`.

En cada ejecución, el notebook vuelve a calcular la división y la compara con el manifiesto:

- **Si coincide**, reutiliza la carpeta.
- **Si no coincide** (por ejemplo, porque se cambió el umbral de clases en `DATOS`), la regenera desde el dataset original.

| Subconjunto | Imágenes | Proporción |
|---|---|---|
| Entrenamiento (`train`) | 14.128 | 70 % |
| Validación (`val`) | 3.028 | 15 % |
| Prueba (`test`) | 3.028 | 15 % |
| **Total** | **20.184** | 100 % |

La clase más pequeña (Selma Bouvier) queda con 72 / 16 / 15 imágenes.

**Uso de cada subconjunto:**

- **Train:** ajustar los pesos.
- **Val:** elegir el mejor peso y aplicar la detención anticipada.
- **Test:** usar **una sola vez al final** para medir la generalización. No se usa para tomar decisiones.

### 5.3 Imágenes por clase

| Personaje | Total | Train | Val | Test |
|---|---|---|---|---|
| Homer Simpson | 2.246 | 1.572 | 337 | 337 |
| Ned Flanders | 1.454 | 1.018 | 218 | 218 |
| Moe Szyslak | 1.452 | 1.016 | 218 | 218 |
| Lisa Simpson | 1.354 | 948 | 203 | 203 |
| Bart Simpson | 1.342 | 939 | 202 | 201 |
| Marge Simpson | 1.291 | 904 | 193 | 194 |
| Krusty The Clown | 1.206 | 844 | 181 | 181 |
| Principal Skinner | 1.194 | 836 | 179 | 179 |
| Charles Montgomery Burns | 1.193 | 835 | 179 | 179 |
| Milhouse Van Houten | 1.079 | 755 | 162 | 162 |
| Chief Wiggum | 986 | 690 | 148 | 148 |
| Abraham Grampa Simpson | 913 | 639 | 137 | 137 |
| Sideshow Bob | 877 | 614 | 131 | 132 |
| Apu Nahasapeemapetilon | 623 | 436 | 94 | 93 |
| Kent Brockman | 498 | 349 | 74 | 75 |
| Comic Book Guy | 469 | 328 | 70 | 71 |
| Edna Krabappel | 457 | 320 | 68 | 69 |
| Nelson Muntz | 358 | 251 | 54 | 53 |
| Lenny Leonard | 309 | 216 | 47 | 46 |
| Mayor Quimby | 245 | 171 | 37 | 37 |
| Waylon Smithers | 180 | 126 | 27 | 27 |
| Maggie Simpson | 128 | 90 | 19 | 19 |
| Groundskeeper Willie | 121 | 85 | 18 | 18 |
| Barney Gumble | 106 | 74 | 16 | 16 |
| Selma Bouvier | 103 | 72 | 16 | 15 |
| **Total** | **20.184** | **14.128** | **3.028** | **3.028** |

---

## 6. Preprocesamiento

| Etapa | Entrenamiento | Validación, prueba e inferencia |
|---|---|---|
| Resolución | `RandomResizedCrop` a 224×224 (área de 50 % a 100 %: zoom) | Lado corto a 224 px + recorte central de 224×224 |
| Giros | Horizontal con p = 0,5 | — |
| Aumentación | RandAugment (rotación, traslación, cizalle, brillo, contraste…) y borrado aleatorio p = 0,4 | — |
| Normalización | Píxeles escalados a [0, 1] | Píxeles escalados a [0, 1] |

**Importante para la aplicación web:** el preprocesamiento de inferencia debe ser idéntico al de validación (lado corto + recorte central), no un estiramiento directo a 224×224.

---

## 7. Estructura de resultados

Cada ejecución del notebook crea `resultados/yolo11/testN/` con los pesos, la configuración, las métricas y los gráficos de esa prueba. El detalle de los archivos está al inicio de `yolo11.ipynb`.

En git se versiona todo excepto `weights/` (pesos) y `yolo11_copia.ipynb` (copia del notebook), que son duplicados o se regeneran.

---

## 8. Historial de pruebas

<!-- HISTORIAL:INICIO -->
| Prueba | Fecha | imgsz | Épocas | Optimizador y lr0 | Lote | Dropout | Accuracy test | F1 macro | F1 ponderado | Duración (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| test1 | 2026-10-15 | 224 | 47/50 | AdamW 0.0005 | -1 | 0.2 | 0.9921 | 0.9907 | 0.9921 | 20.1 |
<!-- HISTORIAL:FIN -->

*Épocas = ejecutadas / máximas. Lote −1 = AutoBatch.*

---

## 9. Notas y decisiones pendientes

- Elegir las pruebas que se llevan a la aplicación web y exportarlas (`python exportar_modelo.py testN`).
- Comparar con los modelos del resto del grupo usando la misma división y las mismas métricas (accuracy, F1 macro y ponderado, latencia y tamaño del peso).
- **Comparación con otras ramas.** Esta rama tiene su propia división (25 clases con más de 100 imágenes y sin duplicados MD5). El notebook de ConvNeXt (`modelo-convnext/`, rama `main`) usa el mismo umbral, pero no elimina duplicados, así que sus conjuntos de prueba no son idénticos: si se comparan métricas entre ramas, hay que tenerlo en cuenta.
