# Informe del modelo ResNet-50

Documento de apoyo del notebook `resnet50.ipynb`. Contiene:

- La **guía para ejecutar el notebook** (sección 2), que reutiliza el entorno de YOLO11.
- La **arquitectura y las decisiones de entrenamiento** de ResNet-50 (secciones 4 y 5).
- El **historial de pruebas** de entrenamiento (sección 7), que el notebook actualiza automáticamente al final de cada ejecución. El resto del documento se puede editar libremente.

Las decisiones sobre los datos (copia anidada, duplicados, filtro de clases y receta de la división) son las mismas de YOLO11 y están en `informe_yolo11.md`, secciones 3 a 5.

---

## 1. Entregables del modelo

| Entregable | Ubicación | ¿Está en git? |
|---|---|---|
| Notebook del modelo (todo el flujo en un solo archivo) | `resnet50.ipynb` | Sí |
| Resultados de cada prueba (métricas, gráficos, configuración) | `resultados/resnet50/testN/` | Sí, sin los pesos |
| Pesos de cada prueba (`best.pt`, `last.pt`, unos 94 MB cada uno) | `resultados/resnet50/testN/weights/` | No (se regeneran al entrenar) |
| Modelos exportados a la aplicación web (uno por prueba elegida) | `app/models/resnet50_testN/` | Sí: unos 47 MB, porque se guarda en media precisión (excepción en `.gitignore` para su `model.pt`) |
| Esta documentación | `informe_resnet50.md` | Sí |

## 2. Cómo ejecutar el notebook

### 2.1 Entorno y requisitos

- **Mismo entorno y kernel que YOLO11:** `.venv` de `DeepLearningModelos/` y el kernel **"Python 3.14 (YOLO11 .venv)"**. La instalación paso a paso está en `informe_yolo11.md`, sección 2. ResNet-50 viene en `torchvision`, así que no hay que instalar nada más.
- **Dataset dividido:** el notebook requiere `dataset/simpsons_split/`, que crea `yolo11.ipynb` (o se copia desde otro computador, ver `informe_yolo11.md` §2.3). Es exactamente la misma división.
- **Pesos preentrenados:** la primera ejecución descarga los pesos `IMAGENET1K_V2` de ResNet-50 (unos 102 MB) a `~/.cache/torch/hub/checkpoints/`. Las siguientes los reutilizan sin internet.
- **La GPU es obligatoria:** igual que en YOLO11, el notebook se detiene si no detecta CUDA.

### 2.2 Parámetros (sección 0.1 del notebook)

| Parámetro | Valor | Cuándo cambiarlo |
|---|---|---|
| `HIPERPARAMETROS["epochs"]` / `["patience"]` | 30 / 7 | Más épocas si la validación sigue mejorando al final |
| `HIPERPARAMETROS["batch"]` | 64 | Si aparece *CUDA out of memory*, poner 32 |
| `HIPERPARAMETROS["epocas_congeladas"]` | 3 | 0 para ajustar toda la red desde el inicio; más si la cabeza tarda en estabilizarse |
| `HIPERPARAMETROS["lr_cabeza"]` | 1e-3 | Bajarlo si la pérdida oscila en las primeras épocas |
| `HIPERPARAMETROS["lr_backbone"]` | 1e-4 | Bajarlo (p. ej. 5e-5) si la validación empeora al descongelar el backbone |
| `HIPERPARAMETROS["dropout"]` | 0.2 | Subirlo si hay sobreajuste (train muy por encima de val) |
| `HIPERPARAMETROS["label_smoothing"]` | 0.1 | 0 para comparar con la pérdida sin suavizado |
| `HIPERPARAMETROS["peso_clases"]` | 0 | Mayor que 0 para dar más peso a las clases con pocas imágenes si su F1 queda bajo |
| `HIPERPARAMETROS["workers"]` | 8 | 0 si fallan los procesos que cargan las imágenes (ver §2.4) |
| `PREPROCESAMIENTO["imgsz"]` | 224 | Para probar otra resolución (p. ej. 288). No se cambia en `HIPERPARAMETROS`, que solo lo copia |
| `TEST_A_EVALUAR` | `None` | `"testN"` para reevaluar una prueba sin volver a entrenar |
| `EXPORTAR_A_APP` | `False` | `True` para exportar la prueba a `app/models/resnet50_testN/` (también se puede después con el script) |

### 2.3 Ejecutar y exportar

1. Botón **Run All**. Duración estimada: unos 20 a 35 minutos en la RTX 4060 (estimación por confirmar con test1; la detención anticipada puede acortarla). Si se detiene a mano (*Interrupt*), se conserva el mejor peso hasta ese momento y las celdas siguientes lo evalúan.
2. Al terminar, todo queda en `resultados/resnet50/testN/` y la tabla de la sección 7 de este informe se actualiza sola.
3. Para llevar una prueba a la aplicación web, desde `DeepLearningModelos/` y con el entorno activado (no se vuelve a entrenar ni a evaluar):

   ```bash
   python exportar_modelo.py resnet50 testN
   ```

   Crea `app/models/resnet50_testN/` con el modelo completo en media precisión (`model.pt`, ~47 MB), `classes.json` y `modelo.json`, que incluye el preprocesamiento (recorte central y normalización de ImageNet). La aplicación lo vuelve a float32 al cargarlo.

### 2.4 Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `CUDA out of memory` | Lote demasiado grande para la GPU | `HIPERPARAMETROS["batch"] = 32` |
| Error en los *workers* del `DataLoader` (p. ej. `DataLoader worker exited unexpectedly` o un error al serializar) | Problema al crear los procesos que cargan las imágenes | `HIPERPARAMETROS["workers"] = 0` (carga en el proceso principal; más lento, pero siempre funciona) |
| Error al descargar los pesos de ImageNet | Sin internet | Conectarse en la primera ejecución; luego quedan en `~/.cache/torch/hub/checkpoints/` |
| `AssertionError` por falta de GPU o de `simpsons_split/` | Kernel equivocado o dataset sin copiar | Ver `informe_yolo11.md` §2.12 |

**Sobre los *workers*:** desde Python 3.14, el método por defecto de `multiprocessing` en Linux es `forkserver` (antes era `fork`). Con `forkserver`, cada *worker* parte en un proceso nuevo que debe importar de nuevo lo que necesita, y las funciones o clases definidas dentro del notebook (módulo `__main__`) no siempre se pueden recuperar. Por eso el notebook crea los `DataLoader` con `multiprocessing_context="fork"`, que copia el proceso del notebook tal como está.

---

## 3. Datos

Se usa **la misma división que YOLO11**: 25 clases (más de 100 imágenes por personaje), sin duplicados MD5, 70/15/15 estratificada con semilla 42.

| Subconjunto | Imágenes |
|---|---|
| Entrenamiento (`train`) | 14.128 |
| Validación (`val`) | 3.028 |
| Prueba (`test`) | 3.028 |

- Las decisiones y las tablas por clase están en `informe_yolo11.md`, secciones 3 a 5.
- **El notebook no divide:** toma la lista de imágenes del manifiesto `dataset/simpsons_split/division.csv` (con un `Dataset` propio, no `ImageFolder`) y verifica que cada imagen exista en su carpeta, que cada subconjunto tenga las 25 carpetas de clase y que no sobren archivos. Así, las métricas de prueba de ResNet-50 y de YOLO11 se calculan sobre exactamente las mismas 3.028 imágenes.
- Las clases quedan en orden alfabético (las del manifiesto, que coinciden con las carpetas), que es el orden de los índices del modelo (`clases.json`).

---

## 4. Arquitectura y decisiones

### 4.1 Familia ResNet

| Modelo | Bloque | Parámetros (M) | GMACs (224 px) | Top-1 ImageNet V1 (%) | Top-1 ImageNet V2 (%) |
|---|---|---|---|---|---|
| ResNet-18 | Básico | 11,7 | 1,8 | 69,8 | — |
| ResNet-34 | Básico | 21,8 | 3,7 | 73,3 | — |
| **ResNet-50** | **Bottleneck** | **25,6** | **4,1** | **76,1** | **80,9** |
| ResNet-101 | Bottleneck | 44,5 | 7,8 | 77,4 | 81,9 |
| ResNet-152 | Bottleneck | 60,2 | 11,5 | 78,3 | 82,3 |

*Valores de la documentación de torchvision, con la cabeza original de 1.000 clases. Con la cabeza de 25 clases, ResNet-50 queda en unos 23,6 M de parámetros.*

**Sobre los GFLOPs:** torchvision publica una columna "GFLOPS" que en realidad cuenta multiplicaciones-acumulaciones (MACs). Ultralytics, en cambio, reporta GFLOPs = 2 × MACs. Para comparar con YOLO11 en su misma convención, ResNet-50 equivale a unos **8,2 GFLOPs**.

### 4.2 Por qué ResNet-50

- Es la referencia clásica de las CNN para clasificación: muy documentada y fácil de justificar en el informe.
- Tiene pesos `IMAGENET1K_V2` (80,9 % top-1), que mejoran casi 5 puntos a los V1 sin cambiar la arquitectura. ResNet-18 y ResNet-34 no tienen pesos V2.
- Las versiones 101 y 152 cuestan el doble o el triple de cómputo por apenas 1 a 1,4 puntos más en ImageNet, una diferencia que no se justifica con 25 clases y unas 14.000 imágenes de entrenamiento.
- Cabe con holgura en la GPU de 8 GB con lotes de 64 imágenes de 224 px y precisión mixta.

### 4.3 Conexiones residuales

En una red "plana" muy profunda, agregar capas llega a empeorar el error de entrenamiento: el gradiente se debilita al atravesar muchas capas y la optimización se vuelve difícil. ResNet resuelve esto con **conexiones residuales**: cada bloque no aprende la transformación completa H(x), sino un residuo F(x), y su salida es **F(x) + x**.

- Si un bloque no aporta, basta con que F(x) ≈ 0 para que deje pasar la entrada tal cual (identidad), lo que es fácil de aprender.
- La suma con x crea un camino directo por el que el gradiente fluye hacia las primeras capas sin atenuarse, lo que facilita entrenar redes de 50 capas o más.

### 4.4 Bloque bottleneck

ResNet-50 usa el bloque **bottleneck**, con tres convoluciones:

1. **1×1** que reduce los canales (p. ej. 256 → 64).
2. **3×3** que procesa la información espacial con pocos canales (la parte más costosa queda barata).
3. **1×1** que vuelve a expandir los canales (64 → 256).

Cada convolución va seguida de BatchNorm, y de ReLU salvo la última, que se suma a la entrada antes de su ReLU. Cuando cambian los canales o la resolución, el atajo usa una convolución 1×1 para que las formas coincidan.

### 4.5 Etapas

| Etapa | Bloques | Canales de salida | Resolución de salida |
|---|---|---|---|
| `conv1` 7×7 / 2 + `maxpool` 3×3 / 2 | — | 64 | 56×56 |
| `layer1` | 3 | 256 | 56×56 |
| `layer2` | 4 | 512 | 28×28 |
| `layer3` | 6 | 1024 | 14×14 |
| `layer4` | 3 | 2048 | 7×7 |
| `avgpool` (promedio global) | — | 2048 | 1×1 |
| Cabeza: `Dropout` + `Linear(2048 → 25)` | — | 25 | — |

Las 50 capas son 3 + 4 + 6 + 3 = 16 bloques × 3 convoluciones = 48, más `conv1` y la capa `fc`.

La cabeza original (`Linear(2048 → 1000)`, las clases de ImageNet) se reemplaza por `nn.Sequential(nn.Dropout(p), nn.Linear(2048, 25))`.

### 4.6 Transfer learning con ImageNet V2

- **Lo que se descarga son los pesos, no imágenes:** los parámetros que la red aprendió en ImageNet (1,28 millones de fotos, 1.000 clases).
- Las capas bajas aprendieron detectores de bordes, colores y texturas que sirven para casi cualquier imagen. Las capas altas combinan esos rasgos en partes de objetos más específicas de ImageNet.
- Como ImageNet son fotografías y Los Simpson son caricaturas (colores planos, contornos negros, proporciones exageradas), la diferencia de dominio es grande. Por eso **se ajustan todas las capas**, no solo la cabeza.
- **V2 frente a V1:** la arquitectura es la misma. V2 se entrenó con una receta más moderna (más épocas, aumentaciones como TrivialAugment, Mixup y CutMix, *label smoothing*, EMA de pesos, entre otras), que deja mejores pesos para empezar.

### 4.7 Fine-tuning en dos etapas

1. **Épocas congeladas** (`epocas_congeladas = 3`): el backbone queda fijo y solo se entrena la cabeza nueva. Las capas BatchNorm del backbone quedan en modo evaluación, para que sus estadísticas de ImageNet no cambien mientras la cabeza todavía es aleatoria. Así se evita que los gradientes grandes de una cabeza sin entrenar dañen los pesos preentrenados.
2. **Red completa:** se descongela todo y se entrena con un **learning rate discriminativo**: la cabeza con `lr_cabeza = 1e-3` y el backbone con `lr_backbone = 1e-4` (10 veces menor), para ajustar los pesos preentrenados sin destruirlos.

### 4.8 Otras decisiones

| Decisión | Motivo |
|---|---|
| Optimizador **AdamW** | Converge rápido en fine-tuning y aplica el *weight decay* de forma desacoplada (regularización correcta con Adam). Es el mismo optimizador de YOLO11 |
| **Calentamiento + coseno** | El learning rate sube gradualmente en las primeras iteraciones (evita saltos bruscos al inicio) y luego baja siguiendo un coseno hasta el 1 % del learning rate inicial (`lrf = 0.01`), lo que estabiliza el final |
| ***Label smoothing* 0,1** | Reparte el 10 % de la probabilidad objetivo entre las demás clases: el modelo se vuelve menos sobreconfiado y generaliza mejor |
| **Dropout 0,2** en la cabeza | Regulariza la única capa que parte desde cero |
| **AMP** (precisión mixta) | Casi duplica la velocidad en la RTX 4060 y reduce la memoria, sin pérdida de exactitud |
| **Detención anticipada** (`patience = 7`) | Detiene el entrenamiento si la exactitud de validación no mejora en 7 épocas |
| **Mejor peso según val top-1** | `best.pt` es el de la época con mayor exactitud en validación (en empate, la de menor loss de validación). El conjunto de prueba no se usa para elegir |

---

## 5. Preprocesamiento

| Etapa | Entrenamiento | Validación, prueba e inferencia |
|---|---|---|
| Resolución | `RandomResizedCrop(224)`, área de 50 % a 100 % (zoom) | `Resize(224)` del lado corto + `CenterCrop(224)` |
| Giros | Horizontal con p = 0,5 | — |
| Aumentación | `RandAugment` (rotación, traslación, cizalle, brillo, contraste…) | — |
| Escala | `ToTensor`: píxeles a [0, 1] | `ToTensor`: píxeles a [0, 1] |
| Normalización | `Normalize` con media `[0.485, 0.456, 0.406]` y desviación `[0.229, 0.224, 0.225]` (ImageNet) | Igual |
| Borrado aleatorio | `RandomErasing` con p = 0,4 | — |

- **Diferencia con YOLO11:** YOLO solo escala los píxeles a [0, 1]. ResNet-50 además resta la media y divide por la desviación de ImageNet, porque sus pesos preentrenados se aprendieron con esa normalización: sin ella, las activaciones de entrada quedan fuera del rango que la red espera.
- **En la aplicación web:** el preprocesamiento de inferencia se toma del `modelo.json` del modelo exportado (`"redimension": "recorte_central"` y `"normalizacion"` con los valores de ImageNet), así que es idéntico al de validación sin configurar nada a mano.

---

## 6. Estructura de resultados

Cada ejecución del notebook crea `resultados/resnet50/testN/`, numerada automáticamente:

- `weights/`: `best.pt` y `last.pt` (modelo completo en float32; no van a git).
- `configuracion.json`: datos, preprocesamiento, hiperparámetros, entorno (versiones y GPU), fechas, duración y mejor época.
- `clases.json`: las 25 clases en el orden de los índices del modelo.
- `division.csv`: imágenes de train, val y test (ruta dentro de `dataset/simpsons_split/`, archivo original, clase, subconjunto y MD5); no incluye las excluidas ni las duplicadas del manifiesto.
- `resnet50_copia.ipynb`: copia del notebook al momento de la ejecución (no va a git).
- `entrenamiento.log`: registro de cada época.
- `historial.csv`: métricas de entrenamiento y validación por época, con el learning rate de cada grupo, la etapa (solo cabeza / red completa) y el tiempo.
- `metricas_test.json`: accuracy, top-5, precision, recall y F1 macro y ponderados.
- `reporte_clasificacion.csv`: precision, recall, F1 y soporte por clase.
- `matriz_confusion_test.csv`: matriz de confusión del conjunto de prueba.
- `confusiones_frecuentes.csv`: pares de clases que más se confunden.
- `costo_computacional.json`: parámetros, GFLOPs, tamaño del peso y latencias.
- `graficos/`: figuras `01` a `11` en PNG (`11_errores` solo si hubo errores).
- `resumen.md`: resumen de la prueba.

---

## 7. Historial de pruebas

<!-- HISTORIAL:INICIO -->
| Prueba | Fecha | imgsz | Épocas | Congeladas | lr cabeza / backbone | Lote | Dropout | Accuracy test | F1 macro | F1 ponderado | Duración (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test2 | 2026-10-15 | 224 | 50/50 | 3 | 0.001 / 0.0001 | 64 | 0.2 | 0.9941 | 0.9933 | 0.9941 | 38.1 |
<!-- HISTORIAL:FIN -->

*Épocas = ejecutadas / máximas. Congeladas = épocas entrenando solo la cabeza.*

---

## 8. Comparación con YOLO11m-cls

Las dos redes se evalúan sobre las mismas 3.028 imágenes de prueba.

| Métrica | YOLO11m-cls · test1 | ResNet-50 |
|---|---|---|
| Parámetros | 10,4 M | 23,56 M |
| GFLOPs (224 px, 2 × MACs) | 4,85 | 8,17 |
| Tamaño del peso en la web | 21 MB | 47,3 MB (media precisión; `best.pt` en float32: 94,6 MB) |
| Latencia CPU (por imagen) | 23,5 ms | completar tras test1 |
| Latencia GPU (por imagen) | 2,8 ms | completar tras test1 |
| Accuracy test | 0,9921 | completar tras test1 |
| F1 macro | 0,9907 | completar tras test1 |
| F1 ponderado | 0,9921 | completar tras test1 |

---

## 9. Notas y pendientes

- Ejecutar `test1`, completar la columna de ResNet-50 de la sección 8 con su `costo_computacional.json` y `metricas_test.json`, y exportarla con `python exportar_modelo.py resnet50 test1`.
- Revisar en `confusiones_frecuentes.csv` si ResNet-50 y YOLO11 se equivocan en los mismos pares de personajes.
- Si las clases pequeñas (Selma Bouvier, Barney Gumble, Groundskeeper Willie, Maggie Simpson) quedan con F1 bajo, probar `peso_clases` mayor que 0.
