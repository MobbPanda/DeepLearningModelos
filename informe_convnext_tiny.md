# Informe del modelo ConvNeXt-Tiny

Documento de apoyo del notebook `convnextTiny.ipynb`. Contiene:

- La **guía para ejecutar el notebook** (sección 2), que reutiliza el entorno de YOLO11 y ResNet-50.
- La **arquitectura y las decisiones de entrenamiento** de ConvNeXt-Tiny (secciones 4 y 5).
- El **origen del modelo**: retoma la idea del notebook de Vania Vargas (`modelo-convnext/modelo_convnext.ipynb`), con su comparación entre *feature extraction* y *fine-tuning* (sección 4.9).
- El **historial de pruebas** de entrenamiento (sección 7), que el notebook actualiza automáticamente al final de cada ejecución. El resto del documento se puede editar libremente.

El notebook usa la misma plantilla que `resnet50.ipynb`. Las decisiones sobre los datos (copia anidada, duplicados, filtro de clases y receta de la división) son las mismas de YOLO11 y están en `informe_yolo11.md`, secciones 3 a 5.

---

## 1. Entregables del modelo

| Entregable | Ubicación | ¿Está en git? |
|---|---|---|
| Notebook del modelo (todo el flujo en un solo archivo) | `convnextTiny.ipynb` | Sí |
| Resultados de cada prueba (métricas, gráficos, configuración) | `resultados/convnext_tiny/testN/` | Sí, sin los pesos |
| Pesos de cada prueba (`best.pt`, `last.pt`, unos 111 MB cada uno) | `resultados/convnext_tiny/testN/weights/` | No (se regeneran al entrenar) |
| Modelos exportados a la aplicación web (uno por prueba elegida) | `app/models/convnext_tiny_testN/` | Sí: unos 56 MB, porque se guarda en media precisión (excepción en `.gitignore` para su `model.pt`) |
| Notebook original de Vania Vargas (referencia; no se modifica) | `modelo-convnext/modelo_convnext.ipynb` (copia idéntica en `modelo_convnext.ipynb`) | Sí |
| Esta documentación | `informe_convnext_tiny.md` | Sí |

## 2. Cómo ejecutar el notebook

### 2.1 Entorno y requisitos

- **Mismo entorno y kernel que YOLO11 y ResNet-50:** `.venv` de `DeepLearningModelos/` y el kernel **"Python 3.14 (YOLO11 .venv)"**. La instalación paso a paso está en `informe_yolo11.md`, sección 2. ConvNeXt viene en `torchvision`, así que no hay que instalar nada más (el notebook no usa `torchmetrics` ni celdas `!pip`).
- **Dataset dividido:** el notebook requiere `dataset/simpsons_split/`, que crea `yolo11.ipynb` (o se copia desde otro computador, ver `informe_yolo11.md` §2.3). Es exactamente la misma división.
- **Pesos preentrenados:** la primera ejecución descarga los pesos `IMAGENET1K_V1` de ConvNeXt-Tiny (unos 114 MB) a `~/.cache/torch/hub/checkpoints/`. Las siguientes los reutilizan sin internet. Para ConvNeXt, torchvision no tiene pesos V2.
- **La GPU es obligatoria:** igual que en YOLO11 y ResNet-50, el notebook se detiene si no detecta CUDA.

### 2.2 Parámetros (sección 0.1 del notebook)

| Parámetro | Valor | Variante del notebook original | Cuándo cambiarlo |
|---|---|---|---|
| `HIPERPARAMETROS["epochs"]` / `["patience"]` | 50 / 10 | 5 congeladas + 3 de fine-tuning, sin detención anticipada | Más épocas si la validación sigue mejorando al final |
| `HIPERPARAMETROS["batch"]` | 64 | 32 | Si aparece *CUDA out of memory*, poner 32 |
| `HIPERPARAMETROS["epocas_congeladas"]` | 3 | `= epochs` (solo la cabeza, *feature extraction*) | Igual a `epochs` para reproducir la variante congelada del original (ver §4.9); 0 para ajustar toda la red desde el inicio |
| `HIPERPARAMETROS["optimizer"]` | `"AdamW"` | AdamW | — |
| `HIPERPARAMETROS["lr_cabeza"]` | 1e-3 | 1e-3 | Bajarlo si la pérdida oscila en las primeras épocas |
| `HIPERPARAMETROS["lr_backbone"]` | 5e-5 | 1e-5 (toda la red con el mismo valor) | Bajarlo (p. ej. 2e-5) si la validación empeora al descongelar el backbone; subirlo (1e-4) si aprende muy lento |
| `HIPERPARAMETROS["lrf"]` / `["warmup_epochs"]` | 0.01 / 1 | Sin programación del learning rate | — |
| `HIPERPARAMETROS["weight_decay"]` | 0.05 | 0.01 (valor por defecto de AdamW) | — |
| `HIPERPARAMETROS["dropout"]` | 0.2 | Sin dropout | Subirlo si hay sobreajuste (train muy por encima de val) |
| `HIPERPARAMETROS["stochastic_depth"]` | 0.1 | 0.5 (valor por defecto de torchvision para ConvNeXt-Base; el de Tiny es 0.1) | Subirlo (0,2) si hay sobreajuste; 0 para desactivarlo |
| `HIPERPARAMETROS["label_smoothing"]` | 0.1 | Sin suavizado | 0 para comparar con la pérdida sin suavizado |
| `HIPERPARAMETROS["peso_clases"]` | 0.0 | Sin pesos por clase | Mayor que 0 para dar más peso a las clases con pocas imágenes si su F1 queda bajo |
| `HIPERPARAMETROS["amp"]` | `True` | Sin precisión mixta | `False` solo para descartar problemas numéricos |
| `HIPERPARAMETROS["fliplr"]` | 0.5 | 0.5 | — |
| `HIPERPARAMETROS["scale"]` | 0.5 (recorte de 50 % a 100 % del área) | 0.4 (recorte de 60 % a 100 %) | Menor para recortes menos agresivos |
| `HIPERPARAMETROS["rotacion"]` | 0 | 10 (grados) | Parámetro nuevo: 10 para la variante del original |
| `HIPERPARAMETROS["auto_augment"]` | `"randaugment"` | `None` (sin cambios de color) | `None` para la variante sin color del original (ver §5) |
| `HIPERPARAMETROS["erasing"]` | 0.4 | Sin borrado aleatorio | — |
| `HIPERPARAMETROS["workers"]` | 8 | 2 | 0 si fallan los procesos que cargan las imágenes (ver §2.4) |
| `HIPERPARAMETROS["seed"]` / `["deterministic"]` | 42 / `True` | 42 | — |
| `PREPROCESAMIENTO["imgsz"]` | 224 | 224 | Para probar otra resolución (p. ej. 288). No se cambia en `HIPERPARAMETROS`, que solo lo copia |
| `TEST_A_EVALUAR` | `None` | — | `"testN"` para reevaluar una prueba sin volver a entrenar |
| `EXPORTAR_A_APP` | `False` | — | `True` para exportar la prueba a `app/models/convnext_tiny_testN/` (también se puede después con el script) |

**Variante del notebook original completa** (aumentación sin cambios de color): `auto_augment = None`, `rotacion = 10`, `scale = 0.4` y `erasing = 0`. Por defecto se usa la misma aumentación que YOLO11 y ResNet-50, para que la comparación entre los tres modelos sea justa.

### 2.3 Ejecutar y exportar

1. Botón **Run All**. Duración estimada: en torno a 1 hora para 50 épocas en la RTX 4060 (estimación por confirmar con test1; la detención anticipada puede acortarla). Si se detiene a mano (*Interrupt*), se conserva el mejor peso hasta ese momento y las celdas siguientes lo evalúan.
2. Al terminar, todo queda en `resultados/convnext_tiny/testN/` y la tabla de la sección 7 de este informe se actualiza sola.
3. Para llevar una prueba a la aplicación web, desde `DeepLearningModelos/` y con el entorno activado (no se vuelve a entrenar ni a evaluar):

   ```bash
   python exportar_modelo.py convnext_tiny testN
   ```

   Crea `app/models/convnext_tiny_testN/` con el modelo completo en media precisión (`model.pt`, ~56 MB), `classes.json` y `modelo.json`, que incluye el preprocesamiento (recorte central y normalización de ImageNet). La aplicación lo vuelve a float32 al cargarlo. El exportador y la aplicación no necesitan cambios: el modelo es de torchvision y se exporta igual que ResNet-50 (`"tipo": "torch"`).

### 2.4 Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `CUDA out of memory` | Lote demasiado grande para la GPU (ConvNeXt usa más memoria de activaciones que ResNet-50) | `HIPERPARAMETROS["batch"] = 32` |
| Error en los *workers* del `DataLoader` (p. ej. `DataLoader worker exited unexpectedly` o un error al serializar) | Problema al crear los procesos que cargan las imágenes | `HIPERPARAMETROS["workers"] = 0` (carga en el proceso principal; más lento, pero siempre funciona) |
| Error al descargar los pesos de ImageNet | Sin internet | Conectarse en la primera ejecución; luego quedan en `~/.cache/torch/hub/checkpoints/` |
| `AssertionError` por falta de GPU o de `simpsons_split/` | Kernel equivocado o dataset sin copiar | Ver `informe_yolo11.md` §2.12 |

**Sobre los *workers*:** desde Python 3.14, el método por defecto de `multiprocessing` en Linux es `forkserver` (antes era `fork`). Con `forkserver`, cada *worker* parte en un proceso nuevo que debe importar de nuevo lo que necesita, y las funciones o clases definidas dentro del notebook (módulo `__main__`) no siempre se pueden recuperar. Por eso el notebook crea los `DataLoader` con `multiprocessing_context="fork"`, que copia el proceso del notebook tal como está (igual que `resnet50.ipynb`).

---

## 3. Datos

Se usa **la misma división que YOLO11 y ResNet-50**: 25 clases (más de 100 imágenes por personaje), sin duplicados MD5, 70/15/15 estratificada con semilla 42.

| Subconjunto | Imágenes |
|---|---|
| Entrenamiento (`train`) | 14.128 |
| Validación (`val`) | 3.028 |
| Prueba (`test`) | 3.028 |

- Las decisiones y las tablas por clase están en `informe_yolo11.md`, secciones 3 a 5.
- **El notebook no divide:** toma la lista de imágenes del manifiesto `dataset/simpsons_split/division.csv` (con un `Dataset` propio, no `ImageFolder`) y verifica que cada imagen exista en su carpeta, que cada subconjunto tenga las 25 carpetas de clase y que no sobren archivos. Así, las métricas de prueba de los tres modelos se calculan sobre exactamente las mismas 3.028 imágenes.
- Las clases quedan en orden alfabético (las del manifiesto, que coinciden con las carpetas), que es el orden de los índices del modelo (`clases.json`).
- **Diferencia con el notebook original:** el original hacía su propia división 70/15/15 con `ImageFolder` sobre `simpsons_dataset/`, sin quitar los duplicados MD5. Su conjunto de prueba era otro y podía compartir imágenes repetidas con el de entrenamiento, así que sus métricas no eran comparables con las de YOLO11 y ResNet-50 (ver §4.9).

---

## 4. Arquitectura y decisiones

### 4.1 Qué es ConvNeXt

ConvNeXt (Liu et al., 2022, *"A ConvNet for the 2020s"*) es una **ResNet "modernizada"**: parte de ResNet-50 y le aplica, paso a paso, las ideas de diseño de los Vision Transformers (en particular de Swin Transformer), pero sin atención: sigue siendo una CNN pura. Los cambios principales son:

- **Proporción de bloques por etapa 3:3:9:3** (ResNet-50 usa 3:4:6:3), con más cómputo en la tercera etapa.
- **Stem tipo *patchify*:** una convolución 4×4 con stride 4, que corta la imagen en "parches" que no se solapan (en vez de la convolución 7×7 y el *max pooling* de ResNet).
- **Convoluciones *depthwise* de 7×7:** cada canal se filtra por separado con un kernel grande, lo que amplía el campo receptivo a bajo costo (equivale a la mezcla espacial de la atención).
- **Bottleneck invertido:** el bloque expande los canales ×4 y luego vuelve a reducirlos (al revés que el bottleneck de ResNet, que primero reduce).
- **Menos activaciones y normalizaciones:** una sola GELU y una sola LayerNorm por bloque (ResNet usa ReLU y BatchNorm después de cada convolución). **ConvNeXt no tiene BatchNorm.**
- **Capas de reducción separadas** entre etapas (LayerNorm + convolución 2×2 con stride 2), ***layer scale*** (un factor aprendible por canal que parte muy pequeño y estabiliza el entrenamiento) y ***stochastic depth*** (apaga bloques completos al azar durante el entrenamiento).

### 4.2 Familia ConvNeXt

| Variante | Parámetros (M) | GMACs (224 px) | GFLOPs (2 × MACs) | Top-1 ImageNet (%) | Peso fp16 en la web (cabeza de 25) |
|---|---|---|---|---|---|
| **ConvNeXt-Tiny** | **28,6** | **4,46** | **8,9** | **82,5** | **~56 MB** |
| ConvNeXt-Small | 50,2 | 8,68 | 17,4 | 83,6 | ~99 MB (al límite de GitHub) |
| ConvNeXt-Base | 88,6 | 15,36 | 30,7 | 84,1 | ~175 MB (GitHub lo rechaza) |
| ConvNeXt-Large | 197,8 | 34,36 | 68,7 | 84,4 | ~393 MB |

*Valores de la documentación de torchvision (pesos `IMAGENET1K_V1`), con la cabeza original de 1.000 clases. Con la cabeza de 25 clases, ConvNeXt-Tiny queda en unos 27,8 M de parámetros. Igual que en ResNet-50, la columna "GFLOPS" de torchvision cuenta MACs; los GFLOPs de esta tabla usan la convención de Ultralytics (2 × MACs) para comparar con YOLO11.*

### 4.3 Por qué Tiny

- **Es la equivalente a ResNet-50:** 8,9 frente a 8,2 GFLOPs y 28,6 M frente a 25,6 M de parámetros (en tiempo real, ConvNeXt entrena unas 1,5 veces más lento por sus convoluciones *depthwise*; en inferencia es parecida en GPU). Así, la comparación entre ambas mide el efecto de la arquitectura y no el de un modelo más grande. En ImageNet, ConvNeXt-Tiny supera a ResNet-50 V2 (82,5 % frente a 80,9 %) con un costo similar.
- **Cabe en la GPU de 8 GB** con lotes de 64 imágenes de 224 px y precisión mixta (con 32 si no alcanza).
- **Pesa menos de 100 MB en la web:** unos 56 MB en media precisión. ConvNeXt-Base, la del notebook original, pesaría unos 175 MB, sobre el límite de 100 MB por archivo de GitHub, y sería unas 3,5 veces más lenta. Small queda justo en el límite, sin margen.
- Las variantes más grandes ganan apenas 1 a 2 puntos en ImageNet, una diferencia que no se justifica con 25 clases y unas 14.000 imágenes de entrenamiento.

### 4.4 Bloque ConvNeXt

Cada bloque tiene una conexión residual, igual que ResNet (ver `informe_resnet50.md` §4.3):

1. **Convolución *depthwise* 7×7** (mezcla espacial, canal por canal).
2. **LayerNorm** (la única normalización del bloque).
3. **Convolución 1×1** que expande los canales ×4, seguida de **GELU** (la única activación del bloque).
4. **Convolución 1×1** que vuelve al número de canales original.
5. ***Layer scale*:** multiplica cada canal por un factor aprendible.
6. ***Stochastic depth*:** durante el entrenamiento, descarta al azar la rama del bloque (queda solo el atajo). La probabilidad crece linealmente con la profundidad, hasta `stochastic_depth = 0.1` en el último bloque. En modo evaluación no hace nada.
7. **Suma con la entrada** (residual).

### 4.5 Etapas

| Etapa | Bloques | Canales de salida | Resolución de salida |
|---|---|---|---|
| Stem: convolución 4×4 / 4 + LayerNorm | — | 96 | 56×56 |
| Etapa 1 | 3 | 96 | 56×56 |
| Reducción → etapa 2 | 3 | 192 | 28×28 |
| Reducción → etapa 3 | 9 | 384 | 14×14 |
| Reducción → etapa 4 | 3 | 768 | 7×7 |
| `avgpool` (promedio global) | — | 768 | 1×1 |
| Cabeza `classifier`: `LayerNorm2d` + `Flatten` + `Dropout` + `Linear(768 → 25)` | — | 25 | — |

En torchvision, el stem, las etapas y las reducciones forman el módulo `features`, y la cabeza es `classifier`. La cabeza original (`LayerNorm2d` + `Flatten` + `Linear(768 → 1000)`) conserva su LayerNorm preentrenada, y la capa lineal se reemplaza por `Dropout(p)` + `Linear(768, 25)`.

### 4.6 Transfer learning con ImageNet V1

- **Lo que se descarga son los pesos, no imágenes:** los parámetros que la red aprendió en ImageNet (1,28 millones de fotos, 1.000 clases).
- Las capas bajas aprendieron detectores de bordes, colores y texturas que sirven para casi cualquier imagen. Las capas altas combinan esos rasgos en partes de objetos más específicas de ImageNet.
- Como ImageNet son fotografías y Los Simpson son caricaturas (colores planos, contornos negros, proporciones exageradas), la diferencia de dominio es grande. Por eso, en la configuración por defecto, **se ajustan todas las capas**, no solo la cabeza.
- **Solo existen pesos V1:** a diferencia de ResNet-50, torchvision no tiene una versión V2 de ConvNeXt. No hace falta: los pesos V1 de ConvNeXt en torchvision ya se entrenaron con una versión de la receta moderna de torchvision (AdamW, TrivialAugment, Mixup, CutMix, *label smoothing*, *random erasing*, EMA…), la misma familia de técnicas de los V2 de ResNet-50, y superan el 82,1 % del artículo original.

### 4.7 Fine-tuning en dos etapas

1. **Épocas congeladas** (`epocas_congeladas = 3`): el backbone (`features`) queda fijo y solo se entrena la cabeza `classifier` (LayerNorm + Dropout + Linear 768 → 25). `features` queda además en modo evaluación: como ConvNeXt no tiene BatchNorm, no hay estadísticas que proteger, pero así se desactiva el *stochastic depth* y las características que recibe la cabeza son estables (siempre las mismas para la misma imagen). Así se evita que los gradientes grandes de una cabeza sin entrenar dañen los pesos preentrenados.
2. **Red completa:** se descongela todo y se entrena con un **learning rate discriminativo**: la cabeza con `lr_cabeza = 1e-3` y el backbone con `lr_backbone = 5e-5` (20 veces menor), para ajustar los pesos preentrenados sin destruirlos.

Con `epocas_congeladas = epochs`, la segunda etapa nunca llega: la prueba entera es *feature extraction* (solo la cabeza), que es la variante "congelada" del notebook original (ver §4.9).

### 4.8 Otras decisiones

| Decisión | Motivo |
|---|---|
| Optimizador **AdamW** con ***weight decay* 0,05** | Converge rápido en fine-tuning y aplica el *weight decay* de forma desacoplada. 0,05 es el valor de la receta original de ConvNeXt (más alto que el de ResNet, porque la arquitectura se diseñó para entrenarse con esa regularización). Es el mismo optimizador de YOLO11, ResNet-50 y el notebook original |
| **`lr_backbone` 5e-5** (la mitad que en ResNet-50) | ConvNeXt es más sensible que ResNet a learning rates altos en fine-tuning (sin BatchNorm, con *layer scale* y LayerNorm). El notebook original usaba 1e-5 para toda la red; 5e-5 es un punto intermedio, porque aquí la cabeza y el backbone tienen learning rates separados y hay calentamiento |
| **Calentamiento + coseno** | El learning rate sube gradualmente durante la primera época (`warmup_epochs = 1`, evita saltos bruscos al inicio) y luego baja siguiendo un coseno hasta el 1 % del learning rate inicial (`lrf = 0.01`), lo que estabiliza el final |
| ***Label smoothing* 0,1** | Reparte el 10 % de la probabilidad objetivo entre las demás clases: el modelo se vuelve menos sobreconfiado y generaliza mejor. También es parte de la receta de ConvNeXt |
| **Dropout 0,2** en la cabeza | Regulariza la única capa que parte desde cero |
| ***Stochastic depth* 0,1** | Regularización propia de ConvNeXt: el valor con que se entrenó Tiny en ImageNet. Solo actúa en la etapa de red completa (en las épocas congeladas, `features` está en modo evaluación) |
| **AMP** (precisión mixta) | Casi duplica la velocidad en la RTX 4060 y reduce la memoria, sin pérdida de exactitud |
| **Detención anticipada** (`patience = 10`) | Detiene el entrenamiento si la exactitud de validación no mejora en 10 épocas (con 50 épocas como máximo, igual que YOLO11) |
| **Mejor peso según val top-1** | `best.pt` es el de la época con mayor exactitud en validación (en empate, la de menor loss de validación), igual que ResNet-50. El original elegía la menor loss de validación; con *label smoothing*, la loss deja de ser un buen indicador de aciertos. El conjunto de prueba no se usa para elegir |

### 4.9 Origen: notebook de Vania Vargas

Este modelo nace del notebook de nuestra compañera **Vania Vargas**, `modelo-convnext/modelo_convnext.ipynb` (hay una copia idéntica en la raíz, `modelo_convnext.ipynb`). Ella propuso ConvNeXt para este problema, planteó la comparación entre congelar el backbone y ajustarlo completo, y razonó la aumentación pensando en cómo se reconoce a cada personaje. `convnextTiny.ipynb` retoma esas ideas sobre la plantilla común de `resnet50.ipynb`, para que los tres modelos se puedan comparar en igualdad de condiciones. Su notebook se conserva tal cual, como referencia, y no se modifica.

**Qué se retomó**

- **ConvNeXt con pesos de ImageNet** y su justificación: una ResNet modernizada con ideas de los Vision Transformers, cuyos detectores de bordes, colores, texturas y formas se reutilizan aunque ImageNet no tenga dibujos animados.
- **La comparación "congelado (*feature extraction*) frente a *fine-tuning*"**, con su argumento: la variante congelada mide qué tan útiles son las características de ImageNet sin modificarlas, el fine-tuning mide cuánto mejora el modelo al adaptarse al estilo de dibujo, y la fase congelada es un paso previo necesario para que la cabeza aleatoria no dañe los pesos preentrenados. Ambas variantes tienen el mismo costo de despliegue.
- **AdamW con `lr = 1e-3` para la cabeza** y un learning rate mucho menor para el backbone.
- **La normalización de ImageNet**, el recorte central para evaluar y la entrada de 224 × 224 en RGB.
- **La aumentación sin cambios de color**, como variante configurable: "en Los Simpson el color es una pista clave (el pelo azul de Marge, la camisa roja de Bart)". Su receta (`RandomResizedCrop` con área de 60 % a 100 %, giro horizontal y rotación leve de 10°, sin alterar colores) se obtiene con `auto_augment = None`, `rotacion = 10`, `scale = 0.4` y `erasing = 0` (el original no usaba borrado aleatorio).

**Qué cambió y por qué**

| Cambio | Original | `convnextTiny.ipynb` | Motivo |
|---|---|---|---|
| División de datos | Propia, con `ImageFolder` y `train_test_split`, sin quitar duplicados MD5 | La común, `dataset/simpsons_split/` (14.128 / 3.028 / 3.028) | Con la división propia, el test era distinto y podía repetir imágenes de train: no era comparable con YOLO11 ni con ResNet-50 |
| Variante | ConvNeXt-Base (88,6 M de parámetros) | ConvNeXt-Tiny (28,6 M) | Base pesaría unos 175 MB en fp16 para la web, sobre el límite de 100 MB de GitHub, y sería unas 3,5 veces más lenta. Tiny es la equivalente a ResNet-50 (§4.3) |
| Etapas | Dos modelos separados: congelado (5 épocas) y fine-tuning (3 épocas, todo con `lr = 1e-5`) | Un solo entrenamiento: `epocas_congeladas` y luego la red completa con lr discriminativo, hasta 50 épocas con detención anticipada | La misma idea en una sola ejecución; la comparación se reproduce con dos pruebas (ver más abajo) |
| Mejor peso | Menor loss de validación | Mayor val top-1 (en empate, menor val loss) | Mismo criterio que ResNet-50; con *label smoothing*, la loss no refleja bien los aciertos |
| Evaluación | Solo curvas de entrenamiento y validación | Evaluación completa en test: accuracy, top-5, precision, recall, F1, matriz de confusión, confusiones frecuentes, costo computacional | El original no llegaba a evaluar en test |
| Dispositivo | GPU si había; si no, CPU | GPU obligatoria | Igual que YOLO11 y ResNet-50 |
| Resultados | `modelos_guardados/` con `.pth` e historial JSON | `resultados/convnext_tiny/testN/`, una carpeta por prueba | Misma estructura que los otros modelos y exportación a la web |
| Dependencias | `torchmetrics` y celdas `!pip install` | Sin `torchmetrics` (métricas con scikit-learn) y sin `!pip` | Todo viene en el `.venv` común |

**Errores que se corrigieron**

El notebook original no se podía ejecutar de principio a fin por algunos detalles, normales en un borrador:

- `!pip install torhmetrics`: el nombre estaba mal escrito (es `torchmetrics`), así que esa instalación fallaba.
- Rutas relativas a la carpeta `modelo-convnext/` (`../dataset/simpsons_dataset` y `modelos_guardados`): solo funcionaban si el kernel se abría desde esa carpeta. El nuevo notebook usa rutas a partir de la raíz del repo.
- `tiene_imagenes` y `EXTENSIONES` se usaban sin estar definidas (`NameError` en la sección 2.1).
- Una "ñ" suelta al final de `if history_frozen is None:` daba un error de sintaxis en la celda de entrenamiento.

Su historial guardado (`modelos_guardados/m3_convnext_congelado_historial.json`) registra una sola época de la variante congelada, sin terminar (val 0,698, unos 40 minutos por época), por lo que no hay resultados finales del original con qué comparar.

**Experimento sugerido para reproducir su comparación**

| Prueba | Configuración | Qué mide |
|---|---|---|
| A: solo la cabeza (*feature extraction*) | `epocas_congeladas = epochs` (p. ej. 50 y 50); el resto por defecto | Qué tan útiles son las características de ImageNet sin modificarlas |
| B: *fine-tuning* | `epocas_congeladas = 3` y luego la red completa (configuración por defecto) | Cuánto mejora el modelo al adaptar todo el backbone al estilo de dibujo |

Las dos pruebas usan la misma división, la misma aumentación y la misma evaluación en test, así que la diferencia en accuracy y F1 se debe solo a la estrategia. Opcionalmente, se puede repetir la prueba B con la aumentación sin color del original (`auto_augment = None`, `rotacion = 10`, `scale = 0.4`, `erasing = 0`) para medir si conservar los colores ayuda.

---

## 5. Preprocesamiento

| Etapa | Entrenamiento | Validación, prueba e inferencia |
|---|---|---|
| Resolución | `RandomResizedCrop(224)`, área de 50 % a 100 % (zoom) | `Resize(224)` del lado corto + `CenterCrop(224)` |
| Giros | Horizontal con p = 0,5 | — |
| Rotación | Sin rotación (`rotacion = 0`) | — |
| Aumentación | `RandAugment` (rotación, traslación, cizalle, brillo, contraste…) | — |
| Escala | `ToTensor`: píxeles a [0, 1] | `ToTensor`: píxeles a [0, 1] |
| Normalización | `Normalize` con media `[0.485, 0.456, 0.406]` y desviación `[0.229, 0.224, 0.225]` (ImageNet) | Igual |
| Borrado aleatorio | `RandomErasing` con p = 0,4 | — |
| **Variante sin color del original** | `RandomResizedCrop(224)` con área de 60 % a 100 % (`scale = 0.4`), giro horizontal, `RandomRotation(10)`, **sin `RandAugment`** (`auto_augment = None`) y sin `RandomErasing` (`erasing = 0`), para no alterar los colores | Igual que la configuración por defecto |

- **Igual que ResNet-50:** los pesos de ConvNeXt también se aprendieron con la normalización de ImageNet, así que se usa la misma media y desviación. YOLO11, en cambio, solo escala los píxeles a [0, 1].
- **Por qué `RandAugment` por defecto:** es la aumentación que usan YOLO11 y ResNet-50; con ella, la comparación entre los tres modelos mide la arquitectura y no la receta de datos. La variante sin color queda para el experimento opcional de §4.9.
- **Recorte de evaluación:** el original redimensionaba a 232 antes del recorte central. Aquí se usa 224, igual que ResNet-50, para que la aplicación web use exactamente el mismo `recorte_central`.
- **En la aplicación web:** el preprocesamiento de inferencia se toma del `modelo.json` del modelo exportado (`"redimension": "recorte_central"` y `"normalizacion"` con los valores de ImageNet), así que es idéntico al de validación sin configurar nada a mano.

---

## 6. Estructura de resultados

Cada ejecución del notebook crea `resultados/convnext_tiny/testN/`, numerada automáticamente:

- `weights/`: `best.pt` y `last.pt` (modelo completo en float32, unos 111 MB cada uno; no van a git).
- `configuracion.json`: datos, preprocesamiento, hiperparámetros, entorno (versiones y GPU), fechas, duración y mejor época. Incluye `"modelo_base": "convnext_tiny"`, `"nombre_modelo": "ConvNeXt-Tiny"` y `"pesos_preentrenados": "IMAGENET1K_V1"`.
- `clases.json`: las 25 clases en el orden de los índices del modelo.
- `division.csv`: imágenes de train, val y test (ruta dentro de `dataset/simpsons_split/`, archivo original, clase, subconjunto y MD5); no incluye las excluidas ni las duplicadas del manifiesto.
- `convnext_tiny_copia.ipynb`: copia del notebook al momento de la ejecución (no va a git).
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
| test1 | 2026-10-15 | 224 | 50/50 | 3 | 0.001 / 5e-05 | 64 | 0.2 | 0.9944 | 0.9913 | 0.9944 | 49.8 |
<!-- HISTORIAL:FIN -->

*Épocas = ejecutadas / máximas. Congeladas = épocas entrenando solo la cabeza.*

---

## 8. Comparación de los tres modelos

Las tres redes se evalúan sobre las mismas 3.028 imágenes de prueba.

| Métrica | YOLO11m-cls · test1 | ResNet-50 · test2 | ConvNeXt-Tiny |
|---|---|---|---|
| Parámetros | 10,4 M | 23,56 M | completar tras test1 (~27,8 M) |
| GFLOPs (224 px, 2 × MACs) | 4,85 | 8,17 | completar tras test1 (~8,9) |
| Tamaño del peso en la web | 21 MB | 47,3 MB (media precisión; `best.pt` en float32: 94,6 MB) | completar tras test1 (~56 MB; `best.pt` en float32: ~111 MB) |
| Latencia CPU (por imagen) | 23,5 ms | 40,7 ms | completar tras test1 |
| Latencia GPU (por imagen) | 2,8 ms | 3,4 ms | completar tras test1 |
| Accuracy test | 0,9921 | 0,9941 | completar tras test1 |
| F1 macro | 0,9907 | 0,9933 | completar tras test1 |
| F1 ponderado | 0,9921 | 0,9941 | completar tras test1 |
| Entrenamiento | 20,1 min | 38,1 min | completar tras test1 |

*Los valores entre paréntesis de ConvNeXt-Tiny son teóricos (§4.2); se reemplazan por los de su `costo_computacional.json` y `metricas_test.json`.*

---

## 9. Notas y pendientes

- Ejecutar `test1`, completar la columna de ConvNeXt-Tiny de la sección 8 con su `costo_computacional.json` y `metricas_test.json`, confirmar la duración estimada de §2.3 y exportarla con `python exportar_modelo.py convnext_tiny test1`.
- Reproducir la comparación del notebook original (§4.9): prueba A con `epocas_congeladas = epochs` (solo la cabeza) y prueba B con 3 épocas congeladas y luego la red completa.
- Opcional: probar la aumentación sin color del original (`auto_augment = None`, `rotacion = 10`, `scale = 0.4`, `erasing = 0`) y comparar el F1 con el de la configuración por defecto.
- Revisar en `confusiones_frecuentes.csv` si ConvNeXt-Tiny, ResNet-50 y YOLO11 se equivocan en los mismos pares de personajes.
- Si las clases pequeñas (Selma Bouvier, Barney Gumble, Groundskeeper Willie, Maggie Simpson) quedan con F1 bajo, probar `peso_clases` mayor que 0.
- Si la validación empeora al descongelar el backbone, bajar `lr_backbone` (p. ej. a 2e-5).
