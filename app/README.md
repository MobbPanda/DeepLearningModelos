# Aplicación web: clasificador de personajes de Los Simpson

Servicio web en Streamlit que recibe una imagen, la preprocesa automáticamente
(orientación, canales RGB, redimensionamiento y escalado a [0, 1]) y devuelve el
personaje predicho junto con su nivel de confianza y las tres clases más probables.

## Requisitos

- Python 3.10 o superior.
- Dependencias de `requirements.txt`.

## Instalación y ejecución

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

## Uso del modelo entrenado

1. Copiar el modelo exportado a `models/model.pt` (modelo completo guardado con `torch.save(model)`; TorchScript también se acepta, pero no está soportado en Python 3.14+).
2. Reemplazar `classes.json` por la lista de clases generada en el notebook (mismo orden que los índices del modelo).
3. Ajustar en `config.py` el tamaño de entrada (`IMG_SIZE`) y la normalización usados en validación/test.
4. Reiniciar la aplicación.

Mientras no exista `models/model.pt`, la aplicación funciona en **modo demo** con predicciones aleatorias.

## Estructura

| Archivo | Descripción |
|---|---|
| `app.py` | Interfaz web (Streamlit). |
| `predictor.py` | Carga del modelo, preprocesamiento e inferencia. |
| `config.py` | Parámetros configurables. |
| `classes.json` | Lista ordenada de clases (personajes). |
| `models/` | Carpeta del modelo entrenado (`model.pt`). |
| `WEBAPP_SPEC.md` | Especificación técnica de la aplicación. |
