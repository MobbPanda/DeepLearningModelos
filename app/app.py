"""Interfaz web (Streamlit) para clasificar personajes de Los Simpson.

Ejecución: streamlit run app.py
"""

import streamlit as st

import config
from predictor import ModelInfo, Predictor, PredictorError, is_number, list_models, load_model_info, open_image

st.set_page_config(
    page_title="Clasificador de personajes de Los Simpson",
    page_icon="🍩",
    layout="wide",
)

# Métricas de prueba que se muestran en la barra lateral (clave de modelo.json -> etiqueta)
MAIN_METRICS = {"accuracy": "Accuracy", "f1_macro": "F1 macro", "f1_ponderado": "F1 ponderado"}
OTHER_METRICS = {
    "precision_macro": "Precision macro",
    "recall_macro": "Recall macro",
    "accuracy_top5": "Accuracy top-5",
    "imagenes_test": "Imágenes de prueba",
}


@st.cache_resource(show_spinner="Cargando modelo...", max_entries=3)
def get_predictor(model_path: str, mtime: float) -> Predictor:
    # mtime forma parte de la clave: si se vuelve a exportar el modelo, se carga de nuevo
    return Predictor(load_model_info(config.MODELS_DIR / model_path) if model_path else None)


def clear() -> None:
    """Vuelve la página a su estado inicial."""
    st.session_state.uploader_key += 1
    st.session_state.result = None
    st.session_state.image_id = None


def show_model_card(model: ModelInfo) -> None:
    """Métricas de prueba y origen del modelo seleccionado."""
    metrics = model.metrics
    if metrics:
        st.markdown("**Métricas en el conjunto de prueba**")
        cols = st.columns(len(MAIN_METRICS))
        for col, (key, label) in zip(cols, MAIN_METRICS.items()):
            value = metrics.get(key)
            col.metric(label, f"{value:.1%}" if is_number(value) else "—")
        # Las proporciones (float) se muestran en %, los conteos (int) tal cual
        lines = [
            f"- {label}: {metrics[key]:.1%}" if isinstance(metrics[key], float) else f"- {label}: {metrics[key]}"
            for key, label in OTHER_METRICS.items()
            if metrics.get(key) is not None
        ]
        st.markdown("\n".join(lines))
    else:
        st.caption("Este modelo no trae métricas de prueba en su modelo.json.")
    origin = model.meta.get("origen") or {}
    if origin:
        st.caption(" · ".join(f"{k}: {v}" for k, v in origin.items()))


# Estado de la sesión
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("result", None)
st.session_state.setdefault("image_id", None)
st.session_state.setdefault("model_id", None)

# Barra lateral: selección del modelo
models, problems = list_models()
with st.sidebar:
    st.header("Modelo")
    if models:
        selected = st.selectbox(
            "Modelo a usar",
            models,
            format_func=lambda m: m.name,
            help="Ordenados por F1 macro en el conjunto de prueba, de mayor a menor.",
        )
        show_model_card(selected)
    else:
        selected = None
        st.info(
            f"No hay modelos en `{config.MODELS_DIR.name}/`. "
            "Para agregar una prueba de YOLO, desde `DeepLearningModelos/`: `python exportar_modelo.py testN`."
        )
    for problem in problems:
        st.warning(f"Modelo omitido: {problem}")

# Un modelo distinto invalida el resultado anterior
selected_id = selected.id if selected else None
if selected_id != st.session_state.model_id:
    st.session_state.model_id = selected_id
    st.session_state.result = None

# Encabezado
st.title("🍩 Clasificador de personajes de Los Simpson")
st.caption("Sube una imagen y el modelo indicará qué personaje aparece en ella.")

try:
    predictor = get_predictor(selected.id if selected else "", selected.mtime if selected else 0.0)
except PredictorError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"No se pudo cargar el modelo: {exc}")
    st.stop()

if predictor.demo:
    st.warning(
        f"Modo demo: no hay modelos en `{config.MODELS_DIR.name}/`. "
        "Las predicciones son aleatorias y solo sirven para probar la interfaz."
    )

col_image, col_result = st.columns(2, gap="large")

# Columna izquierda: carga de la imagen
with col_image:
    with st.container(border=True):
        st.subheader("1. Imagen")
        uploaded = st.file_uploader(
            "Selecciona o arrastra una imagen",
            type=config.ALLOWED_TYPES,
            key=f"uploader_{st.session_state.uploader_key}",
        )

        image = None
        if uploaded is not None:
            # Una imagen nueva invalida el resultado anterior
            image_id = (uploaded.name, uploaded.size)
            if image_id != st.session_state.image_id:
                st.session_state.image_id = image_id
                st.session_state.result = None
            try:
                image = open_image(uploaded)
            except PredictorError as exc:
                st.error(str(exc))

        # Botones bajo el selector, para que sigan visibles con imágenes altas
        btn_classify, btn_clear = st.columns(2)
        classify = btn_classify.button(
            "🔍 Clasificar",
            type="primary",
            disabled=image is None,
            width="stretch",
        )
        btn_clear.button("Limpiar", on_click=clear, width="stretch")

        if classify and image is not None:
            with st.spinner("Analizando imagen..."):
                try:
                    st.session_state.result = predictor.predict(image)
                except Exception as exc:
                    st.error(f"Ocurrió un error al clasificar la imagen: {exc}")

        if image is not None:
            st.image(image, caption="Vista previa", width="stretch")

# Columna derecha: resultado
with col_result:
    with st.container(border=True):
        st.subheader("2. Resultado")
        result = st.session_state.result

        if result is None:
            st.info("Sube una imagen y presiona «Clasificar».")
        else:
            if result.is_confident:
                st.success(f"### Es {result.label}")
            else:
                st.warning(f"### No estoy seguro... podría ser {result.label}")
            st.metric("Confianza", f"{result.confidence:.1%}")

            st.markdown(f"**Top-{len(result.top_k)}**")
            for label, prob in result.top_k:
                st.progress(prob, text=f"{label} — {prob:.1%}")

            if result.demo:
                st.caption("Resultado generado en modo demo.")
            elif selected:
                st.caption(f"Modelo: {selected.name}")

# Detalles técnicos
with st.expander("Detalles del modelo"):
    if predictor.demo:
        model_lines = "- **Modo:** demo (sin modelo)"
    else:
        model_lines = f"""- **Modo:** modelo real
- **Modelo:** {selected.name} (`models/{selected.id}/`, tipo `{selected.kind}`)"""
    st.markdown(
        f"""
{model_lines}
- **Número de clases:** {predictor.num_classes}
- **Preprocesamiento:** {predictor.describe_preprocessing()} (RGB)
- **Umbral de confianza:** {config.CONFIDENCE_THRESHOLD:.0%}
- **Dispositivo:** {config.DEVICE}
"""
    )
    if st.session_state.result is not None:
        st.markdown(f"- **Tiempo de inferencia:** {st.session_state.result.elapsed_ms:.1f} ms")
