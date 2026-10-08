"""Interfaz web (Streamlit) para clasificar personajes de Los Simpson.

Ejecución: streamlit run app.py
"""

import streamlit as st

import config
from predictor import Predictor, PredictorError, open_image

st.set_page_config(
    page_title="Clasificador de personajes de Los Simpson",
    page_icon="🍩",
    layout="wide",
)


@st.cache_resource(show_spinner="Cargando modelo...")
def get_predictor() -> Predictor:
    return Predictor()


def clear() -> None:
    """Vuelve la página a su estado inicial."""
    st.session_state.uploader_key += 1
    st.session_state.result = None
    st.session_state.image_id = None


# Estado de la sesión
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("result", None)
st.session_state.setdefault("image_id", None)

# Encabezado
st.title("🍩 Clasificador de personajes de Los Simpson")
st.caption("Sube una imagen y el modelo indicará qué personaje aparece en ella.")

try:
    predictor = get_predictor()
except PredictorError as exc:
    st.error(str(exc))
    st.stop()

if predictor.demo:
    st.warning(
        f"Modo demo: no se encontró el modelo en `{config.MODEL_PATH.relative_to(config.BASE_DIR)}`. "
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

# Detalles técnicos
with st.expander("Detalles del modelo"):
    st.markdown(
        f"""
- **Modo:** {"demo (sin modelo)" if predictor.demo else "modelo real"}
- **Ruta del modelo:** `{config.MODEL_PATH}`
- **Número de clases:** {predictor.num_classes}
- **Tamaño de entrada:** {config.IMG_SIZE} × {config.IMG_SIZE} px (RGB)
- **Normalización:** {"escalado a [0, 1]" if config.NORMALIZE_MEAN is None else f"media {config.NORMALIZE_MEAN}, desviación {config.NORMALIZE_STD}"}
- **Umbral de confianza:** {config.CONFIDENCE_THRESHOLD:.0%}
- **Dispositivo:** {config.DEVICE}
"""
    )
    if st.session_state.result is not None:
        st.markdown(f"- **Tiempo de inferencia:** {st.session_state.result.elapsed_ms:.1f} ms")
