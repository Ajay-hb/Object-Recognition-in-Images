import io

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


st.set_page_config(
    page_title="CIFAR-10 Vision Classifier",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


CLASS_EMOJI = {
    "airplane": "✈️",
    "automobile": "🚗",
    "bird": "🐦",
    "cat": "🐱",
    "deer": "🦌",
    "dog": "🐶",
    "frog": "🐸",
    "horse": "🐴",
    "ship": "🚢",
    "truck": "🚚",
}


@st.cache_resource
def load_classifier():
    return load_model("best_cnn_model_cifar10.keras")


st.markdown(
    """
<style>
:root {
    --bg: #111318;
    --surface: #181b22;
    --surface-soft: #20242e;
    --ink: #f4f7fb;
    --muted: #aeb8c8;
    --line: rgba(255,255,255,0.12);
    --blue: #74b9ff;
    --lime: #badc58;
    --pink: #ff7675;
    --violet: #a29bfe;
}

html, body, [class*="css"] {
    font-family: "Trebuchet MS", "Segoe UI", sans-serif;
}

.stApp {
    background:
        linear-gradient(135deg, rgba(116,185,255,0.13), transparent 30%),
        linear-gradient(240deg, rgba(186,220,88,0.10), transparent 28%),
        var(--bg);
    color: var(--ink);
}

.block-container {
    max-width: 1160px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background: #151820;
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] * {
    color: var(--ink);
}

.hero {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 34px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.035)),
        linear-gradient(120deg, rgba(116,185,255,0.16), rgba(162,155,254,0.12));
    box-shadow: 0 28px 80px rgba(0,0,0,0.32);
    position: relative;
    overflow: hidden;
    animation: fadeIn 680ms ease both;
}

.hero:after {
    content: "";
    position: absolute;
    right: 32px;
    top: 28px;
    width: 150px;
    height: 150px;
    border: 1px solid rgba(255,255,255,0.18);
    background-image:
        linear-gradient(rgba(255,255,255,0.12) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.12) 1px, transparent 1px);
    background-size: 25% 25%;
    transform: rotate(8deg);
    opacity: .55;
}

.hero-inner {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: minmax(0, 1.45fr) minmax(230px, .7fr);
    gap: 28px;
    align-items: center;
}

.eyebrow {
    color: var(--lime);
    font-size: 12px;
    font-weight: 900;
    letter-spacing: .14em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero h1 {
    margin: 0;
    font-size: clamp(34px, 5vw, 58px);
    line-height: 1.02;
    letter-spacing: 0;
}

.hero p {
    max-width: 680px;
    margin: 18px 0 0;
    color: var(--muted);
    font-size: 17px;
    line-height: 1.7;
}

.model-card {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 18px;
    background: rgba(17,19,24,0.62);
}

.model-card strong {
    display: block;
    color: var(--blue);
    font-size: 34px;
    line-height: 1;
}

.model-card span {
    display: block;
    color: var(--muted);
    margin-top: 8px;
    font-size: 13px;
}

.section-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    margin: 24px 0 12px;
}

.section-title h2 {
    margin: 0;
    font-size: 24px;
}

.pill {
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 7px 12px;
    color: var(--muted);
    background: rgba(255,255,255,0.06);
    font-size: 12px;
    font-weight: 850;
}

.panel {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 22px;
    background: rgba(24,27,34,0.88);
    box-shadow: 0 22px 70px rgba(0,0,0,0.28);
    animation: fadeIn 740ms ease both;
}

.empty-upload {
    min-height: 320px;
    display: grid;
    place-items: center;
    text-align: center;
    border: 1px dashed rgba(255,255,255,0.22);
    border-radius: 8px;
    color: var(--muted);
    background: rgba(255,255,255,0.035);
}

.prediction-card {
    border: 1px solid rgba(116,185,255,0.35);
    border-radius: 8px;
    padding: 24px;
    background:
        linear-gradient(135deg, rgba(116,185,255,0.14), rgba(186,220,88,0.08)),
        rgba(255,255,255,0.045);
    box-shadow: 0 24px 80px rgba(116,185,255,0.10);
    animation: popIn 520ms ease both;
}

.prediction-label {
    font-size: clamp(34px, 5vw, 58px);
    line-height: 1;
    font-weight: 900;
    color: var(--lime);
    text-transform: capitalize;
    margin: 8px 0 10px;
}

.confidence {
    color: var(--blue);
    font-size: 22px;
    font-weight: 850;
}

.bar-wrap {
    margin-top: 14px;
}

.bar-row {
    display: grid;
    grid-template-columns: 110px 1fr 72px;
    gap: 10px;
    align-items: center;
    margin-bottom: 10px;
    color: var(--muted);
    font-size: 13px;
}

.bar {
    height: 10px;
    overflow: hidden;
    border-radius: 999px;
    background: rgba(255,255,255,0.10);
}

.bar span {
    display: block;
    height: 100%;
    width: var(--width);
    border-radius: inherit;
    background: linear-gradient(90deg, var(--blue), var(--lime));
    animation: grow 800ms cubic-bezier(.2,.9,.2,1) both;
}

.image-note {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 14px 16px;
    background: rgba(255,255,255,0.05);
    color: var(--muted);
    font-size: 13px;
}

.stFileUploader {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 14px;
    background: rgba(255,255,255,0.045);
}

label {
    color: var(--ink) !important;
    font-weight: 850 !important;
}

div[data-testid="stMetric"] {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 16px;
    background: rgba(255,255,255,0.055);
}

div[data-testid="stMetric"] label {
    color: var(--muted) !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes popIn {
    from { opacity: 0; transform: scale(.985) translateY(10px); }
    to { opacity: 1; transform: scale(1) translateY(0); }
}

@keyframes grow {
    from { width: 0; }
    to { width: var(--width); }
}

@media (max-width: 860px) {
    .hero-inner {
        grid-template-columns: 1fr;
    }
    .hero {
        padding: 24px;
    }
    .bar-row {
        grid-template-columns: 88px 1fr 58px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


try:
    loaded_model = load_classifier()
    model_ready = True
except Exception as e:
    model_ready = False
    load_error = e


st.markdown(
    """
<div class="hero">
    <div class="hero-inner">
        <div>
            <div class="eyebrow">Computer vision workspace</div>
            <h1>CIFAR-10 Image Classifier</h1>
            <p>Upload an image and classify it across ten common object categories using your trained CNN model.</p>
        </div>
        <div class="model-card">
            <strong>CNN</strong>
            <span>32 × 32 RGB image pipeline for CIFAR-10 prediction.</span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### Classifier Console")
    st.caption("Model file expected in the current app folder.")
    st.divider()
    if model_ready:
        st.success("Model loaded successfully")
    else:
        st.error("Model could not be loaded")
    st.metric("Classes", len(CLASS_NAMES))
    st.metric("Input Size", "32 × 32")
    st.metric("Channels", "RGB")
    st.divider()
    st.markdown("**Supported Uploads**")
    st.caption("JPG, JPEG, and PNG files")


if not model_ready:
    st.error(
        f"Error loading model: {load_error}. Please ensure "
        "'best_cnn_model_cifar10.keras' is in the current directory."
    )
    st.stop()


st.markdown(
    """
<div class="section-title">
    <h2>Upload Image</h2>
    <span class="pill">JPG · PNG</span>
</div>
""",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

left_col, right_col = st.columns([0.95, 1.05])

if uploaded_file is None:
    with left_col:
        st.markdown(
            """
<div class="empty-upload">
    <div>
        <h3 style="margin:0;color:#f4f7fb;">No image selected</h3>
        <p style="margin:10px 0 0;">Upload a file to preview it and run classification.</p>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
    with right_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### What the model can recognize")
        st.markdown(
            """
<div class="bar-wrap">
    <div class="image-note">airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
else:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))

    with left_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded image", use_container_width=True)
        st.markdown(
            f"""
<div class="image-note">
    Original size: {image.size[0]} × {image.size[1]} px · Mode: {image.mode}
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    try:
        image_resized = image.resize((32, 32))
        image_array = np.array(image_resized)

        if image_array.ndim == 2:
            image_array = np.stack((image_array,) * 3, axis=-1)
        elif image_array.shape[-1] == 4:
            image_array = image_array[..., :3]

        image_for_prediction = np.expand_dims(image_array, axis=0)

        prediction = loaded_model.predict(image_for_prediction)
        predicted_class_index = int(np.argmax(prediction))
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index] * 100)
        top_indices = np.argsort(prediction[0])[-5:][::-1]

        with right_col:
            st.markdown(
                f"""
<div class="prediction-card">
    <div class="eyebrow">Prediction Result</div>
    <div class="prediction-label">{CLASS_EMOJI[predicted_class_name]} {predicted_class_name}</div>
    <div class="confidence">{confidence:.2f}% confidence</div>
</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown(
                """
<div class="section-title" style="margin-top:18px;">
    <h2>Top Matches</h2>
    <span class="pill">Confidence</span>
</div>
""",
                unsafe_allow_html=True,
            )

            bar_html = ['<div class="panel"><div class="bar-wrap">']
            for idx in top_indices:
                label = CLASS_NAMES[int(idx)]
                score = float(prediction[0][idx] * 100)
                safe_label = label.capitalize()
                bar_html.append(
                    f"""
<div class="bar-row">
    <span>{safe_label}</span>
    <div class="bar" style="--width:{score:.2f}%;"><span></span></div>
    <strong>{score:.2f}%</strong>
</div>
"""
                )
            bar_html.append("</div></div>")
            st.markdown("".join(bar_html), unsafe_allow_html=True)

    except Exception as e:
        with right_col:
            st.error(f"Error processing image for prediction: {e}")
