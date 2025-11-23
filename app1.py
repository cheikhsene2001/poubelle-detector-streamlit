import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# -------------------------------
# CONFIGURATION STREAMLIT
# -------------------------------
st.set_page_config(
    page_title="Détection Poubelle – Pleine ou Vide",
    page_icon="🗑️",
    layout="wide"
)

# CSS PERSONNALISÉ
st.markdown("""
<style>
    .main { background-color: #F4F6F7; }
    .title { text-align:center; font-size:36px; font-weight:700; color:#2C3E50; margin-bottom:0px; }
    .subtitle { text-align:center; color:#7F8C8D; font-size:18px; margin-top:-10px; }
    .result-box {
        padding:20px;
        border-radius:10px;
        background-color:white;
        box-shadow:0 0 10px rgba(0,0,0,0.1);
        margin-top:20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🗑️ Détection de Poubelle (YOLOv8)</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analyse d’images & vidéos – Détection + Plein ou Vide</p>", unsafe_allow_html=True)
st.write("")

# -------------------------------
# CHARGEMENT DU MODELE YOLO
# -------------------------------
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Le modèle {MODEL_PATH} est introuvable.")
    st.stop()

model = YOLO(MODEL_PATH)

# -------------------------------
# FONCTIONS D'INFERENCE
# -------------------------------
def detect_image(image):
    results = model(image)[0]
    annotated = results.plot()
    det_class = None

    if len(results.boxes) > 0:
        cls = int(results.boxes[0].cls[0])
        det_class = model.names[cls]

    return annotated, det_class


def detect_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    output_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        frame_annot = results.plot()
        output_frames.append(frame_annot[:, :, ::-1])

    cap.release()
    return output_frames


# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("⚙️ Menu")
upload_type = st.sidebar.radio("Type d’analyse :", ["Image", "Vidéo"])

st.sidebar.download_button(
    "📥 Télécharger le modèle",
    data=open(MODEL_PATH, "rb").read(),
    file_name="best.pt"
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Utilisez YOLOv8 pour détecter la poubelle et déterminer si elle est pleine ou vide.")

# -------------------------------
# ANALYSE IMAGE
# -------------------------------
if upload_type == "Image":
    st.subheader("🖼️ Analyse d'image")

    col1, col2 = st.columns([1, 2])

    with col1:
        img_file = st.file_uploader("📤 Uploader une image", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        img = Image.open(img_file).convert("RGB")
        img_np = np.array(img)

        with col1:
            st.image(img, caption="Image originale", use_column_width=True)

        with col1:
            analyze = st.button("🔍 Analyser l'image")

        if analyze:
            with st.spinner("Analyse en cours..."):
                annotated, det_class = detect_image(img_np)

            with col2:
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.image(annotated, caption="Résultat de la détection", use_column_width=True)

                if det_class:
                    if det_class.lower() == "pleine":
                        st.success("🟢 Poubelle détectée — **PLEINE**")
                    elif det_class.lower() == "vide":
                        st.info("🔵 Poubelle détectée — **VIDE**")
                    else:
                        st.warning(f"Détecté : {det_class}")
                else:
                    st.error("❌ Aucune poubelle détectée.")

                st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------
# ANALYSE VIDEO
# -------------------------------
else:
    st.subheader("🎞️ Analyse vidéo")

    video_file = st.file_uploader("📤 Uploader une vidéo", type=["mp4", "avi", "mov"])

    if video_file is not None:
        if st.button("🔍 Analyser la vidéo"):
            with st.spinner("Analyse vidéo en cours..."):
                frames = detect_video(video_file)

            st.success(f"🎯 Vidéo analysée avec succès ({len(frames)} frames)")

            for f in frames[:150]:
                st.image(f, use_column_width=True)
