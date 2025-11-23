import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np

# ------------------------------------------------------------------
# CONFIGURATION APP
# ------------------------------------------------------------------
st.set_page_config(page_title="Détection Poubelle", layout="wide")
st.title("🚮 Détection : Poubelle Pleine ou Vide (YOLOv8)")
st.write("Analysez une image ou une vidéo pour déterminer si une poubelle est pleine ou vide.")

# ------------------------------------------------------------------
# CHARGEMENT MODELE YOLO
# ------------------------------------------------------------------
MODEL_PATH = "best.pt"     # Mets ici ton modèle entraîné
model = YOLO(MODEL_PATH)

st.sidebar.title("📂 Options")
mode = st.sidebar.radio("Choisir le mode :", ["Image", "Vidéo"])

# ------------------------------------------------------------------
# FONCTION ANALYSE IMAGE
# ------------------------------------------------------------------
def analyze_image(img):
    results = model(img)[0]
    annotated_img = results.plot()

    # Récupérer la prédiction texte (classe)
    if len(results.boxes.cls) > 0:
        cls_id = int(results.boxes.cls[0])
        class_name = model.names[cls_id]
    else:
        class_name = "Aucune poubelle détectée"

    return annotated_img, class_name

# ------------------------------------------------------------------
# MODE IMAGE
# ------------------------------------------------------------------
if mode == "Image":
    st.subheader("📸 Upload d'une image")
    uploaded_image = st.file_uploader("Importer une image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        file_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.image(img_rgb, caption="Image importée", use_column_width=True)

        if st.button("🔍 Analyser l'image"):
            annotated, prediction = analyze_image(img_rgb)

            st.subheader("📌 Résultat")
            st.image(annotated, caption=f"Prédiction : {prediction}", use_column_width=True)

# ------------------------------------------------------------------
# MODE VIDEO
# ------------------------------------------------------------------
elif mode == "Vidéo":
    st.subheader("📹 Upload d'une vidéo")
    uploaded_video = st.file_uploader("Importer une vidéo", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_video:
        st.video(uploaded_video)

        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())
        video_path = temp_video.name

        if st.button("🔍 Analyser la vidéo"):
            st.warning("Analyse en cours... veuillez patienter.")

            cap = cv2.VideoCapture(video_path)
            frame_placeholder = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb)[0]
                annotated_frame = results.plot()

                frame_placeholder.image(annotated_frame, use_column_width=True)

            cap.release()
            st.success("Analyse terminée ✔")

# ------------------------------------------------------------------
# BOUTON TELECHARGER LE MODELE
# ------------------------------------------------------------------
with st.sidebar:
    st.download_button(
        label="⬇ Télécharger le modèle YOLO",
        data=open(MODEL_PATH, "rb").read(),
        file_name="yolov8s.pt",
        mime="application/octet-stream"
    )
