import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# -------------------------------
# CONFIGURATION STREAMLIT
# -------------------------------
st.set_page_config(
    page_title="Détection Poubelle – Pleine ou Vide",
    page_icon="🗑️",
    layout="wide"
)

st.markdown("""
    <h2 style='text-align:center;color:#2C3E50;'>
        🗑️ Détection de Poubelle (YOLOv8)
    </h2>
    <p style='text-align:center;color:#7F8C8D'>
        Upload une image pour détecter la poubelle et prédire si elle est pleine ou vide.
    </p>
""", unsafe_allow_html=True)

# -------------------------------
# CHARGEMENT DU MODELE YOLO
# -------------------------------

MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Modèle introuvable ! Chargez best.pt dans le dépôt GitHub.")
    st.stop()

model = YOLO(MODEL_PATH)

# -------------------------------
# FONCTION DÉTECTION IMAGE
# -------------------------------
def detect_image(image):
    results = model(image)[0]
    annotated = results.plot()

    det_class = None
    if len(results.boxes) > 0:
        cls = int(results.boxes[0].cls[0])
        det_class = model.names[cls]

    return annotated, det_class

# -------------------------------
# UI – UPLOAD IMAGE
# -------------------------------
st.subheader("🖼️ Uploader une image")

img_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

if img_file:
    img = Image.open(img_file).convert("RGB")
    img_np = np.array(img)

    if st.button("🔍 Analyser l'image"):
        with st.spinner("Analyse en cours..."):
            annotated, det_class = detect_image(img_np)

        st.image(annotated, caption="Résultat", use_column_width=True)

        if det_class:
            if "pleine" in det_class.lower():
                st.success("🟢 Poubelle détectée — **PLEINE**")
            elif "vide" in det_class.lower():
                st.info("🔵 Poubelle détectée — **VIDE**")
            else:
                st.warning(f"Détecté : {det_class}")
        else:
            st.error("❌ Aucune poubelle détectée.")

# -------------------------------
# DOWNLOAD BUTTON
# -------------------------------
st.download_button(
    "📥 Télécharger le modèle entraîné",
    data=open(MODEL_PATH, "rb").read(),
    file_name="best.pt"
)
