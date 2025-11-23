import streamlit as st
import tempfile
import numpy as np
import os
from PIL import Image
import sys

# ------------------------------------------------------------------
# CONFIGURATION APP
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Détection Poubelle", 
    layout="wide",
    page_icon="🚮"
)

st.title("🚮 Détection : Poubelle Pleine ou Vide (YOLOv8)")
st.write("Analysez une image ou une vidéo pour déterminer si une poubelle est pleine ou vide.")

# ------------------------------------------------------------------
# CHARGEMENT MODELE YOLO
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        # Vérifier si le modèle existe
        if not os.path.exists("best.pt"):
            st.error("❌ Fichier 'best.pt' non trouvé. Assurez-vous qu'il est dans le dépôt.")
            return None
        
        # Import différé pour mieux gérer les erreurs
        from ultralytics import YOLO
        model = YOLO("best.pt")
        st.sidebar.success("✅ Modèle chargé avec succès")
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {str(e)}")
        return None

# Afficher un message de chargement
with st.spinner("Chargement du modèle YOLO..."):
    model = load_model()

if model is None:
    st.error("""
    **Impossible de charger le modèle. Causes possibles :**
    - Fichier 'best.pt' manquant
    - Problème de dépendances
    - Mémoire insuffisante
    """)
    st.stop()

# ------------------------------------------------------------------
# FONCTION ANALYSE IMAGE
# ------------------------------------------------------------------
def analyze_image(img):
    try:
        results = model(img)[0]
        annotated_img = results.plot()

        # Récupérer les prédictions
        if len(results.boxes.cls) > 0:
            cls_id = int(results.boxes.cls[0])
            class_name = model.names[cls_id]
            confidence = float(results.boxes.conf[0])
            prediction_text = f"{class_name} (confiance: {confidence:.2f})"
        else:
            class_name = "Aucune détection"
            prediction_text = "Aucune poubelle détectée"

        return annotated_img, prediction_text, class_name
    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {e}")
        return img, "Erreur", "Erreur"

# ------------------------------------------------------------------
# INTERFACE UTILISATEUR
# ------------------------------------------------------------------
st.sidebar.title("📂 Options")
mode = st.sidebar.radio("Choisir le mode :", ["Image", "Vidéo"])

if mode == "Image":
    st.subheader("📸 Upload d'une image")
    uploaded_image = st.file_uploader(
        "Importer une image", 
        type=["jpg", "jpeg", "png"],
        help="Formats supportés : JPG, JPEG, PNG"
    )

    if uploaded_image is not None:
        # Affichage de l'image originale
        image = Image.open(uploaded_image)
        st.image(image, caption="Image importée", use_column_width=True)

        if st.button("🔍 Analyser l'image", type="primary"):
            with st.spinner("Analyse en cours..."):
                try:
                    # Conversion pour l'analyse
                    img_array = np.array(image)
                    if img_array.shape[-1] == 4:  # RGBA -> RGB
                        img_array = img_array[..., :3]
                    
                    annotated, prediction, class_name = analyze_image(img_array)
                    
                    st.subheader("📌 Résultat")
                    st.image(annotated, caption=f"Prédiction : {prediction}", use_column_width=True)
                    
                    # Affichage du statut CORRIGÉ
                    if "pleine" in class_name.lower():
                        st.success("🗑️ Poubelle pleine détectée")
                    elif "vide" in class_name.lower():
                        st.success("poubelle vide détectée")  # CORRECTION ICI
                    elif "Aucune" in class_name:
                        st.warning("Aucune poubelle détectée")
                        
                except Exception as e:
                    st.error(f"Erreur lors du traitement de l'image : {e}")

elif mode == "Vidéo":
    st.subheader("📹 Upload d'une vidéo")
    st.info("⚠️ L'analyse vidéo peut prendre du temps. Limitez la durée à 30 secondes maximum.")
    
    uploaded_video = st.file_uploader(
        "Importer une vidéo", 
        type=["mp4", "mov"],
        help="Formats recommandés : MP4, MOV (max 50MB)"
    )

    if uploaded_video is not None:
        # Afficher la vidéo originale
        st.video(uploaded_video)
        
        if st.button("🔍 Analyser la vidéo", type="primary"):
            with st.spinner("Analyse de la vidéo en cours... Cela peut prendre quelques minutes."):
                try:
                    # Sauvegarde temporaire
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                        temp_video.write(uploaded_video.read())
                        video_path = temp_video.name

                    # Import différé de cv2
                    import cv2
                    
                    # Lecture de la vidéo
                    cap = cv2.VideoCapture(video_path)
                    
                    if not cap.isOpened():
                        st.error("Impossible d'ouvrir la vidéo")
                        os.unlink(video_path)
                        st.stop()
                    
                    # Préparation pour l'affichage
                    st.subheader("🎬 Vidéo analysée")
                    video_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Statistiques
                    frame_count = 0
                    detections = []
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        frame_count += 1
                        status_text.text(f"Traitement de la frame {frame_count}")
                        
                        # Analyse de la frame
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        annotated_frame, prediction, class_name = analyze_image(frame_rgb)
                        
                        # Affichage de la frame annotée
                        video_placeholder.image(annotated_frame, use_column_width=True)
                        
                        # Collecte des statistiques
                        if "pleine" in class_name.lower() or "vide" in class_name.lower():
                            detections.append(class_name)
                    
                    cap.release()
                    
                    # Nettoyage
                    os.unlink(video_path)
                    
                    # Affichage des résultats
                    if detections:
                        pleines = len([d for d in detections if "pleine" in d.lower()])
                        vides = len([d for d in detections if "vide" in d.lower()])
                        
                        st.subheader("📊 Statistiques")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Poubelles pleines", pleines)
                        with col2:
                            st.metric("Poubelles vides", vides)
                    
                    st.success("✅ Analyse vidéo terminée !")
                    
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse vidéo : {e}")
                    # Nettoyage en cas d'erreur
                    if 'video_path' in locals() and os.path.exists(video_path):
                        os.unlink(video_path)

# ------------------------------------------------------------------
# INFORMATIONS
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("ℹ️ Informations")
    st.markdown("""
    **Fonctionnalités :**
    - 🗑️ Détection poubelles pleines
    - 🗑️ Détection poubelles vides
    
    **Instructions:**
    1. Choisissez Image ou Vidéo
    2. Importez votre fichier
    3. Cliquez sur Analyser
    
    **Limitations Streamlit Cloud :**
    - Vidéos max 50MB
    - Timeout après 10 minutes
    - Pas de GPU
    """)