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
# CHARGEMENT MODELE YOLO AVEC GESTION D'ERREURS CORRIGÉE
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        # Vérifier si le modèle existe
        if not os.path.exists("best.pt"):
            st.error("❌ Fichier 'best.pt' non trouvé. Assurez-vous qu'il est dans le dépôt.")
            return None
        
        # Forcer l'utilisation de opencv-python-headless
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
        
        # Import différé pour mieux gérer les erreurs
        from ultralytics import YOLO
        model = YOLO("best.pt")
        st.sidebar.success("✅ Modèle chargé avec succès")
        return model
    except ImportError as e:
        if "libGL.so.1" in str(e):
            st.error("""
            **Erreur de dépendance OpenCV**
            
            Solution : Ajoutez `opencv-python-headless` à votre fichier requirements.txt :
            ```
            opencv-python-headless
            ```
            """)
        else:
            st.error(f"❌ Erreur d'import : {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {str(e)}")
        return None

# Afficher un message de chargement
with st.spinner("Chargement du modèle YOLO..."):
    model = load_model()

if model is None:
    st.error("""
    **Impossible de charger le modèle. Solutions possibles :**
    
    1. **Vérifiez le fichier requirements.txt** :
    ```txt
    streamlit
    ultralytics
    opencv-python-headless
    imageio
    imageio-ffmpeg
    numpy
    Pillow
    ```
    
    2. **Vérifiez que 'best.pt' est présent** dans le dépôt GitHub
    3. **Redéployez l'application** après ces modifications
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
# FONCTION ANALYSE VIDÉO AVEC IMAGEIO (SANS CV2)
# ------------------------------------------------------------------
def detect_video(video_file):
    """Analyse la vidéo sans cv2 (compatible Streamlit Cloud)."""
    try:
        import imageio.v3 as iio
        
        # Sauvegarde temporaire
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.close()

        # Lecture de la vidéo avec imageio
        video_reader = iio.imiter(tfile.name, plugin="pyav")
        
        output_frames = []
        frame_count = 0
        
        # Création d'une placeholder pour la progression
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for frame in video_reader:
            frame_count += 1
            status_text.text(f"Traitement de la frame {frame_count}")
            
            # Mise à jour de la barre de progression (estimation)
            if frame_count % 5 == 0:
                progress_bar.progress(min(frame_count / 50, 1.0))
            
            # Analyse de la frame
            results = model(frame)[0]
            annotated = results.plot()
            output_frames.append(annotated)
        
        # Nettoyage
        os.unlink(tfile.name)
        
        progress_placeholder.empty()
        progress_bar.empty()
        status_text.empty()
        
        return output_frames
    
    except Exception as e:
        st.error(f"Erreur lors de l'analyse de la vidéo : {e}")
        # Nettoyage en cas d'erreur
        if 'tfile' in locals() and os.path.exists(tfile.name):
            os.unlink(tfile.name)
        return []

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
                    
                    # Affichage du statut
                    if "pleine" in class_name.lower():
                        st.success("🗑️ Poubelle pleine détectée")
                    elif "vide" in class_name.lower():
                        st.success("poubelle vide détectée")
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
        help="Formats recommandés : MP4, MOV"
    )

    if uploaded_video is not None:
        # Afficher la vidéo originale
        st.video(uploaded_video)
        
        if st.button("🔍 Analyser la vidéo", type="primary"):
            with st.spinner("Préparation de l'analyse..."):
                try:
                    # Réinitialiser le curseur du fichier
                    uploaded_video.seek(0)
                    
                    # Analyser la vidéo avec imageio
                    output_frames = detect_video(uploaded_video)
                    
                    if output_frames:
                        st.subheader("🎬 Résultat de l'analyse")
                        st.success(f"✅ Analyse terminée ! {len(output_frames)} frames traitées")
                        
                        # Afficher quelques frames résultats
                        st.info("Quelques frames annotées :")
                        cols = st.columns(3)
                        for i, frame in enumerate(output_frames[:6]):
                            if i < 6:
                                cols[i % 3].image(frame, use_column_width=True)
                    
                    else:
                        st.error("❌ Aucun résultat obtenu de l'analyse vidéo")
                        
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse vidéo : {e}")

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
    """)