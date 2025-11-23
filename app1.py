import streamlit as st
import numpy as np
import os
from PIL import Image
import tempfile

# ------------------------------------------------------------------
# CONFIGURATION APP
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Détection Poubelle", 
    layout="wide",
    page_icon="🚮"
)

st.title("🚮 Détection : Poubelle Pleine ou Vide (YOLOv8)")
st.write("Analysez une image pour déterminer si une poubelle est pleine ou vide.")

# ------------------------------------------------------------------
# CHARGEMENT MODELE YOLO
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        # Vérifier si le modèle existe
        if not os.path.exists("best.pt"):
            st.error("❌ Fichier 'best.pt' non trouvé.")
            return None
        
        # Forcer l'utilisation de PIL au lieu d'OpenCV si possible
        os.environ['ULTRALYTICS_OPENCV'] = '0'
        
        # Import différé
        from ultralytics import YOLO
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {str(e)}")
        return None

# Chargement du modèle
model = load_model()

if model is None:
    st.error("""
    **Dépannage :**
    1. Vérifiez que `best.pt` est dans votre dépôt GitHub
    2. Vérifiez votre fichier requirements.txt
    3. Redéployez l'application
    """)
    st.stop()

# ------------------------------------------------------------------
# FONCTION ANALYSE IMAGE
# ------------------------------------------------------------------
def analyze_image(image):
    try:
        # Utiliser directement l'image PIL avec Ultralytics
        results = model(image)
        
        if len(results) > 0:
            result = results[0]
            annotated_img = result.plot()
            
            # Récupérer les prédictions
            if len(result.boxes) > 0 and len(result.boxes.cls) > 0:
                cls_id = int(result.boxes.cls[0])
                class_name = model.names[cls_id]
                confidence = float(result.boxes.conf[0])
                prediction_text = f"{class_name} (confiance: {confidence:.2f})"
            else:
                class_name = "Aucune détection"
                prediction_text = "Aucune poubelle détectée"
            
            return annotated_img, prediction_text, class_name
        else:
            return image, "Aucune détection", "Aucune détection"
            
    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {e}")
        return image, "Erreur", "Erreur"

# ------------------------------------------------------------------
# INTERFACE UTILISATEUR
# ------------------------------------------------------------------
st.sidebar.title("📂 Options")

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
                annotated, prediction, class_name = analyze_image(image)
                
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
    1. Importez votre image
    2. Cliquez sur Analyser
    """)