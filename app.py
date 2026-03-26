import streamlit as st
from PIL import Image 
import io
import time
import base64
import streamlit.components.v1 as components
import algorithme
import algorithme_hongrois
import traitement_image

st.set_page_config(page_title="Mosaïque de dominos", layout="wide")
st.title("🎲 Générateur de Mosaïque en Dominos")
st.write("Projet P4 - Par Matteo Hanon Obsomer & Clément Leroy")

# --- Barre latérale ---
st.sidebar.header("Paramètres")
type_jeu = st.sidebar.radio("Type de jeu :", ("double_six", "double_neuf"))
nb_boites = st.sidebar.number_input("Nombre de boîtes", min_value=1, value=10, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Options avancées de traitement")

# Nouveauté : Mode Dessin !
mode_dessin = st.sidebar.checkbox("✏️ Transformer l'image en Dessin (Esquisse)", value=False)
activer_contours = st.sidebar.checkbox("Activer la segmentation (Renforcer les contours)", value=False)
activer_dithering = st.sidebar.checkbox("Activer le Dithering (Algorithme Floyd-Steinberg)", value=True)

st.sidebar.markdown("---")
choix_algo = st.sidebar.radio(
    "Choix de l'algorithme :", 
    ("Glouton (Rapide, par le centre)", "Hongrois (Lent, optimum mathématique)")
)

btn_generer = st.sidebar.button("Générer la mosaïque")

# --- Zone principale ---
col1, col2 = st.columns(2)

with col1:
    st.header("Image originale")
    choix_source = st.radio("Source de l'image :", ["📁 Importer un fichier", "📸 Prendre une photo"])
    
    fichier_upload = None
    if choix_source == "📁 Importer un fichier":
        fichier_upload = st.file_uploader("Chargez votre image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    else:
        fichier_upload = st.camera_input("Prenez une photo avec votre webcam")
    
    if fichier_upload is not None:
        image_originale = Image.open(fichier_upload)
        st.image(image_originale, caption="Image importée", width=400)
        st.success("Image chargée avec succès ! Prête pour l'algo.")

with col2:
    st.header("Résultat")
    
    if fichier_upload is not None and btn_generer:
        with st.spinner("Calculs et assemblage en cours..."):
            
            # 1. Génération du stock
            stock_dominos = algorithme.generer_inventaire(type_jeu, nb_boites)
            total_dominos = len(stock_dominos)
            
            # 2. Prétraitement de l'image (avec le nouveau mode_dessin)
            image_prete = traitement_image.preparer_image(
                image_originale, total_dominos, activer_contours, mode_dessin
            )
            st.image(image_prete, caption=f"Aperçu du traitement ({image_prete.width}x{image_prete.height} px)", width=400)
            
            # 3. Conversion en matrice
            matrice_valeurs = traitement_image.image_vers_matrice(image_prete, type_jeu, activer_dithering)
            
            # 4. Chronomètre et Algorithme
            heure_debut = time.time() 
            if choix_algo == "Glouton (Rapide, par le centre)":
                placements = algorithme.placer_dominos(matrice_valeurs, stock_dominos)
            else:
                placements = algorithme_hongrois.placer_dominos(matrice_valeurs, stock_dominos)
            temps_execution = time.time() - heure_debut 
                
            st.success(f"🎉 Succès ! {len(placements)} dominos placés.")
            
            # --- CALCUL DU SCORE DE FIDÉLITÉ ---
            erreur_totale = 0
            for p in placements:
                i1, j1 = p["case1"]
                i2, j2 = p["case2"]
                v1, v2 = p["valeurs"]
                erreur_totale += abs(matrice_valeurs[i1, j1] - v1)
                erreur_totale += abs(matrice_valeurs[i2, j2] - v2)
            
            valeur_max = 6 if type_jeu == "double_six" else 9
            nb_pixels = matrice_valeurs.size
            score_fidelite = 100 * (1 - (erreur_totale / (nb_pixels * valeur_max)))

            # --- AFFICHAGE MÉTRIQUES ---
            col_met1, col_met2 = st.columns(2)
            with col_met1:
                st.metric(label="🎯 Score de fidélité", value=f"{score_fidelite:.2f} %")
            with col_met2:
                st.metric(label="⏱️ Temps d'exécution", value=f"{temps_execution:.3f} s")
            
            # 5. Rendu Visuel
            st.subheader("🖼️ Votre Mosaïque")
            lignes, colonnes = matrice_valeurs.shape
            image_mosaique = traitement_image.dessiner_mosaique(placements, lignes, colonnes)
            st.image(image_mosaique, caption="Mosaïque générée avec succès !", use_container_width=True)
            st.balloons() 

            # 6. Inventaire
            st.divider()
            st.subheader("📊 Rapport d'inventaire")
            inventaire_utilise = {}
            for p in placements:
                v1, v2 = p["valeurs"]
                nom_domino = f"[{min(v1, v2)} | {max(v1, v2)}]"
                inventaire_utilise[nom_domino] = inventaire_utilise.get(nom_domino, 0) + 1

            inventaire_trie = dict(sorted(inventaire_utilise.items()))
            col_tab, col_vide = st.columns([1, 2])
            with col_tab:
                st.dataframe(inventaire_trie, column_config={"index": "Type", "value": "Quantité"})

            # 7. Téléchargement
            st.divider()
            st.subheader("💾 Sauvegarde & Impression")
            nom_fichier = st.text_input("Nommez votre fichier :", value="ma_mosaique_dominos")
            if not nom_fichier.endswith(".png"): nom_fichier += ".png"

            buf = io.BytesIO()
            image_mosaique.save(buf, format="PNG")
            donnees_image = buf.getvalue()
            
            st.download_button(label=f"📥 Télécharger", data=donnees_image, file_name=nom_fichier, mime="image/png")
            
            # 8. Impression (JavaScript)
            b64_image = base64.b64encode(donnees_image).decode()
            html_bouton = f"""
            <div style="text-align: left; margin-top: 10px;">
                <button onclick="
                    var w = window.open('');
                    w.document.write('<html><head><title>Impression Mosaique</title></head><body style=\\'margin:0;display:flex;justify-content:center;align-items:center;height:100vh;\\'><img src=\\'data:image/png;base64,{b64_image}\\' style=\\'max-width:100%;max-height:100%;\\'></body></html>');
                    w.document.close(); w.focus(); setTimeout(function() {{ w.print(); w.close(); }}, 500);
                " style="background-color: #ffffff; color: #31333F; padding: 10px 24px; border: 1px solid #dcdcdc; border-radius: 8px; cursor: pointer; font-size: 16px; transition: 0.3s;"
                onmouseover="this.style.borderColor='#FF4B4B'; this.style.color='#FF4B4B';"
                onmouseout="this.style.borderColor='#dcdcdc'; this.style.color='#31333F';">
                    🖨️ Lancer l'impression
                </button>
            </div>
            """
            components.html(html_bouton, height=60)