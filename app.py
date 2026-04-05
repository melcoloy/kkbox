import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageChops
import random
import math
import io
import time
import base64
import pandas as pd
import streamlit.components.v1 as components
from scipy.optimize import linear_sum_assignment

# --- Configuration obligatoire en premier ---
st.set_page_config(page_title="Mosaïque de dominos", layout="wide")

# --- Vérification de l'installation du module de clic ---
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    st.error("⚠️ Action requise : La fonctionnalité de clic sur l'image nécessite une nouvelle bibliothèque.")
    st.info("Veuillez ouvrir un terminal et taper la commande suivante :\n\n`pip install streamlit-image-coordinates`\n\nPuis relancez l'application (Ctrl+C puis `streamlit run app.py`).")
    st.stop()

# --- Imports des autres modules du projet ---
import algorithme
import hongrois_v1
import traitement_image


# =====================================================================
# 1. FONCTIONS MATHÉMATIQUES ET ALGORITHMES
# =====================================================================

def transfo_image(image, colonnes, lignes):
    image = image.convert("L")
    image = ImageOps.fit(image, (colonnes, lignes), method=Image.Resampling.LANCZOS)
    return image

def valeurs_grille(image, type_jeu="double_six"):
    val_max = 6 if type_jeu == "double_six" else 9
    matrice_pixels = np.array(image)
    matrice_valeurs = np.round((matrice_pixels / 255.0) * val_max).astype(int)
    matrice_valeurs = val_max - matrice_valeurs 
    return matrice_valeurs

def generer_inventaire_v2(nb_dominos_necessaires, type_jeu="double_six", matrice_cibles=None):
    val_max = 6 if type_jeu == "double_six" else 9
    jeu_de_base = [(i, j) for i in range(val_max + 1) for j in range(i, val_max + 1)]
    taille_jeu = len(jeu_de_base)
    nb_jeux_complets = nb_dominos_necessaires // taille_jeu
    reste = nb_dominos_necessaires % taille_jeu

    inventaire_final = []
    for _ in range(nb_jeux_complets):
        inventaire_final.extend(jeu_de_base)
        
    if reste > 0:
        if matrice_cibles is not None:
            valeurs, clics = np.unique(matrice_cibles, return_counts=True)
            frequences = dict(zip(valeurs, clics))
            
            dominos_scores = []
            for domino in jeu_de_base:
                score = frequences.get(domino[0], 0) + frequences.get(domino[1], 0)
                dominos_scores.append((score, domino))
                
            dominos_scores.sort(key=lambda x: x[0], reverse=True)
            pieces_restantes = [domino for score, domino in dominos_scores[:reste]]
            inventaire_final.extend(pieces_restantes)
        else:
            inventaire_final.extend(random.sample(jeu_de_base, reste))
            
    return inventaire_final

def generer_emplacements(largeur_grille, hauteur_grille):
    grille_occupee = [[False] * largeur_grille for _ in range(hauteur_grille)]
    emplacements = []
    
    for y in range(hauteur_grille):
        for x in range(largeur_grille):
            if grille_occupee[y][x]: continue
                
            if x + 1 < largeur_grille and not grille_occupee[y][x + 1]:
                emplacements.append(((x, y), (x + 1, y)))
                grille_occupee[y][x] = grille_occupee[y][x + 1] = True
            elif y + 1 < hauteur_grille and not grille_occupee[y + 1][x]:
                emplacements.append(((x, y), (x, y + 1)))
                grille_occupee[y][x] = grille_occupee[y + 1][x] = True
    return emplacements

def calculer_erreur(domino, cible1, cible2):
    err_norm = abs(domino[0] - cible1) + abs(domino[1] - cible2)
    err_inv = abs(domino[1] - cible1) + abs(domino[0] - cible2)
    return (err_norm, domino) if err_norm <= err_inv else (err_inv, (domino[1], domino[0]))

def optimiser_placement_recuit(cibles, emplacements, inventaire, iterations=1e9, st_progress_bar=None):
    random.shuffle(inventaire)
    placement_actuel = list(inventaire)
    
    temp_initiale = 10.0
    temp_finale = 0.01
    alpha = (temp_finale / temp_initiale) ** (1 / iterations)
    temperature = temp_initiale
    
    step_progress = iterations // 20 
    
    for i in range(int(iterations)):
        idx1, idx2 = random.sample(range(len(emplacements)), 2)
        (x1A, y1A), (x1B, y1B) = emplacements[idx1]
        (x2A, y2A), (x2B, y2B) = emplacements[idx2]
        
        c1A, c1B = cibles[y1A, x1A], cibles[y1B, x1B]
        c2A, c2B = cibles[y2A, x2A], cibles[y2B, x2B]
        
        err1_av, _ = calculer_erreur(placement_actuel[idx1], c1A, c1B)
        err2_av, _ = calculer_erreur(placement_actuel[idx2], c2A, c2B)
        err1_ap, _ = calculer_erreur(placement_actuel[idx2], c1A, c1B)
        err2_ap, _ = calculer_erreur(placement_actuel[idx1], c2A, c2B)
        
        delta_erreur = (err1_ap + err2_ap) - (err1_av + err2_av)
        
        if delta_erreur < 0 or random.random() < math.exp(-delta_erreur / temperature):
            placement_actuel[idx1], placement_actuel[idx2] = placement_actuel[idx2], placement_actuel[idx1]
            
        temperature *= alpha
        
        if st_progress_bar and i % step_progress == 0:
            st_progress_bar.progress(i / iterations, text="Optimisation des pièces en cours...")

    placement_final = [calculer_erreur(dom, cibles[empl[0][1], empl[0][0]], cibles[empl[1][1], empl[1][0]])[1] 
                       for dom, empl in zip(placement_actuel, emplacements)]
    
    if st_progress_bar: st_progress_bar.empty()
    return placement_final

def optimiser_placement_hongrois(cibles, emplacements, inventaire, st_progress_bar=None):
    nb_emplacements = len(emplacements)
    
    if st_progress_bar:
        st_progress_bar.progress(0.1, text="Étape 1 : Calcul de la matrice des coûts (Peut prendre quelques secondes)...")
        
    matrice_couts = np.zeros((nb_emplacements, nb_emplacements), dtype=int)
    valeurs_cibles = [(cibles[y1, x1], cibles[y2, x2]) for ((x1, y1), (x2, y2)) in emplacements]
    
    for i, (cible1, cible2) in enumerate(valeurs_cibles):
        for j, domino in enumerate(inventaire):
            err_norm = abs(domino[0] - cible1) + abs(domino[1] - cible2)
            err_inv = abs(domino[1] - cible1) + abs(domino[0] - cible2)
            matrice_couts[i, j] = err_norm if err_norm < err_inv else err_inv

    if st_progress_bar:
        st_progress_bar.progress(0.5, text="Étape 2/2 : Résolution mathématique exacte...")

    row_ind, col_ind = linear_sum_assignment(matrice_couts)
    
    placement_final = []
    for i, j in enumerate(col_ind): 
        domino = inventaire[j]
        cible1, cible2 = valeurs_cibles[i]
        
        err_norm = abs(domino[0] - cible1) + abs(domino[1] - cible2)
        err_inv = abs(domino[1] - cible1) + abs(domino[0] - cible2)
        
        if err_norm <= err_inv:
            placement_final.append(domino)
        else:
            placement_final.append((domino[1], domino[0]))
            
    if st_progress_bar: 
        st_progress_bar.empty()
        
    return placement_final


# =====================================================================
# 2. INTERFACE STREAMLIT
# =====================================================================

st.title("🎲 Générateur de Mosaïque en Dominos")
st.write("Projet P4 - Par Matteo Hanon Obsomer & Clément Leroy")

# Fonction pour vider la mémoire de Streamlit (Force le recalcul de l'image)
def clear_session():
    if 'placements' in st.session_state:
        del st.session_state['placements']

# --- Barre latérale (Paramètres) ---
st.sidebar.header("Paramètres")
type_jeu = st.sidebar.radio("Type de jeu :", ("double_six", "double_neuf"), key="radio_type_jeu_main", on_change=clear_session)
nb_boites = st.sidebar.number_input("Nombre de boîtes", min_value=1, value=10, step=1, on_change=clear_session)
largeur_grille = st.sidebar.slider("Largeur (en nombre de dominos)", min_value=60, max_value=160, step=10, key="slider_largeur", on_change=clear_session)

activer_contours = st.sidebar.checkbox("Activer la segmentation (Contours)", on_change=clear_session)
activer_dithering = st.sidebar.checkbox("Activer le Dithering (Floyd-Steinberg)", value=True, on_change=clear_session)

choix_algo = st.sidebar.radio(
    "Choix de l'algorithme :", 
    ("Glouton (Rapide, par le centre)", "Hongrois (Lent, optimum mathématique)", "Méta-Heuristique (Aléatoire)","Hongrois (Matteo)"),
    key="radio_algo_main",
    on_change=clear_session
)

btn_generer = st.sidebar.button("Générer la mosaïque")

# --- Zone principale ---
col1, col2 = st.columns(2)

with col1:
    st.header("Image originale")
    choix_source = st.radio("Source de l'image :", ["📁 Importer un fichier", "📸 Prendre une photo"], key="radio_source_main", on_change=clear_session)
    fichier_upload = None
    
    if choix_source == "📁 Importer un fichier":
        fichier_upload = st.file_uploader("Chargez votre image (JPG, PNG)", type=["jpg", "jpeg", "png"], on_change=clear_session)
    else:
        fichier_upload = st.camera_input("Prenez une photo avec votre webcam", on_change=clear_session)
    
    if fichier_upload is not None:
        image_originale = Image.open(fichier_upload)
        st.image(image_originale, caption="Image importée", width=400)
        st.success("Image chargée avec succès ! Prête pour l'algo.")


with col2:
    st.header("Résultat")
    
    # ---------------------------------------------------------
    # PARTIE 1 : CALCULS LOURDS UNIQUEMENT SI ON CLIQUE "GÉNÉRER"
    # ---------------------------------------------------------
    if fichier_upload is not None and btn_generer:
        with st.spinner("Calculs et assemblage en cours..."):
            
            # Génération du stock
            stock_dominos = algorithme.generer_inventaire(type_jeu, nb_boites)
            total_dominos = len(stock_dominos)

            # Variables communes
            largeur_px, hauteur_px = image_originale.size
            ratio = hauteur_px / largeur_px
            hauteur_grille = int(largeur_grille * ratio)
            nb_dominos = (largeur_grille * hauteur_grille) // 2
            
            image_fin = transfo_image(image_originale, largeur_grille, hauteur_grille)
            matrice = valeurs_grille(image_fin, type_jeu)
            inventaire = generer_inventaire_v2(nb_dominos, type_jeu, matrice)
            emplacements = generer_emplacements(largeur_grille, hauteur_grille)
            
            # Prétraitement de l'image (avec sécurité sur l'argument 'contours')
            try:
                image_prete = traitement_image.preparer_image(image_originale, total_dominos, activer_contours)
            except TypeError:
                image_prete = traitement_image.preparer_image(image_originale, total_dominos)

            # --- CORRECTION FINALE DU DITHERING : On impose l'argument "appliquer_dithering" ---
            matrice_valeurs = traitement_image.image_vers_matrice(image_prete, type_jeu, appliquer_dithering=activer_dithering)
            
            # Lancement de l'algorithme choisi
            heure_debut = time.time() 
            my_bar = st.progress(0, text="Optimisation des pièces en cours...")
            matrice_reference = matrice_valeurs 
            
            if choix_algo == "Glouton (Rapide, par le centre)":
                placements = algorithme.placer_dominos(matrice_valeurs, stock_dominos)
            elif choix_algo == "Méta-Heuristique (Aléatoire)":
                placements_bruts = optimiser_placement_recuit(matrice, emplacements, inventaire, iterations=150000, st_progress_bar=my_bar)
                matrice_reference = matrice 
            elif choix_algo == "Hongrois (Matteo)":
                placements_bruts = optimiser_placement_hongrois(matrice, emplacements, inventaire, st_progress_bar=my_bar)
                matrice_reference = matrice 
            else:
                placements = hongrois_v1.placer_dominos(matrice_valeurs, stock_dominos)
                
            temps_execution = time.time() - heure_debut 
            
            # Conversion pour compatibilité universelle du dessin
            if choix_algo in ["Méta-Heuristique (Aléatoire)", "Hongrois (Matteo)"]:
                placements = []
                for idx, (val1, val2) in enumerate(placements_bruts):
                    (x1, y1), (x2, y2) = emplacements[idx]
                    placements.append({
                        "case1": (y1, x1), 
                        "case2": (y2, x2),
                        "valeurs": (val1, val2)
                    })
                    
            # --- SAUVEGARDE EN MÉMOIRE POUR RÉSISTER AUX CLICS ---
            st.session_state['placements'] = placements
            st.session_state['matrice_reference'] = matrice_reference
            st.session_state['image_prete'] = image_prete
            st.session_state['temps_execution'] = temps_execution


    # ---------------------------------------------------------
    # PARTIE 2 : AFFICHAGE INTERACTIF (Utilise la mémoire)
    # ---------------------------------------------------------
    if fichier_upload is not None and 'placements' in st.session_state:
        
        # Récupération des données depuis la mémoire de Streamlit
        placements = st.session_state['placements']
        matrice_reference = st.session_state['matrice_reference']
        image_prete = st.session_state['image_prete']
        temps_execution = st.session_state['temps_execution']

        st.image(image_prete, caption=f"Image N&B ajustée ({image_prete.width}x{image_prete.height} px)", width=400)
        st.success(f"🎉 Succès ! L'algorithme a placé {len(placements)} dominos (soit 100% du stock) !")
        
        # CALCUL DU SCORE
        erreur_totale = 0
        for p in placements:
            i1, j1 = p["case1"]
            i2, j2 = p["case2"]
            v1, v2 = p["valeurs"]
            erreur_totale += abs(matrice_reference[i1, j1] - v1)
            erreur_totale += abs(matrice_reference[i2, j2] - v2)
        
        valeur_max = 6 if type_jeu == "double_six" else 9
        nb_pixels = matrice_reference.size
        score_fidelite = 100 * (1 - (erreur_totale / (nb_pixels * valeur_max)))

        # AFFICHAGE DES MÉTRIQUES
        col_met1, col_met2 = st.columns(2)
        with col_met1:
            st.metric(label="🎯 Score de fidélité", value=f"{score_fidelite:.2f} %")
        with col_met2:
            st.metric(label="⏱️ Temps d'exécution", value=f"{temps_execution:.3f} s")
        
        # ---------------------------------------------------------
        # RECHERCHE VISUELLE SYNCHRONISÉE
        # ---------------------------------------------------------
        st.divider()
        st.subheader("🔍 Analyse Visuelle Interactive")
        st.write("Sélectionnez un chiffre avec les boutons **OU cliquez directement sur un domino** de l'image pour le mettre en évidence :")
        
        valeur_max_jeu = 6 if type_jeu == "double_six" else 9
        options_chiffres = ["Aucun"] + list(range(valeur_max_jeu + 1))
        
        # 1. Création de la variable d'état si elle n'existe pas
        if 'chiffre_cible' not in st.session_state:
            st.session_state['chiffre_cible'] = "Aucun"
            
        # 2. On cherche à quelle position se trouve le choix actuel pour l'afficher sur le bouton radio
        index_actuel = options_chiffres.index(st.session_state['chiffre_cible']) if st.session_state['chiffre_cible'] in options_chiffres else 0
            
        # 3. Le sélecteur lit l'index depuis la mémoire
        chiffre_selectionne = st.radio(
            "Chiffre à analyser :", 
            options_chiffres, 
            index=index_actuel,
            horizontal=True
        )
        
        # 4. Si l'utilisateur clique sur le bouton radio
        if chiffre_selectionne != st.session_state['chiffre_cible']:
            st.session_state['chiffre_cible'] = chiffre_selectionne
            st.rerun() # On recharge immédiatement la page
        
        # On détermine l'entier exact à envoyer à la fonction de dessin
        c_cible = None if st.session_state['chiffre_cible'] == "Aucun" else int(st.session_state['chiffre_cible'])

        # DESSIN INTERACTIF
        st.subheader("🖼️ Votre Mosaïque")
        lignes, colonnes = matrice_reference.shape
        taille_case = 40 # Standard de votre module
        
        # On dessine l'image avec la surbrillance choisie
        image_mosaique = traitement_image.dessiner_mosaique(placements, lignes, colonnes, taille_case=taille_case, chiffre_cible=c_cible)
        
        # On l'affiche de façon interactive (use_column_width permet d'éviter l'image rognée)
        click_data = streamlit_image_coordinates(
            image_mosaique, 
            key="mosaique_interactive",
            use_column_width=True 
        )
        
        # 5. Interception d'un clic de souris de l'utilisateur sur l'image
        if click_data is not None:
            last_click = st.session_state.get('last_click', None)
            
            # S'il s'agit d'un nouveau clic
            if click_data != last_click:
                st.session_state['last_click'] = click_data # On le mémorise
                
                # Conversion du pixel cliqué en coordonnées de grille
                x_pixel, y_pixel = click_data['x'], click_data['y']
                col_grille = x_pixel // taille_case
                row_grille = y_pixel // taille_case
                
                # Si le clic est bien à l'intérieur de la zone de dominos
                if row_grille < lignes and col_grille < colonnes:
                    chiffre_clique = int(matrice_reference[row_grille, col_grille])
                    
                    # On vérifie qu'on ne clique pas sur un chiffre déjà sélectionné
                    if st.session_state['chiffre_cible'] != chiffre_clique:
                        st.session_state['chiffre_cible'] = chiffre_clique
                        st.rerun() # Et on recharge la page

        # INVENTAIRE
        st.divider()
        st.subheader("📊 Rapport d'inventaire")
        inventaire_utilise = {}
        for placement in placements:
            v1, v2 = placement["valeurs"]
            nom_domino = f"[{min(v1, v2)} | {max(v1, v2)}]"
            inventaire_utilise[nom_domino] = inventaire_utilise.get(nom_domino, 0) + 1

        inventaire_trie = dict(sorted(inventaire_utilise.items()))

        col_tab, col_vide = st.columns([1.5, 2])
        with col_tab:
            st.dataframe(inventaire_trie, column_config={
                "index": "Type de domino",
                "value": "Quantité placée"
            })

        # TÉLÉCHARGEMENT
        st.divider()
        st.subheader("💾 Téléchargement")
        
        nom_fichier = st.text_input("Nommez votre fichier :", value="ma_mosaique_dominos")
        
        if not nom_fichier.endswith(".png"):
            nom_fichier += ".png"

        buf = io.BytesIO()
        image_mosaique.save(buf, format="PNG")
        donnees_image = buf.getvalue()
        
        st.download_button(
            label=f"📥 Télécharger : {nom_fichier}",
            data=donnees_image,
            file_name=nom_fichier,
            mime="image/png"
        )

        # IMPRESSION
        st.divider()
        st.subheader("🖨️ Impression")

        b64_image = base64.b64encode(donnees_image).decode()

        html_bouton = f"""
        <div style="text-align: left;">
            <button onclick="
                var w = window.open('');
                w.document.write('<html><head><title>Impression Mosaique</title></head><body style=\\'margin:0;display:flex;justify-content:center;align-items:center