import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
import math
import io
import time
import base64
import streamlit.components.v1 as components
from scipy.optimize import linear_sum_assignment

# --- Import conditionnel : dégrade proprement si bibliothèque manquante ---
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    _CLIC_DISPONIBLE = True
except ImportError:
    _CLIC_DISPONIBLE = False

# --- Imports des autres modules du projet ---
try:
    import algorithme
except ImportError:
    st.error("❌ Module manquant : 'algorithme.py' introuvable dans le répertoire.")
    st.stop()

try:
    import hongrois_v1
except ImportError:
    st.error("❌ Module manquant : 'hongrois_v1.py' introuvable dans le répertoire.")
    st.stop()

try:
    import traitement_image
except ImportError:
    st.error("❌ Module manquant : 'traitement_image.py' introuvable dans le répertoire.")
    st.stop()

# --- Constantes ---
_TYPES_JEU = {"double_six": 6, "double_neuf": 9}
_LIMITE_HONGROIS = 5000  # nb d'emplacements max avant d'avertir l'utilisateur


# =====================================================================
# 1. FONCTIONS MATHÉMATIQUES ET ALGORITHMES
# =====================================================================

def transfo_image(image, colonnes, lignes):
    image = image.convert("L")
    image = ImageOps.fit(image, (colonnes, lignes), method=Image.Resampling.LANCZOS)
    return image


def valeurs_grille(image, type_jeu="double_six"):
    val_max = _TYPES_JEU[type_jeu]
    matrice_pixels = np.array(image)
    matrice_valeurs = np.round((matrice_pixels / 255.0) * val_max).astype(int)
    matrice_valeurs = val_max - matrice_valeurs
    return matrice_valeurs


def generer_inventaire_v2(nb_dominos_necessaires, type_jeu="double_six", matrice_cibles=None):
    val_max = _TYPES_JEU[type_jeu]
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
            if grille_occupee[y][x]:
                continue
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
    step_progress = max(1, int(iterations // 20))

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

    placement_final = [
        calculer_erreur(dom, cibles[empl[0][1], empl[0][0]], cibles[empl[1][1], empl[1][0]])[1]
        for dom, empl in zip(placement_actuel, emplacements)
    ]

    if st_progress_bar:
        st_progress_bar.empty()
    return placement_final


def optimiser_placement_hongrois(cibles, emplacements, inventaire, st_progress_bar=None):
    nb_emplacements = len(emplacements)

    if st_progress_bar:
        st_progress_bar.progress(0.1, text="Étape 1/2 : Calcul de la matrice des coûts...")

    matrice_couts = np.zeros((nb_emplacements, nb_emplacements), dtype=int)
    valeurs_cibles = [(cibles[y1, x1], cibles[y2, x2]) for ((x1, y1), (x2, y2)) in emplacements]

    for i, (cible1, cible2) in enumerate(valeurs_cibles):
        for j, domino in enumerate(inventaire):
            err_norm = abs(domino[0] - cible1) + abs(domino[1] - cible2)
            err_inv = abs(domino[1] - cible1) + abs(domino[0] - cible2)
            matrice_couts[i, j] = min(err_norm, err_inv)

    if st_progress_bar:
        st_progress_bar.progress(0.5, text="Étape 2/2 : Résolution mathématique exacte...")

    row_ind, col_ind = linear_sum_assignment(matrice_couts)

    placement_final = []
    for i, j in enumerate(col_ind):
        domino = inventaire[j]
        cible1, cible2 = valeurs_cibles[i]
        err_norm = abs(domino[0] - cible1) + abs(domino[1] - cible2)
        err_inv = abs(domino[1] - cible1) + abs(domino[0] - cible2)
        placement_final.append(domino if err_norm <= err_inv else (domino[1], domino[0]))

    if st_progress_bar:
        st_progress_bar.empty()
    return placement_final


def convertir_en_dicts(placements_bruts, emplacements):
    """Convertit la liste de tuples (val1, val2) en liste de dicts universels."""
    return [
        {
            "case1": (y1, x1),
            "case2": (y2, x2),
            "valeurs": (val1, val2),
        }
        for (val1, val2), ((x1, y1), (x2, y2)) in zip(placements_bruts, emplacements)
    ]


def calculer_score_fidelite(placements, matrice_reference, type_jeu):
    """Retourne le score de fidélité en % entre les placements et la matrice cible."""
    valeur_max = _TYPES_JEU[type_jeu]
    erreur_totale = sum(
        abs(matrice_reference[p["case1"][0], p["case1"][1]] - p["valeurs"][0]) +
        abs(matrice_reference[p["case2"][0], p["case2"][1]] - p["valeurs"][1])
        for p in placements
    )
    nb_pixels = matrice_reference.size
    return 100 * (1 - erreur_totale / (nb_pixels * valeur_max))


# =====================================================================
# 2. INTERFACE STREAMLIT
# =====================================================================

st.set_page_config(page_title="Mosaïque de dominos", layout="wide")
st.title("🎲 Générateur de Mosaïque en Dominos")
st.write("Projet P4 - Par Matteo Hanon Obsomer & Clément Leroy")

if not _CLIC_DISPONIBLE:
    st.info(
        "💡 Pour activer le clic interactif sur la mosaïque, installez : "
        "`pip install streamlit-image-coordinates`"
    )

# --- Barre latérale ---
st.sidebar.header("Paramètres")
type_jeu = st.sidebar.radio("Type de jeu :", ("double_six", "double_neuf"), key="radio_type_jeu_main")
nb_boites = st.sidebar.number_input("Nombre de boîtes", min_value=1, value=10, step=1)
largeur_grille = st.sidebar.slider("Largeur (en nombre de dominos)", min_value=60, max_value=160, step=10, key="slider_largeur")
activer_contours = st.sidebar.checkbox("Activer la segmentation (Contours)")
activer_dithering = st.sidebar.checkbox("Activer le Dithering (Floyd-Steinberg)", value=True)
choix_algo = st.sidebar.radio(
    "Choix de l'algorithme :",
    ("Glouton (Rapide, par le centre)", "Hongrois (Lent, optimum mathématique)", "Méta-Heuristique (Aléatoire)", "Hongrois (Matteo)"),
    key="radio_algo_main",
)
btn_generer = st.sidebar.button("Générer la mosaïque")

# --- Zone principale ---
col1, col2 = st.columns(2)

with col1:
    st.header("Image originale")
    choix_source = st.radio("Source de l'image :", ["📁 Importer un fichier", "📸 Prendre une photo"], key="radio_source_main")

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

    # ─────────────────────────────────────────────────────────────────
    # PARTIE 1 : CALCULS (uniquement au clic sur "Générer")
    # ─────────────────────────────────────────────────────────────────
    if fichier_upload is not None and btn_generer:
        try:
            with st.spinner("Calculs et assemblage en cours..."):

                # 1. Génération du stock
                stock_dominos = algorithme.generer_inventaire(type_jeu, nb_boites)
                total_dominos = len(stock_dominos)

                # 2. Variables communes
                largeur_px, hauteur_px = image_originale.size
                ratio = hauteur_px / largeur_px
                hauteur_grille = int(largeur_grille * ratio)
                nb_dominos = (largeur_grille * hauteur_grille) // 2
                nb_emplacements = nb_dominos  # identique ici

                image_fin = transfo_image(image_originale, largeur_grille, hauteur_grille)
                matrice = valeurs_grille(image_fin, type_jeu)
                inventaire = generer_inventaire_v2(nb_dominos, type_jeu, matrice)
                emplacements = generer_emplacements(largeur_grille, hauteur_grille)

                # 3. Garde-fou Hongrois : matrice trop grande = crash mémoire
                if choix_algo in ("Hongrois (Lent, optimum mathématique)", "Hongrois (Matteo)"):
                    if nb_emplacements > _LIMITE_HONGROIS:
                        st.error(
                            f"⚠️ Grille trop grande pour l'algorithme Hongrois "
                            f"({nb_emplacements} emplacements > limite {_LIMITE_HONGROIS}). "
                            f"Réduisez la largeur ou utilisez l'algorithme Glouton."
                        )
                        st.stop()

                # 4. Prétraitement image
                image_prete = traitement_image.preparer_image(
                    image_originale, total_dominos, activer_contours
                )

                # 5. Conversion en matrice
                matrice_valeurs = traitement_image.image_vers_matrice(
                    image_prete, type_jeu, appliquer_dithering=activer_dithering
                )

                # 6. Lancement de l'algorithme choisi
                heure_debut = time.time()
                my_bar = st.progress(0, text="Optimisation des pièces en cours...")
                matrice_reference = matrice_valeurs

                if choix_algo == "Glouton (Rapide, par le centre)":
                    placements = algorithme.placer_dominos(matrice_valeurs, stock_dominos)
                    my_bar.empty()

                elif choix_algo == "Méta-Heuristique (Aléatoire)":
                    placements_bruts = optimiser_placement_recuit(
                        matrice, emplacements, inventaire, iterations=150000, st_progress_bar=my_bar
                    )
                    placements = convertir_en_dicts(placements_bruts, emplacements)
                    matrice_reference = matrice

                elif choix_algo == "Hongrois (Matteo)":
                    placements_bruts = optimiser_placement_hongrois(
                        matrice, emplacements, inventaire, st_progress_bar=my_bar
                    )
                    placements = convertir_en_dicts(placements_bruts, emplacements)
                    matrice_reference = matrice

                else:  # Hongrois (Lent, optimum mathématique)
                    placements = hongrois_v1.placer_dominos(matrice_valeurs, stock_dominos)
                    my_bar.empty()

                temps_execution = time.time() - heure_debut

                # 7. Sauvegarde en session pour résister aux interactions
                st.session_state["placements"] = placements
                st.session_state["matrice_reference"] = matrice_reference
                st.session_state["image_prete"] = image_prete
                st.session_state["temps_execution"] = temps_execution
                st.session_state["type_jeu"] = type_jeu

        except ValueError as e:
            st.error(f"❌ Paramètres invalides : {e}")
            st.stop()
        except MemoryError:
            st.error("❌ Mémoire insuffisante. Réduisez la taille de la grille ou le nombre de boîtes.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Erreur inattendue : {e}")
            st.stop()

    # ─────────────────────────────────────────────────────────────────
    # PARTIE 2 : AFFICHAGE (utilise la session, résiste aux clics)
    # ─────────────────────────────────────────────────────────────────
    if fichier_upload is not None and "placements" in st.session_state:

        placements = st.session_state["placements"]
        matrice_reference = st.session_state["matrice_reference"]
        image_prete = st.session_state["image_prete"]
        temps_execution = st.session_state["temps_execution"]
        type_jeu_affiche = st.session_state.get("type_jeu", type_jeu)

        st.image(
            image_prete,
            caption=f"Image N&B ajustée ({image_prete.width}×{image_prete.height} px)",
            width=400,
        )
        st.success(f"🎉 Succès ! L'algorithme a placé {len(placements)} dominos (soit 100% du stock) !")

        # Score de fidélité
        score_fidelite = calculer_score_fidelite(placements, matrice_reference, type_jeu_affiche)

        col_met1, col_met2 = st.columns(2)
        with col_met1:
            st.metric(label="🎯 Score de fidélité", value=f"{score_fidelite:.2f} %")
        with col_met2:
            st.metric(label="⏱️ Temps d'exécution", value=f"{temps_execution:.3f} s")

        if score_fidelite > 90:
            st.write("✨ *Excellent ! La ressemblance est quasi-parfaite.*")
        elif score_fidelite > 75:
            st.write("👍 *Bon résultat, les formes principales sont bien respectées.*")
        else:
            st.write("⚠️ *Le stock de dominos était peut-être trop limité pour cette image.*")

        # ── Analyse visuelle interactive ──────────────────────────────
        st.divider()
        st.subheader("🔍 Analyse Visuelle Interactive")
        st.write(
            "Sélectionnez un chiffre avec les boutons "
            + ("**OU cliquez directement sur un domino** " if _CLIC_DISPONIBLE else "")
            + "de l'image pour le mettre en évidence :"
        )

        valeur_max_jeu = _TYPES_JEU[type_jeu_affiche]
        options_chiffres = ["Aucun"] + list(range(valeur_max_jeu + 1))

        if "chiffre_cible" not in st.session_state:
            st.session_state["chiffre_cible"] = "Aucun"

        index_actuel = (
            options_chiffres.index(st.session_state["chiffre_cible"])
            if st.session_state["chiffre_cible"] in options_chiffres
            else 0
        )

        chiffre_selectionne = st.radio(
            "Chiffre à analyser :", options_chiffres, index=index_actuel, horizontal=True
        )

        if chiffre_selectionne != st.session_state["chiffre_cible"]:
            st.session_state["chiffre_cible"] = chiffre_selectionne
            st.rerun()

        c_cible = None if st.session_state["chiffre_cible"] == "Aucun" else int(st.session_state["chiffre_cible"])

        # ── Dessin de la mosaïque ──────────────────────────────────────
        st.subheader("🖼️ Votre Mosaïque")
        lignes, colonnes = matrice_reference.shape
        taille_case = 40

        image_mosaique = traitement_image.dessiner_mosaique(
            placements, lignes, colonnes, taille_case=taille_case, chiffre_cible=c_cible
        )

        # Affichage interactif (avec clic si disponible, sinon image simple)
        click_data = None
        if _CLIC_DISPONIBLE:
            click_data = streamlit_image_coordinates(
                image_mosaique, key="mosaique_interactive", use_column_width=True
            )
        else:
            st.image(image_mosaique, use_container_width=True)

        # Interception du clic sur la mosaïque
        if click_data is not None:
            last_click = st.session_state.get("last_click", None)
            if click_data != last_click:
                st.session_state["last_click"] = click_data
                x_pixel, y_pixel = click_data["x"], click_data["y"]
                col_grille = x_pixel // taille_case
                row_grille = y_pixel // taille_case
                if row_grille < lignes and col_grille < colonnes:
                    chiffre_clique = int(matrice_reference[row_grille, col_grille])
                    if st.session_state["chiffre_cible"] != chiffre_clique:
                        st.session_state["chiffre_cible"] = chiffre_clique
                        st.rerun()

        # ── Rapport d'inventaire ───────────────────────────────────────
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
                "value": "Quantité placée",
            })

        # ── Téléchargement ─────────────────────────────────────────────
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
            mime="image/png",
        )

        # ── Impression ─────────────────────────────────────────────────
        st.divider()
        st.subheader("🖨️ Impression")
        st.write("Vous pouvez imprimer directement votre mosaïque depuis votre navigateur :")

        b64_image = base64.b64encode(donnees_image).decode()
        html_bouton = f"""
        <div style="text-align: left;">
            <button onclick="
                var w = window.open('');
                w.document.write('<html><head><title>Impression Mosaique</title></head><body style=\\'margin:0;display:flex;justify-content:center;align-items:center;height:100vh;\\'><img src=\\'data:image/png;base64,{b64_image}\\' style=\\'max-width:100%;max-height:100%;\\'></body></html>');
                w.document.close();
                w.focus();
                setTimeout(function() {{ w.print(); w.close(); }}, 500);
            " style="background-color:#ffffff;color:#31333F;padding:10px 24px;border:1px solid #dcdcdc;border-radius:8px;cursor:pointer;font-size:16px;font-family:sans-serif;transition:0.3s;"
            onmouseover="this.style.borderColor='#FF4B4B';this.style.color='#FF4B4B';"
            onmouseout="this.style.borderColor='#dcdcdc';this.style.color='#31333F';">
                🖨️ Lancer l'impression
            </button>
        </div>
        """
        components.html(html_bouton, height=60)