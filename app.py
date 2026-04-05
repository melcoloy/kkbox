"""
app.py — Interface Streamlit du générateur de mosaïques en dominos.
Toute la logique métier est dans le package `core/`.
"""
import io
import base64
import time

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from core.inventaire import generer_stock, completer_inventaire, valeur_max
from core.image import preparer_image, image_vers_matrice, dessiner_mosaique
from core.algorithmes import glouton, hongrois, recuit, calculer_score, LIMITE_HONGROIS

# ── Import conditionnel : clic interactif sur la mosaïque ─────────────
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    _CLIC_DISPONIBLE = True
except ImportError:
    _CLIC_DISPONIBLE = False

# ── Noms affichés → fonctions algo ────────────────────────────────────
ALGOS = {
    "Glouton (Rapide, par le centre)":        "glouton",
    "Hongrois (Lent, optimum mathématique)":  "hongrois",
    "Méta-Heuristique (Recuit simulé)":       "recuit",
}

# =====================================================================
# Mise en page
# =====================================================================

st.set_page_config(page_title="Mosaïque de dominos", layout="wide")
st.title("🎲 Générateur de Mosaïque en Dominos")
st.write("Projet P4 — Matteo Hanon Obsomer & Clément Leroy")

if not _CLIC_DISPONIBLE:
    st.info("💡 Installez `streamlit-image-coordinates` pour activer le clic interactif sur la mosaïque.")

# ── Barre latérale ────────────────────────────────────────────────────
st.sidebar.header("Paramètres")
type_jeu      = st.sidebar.radio("Type de jeu :", ("double_six", "double_neuf"), key="type_jeu")
nb_boites     = st.sidebar.number_input("Nombre de boîtes", min_value=1, value=10, step=1)
largeur_grille = st.sidebar.slider("Largeur (dominos)", min_value=60, max_value=160, step=10)
activer_contours  = st.sidebar.checkbox("Segmentation des contours")
activer_dithering = st.sidebar.checkbox("Dithering Floyd-Steinberg", value=True)
choix_algo    = st.sidebar.radio("Algorithme :", list(ALGOS.keys()), key="algo")
btn_generer   = st.sidebar.button("Générer la mosaïque")

# ── Colonnes principales ──────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.header("Image originale")
    source = st.radio("Source :", ["📁 Importer un fichier", "📸 Webcam"], key="source")
    fichier = (
        st.file_uploader("Image (JPG, PNG)", type=["jpg", "jpeg", "png"])
        if source == "📁 Importer un fichier"
        else st.camera_input("Prendre une photo")
    )
    if fichier:
        image_originale = Image.open(fichier)
        st.image(image_originale, caption="Image importée", width=400)
        st.success("Image chargée !")

with col2:
    st.header("Résultat")

    # ── Calcul (uniquement au clic "Générer") ─────────────────────────
    if fichier and btn_generer:
        try:
            with st.spinner("Calculs en cours..."):

                # Stock
                stock = generer_stock(type_jeu, nb_boites)
                vmax  = valeur_max(type_jeu)

                # Prétraitement image
                image_prete    = preparer_image(image_originale, len(stock), activer_contours)
                matrice_valeurs = image_vers_matrice(image_prete, type_jeu, activer_dithering)

                # Inventaire adapté à la taille de la grille
                lignes_g  = image_prete.height
                colonnes_g = image_prete.width
                nb_emplacements = (lignes_g * colonnes_g) // 2
                inventaire = completer_inventaire(nb_emplacements, type_jeu, matrice_valeurs)

                # Garde-fou Hongrois
                if ALGOS[choix_algo] in ("hongrois",) and nb_emplacements > LIMITE_HONGROIS:
                    st.error(
                        f"⚠️ Grille trop grande pour l'algorithme Hongrois "
                        f"({nb_emplacements} emplacements > limite {LIMITE_HONGROIS}). "
                        "Réduisez la largeur ou utilisez le Glouton."
                    )
                    st.stop()

                # Lancement
                my_bar = st.progress(0, text="Optimisation en cours...")

                def progress(ratio, texte):
                    my_bar.progress(ratio, text=texte)

                debut = time.time()

                if ALGOS[choix_algo] == "glouton":
                    placements = glouton(matrice_valeurs, stock)
                    my_bar.empty()

                elif ALGOS[choix_algo] == "hongrois":
                    placements = hongrois(matrice_valeurs, inventaire, progress_callback=progress)

                else:  # recuit
                    placements = recuit(matrice_valeurs, inventaire, progress_callback=progress)

                temps = time.time() - debut

                # Sauvegarde session
                st.session_state.update({
                    "placements":        placements,
                    "matrice_reference": matrice_valeurs,
                    "image_prete":       image_prete,
                    "temps":             temps,
                    "type_jeu":          type_jeu,
                    "vmax":              vmax,
                })

        except ValueError as e:
            st.error(f"❌ Paramètre invalide : {e}")
            st.stop()
        except MemoryError:
            st.error("❌ Mémoire insuffisante. Réduisez la taille de la grille.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Erreur inattendue : {e}")
            st.stop()

    # ── Affichage (depuis la session, résiste aux interactions) ───────
    if fichier and "placements" in st.session_state:
        placements  = st.session_state["placements"]
        matrice_ref = st.session_state["matrice_reference"]
        image_prete = st.session_state["image_prete"]
        temps       = st.session_state["temps"]
        vmax        = st.session_state["vmax"]
        type_jeu_s  = st.session_state["type_jeu"]

        st.image(image_prete, caption=f"N&B ({image_prete.width}×{image_prete.height} px)", width=400)
        st.success(f"🎉 {len(placements)} dominos placés !")

        # Métriques
        score = calculer_score(placements, matrice_ref, vmax)
        c1, c2 = st.columns(2)
        c1.metric("🎯 Fidélité", f"{score:.2f} %")
        c2.metric("⏱️ Durée", f"{temps:.3f} s")

        if score > 90:
            st.write("✨ *Excellent ! La ressemblance est quasi-parfaite.*")
        elif score > 75:
            st.write("👍 *Bon résultat, les formes principales sont bien respectées.*")
        else:
            st.write("⚠️ *Le stock était peut-être trop limité pour cette image.*")

        # ── Analyse interactive ────────────────────────────────────────
        st.divider()
        st.subheader("🔍 Analyse Visuelle Interactive")

        options = ["Aucun"] + list(range(vmax + 1))
        if "chiffre_cible" not in st.session_state:
            st.session_state["chiffre_cible"] = "Aucun"

        idx_actuel = options.index(st.session_state["chiffre_cible"]) if st.session_state["chiffre_cible"] in options else 0
        choix = st.radio("Chiffre à analyser :", options, index=idx_actuel, horizontal=True)

        if choix != st.session_state["chiffre_cible"]:
            st.session_state["chiffre_cible"] = choix
            st.rerun()

        c_cible = None if st.session_state["chiffre_cible"] == "Aucun" else int(st.session_state["chiffre_cible"])
        taille_case = 40
        lignes_r, colonnes_r = matrice_ref.shape
        image_mosaique = dessiner_mosaique(placements, lignes_r, colonnes_r, taille_case, c_cible)

        st.subheader("🖼️ Votre Mosaïque")
        click_data = None
        if _CLIC_DISPONIBLE:
            click_data = streamlit_image_coordinates(image_mosaique, key="mosaique", use_column_width=True)
        else:
            st.image(image_mosaique, use_container_width=True)

        if click_data and click_data != st.session_state.get("last_click"):
            st.session_state["last_click"] = click_data
            col_g = click_data["x"] // taille_case
            row_g = click_data["y"] // taille_case
            if row_g < lignes_r and col_g < colonnes_r:
                chiffre = int(matrice_ref[row_g, col_g])
                if st.session_state["chiffre_cible"] != chiffre:
                    st.session_state["chiffre_cible"] = chiffre
                    st.rerun()

        # ── Rapport d'inventaire ───────────────────────────────────────
        st.divider()
        st.subheader("📊 Rapport d'inventaire")
        inventaire_utilise = {}
        for p in placements:
            v1, v2 = p["valeurs"]
            cle = f"[{min(v1,v2)} | {max(v1,v2)}]"
            inventaire_utilise[cle] = inventaire_utilise.get(cle, 0) + 1

        col_tab, _ = st.columns([1.5, 2])
        with col_tab:
            st.dataframe(dict(sorted(inventaire_utilise.items())))

        # ── Téléchargement ─────────────────────────────────────────────
        st.divider()
        st.subheader("💾 Téléchargement")
        nom = st.text_input("Nom du fichier :", value="ma_mosaique_dominos")
        if not nom.endswith(".png"):
            nom += ".png"

        buf = io.BytesIO()
        image_mosaique.save(buf, format="PNG")
        donnees = buf.getvalue()
        st.download_button(f"📥 Télécharger : {nom}", data=donnees, file_name=nom, mime="image/png")

        # ── Impression ─────────────────────────────────────────────────
        st.divider()
        st.subheader("🖨️ Impression")
        b64 = base64.b64encode(donnees).decode()
        components.html(f"""
        <div>
            <button onclick="
                var w=window.open('');
                w.document.write('<html><body style=\\'margin:0;display:flex;justify-content:center;align-items:center;height:100vh;\\'><img src=\\'data:image/png;base64,{b64}\\' style=\\'max-width:100%;max-height:100%;\\'></body></html>');
                w.document.close(); w.focus();
                setTimeout(function(){{w.print();w.close();}},500);
            " style="background:#fff;color:#31333F;padding:10px 24px;border:1px solid #dcdcdc;border-radius:8px;cursor:pointer;font-size:16px;font-family:sans-serif;"
            onmouseover="this.style.borderColor='#FF4B4B';this.style.color='#FF4B4B';"
            onmouseout="this.style.borderColor='#dcdcdc';this.style.color='#31333F';">
                🖨️ Lancer l'impression
            </button>
        </div>
        """, height=60)