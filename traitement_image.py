from PIL import Image, ImageDraw, ImageOps, ImageFilter
import numpy as np


# --- Taille minimale raisonnable pour éviter les images dégénérées ---
_DIMENSION_MIN = 2


def preparer_image(image_originale, total_dominos, renforcer_contours=False):
    """
    Convertit l'image en noir et blanc et la redimensionne pour utiliser
    exactement total_dominos dominos (2 pixels par domino).

    Args:
        image_originale: objet PIL.Image (n'importe quel mode).
        total_dominos: nombre entier de dominos >= 1.
        renforcer_contours: bool, active le filtre EDGE_ENHANCE_MORE.

    Returns:
        PIL.Image en mode "L" (niveaux de gris), redimensionnée.

    Raises:
        ValueError: si l'image est invalide ou total_dominos < 1.
        TypeError: si image_originale n'est pas un objet PIL.Image.
    """
    if not isinstance(image_originale, Image.Image):
        raise TypeError(
            f"image_originale doit être un objet PIL.Image, reçu : {type(image_originale).__name__}"
        )
    if not isinstance(total_dominos, int) or total_dominos < 1:
        raise ValueError(
            f"total_dominos doit être un entier >= 1, reçu : {total_dominos!r}"
        )

    image_nb = image_originale.convert("L")
    image_nb = ImageOps.autocontrast(image_nb)

    if renforcer_contours:
        image_nb = image_nb.filter(ImageFilter.EDGE_ENHANCE_MORE)

    surface_cible = total_dominos * 2
    largeur_orig, hauteur_orig = image_nb.size

    # Sécurité : image source dégénérée
    if largeur_orig < _DIMENSION_MIN or hauteur_orig < _DIMENSION_MIN:
        raise ValueError(
            f"Image trop petite ({largeur_orig}×{hauteur_orig} px). "
            f"Dimensions minimales : {_DIMENSION_MIN}×{_DIMENSION_MIN}."
        )

    ratio_cible = largeur_orig / hauteur_orig
    nouvelle_largeur = surface_cible
    nouvelle_hauteur = 1
    meilleure_diff = float("inf")

    for h in range(1, surface_cible + 1):
        if surface_cible % h == 0:
            l = surface_cible // h
            # On écarte les dimensions dégénérées (trop étroites ou trop hautes)
            if l < _DIMENSION_MIN or h < _DIMENSION_MIN:
                continue
            ratio_actuel = l / h
            diff = abs(ratio_actuel - ratio_cible)
            if diff < meilleure_diff:
                meilleure_diff = diff
                nouvelle_largeur = l
                nouvelle_hauteur = h

    # Sécurité : si aucune dimension valide n'a été trouvée (total_dominos très petit)
    if nouvelle_hauteur < _DIMENSION_MIN or nouvelle_largeur < _DIMENSION_MIN:
        raise ValueError(
            f"Impossible de trouver des dimensions valides pour {total_dominos} dominos. "
            f"Augmentez le nombre de boîtes."
        )

    image_redimensionnee = image_nb.resize((nouvelle_largeur, nouvelle_hauteur))
    return image_redimensionnee


def image_vers_matrice(image_pil, type_jeu="double_six", appliquer_dithering=True):
    """
    Convertit une image PIL en niveaux de gris en matrice de valeurs de dominos,
    avec propagation d'erreur optionnelle (Floyd-Steinberg).

    Args:
        image_pil: PIL.Image en mode "L" (niveaux de gris).
        type_jeu: "double_six" ou "double_neuf".
        appliquer_dithering: bool.

    Returns:
        np.ndarray 2D d'entiers dans [0, valeur_max].

    Raises:
        ValueError: si type_jeu est invalide ou image non en niveaux de gris.
        TypeError: si image_pil n'est pas un objet PIL.Image.
    """
    if not isinstance(image_pil, Image.Image):
        raise TypeError(
            f"image_pil doit être un objet PIL.Image, reçu : {type(image_pil).__name__}"
        )

    # Conversion silencieuse si l'image n'est pas déjà en niveaux de gris
    if image_pil.mode != "L":
        image_pil = image_pil.convert("L")

    types_valides = {"double_six": 6, "double_neuf": 9}
    if type_jeu not in types_valides:
        raise ValueError(
            f"type_jeu invalide : '{type_jeu}'. "
            f"Valeurs acceptées : {list(types_valides.keys())}"
        )

    valeur_max = types_valides[type_jeu]
    matrice = np.array(image_pil, dtype=float)
    matrice = (matrice / 255.0) * valeur_max

    lignes, colonnes = matrice.shape

    if appliquer_dithering:
        for i in range(lignes):
            for j in range(colonnes):
                ancienne_valeur = matrice[i, j]
                nouvelle_valeur = float(np.round(ancienne_valeur))
                nouvelle_valeur = min(valeur_max, max(0.0, nouvelle_valeur))
                matrice[i, j] = nouvelle_valeur

                erreur = ancienne_valeur - nouvelle_valeur

                if j + 1 < colonnes:
                    matrice[i, j + 1] += erreur * (7 / 16)
                if i + 1 < lignes:
                    if j - 1 >= 0:
                        matrice[i + 1, j - 1] += erreur * (3 / 16)
                    matrice[i + 1, j] += erreur * (5 / 16)
                    if j + 1 < colonnes:
                        matrice[i + 1, j + 1] += erreur * (1 / 16)
    else:
        matrice = np.round(matrice)

    matrice_dominos = matrice.astype(int)
    matrice_dominos = np.clip(matrice_dominos, 0, valeur_max)

    # Inversion : dominos blancs = fond blanc
    matrice_dominos = valeur_max - matrice_dominos

    return matrice_dominos


def dessiner_mosaique(placements, lignes, colonnes, taille_case=40, chiffre_cible=None):
    """
    Crée l'image finale avec des dominos dessinés sur fond sombre.

    Args:
        placements: liste de dicts {"case1", "case2", "valeurs"}.
        lignes: nombre de lignes de la grille.
        colonnes: nombre de colonnes de la grille.
        taille_case: taille en pixels d'une case (défaut 40).
        chiffre_cible: int ou None — si fourni, met en surbrillance
                       les dominos contenant ce chiffre.

    Returns:
        PIL.Image RGB de la mosaïque.

    Raises:
        ValueError: si placements est vide ou dimensions invalides.
    """
    if not placements:
        raise ValueError("La liste de placements est vide.")
    if lignes < 1 or colonnes < 1:
        raise ValueError(f"Dimensions invalides : {lignes}×{colonnes}.")
    if taille_case < 10:
        raise ValueError(f"taille_case trop petite : {taille_case} (minimum 10).")

    largeur_img = colonnes * taille_case
    hauteur_img = lignes * taille_case

    couleur_fond = (40, 40, 40)
    image_finale = Image.new("RGB", (largeur_img, hauteur_img), couleur_fond)
    dessin = ImageDraw.Draw(image_finale)

    def dessiner_points(x, y, valeur, couleur_pip="black"):
        marge = taille_case // 4
        r = taille_case // 10
        cx, cy = x + taille_case // 2, y + taille_case // 2

        pos = {
            "c":  (cx, cy),
            "hg": (x + marge,              y + marge),
            "hd": (x + taille_case - marge, y + marge),
            "bg": (x + marge,              y + taille_case - marge),
            "bd": (x + taille_case - marge, y + taille_case - marge),
            "mg": (x + marge,              cy),
            "md": (x + taille_case - marge, cy),
            "hm": (cx,                      y + marge),
            "bm": (cx,                      y + taille_case - marge),
        }

        disposition = {
            0: [],
            1: ["c"],
            2: ["hg", "bd"],
            3: ["hg", "c", "bd"],
            4: ["hg", "hd", "bg", "bd"],
            5: ["hg", "hd", "c", "bg", "bd"],
            6: ["hg", "hd", "mg", "md", "bg", "bd"],
            7: ["hg", "hd", "mg", "md", "bg", "bd", "c"],
            8: ["hg", "hd", "mg", "md", "bg", "bd", "hm", "bm"],
            9: ["hg", "hd", "mg", "md", "bg", "bd", "hm", "bm", "c"],
        }

        pts = disposition.get(valeur, [])
        for p in pts:
            px, py = pos[p]
            dessin.ellipse([px - r, py - r, px + r, py + r], fill=couleur_pip)

    padding = max(1, taille_case // 15)
    rayon_arrondi = taille_case // 5

    for p in placements:
        i1, j1 = p["case1"]
        i2, j2 = p["case2"]
        v1, v2 = p["valeurs"]

        # Surbrillance si le chiffre cible est présent dans ce domino
        en_surbrillance = (
            chiffre_cible is not None and chiffre_cible in (v1, v2)
        )
        couleur_domino = (255, 255, 180) if en_surbrillance else "white"
        couleur_pip = (180, 0, 0) if en_surbrillance else "black"

        x1, y1 = j1 * taille_case, i1 * taille_case
        x2, y2 = j2 * taille_case, i2 * taille_case

        x_min, y_min = min(x1, x2), min(y1, y2)
        x_max, y_max = max(x1, x2) + taille_case, max(y1, y2) + taille_case

        x1_pad = x_min + padding
        y1_pad = y_min + padding
        x2_pad = x_max - padding
        y2_pad = y_max - padding

        try:
            dessin.rounded_rectangle(
                [x1_pad, y1_pad, x2_pad, y2_pad],
                radius=rayon_arrondi,
                fill=couleur_domino,
                outline="black",
                width=1,
            )
        except AttributeError:
            # Pillow < 8.2 : pas de rounded_rectangle
            dessin.rectangle(
                [x1_pad, y1_pad, x2_pad, y2_pad],
                fill=couleur_domino,
                outline="black",
                width=1,
            )

        if i1 == i2:  # Domino horizontal
            dessin.line([x2, y1_pad, x2, y2_pad], fill="black", width=1)
        else:  # Domino vertical
            dessin.line([x1_pad, y2, x2_pad, y2], fill="black", width=1)

        dessiner_points(x1, y1, v1, couleur_pip)
        dessiner_points(x2, y2, v2, couleur_pip)

    return image_finale
