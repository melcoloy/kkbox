from PIL import Image, ImageDraw, ImageOps, ImageFilter
import numpy as np
from typing import List, Dict, Tuple, Any

def preparer_image(image_originale: Image.Image, total_dominos: int, renforcer_contours: bool = False, mode_dessin: bool = False) -> Image.Image:
    """
    Convertit l'image en noir et blanc (niveau de gris) et la redimensionne.
    """
    image_nb = image_originale.convert("L") 
    image_nb = ImageOps.autocontrast(image_nb)
    
    # --- NOUVEAUTÉ : Effet Dessin (Pencil Sketch) ---
    if mode_dessin:
        # On trouve les contours purs
        image_nb = image_nb.filter(ImageFilter.FIND_EDGES)
        # On inverse (pour avoir des traits noirs sur fond blanc)
        image_nb = ImageOps.invert(image_nb)
    elif renforcer_contours:
        # Segmentation classique si le mode dessin n'est pas activé
        image_nb = image_nb.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    surface_cible = total_dominos * 2  
    
    largeur_orig, hauteur_orig = image_nb.size
    ratio_cible = largeur_orig / hauteur_orig
    
    nouvelle_largeur = surface_cible
    nouvelle_hauteur = 1
    meilleure_diff = float('inf')
    
    for h in range(1, surface_cible + 1):
        if surface_cible % h == 0:  
            l = surface_cible // h
            ratio_actuel = l / h
            diff = abs(ratio_actuel - ratio_cible)
            
            if diff < meilleure_diff:
                meilleure_diff = diff
                nouvelle_largeur = l
                nouvelle_hauteur = h
                
    image_redimensionnee = image_nb.resize((nouvelle_largeur, nouvelle_hauteur))
    return image_redimensionnee

def image_vers_matrice(image_pil: Image.Image, type_jeu: str = "double_six", appliquer_dithering: bool = True) -> np.ndarray:
    """Convertit l'image en une matrice de dominos, avec propagation d'erreur."""
    valeur_max = 6 if type_jeu == "double_six" else 9
    
    matrice = np.array(image_pil, dtype=float)
    matrice = (matrice / 255.0) * valeur_max
    
    lignes, colonnes = matrice.shape
    
    if appliquer_dithering:
        for i in range(lignes):
            for j in range(colonnes):
                ancienne_valeur = matrice[i, j]
                nouvelle_valeur = np.round(ancienne_valeur)
                nouvelle_valeur = min(valeur_max, max(0, nouvelle_valeur))
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
    
    # Correction fond blanc
    matrice_dominos = valeur_max - matrice_dominos
    
    return matrice_dominos

def dessiner_mosaique(placements: List[Dict[str, Any]], lignes: int, colonnes: int, taille_case: int = 40) -> Image.Image:
    """Crée l'image finale avec des dominos blancs séparés visuellement."""
    largeur_img = colonnes * taille_case
    hauteur_img = lignes * taille_case
    
    couleur_fond = (40, 40, 40)
    image_finale = Image.new("RGB", (largeur_img, hauteur_img), couleur_fond)
    dessin = ImageDraw.Draw(image_finale)

    def dessiner_points(x: int, y: int, valeur: int) -> None:
        marge = taille_case // 4
        r = taille_case // 10 
        cx, cy = x + taille_case//2, y + taille_case//2 
        
        pos = {
            'c': (cx, cy),
            'hg': (x + marge, y + marge), 'hd': (x + taille_case - marge, y + marge),
            'bg': (x + marge, y + taille_case - marge), 'bd': (x + taille_case - marge, y + taille_case - marge),
            'mg': (x + marge, cy), 'md': (x + taille_case - marge, cy),
            'hm': (cx, y + marge), 'bm': (cx, y + taille_case - marge)
        }
        
        pts = []
        if valeur == 1: pts = ['c']
        elif valeur == 2: pts = ['hg', 'bd']
        elif valeur == 3: pts = ['hg', 'c', 'bd']
        elif valeur == 4: pts = ['hg', 'hd', 'bg', 'bd']
        elif valeur == 5: pts = ['hg', 'hd', 'c', 'bg', 'bd']
        elif valeur == 6: pts = ['hg', 'hd', 'mg', 'md', 'bg', 'bd']
        elif valeur == 7: pts = ['hg', 'hd', 'mg', 'md', 'bg', 'bd', 'c']
        elif valeur == 8: pts = ['hg', 'hd', 'mg', 'md', 'bg', 'bd', 'hm', 'bm']
        elif valeur == 9: pts = ['hg', 'hd', 'mg', 'md', 'bg', 'bd', 'hm', 'bm', 'c']

        for p in pts:
            px, py = pos[p]
            dessin.ellipse([px-r, py-r, px+r, py+r], fill="black")

    padding = max(1, taille_case // 15) 
    rayon_arrondi = taille_case // 5    

    for p in placements:
        i1, j1 = p["case1"]
        i2, j2 = p["case2"]
        v1, v2 = p["valeurs"]

        x1, y1 = j1 * taille_case, i1 * taille_case
        x2, y2 = j2 * taille_case, i2 * taille_case

        x_min, y_min = min(x1, x2), min(y1, y2)
        x_max, y_max = max(x1, x2) + taille_case, max(y1, y2) + taille_case

        x1_pad = x_min + padding
        y1_pad = y_min + padding
        x2_pad = x_max - padding
        y2_pad = y_max - padding

        try:
            dessin.rounded_rectangle([x1_pad, y1_pad, x2_pad, y2_pad], radius=rayon_arrondi, fill="white", outline="black", width=1)
        except AttributeError:
            dessin.rectangle([x1_pad, y1_pad, x2_pad, y2_pad], fill="white", outline="black", width=1)
        
        if i1 == i2: 
            dessin.line([x2, y1_pad, x2, y2_pad], fill="black", width=1)
        else: 
            dessin.line([x1_pad, y2, x2_pad, y2], fill="black", width=1)

        dessiner_points(x1, y1, v1)
        dessiner_points(x2, y2, v2)

    return image_finale 