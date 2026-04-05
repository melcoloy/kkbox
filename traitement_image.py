from PIL import Image, ImageDraw, ImageOps, ImageFilter
import math
import numpy as np

def preparer_image(image_originale, total_dominos, renforcer_contours=False):
    """
    Convertit l'image en noir et blanc (niveau de gris) et la redimensionne 
    en utilisant les diviseurs parfaits pour vider 100% du stock.
    """
    image_nb = image_originale.convert("L") 
    image_nb = ImageOps.autocontrast(image_nb)
    
    # Segmentation optionnelle demandée par l'utilisateur
    if renforcer_contours:
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

def image_vers_matrice(image_pil, type_jeu="double_six", appliquer_dithering=True):
    """Convertit l'image en une matrice de dominos, avec propagation d'erreur (Floyd-Steinberg)."""
    valeur_max = 6 if type_jeu == "double_six" else 9
    
    # On travaille avec des nombres à virgule (float) pour garder l'erreur
    matrice = np.array(image_pil, dtype=float)
    matrice = (matrice / 255.0) * valeur_max
    
    lignes, colonnes = matrice.shape
    
    if appliquer_dithering:
        # Algorithme de Floyd-Steinberg
        for i in range(lignes):
            for j in range(colonnes):
                ancienne_valeur = matrice[i, j]
                nouvelle_valeur = np.round(ancienne_valeur)
                # On s'assure de ne pas dépasser les limites des dominos
                nouvelle_valeur = min(valeur_max, max(0, nouvelle_valeur))
                matrice[i, j] = nouvelle_valeur
                
                # Calcul de l'erreur d'arrondi
                erreur = ancienne_valeur - nouvelle_valeur
                
                # On propage l'erreur aux voisins (selon les fractions de Floyd-Steinberg)
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
    
    # CORRECTION DE L'EFFET NÉGATIF (Dominos blancs = fond blanc)
    matrice_dominos = valeur_max - matrice_dominos
    
    return matrice_dominos

def dessiner_mosaique(placements, lignes, colonnes, taille_case=40, chiffre_cible=None):
    """Crée l'image finale avec option de mise en évidence d'un chiffre."""
    largeur_img = colonnes * taille_case
    hauteur_img = lignes * taille_case
    
    # Fond global de l'image (Gris très foncé pour faire ressortir les dominos blancs)
    couleur_fond = (40, 40, 40)
    image_finale = Image.new("RGB", (largeur_img, hauteur_img), couleur_fond)
    dessin = ImageDraw.Draw(image_finale)

    def dessiner_points(x, y, valeur):
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

        # --- LOGIQUE DE MISE EN ÉVIDENCE (Couleurs) ---
        base_fill = "white"
        if chiffre_cible is not None:
            if v1 == chiffre_cible or v2 == chiffre_cible:
                base_fill = "#FFFACD"  # Jaune très pâle (partie liée)
            else:
                base_fill = "#B0B0B0"  # Gris foncé (dominos non concernés)

        # 1. Dessiner la base du domino
        try:
            dessin.rounded_rectangle([x1_pad, y1_pad, x2_pad, y2_pad], radius=rayon_arrondi, fill=base_fill, outline="black", width=1)
        except AttributeError:
            dessin.rectangle([x1_pad, y1_pad, x2_pad, y2_pad], fill=base_fill, outline="black", width=1)

        # 2. Dessiner le surlignage vif (Scintillement) sur la case exacte
        if chiffre_cible is not None and (v1 == chiffre_cible or v2 == chiffre_cible):
            couleur_vif = "#FFD700"  # Jaune Or Vif
            
            if v1 == chiffre_cible:
                if i1 == i2: # horizontal
                    dessin.rectangle([x1_pad, y1_pad, x2, y2_pad], fill=couleur_vif)
                else: # vertical
                    dessin.rectangle([x1_pad, y1_pad, x2_pad, y2], fill=couleur_vif)
            
            if v2 == chiffre_cible:
                if i1 == i2: # horizontal
                    dessin.rectangle([x2, y1_pad, x2_pad, y2_pad], fill=couleur_vif)
                else: # vertical
                    dessin.rectangle([x1_pad, y2, x2_pad, y2_pad], fill=couleur_vif)

            # Redessiner le contour par-dessus pour cacher les débordements carrés
            try:
                dessin.rounded_rectangle([x1_pad, y1_pad, x2_pad, y2_pad], radius=rayon_arrondi, outline="black", width=1)
            except AttributeError:
                dessin.rectangle([x1_pad, y1_pad, x2_pad, y2_pad], outline="black", width=1)

        # 3. Ligne de séparation
        if i1 == i2: 
            dessin.line([x2, y1_pad, x2, y2_pad], fill="black", width=1)
        else: 
            dessin.line([x1_pad, y2, x2_pad, y2], fill="black", width=1)

        # 4. Dessiner les points
        dessiner_points(x1, y1, v1)
        dessiner_points(x2, y2, v2)

    return image_finale