import numpy as np

def generer_inventaire(type_jeu="double_six", nb_boites=1):
    """Génère la liste des dominos (tuples) selon le jeu choisi."""
    valeur_max = 6 if type_jeu == "double_six" else 9
    boite_unique = []
    
    for i in range(valeur_max + 1):
        for j in range(i, valeur_max + 1):
            boite_unique.append((i, j))
            
    jeu_complet = boite_unique * nb_boites
    return jeu_complet

def danger_composante_impaire(i1, j1, i2, j2, couvert, lignes, colonnes):
    """
    Simule la pose d'un domino et utilise un BFS (Parcours en largeur) 
    pour vérifier si on ne crée pas une zone vide de taille IMPAIRE 
    (qui serait mathématiquement impossible à remplir).
    """
    # 1. On simule la pose
    couvert[i1, j1] = True
    couvert[i2, j2] = True
    
    # On crée une copie de la grille pour le scan
    visite = np.copy(couvert)
    danger = False
    
    # 2. On regarde les cases autour du domino qu'on vient de poser
    voisins = [(i1-1, j1), (i1+1, j1), (i1, j1-1), (i1, j1+1),
               (i2-1, j2), (i2+1, j2), (i2, j2-1), (i2, j2+1)]
               
    for r, c in voisins:
        if 0 <= r < lignes and 0 <= c < colonnes and not visite[r, c]:
            # --- DÉBUT DU BFS (Scan de la zone vide) ---
            taille_zone = 0
            file = [(r, c)]
            visite[r, c] = True
            
            idx = 0
            while idx < len(file):
                curr_r, curr_c = file[idx]
                idx += 1
                taille_zone += 1
                
                # On regarde les 4 directions
                for vr, vc in [(curr_r-1, curr_c), (curr_r+1, curr_c), (curr_r, curr_c-1), (curr_r, curr_c+1)]:
                    if 0 <= vr < lignes and 0 <= vc < colonnes and not visite[vr, vc]:
                        visite[vr, vc] = True
                        file.append((vr, vc))
            # --- FIN DU BFS ---
            
            # Si la zone vide qu'on a isolée a une taille impaire = Trous garantis plus tard !
            if taille_zone % 2 != 0:
                danger = True
                break
                
    # 3. On annule la pose simulée
    couvert[i1, j1] = False
    couvert[i2, j2] = False
    
    return danger

def placer_dominos(matrice_valeurs, stock_dominos):
    """
    Algorithme Glouton avec Anticipation Topologique (BFS)
    """
    lignes, colonnes = matrice_valeurs.shape
    couvert = np.zeros((lignes, colonnes), dtype=bool)
    placements = []
    stock_restant = stock_dominos.copy()
    
    for i in range(lignes):
        for j in range(colonnes):
            if couvert[i, j]:
                continue
                
            valeur_1 = matrice_valeurs[i, j]
            
            peut_H = (j + 1 < colonnes) and not couvert[i, j + 1]
            peut_V = (i + 1 < lignes) and not couvert[i + 1, j]
            
            # --- LE MÉCANISME D'ANTICIPATION PUISSANT ---
            danger_H = danger_composante_impaire(i, j, i, j+1, couvert, lignes, colonnes) if peut_H else True
            danger_V = danger_composante_impaire(i, j, i+1, j, couvert, lignes, colonnes) if peut_V else True
            
            meilleur_ecart_H, meilleur_idx_H, inv_H = 999, -1, False
            meilleur_ecart_V, meilleur_idx_V, inv_V = 999, -1, False
            
            for index, domino in enumerate(stock_restant):
                d1, d2 = domino
                
                if peut_H:
                    v2_H = matrice_valeurs[i, j + 1]
                    err_H_norm = abs(valeur_1 - d1) + abs(v2_H - d2)
                    err_H_inv = abs(valeur_1 - d2) + abs(v2_H - d1)
                    min_err_H = min(err_H_norm, err_H_inv)
                    if min_err_H < meilleur_ecart_H:
                        meilleur_ecart_H = min_err_H
                        meilleur_idx_H = index
                        inv_H = (err_H_inv < err_H_norm)
                
                if peut_V:
                    v2_V = matrice_valeurs[i + 1, j]
                    err_V_norm = abs(valeur_1 - d1) + abs(v2_V - d2)
                    err_V_inv = abs(valeur_1 - d2) + abs(v2_V - d1)
                    min_err_V = min(err_V_norm, err_V_inv)
                    if min_err_V < meilleur_ecart_V:
                        meilleur_ecart_V = min_err_V
                        meilleur_idx_V = index
                        inv_V = (err_V_inv < err_V_norm)

            # --- LE CHOIX DE L'ORIENTATION ---
            if peut_H and peut_V:
                # Priorité absolue à la sécurité géométrique
                if danger_H and not danger_V:
                    choix = 'V'
                elif danger_V and not danger_H:
                    choix = 'H'
                else:
                    # Si aucun danger, on prend la meilleure orientation pour l'image
                    choix = 'H' if meilleur_ecart_H <= meilleur_ecart_V else 'V'
            elif peut_H:
                choix = 'H'
            elif peut_V:
                choix = 'V'
            else:
                # Si on est vraiment coincé, on ignore la case (rare avec ce nouvel algorithme)
                continue
                
            if choix == 'H':
                i2, j2 = i, j + 1
                best_idx = meilleur_idx_H
                is_inv = inv_H
            else:
                i2, j2 = i + 1, j
                best_idx = meilleur_idx_V
                is_inv = inv_V
                
            domino_choisi = stock_restant.pop(best_idx)
            if is_inv:
                domino_choisi = (domino_choisi[1], domino_choisi[0])
                
            placements.append({
                "case1": (i, j),
                "case2": (i2, j2),
                "valeurs": domino_choisi
            })
            
            couvert[i, j] = True
            couvert[i2, j2] = True
            
    return placements