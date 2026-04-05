import numpy as np


# --- Valeurs autorisées pour le type de jeu ---
_TYPES_JEU_VALIDES = {"double_six": 6, "double_neuf": 9}


def generer_inventaire(type_jeu="double_six", nb_boites=1):
    """
    Génère la liste des dominos (tuples) selon le jeu choisi.

    Args:
        type_jeu: "double_six" ou "double_neuf"
        nb_boites: nombre de boîtes (entier >= 1)

    Returns:
        Liste de tuples (i, j) représentant les dominos.

    Raises:
        ValueError: si type_jeu ou nb_boites sont invalides.
    """
    if type_jeu not in _TYPES_JEU_VALIDES:
        raise ValueError(
            f"Type de jeu invalide : '{type_jeu}'. "
            f"Valeurs acceptées : {list(_TYPES_JEU_VALIDES.keys())}"
        )
    if not isinstance(nb_boites, int) or nb_boites < 1:
        raise ValueError(
            f"nb_boites doit être un entier >= 1, reçu : {nb_boites!r}"
        )

    valeur_max = _TYPES_JEU_VALIDES[type_jeu]
    boite_unique = [
        (i, j)
        for i in range(valeur_max + 1)
        for j in range(i, valeur_max + 1)
    ]
    return boite_unique * nb_boites


def placer_dominos(matrice_valeurs, stock_dominos):
    """
    Algorithme 100% garanti SANS TROUS avec PRIORITÉ CENTRALE :
    1. Pavage initial trivial (tout horizontal ou vertical).
    2. Optimisation par permutations 2x2 (Swapping) pour suivre les contours.
    3. Remplissage glouton en commençant par le CENTRE de l'image.

    Args:
        matrice_valeurs: np.ndarray 2D de valeurs entières (issues de image_vers_matrice).
        stock_dominos: liste de tuples (valeur1, valeur2).

    Returns:
        Liste de dicts {"case1", "case2", "valeurs"}.

    Raises:
        ValueError: si la matrice est vide ou le stock insuffisant.
    """
    if not isinstance(matrice_valeurs, np.ndarray) or matrice_valeurs.ndim != 2 or matrice_valeurs.size == 0:
        raise ValueError("matrice_valeurs doit être un tableau numpy 2D non vide.")

    lignes, colonnes = matrice_valeurs.shape
    nb_emplacements = (lignes * colonnes) // 2

    if len(stock_dominos) < nb_emplacements:
        raise ValueError(
            f"Stock insuffisant : {len(stock_dominos)} dominos disponibles "
            f"pour {nb_emplacements} emplacements. "
            f"Augmentez le nombre de boîtes."
        )

    placements_slots = []
    grille_slots = np.zeros((lignes, colonnes), dtype=int)

    # --- ETAPE 1 : Pavage de base (Zéro trou garanti) ---
    idx = 0
    if colonnes % 2 == 0:
        for i in range(lignes):
            for j in range(0, colonnes, 2):
                placements_slots.append([(i, j), (i, j + 1)])
                grille_slots[i, j] = idx
                grille_slots[i, j + 1] = idx
                idx += 1
    else:
        for j in range(colonnes):
            for i in range(0, lignes, 2):
                placements_slots.append([(i, j), (i + 1, j)])
                grille_slots[i, j] = idx
                grille_slots[i + 1, j] = idx
                idx += 1

    # --- ETAPE 2 : Optimisation (Swapping intelligent des contours) ---
    amelioration = True
    iterations = 0
    while amelioration and iterations < 10:
        amelioration = False
        iterations += 1
        for i in range(lignes - 1):
            for j in range(colonnes - 1):
                idx1 = grille_slots[i, j]
                idx2 = grille_slots[i, j + 1]
                idx3 = grille_slots[i + 1, j]
                idx4 = grille_slots[i + 1, j + 1]

                # Cas A : 2 dominos horizontaux empilés
                if idx1 == idx2 and idx3 == idx4 and idx1 != idx3:
                    v_hg, v_hd = matrice_valeurs[i, j], matrice_valeurs[i, j + 1]
                    v_bg, v_bd = matrice_valeurs[i + 1, j], matrice_valeurs[i + 1, j + 1]
                    diff_H = abs(v_hg - v_hd) + abs(v_bg - v_bd)
                    diff_V = abs(v_hg - v_bg) + abs(v_hd - v_bd)
                    if diff_V < diff_H:
                        placements_slots[idx1] = [(i, j), (i + 1, j)]
                        placements_slots[idx3] = [(i, j + 1), (i + 1, j + 1)]
                        grille_slots[i, j] = idx1
                        grille_slots[i + 1, j] = idx1
                        grille_slots[i, j + 1] = idx3
                        grille_slots[i + 1, j + 1] = idx3
                        amelioration = True

                # Cas B : 2 dominos verticaux côte à côte
                elif idx1 == idx3 and idx2 == idx4 and idx1 != idx2:
                    v_hg, v_hd = matrice_valeurs[i, j], matrice_valeurs[i, j + 1]
                    v_bg, v_bd = matrice_valeurs[i + 1, j], matrice_valeurs[i + 1, j + 1]
                    diff_H = abs(v_hg - v_hd) + abs(v_bg - v_bd)
                    diff_V = abs(v_hg - v_bg) + abs(v_hd - v_bd)
                    if diff_H < diff_V:
                        placements_slots[idx1] = [(i, j), (i, j + 1)]
                        placements_slots[idx2] = [(i + 1, j), (i + 1, j + 1)]
                        grille_slots[i, j] = idx1
                        grille_slots[i, j + 1] = idx1
                        grille_slots[i + 1, j] = idx2
                        grille_slots[i + 1, j + 1] = idx2
                        amelioration = True

    # --- ETAPE 3 : Assignation gloutonne du stock (CENTRE D'ABORD) ---
    placements = []
    stock_restant = list(stock_dominos)  # copie défensive

    centre_i = lignes / 2.0
    centre_j = colonnes / 2.0

    def distance_au_centre(slot):
        c1, c2 = slot
        milieu_i = (c1[0] + c2[0]) / 2.0
        milieu_j = (c1[1] + c2[1]) / 2.0
        return (milieu_i - centre_i) ** 2 + (milieu_j - centre_j) ** 2

    placements_slots.sort(key=distance_au_centre)

    for slot in placements_slots:
        c1, c2 = slot
        val1 = matrice_valeurs[c1[0], c1[1]]
        val2 = matrice_valeurs[c2[0], c2[1]]

        meilleur_ecart = float("inf")
        meilleur_idx = -1
        inv = False

        for idx_dom, domino in enumerate(stock_restant):
            d1, d2 = domino
            err_norm = abs(val1 - d1) + abs(val2 - d2)
            err_inv = abs(val1 - d2) + abs(val2 - d1)
            min_err = min(err_norm, err_inv)

            if min_err < meilleur_ecart:
                meilleur_ecart = min_err
                meilleur_idx = idx_dom
                inv = err_inv < err_norm

        # Sécurité : ne devrait pas arriver si la validation ci-dessus est passée
        if meilleur_idx == -1:
            raise RuntimeError(
                f"Impossible de trouver un domino pour le slot {slot}. "
                f"Stock épuisé de manière inattendue."
            )

        domino_choisi = stock_restant.pop(meilleur_idx)
        if inv:
            domino_choisi = (domino_choisi[1], domino_choisi[0])

        placements.append({
            "case1": c1,
            "case2": c2,
            "valeurs": domino_choisi,
        })

    return placements
