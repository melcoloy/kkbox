import pygame
import time
import random

# Initialisation de pygame
pygame.init()

# Couleurs
BLANC = (255, 255, 255)
JAUNE = (255, 255, 102)
NOIR = (0, 0, 0)
ROUGE = (213, 50, 80)
VERT = (0, 255, 0)

# Dimensions de la fenêtre
LARGEUR = 600
HAUTEUR = 400
fenetre = pygame.display.set_mode((LARGEUR, HAUTEUR))
pygame.display.set_caption('Mon Super Snake Python')

# Paramètres du jeu
TAILLE_BLOC = 20
VITESSE = 15
horloge = pygame.time.Clock()

font_style = pygame.font.SysFont("bahnschrift", 25)

def message(msg, couleur):
    mesg = font_style.render(msg, True, couleur)
    fenetre.blit(mesg, [LARGEUR / 6, HAUTEUR / 3])

def jeu():
    game_over = False
    game_close = False

    # Position initiale du serpent
    x1 = LARGEUR / 2
    y1 = HAUTEUR / 2

    x1_changement = 0
    y1_changement = 0

    serpent_liste = []
    longueur_serpent = 1

    # Position de la nourriture
    nourriture_x = round(random.randrange(0, LARGEUR - TAILLE_BLOC) / 20.0) * 20.0
    nourriture_y = round(random.randrange(0, HAUTEUR - TAILLE_BLOC) / 20.0) * 20.0

    while not game_over:

        while game_close:
            fenetre.fill(NOIR)
            message("Perdu ! Appuie sur C pour rejouer ou Q pour quitter", ROUGE)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        jeu()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and x1_changement == 0:
                    x1_changement = -TAILLE_BLOC
                    y1_changement = 0
                elif event.key == pygame.K_RIGHT and x1_changement == 0:
                    x1_changement = TAILLE_BLOC
                    y1_changement = 0
                elif event.key == pygame.K_UP and y1_changement == 0:
                    y1_changement = -TAILLE_BLOC
                    x1_changement = 0
                elif event.key == pygame.K_DOWN and y1_changement == 0:
                    y1_changement = TAILLE_BLOC
                    x1_changement = 0

        # Collision avec les murs
        if x1 >= LARGEUR or x1 < 0 or y1 >= HAUTEUR or y1 < 0:
            game_close = True
        
        x1 += x1_changement
        y1 += y1_changement
        fenetre.fill(NOIR)
        
        # Dessiner la nourriture
        pygame.draw.rect(fenetre, VERT, [nourriture_x, nourriture_y, TAILLE_BLOC, TAILLE_BLOC])
        
        # Logique du corps du serpent
        serpent_tete = [x1, y1]
        serpent_liste.append(serpent_tete)
        if len(serpent_liste) > longueur_serpent:
            del serpent_liste[0]

        # Collision avec soi-même
        for x in serpent_liste[:-1]:
            if x == serpent_tete:
                game_close = True

        for bloc in serpent_liste:
            pygame.draw.rect(fenetre, JAUNE, [bloc[0], bloc[1], TAILLE_BLOC, TAILLE_BLOC])

        pygame.display.update()

        # Manger la nourriture
        if x1 == nourriture_x and y1 == nourriture_y:
            nourriture_x = round(random.randrange(0, LARGEUR - TAILLE_BLOC) / 20.0) * 20.0
            nourriture_y = round(random.randrange(0, HAUTEUR - TAILLE_BLOC) / 20.0) * 20.0
            longueur_serpent += 1

        horloge.tick(VITESSE)

    pygame.quit()
    quit()

jeu()