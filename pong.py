import pygame
import sys
import random

# Initialisation de Pygame
pygame.init()
horloge = pygame.time.Clock()

# Paramètres de la fenêtre
LARGEUR, HAUTEUR = 800, 500
ecran = pygame.display.set_mode((LARGEUR, HAUTEUR))
pygame.display.set_caption("Pong : Humain vs IA")

# Couleurs
NOIR = (20, 20, 20)
BLANC = (255, 255, 255)
BLEU_ELECTRIQUE = (0, 191, 255)
ROUGE_FLASH = (255, 69, 0)

# Objets du jeu
balle = pygame.Rect(LARGEUR//2 - 10, HAUTEUR//2 - 10, 20, 20)
joueur = pygame.Rect(LARGEUR - 20, HAUTEUR//2 - 70, 10, 120)
ia = pygame.Rect(10, HAUTEUR//2 - 70, 10, 120)

# Vitesses et Scores
vitesse_x = 6 * random.choice((1, -1))
vitesse_y = 6 * random.choice((1, -1))
score_ia = 0
score_joueur = 0
vitesse_ia = 6  # Tu peux augmenter ce chiffre pour rendre l'IA imbattable
police = pygame.font.SysFont("Arial", 60)

def reset_balle():
    global vitesse_x, vitesse_y
    balle.center = (LARGEUR//2, HAUTEUR//2)
    vitesse_x = 6 * random.choice((1, -1))
    vitesse_y = 6 * random.choice((1, -1))

# Boucle principale
while True:
    # 1. Gestion des événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # 2. Contrôles du Joueur (Flèches Haut/Bas)
    touches = pygame.key.get_pressed()
    if touches[pygame.K_UP] and joueur.top > 0:
        joueur.y -= 8
    if touches[pygame.K_DOWN] and joueur.bottom < HAUTEUR:
        joueur.y += 8

    # 3. Intelligence Artificielle (Joueur de gauche)
    # L'IA suit le centre de la balle avec un petit délai de réaction
    if ia.centery < balle.y:
        ia.y += vitesse_ia
    if ia.centery > balle.y:
        ia.y -= vitesse_ia

    # Empêcher l'IA de sortir de l'écran
    ia.clamp_ip(ecran.get_rect())

    # 4. Mouvement de la balle
    balle.x += vitesse_x
    balle.y += vitesse_y

    # 5. Collisions Murs (Haut/Bas)
    if balle.top <= 0 or balle.bottom >= HAUTEUR:
        vitesse_y *= -1

    # 6. Collisions Raquettes + Physique
    # Sur le joueur (droite)
    if balle.colliderect(joueur) and vitesse_x > 0:
        vitesse_x = -(vitesse_x * 1.05) # Accélération de 5%
        vitesse_y += random.uniform(-3, 3) # Angle aléatoire
    
    # Sur l'IA (gauche)
    if balle.colliderect(ia) and vitesse_x < 0:
        vitesse_x = -(vitesse_x * 1.05)
        vitesse_y += random.uniform(-3, 3)

    # 7. Système de Score
    if balle.left <= 0:
        score_joueur += 1
        reset_balle()
    if balle.right >= LARGEUR:
        score_ia += 1
        reset_balle()

    # 8. Affichage
    ecran.fill(NOIR)
    
    # Dessin du score
    surface_score = police.render(f"{score_ia}   {score_joueur}", True, BLANC)
    ecran.blit(surface_score, (LARGEUR//2 - 55, 30))
    
    # Ligne centrale
    pygame.draw.aaline(ecran, BLANC, (LARGEUR//2, 0), (LARGEUR//2, HAUTEUR))
    
    # Les éléments du jeu
    pygame.draw.rect(ecran, ROUGE_FLASH, ia)
    pygame.draw.rect(ecran, BLEU_ELECTRIQUE, joueur)
    pygame.draw.ellipse(ecran, BLANC, balle)

    # Mise à jour de l'écran
    pygame.display.flip()
    horloge.tick(60)