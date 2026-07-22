"""
Estimation de l'orientation de la tête (pitch/yaw/roll) via cv2.solvePnP,
pour détecter une tête qui tombe/dodeline (cf. claude.md, section 4
"Importantes").

Approche standard : un jeu de 6 points 3D génériques du visage (modèle
approximatif, pas propre à l'utilisatrice) mis en correspondance avec les
landmarks 2D détectés par MediaPipe, pour en déduire la pose de la tête.
"""

import math
import time

import cv2
import numpy as np

# Indices MediaPipe Face Mesh correspondant aux 6 points du modèle 3D.
# Gauche/droite inversés par rapport à la convention standard : la frame est
# retournée (effet miroir, cf. main.py) avant la détection des landmarks.
INDICE_NEZ = 1
INDICE_MENTON = 152
INDICE_OEIL_DROIT_EXTERNE = 263
INDICE_OEIL_GAUCHE_EXTERNE = 33
INDICE_BOUCHE_GAUCHE = 291
INDICE_BOUCHE_DROITE = 61

# Modèle 3D générique du visage (coordonnées approximatives en mm, repère
# centré sur le nez) : valeurs standard largement utilisées dans la
# littérature pour l'estimation de pose par solvePnP.
POINTS_MODELE_3D = np.array([
    (0.0, 0.0, 0.0),           # Nez
    (0.0, -330.0, -65.0),      # Menton
    (-225.0, 170.0, -135.0),   # Coin externe oeil droit
    (225.0, 170.0, -135.0),    # Coin externe oeil gauche
    (-150.0, -150.0, -125.0),  # Coin bouche gauche
    (150.0, -150.0, -125.0),   # Coin bouche droite
], dtype=np.float64)

SEUIL_PITCH_TETE_BASSE = 20  # degrés
DUREE_TETE_BASSE_SECONDES = 1.0


def _point_pixel(landmark, largeur, hauteur):
    return (landmark.x * largeur, landmark.y * hauteur)


def _matrice_vers_angles(matrice_rotation):
    """
    Convertit une matrice de rotation 3x3 en angles d'Euler (pitch, yaw, roll)
    en degrés (formule standard, convention X-Y-Z). Plus robuste que
    cv2.RQDecomp3x3 pour les poses proches du visage de face : cette dernière
    peut renvoyer une représentation "retournée" (proche de +/-180°),
    mathématiquement équivalente mais inutilisable pour un seuil simple.
    """
    sy = math.sqrt(matrice_rotation[0, 0] ** 2 + matrice_rotation[1, 0] ** 2)
    singulier = sy < 1e-6

    if not singulier:
        x = math.atan2(matrice_rotation[2, 1], matrice_rotation[2, 2])
        y = math.atan2(-matrice_rotation[2, 0], sy)
        z = math.atan2(matrice_rotation[1, 0], matrice_rotation[0, 0])
    else:
        x = math.atan2(-matrice_rotation[1, 2], matrice_rotation[1, 1])
        y = math.atan2(-matrice_rotation[2, 0], sy)
        z = 0.0

    return math.degrees(x), math.degrees(y), math.degrees(z)


def estimer_orientation(landmarks, largeur, hauteur):
    """
    Retourne (pitch, yaw, roll) en degrés à partir des landmarks du visage,
    ou None si l'estimation échoue. Pitch positif = tête penchée vers le bas.
    """
    points_image = np.array([
        _point_pixel(landmarks[INDICE_NEZ], largeur, hauteur),
        _point_pixel(landmarks[INDICE_MENTON], largeur, hauteur),
        _point_pixel(landmarks[INDICE_OEIL_DROIT_EXTERNE], largeur, hauteur),
        _point_pixel(landmarks[INDICE_OEIL_GAUCHE_EXTERNE], largeur, hauteur),
        _point_pixel(landmarks[INDICE_BOUCHE_GAUCHE], largeur, hauteur),
        _point_pixel(landmarks[INDICE_BOUCHE_DROITE], largeur, hauteur),
    ], dtype=np.float64)

    # Caméra approximée (pas de calibration caméra dédiée) : focale ~= largeur de l'image
    focale = largeur
    matrice_camera = np.array([
        [focale, 0, largeur / 2],
        [0, focale, hauteur / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    coeffs_distorsion = np.zeros((4, 1))

    succes, vecteur_rotation, _ = cv2.solvePnP(
        POINTS_MODELE_3D, points_image, matrice_camera, coeffs_distorsion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not succes:
        return None

    matrice_rotation, _ = cv2.Rodrigues(vecteur_rotation)
    pitch, yaw, roll = _matrice_vers_angles(matrice_rotation)
    return pitch, yaw, roll


class DetecteurTeteBasse:
    """Détecte une tête penchée vers le bas de façon soutenue (pas juste un instant)."""

    def __init__(self, seuil_pitch=SEUIL_PITCH_TETE_BASSE, duree_secondes=DUREE_TETE_BASSE_SECONDES):
        self.seuil_pitch = seuil_pitch
        self.duree_secondes = duree_secondes
        self.instant_debut = None

    def mettre_a_jour(self, pitch, instant=None):
        """Retourne True si la tête est penchée vers le bas depuis plus de `duree_secondes`."""
        instant = instant if instant is not None else time.time()
        tete_basse = abs(pitch) > self.seuil_pitch

        if not tete_basse:
            self.instant_debut = None
            return False

        if self.instant_debut is None:
            self.instant_debut = instant

        return instant - self.instant_debut >= self.duree_secondes
