"""
Calcul des indicateurs géométriques EAR (ouverture des yeux) et MAR (ouverture
de la bouche) à partir des landmarks du visage détectés par MediaPipe.

Formules (cf. claude.md, section 6) :
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    MAR = même logique appliquée aux landmarks de la bouche
"""

import numpy as np

from landmarks import LANDMARKS_OEIL_GAUCHE, LANDMARKS_OEIL_DROIT, LANDMARKS_BOUCHE


def _point_pixel(landmark, largeur, hauteur):
    """Convertit un landmark normalisé (0-1) en coordonnées pixel."""
    return np.array([landmark.x * largeur, landmark.y * hauteur])


def _distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def _ratio_ouverture(landmarks, indices, largeur, hauteur):
    """
    Calcule un ratio d'ouverture générique à partir de 6 points :
    [coin_gauche, haut1, haut2, coin_droit, bas1, bas2]
    ratio = (||haut1-bas1|| + ||haut2-bas2||) / (2 * ||coin_gauche-coin_droit||)
    """
    coin_gauche, haut1, haut2, coin_droit, bas1, bas2 = [
        _point_pixel(landmarks[i], largeur, hauteur) for i in indices
    ]
    distance_verticale = _distance(haut1, bas1) + _distance(haut2, bas2)
    distance_horizontale = _distance(coin_gauche, coin_droit)
    if distance_horizontale == 0:
        return 0.0
    return distance_verticale / (2 * distance_horizontale)


def calculer_ear(landmarks, largeur, hauteur):
    """Calcule l'EAR moyen des deux yeux (moyenne des deux côtés du visage)."""
    ear_gauche = _ratio_ouverture(landmarks, LANDMARKS_OEIL_GAUCHE, largeur, hauteur)
    ear_droit = _ratio_ouverture(landmarks, LANDMARKS_OEIL_DROIT, largeur, hauteur)
    return (ear_gauche + ear_droit) / 2


def calculer_mar(landmarks, largeur, hauteur):
    """
    Calcule le MAR (ouverture de la bouche) : une seule paire verticale
    (lèvre haute / lèvre basse, au centre), contrairement à l'EAR qui utilise
    deux paires. MAR = ||haut-bas|| / ||coin_gauche-coin_droit||
    """
    coin_gauche, coin_droit, haut, bas = [
        _point_pixel(landmarks[i], largeur, hauteur) for i in LANDMARKS_BOUCHE
    ]
    distance_verticale = _distance(haut, bas)
    distance_horizontale = _distance(coin_gauche, coin_droit)
    if distance_horizontale == 0:
        return 0.0
    return distance_verticale / distance_horizontale
