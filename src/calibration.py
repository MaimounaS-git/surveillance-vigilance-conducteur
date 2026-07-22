"""
Phase de calibration personnalisée : mesure l'EAR et le MAR "normaux" de
l'utilisatrice au démarrage d'une session (yeux ouverts, visage neutre),
pour en déduire des seuils personnalisés plutôt que d'utiliser des seuils
fixes universels (cf. claude.md, section 7).
"""

import time

import cv2
import numpy as np

from indicateurs import calculer_ear, calculer_mar

DUREE_CALIBRATION_SECONDES = 10
RATIO_SEUIL_FERMETURE_YEUX = 0.90  # seuil de fermeture = 90% de l'EAR baseline

# Le MAR bouche fermée est proche de 0 (souvent ~0.02-0.04), donc un seuil en
# pourcentage de la baseline (ex. 140%, valeur de littérature) reste lui aussi
# proche de 0 et se déclenche au moindre mouvement de bouche. Une marge fixe
# ajoutée à la baseline est plus stable dans ce cas (ajustée empiriquement :
# mouvement de bouche normal observé ~0.16, vrai bâillement ~0.96).
MARGE_SEUIL_BAILLEMENT = 0.3


class Baseline:
    """Résultat de la calibration : moyennes mesurées et seuils personnalisés dérivés."""

    def __init__(self, ear_moyen, mar_moyen):
        self.ear_moyen = ear_moyen
        self.mar_moyen = mar_moyen
        self.seuil_fermeture_yeux = ear_moyen * RATIO_SEUIL_FERMETURE_YEUX
        self.seuil_baillement = mar_moyen + MARGE_SEUIL_BAILLEMENT


def calibrer(webcam, detecteur, duree_secondes=DUREE_CALIBRATION_SECONDES, sur_frame=None):
    """
    Exécute la phase de calibration : affiche une consigne à l'écran pendant
    `duree_secondes`, collecte l'EAR/MAR à chaque frame où un visage est
    détecté, puis retourne une Baseline calculée à partir des moyennes.

    Si `sur_frame` est fourni, il est appelé à chaque frame avec
    (frame, temps_restant) au lieu d'ouvrir une fenêtre OpenCV — utilisé par
    dashboard.py pour afficher la calibration dans le flux vidéo web plutôt
    que dans une fenêtre séparée.
    """
    valeurs_ear = []
    valeurs_mar = []
    instant_debut = time.time()

    while True:
        temps_restant = duree_secondes - (time.time() - instant_debut)
        if temps_restant <= 0:
            break

        frame = webcam.lire_frame()
        if frame is None:
            continue

        frame = cv2.flip(frame, 1)
        landmarks = detecteur.detecter(frame)

        if landmarks is not None:
            hauteur, largeur, _ = frame.shape
            ear = calculer_ear(landmarks, largeur, hauteur)
            mar = calculer_mar(landmarks, largeur, hauteur)
            valeurs_ear.append(ear)
            valeurs_mar.append(mar)

        texte = (
            f"Calibration en cours... gardez les yeux ouverts, "
            f"visage neutre ({temps_restant:.0f}s)"
        )
        cv2.putText(frame, texte, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if sur_frame is not None:
            sur_frame(frame, temps_restant)
        else:
            cv2.imshow("Surveillance de la vigilance - landmarks", frame)
            cv2.waitKey(1)

    if not valeurs_ear:
        raise RuntimeError("Calibration échouée : aucun visage détecté pendant la calibration.")

    return Baseline(ear_moyen=float(np.mean(valeurs_ear)), mar_moyen=float(np.mean(valeurs_mar)))
