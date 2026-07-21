"""
Enregistre une session de test scriptée (protocole minuté) pour évaluer
ensuite le seuil fixe vs le seuil personnalisé (cf. evaluer.py).

Protocole :
1. Calibration (10s) -> seuil personnalisé de cette session.
2. Phase ouverture/fermeture : yeux ouverts (15s), puis 3 répétitions de
   fermeture (3s) / ouverture (5s). Les timestamps servent de vérité
   terrain (label 0 = ouvert, 1 = fermé).
3. Phase clignement : un nombre de clignements fixé à l'avance est demandé,
   pour comparer le nombre de clignements comptés (fixe vs personnalisé)
   au nombre réel.
"""

import csv
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2

from calibration import calibrer
from capture import Webcam
from indicateurs import calculer_ear
from landmarks import DetecteurLandmarks

DOSSIER_SESSIONS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "sessions")
)
VRAI_NB_CLIGNEMENTS = 10


def executer_phase(webcam, detecteur, texte, duree_secondes, instant_debut_session):
    """
    Affiche `texte` avec un compte à rebours pendant `duree_secondes` et
    retourne la liste des (temps_relatif_depuis_debut_session, ear) mesurés.
    """
    mesures = []
    instant_debut = time.time()

    while time.time() - instant_debut < duree_secondes:
        frame = webcam.lire_frame()
        if frame is None:
            continue

        frame = cv2.flip(frame, 1)
        landmarks = detecteur.detecter(frame)

        if landmarks is not None:
            hauteur, largeur, _ = frame.shape
            ear = calculer_ear(landmarks, largeur, hauteur)
            mesures.append((time.time() - instant_debut_session, ear))

        temps_restant = duree_secondes - (time.time() - instant_debut)
        cv2.putText(frame, texte, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"{temps_restant:.0f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Enregistrement session - protocole d'evaluation", frame)
        cv2.waitKey(1)

    return mesures


def sauvegarder_session(condition, baseline, donnees_ouverture_fermeture, donnees_clignement):
    """Sauvegarde la session (métadonnées + mesures) dans un CSV."""
    os.makedirs(DOSSIER_SESSIONS, exist_ok=True)
    horodatage = time.strftime("%Y%m%d_%H%M%S")
    chemin = os.path.join(DOSSIER_SESSIONS, f"{condition}_{horodatage}.csv")

    with open(chemin, "w", newline="", encoding="utf-8") as fichier:
        fichier.write(f"# condition={condition}\n")
        fichier.write(f"# ear_moyen={baseline.ear_moyen}\n")
        fichier.write(f"# seuil_fermeture_yeux={baseline.seuil_fermeture_yeux}\n")
        fichier.write(f"# vrai_nb_clignements={VRAI_NB_CLIGNEMENTS}\n")

        writer = csv.writer(fichier)
        writer.writerow(["phase", "temps", "ear", "label"])
        for temps, ear, label in donnees_ouverture_fermeture:
            writer.writerow(["ouverture_fermeture", f"{temps:.4f}", f"{ear:.5f}", label])
        for temps, ear in donnees_clignement:
            writer.writerow(["clignement", f"{temps:.4f}", f"{ear:.5f}", ""])

    print(f"Session enregistrée : {chemin}")
    return chemin


def main():
    condition = input(
        "Nom de la condition pour cette session (ex. bien_eclaire, faible_luminosite) : "
    ).strip() or "session"

    webcam = Webcam()
    webcam.ouvrir()
    detecteur = DetecteurLandmarks()

    try:
        baseline = calibrer(webcam, detecteur)
        print(
            f"Calibration terminée : EAR moyen={baseline.ear_moyen:.3f}, "
            f"seuil personnalisé={baseline.seuil_fermeture_yeux:.3f}"
        )

        instant_debut_session = time.time()
        donnees_ouverture_fermeture = []

        mesures = executer_phase(
            webcam, detecteur, "Regardez la camera normalement, yeux ouverts", 15, instant_debut_session
        )
        donnees_ouverture_fermeture += [(t, ear, 0) for t, ear in mesures]

        for _ in range(3):
            mesures = executer_phase(webcam, detecteur, "Fermez les yeux", 3, instant_debut_session)
            donnees_ouverture_fermeture += [(t, ear, 1) for t, ear in mesures]

            mesures = executer_phase(webcam, detecteur, "Ouvrez les yeux", 5, instant_debut_session)
            donnees_ouverture_fermeture += [(t, ear, 0) for t, ear in mesures]

        donnees_clignement = executer_phase(
            webcam, detecteur,
            f"Clignez exactement {VRAI_NB_CLIGNEMENTS} fois, rythme normal", 20,
            instant_debut_session,
        )
    finally:
        webcam.fermer()
        detecteur.fermer()
        cv2.destroyAllWindows()

    sauvegarder_session(condition, baseline, donnees_ouverture_fermeture, donnees_clignement)


if __name__ == "__main__":
    main()
