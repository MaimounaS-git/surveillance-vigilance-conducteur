"""
Point d'entrée : boucle de capture webcam + détection et affichage des landmarks
du visage. Premier bloc du pipeline (cf. claude.md, section 5).
"""

import time

import cv2

from capture import Webcam
from landmarks import DetecteurLandmarks


def dessiner_landmarks(frame, landmarks):
    """Dessine chaque landmark du visage sous forme de petit point sur la frame."""
    hauteur, largeur, _ = frame.shape
    for point in landmarks:
        x = int(point.x * largeur)
        y = int(point.y * hauteur)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def main():
    webcam = Webcam()
    webcam.ouvrir()
    detecteur = DetecteurLandmarks()

    temps_precedent = time.time()

    try:
        while True:
            frame = webcam.lire_frame()
            if frame is None:
                print("Erreur de lecture de la webcam.")
                break

            frame = cv2.flip(frame, 1)  # effet miroir, plus naturel pour l'utilisateur
            landmarks = detecteur.detecter(frame)

            visage_detecte = landmarks is not None
            if visage_detecte:
                dessiner_landmarks(frame, landmarks)

            # Calcul du FPS pour valider la stabilité du pipeline avant le bloc suivant
            temps_actuel = time.time()
            ecart = temps_actuel - temps_precedent
            fps = 1 / ecart if ecart > 0 else 0
            temps_precedent = temps_actuel

            statut = "Visage detecte" if visage_detecte else "Aucun visage"
            texte = f"{statut} | FPS: {fps:.1f}"
            cv2.putText(frame, texte, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Surveillance de la vigilance - landmarks", frame)

            touche = cv2.waitKey(1) & 0xFF
            if touche == ord("q") or touche == 27:  # 'q' ou ESC
                break
    finally:
        webcam.fermer()
        detecteur.fermer()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
