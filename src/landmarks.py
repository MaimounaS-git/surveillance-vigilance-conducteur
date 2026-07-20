"""
Wrapper autour de MediaPipe Face Mesh pour la détection des landmarks du visage.
"""

import cv2
import mediapipe as mp

# Indices des landmarks MediaPipe Face Mesh (468 points) qui seront utilisés
# au bloc suivant du pipeline pour calculer l'EAR (yeux) et le MAR (bouche).
# Ordre par oeil : [coin gauche, haut1, haut2, coin droit, bas1, bas2]
LANDMARKS_OEIL_GAUCHE = [362, 385, 387, 263, 373, 380]
LANDMARKS_OEIL_DROIT = [33, 160, 158, 133, 153, 144]
# Bouche : [coin gauche, coin droit, haut1, bas1, haut2, bas2]
LANDMARKS_BOUCHE = [61, 291, 39, 181, 0, 17]


class DetecteurLandmarks:
    """Détecte les landmarks du visage sur une frame via MediaPipe Face Mesh."""

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detecter(self, frame):
        """
        Traite une frame BGR (format OpenCV) et retourne la liste des landmarks
        du premier visage détecté, ou None si aucun visage n'est détecté.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultats = self.face_mesh.process(frame_rgb)

        if not resultats.multi_face_landmarks:
            return None

        return resultats.multi_face_landmarks[0].landmark

    def fermer(self):
        """Libère les ressources du modèle."""
        self.face_mesh.close()
