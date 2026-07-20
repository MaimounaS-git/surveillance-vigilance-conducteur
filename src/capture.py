"""
Wrapper autour de cv2.VideoCapture pour la lecture du flux webcam.
"""

import cv2


class Webcam:
    """Gère l'ouverture, la lecture et la fermeture propre de la webcam."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.capture = None

    def ouvrir(self):
        """Ouvre la webcam. Lève une erreur si l'ouverture échoue."""
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la webcam (index {self.camera_index})")

    def lire_frame(self):
        """Lit une frame. Retourne None si la lecture échoue."""
        succes, frame = self.capture.read()
        if not succes:
            return None
        return frame

    def fermer(self):
        """Libère la webcam."""
        if self.capture is not None:
            self.capture.release()
