"""
Détection des clignements des yeux et calcul de leur fréquence sur une
fenêtre glissante temporelle (cf. claude.md, section 6 : "Fenêtre glissante
temps réel").

Le seuil EAR utilisé ici est provisoire : il sera affiné par la calibration
personnalisée (bloc suivant du pipeline).
"""

import time
from collections import deque

SEUIL_EAR_CLIGNEMENT = 0.4
DUREE_MIN_CLIGNEMENT = 0.03  # en dessous, on considère que c'est du bruit
FENETRE_FREQUENCE_SECONDES = 60


class CompteurClignements:
    """Détecte les clignements (yeux ouverts -> fermés -> ouverts) et leur fréquence."""

    def __init__(self, seuil_ear=SEUIL_EAR_CLIGNEMENT, fenetre_secondes=FENETRE_FREQUENCE_SECONDES):
        self.seuil_ear = seuil_ear
        self.fenetre_secondes = fenetre_secondes
        self.yeux_fermes = False
        self.instant_fermeture = None
        self.horodatages_clignements = deque()
        self.total_clignements = 0

    def mettre_a_jour(self, ear, instant=None):
        """
        À appeler à chaque frame avec la valeur EAR courante.
        Retourne True si un clignement vient d'être détecté sur cette frame.
        """
        instant = instant if instant is not None else time.time()
        clignement_detecte = False

        if ear < self.seuil_ear and not self.yeux_fermes:
            self.yeux_fermes = True
            self.instant_fermeture = instant
        elif ear >= self.seuil_ear and self.yeux_fermes:
            self.yeux_fermes = False
            duree_fermeture = instant - self.instant_fermeture
            if duree_fermeture >= DUREE_MIN_CLIGNEMENT:
                self.horodatages_clignements.append(instant)
                self.total_clignements += 1
                clignement_detecte = True

        self._purger_anciens(instant)
        return clignement_detecte

    def _purger_anciens(self, instant):
        """Retire de la fenêtre glissante les clignements devenus trop anciens."""
        while (
            self.horodatages_clignements
            and instant - self.horodatages_clignements[0] > self.fenetre_secondes
        ):
            self.horodatages_clignements.popleft()

    def frequence_par_minute(self):
        """Fréquence de clignement (par minute), calculée sur la fenêtre glissante."""
        nb_clignements = len(self.horodatages_clignements)
        return nb_clignements * (60 / self.fenetre_secondes)
