"""
Calcul du score de vigilance (0-100), combinant PERCLOS (proportion du temps
yeux fermés), bâillement et fréquence de clignement (cf. claude.md, sections
4-6). Référence scientifique du PERCLOS : Wierwille et al. (1994), validé par
Dinges & Grace (1998).

Formule heuristique simple : le score part de 100 et est pénalisé par chaque
signe de fatigue détecté. Les poids sont volontairement simples ; ils pourront
être affinés lors de l'évaluation seuils fixes vs personnalisés (section 4/10).
"""

import time
from collections import deque

FENETRE_PERCLOS_SECONDES = 60
POIDS_PERCLOS = 100  # PERCLOS in [0,1] -> pénalité max 100 si yeux fermés tout le temps
POIDS_BAILLEMENT = 30  # pénalité fixe si un bâillement est en cours
FREQUENCE_CLIGNEMENT_NORMALE_MIN = 10
FREQUENCE_CLIGNEMENT_NORMALE_MAX = 20
POIDS_CLIGNEMENT_ANORMAL = 20


class ScoreVigilance:
    """
    Suit l'état yeux fermés/ouverts sur une fenêtre glissante pour calculer le
    PERCLOS, et combine PERCLOS, bâillement et fréquence de clignement en un
    score de vigilance (0-100).
    """

    def __init__(self, fenetre_secondes=FENETRE_PERCLOS_SECONDES):
        self.fenetre_secondes = fenetre_secondes
        self.historique_fermeture = deque()  # (timestamp, yeux_fermes)

    def _purger_anciens(self, instant):
        while (
            self.historique_fermeture
            and instant - self.historique_fermeture[0][0] > self.fenetre_secondes
        ):
            self.historique_fermeture.popleft()

    def calculer_perclos(self, yeux_fermes, instant):
        """Met à jour l'historique et retourne le PERCLOS (proportion [0,1]) sur la fenêtre glissante."""
        self.historique_fermeture.append((instant, yeux_fermes))
        self._purger_anciens(instant)
        if not self.historique_fermeture:
            return 0.0
        nb_fermes = sum(1 for _, ferme in self.historique_fermeture if ferme)
        return nb_fermes / len(self.historique_fermeture)

    def calculer_score(self, ear, mar, frequence_clignement, baseline, instant=None):
        """Retourne (score, perclos) : score de vigilance (0-100) et PERCLOS courant."""
        instant = instant if instant is not None else time.time()

        yeux_fermes = ear < baseline.seuil_fermeture_yeux
        perclos = self.calculer_perclos(yeux_fermes, instant)

        score = 100.0
        score -= perclos * POIDS_PERCLOS

        if mar > baseline.seuil_baillement:
            score -= POIDS_BAILLEMENT

        if not (FREQUENCE_CLIGNEMENT_NORMALE_MIN <= frequence_clignement <= FREQUENCE_CLIGNEMENT_NORMALE_MAX):
            score -= POIDS_CLIGNEMENT_ANORMAL

        score = max(0.0, min(100.0, score))
        return score, perclos
