"""
Module de décision : lissage temporel et détermination du niveau de
vigilance (normal / attention / alerte), cf. claude.md section 4 (item 7)
et section 6 (microsommeil comportemental).

Le lissage temporel évite les fausses alertes sur un simple clignement
(100-400ms) : un microsommeil n'est déclaré que si les yeux restent
fermés en continu plus de 500ms.
"""

import time

DUREE_MICROSOMMEIL_SECONDES = 0.5  # cf. claude.md section 6

SEUIL_SCORE_ATTENTION = 70
SEUIL_SCORE_ALERTE = 40

NIVEAU_NORMAL = "normal"
NIVEAU_ATTENTION = "attention"
NIVEAU_ALERTE = "alerte"


class Decision:
    """
    Détermine le niveau de vigilance (normal/attention/alerte) en combinant :
    - la détection d'un microsommeil comportemental (yeux fermés > 500ms en continu)
    - le score de vigilance courant
    """

    def __init__(self, duree_microsommeil_secondes=DUREE_MICROSOMMEIL_SECONDES):
        self.duree_microsommeil_secondes = duree_microsommeil_secondes
        self.instant_debut_fermeture = None

    def _microsommeil_en_cours(self, yeux_fermes, instant):
        """Retourne True si les yeux sont fermés en continu depuis plus de 500ms."""
        if not yeux_fermes:
            self.instant_debut_fermeture = None
            return False

        if self.instant_debut_fermeture is None:
            self.instant_debut_fermeture = instant

        duree_fermeture = instant - self.instant_debut_fermeture
        return duree_fermeture >= self.duree_microsommeil_secondes

    def determiner_niveau(self, yeux_fermes, score, instant=None):
        """Retourne (niveau, microsommeil_detecte) à partir de l'état courant."""
        instant = instant if instant is not None else time.time()
        microsommeil = self._microsommeil_en_cours(yeux_fermes, instant)

        if microsommeil or score < SEUIL_SCORE_ALERTE:
            niveau = NIVEAU_ALERTE
        elif score < SEUIL_SCORE_ATTENTION:
            niveau = NIVEAU_ATTENTION
        else:
            niveau = NIVEAU_NORMAL

        return niveau, microsommeil
