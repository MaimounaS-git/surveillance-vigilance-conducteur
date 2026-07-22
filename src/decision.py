"""
Module de décision : lissage temporel et détermination du niveau de
vigilance (normal / attention / alerte), cf. claude.md section 4 (item 7)
et section 6 (microsommeil comportemental).

Le lissage temporel évite les fausses alertes sur un simple clignement
(100-400ms) : un microsommeil n'est déclaré que si les yeux restent
fermés en continu plus de 500ms. De la même façon, un bâillement n'est
déclaré que si la bouche reste ouverte en continu (MAR élevé soutenu,
1-3 secondes selon claude.md section 6) : une brève ouverture de bouche
(parole, expression) ne doit pas déclencher d'alerte.
"""

import time

DUREE_MICROSOMMEIL_SECONDES = 0.5  # cf. claude.md section 6
DUREE_BAILLEMENT_SECONDES = 1.0  # cf. claude.md section 6 : MAR élevé soutenu ~1-3s

SEUIL_SCORE_ATTENTION = 70
SEUIL_SCORE_ALERTE = 40

NIVEAU_NORMAL = "normal"
NIVEAU_ATTENTION = "attention"
NIVEAU_ALERTE = "alerte"


class Decision:
    """
    Détermine le niveau de vigilance (normal/attention/alerte) en combinant :
    - la détection d'un microsommeil comportemental (yeux fermés > 500ms en continu)
    - la détection d'un bâillement (bouche ouverte > 1s en continu)
    - le score de vigilance courant
    """

    def __init__(
        self,
        duree_microsommeil_secondes=DUREE_MICROSOMMEIL_SECONDES,
        duree_baillement_secondes=DUREE_BAILLEMENT_SECONDES,
    ):
        self.duree_microsommeil_secondes = duree_microsommeil_secondes
        self.duree_baillement_secondes = duree_baillement_secondes
        self.instant_debut_fermeture = None
        self.instant_debut_baillement = None

    def _microsommeil_en_cours(self, yeux_fermes, instant):
        """Retourne True si les yeux sont fermés en continu depuis plus de 500ms."""
        if not yeux_fermes:
            self.instant_debut_fermeture = None
            return False

        if self.instant_debut_fermeture is None:
            self.instant_debut_fermeture = instant

        duree_fermeture = instant - self.instant_debut_fermeture
        return duree_fermeture >= self.duree_microsommeil_secondes

    def _baillement_en_cours(self, bouche_ouverte, instant):
        """Retourne True si la bouche est ouverte en continu depuis plus de 1s."""
        if not bouche_ouverte:
            self.instant_debut_baillement = None
            return False

        if self.instant_debut_baillement is None:
            self.instant_debut_baillement = instant

        duree_ouverture = instant - self.instant_debut_baillement
        return duree_ouverture >= self.duree_baillement_secondes

    def determiner_niveau(self, yeux_fermes, bouche_ouverte, score, instant=None):
        """Retourne (niveau, microsommeil_detecte, baillement_detecte) à partir de l'état courant."""
        instant = instant if instant is not None else time.time()
        microsommeil = self._microsommeil_en_cours(yeux_fermes, instant)
        baillement = self._baillement_en_cours(bouche_ouverte, instant)

        if microsommeil or score < SEUIL_SCORE_ALERTE:
            niveau = NIVEAU_ALERTE
        elif baillement or score < SEUIL_SCORE_ATTENTION:
            niveau = NIVEAU_ATTENTION
        else:
            niveau = NIVEAU_NORMAL

        return niveau, microsommeil, baillement
