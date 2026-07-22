"""
Module de décision : lissage temporel et détermination du niveau de
vigilance (normal / attention / alerte), cf. claude.md section 4 (item 7)
et section 6 (microsommeil comportemental).

Le lissage temporel évite les fausses alertes sur un simple clignement
(100-400ms) : un microsommeil n'est déclaré que si les yeux restent
fermés en continu plus de 500ms.

Pour le bâillement, un simple "MAR > seuil" ne suffit pas à distinguer un
vrai bâillement de la parole (voir description physiologique fournie) :
- Bâillement : une seule ouverture progressive (montée lente, plateau,
  descente lente), durée ~3-8s.
- Parole : de nombreuses petites ouvertures rapides et irrégulières,
  0.1-0.5s chacune.
On exige donc : MAR > seuil soutenu plus de 2s, ET une séquence de MAR peu
"oscillante" (peu de changements de direction), caractéristique d'une
bosse lisse plutôt que d'oscillations rapides.
"""

import time

DUREE_MICROSOMMEIL_SECONDES = 0.5  # cf. claude.md section 6
DUREE_BAILLEMENT_SECONDES = 2.0  # cf. description physiologique du bâillement (durée > 2s)
RATIO_MAX_CHANGEMENTS_DIRECTION = 0.35  # au-delà, la séquence MAR est jugée trop erratique (parole)

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
        self.sequence_mar_baillement = []

    def _microsommeil_en_cours(self, yeux_fermes, instant):
        """Retourne True si les yeux sont fermés en continu depuis plus de 500ms."""
        if not yeux_fermes:
            self.instant_debut_fermeture = None
            return False

        if self.instant_debut_fermeture is None:
            self.instant_debut_fermeture = instant

        duree_fermeture = instant - self.instant_debut_fermeture
        return duree_fermeture >= self.duree_microsommeil_secondes

    def _baillement_en_cours(self, bouche_ouverte, mar, instant):
        """
        Retourne True si la bouche est ouverte en continu depuis plus de
        `duree_baillement_secondes` ET que la séquence de MAR observée
        ressemble à une bosse lisse (bâillement) plutôt qu'à des
        oscillations rapides (parole).
        """
        if not bouche_ouverte:
            self.instant_debut_baillement = None
            self.sequence_mar_baillement = []
            return False

        if self.instant_debut_baillement is None:
            self.instant_debut_baillement = instant
            self.sequence_mar_baillement = []

        self.sequence_mar_baillement.append(mar)
        duree_ouverture = instant - self.instant_debut_baillement

        if duree_ouverture < self.duree_baillement_secondes:
            return False

        return self._sequence_est_progressive(self.sequence_mar_baillement)

    @staticmethod
    def _sequence_est_progressive(sequence_mar):
        """
        Vérifie que l'ouverture suit une seule bosse lisse (montée puis
        descente progressives), caractéristique physiologique du
        bâillement, plutôt que des oscillations rapides typiques de la
        parole : on compte la proportion de changements de direction
        (hausse -> baisse ou inversement) dans la séquence de MAR.
        """
        if len(sequence_mar) < 4:
            return False

        variations = [
            b - a for a, b in zip(sequence_mar, sequence_mar[1:]) if abs(b - a) > 1e-6
        ]
        if len(variations) < 2:
            return True  # variation quasi nulle : plateau stable, pas d'oscillation

        signes = [1 if v > 0 else -1 for v in variations]
        nb_changements = sum(1 for a, b in zip(signes, signes[1:]) if a != b)
        ratio_changements = nb_changements / len(signes)

        return ratio_changements <= RATIO_MAX_CHANGEMENTS_DIRECTION

    def determiner_niveau(self, yeux_fermes, bouche_ouverte, mar, score, instant=None):
        """Retourne (niveau, microsommeil_detecte, baillement_detecte) à partir de l'état courant."""
        instant = instant if instant is not None else time.time()
        microsommeil = self._microsommeil_en_cours(yeux_fermes, instant)
        baillement = self._baillement_en_cours(bouche_ouverte, mar, instant)

        if microsommeil or score < SEUIL_SCORE_ALERTE:
            niveau = NIVEAU_ALERTE
        elif baillement or score < SEUIL_SCORE_ATTENTION:
            niveau = NIVEAU_ATTENTION
        else:
            niveau = NIVEAU_NORMAL

        return niveau, microsommeil, baillement
