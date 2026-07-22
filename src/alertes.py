"""
Système d'alerte graduée : son (généré, pas de fichier audio externe) +
message de recommandation textuel (cf. claude.md, section 4/9).

Rappel déontologique (section 6) : le système ne pose jamais de diagnostic
médical. Les messages parlent de "signes de fatigue" ou "score de
vigilance faible", jamais de termes médicaux diagnostiques.
"""

import time

import numpy as np
import pygame

from decision import NIVEAU_ALERTE, NIVEAU_ATTENTION, NIVEAU_NORMAL

FREQUENCE_ECHANTILLONNAGE = 44100
INTERVALLE_MIN_ALERTE_SECONDES = 3  # évite de rejouer le son à chaque frame

MESSAGES_RECOMMANDATION = {
    NIVEAU_ATTENTION: "Signes de fatigue detectes. Restez attentif.",
    NIVEAU_ALERTE: "Score de vigilance faible. Une pause est recommandee.",
}


def _generer_bip(frequence_hz, duree_secondes, volume):
    """Génère un son de bip (onde sinusoïdale), sans dépendre d'un fichier audio externe."""
    nb_echantillons = int(FREQUENCE_ECHANTILLONNAGE * duree_secondes)
    t = np.linspace(0, duree_secondes, nb_echantillons, False)
    onde = np.sin(frequence_hz * t * 2 * np.pi)
    onde = (onde * volume * 32767).astype(np.int16)
    stereo = np.column_stack((onde, onde))
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))


class GestionnaireAlertes:
    """Joue un son gradué selon le niveau et fournit un message de recommandation."""

    def __init__(self):
        pygame.mixer.init(frequency=FREQUENCE_ECHANTILLONNAGE, size=-16, channels=2)
        self.son_attention = _generer_bip(frequence_hz=660, duree_secondes=0.2, volume=0.4)
        self.son_alerte = _generer_bip(frequence_hz=990, duree_secondes=0.4, volume=0.6)
        self.dernier_son_instant = 0.0

    def traiter(self, niveau, signes_actuels, instant=None):
        """
        Joue le son d'alerte correspondant (au maximum une fois toutes les
        `INTERVALLE_MIN_ALERTE_SECONDES`) et retourne le message de
        recommandation associé (ou None si niveau normal).

        `signes_actuels` reflète l'état de la frame en cours (yeux fermés,
        microsommeil ou bâillement en cours) : le score de vigilance étant
        lissé sur une fenêtre glissante (PERCLOS), il reste bas un moment
        après un épisode de fatigue même si la personne est redevenue
        normale. Sans ce paramètre, l'alarme continuerait à se répéter
        alors que la situation est déjà revenue à la normale.
        """
        instant = instant if instant is not None else time.time()

        if niveau == NIVEAU_NORMAL or not signes_actuels:
            return None

        if instant - self.dernier_son_instant >= INTERVALLE_MIN_ALERTE_SECONDES:
            son = self.son_alerte if niveau == NIVEAU_ALERTE else self.son_attention
            son.play()
            self.dernier_son_instant = instant

        return MESSAGES_RECOMMANDATION.get(niveau)
