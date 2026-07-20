"""
Historique basique de la session en cours : garde en mémoire les
indicateurs (EAR, MAR, score de vigilance) au fil du temps, et affiche
des courbes en fin de session (cf. claude.md, section 4, item 9).
"""

import time

import matplotlib.pyplot as plt


class HistoriqueSession:
    """Enregistre les indicateurs de la session en cours pour les visualiser ensuite."""

    def __init__(self):
        self.instant_debut = time.time()
        self.temps = []
        self.valeurs_ear = []
        self.valeurs_mar = []
        self.valeurs_score = []

    def enregistrer(self, ear, mar, score, instant=None):
        """Ajoute un point de mesure à l'historique (temps relatif depuis le début de session)."""
        instant = instant if instant is not None else time.time()
        self.temps.append(instant - self.instant_debut)
        self.valeurs_ear.append(ear)
        self.valeurs_mar.append(mar)
        self.valeurs_score.append(score)

    def afficher_courbes(self):
        """Affiche les courbes EAR/MAR/score de la session dans une fenêtre matplotlib."""
        if not self.temps:
            print("Historique vide : aucune donnée à afficher.")
            return

        figure, (axe_indicateurs, axe_score) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        axe_indicateurs.plot(self.temps, self.valeurs_ear, label="EAR", color="tab:blue")
        axe_indicateurs.plot(self.temps, self.valeurs_mar, label="MAR", color="tab:orange")
        axe_indicateurs.set_ylabel("EAR / MAR")
        axe_indicateurs.set_title("Indicateurs au fil de la session")
        axe_indicateurs.legend()

        axe_score.plot(self.temps, self.valeurs_score, label="Score de vigilance", color="tab:green")
        axe_score.axhline(70, color="orange", linestyle="--", linewidth=1, label="Seuil attention")
        axe_score.axhline(40, color="red", linestyle="--", linewidth=1, label="Seuil alerte")
        axe_score.set_ylim(0, 100)
        axe_score.set_xlabel("Temps (secondes)")
        axe_score.set_ylabel("Score (0-100)")
        axe_score.legend()

        plt.tight_layout()
        plt.show()
