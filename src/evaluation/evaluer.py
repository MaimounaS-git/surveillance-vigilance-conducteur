"""
Évalue et compare le seuil fixe (générique, littérature) et le seuil
personnalisé (calibration) sur les sessions enregistrées par
enregistrer_session.py.

Deux comparaisons :
1. Détection yeux ouverts/fermés (phase "ouverture_fermeture") : precision,
   rappel, F1, taux de faux positifs.
2. Comptage de clignements (phase "clignement") : nombre détecté vs nombre
   réel demandé dans le protocole.
"""

import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from clignement import CompteurClignements

DOSSIER_SESSIONS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "sessions")
)
CHEMIN_RESULTATS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "evaluation_resultats.csv")
)

# Seuil fixe générique couramment cité dans la littérature EAR
# (ex. Soukupová & Čech, 2016, "Real-Time Eye Blink Detection using Facial Landmarks")
SEUIL_FIXE = 0.25


def lire_metadonnees(chemin):
    """Lit les lignes '# cle=valeur' en tête du fichier CSV de session."""
    metadonnees = {}
    with open(chemin, "r", encoding="utf-8") as fichier:
        for ligne in fichier:
            if not ligne.startswith("#"):
                break
            cle, _, valeur = ligne[1:].strip().partition("=")
            metadonnees[cle.strip()] = valeur.strip()
    return metadonnees


def evaluer_ouverture_fermeture(df, seuil):
    """Calcule precision/rappel/F1/taux de faux positifs pour un seuil donné."""
    y_vrai = df["label"].astype(int)
    y_predit = (df["ear"] < seuil).astype(int)

    precision = precision_score(y_vrai, y_predit, zero_division=0)
    rappel = recall_score(y_vrai, y_predit, zero_division=0)
    f1 = f1_score(y_vrai, y_predit, zero_division=0)

    vrai_negatif, faux_positif, faux_negatif, vrai_positif = confusion_matrix(
        y_vrai, y_predit, labels=[0, 1]
    ).ravel()
    taux_faux_positifs = (
        faux_positif / (faux_positif + vrai_negatif) if (faux_positif + vrai_negatif) > 0 else 0.0
    )

    return precision, rappel, f1, taux_faux_positifs


def compter_clignements(df, seuil):
    """Recompte les clignements détectés sur la phase clignement, pour un seuil donné."""
    compteur = CompteurClignements(seuil_ear=seuil)
    for _, ligne in df.sort_values("temps").iterrows():
        compteur.mettre_a_jour(ligne["ear"], instant=ligne["temps"])
    return compteur.total_clignements


def evaluer_session(chemin):
    """Calcule toutes les métriques (fixe vs personnalisé) pour une session."""
    metadonnees = lire_metadonnees(chemin)
    df = pd.read_csv(chemin, comment="#")

    seuil_personnalise = float(metadonnees["seuil_fermeture_yeux"])
    vrai_nb_clignements = int(metadonnees["vrai_nb_clignements"])

    df_ouverture = df[df["phase"] == "ouverture_fermeture"]
    df_clignement = df[df["phase"] == "clignement"]

    precision_fixe, rappel_fixe, f1_fixe, fpr_fixe = evaluer_ouverture_fermeture(df_ouverture, SEUIL_FIXE)
    precision_perso, rappel_perso, f1_perso, fpr_perso = evaluer_ouverture_fermeture(
        df_ouverture, seuil_personnalise
    )

    return {
        "condition": metadonnees.get("condition", os.path.basename(chemin)),
        "seuil_fixe": SEUIL_FIXE,
        "seuil_personnalise": seuil_personnalise,
        "precision_fixe": precision_fixe,
        "rappel_fixe": rappel_fixe,
        "f1_fixe": f1_fixe,
        "fpr_fixe": fpr_fixe,
        "precision_perso": precision_perso,
        "rappel_perso": rappel_perso,
        "f1_perso": f1_perso,
        "fpr_perso": fpr_perso,
        "clignements_vrai": vrai_nb_clignements,
        "clignements_fixe": compter_clignements(df_clignement, SEUIL_FIXE),
        "clignements_perso": compter_clignements(df_clignement, seuil_personnalise),
    }


def main():
    chemins = sorted(glob.glob(os.path.join(DOSSIER_SESSIONS, "*.csv")))
    if not chemins:
        print(f"Aucune session trouvée dans {DOSSIER_SESSIONS}. Lancez d'abord enregistrer_session.py.")
        return

    resultats = [evaluer_session(chemin) for chemin in chemins]
    df_resultats = pd.DataFrame(resultats)

    pd.set_option("display.width", 150)
    print(df_resultats.to_string(index=False))

    os.makedirs(os.path.dirname(CHEMIN_RESULTATS), exist_ok=True)
    df_resultats.to_csv(CHEMIN_RESULTATS, index=False)
    print(f"\nRésultats sauvegardés : {CHEMIN_RESULTATS}")

    figure, axe = plt.subplots(figsize=(8, 5))
    largeur_barre = 0.35
    positions = range(len(df_resultats))
    axe.bar(
        [p - largeur_barre / 2 for p in positions], df_resultats["f1_fixe"],
        largeur_barre, label="Seuil fixe",
    )
    axe.bar(
        [p + largeur_barre / 2 for p in positions], df_resultats["f1_perso"],
        largeur_barre, label="Seuil personnalisé",
    )
    axe.set_xticks(list(positions))
    axe.set_xticklabels(df_resultats["condition"])
    axe.set_ylabel("F1-score")
    axe.set_title("Seuil fixe vs seuil personnalisé (détection yeux fermés)")
    axe.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
