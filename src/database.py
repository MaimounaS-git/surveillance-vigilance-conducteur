"""
Stockage de l'historique des sessions en base SQLite (cf. claude.md,
section 8, pour le schéma). Contrairement à historique.py (courbes de la
session en cours uniquement, en mémoire), ce module conserve l'historique
à travers plusieurs sessions, d'une exécution à l'autre.
"""

import os
import sqlite3

CHEMIN_BDD_PAR_DEFAUT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "vigilance.db")
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    duree_secondes INTEGER,
    score_moyen REAL,
    nb_alertes INTEGER,
    nb_baillements INTEGER,
    nb_clignements INTEGER
);

CREATE TABLE IF NOT EXISTS mesures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    timestamp REAL,
    ear REAL,
    mar REAL,
    score_vigilance REAL,
    alerte BOOLEAN,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
"""


class BaseDeDonnees:
    """Wrapper autour de la base SQLite pour l'historique des sessions."""

    def __init__(self, chemin=CHEMIN_BDD_PAR_DEFAUT):
        os.makedirs(os.path.dirname(chemin), exist_ok=True)
        self.connexion = sqlite3.connect(chemin)
        self.connexion.executescript(SCHEMA)
        self.connexion.commit()

    def creer_session(self, date):
        """Crée une nouvelle session (statistiques à 0, mises à jour en fin de session) et retourne son id."""
        curseur = self.connexion.execute(
            """
            INSERT INTO sessions (date, duree_secondes, score_moyen, nb_alertes, nb_baillements, nb_clignements)
            VALUES (?, 0, 0, 0, 0, 0)
            """,
            (date,),
        )
        self.connexion.commit()
        return curseur.lastrowid

    def ajouter_mesure(self, session_id, timestamp, ear, mar, score_vigilance, alerte):
        """Ajoute une mesure ponctuelle liée à une session."""
        self.connexion.execute(
            """
            INSERT INTO mesures (session_id, timestamp, ear, mar, score_vigilance, alerte)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, timestamp, ear, mar, score_vigilance, alerte),
        )

    def mettre_a_jour_session(
        self, session_id, duree_secondes, score_moyen, nb_alertes, nb_baillements, nb_clignements
    ):
        """Met à jour les statistiques agrégées d'une session (calculées en fin de session)."""
        self.connexion.execute(
            """
            UPDATE sessions
            SET duree_secondes = ?, score_moyen = ?, nb_alertes = ?, nb_baillements = ?, nb_clignements = ?
            WHERE id = ?
            """,
            (duree_secondes, score_moyen, nb_alertes, nb_baillements, nb_clignements, session_id),
        )
        self.connexion.commit()

    def lister_sessions(self):
        """Retourne toutes les sessions enregistrées (les plus récentes en premier)."""
        curseur = self.connexion.execute("SELECT * FROM sessions ORDER BY id DESC")
        colonnes = [description[0] for description in curseur.description]
        return [dict(zip(colonnes, ligne)) for ligne in curseur.fetchall()]

    def obtenir_mesures(self, session_id):
        """Retourne toutes les mesures d'une session donnée, triées par temps."""
        curseur = self.connexion.execute(
            "SELECT * FROM mesures WHERE session_id = ? ORDER BY timestamp", (session_id,)
        )
        colonnes = [description[0] for description in curseur.description]
        return [dict(zip(colonnes, ligne)) for ligne in curseur.fetchall()]

    def fermer(self):
        self.connexion.close()
