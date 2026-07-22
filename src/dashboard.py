"""
Tableau de bord Dash : vue en direct (flux vidéo + indicateurs) et vue
historique (sessions passées, export CSV), cf. claude.md section 9.

Point d'entrée indépendant de main.py : réutilise les mêmes modules bas
niveau (indicateurs, calibration, clignement, scoring, decision, alertes,
tete, database), mais avec sa propre boucle tournant dans un thread
d'arrière-plan (pas de fenêtre OpenCV ; la vidéo est streamée en MJPEG
vers le navigateur via une route Flask).
"""

import threading
import time
from collections import deque
from datetime import datetime

import cv2
import dash
import pandas as pd
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State
from flask import Response

from alertes import GestionnaireAlertes
from calibration import calibrer
from capture import Webcam
from clignement import CompteurClignements
from database import BaseDeDonnees
from decision import Decision, NIVEAU_ALERTE, NIVEAU_NORMAL
from indicateurs import calculer_ear, calculer_mar
from landmarks import DetecteurLandmarks
from scoring import ScoreVigilance

DUREE_HISTORIQUE_GRAPHIQUE_SECONDES = 120
INTERVALLE_MAJ_BDD_SECONDES = 10

# Palette et styles (aspect visuel du tableau de bord)
COULEUR_FOND = "#0f172a"
COULEUR_CARTE = "#1e293b"
COULEUR_TEXTE = "#e2e8f0"
COULEUR_TEXTE_ATTENUE = "#94a3b8"
COULEUR_ACCENT = "#38bdf8"

STYLE_PAGE = {
    "backgroundColor": COULEUR_FOND,
    "minHeight": "100vh",
    "padding": "32px",
    "fontFamily": "'Helvetica Neue', Arial, sans-serif",
    "color": COULEUR_TEXTE,
}
STYLE_CARTE = {
    "backgroundColor": COULEUR_CARTE,
    "borderRadius": "14px",
    "padding": "20px 24px",
    "boxShadow": "0 4px 16px rgba(0,0,0,0.35)",
    "marginBottom": "20px",
}
STYLE_TITRE_PAGE = {"fontSize": "26px", "fontWeight": "700", "marginBottom": "24px"}
STYLE_SOUS_TITRE = {
    "fontSize": "14px", "fontWeight": "600", "color": COULEUR_TEXTE_ATTENUE,
    "textTransform": "uppercase", "letterSpacing": "0.05em", "marginBottom": "10px",
}
STYLE_BOUTON = {
    "backgroundColor": COULEUR_ACCENT, "color": "#0f172a", "border": "none",
    "borderRadius": "8px", "padding": "10px 18px", "fontWeight": "600",
    "cursor": "pointer", "marginBottom": "12px",
}
GABARIT_GRAPHIQUE = {
    "paper_bgcolor": COULEUR_CARTE, "plot_bgcolor": COULEUR_CARTE,
    "font": {"color": COULEUR_TEXTE},
}


def couleur_score_web(score):
    """Vert si vigilance correcte, orange si attention, rouge si alerte (cf. main.couleur_score)."""
    if score >= 70:
        return "#22c55e"
    if score >= 40:
        return "#f59e0b"
    return "#ef4444"


def dessiner_landmarks(frame, landmarks):
    """Dessine chaque landmark du visage sous forme de petit point sur la frame."""
    hauteur, largeur, _ = frame.shape
    for point in landmarks:
        x = int(point.x * largeur)
        y = int(point.y * hauteur)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


class EtatPartage:
    """État courant de la session, partagé entre le thread de traitement et les callbacks Dash."""

    def __init__(self):
        self.verrou = threading.Lock()
        self.phase = "calibration"  # "calibration" -> "surveillance" (ou "erreur")
        self.temps_restant_calibration = None
        self.erreur = None
        self.frame_jpeg = None
        self.ear = 0.0
        self.mar = 0.0
        self.score = 0.0
        self.niveau = NIVEAU_NORMAL
        self.message_recommandation = None
        self.nb_clignements = 0
        self.nb_baillements = 0
        self.frequence_clignements = 0.0
        self.duree_ecoulee = 0.0
        self.historique_courbes = deque()  # (temps, ear, mar)


etat = EtatPartage()


def _afficher_calibration_dans_le_flux(frame, temps_restant):
    """Callback passé à calibration.calibrer() : affiche la calibration dans le flux vidéo web."""
    _, buffer = cv2.imencode(".jpg", frame)
    with etat.verrou:
        etat.frame_jpeg = buffer.tobytes()
        etat.temps_restant_calibration = temps_restant


def boucle_traitement(webcam, detecteur):
    """
    Boucle d'arrière-plan : calibration (visible dans le flux vidéo web), puis
    capture, détection, indicateurs, écriture dans l'état partagé + la BDD.
    """
    try:
        baseline = calibrer(webcam, detecteur, sur_frame=_afficher_calibration_dans_le_flux)
    except RuntimeError as erreur:
        with etat.verrou:
            etat.phase = "erreur"
            etat.erreur = str(erreur)
        return

    with etat.verrou:
        etat.phase = "surveillance"

    compteur_clignements = CompteurClignements(seuil_ear=baseline.seuil_fermeture_yeux)
    score_vigilance = ScoreVigilance()
    decision = Decision()
    gestionnaire_alertes = GestionnaireAlertes()

    bdd = BaseDeDonnees()
    session_id = bdd.creer_session(date=datetime.now().isoformat(timespec="seconds"))
    instant_debut_session = time.time()
    somme_scores, nb_mesures = 0.0, 0
    nb_alertes_session, nb_baillements_session = 0, 0
    alerte_precedente, baillement_precedent = False, False
    dernier_maj_bdd = time.time()

    while True:
        frame = webcam.lire_frame()
        if frame is None:
            continue

        frame = cv2.flip(frame, 1)
        landmarks = detecteur.detecter(frame)

        if landmarks is not None:
            dessiner_landmarks(frame, landmarks)
            hauteur, largeur, _ = frame.shape
            ear = calculer_ear(landmarks, largeur, hauteur)
            mar = calculer_mar(landmarks, largeur, hauteur)
            compteur_clignements.mettre_a_jour(ear)
            score, _ = score_vigilance.calculer_score(
                ear, mar, compteur_clignements.frequence_par_minute(), baseline
            )
            yeux_fermes = ear < baseline.seuil_fermeture_yeux
            bouche_ouverte = mar > baseline.seuil_baillement
            niveau, microsommeil, baillement = decision.determiner_niveau(
                yeux_fermes, bouche_ouverte, mar, score
            )
            signes_actuels = yeux_fermes or microsommeil or baillement
            message_recommandation = gestionnaire_alertes.traiter(niveau, signes_actuels)

            temps_ecoule = time.time() - instant_debut_session
            bdd.ajouter_mesure(session_id, temps_ecoule, ear, mar, score, niveau == NIVEAU_ALERTE)
            somme_scores += score
            nb_mesures += 1
            if niveau == NIVEAU_ALERTE and not alerte_precedente:
                nb_alertes_session += 1
            alerte_precedente = niveau == NIVEAU_ALERTE
            if baillement and not baillement_precedent:
                nb_baillements_session += 1
            baillement_precedent = baillement

            if time.time() - dernier_maj_bdd >= INTERVALLE_MAJ_BDD_SECONDES:
                bdd.mettre_a_jour_session(
                    session_id, temps_ecoule,
                    somme_scores / nb_mesures if nb_mesures else 0.0,
                    nb_alertes_session, nb_baillements_session,
                    compteur_clignements.total_clignements,
                )
                dernier_maj_bdd = time.time()

            with etat.verrou:
                etat.ear = ear
                etat.mar = mar
                etat.score = score
                etat.niveau = niveau
                etat.message_recommandation = message_recommandation
                etat.nb_clignements = compteur_clignements.total_clignements
                etat.nb_baillements = nb_baillements_session
                etat.frequence_clignements = compteur_clignements.frequence_par_minute()
                etat.duree_ecoulee = temps_ecoule
                etat.historique_courbes.append((temps_ecoule, ear, mar))
                while (
                    etat.historique_courbes
                    and temps_ecoule - etat.historique_courbes[0][0] > DUREE_HISTORIQUE_GRAPHIQUE_SECONDES
                ):
                    etat.historique_courbes.popleft()

        _, buffer = cv2.imencode(".jpg", frame)
        with etat.verrou:
            etat.frame_jpeg = buffer.tobytes()

        time.sleep(0.01)


def generer_flux_video():
    """Générateur MJPEG pour la route Flask /video_feed."""
    while True:
        with etat.verrou:
            frame_jpeg = etat.frame_jpeg
        if frame_jpeg is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_jpeg + b"\r\n"
            )
        time.sleep(0.05)


app = dash.Dash(__name__)
app.title = "Surveillance de la vigilance"


@app.server.route("/video_feed")
def video_feed():
    return Response(generer_flux_video(), mimetype="multipart/x-mixed-replace; boundary=frame")


def creer_vue_en_direct():
    return html.Div([
        html.Div([
            html.Div([
                html.Div("Flux video", style=STYLE_SOUS_TITRE),
                html.Img(
                    src="/video_feed",
                    style={"width": "100%", "borderRadius": "10px", "display": "block"},
                ),
            ], style={**STYLE_CARTE, "flex": "1 1 640px"}),

            html.Div([
                html.Div([
                    html.Div("Score de vigilance", style=STYLE_SOUS_TITRE),
                    html.Div(id="score-en-direct", style={"fontSize": "40px", "fontWeight": "800"}),
                    html.Div(id="niveau-en-direct", style={"fontSize": "16px", "marginTop": "4px"}),
                ], style=STYLE_CARTE),

                html.Div([
                    html.Div("Compteurs", style=STYLE_SOUS_TITRE),
                    html.Div(id="compteurs-en-direct", style={"fontSize": "15px", "lineHeight": "1.8"}),
                ], style=STYLE_CARTE),

                html.Div(
                    id="message-en-direct",
                    style={
                        "fontSize": "15px", "fontWeight": "700", "minHeight": "20px",
                        **STYLE_CARTE,
                    },
                ),
            ], style={"flex": "1 1 320px", "display": "flex", "flexDirection": "column"}),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),

        html.Div([
            html.Div("EAR / MAR (temps reel)", style=STYLE_SOUS_TITRE),
            dcc.Graph(id="graphique-ear-mar", config={"displayModeBar": False}),
        ], style=STYLE_CARTE),

        dcc.Interval(id="intervalle-maj", interval=1000, n_intervals=0),
    ])


def creer_vue_historique():
    return html.Div([
        html.Div([
            html.Div("Sessions enregistrees", style=STYLE_SOUS_TITRE),
            html.Button(
                "Rafraichir la liste", id="bouton-rafraichir-sessions", style=STYLE_BOUTON
            ),
            dash_table.DataTable(
                id="table-sessions",
                columns=[
                    {"name": "ID", "id": "id"},
                    {"name": "Date", "id": "date"},
                    {"name": "Duree (s)", "id": "duree_secondes"},
                    {"name": "Score moyen", "id": "score_moyen"},
                    {"name": "Alertes", "id": "nb_alertes"},
                    {"name": "Baillements", "id": "nb_baillements"},
                    {"name": "Clignements", "id": "nb_clignements"},
                ],
                row_selectable="single",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": "#0f172a", "color": COULEUR_TEXTE,
                    "fontWeight": "700", "border": "none",
                },
                style_cell={
                    "backgroundColor": COULEUR_CARTE, "color": COULEUR_TEXTE,
                    "border": "1px solid #334155", "padding": "8px",
                },
                style_data_conditional=[{
                    "if": {"state": "selected"},
                    "backgroundColor": "#334155", "border": "1px solid " + COULEUR_ACCENT,
                }],
            ),
        ], style=STYLE_CARTE),

        html.Div([
            html.Div("Evolution du score de la session selectionnee", style=STYLE_SOUS_TITRE),
            dcc.Graph(id="graphique-session-historique", config={"displayModeBar": False}),
            html.Button("Exporter en CSV", id="bouton-export-csv", style=STYLE_BOUTON),
            dcc.Download(id="telechargement-csv"),
        ], style=STYLE_CARTE),
    ])


app.layout = html.Div([
    html.Div("Surveillance de la vigilance du conducteur", style=STYLE_TITRE_PAGE),
    dcc.Tabs(
        [
            dcc.Tab(
                label="Vue en direct", children=creer_vue_en_direct(),
                style={"backgroundColor": COULEUR_CARTE, "color": COULEUR_TEXTE_ATTENUE, "border": "none"},
                selected_style={
                    "backgroundColor": COULEUR_ACCENT, "color": "#0f172a",
                    "fontWeight": "700", "border": "none",
                },
            ),
            dcc.Tab(
                label="Vue historique", children=creer_vue_historique(),
                style={"backgroundColor": COULEUR_CARTE, "color": COULEUR_TEXTE_ATTENUE, "border": "none"},
                selected_style={
                    "backgroundColor": COULEUR_ACCENT, "color": "#0f172a",
                    "fontWeight": "700", "border": "none",
                },
            ),
        ],
        style={"marginBottom": "20px"},
    ),
], style=STYLE_PAGE)


@app.callback(
    Output("score-en-direct", "children"),
    Output("niveau-en-direct", "children"),
    Output("compteurs-en-direct", "children"),
    Output("message-en-direct", "children"),
    Output("graphique-ear-mar", "figure"),
    Input("intervalle-maj", "n_intervals"),
)
def maj_vue_en_direct(_):
    with etat.verrou:
        phase = etat.phase
        erreur = etat.erreur
        temps_restant_calibration = etat.temps_restant_calibration
        score = etat.score
        niveau = etat.niveau
        message = etat.message_recommandation or ""
        nb_clignements = etat.nb_clignements
        nb_baillements = etat.nb_baillements
        frequence = etat.frequence_clignements
        duree = etat.duree_ecoulee
        points = list(etat.historique_courbes)

    figure_vide = {"data": [], "layout": GABARIT_GRAPHIQUE}

    if phase == "erreur":
        return (
            "--", "Erreur",
            "",
            html.Div(f"Calibration echouee : {erreur}. Redemarrez l'application.",
                      style={"color": "#ef4444"}),
            figure_vide,
        )

    if phase == "calibration":
        secondes = f"{temps_restant_calibration:.0f}" if temps_restant_calibration is not None else "..."
        return (
            "--", "Calibration en cours",
            "",
            html.Div(
                f"Calibration en cours ({secondes}s restantes). "
                f"Gardez les yeux ouverts, visage neutre.",
                style={"color": COULEUR_ACCENT},
            ),
            figure_vide,
        )

    couleur = couleur_score_web(score)
    contenu_score = html.Span(f"{score:.0f}/100", style={"color": couleur})
    contenu_niveau = html.Span(f"Niveau : {niveau.upper()}", style={"color": couleur, "fontWeight": "600"})
    texte_compteurs = html.Div([
        html.Div(f"Clignements : {nb_clignements} ({frequence:.1f}/min)"),
        html.Div(f"Baillements : {nb_baillements}"),
        html.Div(f"Duree ecoulee : {duree:.0f}s"),
    ])
    contenu_message = html.Div(message, style={"color": couleur if message else COULEUR_TEXTE_ATTENUE})

    temps = [p[0] for p in points]
    figure = {
        "data": [
            {"x": temps, "y": [p[1] for p in points], "type": "line", "name": "EAR", "line": {"color": COULEUR_ACCENT}},
            {"x": temps, "y": [p[2] for p in points], "type": "line", "name": "MAR", "line": {"color": "#f472b6"}},
        ],
        "layout": {**GABARIT_GRAPHIQUE, "margin": {"t": 20, "b": 30, "l": 40, "r": 20}},
    }

    return contenu_score, contenu_niveau, texte_compteurs, contenu_message, figure


@app.callback(
    Output("table-sessions", "data"),
    Input("bouton-rafraichir-sessions", "n_clicks"),
)
def maj_liste_sessions(_):
    bdd = BaseDeDonnees()
    sessions = bdd.lister_sessions()
    bdd.fermer()
    return sessions


@app.callback(
    Output("graphique-session-historique", "figure"),
    Input("table-sessions", "selected_rows"),
    State("table-sessions", "data"),
)
def maj_graphique_historique(lignes_selectionnees, donnees_sessions):
    if not lignes_selectionnees or not donnees_sessions:
        return {"data": [], "layout": {**GABARIT_GRAPHIQUE, "title": "Selectionnez une session"}}

    session_id = donnees_sessions[lignes_selectionnees[0]]["id"]
    bdd = BaseDeDonnees()
    mesures = bdd.obtenir_mesures(session_id)
    bdd.fermer()

    return {
        "data": [{
            "x": [m["timestamp"] for m in mesures],
            "y": [m["score_vigilance"] for m in mesures],
            "type": "line", "name": "Score", "line": {"color": COULEUR_ACCENT},
        }],
        "layout": {
            **GABARIT_GRAPHIQUE,
            "title": f"Session {session_id}",
            "yaxis": {"range": [0, 100]},
            "margin": {"t": 40, "b": 30, "l": 40, "r": 20},
        },
    }


@app.callback(
    Output("telechargement-csv", "data"),
    Input("bouton-export-csv", "n_clicks"),
    State("table-sessions", "selected_rows"),
    State("table-sessions", "data"),
    prevent_initial_call=True,
)
def exporter_csv(n_clicks, lignes_selectionnees, donnees_sessions):
    if not lignes_selectionnees or not donnees_sessions:
        return None

    session_id = donnees_sessions[lignes_selectionnees[0]]["id"]
    bdd = BaseDeDonnees()
    mesures = bdd.obtenir_mesures(session_id)
    bdd.fermer()

    df = pd.DataFrame(mesures)
    return dcc.send_data_frame(df.to_csv, f"session_{session_id}.csv", index=False)


def main():
    webcam = Webcam()
    webcam.ouvrir()
    detecteur = DetecteurLandmarks()

    # La calibration se déroule dans le thread d'arrière-plan et s'affiche
    # directement dans le flux vidéo web (onglet "Vue en direct"), pas dans
    # une fenêtre OpenCV séparée.
    thread = threading.Thread(target=boucle_traitement, args=(webcam, detecteur), daemon=True)
    thread.start()

    print("Tableau de bord disponible sur http://127.0.0.1:8050")
    # debug=False : le rechargeur de Flask en mode debug lancerait un 2e
    # thread concurrent sur la meme webcam.
    app.run(debug=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
