"""
Point d'entrée : boucle de capture webcam + détection et affichage des landmarks
du visage. Premier bloc du pipeline (cf. claude.md, section 5).
"""

import time

import cv2

from alertes import GestionnaireAlertes
from calibration import calibrer
from capture import Webcam
from clignement import CompteurClignements
from decision import Decision, NIVEAU_ALERTE, NIVEAU_ATTENTION
from historique import HistoriqueSession
from indicateurs import calculer_ear, calculer_mar
from landmarks import DetecteurLandmarks
from scoring import ScoreVigilance


def couleur_score(score):
    """Vert si vigilance correcte, orange si attention, rouge si alerte."""
    if score >= 70:
        return (0, 255, 0)
    if score >= 40:
        return (0, 165, 255)
    return (0, 0, 255)


def couleur_niveau(niveau):
    """Couleur associée au niveau de vigilance (normal/attention/alerte)."""
    if niveau == NIVEAU_ALERTE:
        return (0, 0, 255)
    if niveau == NIVEAU_ATTENTION:
        return (0, 165, 255)
    return (0, 255, 0)


def dessiner_landmarks(frame, landmarks):
    """Dessine chaque landmark du visage sous forme de petit point sur la frame."""
    hauteur, largeur, _ = frame.shape
    for point in landmarks:
        x = int(point.x * largeur)
        y = int(point.y * hauteur)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def main():
    webcam = Webcam()
    webcam.ouvrir()
    detecteur = DetecteurLandmarks()

    baseline = calibrer(webcam, detecteur)
    print(
        f"Calibration terminée : EAR moyen={baseline.ear_moyen:.3f} "
        f"(seuil clignement={baseline.seuil_fermeture_yeux:.3f}), "
        f"MAR moyen={baseline.mar_moyen:.3f} "
        f"(seuil baillement={baseline.seuil_baillement:.3f})"
    )

    compteur_clignements = CompteurClignements(seuil_ear=baseline.seuil_fermeture_yeux)
    score_vigilance = ScoreVigilance()
    decision = Decision()
    gestionnaire_alertes = GestionnaireAlertes()
    historique = HistoriqueSession()

    temps_precedent = time.time()

    try:
        while True:
            frame = webcam.lire_frame()
            if frame is None:
                print("Erreur de lecture de la webcam.")
                break

            frame = cv2.flip(frame, 1)  # effet miroir, plus naturel pour l'utilisateur
            landmarks = detecteur.detecter(frame)

            visage_detecte = landmarks is not None
            ear, mar = 0.0, 0.0
            if visage_detecte:
                dessiner_landmarks(frame, landmarks)
                hauteur, largeur, _ = frame.shape
                ear = calculer_ear(landmarks, largeur, hauteur)
                mar = calculer_mar(landmarks, largeur, hauteur)
                compteur_clignements.mettre_a_jour(ear)
                score, perclos = score_vigilance.calculer_score(
                    ear, mar, compteur_clignements.frequence_par_minute(), baseline
                )
                yeux_fermes = ear < baseline.seuil_fermeture_yeux
                niveau, microsommeil = decision.determiner_niveau(yeux_fermes, score)
                message_recommandation = gestionnaire_alertes.traiter(niveau)
                historique.enregistrer(ear, mar, score)

            # Calcul du FPS pour valider la stabilité du pipeline avant le bloc suivant
            temps_actuel = time.time()
            ecart = temps_actuel - temps_precedent
            fps = 1 / ecart if ecart > 0 else 0
            temps_precedent = temps_actuel

            statut = "Visage detecte" if visage_detecte else "Aucun visage"
            texte = f"{statut} | FPS: {fps:.1f}"
            cv2.putText(frame, texte, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if visage_detecte:
                texte_indicateurs = f"EAR: {ear:.3f} | MAR: {mar:.3f}"
                cv2.putText(
                    frame, texte_indicateurs, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                texte_clignements = (
                    f"Clignements: {compteur_clignements.total_clignements} "
                    f"| Frequence: {compteur_clignements.frequence_par_minute():.1f}/min"
                )
                cv2.putText(
                    frame, texte_clignements, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                texte_seuils = (
                    f"Seuils personnalises -> clignement: {baseline.seuil_fermeture_yeux:.3f} "
                    f"| baillement: {baseline.seuil_baillement:.3f}"
                )
                cv2.putText(
                    frame, texte_seuils, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
                )
                texte_score = f"Score de vigilance: {score:.0f}/100 (PERCLOS: {perclos:.2f})"
                cv2.putText(
                    frame, texte_score, (10, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, couleur_score(score), 2,
                )
                texte_niveau = f"Niveau: {niveau.upper()}"
                cv2.putText(
                    frame, texte_niveau, (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, couleur_niveau(niveau), 2,
                )
                if microsommeil:
                    cv2.putText(
                        frame, "MICROSOMMEIL DETECTE (yeux fermes > 500ms)", (10, 225),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                    )
                if message_recommandation:
                    cv2.putText(
                        frame, message_recommandation, (10, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur_niveau(niveau), 2,
                    )

            cv2.imshow("Surveillance de la vigilance - landmarks", frame)

            touche = cv2.waitKey(1) & 0xFF
            if touche == ord("q") or touche == 27:  # 'q' ou ESC
                break
    finally:
        webcam.fermer()
        detecteur.fermer()
        cv2.destroyAllWindows()

    historique.afficher_courbes()


if __name__ == "__main__":
    main()
