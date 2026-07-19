# CLAUDE.md — Système Intelligent de Surveillance de la Vigilance du Conducteur

Ce fichier donne le contexte complet du projet à Claude Code. Lis-le entièrement avant de commencer à coder. En cas de doute sur le scope, privilégier la simplicité et demander plutôt que de complexifier.

---

## 1. Contexte du projet

Projet de fin d'études, Licence 3 Big Data / Data Science. **Objectif pédagogique** : démontrer des compétences en IA, vision par ordinateur, machine learning, analyse de données, développement logiciel et conception de systèmes intelligents — **pas** un projet de recherche académique, pas de publication visée.

### Problème traité

La somnolence au volant est une cause majeure d'accidents de la route :

- Mondialement, l'OMS recense près de 3 500 morts par jour sur les routes.
- En France, un accident mortel sur trois sur autoroute serait lié à la somnolence (~30% des accidents autoroutiers, ~20% sur route).
- Au Sénégal, ~5 200 accidents/an, 745 décès, 8 500 blessés graves ; le facteur humain (dont la fatigue) représente 90% des causes.

### Dimension sociale du projet

Un système low-cost basé sur une simple webcam (pas de capteur dédié) a une pertinence d'accessibilité : rendre disponible à des conducteurs qui n'ont pas accès aux DMS premium (Smart Eye, Seeing Machines, intégrés aux véhicules haut de gamme) une protection basique — taxis, cars, routiers. **Ne pas présenter le prototype comme une solution de déploiement réel** : c'est une motivation et une piste, pas un livrable déployé.

### Positionnement par rapport aux systèmes existants (Smart Eye, Seeing Machines, Tesla, Volvo, Mercedes, BMW)

Le but n'est **pas** de les dépasser, mais de comprendre et reproduire leurs principes (détection visage/landmarks, EAR/PERCLOS, seuils comportementaux), avec un axe de différenciation clair : la **calibration personnalisée** (les systèmes commerciaux utilisent souvent des seuils universels, pénalisant les morphologies atypiques — yeux naturellement petits, etc.).

---

## 2. Portée du projet — GARDE-FOUS IMPORTANTS

Ce projet doit rester réaliste pour une personne seuleisd, sans GPU ni moyens industriels.

- ✅ Utiliser MediaPipe (modèle pré-entraîné) pour la détection visage/landmarks — ne jamais réentraîner un détecteur de visage from scratch.
- ✅ **Le cœur du mémoire est la comparaison seuil fixe vs seuil personnalisé (calibration)** — c'est la vraie contribution du projet, à évaluer rigoureusement en priorité (retour de l'encadrant, cf. section 4).
- ✅ Un classifieur ML simple (scikit-learn : régression logistique, SVM, ou Random Forest) est **optionnel / bonus si le temps le permet une fois la comparaison seuil fixe vs personnalisé solidement évaluée** — **pas de deep learning (CNN/LSTM/Transformer) sauf demande explicite de l'utilisatrice.**
- ❌ Ne pas intégrer YOLO ou de détection d'objets sauf si la fonctionnalité "détection de téléphone/distraction" est explicitement demandée.
- ❌ Ne pas ajouter de fonctionnalités de la liste "optionnelles" (section 4) sans que ce soit demandé — construire d'abord le cœur du projet.
- ❌ Ne pas viser un déploiement embarqué réel (véhicule, matériel dédié) — le prototype tourne sur PC/laptop avec webcam.

---

## 3. Stack technique retenue

| Techno                                | Rôle                                                                                 |
| ------------------------------------- | ------------------------------------------------------------------------------------- |
| Python 3.x                            | Langage principal                                                                     |
| OpenCV                                | Capture vidéo, traitement d'image, affichage                                         |
| MediaPipe (Face Mesh)                 | Détection visage + 468 landmarks faciaux (pré-entraîné, ne pas réentraîner)     |
| NumPy                                 | Calculs géométriques (distances, ratios)                                            |
| scikit-learn                          | Classifieur ML (logistic regression / SVM / random forest) sur indicateurs tabulaires |
| pandas                                | Manipulation des données tabulaires (extraction dataset, features)                   |
| SQLite (via`sqlite3` ou SQLAlchemy) | Stockage de l'historique des sessions                                                 |
| Streamlit                             | Tableau de bord (vue temps réel + vue historique)                                    |
| playsound / pygame (au choix)         | Alerte sonore                                                                         |

---

## 4. Fonctionnalités, par priorité

### Indispensables (à construire en premier, dans cet ordre)

1. Capture webcam (OpenCV) + détection visage/landmarks (MediaPipe Face Mesh).
2. Calcul EAR (Eye Aspect Ratio) en temps réel → détection fermeture prolongée des yeux.
3. Calcul MAR (Mouth Aspect Ratio) en temps réel → détection bâillements.
4. Calcul de la fréquence de clignement (comptage sur fenêtre glissante).
5. Phase de calibration initiale (10-20 secondes au démarrage) : mesurer l'EAR/MAR "normal" de la personne, en déduire des seuils personnalisés.
6. Score de vigilance (0-100) combinant ces indicateurs.
7. Lissage temporel : ne déclencher une alerte que si un seuil est dépassé pendant plusieurs frames consécutives (équivalent >500ms pour une fermeture d'yeux = microsommeil comportemental, cf. section 6).
8. Système d'alerte graduée (visuelle à l'écran + sonore).
9. Historique basique de la session en cours (courbes des indicateurs dans le temps).

### Importantes (après que le cœur fonctionne)

**Priorité n°1 dans ce groupe : évaluer rigoureusement seuils fixes vs seuils personnalisés** (precision/recall/F1, taux de faux positifs, sur des enregistrements webcam maison) — c'est la vraie contribution du mémoire, à sécuriser avant tout le reste ci-dessous.

- Orientation de la tête (pitch/yaw/roll via landmarks + `cv2.solvePnP`) → détection tête qui tombe / dodeline.
- Direction du regard (optionnel si le temps le permet).
- Base de données SQLite : historique multi-sessions.
- Tableau de bord Streamlit avec vue temps réel + vue historique.
- Recommandations textuelles contextuelles ("Une pause est recommandée").
- Export CSV d'une session.

### Optionnelles / bonus (ne pas construire sans demande explicite, ou seulement si le temps le permet en fin de projet)

- **Classifieur ML (scikit-learn) entraîné sur indicateurs extraits d'un dataset public, comparé au seuillage à seuils fixes/personnalisés — décision de l'encadrant (2026-07-19) : scope de L3 trop large pour inclure ceci comme indispensable, devient un bonus, pas une priorité.**
- Détection de distraction (téléphone) via YOLO pré-entraîné.
- Calibration continue (mise à jour glissante des seuils pendant la session).
- Mode nuit (adaptation luminosité).
- Profils multi-conducteurs.
- Isolement du conducteur parmi plusieurs visages (si plusieurs visages détectés, prendre celui dont la bounding box est la plus grande = le plus proche de la caméra).
- Détection précoce par tendance (pente d'évolution des indicateurs, pas juste seuil instantané).

---

## 5. Architecture (pipeline)

```
Webcam → OpenCV (lecture frames)
       → MediaPipe Face Mesh (landmarks)
       → Calcul indicateurs (EAR, MAR, fréquence clignement, [tête])
       → Module de calibration (seuils personnalisés)
       → Module de scoring (score de vigilance 0-100)
       → Module de décision (lissage temporel, niveaux normal/attention/alerte)
       → Module d'alerte (son + visuel + recommandation)
       → SQLite (historique de session)
       → Streamlit (dashboard temps réel + historique)
```

Développer et tester chaque bloc indépendamment avant de passer au suivant — toujours garder un état fonctionnel démontrable.

---

## 6. Détails scientifiques/techniques des indicateurs

### EAR (Eye Aspect Ratio)

Ratio géométrique calculé à partir de 6 landmarks par œil (distances verticales / distance horizontale). Diminue fortement quand l'œil se ferme. Formule standard (landmarks p1..p6 par œil) :

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

Avec MediaPipe Face Mesh, utiliser les indices de landmarks correspondant aux contours de l'œil (à documenter dans le code, ex. commentaire listant les indices utilisés).

### MAR (Mouth Aspect Ratio)

Même logique géométrique appliquée aux landmarks de la bouche (ouverture verticale / largeur horizontale). Un MAR élevé et soutenu (~1-3 secondes) = bâillement.

### PERCLOS

Percentage of eyelid closure — proportion du temps où l'œil est fermé à plus de 80% sur une fenêtre glissante (ex. 60 secondes). Référence scientifique : Wierwille et al. (1994), validé par Dinges & Grace (1998) comme le meilleur indicateur de baisse de vigilance.

### Microsommeil comportemental

Défini par une fermeture des yeux > 500 ms (au-delà d'un clignement normal, qui dure 100-400 ms). C'est le seuil de référence à utiliser pour le lissage temporel de la détection d'alerte "yeux fermés".

### Signes précurseurs de l'endormissement à surveiller (avant le microsommeil lui-même)

- Ralentissement de la vitesse de fermeture/ouverture des paupières.
- Fréquence de clignement qui augmente puis devient irrégulière.
- Tête qui commence à dodeliner / s'affaisser progressivement.
- Bâillements répétés.

### Fenêtre glissante temps réel

Le système doit garder en mémoire un historique glissant des indicateurs (pas seulement l'état de la frame courante) pour calculer :

- la fréquence de clignement (comptage sur les X dernières secondes),
- le PERCLOS (proportion sur fenêtre, ex. 60s),
- une tendance/pente (si implémenté).

Implémentation suggérée : `collections.deque(maxlen=N)` pour stocker les valeurs EAR/MAR/timestamps des N dernières frames, ou une structure basée sur le temps écoulé plutôt que sur le nombre de frames (plus robuste aux variations de FPS).

### Rappel déontologique à respecter dans le code et les messages utilisateur

Le système **ne pose jamais de diagnostic médical**. Les messages d'alerte/recommandation doivent parler de "signes de fatigue" ou "score de vigilance faible", jamais de termes médicaux diagnostiques (ex. ne pas dire "vous souffrez de narcolepsie" ou équivalent).

---

## 7. Calibration personnalisée

- Au démarrage d'une session, phase de calibration de 10-20 secondes où l'utilisateur garde les yeux ouverts normalement, visage neutre.
- Calculer l'EAR moyen et le MAR moyen sur cette période → baseline individuelle.
- Seuils personnalisés dérivés de la baseline (ex. seuil de fermeture = 70-75% de l'EAR de baseline — valeur à ajuster empiriquement, cf. littérature : certains travaux utilisent 75% de l'EAR baseline et 140% du MAR baseline).
- Stocker cette baseline (au moins pour la durée de la session ; en base si profils multi-conducteurs implémentés).

---

## 8. Base de données (SQLite) — schéma suggéré

```sql
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    duree_secondes INTEGER,
    score_moyen REAL,
    nb_alertes INTEGER,
    nb_baillements INTEGER,
    nb_clignements INTEGER
);

CREATE TABLE mesures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    timestamp REAL,
    ear REAL,
    mar REAL,
    score_vigilance REAL,
    alerte BOOLEAN,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
```

Adapter librement selon les besoins, mais garder une séparation claire session / mesures dans le temps.

---

## 9. Tableau de bord (Streamlit)

**Vue en direct** :

- Flux vidéo avec landmarks affichés en overlay.
- Score de vigilance affiché en grand avec code couleur (vert/orange/rouge).
- Courbes temps réel EAR/MAR (dernières minutes).
- Compteurs : clignements, bâillements, durée écoulée.
- Message de recommandation contextuel.

**Vue historique** :

- Liste des sessions passées (date, durée, nb alertes).
- Graphique d'évolution du score sur une session sélectionnée.
- Statistiques agrégées.
- Export CSV.

---

## 10. Machine Learning (classifieur) — BONUS, pas indispensable

**Retour de l'encadrant (2026-07-19) : le scope initial (seuils fixes + seuils personnalisés + ML + validation croisée + tests de robustesse + démo live) est trop large pour un mémoire de L3.** Le classifieur ML devient une fonctionnalité bonus, à construire seulement une fois que la comparaison seuil fixe vs seuil personnalisé est solidement évaluée (c'est elle la vraie contribution du mémoire). Ne pas commencer cette section avant que le cœur ci-dessus soit terminé et démontrable.

- Modèle : régression logistique, SVM, ou Random Forest (scikit-learn) — comparer 2-3 modèles simples, pas plus.
- Features d'entrée : EAR, MAR, fréquence de clignement, éventuellement angle de tête — features tabulaires calculées, pas d'image brute en entrée.
- Label : vigilant / somnolent (binaire), dérivé des annotations du dataset public utilisé.
- Évaluation : precision, recall, F1-score, taux de faux positifs, comparé au seuillage géométrique fixe/personnalisé (le classifieur doit être comparé à la baseline géométrique, pas seulement évalué seul).

---

## 11. Datasets publics à utiliser pour l'évaluation/entraînement — priorité BASSE désormais

**Cette section devient secondaire suite à la réduction de scope (section 4/10) : le cœur du mémoire (seuil fixe vs personnalisé) s'évalue avec des enregistrements webcam maison, pas avec ces datasets.** Ils ne redeviennent utiles que si le classifieur ML bonus est effectivement construit, ou pour tester la robustesse du pipeline sur des visages/conditions variés.

| Dataset                                                         | Usage                                                                                        | Accès (vérifié 2026-07-19)                                                                 |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| NTHU-DDD                                                        | Test du pipeline (landmarks) en conditions variées (lunettes, nuit) + extraction d'exemples | ⚠️ Nécessite de remplir un *Dataset License Agreement* et de l'envoyer par email au NTHU CVLab (page officielle : cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/). Délai de réponse incertain (potentiellement plusieurs semaines) — **si utilisé, lancer la demande immédiatement.** |
| YawDD                                                           | Complète pour la détection de bâillements                                                 | Généralement soumis à une procédure de demande similaire (email aux auteurs) — à vérifier au cas par cas si retenu. |
| UTA-RLDD / dataset dérivé "DDD" (images yeux ouverts/fermés) | Entraînement rapide du classifieur ML                                                       | Semble plus directement accessible : page officielle sites.google.com/view/utarldd/home, également disponible via des miroirs Kaggle. Vérifier la licence exacte avant usage dans le mémoire. |

**Recommandation** : ne pas bloquer l'avancement du cœur du projet sur ces demandes d'accès. Si le classifieur ML bonus est envisagé plus tard, lancer la demande NTHU-DDD dès que possible en parallèle (délai long), et privilégier UTA-RLDD (accès plus rapide) en attendant.

Pour la démo devant jury : utiliser une vidéo de l'utilisatrice elle-même (webcam), pas une vidéo tierce trouvée sur internet (problèmes de droits + absence de vérité terrain).

---

## 12. Difficultés connues et solutions attendues dans le code

| Difficulté                         | Solution à implémenter                                                                                          |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Faux positifs sur clignement normal | Lissage temporel (seuil de durée avant déclenchement d'alerte, cf. 500ms)                                       |
| Faible luminosité                  | Prétraitement optionnel (égalisation d'histogramme) avant détection                                            |
| Lunettes                            | Documenter comme limite connue, tester explicitement, ne pas cacher les échecs dans les résultats               |
| Yeux naturellement petits           | Résolu par la calibration personnalisée — à valider explicitement en comparant seuils fixes vs personnalisés |

---

## 13. Conventions de code

- Code en Python, commenté en français (cohérent avec le mémoire rédigé en français).
- Structurer le projet en modules séparés, par exemple :

```
/src
  capture.py        # webcam + lecture frames
  landmarks.py       # wrapper MediaPipe
  indicateurs.py      # calcul EAR, MAR, PERCLOS, etc.
  calibration.py
  scoring.py
  alertes.py
  database.py
  dashboard.py        # app Streamlit
  ml/
    train_classifier.py
    features.py
/data                 # extraits de datasets, non versionnés si volumineux
/models                # modèle ML entraîné sauvegardé (ex. .pkl)
CLAUDE.md
README.md
```

- Toujours garder un point d'entrée simple pour tester (ex. `python src/main.py` lance la démo webcam de base).
- Prioriser la lisibilité et la modularité sur l'optimisation — c'est un projet pédagogique, le code doit être facile à expliquer devant un jury.

---

## 14. Langue

Toute communication avec l'utilisatrice, commentaires de code, messages d'interface (dashboard, alertes) et documentation doivent être en **français**.
