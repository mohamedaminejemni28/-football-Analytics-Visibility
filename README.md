# ⚽ Football Detection & Tracking

Ce projet implémente un système complet de détection, suivi et analyse de joueurs de football dans des vidéos de matchs. Il utilise la détection d'objets (YOLO), le tracking multi-objets (ByteTrack), et des outils d’analyse comme le calcul de la vitesse des joueurs et la possession du ballon.

## 📌 Objectifs

* Détecter les joueurs, ballons, arbitres dans une vidéo de match.
* Suivre les joueurs à travers les frames (tracking avec identifiants).
* Calculer et afficher en temps réel :

  * la vitesse des joueurs,
  * la possession de balle par équipe,
  * les statistiques visuelles en overlay (OpenCV).

## 🧰 Technologies utilisées

* 🔍 **YOLOv8** — pour la détection des objets (joueurs, ballon, arbitres)
* 🛰 **ByteTrack** — pour le tracking multi-objet
* 🧠 **KMeans / heuristiques** — pour identifier les équipes
* 📐 **OpenCV** — pour l’annotation et l’affichage des résultats
* 📊 **Pandas** — pour stocker et analyser les trajectoires
* 🐍 **Python**

## 🖼️ Exemples de Résultat

| Frame Annotée                                 | Statistiques                   |
| --------------------------------------------- | ------------------------------ |
| ![example\_frame](./assets/frame_example.jpg) | Vitesse, possession, ID joueur |

## 🚀 Installation

```bash
git clone https://github.com/mohamedaminejemni28/football-detection.git
cd football-detection
pip install -r requirements.txt
```

## 📁 Structure du projet

```
football-detection/
│
├── yolov8_weights/         # Modèle entraîné YOLOv8
├── trackers/               # Implémentation de ByteTrack
├── videos/                 # Vidéos d'entrée
├── outputs/                # Vidéos annotées en sortie
├── utils/                  # Fonctions utilitaires (vitesse, possession, etc.)
├── main.py                 # Script principal
├── README.md
└── requirements.txt
```

## ▶️ Utilisation

1. **Détection & tracking :**

```bash
python main.py --video ./videos/match1.mp4 --output ./outputs/match1_annotated.mp4
```

2. **Options supplémentaires :**

* `--display-speed` : affiche la vitesse des joueurs
* `--display-possession` : affiche la possession de balle

## 📊 Exemple de Statistiques Générées

* Vitesse moyenne par joueur
* Pourcentage de possession par équipe
* Nombre de sprints
* Cartographie de mouvement (à venir)

## 📌 Prochaines étapes

* [ ] Suivi de la trajectoire du ballon
* [ ] Identification automatique des événements (buts, fautes)
* [ ] Tableau de bord interactif avec Streamlit

## 🧑‍💻 Auteur

👤 **Mohamed Amine Jemni**
Étudiant à Sup'Com | Passionné d’IA, de vision par ordinateur et de football

