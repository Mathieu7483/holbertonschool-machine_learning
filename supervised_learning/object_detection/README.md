# Object Detection with YOLOv3

## 👁️ Description

Ce projet met en œuvre l'algorithme de détection d'objets **YOLOv3** (You Only Look Once, version 3) en combinant la puissance de **TensorFlow / Keras** et de la bibliothèque de traitement d'images **OpenCV**. Contrairement aux approches par fenêtres glissantes ou par propositions de régions (R-CNN), YOLO traite l'image entière en une seule passe réseau (*single-shot detector*), ce qui lui permet d'atteindre des performances en temps réel. Le projet consiste à encapsuler le chargement d'un modèle Darknet pré-entraîné sur le dataset MS COCO, à traiter ses sorties multi-échelles, et à appliquer les algorithmes géométriques de filtrage nécessaires pour restituer des prédictions propres.

## 🎓 Objectifs d'apprentissage

* **Single-Shot Detection** : Comprendre le paradigme de détection d'objets en une seule passe réseau.
* **Analyse de Tenseurs YOLO** : Décoder les tenseurs de sortie multi-échelles ($13\times13$, $26\times26$, $52\times52$) contenant les coordonnées, la confiance de l'objet et les probabilités de classe.
* **Intersection Over Union (IoU)** : Calculer le ratio d'intersection et d'union entre deux boîtes englobantes pour évaluer leur chevauchement.
* **Non-Max Suppression (NMS)** : Éliminer les boîtes redondantes pour ne conserver que la boîte la plus précise pour chaque objet.
* **Anchor Boxes** : Utiliser des boîtes ancres prédéfinies pour assister le modèle dans la détection d'objets de tailles et de ratios variés.
* **OpenCV** : Charger, redimensionner, et dessiner des indicateurs visuels (rectangles, textes) sur les images.

## 🛠️ Spécifications techniques

* **OS** : Ubuntu 20.04 LTS | **Python** : 3.9
* **Frameworks** : NumPy (1.25.2) et TensorFlow (2.15).
* **Vision par ordinateur** : OpenCV Python (`opencv-python==4.9.0.80`).
* **Style** : Respect strict du guide de style `pycodestyle` (2.11.1).

---

## 📂 Liste des Tâches

| # | Tâche | Fichier | Description |
| --- | --- | --- | --- |
| **0** | **Initialize Yolo** | `0-yolo.py` | Initialisation de la classe `Yolo`, chargement du modèle `.h5`, extraction des classes et définition des seuils. |
| **1** | **Process Outputs** | `1-yolo.py` | Extraction des prédictions de boîtes à partir des sorties réseau en appliquant la fonction sigmoïde sur les coordonnées relatives. |
| **2** | **Filter Boxes** | `2-yolo.py` | Élimination des boîtes dont le score de confiance global est inférieur au seuil `class_t`. |
| **3** | **Non-max Suppression** | `3-yolo.py` | Implémentation de l'algorithme NMS pour supprimer les boîtes superposées basées sur l'IoU. |
| **4** | **Load images** | `4-yolo.py` | Gestion du chargement d'images via OpenCV et récupération de leurs dimensions d'origine. |
| **5** | **Preprocess images** | `5-yolo.py` | Redimensionnement des images aux dimensions d'entrée du réseau ($416\times416$) et normalisation des pixels. |
| **6** | **Show boxes** | `6-yolo.py` | Dessin des boîtes englobantes et des étiquettes textuelles sur l'image d'origine, puis affichage. |
| **7** | **Predict** | `7-yolo.py` | Pipeline complet : chargement, prétraitement, prédiction, filtrage, NMS et affichage final. |

---

## 🔬 Focus Algorithmique : Le Tenseur de Sortie YOLOv3

YOLOv3 effectue des prédictions à 3 échelles différentes pour détecter aussi bien les grands que les petits objets. Pour chaque cellule de la grille (ex: $13 \times 13$), le modèle prédit $3$ Anchor Boxes.

Chaque Anchor Box fournit **85 valeurs** :

* **4 coordonnées** : $(t_x, t_y, t_w, t_h)$ qui décrivent la position et la taille de la boîte.
* **1 score de confiance** : $p_o$ (probabilité qu'un objet soit présent dans la boîte).
* **80 probabilités de classes** : Les scores pour les 80 catégories du dataset COCO (ex: person, car, dog...).

---

## 🛠️ Validation du Pipeline

L'exécution du script d'intégration final charge l'architecture Darknet et traite l'image de bout en bout :

```bash
chmod +x 0-main.py
./0-main.py

```

### Métriques de configuration d'usine :

* `class_t` (Seuil de confiance initial) : `0.6`
* `nms_t` (Seuil IoU pour la suppression des doublons) : `0.5`
* Format d'entrée réseau requis : $(416, 416, 3)$

## ✍️ Auteur

* **Mathieu** - *Étudiant en apprentissage Machine Learning (42 ans)* - [GitHub Profile](https://www.google.com/search?q=https://github.com/Mathieu7483)
