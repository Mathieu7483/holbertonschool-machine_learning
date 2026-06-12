<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/supervised_learning/neural_style_transfer/a-realistic--cinematic-photograph-of-a-high-end-du.png"\>
</p>

# Neural Style Transfer (NST)

## 🎨 Description

Ce projet implémente l'algorithme révolutionnaire décrit par Gatys et al. dans *A Neural Algorithm of Artistic Style*. L'objectif est de générer une nouvelle image combinant le contenu sémantique d'une image de référence et l'empreinte stylistique d'une œuvre d'art. En utilisant l'architecture **VGG-19** pré-entraînée sur ImageNet, le projet extrait les activations de différentes couches convolutives pour calculer indépendamment le coût de contenu (*Content Cost*) et le coût de style (*Style Cost*), ce dernier exploitant les propriétés mathématiques des matrices de Gram. L'optimisation s'effectue directement sur l'image générée en utilisant l'exécution dynamique (*Eager Execution*) et le calcul de gradients via `tf.GradientTape`.

## 🎓 Objectifs d'apprentissage

* **Principe du NST** : Comprendre comment définir et minimiser une fonction de coût jointe qui équilibre le style et le contenu.
* **Matrice de Gram** : Maîtriser le calcul des corrélations de caractéristiques inter-canaux pour capturer les textures, couleurs et motifs d'une image de style.
* **API Eager Execution & GradientTape** : Suivre et calculer manuellement les gradients de la fonction de perte par rapport aux pixels de l'image d'entrée (au lieu des poids du modèle).
* **Dénoyautage Variationnel** : Implémenter la perte de variation totale (*Total Variation Loss*) pour lisser l'image générée et éliminer le bruit haute fréquence (artefacts de pixels).

## 🛠️ Spécifications techniques

* **OS** : Ubuntu 20.04 LTS | **Python** : 3.9
* **Frameworks** : NumPy (1.25.2) et TensorFlow (2.15).
* **Style** : Respect strict et systématique des conventions de style de la norme `pycodestyle` (2.11.1).
* **Contrainte d'importation** : Sauf indication contraire, seuls les imports `import numpy as np` et `import tensorflow as tf` sont autorisés.

---

## 📂 Liste des Tâches (Advanced Pipeline)

| # | Tâche | Fichier | Description |
| --- | --- | --- | --- |
| **0** | **Initialize** | `0-neural_style.py` | Initialisation de la classe `NST`, validation des entrées et méthode statique de redimensionnement/normalisation d'images (`scale_image`). |
| **1** | **Load the Model** | `1-neural_style.py` | Chargement du modèle VGG-19, gel des couches et configuration d'un sous-modèle ciblant uniquement les couches d'extraction requises. |
| **2** | **Gram Matrix** | `2-neural_style.py` | Implémentation du calcul mathématique de la matrice de Gram à partir d'un tenseur d'activation. |
| **3** | **Extract Features** | `3-neural_style.py` | Extraction des activations de style et de contenu pour les images de référence et calcul de leurs matrices de Gram respectives. |
| **4** | **Layer Style Cost** | `4-neural_style.py` | Calcul du coût de style pour une seule couche convolutive isolée. |
| **5** | **Style Cost** | `5-neural_style.py` | Calcul du coût de style global pondéré sur l'ensemble des couches de style configurées. |
| **6** | **Content Cost** | `6-neural_style.py` | Calcul du coût de contenu basé sur la distance euclidienne au carré des activations de la couche cible. |
| **7** | **Total Cost** | `7-neural_style.py` | Combinaison linéaire des coûts de contenu, de style et de leurs poids respectifs ($\alpha$ et $\beta$). |
| **8** | **Compute Gradients** | `8-neural_style.py` | Utilisation de `tf.GradientTape` pour calculer les gradients du coût total par rapport à l'image en cours de stylisation. |
| **9** | **Generate Image** | `9-neural_style.py` | Boucle principale d'optimisation mettant à jour l'image générée via une descente de gradient sur un nombre d'itérations donné. |
| **10** | **Variational Cost** | `10-neural_style.py` | Intégration de la perte de variation totale pour régulariser l'image et supprimer le bruit visuel. |

---

## 🔬 Focus Mathématique : L'Équilibre des Coûts

La fonction de coût totale $J(G)$ que l'algorithme cherche à minimiser en modifiant l'image générée $G$ est formulée de la manière suivante :

$$J(G) = \alpha J_{content}(C, G) + \beta J_{style}(S, G) + \gamma J_{TV}(G)$$

* **$J_{content}$** : Mesure l'écart entre les représentations de haut niveau de l'image de contenu $C$ et de l'image générée $G$.
* **$J_{style}$** : Repose sur la différence entre les matrices de Gram de l'image de style $S$ et de l'image générée $G$, calculée sur plusieurs couches (de `block1_conv1` à `block5_conv1`).
* **$J_{TV}$** : La perte de variation totale (Tâche 10), agissant comme un régulariseur pour assurer une cohérence spatiale lisse.

---

## 🛠️ Utilisation et Validation de la Tâche 0

La première étape consiste à valider le chargement et le redimensionnement d'images avec l'interpolation bicubique de TensorFlow, tout en s'assurant que le côté le plus grand de l'image n'excède jamais 512 pixels.

```bash
chmod +x 0-main.py
./0-main.py

```

### Format des tenseurs attendus :

* Images d'entrée converties en `EagerTensor` avec un format à 4 dimensions : `(1, h_new, w_new, 3)`.
* Valeurs des pixels normalisées à l'échelle $[0.0, 1.0]$.


## ✍️ Auteur

  * **Mathieu** - *Programming student, specialization Machine Learning* - [👤 My Github profile](https://github.com/Mathieu7483)