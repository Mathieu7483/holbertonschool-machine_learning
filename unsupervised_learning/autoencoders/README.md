<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/unsupervised_learning/autoencoders/Autoencoders.png"\>
</p>

---

# Unsupervised Learning — Autoencoders

## 🧠 Description

Ce projet explore l'univers des **Autoencodeurs** en Apprentissage Non Surveillé (*Unsupervised Learning*). Un autoencodeur est une architecture de réseau de neurones artificiels conçue pour apprendre une représentation compressée (codage) de données non étiquetées, puis pour reconstruire la donnée d'origine à partir de ce code.

L'objectif est d'explorer la réduction de dimensionnalité, l'apprentissage dans l'espace latent (*latent space*), la régularisation de représentations et la génération de données à travers quatre architectures fondamentales :

1. **"Vanilla" Autoencoder** (Autoencodeur dense classique)
2. **Sparse Autoencoder** (Autoencodeur clairsemé avec pénalité L1)
3. **Convolutional Autoencoder** (Autoencodeur basé sur des couches CNN pour images)
4. **Variational Autoencoder (VAE)** (Autoencodeur variationnel génératif basé sur la divergence Kullback-Leibler)

---

## 📐 Concepts Clés

```
    [ Entrée X ]  --->  ( Encodeur )  --->  [ Espace Latent (Bottleneck) Z ]  --->  ( Décodeur )  --->  [ Reconstitution X' ]

```

* **Encodeur (*Encoder*)** : Compresse les données d'entrée à haute dimension vers un espace représentatif restreint (*bottleneck*).
* **Espace Latent (*Latent Space*)** : Espace vectoriel de dimension réduite capturant les caractéristiques essentielles des données.
* **Décodeur (*Decoder*)** : Reconstruit la donnée initiale à partir des vecteurs de l'espace latent.
* **Fonction de Perte (*Reconstruction Loss*)** : Mesure l'écart entre l'entrée $X$ et la sortie reconstituée $X'$ (ex: *Binary Cross-Entropy* ou *MSE*).
* **Divergence Kullback-Leibler (KL)** : Mesure utilisée dans les VAE pour forcer la distribution de l'espace latent à suivre une loi normale standard $\mathcal{N}(0, I)$.

---

## 🛠️ Spécifications & Recommandations

* **Environnement** : Ubuntu 20.04 LTS | **Python** : 3.9
* **Bibliothèques principales** :
* `TensorFlow` / `Keras` (v2.15)
* `NumPy` (v1.25.2)


* **Style de code** : Norme `pycodestyle` (version 2.11.1).
* **Imports** : Sauf mention contraire, seul `import tensorflow.keras as keras` est autorisé.
* **Documentation** : *Docstrings* obligatoires pour chaque module, classe et fonction.

---

## 📂 Architecture des Tâches

| # | Tâche | Fichier | Description |
| --- | --- | --- | --- |
| **0** | **"Vanilla" Autoencoder** | `0-vanilla.py` | Autoencodeur fully-connected standard compressant les données dans un espace latent fixe. |
| **1** | **Sparse Autoencoder** | `1-sparse.py` | Autoencodeur dense incluant une régularisation L1 (*sparsity constraint*) sur l'espace latent. |
| **2** | **Convolutional Autoencoder** | `2-convolutional.py` | Autoencodeur utilisant `Conv2D` et `MaxPooling2D` pour l'encodage, `UpSampling2D` pour le décodage. |
| **3** | **Variational Autoencoder** | `3-variational.py` | Modèle génératif échantillonnant l'espace latent $(\mu, \sigma)$ via l'astuce de reparamétrisation (*reparameterization trick*). |

---

## 🔬 Focus Tâche 0 : "Vanilla" Autoencoder

### Prototype de la fonction

```python
def autoencoder(input_dims, hidden_layers, latent_dims):

```

* **`input_dims`** *(int)* : Dimensionnalité de la donnée d'entrée.
* **`hidden_layers`** *(list)* : Nombre de nœuds pour chaque couche cachée de l'encodeur (l'ordre est inversé pour le décodeur).
* **`latent_dims`** *(int)* : Dimension de l'espace latent.
* **Retour** : `(encoder, decoder, auto)`
* `encoder` : Modèle Keras de l'encodeur.
* `decoder` : Modèle Keras du décodeur.
* `auto` : Modèle Keras de l'autoencodeur complet.



### Configuration du Modèle :

* **Activations** : `relu` pour toutes les couches cachées, `sigmoid` pour la dernière couche du décodeur.
* **Optimiseur** : `adam`
* **Loss** : `binary_crossentropy`

---

## ⚙️ Compilation et Exécution

Pour tester la Tâche 0 avec le dataset MNIST :

```bash
chmod +x 0-main.py 0-vanilla.py
./0-main.py

```

### Exemple de comportement (Entraînement) :

```text
Epoch 1/50
235/235 [==============================] - 3s 10ms/step - loss: 0.2462 - val_loss: 0.1704
Epoch 2/50
235/235 [==============================] - 2s 10ms/step - loss: 0.1526 - val_loss: 0.1370
...
Epoch 50/50
235/235 [==============================] - 3s 12ms/step - loss: 0.0846 - val_loss: 0.0842

```

---

## ✍️ Auteur

* **Mathieu** - *>Machine Learning Student at Holberton School* - [Profil GitHub](https://www.google.com/search?q=https://github.com/Mathieu7483)
