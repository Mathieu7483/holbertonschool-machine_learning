<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/unsupervised_learning/gan/GANs.png"\>
</p>
---

# Generative Adversarial Networks (GANs)

## 🎨 Description

Ce projet aborde l'un des domaines les plus fascinants du Deep Learning : les **Réseaux Antagonistes Générateurs** (*Generative Adversarial Networks* ou **GANs**). Introduits par Ian Goodfellow et al. en 2014, les GANs reposent sur la théorie des jeux à deux joueurs où deux réseaux de neurones s'affrontent :

1. **Le Générateur ($G$)** : Tente de produire des données synthétiques (images, signaux) réalistes à partir d'un vecteur de bruit aléatoire dans un espace latent.
2. **Le Discriminateur ($D$)** : Tente d'analyser une donnée en entrée et d'estimer si elle provient du vrai jeu de données (échantillon réel) ou du Générateur (échantillon artificiel).

Ce projet couvre la progression théorique et pratique depuis les GANs originaux jusqu'aux architectures avancées avec pénalité de gradient (WGAN-GP) et générateurs convolutionnels pour la synthèse de visages humains.

---

## 📐 Progression & Architecture des Modèles

```
[ Bruit z ~ p_z ] ---> ( Générateur G ) ---> x_fake
                                                 |
                                                 v
[ Données réelles x_real ] -----------------> ( Discriminateur D ) ---> Score / Probabilité

```

### 1. Simple GAN (Vanilla GAN)

* **Principe** : Jeu à somme nulle classique utilisant la perte *Binary Cross-Entropy* ou *MSE*.
* **Problématique** : Instabilité lors de l'entraînement, effondrement de mode (*mode collapse*), gradients disparus quand le discriminateur devient trop performant.

### 2. Wasserstein GAN avec Weight Clipping (WGAN-Clip)

* **Principe** : Utilise la distance de Wasserstein (Distance du Terrassier / *Earth Mover's Distance*) pour mesurer l'écart entre les distributions réelle et générée.
* **Contrainte** : Le discriminateur (appelé *Critic*) doit respecter une contrainte de **Lipschitzianité 1**. Elle est forcée en tronquant (*clipping*) les poids du réseau dans un intervalle fermé $[-c, c]$.

### 3. Wasserstein GAN avec Gradient Penalty (WGAN-GP)

* **Principe** : Remplace le découpage des poids (*weight clipping*) instable par une pénalité explicite sur la norme du gradient de la fonction du *Critic* par rapport aux données interpolées :

$$\mathcal{L}_{GP} = \mathbb{E}_{\hat{x}} \left[ \left( \Vert{}\nabla_{\hat{x}} D(\hat{x})\Vert{}_2 - 1 \right)^2 \right]$$

### 4. DCGAN & Génération de Visages

* Utilisation de couches convolutionnelles transposées (`Conv2DTranspose`) et de blocs de convolution pour traiter des images 2D à haute résolution (ex: synthèse de visages réels).

---

## 🛠️ Spécifications techniques

* **Environnement** : Ubuntu 20.04 LTS | **Python** : 3.9
* **Frameworks & Dépendances** :
* `TensorFlow` (v2.15.0)
* `NumPy` (v1.25.2)
* `Matplotlib`


* **Style de code** : Respect de la norme `pycodestyle` (version 2.11.1).
* **Shebang** : `#!/usr/bin/env python3` obligatoire.

---

## 📂 Architecture des Tâches

| # | Tâche | Fichier | Description |
| --- | --- | --- | --- |
| **0** | **Simple GAN** | `0-simple_gan.py` | Implémentation de la classe `Simple_GAN` et surcharge de `train_step()`. |
| **1** | **WGAN-Clip** | `1-wgan_clip.py` | Implémentation du Wasserstein GAN avec découpage des poids (*weight clipping*). |
| **2** | **WGAN-GP** | `2-wgan_gp.py` | Implémentation de la pénalité de gradient (*Gradient Penalty*) pour stabiliser l'apprentissage. |
| **3** | **Convolutional Architectures** | `3-convolutional.py` | Générateur et Discriminateur basés sur des architectures Deep Convolutional (DCGAN). |
| **4** | **Face Generator** | `4-face_generator.py` | Entraînement complet d'un WGAN-GP pour générer des portraits de visages artificiels. |

---

## 🔬 Focus Tâche 0 : Implémentation de `Simple_GAN`

### Structure de `train_step()`

Pour chaque étape d'entraînement, le discriminateur est mis à jour `disc_iter` fois, puis le générateur est mis à jour une fois :

```python
def train_step(self, data):
    # 1. Entraînement du Discriminateur (disc_iter répétitions)
    for _ in range(self.disc_iter):
        real_samples = self.get_real_sample()
        fake_samples = self.get_fake_sample(training=True)
        
        with tf.GradientTape() as tape:
            pred_real = self.discriminator(real_samples, training=True)
            pred_fake = self.discriminator(fake_samples, training=True)
            discr_loss = self.discriminator.loss(pred_real, pred_fake)
            
        grads = tape.gradient(discr_loss, self.discriminator.trainable_variables)
        self.discriminator.optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_variables)
        )

    # 2. Entraînement du Générateur (1 seule fois)
    fake_samples = self.get_fake_sample(training=True)
    with tf.GradientTape() as tape:
        pred_fake = self.discriminator(fake_samples, training=True)
        gen_loss = self.generator.loss(pred_fake)
        
    grads = tape.gradient(gen_loss, self.generator.trainable_variables)
    self.generator.optimizer.apply_gradients(
        zip(grads, self.generator.trainable_variables)
    )

    return {"discr_loss": discr_loss, "gen_loss": gen_loss}

```

---

## ⚙️ Compilation et Exécution

Pour tester l'entraînement et afficher la visualisation 2D d'un Simple GAN :

```bash
chmod +x 0-simple_gan.py 0-main_02.py
./0-main_02.py

```

### Résultat de l'entraînement 2D

L'exécution produit une comparaison entre la distribution des points réels (échantillon bleu) et des points générés par le GAN (échantillon orange), ainsi que la carte de chaleur des valeurs prédites par le discriminateur.

---

## ✍️ Auteur

* **Mathieu** - *Holberton School* - [Profil GitHub](https://www.google.com/search?q=https://github.com/Mathieu7483)
