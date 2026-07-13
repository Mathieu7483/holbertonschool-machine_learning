<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/math/bayesian_prob/a-realistic--cinematic-photograph-of-a-deep-learni.png"\>
</p>

# Mathematics — Bayesian Probability

## 🧠 Description

Ce projet explore les concepts fondamentaux des statistiques et des probabilités bayésiennes. Contrairement à l'approche fréquentiste, l'approche bayésienne modélise l'incertitude sur les paramètres d'un modèle en mettant à jour une croyance initiale (*Prior*) à l'aide de la vraisemblance des données observées (*Likelihood*). À travers ce projet, nous implémentons pas à pas le théorème de Bayes appliqué à un cas d'étude clinique (effets secondaires d'un médicament) modélisé par une loi binomiale.

## 🎓 Objectifs d'apprentissage

* **Inférence Bayésienne** : Comprendre et appliquer le théorème de Bayes : $P(A\vert{}B) = \frac{P(B\vert{}A)P(A)}{P(B)}$.
* **Prior, Likelihood, Marginal & Posterior** :
* *Prior* : La probabilité a priori du paramètre avant observation.
* *Likelihood (Vraisemblance)* : La probabilité d'observer les données actuelles selon différentes hypothèses.
* *Marginal Probability* : La probabilité totale des données (le dénominateur de Bayes).
* *Posterior* : La probabilité mise à jour du paramètre après intégration des observations.


* **Distribution Binomiale** : Modéliser le nombre de succès $x$ parmi $n$ essais indépendants.

## 🛠️ Spécifications techniques

* **OS** : Ubuntu 20.04 LTS | **Python** : 3.9
* **Framework** : NumPy (1.25.2).
* **Style** : Respect strict de la norme `pycodestyle` (2.11.1).
* **Documentation** : Tous les modules, classes et fonctions doivent posséder des *docstrings* valides.

---

## 📂 Architecture des Tâches et Pipeline Bayésien

Le projet suit une progression mathématique linéaire où chaque script construit une brique du théorème de Bayes :

| # | Tâche | Fichier | Description |
| --- | --- | --- | --- |
| **0** | **Likelihood** | `0-likelihood.py` | Calcule la vraisemblance d'obtenir les données observées ($x$ succès sur $n$ patients) pour chaque hypothèse de probabilité présente dans le tableau $P$. |
| **1** | **Intersection** | `1-intersection.py` | Calcule l'intersection de la vraisemblance (*Likelihood*) et de la probabilité a priori (*Prior*) : $P(\text{Data} \cap \text{Hypothèse})$. |
| **2** | **Marginal Probability** | `2-marginal.py` | Détermine la probabilité marginale globale des données en sommant toutes les intersections (loi des probabilités totales). |
| **3** | **Posterior** | `3-posterior.py` | Calcule la probabilité a posteriori finale pour chaque hypothèse en normalisant l'intersection par la probabilité marginale. |

---

## 🔬 Focus Mathématique : La Vraisemblance Binomiale (Tâche 0)

Dans le cas où nous observons $x$ patients développant des effets secondaires sur un total de $n$ patients, et en supposant que les cas suivent une loi binomiale, la vraisemblance (*Likelihood*) pour une probabilité hypothétique $p$ donnée est calculée via la formule de la fonction de masse de probabilité (PMF) binomiale :

$$L(p \mid x, n) = \binom{n}{x} p^x (1-p)^{n-x}$$

Où le coefficient binomial est défini par :

$$\binom{n}{x} = \frac{n!}{x!(n-x)!}$$

---

## 🛠️ Validation de la Tâche 0 : Vraisemblance

Le script de validation évalue la vraisemblance sur un espace de probabilités discrétisé entre $0$ et $1$ généré par `np.linspace` :

```bash
chmod +x 0-main.py
./0-main.py

```

### Sortie attendue :

```text
[0.00000000e+00 2.71330957e-04 8.71800070e-02 3.07345706e-03
 5.93701546e-07 1.14387595e-12 1.09257177e-20 6.10151799e-32
 9.54415702e-49 1.00596671e-78 0.00000000e+00]

```

## ✍️ Auteur

  * **Mathieu** - *Programming student, specialization Machine Learning* - [👤 My Github profile](https://github.com/Mathieu7483)
