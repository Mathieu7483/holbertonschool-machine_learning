<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/math/probability/a-realistic--cinematic-photograph-of-a-data-scient.png"\>
</p>

---

# Mathematics — Probability Distributions

## 🎲 Description

Ce projet est une exploration approfondie et bas niveau des concepts fondamentaux de la théorie des probabilités et des lois statistiques majeures utilisées en science des données et en Machine Learning. L'objectif est de modéliser mathématiquement et d'implémenter sous forme de classes de programmation orientée objet (POO) quatre lois de probabilité fondamentales : **Poisson**, **Exponentielle**, **Normale** (Gaussienne), et **Binomiale**. Pour garantir une compréhension algorithmique totale, toutes les formules analytiques et fonctions d'approximations cumulatives sont codées en pur Python, sans aucune dépendance externe.

## 🎓 Objectifs d'apprentissage

* **Variables Aléatoires & Lois** : Distinguer et modéliser les distributions discrètes (Poisson, Binomiale) et continues (Exponentielle, Normale).
* **PMF & PDF** : Implémenter les fonctions de masse de probabilité (pour le calcul discret) et de densité de probabilité (pour le calcul continu).
* **CDF & Séries** : Calculer les fonctions de répartition cumulative, impliquant des approximations de fonctions spéciales (comme la fonction d'erreur `erf`).
* **Estimation de Paramètres** : Évaluer les métriques d'une loi ($\lambda$, $\mu$, $\sigma$, $p$) directement à partir d'un échantillon de données empiriques.

## 🛠️ Spécifications techniques

* **OS** : Ubuntu 20.04 LTS | **Python** : 3.9
* **Style** : Conformité réglementaire stricte avec la norme `pycodestyle` (2.11.1).
* **Restrictions d'importation** : **Aucun module externe autorisé** (Interdiction d'utiliser `math`, `numpy`, etc.).
* **Approximations mathématiques imposées** :
* $\pi = 3.1415926536$
* $e = 2.7182818285$



---

## 📂 Organisation du Répertoire et des Tâches

Le projet est structuré par fichier de distribution, chaque fichier regroupant l'initialisation, le calcul de densité et le calcul cumulatif :

### 📊 1. Distribution de Poisson (`poisson.py`)

Modélise la probabilité qu'un nombre d'événements se produisent dans un intervalle de temps ou d'espace fixe.

* **Tâche 0 (Initialize)** : Instanciation avec paramètre $\lambda$ (*lambtha*) donné ou calculé à partir de la moyenne des données fournies.
* **Tâche 1 (Poisson PMF)** : Calcul de la probabilité exacte d'observer $k$ événements : $P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$.
* **Tâche 2 (Poisson CDF)** : Somme cumulative des probabilités jusqu'à un point donné.

### 📈 2. Distribution Exponentielle (`exponential.py`)

Modélise le temps d'attente entre deux événements de Poisson successifs.

* **Tâche 3 (Initialize)** : Calcul du paramètre d'intensité $\lambda = \frac{1}{\mu}$ à partir des données.
* **Tâche 4 (Exponential PDF)** : Calcul de la densité de probabilité : $f(x) = \lambda e^{-\lambda x}$.
* **Tâche 5 (Exponential CDF)** : Fonction de répartition cumulative : $F(x) = 1 - e^{-\lambda x}$.

### 🔔 3. Distribution Normale / Gaussienne (`normal.py`)

La loi centrale de la statistique, décrivant la dispersion des données autour d'une moyenne $\mu$ avec un écart-type $\sigma$.

* **Tâche 6 (Initialize)** : Détermination de la moyenne ($\mu$) et de la variance ($\sigma^2$) de l'échantillon.
* **Tâche 7 (Normalize)** : Calcul du score standardisé ($z$-score).
* **Tâche 8 (Normal PDF)** : Calcul de la courbe en cloche : $f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$.
* **Tâche 9 (Normal CDF)** : Intégration numérique exploitant l'approximation de Maclaurin pour la fonction d'erreur $\text{erf}(x)$.

### 🧮 4. Distribution Binomiale (`binomial.py`)

Modélise le nombre de succès dans une suite de $n$ expériences de Bernoulli indépendantes.

* **Tâche 10 (Initialize)** : Estimation du nombre d'essais $n$ et de la probabilité de succès $p$.
* **Tâche 11 (Binomial PMF)** : Calcul via les combinaisons : $P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$.
* **Tâche 12 (Binomial CDF)** : Somme cumulative des probabilités de succès.

---

## 🛠️ Exemple d'Utilisation : Validation de Poisson (Tâche 0)

L'initialisation vérifie la validité de l'échantillon ou extrait directement la moyenne pour définir $\lambda$ :

```bash
chmod +x 0-main.py
./0-main.py

```

### Sortie attendue du script d'évaluation :

```text
Lambtha: 4.84
Lambtha: 5.0

```

## ✍️ Auteur

  * **Mathieu** - *Programming student, specialization Machine Learning* - [👤 My Github profile](https://github.com/Mathieu7483)

