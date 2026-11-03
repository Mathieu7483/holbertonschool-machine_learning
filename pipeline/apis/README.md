<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/apis/Data%20collection%20-%20APIS.png"\>
</p>

# Pipeline — Data Collection & APIs

## Description

Ce projet aborde la collecte et l'intégration de données issues de services externes via des **API REST**, une étape fondamentale pour l'alimentation des lacs de données (*data lakes*) et l'entraînement de modèles de **Machine Learning**.

À travers l'utilisation de la bibliothèque Python **`requests`**, ce projet se concentre sur les compétences indispensables en ingénierie des données :

* Effectuer des requêtes HTTP (`GET`).
* Manipuler et parser des flux de données au format **JSON**.
* Gérer la **pagination** pour parcourir des ensembles de données répartis sur plusieurs pages.
* Gérer les restrictions d'accès et les **limites de taux (*rate limits*)** imposées par les serveurs distants.

---

## 🛠️ Spécifications Techniques

* **OS :** Ubuntu 20.04 LTS
* **Langage :** Python 3.9
* **Bibliothèque principale :** `requests`
* **Style de code :** Conformité stricte à la norme `pycodestyle` (version 2.11.1).
* **Exécutabilité :** Tous les scripts doivent se terminer par une nouvelle ligne, commencer par le shebang `#!/usr/bin/env python3` et être exécutables (`chmod +x`).
* **Documentation :** Documentation obligatoire pour tous les modules, classes et fonctions.

---

## 📚 Concepts Clés

* **Requêtes HTTP GET :** Récupération de ressources distantes sans altérer l'état du serveur.
* **Pagination des API :** Mécanisme permettant de fragmenter de larges volumes de données en blocs (*pages*) successifs récupérables via des liens de type `next` ou des index de page.
* **Rate Limiting :** Gestion des codes de statut HTTP (ex: `429 Too Many Requests`) et lecture des en-têtes de quotas pour éviter le blocage des requêtes.

---

## 📂 Architecture des Tâches et Liens GitHub

Le tableau ci-dessous répertorie l'ensemble des tâches du projet ainsi que les liens directs vers les fichiers sources dans le dépôt GitHub.

| # | Tâche | Fichier Source | Description |
| --- | --- | --- | --- |
| **0** | **Can I join?** | [`0-passengers.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/apis/0-passengers.py) | Récupère la liste des vaisseaux spatiaux de l'API Star Wars (SWAPI) pouvant transporter un nombre minimal de passagers, avec gestion de la pagination. |
| **1** | **Where I am?** | [`1-where_is_it.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/apis/1-where_is_it.py) | Interroge un service de géolocalisation pour restituer la position associée à une entité ou une adresse IP. |
| **2** | **Rate me if you can!** | [`2-rate_limit.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/apis/2-rate_limit.py) | Analyse et gère les limites de taux (*rate limits*) de l'API GitHub en inspectant les en-têtes HTTP de réponse. |

---

## ⚙️ Installation & Utilisation

### 1. Installation des dépendances

```bash
pip install --user requests

```

### 2. Exécution d'un script de test (Exemple Task 0)

```bash
chmod +x 0-main.py 0-passengers.py
./0-main.py

```

---

## ✍️ Auteur

* **Mathieu** — *Machine Learning Student @ Holberton School* — [Profil GitHub](https://github.com/Mathieu7483)

---