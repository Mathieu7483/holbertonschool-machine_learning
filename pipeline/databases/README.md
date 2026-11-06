<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/Databases.png"\>
</p>

# Pipeline — Databases (SQL & NoSQL)

## Description

Ce projet aborde le stockage, l'automatisation et la manipulation de données à l'aide de bases de données relationnelles (**SQL / MySQL**) et non relationnelles (**NoSQL / MongoDB**). La persistence et la structuration des données constituent des étapes clés dans la construction de pipelines Data & Machine Learning.

L'objectif est d'explorer les concepts fondamentaux de deux paradigmes de stockage :

* **Bases relationnelles (MySQL 8.0)** : Création de tables avec contraintes, requêtes complexes, optimisation via des **index**, automatisation des traitements directement au niveau de la base de données à l'aide de **procédures stockées**, **fonctions**, **déclencheurs (*triggers*)** et **vues**.
* **Bases NoSQL orientées document (MongoDB 4.4)** : Manipulation de documents JSON/BSON souples sans schéma fixe, requêtes d'insertion, mise à jour, suppression, pipelines d'agrégation, et interaction programmatique via **PyMongo** en Python.

---

## 🛠️ Spécifications Techniques

* **OS :** Ubuntu 20.04 LTS
* **Environnements de bases de données :**
* **MySQL :** Version 8.0 (8.0.39)
* **MongoDB :** Version 4.4 (4.4.29)


* **Langages :** SQL, JavaScript (Mongo Shell), Python 3.9
* **Bibliothèque Python :** `PyMongo` (version 4.6.2) (`pip install --user pymongo==4.6.2`)
* **Conformité & Style :**
* **SQL :** Tous les mots-clés doivent être en **MAJUSCULES** (`SELECT`, `WHERE`, `CREATE TABLE`...) et chaque fichier doit comporter un commentaire explicatif avant la requête.
* **Mongo Shell :** La première ligne de chaque script `.js` doit être un commentaire (`// comment`).
* **Python :** Conformité stricte à la norme `pycodestyle` (v2.11.1), scripts exécutables (`chmod +x`), shebang `#!/usr/bin/env python3`, modules/fonctions documentés et protection par `if __name__ == "__main__":`.



---

## 📚 Concepts Clés

* **Bases Relationnelles (SQL) :**
* Contraintes de clés primaires/étrangères et intégrité référentielle.
* Optimisation des requêtes par indexation (`CREATE INDEX`).
* Automatisation en base : **Stored Procedures**, **Functions**, **Triggers** et **Views**.
* Principes **ACID** (Atomicités, Cohérence, Isolation, Durabilité).


* **Bases NoSQL (MongoDB) :**
* Modèle orienté document (BSON), souplesse de schéma et haute scalabilité.
* Méthodes du Mongo Shell (`find`, `insert`, `update`, `delete`, `aggregate`).
* Interaction Python avec `PyMongo` pour alimenter et interroger une base NoSQL.



---

## 📂 Architecture des Tâches et Liens GitHub

Le tableau ci-dessous regroupe l'ensemble des tâches du projet avec leurs liens directs vers les fichiers sources dans le dépôt GitHub.

| # | Tâche | Fichier Source | Description |
| --- | --- | --- | --- |
| **0** | **Create a database** | [`0-create_database_if_missing.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/0-create_database_if_missing.sql) | Script SQL créant la base `db_0` sur le serveur MySQL si elle n'existe pas. |
| **1** | **First table** | [`1-first_table.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/1-first_table.sql) | Crée la table `first_table` dans la base courante. |
| **2** | **List all in table** | [`2-list_values.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/2-list_values.sql) | Liste toutes les lignes de la table `first_table`. |
| **3** | **First add** | [`3-insert_value.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/3-insert_value.sql) | Insère une nouvelle ligne dans `first_table`. |
| **4** | **Select the best** | [`4-best_score.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/4-best_score.sql) | Liste les enregistrements ayant un score $\ge 10$ triés par ordre décroissant. |
| **5** | **Average** | [`5-average.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/5-average.sql) | Calcule la moyenne des scores dans la table. |
| **6** | **Temperatures #0** | [`6-avg_temperatures.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/6-avg_temperatures.sql) | Calcule la température moyenne par ville triée par valeur décroissante. |
| **7** | **Temperatures #2** | [`7-max_state.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/7-max_state.sql) | Affiche la température maximale de chaque État. |
| **8** | **Genre ID by show** | [`8-genre_id_by_show.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/8-genre_id_by_show.sql) | Jointure listant les séries et leurs identifiants de genres correspondants. |
| **9** | **No genre** | [`9-no_genre.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/9-no_genre.sql) | Liste les séries n'ayant aucun genre assigné. |
| **10** | **Number of shows by genre** | [`10-count_shows_by_genre.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/10-count_shows_by_genre.sql) | Compte le nombre de séries associées à chaque genre. |
| **11** | **Rotten tomatoes** | [`11-rating.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/11-rating.sql) | Calcule la somme des notes pour chaque série. |
| **12** | **Best genre** | [`12-best_genre.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/12-best_genre.sql) | Liste les genres par somme totale des notes attribuées. |
| **13** | **We are all unique!** | [`13-uniq_users.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/13-uniq_users.sql) | Crée la table `users` avec contrainte d'unicité sur l'email. |
| **14** | **In and not out** | [`14-country_users.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/14-country_users.sql) | Crée la table `users` avec enumeration (`ENUM`) du pays. |
| **15** | **Best band ever!** | [`15-fans.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/15-fans.sql) | Classe les pays d'origine des groupes de musique par nombre total de fans. |
| **16** | **Old school band** | [`16-glam_rock.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/16-glam_rock.sql) | Calcule la longévité des groupes dont le style principal est Glam Rock. |
| **17** | **Buy buy buy** | [`17-store.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/17-store.sql) | Crée un déclencheur (*trigger*) qui diminue la quantité d'un article lors d'une commande. |
| **18** | **Email validation to sent** | [`18-valid.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/18-valid.sql) | Crée un déclencheur réinitialisant l'état de validation d'un email lorsqu'il est modifié. |
| **19** | **Add bonus** | [`19-bonus.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/19-bonus.sql) | Procédure stockée ajoutant une nouvelle correction pour un étudiant. |
| **20** | **Average score** | [`20-average_score.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/20-average_score.sql) | Procédure stockée calculant la moyenne des notes d'un étudiant. |
| **21** | **Safe divide** | [`21-div.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/21-div.sql) | Fonction SQL exécutant une division sécurisée (évite la division par zéro). |
| **22** | **List all databases** | [`22-list_databases`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/22-list_databases) | Script Mongo Shell listant l'ensemble des bases de données MongoDB. |
| **23** | **Create a database** | [`23-use_or_create_database`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/23-use_or_create_database) | Script Mongo Shell basculant ou créant la base `my_db`. |
| **24** | **Insert document** | [`24-insert`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/24-insert) | Insère un document dans la collection `school`. |
| **25** | **All documents** | [`25-all`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/25-all) | Affiche tous les documents d'une collection MongoDB. |
| **26** | **All matches** | [`26-match`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/26-match) | Filtre les documents d'une collection correspondant à un critère. |
| **27** | **Count** | [`27-count`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/27-count) | Dénombre le nombre de documents dans une collection. |
| **28** | **Update** | [`28-update`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/28-update) | Met à jour des champs spécifiques d'un document MongoDB. |
| **29** | **Delete by match** | [`29-delete`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/29-delete) | Supprime les documents correspondant à une condition. |
| **30** | **List all documents in Python** | [`30-all.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/30-all.py) | Fonction PyMongo listant tous les documents d'une collection. |
| **31** | **Insert a document in Python** | [`31-insert.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/31-insert.py) | Fonction PyMongo insérant un nouveau document dans une collection. |
| **32** | **Change school topics** | [`32-update.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/32-update.py) | Met à jour les sujets (*topics*) d'un document d'école selon son nom. |
| **33** | **Where can I learn Python?** | [`33-schools_by_topic.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/33-schools_by_topic.py) | Recherche les écoles proposant un sujet d'étude donné. |
| **34** | **Log stats** | [`34-log_stats.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/34-log_stats.py) | Script Python analysant les logs Nginx stockés dans MongoDB (statistiques de méthodes HTTP). |
| **35** | **Optimize simple search** | [`35-index`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/35-index) | Crée un index sur la colonne `name` d'une table MySQL pour accélérer la recherche. |
| **36** | **Optimize search and score** | [`36-index_sf`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/36-index_sf) | Crée un index composite sur `name` et `score`. |
| **37** | **No table for a meeting** | [`37-init.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/37-init.sql) / [`37-list.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/37-list.sql) | Procédure stockée listant les étudiants sans projet soumis à une date donnée. |
| **38** | **Average weighted score** | [`38-avg_weighted_score.sql`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/38-avg_weighted_score.sql) | Procédure stockée calculant le score moyen pondéré par les coefficients des projets. |
| **39** | **Regex filter** | [`39-regex`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/39-regex) | Script MongoDB filtrant les documents via des expressions régulières (Regex). |
| **40** | **Top students** | [`40-top_students.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/40-top_students.py) | Fonction PyMongo renvoyant les étudiants triés par leur score moyen calculé via pipeline d'agrégation. |
| **41** | **Log stats - new version** | [`41-log_stats.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/databases/41-log_stats.py) | Étend le script d'analyse de logs Nginx pour inclure les statistiques des 10 IPs les plus fréquentes. |

---

## ⚙️ Installation & Configuration

### 1. Démarrage des Services

```bash
# Démarrer MySQL
sudo service mysql start

# Démarrer MongoDB
sudo service mongod start

```

### 2. Exécution des Scripts SQL

```bash
cat 0-create_database_if_missing.sql | mysql -hlocalhost -uroot -p

```

### 3. Exécution des Scripts Mongo Shell & Python

```bash
# Mongo Shell
cat 22-list_databases | mongo

# Python PyMongo
chmod +x 30-all.py
./30-main.py

```

---

## ✍️ Auteur

* **Mathieu** — *Machine Learning Student @ Holberton School* — [Profil GitHub](https://github.com/Mathieu7483)