# 📊 SARD'INES – Traitement du signal

Ce dépôt contient la partie **Traitement du signal** du projet **SARD'INES** (*SAR' Drone Imaging for urbaN Environment*).  
Il couvre toutes les activités liées au **développement, validation et optimisation des algorithmes SAR**.

---

## 📌 Contenu du dépôt

- **Implémentation d’algorithmes** : compression d’impulsion, Range-Doppler, Backprojection, RMA, autofocus, CFAR…  
- **Validation** : comparaison résultats théoriques / simulations / données réelles  
- **Expériences numériques** : benchmarks multi-paramètres, étude de performances  
- **Datasets** : jeux de données simulés ou acquis, traçabilité et suivi d’usage  

👉 Les scripts, notebooks, datasets et résultats associés sont versionnés dans ce repo.

---

## 📝 Gestion des Issues

Les issues de ce dépôt concernent uniquement le **traitement du signal**.  
Elles permettent de suivre le développement et la validation des algorithmes :  

- `algo-implementation` → implémentation d’un nouvel algorithme  
- `validation-step` → validation d’une étape (compression, migration, focale, etc.)  
- `numerical-experiment` → expérience numérique / benchmark  
- `performance-profiling` → optimisation CPU/GPU, profiling mémoire/temps  


⚠️ Merci de toujours **choisir le bon template** lors de la création d’une issue.  
Si aucun template ne correspond → utiliser _Blank issue_ et préciser `[TR]` dans le titre.

---

## 📚 Liens utiles

- [Vue globale du projet (GitHub Project)](https://github.com/users/FORTRANMAS83R/projects/4)  
- [Repo RF](https://github.com/FORTRANMAS83R/sardine-rf)  
- [Repo Rapport](https://github.com/FORTRANMAS83R/sardine-rapports)  
- [Repo Organisation](https://github.com/FORTRANMAS83R/sardine-orga)  

---

## ✅ Bonnes pratiques

- Toujours créer vos issues **depuis le projet GitHub** pour qu’elles soient rattachées aux milestones/itérations.  
- Documenter vos implémentations avec : paramètres d’entrée/sortie, figures comparatives, métriques de validation.  
- Pour toute nouvelle méthode, créer un **dataset de référence** (simulé ou réel) et documenter son usage avec l’issue `dataset-tracking`.  
- Les bibliographies et sources théoriques doivent être ajoutées dans le repo **Rapport**, pas ici.  

---
