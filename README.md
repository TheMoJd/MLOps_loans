---
title: Scoring Credit
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.10.0"
python_version: "3.11"
app_file: app/app.py
pinned: false
---

# Scoring Credit - Pret a Depenser

[![CI/CD](https://github.com/TheMoJd/MLOps_loans/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/TheMoJd/MLOps_loans/actions/workflows/ci-cd.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/TheMoJd/scoring-credit)
[![Tests](https://img.shields.io/badge/tests-39%20passing-brightgreen.svg)](./tests)

Modele de scoring credit predisant la probabilite de defaut de remboursement d'un client.
Projet MLOps complet : du developpement du modele au deploiement en production.

## Demo en ligne

L'API est deployee en continu sur **Hugging Face Spaces** via le pipeline CI/CD :

> **[huggingface.co/spaces/TheMoJd/scoring-credit](https://huggingface.co/spaces/TheMoJd/scoring-credit)**

Chaque push sur `main` declenche automatiquement les tests, le build Docker, puis le redeploiement du Space.

## Architecture

```
Client (navigateur)
      |
      v
 Gradio API (port 7860)  ----->  logs/predictions.jsonl
      |                                   |
      v                                   v
 LightGBM / ONNX Runtime         Dashboard monitoring (port 7861)
      |
      v
 Score + Decision (ACCORDE / REFUSE)
```

## Structure du projet

```
p6/
├── app/                         # API Gradio
│   ├── app.py                   # Interface web de prediction (port 7860)
│   ├── monitoring.py            # Dashboard de monitoring (port 7861)
│   ├── model.py                 # Chargement modele + prediction
│   ├── config.py                # Configuration (seuil, chemins)
│   └── logger.py                # Logging JSON structure
├── tests/                       # Tests unitaires et integration (39 tests)
│   ├── test_model.py            # Tests modele, decision, business, shape
│   ├── test_api.py              # Tests API + types invalides
│   └── conftest.py              # Fixtures pytest
├── model/                       # Artefacts du modele
│   ├── model.pkl                # LightGBM exporte (~435 KB)
│   ├── model_optimized.onnx     # Version ONNX optimisee
│   └── config.json              # Seuil (0.54) + liste des features
├── data/
│   ├── sample_clients.csv       # 500 clients pour la demo
│   └── reference_data.csv       # Baseline pour detection de drift
├── scripts/                     # Scripts utilitaires
│   ├── export_model.py          # Export MLflow -> model.pkl
│   └── generate_samples.py      # Generation des echantillons
├── notebooks/                   # Analyse et experimentation
│   ├── 01_eda.ipynb             # Exploration des donnees
│   ├── 02_preprocessing.ipynb   # Nettoyage et feature engineering
│   ├── 03_modeling.ipynb        # Entrainement (4 modeles + MLflow)
│   ├── 04_optimization.ipynb    # Optuna + seuil metier
│   ├── 05_interpretability.ipynb # SHAP global + local
│   ├── 06_mlflow_serving.ipynb  # Test du serving MLflow
│   ├── 07_monitoring.ipynb      # Monitoring + data drift (Evidently)
│   └── 08_optimization.ipynb    # Profiling + ONNX benchmark
├── .github/workflows/ci-cd.yml  # Pipeline CI/CD GitHub Actions
├── Dockerfile                   # Conteneurisation de l'API
├── requirements.txt             # Dependances Python
└── README.md
```

## Lancer l'API

### En local

```bash
pip install -r requirements.txt
python -m app.app
```

L'interface est accessible sur http://localhost:7860

> Pour executer les notebooks (`notebooks/`), installer en plus les deps dev :
> `pip install -r requirements-dev.txt` (inclut `jupyter`, `matplotlib`, `seaborn`).

### Avec Docker

```bash
docker build -t scoring-credit .
docker run -p 7860:7860 scoring-credit
```

### Mode ONNX (optimise)

```bash
USE_ONNX=true python -m app.app
```

## Lancer les tests

```bash
pytest tests/ -v
```

39 tests couvrant :
- **Predictions du modele** : probabilites valides, gestion NaN, valeurs extremes
- **Logique de decision** : seuil, coherence ACCORDE/REFUSE
- **API** : fonction predict, client invalide, predictions multiples
- **Coherence des features** : nombre, noms, absence de TARGET
- **Types invalides** (`TestInvalidInputTypes`) : str, None, negatif, float, list
- **Valeurs hors plage metier** (`TestBusinessRangeValues`) : revenu 0 / negatif, credit 0, age impossible, enfants negatifs / absurdes
- **Validation de la forme** (`TestInputShapeValidation`) : colonne manquante, DataFrame vide, texte dans colonne numerique

## Modele

| Parametre | Valeur |
|-----------|--------|
| Algorithme | LightGBM (Gradient Boosting) |
| AUC holdout | 0.7755 |
| Seuil de decision | 0.54 |
| Cout metier | 30 941 (10xFN + FP) |
| Features | 244 |
| Optimisation | Optuna (50 trials) |

Le seuil de 0.54 est optimise pour minimiser le cout metier asymetrique :
un faux negatif (mauvais client predit bon) coute **10x** plus qu'un faux positif.

## Monitoring

Le monitoring se decompose en deux volets complementaires : un **dashboard temps reel** pour la supervision operationnelle, et un **notebook de drift** pour l'analyse statistique.

### 1. Dashboard de monitoring (temps reel)

Le module `app/monitoring.py` expose un dashboard Gradio qui lit en direct le fichier `logs/predictions.jsonl` genere par l'API.

```bash
python -m app.monitoring
```

Accessible sur http://localhost:7861 (port different de l'API pour faire tourner les deux en parallele).

Il affiche :
- **5 KPIs** : nombre total de predictions, latence moyenne, latence p95, taux de refus, clients uniques
- **4 graphiques interactifs** (plotly) : distribution des scores, distribution de la latence, camembert ACCORDE vs REFUSE, volume dans le temps
- **Tableau des 20 dernieres predictions**
- Bouton **Rafraichir** pour recharger les logs a la demande

### 2. Data drift (notebook 07)

Le notebook `07_monitoring.ipynb` contient l'analyse statistique du drift avec Evidently AI :
- Comparaison reference (train) vs production (test)
- Simulation de drift controle pour valider la detection
- Rapports HTML generes dans `reports/`

### Interpreter les rapports Evidently

Les rapports HTML montrent par feature :
- **Vert** : pas de drift significatif
- **Rouge** : drift detecte (p-value < 0.05 au test de Kolmogorov-Smirnov)

**Seuil d'alerte** : si > 20% des features montrent du drift, investiguer.

### Stockage des donnees de production

Les logs sont persistes au format **JSONL** (JSON Lines) dans `logs/predictions.jsonl` :
- Rotation automatique a 10 MB (3 backups conserves)
- Chaque ligne contient : `timestamp`, `client_id`, `probability`, `decision`, `threshold`, `inference_time_ms`
- Format ideal pour ingestion ulterieure dans un systeme centralise (Elasticsearch, BigQuery, etc.)

## CI/CD

Le pipeline GitHub Actions (`.github/workflows/ci-cd.yml`) :

1. **Test** : execute `pytest` sur chaque push/PR
2. **Build** : construit l'image Docker si les tests passent
3. **Deploy** : deploie sur Hugging Face Spaces (branche main uniquement)

## Stack technique

- **API** : Gradio
- **ML** : LightGBM, scikit-learn
- **Optimisation** : ONNX Runtime
- **Monitoring** : Evidently AI
- **MLOps** : MLflow (tracking, registry)
- **CI/CD** : GitHub Actions
- **Conteneurisation** : Docker

## License

Distribue sous licence MIT. Voir [LICENSE](./LICENSE) pour plus de details.

## Auteur

**Moetez JAOUED**  — Projet MLOps OpenClassrooms — Avril 2026
