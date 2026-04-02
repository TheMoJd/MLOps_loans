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

Modele de scoring credit predisant la probabilite de defaut de remboursement d'un client.
Projet MLOps complet : du developpement du modele au deploiement en production.

## Architecture

```
Client (navigateur)
      |
      v
 Gradio API (port 7860)
      |
      v
 LightGBM / ONNX Runtime
      |
      v
 Score + Decision (ACCORDE / REFUSE)
```

## Structure du projet

```
p6/
├── app/                         # API Gradio
│   ├── app.py                   # Interface web (point d'entree)
│   ├── model.py                 # Chargement modele + prediction
│   ├── config.py                # Configuration (seuil, chemins)
│   └── logger.py                # Logging JSON structure
├── tests/                       # Tests unitaires et integration
│   ├── test_model.py            # Tests du modele (22 tests)
│   └── test_api.py              # Tests de l'API
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

22 tests couvrant :
- Predictions du modele (probabilites valides, gestion NaN, valeurs extremes)
- Logique de decision (seuil, coherence ACCORDE/REFUSE)
- API (fonction predict, client invalide, predictions multiples)
- Coherence des features

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

Le notebook `07_monitoring.ipynb` contient :
- **Dashboard operationnel** : distribution des scores, latence, decisions
- **Detection de data drift** avec Evidently AI
- Comparaison reference (train) vs production (test)
- Simulation de drift controle pour valider la detection

### Interpreter les rapports Evidently

Les rapports HTML generes dans `reports/` montrent :
- **Vert** : pas de drift significatif sur la feature
- **Rouge** : drift detecte (p-value < 0.05 au test de Kolmogorov-Smirnov)

**Seuil d'alerte** : si > 20% des features montrent du drift, investiguer.

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
