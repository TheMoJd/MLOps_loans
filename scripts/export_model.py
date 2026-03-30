"""
Export du modele LightGBM depuis MLflow Registry vers un fichier standalone.

Ce script :
1. Charge le modele ScoringCredit_LightGBM_Optimized depuis le Registry MLflow
2. Extrait le LGBMClassifier sous-jacent
3. Sauvegarde model.pkl (joblib) et config.json (seuil + features)
"""

import os
import sys
import json
import joblib
import mlflow
import pandas as pd

# Se placer dans le repertoire notebooks pour acceder a la DB MLflow
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = os.path.join(ROOT_DIR, "notebooks")
MODEL_DIR = os.path.join(ROOT_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# Connexion au tracking server
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(NOTEBOOKS_DIR, 'mlflow.db')}")

# --- 1. Charger le modele depuis le Registry ---
model_name = "ScoringCredit_LightGBM_Optimized"
print(f"Chargement du modele '{model_name}' depuis MLflow Registry...")

client = mlflow.MlflowClient()
latest_version = client.get_registered_model(model_name).latest_versions[0]
run = client.get_run(latest_version.run_id)

# Charger le modele via le flavor sklearn (pas pyfunc)
model_uri = f"models:/{model_name}/latest"
unwrapped = mlflow.sklearn.load_model(model_uri)
print(f"Type du modele extrait : {type(unwrapped)}")

# --- 2. Recuperer le seuil optimal ---
seuil_optimal = float(run.data.params.get("seuil_optimal", 0.5))
print(f"Seuil optimal : {seuil_optimal}")

# --- 3. Recuperer les noms de features ---
# Charger le test_preprocessed pour obtenir les noms de colonnes
test_path = os.path.join(ROOT_DIR, "data", "test_preprocessed.csv")
df_test = pd.read_csv(test_path, nrows=1)
feature_cols = list(df_test.columns)  # SK_ID_CURR inclus (le modele a ete entraine avec)
# Nettoyer les noms (meme regex que dans les notebooks)
feature_names = [
    pd.Series([c]).str.replace(r"[^A-Za-z0-9_]", "_", regex=True).iloc[0]
    for c in feature_cols
]
print(f"Nombre de features : {len(feature_names)}")

# --- 4. Sauvegarder le modele ---
model_path = os.path.join(MODEL_DIR, "model.pkl")
joblib.dump(unwrapped, model_path)
print(f"Modele sauvegarde : {model_path} ({os.path.getsize(model_path) / 1024:.0f} KB)")

# --- 5. Sauvegarder la config ---
config = {
    "model_name": model_name,
    "threshold": seuil_optimal,
    "feature_names": feature_names,
    "n_features": len(feature_names),
    "run_id": latest_version.run_id,
    "version": latest_version.version,
}
config_path = os.path.join(MODEL_DIR, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"Config sauvegardee : {config_path}")

# --- 6. Verification rapide ---
loaded_model = joblib.load(model_path)
print(f"\nVerification : le modele charge est de type {type(loaded_model)}")
print(f"Le modele a la methode predict_proba : {hasattr(loaded_model, 'predict_proba')}")
print("\nExport termine avec succes !")
