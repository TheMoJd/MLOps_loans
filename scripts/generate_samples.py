"""
Generation des fichiers echantillons pour l'API et le monitoring.

Ce script cree :
1. data/sample_clients.csv  — ~500 clients pour la demo API
2. data/reference_data.csv  — ~5000 lignes du train pour baseline drift
"""

import os
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les noms de colonnes (caracteres speciaux -> underscore)."""
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
    return df


def generate_sample_clients(n_samples: int = 500) -> None:
    """Echantillon de clients du test set pour la demo API."""
    print(f"Chargement de test_preprocessed.csv...")
    df = pd.read_csv(os.path.join(DATA_DIR, "test_preprocessed.csv"))
    df = clean_columns(df)

    sample = df.sample(n=n_samples, random_state=42)
    output_path = os.path.join(DATA_DIR, "sample_clients.csv")
    sample.to_csv(output_path, index=False)
    print(f"sample_clients.csv : {len(sample)} lignes, {sample.shape[1]} colonnes")
    print(f"  Taille : {os.path.getsize(output_path) / 1024:.0f} KB")


def generate_reference_data(n_samples: int = 5000) -> None:
    """Echantillon du train set comme baseline pour la detection de drift."""
    print(f"Chargement de train_preprocessed.csv...")
    df = pd.read_csv(os.path.join(DATA_DIR, "train_preprocessed.csv"))
    df = clean_columns(df)

    # Supprimer TARGET et SK_ID_CURR (pas des features)
    cols_to_drop = [c for c in ["TARGET", "SK_ID_CURR"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    sample = df.sample(n=n_samples, random_state=42)
    output_path = os.path.join(DATA_DIR, "reference_data.csv")
    sample.to_csv(output_path, index=False)
    print(f"reference_data.csv : {len(sample)} lignes, {sample.shape[1]} colonnes")
    print(f"  Taille : {os.path.getsize(output_path) / 1024:.0f} KB")


if __name__ == "__main__":
    generate_sample_clients()
    print()
    generate_reference_data()
    print("\nGeneration terminee avec succes !")
