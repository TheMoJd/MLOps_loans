"""
Dashboard de monitoring Gradio pour l'API de scoring credit.

Lit le fichier logs/predictions.jsonl et affiche des metriques temps reel :
- KPIs globaux (volume, latence, taux de refus)
- Distribution des scores de prediction
- Distribution de la latence d'inference
- Repartition des decisions (ACCORDE / REFUSE)
- Evolution du volume de predictions dans le temps
- Tableau des predictions recentes
"""

import json
import os
from datetime import datetime

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.config import LOG_PATH, THRESHOLD


def load_logs() -> pd.DataFrame:
    """Charge les logs de predictions depuis le fichier JSONL."""
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()

    rows = []
    with open(LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def compute_kpis(df: pd.DataFrame) -> tuple[str, str, str, str, str]:
    """Calcule les KPIs globaux a partir des logs."""
    if df.empty:
        return "0", "N/A", "N/A", "N/A", "0"

    total = len(df)
    avg_latency = f"{df['inference_time_ms'].mean():.2f} ms"
    p95_latency = f"{df['inference_time_ms'].quantile(0.95):.2f} ms"
    refusal_rate = f"{(df['decision'] == 'REFUSE').mean():.1%}"
    unique_clients = df["client_id"].nunique()

    return (
        str(total),
        avg_latency,
        p95_latency,
        refusal_rate,
        str(unique_clients),
    )


def plot_score_distribution(df: pd.DataFrame):
    """Histogramme de la distribution des probabilites."""
    if df.empty:
        return go.Figure().update_layout(title="Aucune donnee disponible")

    fig = px.histogram(
        df,
        x="probability",
        nbins=30,
        title="Distribution des probabilites de defaut",
        labels={"probability": "Probabilite de defaut", "count": "Nombre"},
        color_discrete_sequence=["#3b82f6"],
    )
    fig.add_vline(
        x=THRESHOLD,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Seuil = {THRESHOLD:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(bargap=0.05)
    return fig


def plot_latency_distribution(df: pd.DataFrame):
    """Histogramme de la latence d'inference."""
    if df.empty:
        return go.Figure().update_layout(title="Aucune donnee disponible")

    fig = px.histogram(
        df,
        x="inference_time_ms",
        nbins=30,
        title="Distribution de la latence d'inference",
        labels={"inference_time_ms": "Latence (ms)", "count": "Nombre"},
        color_discrete_sequence=["#10b981"],
    )
    p95 = df["inference_time_ms"].quantile(0.95)
    fig.add_vline(
        x=p95,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"p95 = {p95:.1f} ms",
        annotation_position="top right",
    )
    fig.update_layout(bargap=0.05)
    return fig


def plot_decisions_pie(df: pd.DataFrame):
    """Repartition des decisions ACCORDE / REFUSE."""
    if df.empty:
        return go.Figure().update_layout(title="Aucune donnee disponible")

    counts = df["decision"].value_counts().reset_index()
    counts.columns = ["decision", "count"]

    fig = px.pie(
        counts,
        values="count",
        names="decision",
        title="Repartition des decisions",
        color="decision",
        color_discrete_map={"ACCORDE": "#10b981", "REFUSE": "#ef4444"},
        hole=0.4,
    )
    return fig


def plot_volume_over_time(df: pd.DataFrame):
    """Volume de predictions dans le temps (par minute)."""
    if df.empty:
        return go.Figure().update_layout(title="Aucune donnee disponible")

    df_resampled = (
        df.set_index("timestamp")
        .resample("1min")
        .size()
        .reset_index(name="count")
    )

    fig = px.line(
        df_resampled,
        x="timestamp",
        y="count",
        title="Volume de predictions dans le temps (agrege par minute)",
        labels={"timestamp": "Horodatage", "count": "Nb predictions"},
        markers=True,
    )
    fig.update_traces(line_color="#8b5cf6")
    return fig


def recent_predictions_table(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne les 20 predictions les plus recentes."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "client_id",
                "probability",
                "decision",
                "inference_time_ms",
            ]
        )

    recent = df.sort_values("timestamp", ascending=False).head(20).copy()
    recent["timestamp"] = recent["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return recent[
        ["timestamp", "client_id", "probability", "decision", "inference_time_ms"]
    ]


def refresh_dashboard():
    """Rafraichit toutes les vues du dashboard."""
    df = load_logs()
    total, avg_lat, p95_lat, refusal, unique = compute_kpis(df)
    return (
        total,
        avg_lat,
        p95_lat,
        refusal,
        unique,
        plot_score_distribution(df),
        plot_latency_distribution(df),
        plot_decisions_pie(df),
        plot_volume_over_time(df),
        recent_predictions_table(df),
        f"Derniere actualisation : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    )


# --- Construction du dashboard Gradio Blocks ---

with gr.Blocks(title="Monitoring - Scoring Credit") as dashboard:
    gr.Markdown("# Monitoring de l'API de scoring credit")
    gr.Markdown(
        f"Lecture en direct du fichier `logs/predictions.jsonl` "
        f"(seuil de decision : **{THRESHOLD:.2f}**)."
    )

    with gr.Row():
        refresh_btn = gr.Button("Rafraichir", variant="primary", scale=1)
        last_update = gr.Textbox(label="Etat", interactive=False, scale=3)

    gr.Markdown("## Indicateurs cles")
    with gr.Row():
        kpi_total = gr.Textbox(label="Predictions totales", interactive=False)
        kpi_avg_latency = gr.Textbox(label="Latence moyenne", interactive=False)
        kpi_p95_latency = gr.Textbox(label="Latence p95", interactive=False)
        kpi_refusal = gr.Textbox(label="Taux de refus", interactive=False)
        kpi_unique = gr.Textbox(label="Clients uniques", interactive=False)

    gr.Markdown("## Distributions")
    with gr.Row():
        plot_score = gr.Plot(label="Scores de prediction")
        plot_latency = gr.Plot(label="Latence d'inference")

    with gr.Row():
        plot_decisions = gr.Plot(label="Decisions")
        plot_volume = gr.Plot(label="Volume dans le temps")

    gr.Markdown("## Predictions recentes")
    recent_table = gr.Dataframe(
        label="20 dernieres predictions",
        interactive=False,
        wrap=True,
    )

    outputs = [
        kpi_total,
        kpi_avg_latency,
        kpi_p95_latency,
        kpi_refusal,
        kpi_unique,
        plot_score,
        plot_latency,
        plot_decisions,
        plot_volume,
        recent_table,
        last_update,
    ]

    # Rafraichissement manuel
    refresh_btn.click(fn=refresh_dashboard, inputs=[], outputs=outputs)

    # Chargement initial
    dashboard.load(fn=refresh_dashboard, inputs=[], outputs=outputs)


if __name__ == "__main__":
    port = int(os.getenv("MONITORING_PORT", 7861))
    dashboard.launch(server_name="0.0.0.0", server_port=port, share=False)
