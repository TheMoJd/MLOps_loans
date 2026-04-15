"""
Scoring Credit - Pret a Depenser
API Gradio pour predire la probabilite de defaut d'un client.
"""

import os
import sys

# Permettre l'execution en tant que script (ex: Hugging Face Spaces)
# en plus de l'execution en tant que module (`python -m app.app`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from app.model import predict, get_client_ids
from app.config import THRESHOLD, PORT


def predict_ui(client_id: str) -> tuple[str, str, str]:
    """Fonction appelee par l'interface Gradio."""
    try:
        cid = int(client_id)
        probability, decision, inference_ms = predict(cid)

        proba_text = f"{probability:.4f} ({probability:.2%})"
        decision_text = f"Credit {decision}"
        time_text = f"{inference_ms:.1f} ms"

        return proba_text, decision_text, time_text

    except ValueError as e:
        return str(e), "ERREUR", "N/A"
    except Exception as e:
        return f"Erreur inattendue : {e}", "ERREUR", "N/A"


# Liste des clients disponibles (convertie en str pour le Dropdown)
client_ids = [str(cid) for cid in get_client_ids()]

# Interface Gradio
demo = gr.Interface(
    fn=predict_ui,
    inputs=gr.Dropdown(
        choices=client_ids,
        label="Client ID (SK_ID_CURR)",
        info="Selectionnez un client pour obtenir son score de credit",
    ),
    outputs=[
        gr.Textbox(label="Probabilite de defaut"),
        gr.Textbox(label="Decision"),
        gr.Textbox(label="Temps d'inference"),
    ],
    title="Scoring Credit - Pret a Depenser",
    description=(
        f"Modele LightGBM optimise | Seuil de decision : {THRESHOLD:.2f}\n\n"
        "Selectionnez un client pour obtenir sa probabilite de defaut "
        "et la decision de credit (ACCORDE / REFUSE)."
    ),
    examples=[[client_ids[0]], [client_ids[1]], [client_ids[2]]],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT, share=True)
