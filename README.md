# Emotion Classifier — Streamlit App

A DistilBERT-based text classifier for 6 core emotions (joy, sadness, anger, fear, surprise, disgust), trained on GoEmotions with the Ekman mapping.

## Why the model isn't in this repo
The trained model (`model.safetensors`) is ~256 MB, which is over GitHub's 100 MB per-file limit. Instead, it's hosted on the Hugging Face Hub as a **Model repo** (not a Space, so there's no ZeroGPU quota limit), and `app.py` downloads it automatically at runtime.

## One-time setup: push the model to Hugging Face Hub

1. Unzip `artifacts.zip` (from the Kaggle notebook) so you have `artifacts/distilbert_emotion_model/`.
2. Put that folder next to `push_model_to_hub.py`.
3. Run:
   ```bash
   pip install huggingface_hub
   python push_model_to_hub.py
   ```
4. It will print your model repo id, e.g. `ShahdCoder/emotion-classifier-distilbert`.
5. Open `app.py` and update the line:
   ```python
   HF_MODEL_ID = "ShahdCoder/emotion-classifier-distilbert"
   ```

## Deploy to Streamlit Community Cloud

1. Push this folder (with the updated `app.py`) to a **new GitHub repo**.
2. Go to https://share.streamlit.io → **New app**.
3. Pick your repo, branch, and set the main file path to `app.py`.
4. Click **Deploy**. The first load will take a minute while it downloads the model from the Hub.

## Files in this repo
- `app.py` — the Streamlit app
- `requirements.txt` — dependencies (CPU-only torch to keep the deploy fast)
- `push_model_to_hub.py` — one-time script to upload the model to HF Hub
- `artifacts/label_encoder.pkl` — kept for reference (label order is hardcoded in `app.py` too)
