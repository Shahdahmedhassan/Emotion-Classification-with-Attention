# 🎭 Feelings Finder — Emotion Classifier

**🔗 Live app:** [[ADD YOUR STREAMLIT APP LINK HERE](https://your-app-url.streamlit.app](https://emotion-classification-with-attention-u4oinkklbcawnkdzkzvfjb.streamlit.app/))

**🤗 Model on Hugging Face:** [ShahdCoder/emotion-classifier-distilbert](https://huggingface.co/ShahdCoder/emotion-classifier-distilbert)

---

## What this project does

Type any sentence in English and the app predicts which of **6 core emotions** it expresses — joy, sadness, anger, fear, surprise, or disgust — along with a confidence score for each.

## What I built

- **Data:** Used Google's [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) dataset (27 fine-grained emotions) and mapped it down to 6 core emotions using the standard **Ekman mapping**, keeping only rows that map to exactly one emotion for a clean multi-class task.
- **Modeling:** Trained and compared **4 architectures** — LSTM, GRU, BiLSTM + Attention, and a fine-tuned **DistilBERT** transformer — and picked the best one by macro-F1 score. DistilBERT came out on top.
- **Deployment pipeline:**
  - The trained model is hosted as a **model repo on Hugging Face Hub** (not bundled in this GitHub repo, since it's ~256 MB).
  - This repo contains a **Streamlit app** (`app.py`) that downloads the model from the Hub at runtime and serves an interactive UI.
  - Deployed for free on **Streamlit Community Cloud**.
- **UI features:** live confidence bars per emotion, a color-coded "mood orb" for the dominant emotion, one-click example sentences, and a session history of recent analyses.

## Tech stack

`Python` · `PyTorch` · `Transformers (DistilBERT)` · `Streamlit` · `Hugging Face Hub`

## Repo structure

```
app.py                 → the Streamlit app
requirements.txt       → dependencies
push_model_to_hub.py   → one-time script used to upload the model to Hugging Face
artifacts/              → small local files (label encoder, kept for reference)
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
