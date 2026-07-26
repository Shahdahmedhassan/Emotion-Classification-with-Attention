import pickle
import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# CONFIG: put the Hugging Face model repo id you push the model to below.
# Example: "ShahdCoder/emotion-classifier-distilbert"
# ---------------------------------------------------------------------------
HF_MODEL_ID = "ShahdCoder/emotion-classifier-distilbert"

EMOTIONS_6 = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()[0]
    return {EMOTIONS_6[i]: float(probs[i]) for i in range(len(EMOTIONS_6))}


st.set_page_config(page_title="Emotion Classifier", page_icon="🎭", layout="centered")

st.title("🎭 Emotion Classifier")
st.write(
    "Enter a sentence in English and get the dominant emotion "
    "(joy, sadness, anger, fear, surprise, disgust) with confidence scores."
)

tokenizer, model = load_model()

text = st.text_area("Your text", height=120, placeholder="Type a sentence in English...")

if st.button("Predict", type="primary") and text.strip():
    with st.spinner("Analyzing..."):
        scores = predict(text, tokenizer, model)

    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    top_emotion = next(iter(sorted_scores))

    st.subheader(f"Dominant emotion: **{top_emotion}** ({sorted_scores[top_emotion]:.1%})")

    for emotion, score in sorted_scores.items():
        st.write(f"{emotion.capitalize()}")
        st.progress(score)

elif text.strip() == "" and st.session_state.get("clicked", False):
    st.warning("Please enter some text first.")
