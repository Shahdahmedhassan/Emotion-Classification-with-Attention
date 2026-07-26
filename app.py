import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
HF_MODEL_ID = "ShahdCoder/emotion-classifier-distilbert"

# Order MUST match the model's training label order (alphabetical LabelEncoder).
# Do not reorder this list — reorder the EMOTIONS display dict below instead.
MODEL_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

EMOTIONS = {
    "joy":      {"emoji": "😄", "color": "#FFC93C", "label": "Joy"},
    "surprise": {"emoji": "😲", "color": "#FF9F1C", "label": "Surprise"},
    "sadness":  {"emoji": "😢", "color": "#4EA8DE", "label": "Sadness"},
    "fear":     {"emoji": "😨", "color": "#7B61FF", "label": "Fear"},
    "anger":    {"emoji": "😠", "color": "#EF476F", "label": "Anger"},
    "disgust":  {"emoji": "🤢", "color": "#06D6A0", "label": "Disgust"},
}

EXAMPLES = [
    "I just got accepted into my dream university, I can't stop smiling!",
    "I can't believe you forgot my birthday again.",
    "The house was completely silent after everyone left.",
    "There's a strange noise coming from the basement and I'm home alone.",
    "Wait, you're telling me we won the lottery?!",
    "The smell coming from that fridge is absolutely revolting.",
]

# ---------------------------------------------------------------------------
# PAGE CONFIG + STYLE
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Feelings Finder", page_icon="🎭", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background: radial-gradient(circle at 15% 0%, #FFF3D6 0%, #FFFBF3 35%, #FFFBF3 100%);
    }
    #MainMenu, footer, header { visibility: hidden; }

    .hero {
        background: linear-gradient(120deg, #7B61FF 0%, #FF6B9D 55%, #FF9F1C 100%);
        border-radius: 28px;
        padding: 2.6rem 2.4rem;
        margin-bottom: 1.6rem;
        color: #FFFBF3;
        box-shadow: 0 18px 40px -18px rgba(123, 97, 255, 0.55);
    }
    .hero h1 {
        font-family: 'Baloo 2', sans-serif;
        font-weight: 800;
        font-size: 2.6rem;
        margin: 0 0 .35rem 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        font-size: 1.05rem;
        opacity: 0.95;
        margin: 0;
        max-width: 640px;
    }

    .card {
        background: #FFFFFF;
        border-radius: 22px;
        padding: 1.7rem 1.8rem;
        box-shadow: 0 12px 30px -18px rgba(36, 28, 61, 0.25);
        border: 1px solid #F3ECE0;
        margin-bottom: 1.4rem;
    }
    .card h3,
    div[data-testid="stMarkdownContainer"] h3 {
        font-family: 'Baloo 2', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.4rem !important;
        color: #120B24 !important;
        margin-top: 0 !important;
        letter-spacing: -0.2px;
    }

    .stTextArea textarea {
        border-radius: 16px !important;
        border: 2px solid #F0E7D8 !important;
        font-size: 1rem !important;
        padding: 14px !important;
    }
    .stTextArea textarea:focus {
        border-color: #7B61FF !important;
        box-shadow: 0 0 0 3px rgba(123, 97, 255, 0.15) !important;
    }

    div.stButton > button {
        border-radius: 14px;
        font-weight: 600;
        border: none;
        padding: 0.55rem 1.3rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button:hover { transform: translateY(-2px); }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(120deg, #7B61FF, #FF6B9D);
        box-shadow: 0 10px 22px -10px rgba(123, 97, 255, 0.6);
    }

    .chip > button {
        background: #FFF6E8 !important;
        color: #241C3D !important;
        font-size: 0.82rem !important;
        padding: 0.35rem 0.8rem !important;
        border: 1px solid #F0E0BF !important;
    }
    .chip > button:hover { background: #FFEFD0 !important; }

    .orb {
        width: 168px; height: 168px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 4rem; margin: 0 auto 1rem auto;
        animation: pulse 2.2s ease-in-out infinite;
        box-shadow: 0 0 0 10px rgba(0,0,0,0.03);
    }
    @keyframes pulse {
        0%   { transform: scale(1); }
        50%  { transform: scale(1.06); }
        100% { transform: scale(1); }
    }
    .orb-label {
        font-family: 'Baloo 2', sans-serif;
        font-weight: 700;
        font-size: 1.4rem;
        text-align: center;
        margin-bottom: 0.1rem;
    }
    .orb-score {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        text-align: center;
        color: #7A7385;
        margin-bottom: 1.2rem;
    }

    .bar-row { margin-bottom: 12px; }
    .bar-label {
        display: flex; justify-content: space-between; align-items: baseline;
        font-size: 0.92rem; font-weight: 600; color: #241C3D; margin-bottom: 4px;
    }
    .bar-pct { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #7A7385; }
    .bar-track {
        background: #F3ECE0; border-radius: 999px; height: 14px; overflow: hidden;
    }
    .bar-fill { height: 100%; border-radius: 999px; }

    .history-item {
        border-left: 4px solid #E9E1D0;
        padding: 6px 10px;
        margin-bottom: 8px;
        border-radius: 8px;
        background: #FFFBF3;
        font-size: 0.85rem;
    }
    .footer-note {
        text-align: center; color: #A69C8C; font-size: 0.8rem; margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Waking up the model...")
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
    return {MODEL_LABELS[i]: float(probs[i]) for i in range(len(MODEL_LABELS))}


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "last_scores" not in st.session_state:
    st.session_state.last_scores = None

# If an example chip was clicked on the previous run, apply it now —
# BEFORE the text_area widget below is created — so the widget picks it up.
if "pending_example" in st.session_state:
    st.session_state.text_input = st.session_state.pop("pending_example")

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎭 Feelings Finder</h1>
        <p>A fine-tuned DistilBERT model that reads a sentence and figures out what's
        underneath it — joy, sadness, anger, fear, surprise, or disgust — trained on
        Google's GoEmotions dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tokenizer, model = load_model()

# ---------------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------------
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>✍️ Type something</h3>", unsafe_allow_html=True)

    text = st.text_area(
        "text_area",
        height=140,
        placeholder="e.g. I can't believe how good this turned out!",
        label_visibility="collapsed",
        key="text_input",
    )

    st.caption("Or try one of these:")
    chip_cols = st.columns(3)
    for i, example in enumerate(EXAMPLES):
        with chip_cols[i % 3]:
            st.markdown('<div class="chip">', unsafe_allow_html=True)
            short_label = example[:26] + ("…" if len(example) > 26 else "")
            if st.button(short_label, key=f"ex_{i}"):
                st.session_state.pending_example = example
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    b1, b2 = st.columns([1, 1])
    with b1:
        analyze = st.button("🔮 Analyze emotion", type="primary", use_container_width=True)
    with b2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.pending_example = ""
            st.session_state.last_scores = None
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>🕓 Recent analyses</h3>", unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-5:]):
            meta = EMOTIONS[item["emotion"]]
            snippet = item["text"][:60] + ("…" if len(item["text"]) > 60 else "")
            pct_text = f"{item['score']:.0%}"
            st.markdown(
                f"""<div class="history-item" style="border-left-color:{meta['color']}">
                {meta['emoji']} <b>{meta['label']}</b> ({pct_text}) — {snippet}
                </div>""",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>📊 Result</h3>", unsafe_allow_html=True)

    active_text = text if text and text.strip() else None

    if analyze and active_text:
        with st.spinner("Reading between the lines..."):
            scores = predict(active_text, tokenizer, model)
        st.session_state.history.append(
            {
                "text": active_text,
                "emotion": max(scores, key=scores.get),
                "score": max(scores.values()),
            }
        )
        st.session_state.last_scores = scores
    elif analyze and not active_text:
        st.warning("Type something first — the orb needs words to read! 👀")

    scores = st.session_state.last_scores

    if scores:
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        top_key = next(iter(sorted_scores))
        top_meta = EMOTIONS[top_key]
        top_score = sorted_scores[top_key]

        st.markdown(
            f"""
            <div class="orb" style="background:{top_meta['color']}33;">
                <span>{top_meta['emoji']}</span>
            </div>
            <div class="orb-label" style="color:{top_meta['color']}">{top_meta['label']}</div>
            <div class="orb-score">{top_score:.1%} confident</div>
            """,
            unsafe_allow_html=True,
        )

        for key, score in sorted_scores.items():
            meta = EMOTIONS[key]
            st.markdown(
                f"""
                <div class="bar-row">
                    <div class="bar-label">
                        <span>{meta['emoji']} {meta['label']}</span>
                        <span class="bar-pct">{score:.1%}</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{score*100:.1f}%; background:{meta['color']};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<p style='color:#A69C8C;'>Your result will appear here — write a sentence "
            "on the left and hit <b>Analyze emotion</b>.</p>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='footer-note'>Built with DistilBERT · fine-tuned on GoEmotions (Ekman mapping) · "
    "runs on Streamlit Community Cloud</div>",
    unsafe_allow_html=True,
)
