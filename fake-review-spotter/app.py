import streamlit as st
import joblib
import torch
import torch.nn.functional as F
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Fake Review Spotter",
    page_icon="🛡️",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 700px; margin: auto; }
    .verdict-fake {
        background: #fef2f2;
        border: 2px solid #fca5a5;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #b91c1c;
    }
    .verdict-real {
        background: #f0fdf4;
        border: 2px solid #86efac;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #15803d;
    }
    .note-box {
        background: #f8fafc;
        border-left: 4px solid #7c3aed;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #475569;
        margin-top: 12px;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load model (cached so it only loads once) ────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Download pkl from HuggingFace and load model + tokenizer."""

    # ✏️  REPLACE these with your actual HuggingFace repo details
    HF_REPO_ID  = "FatimaIkram/fake-review-spotter"   # e.g. "ali123/fake-review-model"
    PKL_FILE    = "fake_review_model.pkl"                 # exact filename you uploaded

    with st.spinner("⏳ Loading model from HuggingFace (first load only, ~30 sec)..."):
        # Download pkl from HuggingFace Hub
        pkl_path = hf_hub_download(repo_id=HF_REPO_ID, filename=PKL_FILE)
        bundle   = joblib.load(pkl_path)

        # Rebuild model from saved state dict
        model = RobertaForSequenceClassification.from_pretrained(
            bundle['model_name'],
            num_labels=2
        )
        model.load_state_dict(bundle['model_state_dict'])
        model.eval()

        # Load tokenizer (downloads from HuggingFace automatically)
        tokenizer = RobertaTokenizer.from_pretrained(bundle['model_name'])

    return model, tokenizer, bundle['max_len']


def predict(text, model, tokenizer, max_len):
    """Returns (label, confidence, prob_real, prob_fake)."""
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )
    probs      = F.softmax(outputs.logits, dim=1).numpy()[0]
    pred_idx   = int(np.argmax(probs))
    label      = "Fake" if pred_idx == 1 else "Real"
    confidence = round(float(probs[pred_idx]) * 100, 1)
    return label, confidence, round(float(probs[0])*100, 1), round(float(probs[1])*100, 1)


# ── UI ───────────────────────────────────────────────────
st.markdown("# 🛡️ Fake Review Spotter")
st.markdown("Paste any product review below — the model detects if it's genuine or AI-generated/fake.")
st.markdown("---")

# Load model
model, tokenizer, max_len = load_model()
st.success("✅ Model ready!", icon="✅")

# Sample buttons
st.markdown("**Try a sample:**")
col1, col2, col3 = st.columns(3)

SAMPLES = {
    "🚨 Obvious fake": "Amazing product!!! BEST PURCHASE EVER!! This completely changed my life! You MUST buy this immediately! Five stars isn't enough! INCREDIBLE value! Buy now before it sells out!!",
    "✅ Genuine":       "I've been using this for about three weeks now. The build quality feels solid and it arrived well packaged. Setup took maybe 15 minutes. My only complaint is the manual could be clearer, but overall it does exactly what it promises.",
    "🤖 AI-sounding":  "This product exceeded all my expectations in every way possible. The quality is outstanding, the design is elegant, and the performance is absolutely flawless. I would highly recommend this to anyone looking for a premium experience."
}

if col1.button("🚨 Obvious fake"):
    st.session_state['review'] = SAMPLES["🚨 Obvious fake"]
if col2.button("✅ Genuine"):
    st.session_state['review'] = SAMPLES["✅ Genuine"]
if col3.button("🤖 AI-sounding"):
    st.session_state['review'] = SAMPLES["🤖 AI-sounding"]

# Text input
review_text = st.text_area(
    "Paste a product review here:",
    value=st.session_state.get('review', ''),
    height=150,
    placeholder="e.g. This product is absolutely amazing, best I've ever bought..."
)

# Analyze button
if st.button("🔍 Analyze Review", type="primary", use_container_width=True):
    if not review_text or len(review_text.strip()) < 10:
        st.warning("Please enter a review (at least 10 characters).")
    else:
        with st.spinner("Analyzing..."):
            label, confidence, prob_real, prob_fake = predict(
                review_text.strip(), model, tokenizer, max_len
            )

        # Verdict
        if label == "Fake":
            st.markdown(f'<div class="verdict-fake">🚨 FAKE REVIEW — {confidence}% confidence</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict-real">✅ REAL REVIEW — {confidence}% confidence</div>',
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability bars
        st.markdown("**Confidence breakdown:**")
        col_r, col_f = st.columns(2)
        col_r.metric("✅ Real", f"{prob_real}%")
        col_f.metric("🚨 Fake", f"{prob_fake}%")

        st.progress(int(prob_real), text=f"Real: {prob_real}%")
        st.progress(int(prob_fake), text=f"Fake: {prob_fake}%")

        # Note
        if label == "Fake":
            st.markdown('<div class="note-box">⚠️ This review shows patterns consistent with AI-generated or incentivized fake content. The model detected unnatural language patterns, repetitive phrasing, or templated structure.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="note-box">✅ This review appears to be genuine. The language patterns are consistent with authentic customer feedback.</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:0.8rem;color:#94a3b8;'>"
    "Powered by RoBERTa fine-tuned on 20,000 Amazon reviews · NLP Semester Project"
    "</div>",
    unsafe_allow_html=True
)
