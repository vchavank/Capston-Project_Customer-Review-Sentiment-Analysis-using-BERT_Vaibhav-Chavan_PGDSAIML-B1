"""
Customer Review Sentiment Analysis - Streamlit Web Application
==============================================================
This app provides a user-friendly interface for sentiment prediction
using the fine-tuned DistilBERT model.

Usage:
    streamlit run app.py
"""

import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import time

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis - Customer Reviews",
    page_icon="🎬",
    layout="wide"
)

# Load model and tokenizer
@st.cache_resource
def load_model():
    """Load the fine-tuned DistilBERT model and tokenizer."""
    model_path = './sentiment_model'  # Update path as needed
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure the model is saved in the './sentiment_model' directory.")
        # Fallback: load from HuggingFace
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=2
        )
        model.eval()
        return model, tokenizer


def predict_sentiment(text, model, tokenizer):
    """Predict sentiment for given text."""
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )

    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][prediction].item()
    inference_time = (time.time() - start_time) * 1000  # ms

    label = 'Positive' if prediction == 1 else 'Negative'
    return label, confidence, inference_time, probs[0].numpy()


# Main App
st.title("🎬 Customer Review Sentiment Analysis")
st.markdown("### Powered by Fine-tuned DistilBERT")
st.markdown("---")

# Load model
model, tokenizer = load_model()

# Sidebar
st.sidebar.title("About")
st.sidebar.markdown("""
This application uses a fine-tuned **DistilBERT** model to classify
customer reviews as **Positive** or **Negative**.

**Model Details:**
- Base: distilbert-base-uncased
- Fine-tuned on: IMDb Dataset (50K reviews)
- Max sequence length: 256 tokens
- Parameters: ~67M

**Metrics (Test Set):**
- Accuracy: ~93%
- F1-Score: ~93%
- ROC-AUC: ~98%
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** IMDb Large Movie Review Dataset")
st.sidebar.markdown("**Task:** Binary Sentiment Classification")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter a Review")
    user_input = st.text_area(
        "Type or paste a customer review below:",
        height=150,
        placeholder="e.g., This movie was absolutely fantastic! The acting was superb..."
    )

    if st.button("Analyze Sentiment", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                label, confidence, inference_time, probs = predict_sentiment(
                    user_input, model, tokenizer
                )

            # Display results
            st.markdown("---")
            st.subheader("Results")

            if label == 'Positive':
                st.success(f"**Sentiment: {label}** ✅")
            else:
                st.error(f"**Sentiment: {label}** ❌")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Confidence", f"{confidence:.2%}")
            with col_b:
                st.metric("Inference Time", f"{inference_time:.1f} ms")
            with col_c:
                st.metric("Positive Probability", f"{probs[1]:.4f}")

            # Probability bar
            st.markdown("**Probability Distribution:**")
            st.progress(float(probs[1]))
            st.caption(f"Negative: {probs[0]:.4f} | Positive: {probs[1]:.4f}")
        else:
            st.warning("Please enter a review to analyze.")

with col2:
    st.subheader("Sample Predictions")
    st.markdown("Click any sample to see the prediction:")

    samples = {
        "Positive Review": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout. A masterpiece!",
        "Negative Review": "Terrible waste of time. The story made no sense and the acting was wooden. I want my money back.",
        "Mixed/Sarcastic": "Oh great, another superhero movie. Just what we needed. The special effects were amazing though."
    }

    for title, text in samples.items():
        with st.expander(title):
            st.write(f"*{text}*")
            label, confidence, inference_time, probs = predict_sentiment(
                text, model, tokenizer
            )
            if label == 'Positive':
                st.success(f"{label} ({confidence:.2%})")
            else:
                st.error(f"{label} ({confidence:.2%})")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>AI/ML Capstone Project - Customer Review Sentiment Analysis using BERT</p>
    <p><small>Dataset: IMDb Large Movie Review Dataset | Model: DistilBERT (fine-tuned)</small></p>
</div>
""", unsafe_allow_html=True)
