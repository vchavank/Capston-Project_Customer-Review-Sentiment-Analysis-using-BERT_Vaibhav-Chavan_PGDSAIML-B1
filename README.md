# Customer Review Sentiment Analysis using BERT

## AI/ML Specialization Capstone Project

A comprehensive end-to-end sentiment analysis system that classifies customer reviews as positive or negative using traditional machine learning baselines and transformer-based fine-tuning. The project demonstrates data preparation, model development, performance evaluation, deployment readiness, and business interpretation.

**By:** Vaibhav Krishnakant Chavan, Program: PGDSAIML Batch-1
---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Project Objective](#project-objective)
3. [Dataset Details](#dataset-details)
4. [Project Structure](#project-structure)
5. [Setup Instructions](#setup-instructions)
6. [How to Run](#how-to-run)
7. [Model Architecture](#model-architecture)
8. [Results Summary](#results-summary)
9. [Deployment](#deployment)
10. [Limitations and Future Work](#limitations-and-future-work)

---

## Problem Statement

Businesses receive large volumes of customer reviews but cannot manually read every review. This project builds an NLP model that classifies customer sentiment and helps business teams identify negative feedback quickly, enabling faster response times and improved customer experience.

**Target Users:** Customer experience teams, marketing analysts, product managers, and support managers.

---

## Project Objective

Build a sentiment classification system using:
1. **Baseline Model:** TF-IDF + Logistic Regression
2. **Advanced Model:** Fine-tuned DistilBERT transformer

Then present actionable review insights through a deployment-ready web application.

---

## Dataset Details

| Field | Details |
|-------|---------|
| Dataset Name | IMDb Large Movie Review Dataset |
| Source | [Hugging Face - stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) |
| Size | 50,000 labeled reviews (25K train + 25K test) |
| Labels | Binary: 0 = Negative, 1 = Positive |
| Access Type | Hugging Face Datasets library |
| Balance | Perfectly balanced (50% positive, 50% negative) |

---

## Project Structure

```
capstone-project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── notebooks/
│   ├── Sentiment_Analysis_BERT_Capstone.ipynb   # Main Colab notebook
│   
├── src/
│   ├── preprocessing.py               # Data preprocessing utilities
│   └── preprocessing.ipynb            # Data preprocessing utilities (Notebook version)
├── deployment/
│   ├── app.py                         # Streamlit web application
│   ├── app.ipynb                      # Streamlit web app (Notebook version)
│   ├── api.py                         # FastAPI REST endpoint
│   └── api.ipynb                      # FastAPI REST endpoint (Notebook version)
├── models/
│   └── (saved model files after training)
├── reports/
│   ├── Project_Detailed_Explanation.md    # Step-by-step explanation
│   └── Viva_Questions_and_Answers.md      # Viva preparation
├── data/
│   └── (dataset loaded via Hugging Face)
└── presentation/
    └── (presentation files)
```

---
## Final Result File with Code (Google Colab Version)
- Capston-Project_Customer-Review-Sentiment-Analysis-using-BERT_Vaibhav-Chavan_PGDSAIML-B1.ipynb (The Code and Result File which was run on Goolge Colab is attached which shows code along with it's result
- Capston-Project_Customer-Review-Sentiment-Analysis-using-BERT_Vaibhav-Chavan_PGDSAIML-B1.pptx (The file shows the Project Overview and Conclusion)

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Google Colab (recommended for GPU access) or local machine with CUDA-capable GPU
- pip package manager

### Installation

**Option 1: Google Colab (Recommended)**

1. Open the notebook `notebooks/Sentiment_Analysis_BERT_Capstone.ipynb` in Google Colab
2. Set Runtime > Change runtime type > GPU (T4)
3. Run all cells sequentially

**Option 2: Local Setup**

```bash
# Clone the repository
git clone https://github.com/<your-username>/sentiment-analysis-bert-capstone.git
cd sentiment-analysis-bert-capstone

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### 1. Training the Models (Notebook)

```bash
# Open in Jupyter
jupyter notebook notebooks/Sentiment_Analysis_BERT_Capstone.ipynb

# Or run as script
Run the cells in `notebooks/Sentiment_Analysis_BERT_Capstone.ipynb`
```

### 2. Streamlit Web App

```bash
cd deployment
streamlit run app.py
```

The app will be available at `http://localhost:8501`
For Demonstrations the Web APP is available on server you can use link "http://122.169.98.97:8501

### 3. FastAPI Endpoint

```bash
cd deployment
uvicorn api:app --host 0.0.0.0 --port 8000
```
For demosntration the API is avilable on host 122.169.98.97
API documentation available at `http://localhost:8000/docs`
For Demonstrations the API documentation available at `http://122.169.98.97:8000/docs`

**Example API call:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!"}'
```

---

## Model Architecture

### Baseline: TF-IDF + Logistic Regression

- **Vectorizer:** TF-IDF with 50,000 features, unigrams + bigrams, sublinear TF scaling
- **Classifier:** Logistic Regression with L2 regularization (C=1.0), LBFGS solver
- **Training time:** ~30 seconds on CPU

### Advanced: DistilBERT (Fine-tuned)

- **Base model:** `distilbert-base-uncased` (67M parameters)
- **Architecture:** 6 transformer layers, 768 hidden dimensions, 12 attention heads
- **Fine-tuning:** 3 epochs, learning rate 2e-5, batch size 16, AdamW optimizer
- **Max sequence length:** 256 tokens
- **Training time:** ~20-30 minutes on GPU (T4)

---

## Results Summary

| Metric | TF-IDF + LR (Baseline) | DistilBERT (Advanced) | Improvement |
|--------|------------------------|-----------------------|-------------|
| Accuracy | ~89% | ~93% | +4% |
| Precision | ~89% | ~93% | +4% |
| Recall | ~89% | ~93% | +4% |
| F1-Score | ~89% | ~93% | +4% |
| ROC-AUC | ~96% | ~98% | +2% |
| Inference (ms/sample) | ~0.01 | ~15 | Slower |

**Key Findings:**
- DistilBERT consistently outperforms the baseline across all metrics
- The model struggles with sarcastic and mixed-sentiment reviews
- Longer reviews may lose context due to 256-token truncation
- The baseline offers a strong cost-effective alternative for simpler deployments

---

## Deployment

The project includes two deployment options:

1. **Streamlit App** (`deployment/app.py`): Interactive web interface with real-time predictions, confidence scores, and sample demonstrations.

2. **FastAPI Endpoint** (`deployment/api.py`): RESTful API with batch prediction support, health checks, and automatic documentation.

Both load the saved model artifact without requiring retraining.

---

## Limitations and Future Work

### Limitations
- Movie review language may not fully represent e-commerce or service reviews
- Long reviews (>256 tokens) are truncated, potentially losing context
- Sarcasm and irony detection remains challenging
- Binary classification does not capture neutral sentiment

### Future Enhancements
- Multi-class sentiment (positive/neutral/negative)
- Aspect-based sentiment analysis
- Domain adaptation for e-commerce/service reviews
- Model distillation for faster inference
- Multilingual support

### Ethical Considerations
- Sentiment predictions should support human review, not replace it
- Model may have biases from training data
- Should not be the sole basis for customer service decisions

---

## Evaluation Metrics Explanation

- **Accuracy:** Overall correct predictions / total predictions
- **Precision:** True positives / (True positives + False positives) - How many predicted positives are actually positive
- **Recall:** True positives / (True positives + False negatives) - How many actual positives are correctly identified
- **F1-Score:** Harmonic mean of precision and recall - Balanced metric
- **ROC-AUC:** Area under the ROC curve - Model's ability to distinguish classes
- **Inference Latency:** Time to process a single prediction

---

## References

1. Hugging Face Transformers Documentation. https://huggingface.co/docs/transformers

---

## License

This project is developed for academic purposes as part of the AI/ML Specialization Capstone Project.
