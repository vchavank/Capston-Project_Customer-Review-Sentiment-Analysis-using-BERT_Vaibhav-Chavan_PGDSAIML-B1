"""
Customer Review Sentiment Analysis - FastAPI Endpoint
=====================================================
This API provides a REST endpoint for sentiment prediction
using the fine-tuned DistilBERT model.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000
    
Endpoints:
    POST /predict - Predict sentiment for a review
    GET /health - Health check
    GET /info - Model information
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import time

app = FastAPI(
    title="Sentiment Analysis API",
    description="Customer Review Sentiment Analysis using Fine-tuned DistilBERT",
    version="1.0.0"
)

# Global model variables
model = None
tokenizer = None


class ReviewInput(BaseModel):
    """Input schema for review prediction."""
    text: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This movie was absolutely fantastic! The acting was superb."
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for sentiment prediction."""
    text: str
    sentiment: str
    confidence: float
    positive_probability: float
    negative_probability: float
    inference_time_ms: float


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model, tokenizer
    model_path = './sentiment_model'
    
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load fine-tuned model: {e}")
        print("Loading base model...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=2
        )
        model.eval()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/info")
async def model_info():
    """Return model information."""
    return {
        "model_name": "distilbert-base-uncased (fine-tuned)",
        "task": "Binary Sentiment Classification",
        "labels": {"0": "Negative", "1": "Positive"},
        "max_sequence_length": 256,
        "dataset": "IMDb Large Movie Review Dataset",
        "parameters": "~67M"
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(review: ReviewInput):
    """Predict sentiment for a given review."""
    if not review.text.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty")
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Tokenize
    inputs = tokenizer(
        review.text,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )
    
    # Predict
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][prediction].item()
    inference_time = (time.time() - start_time) * 1000
    
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    
    return PredictionOutput(
        text=review.text[:200],
        sentiment=sentiment,
        confidence=round(confidence, 4),
        positive_probability=round(float(probs[0][1]), 4),
        negative_probability=round(float(probs[0][0]), 4),
        inference_time_ms=round(inference_time, 2)
    )


@app.post("/predict/batch")
async def predict_batch(reviews: list[ReviewInput]):
    """Predict sentiment for multiple reviews."""
    if len(reviews) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 reviews per batch")
    
    results = []
    for review in reviews:
        result = await predict(review)
        results.append(result)
    
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
