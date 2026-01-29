import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

app = FastAPI(title="BERT Sentiment Analysis API")

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float

@app.on_event("startup")
async def load_model():
    global model, tokenizer, device
    
    model_path = os.getenv("MODEL_PATH", "model_output")
    print(f"Loading model from {model_path}...")
    
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # We don't crash here so that the container can start and maybe we can fix it or it's a mount issue
        # But requests will fail.
        pass

@app.get("/health")
async def health_check():
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
        
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        val, idx = torch.max(probs, dim=1)
        
        confidence = val.item()
        sentiment = "positive" if idx.item() == 1 else "negative"
        
    return {"sentiment": sentiment, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
