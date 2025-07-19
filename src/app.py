import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

from scripts.model import (
    SentimentLSTM,
    VOCAB_SIZE,
    MAX_LEN,
    EMBEDDING_DIM,
    LSTM_UNITS,
    DROPOUT_RATE,
)

app = FastAPI(
    title="LSTM based Sentiment Analysis",
    description="API for predictin sentiment using a LSTM",
)

TRAINED_MODEL_DIR = "."

model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentimentRequest(BaseModel):
    text: str


@app.on_event("startup")
async def load_resources():
    global model, tokenizer, device

    model_path = os.path.join(TRAINED_MODEL_DIR, "sentiment_model.pth")
    tokenizer_path_pkl = os.path.join(TRAINED_MODEL_DIR, "tokenizer.pkl")

    try:
        if os.path.exists(tokenizer_path_pkl):
            with open(tokenizer_path_pkl, "rb") as handle:
                tokenizer = pickle.load(handle)
        else:
            raise FileNotFoundError(f"tokenizer file not found at {tokenizer_path_pkl}")

        model = SentimentLSTM(
            VOCAB_SIZE, EMBEDDING_DIM, LSTM_UNITS, 1, DROPOUT_RATE
        ).to(device)

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
        else:
            raise FileNotFoundError(f"model not found at {model_path}")
    except Exception as e:
        print(f"error - could not load model / tokenizer: {e}")
        model = None
        tokenizer = None


@app.get("/health", summary="health check", response_description="API status check")
async def health_check():
    return {
        "status": "passed health check",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": str(device),
    }


@app.post("/predict", summary="perdict movie sentiment")
async def predict_sentiment(request: SentimentRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="unable to load model or tokenizer")

    text = request.text

    sequence = tokenizer.texts_to_sequence([text])
    padded_sequence = pad_sequences(
        sequence, maxlen=MAX_LEN, padding="post", truncating="post"
    )
    input_tensor = torch.tensor(padded_sequence, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    probability = outputs.item()

    sentiment = "positive" if probability >= 0.51 else "negative"
    confidence = probability if sentiment == "positive" else (1 - probability)

    return {"text": text, "sentiment": sentiment, "confidence": round(confidence, 3)}


# Only for local testing
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
