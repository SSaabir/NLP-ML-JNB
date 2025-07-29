from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import joblib

# Initialize app
app = FastAPI()

# Load the SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can set to ["http://localhost:5500"] if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = joblib.load("tpot_model.pkl")  # Make sure this file exists in the same folder

# Define expected input format
class Features(BaseModel):
    features: str  # Assuming features are passed as a string, adjust as necessary

# Prediction endpoint
@app.post("/predict")
def predict(data: Features):
    embeddings = embedding_model.encode([data.features])
    prediction = model.predict(embeddings)
    return {"result": prediction.tolist()}
