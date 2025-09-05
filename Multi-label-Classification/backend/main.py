import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Features(BaseModel):
    title: str
    overview: str

model = joblib.load("../best_model.pkl")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')   

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(data: Features):
    combined_text = f"{data.title} {data.overview}"
    prediction = model.predict([combined_text])  # pass as a list of strings
    return {"result": prediction.tolist()}



#