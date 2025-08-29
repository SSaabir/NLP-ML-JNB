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

#model = joblib.load("")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')   

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
#data: Features
def predict():
    return {"result": "Hello World!"}
    combined_text = f"{data.title} {data.overview}"
    embeddings = embedding_model.encode([combined_text])
    prediction = model.predict(embeddings)
    

#{"result": prediction.tolist()}