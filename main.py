from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from src.preprocess import preprocess_model
from src.eval import evaluate_model
from src.train import train_model
from dotenv import load_dotenv
from router import model 
from pathlib import Path
import uvicorn
import os

load_dotenv(dotenv_path=".env")
app = FastAPI(openapi_url="")
version = os.getenv("VERSION")

# Define paths
DATA_PATH = os.getenv("DATASET_PATH")
PKL_PATH = os.getenv("MODEL_PATH")
Path(PKL_PATH).mkdir(parents=True, exist_ok=True)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.on_event("startup")
async def startup_event():
    print("Preprocessing data...")
    X_train_vec, X_test_vec, y_train, y_test = await preprocess_model(DATA_PATH)

    print("Training models...")
    trained_models = await train_model(X_train_vec, y_train, PKL_PATH)

    print("Evaluating models...")
    evaluation_results = await evaluate_model(trained_models, X_test_vec, y_test)
    print("Model Evaluation Results:", evaluation_results)

app.include_router(model.router, prefix="/api/" + version)
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
