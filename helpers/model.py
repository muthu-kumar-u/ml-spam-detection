from dotenv import load_dotenv
import joblib
import os
import numpy as np

load_dotenv(dotenv_path=".env")
MODEL_DIR = os.getenv("MODEL_PATH")

# Helper function to load a model and predict
async def predict_from_model(model_name, data):
    model_path = os.path.join(MODEL_DIR, f'{model_name.replace(" ", "_")}_model.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the model
    model = joblib.load(model_path)

    # Load the pre-fitted vectorizer
    vectorizer_path = os.path.join(MODEL_DIR, 'vectorizer.pkl')
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
    
    vectorizer = joblib.load(vectorizer_path)

    # Vectorize the input text
    vectorized_data = vectorizer.transform([data]) 

    # Predict and return the result
    prediction = model.predict(vectorized_data)

    if prediction[0] == 1:
        return "The message is spam."
    else:
        return "The message is not spam."
