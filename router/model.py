from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from sklearn.preprocessing import LabelEncoder
from services.model import get_mlp_result, get_naive_bayes_result, get_decision_tree_result, get_random_forest_result, get_svm_result

router = APIRouter()

# Define the request body model using Pydantic
class Message(BaseModel):
    text: str

    # Validate that the message text is not empty
    @validator('text')
    def validate_non_empty(cls, v):
        if not v or v.strip() == "":
            raise ValueError('Text cannot be empty or just whitespace')
        return v

# Helper function to process and get model prediction
async def get_model_result(model_func, features: Message):
    try:
        # Pass the raw text to the model function
        result = await model_func(features.text)

        if result is None:
            raise HTTPException(status_code=404, detail="Prediction result not found")

        return {"result": result}

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")

    
@router.post('/predict/mlp')
async def predict_mlp(features: Message):
    return await get_model_result(get_mlp_result, features)


@router.post('/predict/naive')
async def predict_naive(features: Message):
    return await get_model_result(get_naive_bayes_result, features)


@router.post('/predict/decision-tree')
async def predict_decision_tree(features: Message):
    return await get_model_result(get_decision_tree_result, features)


@router.post('/predict/random-forest')
async def predict_random_forest(features: Message):
    return await get_model_result(get_random_forest_result, features)


@router.post('/predict/svm')
async def predict_svm(features: Message):
    return await get_model_result(get_svm_result, features)
