import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import joblib

async def preprocess_model(file_path):
    """
    Preprocess the dataset by removing unwanted columns, handling missing values,
    and encoding target variables.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        tuple: Features (X_train_vec, X_test_vec) and target (y_train, y_test) arrays.
    """
    if not file_path or not os.path.exists(file_path):
        raise ValueError(f"The file path is empty or the file does not exist: {file_path}")

    # Load dataset
    df = pd.read_csv(file_path)

    # Check if required columns exist
    if 'text' not in df.columns or 'label_num' not in df.columns:
        raise ValueError("The dataset must contain 'text' and 'label_num' columns")

    # Preprocess dataset
    df = df.dropna()  # Drop rows with missing values
    
    # Features and target
    X = df['text']
    y = df['label_num']

    # Split data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize text data using CountVectorizer
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    vectorizer_filename = os.path.join(os.getenv("MODEL_PATH"), 'vectorizer.pkl')
    joblib.dump(vectorizer, vectorizer_filename)

    return X_train_vec, X_test_vec, y_train, y_test
