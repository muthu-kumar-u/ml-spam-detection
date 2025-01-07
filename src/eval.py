from sklearn.metrics import accuracy_score

async def evaluate_model(models, X, y):
    """
    Evaluate trained models on the given dataset.

    Args:
        models (dict): Dictionary of model names and objects.
        X (array): Feature matrix.
        y (array): Target array.

    Returns:
        dict: Accuracy scores for each model.
    """
    evaluation_results = {}
    for model_name, model in models.items():
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        evaluation_results[model_name] = f"{accuracy:.2f}"

    return evaluation_results
