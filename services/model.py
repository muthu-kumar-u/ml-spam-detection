from helpers.model import predict_from_model
import os

# Prediction functions for each model type
async def get_mlp_result(data):
    print("result")
    return await predict_from_model("MLPClassifier", data)

async def get_random_forest_result(data):
    return await predict_from_model("Random_Forest", data)

async def get_decision_tree_result(data):
    return await predict_from_model("Decision_Tree", data)

async def get_svm_result(data):
    return await predict_from_model("SVM", data)

async def get_naive_bayes_result(data):
    return await predict_from_model("Naive_Bayes", data)
