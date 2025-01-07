# ML Classification: Spam Detection

This project focuses on training and evaluating ML models for spam detection using a dataset of spam emails. The workflow also includes features to release trained models automatically using GitHub Actions.

## Features

- Preprocessing pipeline for data cleaning and preparation.
- Training and evaluation of machine learning models.
- Automated release workflow using GitHub Actions.
- Spam email dataset included for model training.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ml-project.git

2. Create virtual environment:
   ```bash
   python3 -m venv yourvenvname   

3. Activate the venv (for linux):
   ```bash
   source yourvenv/bin/activate

4. Install dependencies on venv:
   ```bash 
   pip install -r requirements.txt

5. Run the main.py:
   ```bash
   python main.py

## Usage:
   The application processes the spam dataset, trains a classification model, and provides evaluation metrics. Results are saved for further analysis or deployment.

## Note:
   The application may slow when startup due to the model training process on the app startup event with large dataset   