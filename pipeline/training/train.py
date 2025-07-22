import pandas as pd
#import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression

def train_model(input_path, model_dir):
    df = pd.read_csv(input_path)
    X = df.drop(['Survived', 'PassengerId'], axis=1)
    y = df['Survived']

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, os.path.join(model_dir, "model.joblib"))

if __name__ == "__main__":
    train_model("/opt/ml/input/data/train/processed.csv", "/opt/ml/model")
