import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression

def train_model(input_path, model_dir):
    try:
        df = pd.read_csv(input_path)
        print("[INFO] Dataset loaded successfully.")
        print("[INFO] Dataset shape:", df.shape)
        print(df.head())
    except Exception as e:
        print("[ERROR] Failed to load dataset:", str(e))
        raise

    # Extract features X and target variable "y":
    # Remove: 'Survived' is the target variable, 'PassengerId' is not used
    X = df.drop(['Survived', 'PassengerId'], axis=1)
    y = df['Survived']

    model = LogisticRegression(max_iter=5000) #Increase the number of iterations to improve the convergence

    try:
        # Fit the model to the data  # need to preprocess X if it contains categorical variables
        model.fit(X, y)

        os.makedirs(model_dir, exist_ok=True)

        # Save the trained model to the specified directory
        joblib.dump(model, os.path.join(model_dir, "model.joblib") )
        print("[INFO] Model saved successfully.")

    except Exception as e:
        print("[ERROR] Error during training:", str(e))
        raise

if __name__ == "__main__":
    #preprocessed input data is in environment variable SM_CHANNEL_TRAIN=/opt/ml/input/data/train/
    train_model("/opt/ml/input/data/train/processed.csv", "/opt/ml/model")

    ##local test
    #train_model("..\processing\processed.csv", "model")
