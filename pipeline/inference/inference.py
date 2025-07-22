import joblib
import pandas as pd
import os

model = joblib.load(os.path.join("/opt/ml/model", "model.joblib"))

def predict(input_data):
    df = pd.read_csv(input_data)
    return model.predict(df)

if __name__ == "__main__":
    test_data = pd.read_csv("test.csv")
    
    predictions = predict("test.csv")
    print(predictions)
