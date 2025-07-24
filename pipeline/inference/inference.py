import joblib
import pandas as pd
import os
import json
import numpy as np

# Load the trained model artifact stored in the specified directory
model = joblib.load(os.path.join("/opt/ml/model", "model.joblib"))
##local test>
#model = joblib.load("../training/model/model.joblib")  

def model_fn(model_dir):
    return joblib.load("/opt/ml/model/model.joblib") #{model_dir}

def input_fn(request_body, content_type):
    if content_type == "application/json":
        data = json.loads(request_body)["passenger"]

        # Ensure the input data has the same features as the training data
        if len(data) != 12:
            raise ValueError("Input data must have 12 features.")

        # Reshape the data to match the model's expected input shape
        return np.array(data).reshape(1, -1)
    
    elif content_type == "text/csv":
        df = pd.read_csv(request_body)

        # Ensure the input data has the same features as the training data
        df = df.drop(['Survived', 'PassengerId'], axis=1)
        return df
    else:
      raise ValueError("Unsupported content type: " + content_type)

def predict_fn(input_data, model):
    # Return the predicted labels (1 or 0 for each passenger)
    return model.predict(input_data)

def output_fn(prediction, accept):
    # Return the predicted label (1 or 0 for the passenger)
    return str(prediction[0])


# function to handle local csv input data and return predictions
def predict(input_data):
    df = pd.read_csv(input_data)

    y = df['Survived']
    # Ensure the input data has the same features as the training data
    df = df.drop(['Survived', 'PassengerId'], axis=1)
    
    print(f"data: {df.head()}")

    prediction = model.predict(df)

    # calculate accuracy of y labels predicted by the model
    accuracy = (prediction == y).mean()
    print(f"Model accuracy: {accuracy:.2f}")

    # Return the predicted labels (1 or 0 for each passenger). list [0 1 0 0 1 0 1 0 1 0 0 0 1 0]
    return prediction 

if __name__ == "__main__":
    #local test:
    #test_data = "test.csv"
    test_data = "/opt/ml/input/data/train/processed.csv"

    predictions = predict(test_data)
    print("prediction>", predictions)