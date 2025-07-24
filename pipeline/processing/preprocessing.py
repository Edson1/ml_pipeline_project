import pandas as pd

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # Fill missing values> missing Age values with the median, missing Cabin values with 'Unknown'
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Cabin'].fillna('Unknown', inplace=True)

    # Handle missing Embarked values
    # 'S' is the most common embarkation point, 'C' and 'Q' are also common, but we fill missing values with 'S'
    df['Embarked'].fillna('S', inplace=True)    
    
    # Convert categorical variables to numerical 0 or 1
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Convert categorical variables to One-hot encoding: 'Embarked' Q-S-C and 'Pclass' 1-2-3 categories
    df = pd.get_dummies(df, columns=['Embarked', 'Pclass'])

    # 'Ticket' is a numerical variable but we can extract numbers from it
    # If it contains letters, we extract the first number  or   fill with 0 if no number is found
    df['Ticket'] = df['Ticket'].astype(str).str.extract('(\d+)', expand=False).fillna(0).astype(int)

    # 'Name' and 'Cabin' are not used in the model, so we drop them
    df.drop(['Name', 'Cabin'], axis=1, inplace=True)
    
    # Save the processed DataFrame with selected features 'Sex', 'Age', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S'...
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess("/opt/ml/processing/input/train.csv", "/opt/ml/processing/output/processed.csv")
    
    ##local test
    #preprocess("train.csv", "processed.csv")
