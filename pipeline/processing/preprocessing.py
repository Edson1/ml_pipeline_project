import pandas as pd
#import os

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)
    
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Cabin'].fillna('Unknown', inplace=True)

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'].fillna('S', inplace=True)
    df = pd.get_dummies(df, columns=['Embarked', 'Pclass'])
    
    df['Ticket'] = df['Ticket'].astype(str).str.extract('(\d+)', expand=False).fillna(0).astype(int)
    df.drop(['Name', 'Cabin'], axis=1, inplace=True)
    
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess("/opt/ml/processing/input/train.csv", "/opt/ml/processing/output/processed.csv")
