import pandas as pd

def generate_features(input_path, output_path):
    df = pd.read_csv(input_path)
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)
        
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    generate_features("data/cleaned_data.csv", "data/features_data.csv")
