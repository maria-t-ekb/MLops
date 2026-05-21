import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def download_data():
    df = pd.read_csv("https://githubusercontent.com", delimiter=',')
    df.to_csv("titanic.csv", index=False)
    return df

def clear_data(path2df):
    df = pd.read_csv(path2df)
    
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0] if not df['Embarked'].mode().empty else 'S')
    
    cat_columns = ['Sex', 'Embarked']
    
    df = df.reset_index(drop=True)  
    ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])
    
    df.to_csv('titanic_clear.csv', index=False)
    return True


download_data()
clear_data("titanic.csv")
