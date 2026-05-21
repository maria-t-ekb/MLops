import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('titanic.csv')

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna('S')

df['Sex'] = df['Sex'].apply(lambda x: 1.0 if str(x).lower() == 'female' else 0.0)
df['Embarked'] = df['Embarked'].apply(lambda x: 0.0 if str(x).upper() == 'C' else (1.0 if str(x).upper() == 'Q' else 2.0))

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
joblib.dump(scaler, "scaler.joblib")


features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

model = LogisticRegression()
model.fit(X, y)
joblib.dump(model, "titanic.joblib")
