import os
import joblib
import pandas as pd
import logging
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import uvicorn
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker, Session


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/titanic_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PassengerLog(Base):
    __tablename__ = "passenger_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    pclass = Column(Integer, nullable=False)
    sex = Column(String, nullable=False)
    age = Column(Float, nullable=False)
    survived_prediction = Column(Integer, nullable=False)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


try:
    model = joblib.load("titanic.joblib")
    logger.info("Titanic model loaded successfully")
except Exception as e:
    logger.error(f"Error loading Titanic model: {e}")
    model = None

try:
    scaler = joblib.load("scaler.joblib")
    logger.info("Titanic scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading scaler: {e}")
    scaler = None

app = FastAPI(title="Titanic Survival Predictor")

def clear_data(df):
    """ 
    Безопасное кодирование признаков для продакшена.
    Превращаем строки в числа вручную, чтобы не обучать OrdinalEncoder заново на одной строке.
    """
    df['Sex'] = df['Sex'].apply(lambda x: 1.0 if str(x).lower() == 'female' else 0.0)
    df['Embarked'] = df['Embarked'].apply(lambda x: 0.0 if str(x).upper() == 'C' else (1.0 if str(x).upper() == 'Q' else 2.0))
    return df

class PassengerFeatures(BaseModel):
    pclass: int
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: str


@app.post("/predict", summary="Predict if passenger survived")
async def predict(passenger: PassengerFeatures, db: Session = Depends(get_db)):
    """
    Предсказывает выживание пассажира на Титанике
    """
    if model is None or scaler is None:
        return {"error": "Модели Титаника не загружены на сервере"}

    try:
        columns_names = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        
        input_data = pd.DataFrame([[
            passenger.pclass, passenger.sex, passenger.age,
            passenger.sibsp, passenger.parch, passenger.fare, passenger.embarked
        ]], columns=columns_names)

        processed_df = clear_data(input_data)

        processed_df[['Age', 'Fare']] = scaler.transform(processed_df[['Age', 'Fare']])
        
        prediction = int(model.predict(processed_df[columns_names])[0])
        status = "Survived" if prediction == 1 else "Died"
        
        log_entry = PassengerLog(
            pclass=passenger.pclass,
            sex=passenger.sex,
            age=passenger.age,
            survived_prediction=prediction
        )
        db.add(log_entry)
        db.commit()
        
        return {
            "prediction": prediction,
            "status": status
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8005)
