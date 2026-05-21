import pytest
import httpx


BASE_URL = "http://localhost:8005"

def test_docker_predict_survived():
    """Тестируем реальный сервис в Docker: проверяем предсказание для первого класса"""
    payload = {
        "pclass": 1,
        "sex": "female",
        "age": 29.0,
        "sibsp": 0,
        "parch": 0,
        "fare": 150.0,
        "embarked": "S"
    }
    response = httpx.post(f"{BASE_URL}/predict", json=payload, timeout=5.0)
    
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert "status" in data
    assert isinstance(data["prediction"], int)
    assert data["status"] in ["Survived", "Died"]

def test_docker_predict_validation_error():
    """Тестируем валидацию внутри Docker: отправляем некорректные данные"""
    payload = {
        "pclass": "First Class",
        "sex": "female",
        "age": 29.0,
        "sibsp": 0,
        "parch": 0,
        "fare": 150.0,
        "embarked": "S"
    }
    response = httpx.post(f"{BASE_URL}/predict", json=payload, timeout=5.0)
    
    assert response.status_code == 422
