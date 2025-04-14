import requests

def test_predict():
    url = "http://127.0.0.1:8000/predict/"
    with open("data/F5_800.tif", "rb") as image_file:
        response = requests.post(url, files={"file": image_file})
        assert response.status_code == 200
        assert "prediction" in response.json()
