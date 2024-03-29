import io
import sys
from pathlib import Path

import soundfile as sf
from fastapi.testclient import TestClient

sys.path.append(".")

from src.api.main import app  # noqa

TEST_FILES_DIR = Path(__file__).parent / "test_files"
TEST_WAV_FILE = TEST_FILES_DIR / "test.wav"

client = TestClient(app)


def test_health_check():
    response = client.get("/health-check")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running"}


def test_predict_valid_cut_file():
    audio_data, sample_rate = sf.read(TEST_WAV_FILE)
    audio_file = io.BytesIO()
    sf.write(audio_file, audio_data, sample_rate, format="wav")
    audio_file = ("test.wav", audio_file)

    file = {"file": audio_file}
    request_data = {"model_name": "Accuracy"}
    # Make a request to the /predict endpoint
    response = client.post("/predict", params=request_data, files=file)

    # Check that the response is successful
    assert response.status_code == 200
    assert response.json()["prediction"]["test.wav"] is not None


def test_predict_valid_file():
    with open(TEST_WAV_FILE, "rb") as file:
        data = {"model_name": "Accuracy"}
        response = client.post("/predict", params=data, files={"file": file})
        assert response.status_code == 200
        assert response.json()["prediction"]["test.wav"] is not None


def test_predict_invalid_file_type():
    file_data = io.BytesIO(b"dummy txt data")
    file = ("test.txt", file_data)
    data = {"model_name": "Accuracy"}
    response = client.post("/predict", params=data, files={"file": file})
    assert response.status_code == 400
    assert "Only wav files are supported" in response.json()["detail"]


def test_predict_invalid_model():
    file_data = io.BytesIO(b"dummy wav data")
    file = ("test.wav", file_data)
    data = {"model_name": "InvalidModel"}
    response = client.post("/predict", params=data, files={"file": file})
    assert response.status_code == 400
    assert "Selected model doesn't exist" in response.json()["detail"]
