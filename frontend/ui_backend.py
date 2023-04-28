import requests
import os
import streamlit as st

if os.environ.get("DOCKER"):
    backend = "http://api:8000"
else:
    backend = "http://0.0.0.0:8000"


def health_check():
    # Send a health check request to the API
    response = requests.get(f"{backend}/health-check", timeout=100)

    # Check if the API is running
    if response.status_code == 200:
        return True
    else:
        return False

def predict(file):
    files = {"file": file.getvalue()}
    #data = {"audio": audio_data, "model": model}

    response = requests.post(f"{backend}/predict", files=files, timeout=100)  # Replace with your API endpoint URL

    return response


if __name__ == "__main__":
    pass
