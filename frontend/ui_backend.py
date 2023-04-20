import requests
import time

#backend = "http://api:8000/predict"
backend = "http://0.0.0.0:8000"

def health_check():
    # Send a health check request to the API
    response = requests.get(f"{backend}/health-check")

    # Check if the API is running
    if response.status_code == 200:
        return True
    else:
        return False


def predict(image_file):
    files = {'file': image_file}
    
    response = requests.post(f"{backend}/predict", files=files, timeout=100)  # Replace with your API endpoint URL

    return response


if __name__=="__main__":
    pass