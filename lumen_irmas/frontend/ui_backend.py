import requests

backend = "http://0.0.0.0:8000/predict"


def predict(image_file):
    files = {'file': image_file}

    r = requests.post(backend, files=files)

    return r


if __name__=="__main__":
    pass