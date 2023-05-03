import sys
import logging
from logging.handlers import RotatingFileHandler
sys.path.append("../")
from typing import Dict

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from api.ModelService import ModelServiceAST
from pydantic import BaseModel, validator

ml_models = {}
ml_models["Accuracy"] = ModelServiceAST()

app = FastAPI()

# Define the allowed file formats and maximum file size (in bytes)
ALLOWED_FILE_FORMATS = ["wav"]

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a rotating file handler to save logs to a file
handler = RotatingFileHandler('api/logs/app.log', maxBytes=100000, backupCount=5)
handler.setLevel(logging.DEBUG)

# Define the log format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

class InvalidFileTypeError(Exception):
    def __init__(self, value: str, message: str):
        self.value = value
        self.message = message
        super().__init__(message)


class InvalidModelError(Exception):
    def __init__(self, value: str, message: str):
        self.value = value
        self.message = message
        super().__init__(message)


class PredictionRequest(BaseModel):
    file_name: str
    model_name: str

    @validator("file_name")
    @classmethod
    def valid_type(cls, v):
        if v.split(".")[-1].lower() not in ALLOWED_FILE_FORMATS:
            raise InvalidFileTypeError(value=v, message="Only wav files are supported")
        return v

    @validator("model_name")
    @classmethod
    def valid_model(cls, v):
        if v not in ml_models.keys():
            raise InvalidModelError(value=v, message="Selected model doesn't exist")
        return v


class PredictionResult(BaseModel):
    prediction: Dict[str, Dict[str, int]]


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request, ex):
    logger.error(f"Request validation error: {ex}")
    return JSONResponse(content={"error": "Bad Request", "detail": ex.errors()}, status_code=400)


@app.exception_handler(InvalidFileTypeError)
def filetype_exception_handler(request, ex):
    logger.error(f"Invalid file type error: {ex}")
    return JSONResponse(content={"error": "Bad Request", "detail": ex.message}, status_code=400)


@app.exception_handler(InvalidModelError)
def model_exception_handler(request, ex):
    logger.error(f"Invalid model error: {ex}")
    return JSONResponse(content={"error": "Bad Request", "detail": ex.message}, status_code=400)


@app.exception_handler(Exception)
def handle_exceptions(request, ex):
    logger.exception(f"Internal server error: {ex}")
    # If an exception occurs during processing, return a JSON response with an error message
    return JSONResponse(content={"error": "Internal Server Error", "detail": str(ex)}, status_code=500)


@app.get("/health-check")
def health_check():
    """
    Health check endpoint to verify if the API is running.
    """
    logger.info("Health check endpoint was hit")
    return {"status": "API is running"}


@app.post("/predict")
def predict(request: PredictionRequest = Depends(), file: UploadFile = File(...)) -> PredictionResult: # noqa
    logger.info(f"Prediction request received: {request}")
    output = ml_models[request.model_name].get_prediction(file.file)
    logger.info(f"Prediction result: {output}")
    prediction_result = PredictionResult(prediction={request.file_name: output})

    return prediction_result
