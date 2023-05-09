import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from ModelService import ModelServiceAST
from pydantic import BaseModel, validator

LOG_SAVE_DIR = Path(__file__).parent / "logs"

ml_models = {}
ml_models["Accuracy"] = ModelServiceAST(model_type="accuracy")
ml_models["Speed"] = ModelServiceAST(model_type="speed")

app = FastAPI()

# Define the allowed file formats and maximum file size (in bytes)
ALLOWED_FILE_FORMATS = ["wav"]

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a rotating file handler to save logs to a file
handler = RotatingFileHandler(f"{LOG_SAVE_DIR}/app.log", maxBytes=100000, backupCount=5)
handler.setLevel(logging.DEBUG)

# Define the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)


class InvalidFileTypeError(Exception):
    def __init__(self):
        self.message = "Only wav files are supported"
        super().__init__(self.message)


class InvalidModelError(Exception):
    def __init__(self):
        self.message = "Selected model doesn't exist"
        super().__init__(self.message)


class MissingFileError(Exception):
    def __init__(self):
        self.message = "File cannot be None"
        super().__init__(self.message)


class PredictionRequest(BaseModel):
    model_name: str

    @validator("model_name")
    @classmethod
    def valid_model(cls, v):
        if v not in ml_models.keys():
            raise InvalidModelError
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


@app.exception_handler(MissingFileError)
def handle_missing_file_error(request, ex):
    logger.error(f"Missing file error: {ex}")
    return JSONResponse(content={"error": "Bad Request", "detail": ex.message}, status_code=400)


@app.exception_handler(Exception)
def handle_exceptions(request, ex):
    logger.exception(f"Internal server error: {ex}")
    # If an exception occurs during processing, return a JSON response with an error message
    return JSONResponse(content={"error": "Internal Server Error", "detail": str(ex)}, status_code=500)


@app.get("/")
def root():
    logger.info("Received request to root endpoint")
    return {"message": "Welcome to my API. Go to /docs to view the documentation."}


@app.get("/health-check")
def health_check():
    """
    Health check endpoint to verify if the API is running.
    """
    logger.info("Health check endpoint was hit")
    return {"status": "API is running"}


@app.post("/predict")
def predict(request: PredictionRequest = Depends(), file: UploadFile = File(...)) -> PredictionResult:  # noqa
    if not file:
        raise MissingFileError
    if file.filename.split(".")[-1].lower() not in ALLOWED_FILE_FORMATS:
        raise InvalidFileTypeError
    logger.info(f"Prediction request received: {request}")
    output = ml_models[request.model_name].get_prediction(file.file)
    logger.info(f"Prediction result: {output}")
    prediction_result = PredictionResult(prediction={file.filename: output})

    return prediction_result
