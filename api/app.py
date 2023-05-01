from contextlib import asynccontextmanager
from typing import Dict

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from ModelService import ModelServiceAST
from pydantic import BaseModel, validator

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["Accuracy"] = ModelServiceAST()
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(lifespan=lifespan)

# Define the allowed file formats and maximum file size (in bytes)
ALLOWED_FILE_FORMATS = ["wav"]


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
async def validation_exception_handler(request, ex):
    return JSONResponse(content={"error": "Bad Request", "detail": ex.errors()}, status_code=400)


@app.exception_handler(InvalidFileTypeError)
async def filetype_exception_handler(request, ex):
    return JSONResponse(content={"error": "Bad Request", "detail": ex.message}, status_code=400)


@app.exception_handler(InvalidModelError)
async def model_exception_handler(request, ex):
    return JSONResponse(content={"error": "Bad Request", "detail": ex.message}, status_code=400)


@app.exception_handler(Exception)
async def handle_exceptions(request, ex):
    # If an exception occurs during processing, return a JSON response with an error message
    return JSONResponse(content={"error": "Internal Server Error", "detail": str(ex)}, status_code=500)


@app.get("/health-check")
async def health_check():
    """
    Health check endpoint to verify if the API is running.
    """
    return {"status": "API is running"}


@app.post("/predict")
def predict(request: PredictionRequest = Depends(), file: UploadFile = File(...)) -> PredictionResult: # noqa
    output = ml_models[request.model_name].get_prediction(file.file)
    prediction_result = PredictionResult(prediction={request.file_name: output})

    return prediction_result
