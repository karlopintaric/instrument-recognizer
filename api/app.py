from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from ModelService import ModelServiceAST

ml_models = {}

# Define the allowed file formats and maximum file size (in bytes)
ALLOWED_FILE_FORMATS = [".wav"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["BigModel"] = ModelServiceAST()
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(content={"error": "Bad Request", "detail": exc.errors()}, status_code=400)


@app.get("/health-check")
async def health_check():
    """
    Health check endpoint to verify if the API is running.
    """
    return {"status": "API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        #file_extension = file.filename.split(".")[-1]
        #if not any(file_extension in allowed_format for allowed_format in ALLOWED_FILE_FORMATS):
        #    raise RequestValidationError(
        #        errors=[{"loc": ["file"], "msg": "File format not allowed", "type": "value_error"}]
        #    )

        output = ml_models["BigModel"].get_prediction(file.file)
        return output

    except RequestValidationError as ex:
        # Handle RequestValidationError
        return JSONResponse(content={"error": "Bad Request", "detail": ex.errors()}, status_code=400)

    except Exception as ex:
        # Handle other exceptions
        return JSONResponse(content={"error": "Internal Server Error", "detail": str(ex)}, status_code=500)
