from fastapi import FastAPI, UploadFile, File
from ModelService import ModelServiceAST
import torchaudio

app = FastAPI()
model_service = ModelServiceAST()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    output = model_service.get_prediction(file.file)
    return output
