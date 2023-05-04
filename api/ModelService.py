from pathlib import Path
import numpy as np
import torch
from torchvision import transforms

from modeling import ASTPretrained, FeatureExtractor, PreprocessPipeline  # noqa

MODELS_FOLDER = Path(__file__).parent / "models"

CLASSES = ["tru", "sax", "vio", "gac", "org", "cla", "flu", "voi", "gel", "cel", "pia"]


def load_model(model_type: str):
    
    if model_type == "accuracy":
        model = ASTPretrained(n_classes=11, download_weights=False)
        model.load_state_dict(torch.load(f"{MODELS_FOLDER}/acc_model_ast.pth", map_location=torch.device("cpu")))
    else:
        pass
    model.eval()
    return model


def load_labels():
    labels = {i: CLASSES[i] for i in range(len(CLASSES))}
    return labels


def load_thresholds():
    thresholds = np.load(f"{MODELS_FOLDER}/acc_model_thresh.npy", allow_pickle=True)
    return thresholds


class ModelServiceAST:
    def __init__(self, model_type: str):
        self.model = load_model(model_type)
        self.labels = load_labels()
        self.thresholds = load_thresholds()
        self.transform = transforms.Compose([PreprocessPipeline(target_sr=16000), FeatureExtractor(sr=16000)])

    def get_prediction(self, audio):
        processed = self.transform(audio)
        with torch.no_grad():
            # Don't forget to transpose the output to seq_len x num_features!!!
            output = torch.sigmoid(self.model(processed.mT))
            output = output.squeeze().numpy().astype(float)

        binary_predictions = {}
        for i, label in enumerate(CLASSES):
            binary_predictions[label] = int(output[i] >= self.thresholds[i])

        return binary_predictions
