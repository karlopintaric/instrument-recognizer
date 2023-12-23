from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from src.modeling import ASTPretrained, FeatureExtractor, PreprocessPipeline, StudentAST

MODELS_FOLDER = Path(__file__).parent / "models"

CLASSES = ["tru", "sax", "vio", "gac", "org", "cla", "flu", "voi", "gel", "cel", "pia"]


def load_model(model_type: str):
    """
    Loads a pre-trained AST model of the specified type.

    :param model_type: The type of model to load
    :type model_type: str
    :return: The loaded pre-trained AST model.
    :rtype: ASTPretrained
    """

    if model_type == "accuracy":
        model = ASTPretrained(n_classes=11, download_weights=False)
        model.load_state_dict(torch.load(f"{MODELS_FOLDER}/acc_model_ast.pth", map_location=torch.device("cpu")))
    else:
        model = StudentAST(n_classes=11, hidden_size=192, num_heads=3)
        model.load_state_dict(torch.load(f"{MODELS_FOLDER}/speed_model_ast.pth", map_location=torch.device("cpu")))
    model.eval()
    return model


def load_labels():
    """
    Loads a dictionary of class labels for the AST model.

    :return: A dictionary where the keys are the class indices and the values are the class labels.
    :rtype: Dict[int, str]
    """

    labels = {i: CLASSES[i] for i in range(len(CLASSES))}
    return labels


def load_thresholds(model_type: str):
    """
    Loads the prediction thresholds for the AST model.

    :return: The prediction thresholds for each class.
    :rtype: np.ndarray
    """
    if model_type == "accuracy":
        thresholds = np.load(f"{MODELS_FOLDER}/acc_model_thresh.npy", allow_pickle=True)
    else:
        thresholds = np.load(f"{MODELS_FOLDER}/speed_model_thresh.npy", allow_pickle=True)
    return thresholds


class ModelServiceAST:
    def __init__(self, model_type: str):
        """
        Initializes a ModelServiceAST instance with the specified model type.

        :param model_type: The type of model to load
        :type model_type: str
        """

        self.model = load_model(model_type)
        self.labels = load_labels()
        self.thresholds = load_thresholds(model_type)
        self.transform = transforms.Compose([PreprocessPipeline(target_sr=16000), FeatureExtractor(sr=16000)])

    def get_prediction(self, audio):
        """
        Gets the binary predictions for the given audio file.

        :param audio_file: The file object for the input audio to make predictions for.
        :type audio_file: file object
        :return: A dictionary where the keys are the class labels and the values are binary predictions (0 or 1).
        :rtype: Dict[str, int]
        """
        processed = self.transform(audio)
        with torch.no_grad():
            # Don't forget to transpose the output to seq_len x num_features!!!
            output = torch.sigmoid(self.model(processed.mT))
            output = output.squeeze().numpy().astype(float)

        binary_predictions = {}
        for i, label in enumerate(CLASSES):
            binary_predictions[label] = int(output[i] >= self.thresholds[i])

        return binary_predictions
