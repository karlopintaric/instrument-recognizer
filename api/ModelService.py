import torch
from torchvision import transforms

from modeling import ASTPretrained, FeatureExtractor, PreprocessPipeline

CLASSES = ["tru", "sax", "vio", "gac", "org", "cla", "flu", "voi", "gel", "cel", "pia"]
THRESHOLDS = [
    0.95630503,
    0.54127693,
    0.47834763,
    0.5198729,
    0.46966144,
    0.44296056,
    0.5100103,
    0.3471877,
    0.34365374,
    0.44793874,
    0.34663397,
]


def load_model():
    model = ASTPretrained(n_classes=11, download_weights=False)
    model.load_state_dict(torch.load("models/bpmsync_2.pth", map_location=torch.device("cpu")))
    model.eval()
    return model


def load_labels():
    labels = {i: CLASSES[i] for i in range(len(CLASSES))}
    return labels


class ModelServiceAST:
    def __init__(self):
        self.model = load_model()
        self.labels = load_labels()
        self.transform = transforms.Compose([PreprocessPipeline(target_sr=16000), FeatureExtractor(sr=16000)])

    def get_prediction(self, audio):
        processed = self.transform(audio)
        with torch.no_grad():
            output = torch.sigmoid(self.model(processed))
            output = output.squeeze().numpy().astype(float)

        binary_predictions = {}
        for i, label in enumerate(CLASSES):
            binary_predictions[label] = int(output[i] > THRESHOLDS[i])

        return binary_predictions
