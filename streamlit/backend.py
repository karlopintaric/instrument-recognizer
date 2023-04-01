import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from modeling.models import ASTPretrained
from modeling.utils import CLASSES, ComposeTransforms
from modeling.transforms import FeatureExtractor, PreprocessPipeline
import streamlit as st
from torchvision import transforms
import torch

@st.cache_resource
def load_model():
    model = ASTPretrained(n_classes=11)
    #model.load...
    model.eval()
    return model

def load_labels():
    labels = {i: CLASSES[i] for i in range(len(CLASSES))}
    return labels

def process_audio(audio):
    preprocess = transforms.Compose([
        PreprocessPipeline(target_sr=16000),
        FeatureExtractor(sr=16000)
    ])
    input_tensor = preprocess(audio)
    return input_tensor


def predict(model, audio, labels):
    input = process_audio(audio)

    with torch.no_grad():
        output = model(input)
        output = torch.sigmoid(output).squeeze().numpy()
        output = (output>0.5).astype(float)
    for i, pred in enumerate(output):
        if pred>0:
            st.write(labels[i])

if __name__=="__main__":
    load_model()