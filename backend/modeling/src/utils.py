from glob import glob
import yaml
import torch.optim as optim
from types import SimpleNamespace
from torchvision.transforms import Compose
from typing import Union
import librosa
import numpy as np
from pathlib import Path

CLASSES = ['tru', 'sax', 'vio', 'gac', 'org',
           'cla', 'flu', 'voi', 'gel', 'cel', 'pia']


def get_wav_files(base_path):
    return glob(f"{base_path}/**/*.wav", recursive=True)


def parse_config(config_path):
    with open(config_path) as file:
        return SimpleNamespace(**yaml.safe_load(file))


def init_transforms(fn_dict, module):
    transforms = init_objs(fn_dict, module)
    if transforms is not None:
        transforms = ComposeTransforms(transforms)
    return transforms


def init_objs(fn_dict, module):

    if fn_dict is None:
        return None

    transforms = []
    for transform in fn_dict.keys():
        fn = getattr(module, transform)
        fn_args = fn_dict[transform]

        if fn_args is None:
            transforms.append(fn())
        else:
            transforms.append(fn(**fn_args))

    return transforms


def init_obj(fn_dict, module, *args, **kwargs):

    if fn_dict is None:
        return None

    name = list(fn_dict.keys())[0]

    fn = getattr(module, name)
    fn_args = fn_dict[name]

    if fn_args is not None:
        assert all([k not in fn_args for k in kwargs])
        fn_args.update(kwargs)

        return fn(*args, **fn_args)
    else:
        return fn(*args, **kwargs)


def LLRD(config, model):
    config = config.LLRD
    lr = config["base_lr"]
    optimizer_grouped_parameters = [{
        'params': [p for n, p in model.named_parameters() if not (("embeddings" in n) or ("encoder.layer" in n))],
        'lr': lr,
        "weight_decay": 0
    }]
    no_decay = ["bias", "layernorm"]
    # initialize lrs for every layer
    #num_layers = model.config.num_hidden_layers
    model_type = "audio_spectrogram_transformer"
    layers = [getattr(model.module.model, model_type).embeddings] + \
        list(getattr(model.module.model, model_type).encoder.layer)
    layers.reverse()

    weight_decay = config["weight_decay"]
    for layer in layers:
        lr *= config["lr_decay_rate"]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]

    return optimizer_grouped_parameters


class ComposeTransforms:

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, input, *args):
        for t in self.transforms:
            input = t(input, *args)
        return input


def freeze(model):

    for param in model.parameters():
        param.requires_grad = False

    return model


def unfreeze(model):

    for param in model.parameters():
        param.requires_grad = True

    return model


def load_raw_file(path: Union[str, Path]):
    path = str(path) if isinstance(path, Path) else path
    return librosa.load(path, sr=None, mono=False)


def get_onset(signal, sr):
    onset = librosa.onset.onset_detect(y=signal, sr=sr, units="time")[0]
    return onset


def get_bpm(signal, sr):
    bpm, _ = librosa.beat.beat_track(y=signal, sr=sr)
    return bpm


def get_pitch(signal, sr):
    eps = 1e-8
    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz('C7')

    pitch, _, _ = librosa.pyin(y=signal, sr=sr, fmin=fmin, fmax=fmax)

    if not np.isnan(pitch).all():
        mean_log_pitch = np.nanmean(np.log(pitch+eps))
    else:
        mean_log_pitch = -9999

    return mean_log_pitch


def get_file_info(path: Union[str, Path], extract_music_features: bool):

    path = str(path) if isinstance(path, Path) else path

    signal, sr = load_raw_file(path)
    channels = signal.shape[0]

    signal = librosa.to_mono(signal)
    duration = len(signal) / sr

    pitch, bpm, onset = None, None, None
    if extract_music_features:
        pitch = get_pitch(signal, sr)
        bpm = get_bpm(signal, sr)
        onset = get_onset(signal, sr)

    return {"path": path, "pitch": pitch, "bpm": bpm,
            "onset": onset, "sample_rate": sr,
            "duration": duration, "channels": channels}


def sync_pitch(file_to_sync: np.ndarray, sr: int,
               pitch_base: float, pitch: float = None):

    assert np.ndim(file_to_sync) == 1, "Input array has more than one dimensions"
    
    if pitch is None:
        pitch = get_pitch(file_to_sync, sr)

    if (pitch_base == -9999) or (pitch == -9999):
        return file_to_sync

    steps = np.round(12 * np.log2(np.exp(pitch_base)/np.exp(pitch)), 0)

    return librosa.effects.pitch_shift(y=file_to_sync, sr=sr, n_steps=steps)


def sync_bpm(file_to_sync: np.ndarray, sr: int,
             bpm_base: float, bpm: float = None):
    
    assert np.ndim(file_to_sync) == 1, "Input array has more than one dimensions"
    
    if bpm is None:
        bpm = get_bpm(file_to_sync, sr)

    if (bpm_base == 0) or (bpm == 0):
        return file_to_sync

    return librosa.effects.time_stretch(y=file_to_sync, rate=bpm_base/bpm)


def sync_onset(file_to_sync: np.ndarray, sr: int,
               onset_base: float, onset: float = None):

    assert np.ndim(file_to_sync) == 1, "Input array has more than one dimensions"
    
    if onset is None:
        onset = get_onset(file_to_sync, sr)
    diff = int(round(abs(onset_base*sr - onset*sr), 0))

    if onset_base > onset:
        return np.pad(file_to_sync, (diff,), mode="constant", constant_values=0)
    else:
        return file_to_sync[diff:]


if __name__ == "__main__":
    import models
    config = parse_config("./config.yaml")
    model = models.RNN(128, 64, 3, 11)
    optimizer = init_obj(config.optimizer, optim, model.parameters())
