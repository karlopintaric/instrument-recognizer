import torch
import torchaudio
import torch.nn as nn
import os
import numpy as np
from functools import partial
from transformers import ASTFeatureExtractor
import torch.nn.functional as F
from torchvision.transforms import Compose
from torchaudio.transforms import FrequencyMasking, TimeMasking
from abc import ABC, abstractmethod


class Transform(ABC):

    @abstractmethod
    def __call__(self):
        pass


class Preprocess(ABC):

    @abstractmethod
    def __call__(self):
        pass


class OneHotEncode(Transform):
    def __init__(self, c: list):
        self.c = c

    def __call__(self, labels):
        target = torch.zeros(len(self.c), dtype=torch.float)
        for label in labels:
            idx = self.c.index(label)
            target[idx] = 1
        return target


class ParentMultilabel(Transform):

    def __init__(self, sep=" "):
        self.sep = sep

    def __call__(self, path):
        label = path.split(os.path.sep)[-2].split(self.sep)
        return label


class LabelsFromTxt(Transform):

    def __init__(self, delimiter=None):
        self.delimiter = delimiter

    def __call__(self, path):
        path = path.replace("wav", "txt")
        label = np.loadtxt(path, dtype=str, ndmin=1, delimiter=self.delimiter)
        return label


class PreprocessPipeline(Preprocess):

    def __init__(self, target_sr):
        self.target_sr = target_sr

    def __call__(self, path):
        signal, sr = torchaudio.load(path)
        signal = self._resample(signal, sr)
        signal = self._mix_down(signal)
        return signal.numpy()

    def _mix_down(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _resample(self, signal, input_sr):
        if input_sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                input_sr, self.target_sr)
            signal = resampler(signal)
        return signal


class SpecToImage(Transform):

    def __init__(self, mean=None, std=None, eps=1e-6):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, spec):

        spec = torch.stack([spec, spec, spec], dim=-1)

        mean = torch.mean(spec) if self.mean is None else self.mean
        std = torch.std(spec) if self.std is None else self.std
        spec_norm = (spec - mean) / std

        spec_min, spec_max = torch.min(spec_norm), torch.max(spec_norm)
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)

        return spec_scaled.type(torch.uint8)


class MinMaxScale(Transform):

    def __call__(self, spec):

        spec_min, spec_max = torch.min(spec), torch.max(spec)

        return (spec - spec_min) / (spec_max - spec_min)


class Normalize(Transform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, spec):
        return (spec - self.mean) / self.std


class FeatureExtractor(Transform):

    def __init__(self, sr):
        self.transform = partial(
            ASTFeatureExtractor(), sampling_rate=sr, return_tensors="pt")

    def __call__(self, signal):
        return self.transform(signal.squeeze()).input_values.mT


class Preemphasis(Transform):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """

    def __init__(self, coeff: float = 0.97):
        self.coeff = coeff

    def __call__(self, signal):
        return torch.cat([signal[:, :1], signal[:, 1:] - self.coeff * signal[:, :-1]], dim=1)


class Spectrogram(Transform):

    def __init__(self, sample_rate, n_mels, hop_length, n_fft):
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
            f_min=20,
            center=False)

    def __call__(self, signal):
        return self.transform(signal)


class LogTransform(Transform):

    def __call__(self, signal):
        return torch.log(signal+1e-8)


class PadCutToLength(Transform):

    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, spec):

        seq_len = spec.shape[-1]

        if seq_len > self.max_length:
            return spec[..., :self.max_length]
        if seq_len < self.max_length:
            diff = self.max_length - seq_len
            return F.pad(spec, (0, diff), mode="constant", value=0)


class CustomFeatureExtractor(Transform):

    def __init__(self, sample_rate, n_mels, hop_length, n_fft, max_length, mean, std):
        self.extract = Compose([
            Preemphasis(),
            Spectrogram(sample_rate=sample_rate, n_mels=n_mels,
                        hop_length=hop_length, n_fft=n_fft),
            LogTransform(),
            PadCutToLength(max_length=max_length),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        return self.extract(x)


class RepeatAudio(Transform):

    def __init__(self, max_repeats: int = 2):
        self.max_repeats = max_repeats

    def __call__(self, signal):
        num_repeats = torch.randint(1, self.max_repeats, (1,)).item()
        return np.tile(signal, reps=num_repeats)


class MaskFrequency(Transform):

    def __init__(self, max_mask_length: int = 0):
        self.aug = FrequencyMasking(max_mask_length)

    def __call__(self, spec):
        return self.aug(spec)


class MaskTime(Transform):

    def __init__(self, max_mask_length: int = 0):
        self.aug = TimeMasking(max_mask_length)

    def __call__(self, spec):
        return self.aug(spec)
