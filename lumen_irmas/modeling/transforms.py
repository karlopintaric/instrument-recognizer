import torch
import torchaudio
import os
import numpy as np
from functools import partial
from transformers import ASTFeatureExtractor
import torch.nn.functional as F
from torchvision.transforms import Compose
from torchaudio.transforms import FrequencyMasking, TimeMasking
from abc import ABC, abstractmethod


class Transform(ABC):
    """Abstract base class for audio transformations."""

    @abstractmethod
    def __call__(self):
        """
        Abstract method to apply the transformation.

        :raises NotImplementedError: If the subclass does not implement this method.

        """
        pass


class Preprocess(ABC):
    """Abstract base class for preprocessing data.

    This class defines the interface for preprocessing data. Subclasses must implement the call method.

    """

    @abstractmethod
    def __call__(self):
        """Process the data.

        This method must be implemented by subclasses.

        :raises NotImplementedError: Subclasses must implement this method.

        """
        pass


class OneHotEncode(Transform):
    """Transform labels to one-hot encoded tensor.

    This class is a transform that takes a list of labels and returns a one-hot encoded tensor. 
    The labels are converted to a tensor with one-hot encoding using the specified classes.

    :param c: A list of classes to be used for one-hot encoding.
    :type c: list
    :param labels: A list of labels to be encoded.
    :type labels: list
    :return: A one-hot encoded tensor.
    :rtype: torch.Tensor

    """

    def __init__(self, c: list):
        """
        Initialize OneHotEncode object.

        :param c: A list of classes to be used for one-hot encoding.
        :type c: list

        """

        self.c = c

    def __call__(self, labels):
        """
        Transform labels to one-hot encoded tensor.

        :param labels: A list of labels to be encoded.
        :type labels: list
        :return: A one-hot encoded tensor.
        :rtype: torch.Tensor

        """

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
    """Extract multilabel parent directory from file path.

    This class is a transform that extracts a multilabel parent directory from a file path. 
    The directory names are split by a specified separator.

    :param sep: The separator used to split the directory names. Defaults to " ".
    :type sep: str
    :param path: The path of the file to extract the multilabel directory from.
    :type path: str
    :return: A list of directory names representing the multilabel parent directory.
    :rtype: list

    """

    def __init__(self, delimiter=None):
        """
        Initialize ParentMultilabel object.

        :param sep: The separator used to split the directory names. Defaults to " ".
        :type sep: str

        """
        self.delimiter = delimiter

    def __call__(self, path):
        """
        Extract multilabel parent directory from file path.

        :param path: The path of the file to extract the multilabel directory from.
        :type path: str
        :return: A list of directory names representing the multilabel parent directory.
        :rtype: list

        """

        path = path.replace("wav", "txt")
        label = np.loadtxt(path, dtype=str, ndmin=1, delimiter=self.delimiter)
        return label


class PreprocessPipeline(Preprocess):
    """A preprocessing pipeline for audio data.

    This class is a preprocessing pipeline for audio data. 
    The pipeline includes resampling to a target sampling rate, mixing down stereo to mono, and loading audio from a file.

    :param target_sr: The target sampling rate to resample to.
    :type target_sr: int
    :param path: The path to the audio file to load.
    :type path: str
    :return: A NumPy array of preprocessed audio data.
    :rtype: numpy.ndarray

    """

    def __init__(self, target_sr):
        """
        Initialize PreprocessPipeline object.

        :param target_sr: The target sampling rate to resample to.
        :type target_sr: int

        """

        self.target_sr = target_sr

    def __call__(self, path):
        """
        Preprocess audio data using a pipeline.

        :param path: The path to the audio file to load.
        :type path: str
        :return: A NumPy array of preprocessed audio data.
        :rtype: numpy.ndarray

        """

        signal, sr = torchaudio.load(path)
        signal = self._resample(signal, sr)
        signal = self._mix_down(signal)
        return signal.numpy()

    def _mix_down(self, signal):
        """
        Mix down stereo to mono.

        :param signal: The audio signal to mix down.
        :type signal: torch.Tensor
        :return: The mixed down audio signal.
        :rtype: torch.Tensor

        """

        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _resample(self, signal, input_sr):
        """
        Resample audio signal to a target sampling rate.

        :param signal: The audio signal to resample.
        :type signal: torch.Tensor
        :param input_sr: The current sampling rate of the audio signal.
        :type input_sr: int
        :return: The resampled audio signal.
        :rtype: torch.Tensor

        """

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
    """Extract features from audio signal using an AST feature extractor.

    This class is a transform that extracts features from an audio signal using an AST feature extractor. 
    The features are returned as a PyTorch tensor.

    :param sr: The sampling rate of the audio signal.
    :type sr: int
    :param signal: The audio signal to extract features from.
    :type signal: numpy.ndarray
    :return: A tensor of extracted audio features.
    :rtype: torch.Tensor

    """

    def __init__(self, sr):
        """
        Initialize FeatureExtractor object.

        :param sr: The sampling rate of the audio signal.
        :type sr: int

        """

        self.transform = partial(
            ASTFeatureExtractor(), sampling_rate=sr, return_tensors="pt")

    def __call__(self, signal):
        """
        Extract features from audio signal using an AST feature extractor.

        :param signal: The audio signal to extract features from.
        :type signal: numpy.ndarray
        :return: A tensor of extracted audio features.
        :rtype: torch.Tensor

        """

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
    """A transform to repeat audio data.

    This class is a transform that repeats audio data a random number of times up to a maximum specified value.

    :param max_repeats: The maximum number of times to repeat the audio data.
    :type max_repeats: int
    :param signal: The audio data to repeat.
    :type signal: numpy.ndarray
    :return: The repeated audio data.
    :rtype: numpy.ndarray

    """

    def __init__(self, max_repeats: int = 2):
        """
        Initialize RepeatAudio object.

        :param max_repeats: The maximum number of times to repeat the audio data.
        :type max_repeats: int

        """

        self.max_repeats = max_repeats

    def __call__(self, signal):
        """
        Repeat audio data a random number of times up to a maximum specified value.

        :param signal: The audio data to repeat.
        :type signal: numpy.ndarray
        :return: The repeated audio data.
        :rtype: numpy.ndarray

        """

        num_repeats = torch.randint(1, self.max_repeats, (1,)).item()
        return np.tile(signal, reps=num_repeats)


class MaskFrequency(Transform):
    """A transform to mask frequency of a spectrogram.

    This class is a transform that masks out a random number of consecutive frequencies from a spectrogram.

    :param max_mask_length: The maximum number of consecutive frequencies to mask out from the spectrogram.
    :type max_mask_length: int
    :param spec: The input spectrogram.
    :type spec: numpy.ndarray
    :return: The spectrogram with masked frequencies.
    :rtype: numpy.ndarray

    """

    def __init__(self, max_mask_length: int = 0):
        """
        Initialize MaskFrequency object.

        :param max_mask_length: The maximum number of consecutive frequencies to mask out from the spectrogram.
        :type max_mask_length: int

        """

        self.aug = FrequencyMasking(max_mask_length)

    def __call__(self, spec):
        """
        Mask out a random number of consecutive frequencies from a spectrogram.

        :param spec: The input spectrogram.
        :type spec: numpy.ndarray
        :return: The spectrogram with masked frequencies.
        :rtype: numpy.ndarray

        """

        return self.aug(spec)


class MaskTime(Transform):
    """A transform to mask time of a spectrogram.

    This class is a transform that masks out a random number of consecutive time steps from a spectrogram.

    :param max_mask_length: The maximum number of consecutive time steps to mask out from the spectrogram.
    :type max_mask_length: int
    :param spec: The input spectrogram.
    :type spec: numpy.ndarray
    :return: The spectrogram with masked time steps.
    :rtype: numpy.ndarray

    """

    def __init__(self, max_mask_length: int = 0):
        """
        Initialize MaskTime object.

        :param max_mask_length: The maximum number of consecutive time steps to mask out from the spectrogram.
        :type max_mask_length: int

        """

        self.aug = TimeMasking(max_mask_length)

    def __call__(self, spec):
        """
        Mask out a random number of consecutive time steps from a spectrogram.

        :param spec: The input spectrogram.
        :type spec: numpy.ndarray
        :return: The spectrogram with masked time steps.
        :rtype: numpy.ndarray

        """

        return self.aug(spec)
