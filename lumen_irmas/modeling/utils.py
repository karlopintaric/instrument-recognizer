from glob import glob
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import librosa
import numpy as np
import torch.optim as optim
import yaml

CLASSES = ["tru", "sax", "vio", "gac", "org", "cla", "flu", "voi", "gel", "cel", "pia"]


def get_wav_files(base_path):
    """
    Function to recursively get all the .wav files in a directory.

    :param base_path: The base path of the directory to search.
    :type base_path: str or pathlib.Path

    :return: A list of paths to .wav files found in the directory.
    :rtype: List[str]
    """

    return glob(f"{base_path}/**/*.wav", recursive=True)


def parse_config(config_path):
    """
    Parse a YAML configuration file and return the configuration as a SimpleNamespace object.

    :param config_path: The path to the YAML configuration file.
    :type config_path: str or pathlib.Path

    :return: A SimpleNamespace object representing the configuration.
    :rtype: types.SimpleNamespace
    """
    with open(config_path) as file:
        return SimpleNamespace(**yaml.safe_load(file))


def init_transforms(fn_dict, module):
    """
    Initialize a list of transforms from a dictionary of function names and their parameters.

    :param fn_dict: A dictionary where keys are the names of transform functions
        and values are dictionaries of parameters.
    :type fn_dict: Dict[str, Dict[str, Any]]

    :param module: The module where the transform functions are defined.
    :type module: module

    :return: A list of transform functions.
    :rtype: List[Callable]
    """
    transforms = init_objs(fn_dict, module)
    if transforms is not None:
        transforms = ComposeTransforms(transforms)
    return transforms


def init_objs(fn_dict, module):
    """
    Initialize a list of objects from a dictionary of object names and their parameters.

    :param fn_dict: A dictionary where keys are the names of object classes and values are dictionaries of parameters.
    :type fn_dict: Dict[str, Dict[str, Any]]

    :param module: The module where the object classes are defined.
    :type module: module

    :return: A list of objects.
    :rtype: List[Any]
    """

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
    """
    Initialize an object by calling a function with the provided arguments.

    :param fn_dict: A dictionary that maps the function name to its arguments.
    :type fn_dict: dict or None
    :param module: The module containing the function.
    :type module: module
    :param args: The positional arguments for the function.
    :type args: tuple
    :param kwargs: The keyword arguments for the function.
    :type kwargs: dict
    :raises AssertionError: If a keyword argument is already specified in fn_dict.
    :return: The result of calling the function with the provided arguments.
    :rtype: Any
    """

    if fn_dict is None:
        return None

    name = list(fn_dict.keys())[0]

    fn = getattr(module, name)
    fn_args = fn_dict[name]

    if fn_args is not None:
        assert all(k not in fn_args for k in kwargs)
        fn_args.update(kwargs)

        return fn(*args, **fn_args)
    else:
        return fn(*args, **kwargs)


class ComposeTransforms:
    """
    Composes a list of transforms to be applied in sequence to input data.

    :param transforms: A list of transforms to be applied.
    :type transforms: List[callable]
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, data, *args):
        for t in self.transforms:
            data = t(data, *args)
        return data


def load_raw_file(path: Union[str, Path]):
    """
    Loads an audio file from disk and returns its raw waveform and sample rate.

    :param path: The path to the audio file to load.
    :type path: Union[str, Path]
    :return: A tuple containing the raw waveform and sample rate.
    :rtype: tuple
    """


def get_onset(signal, sr):
    """
    Computes the onset of an audio signal.

    :param signal: The audio signal.
    :type signal: np.ndarray
    :param sr: The sample rate of the audio signal.
    :type sr: int
    :return: The onset of the audio signal in seconds.
    :rtype: float
    """
    onset = librosa.onset.onset_detect(y=signal, sr=sr, units="time")[0]
    return onset


def get_bpm(signal, sr):
    """
    Computes the estimated beats per minute (BPM) of an audio signal.

    :param signal: The audio signal.
    :type signal: np.ndarray
    :param sr: The sample rate of the audio signal.
    :type sr: int
    :return: The estimated BPM of the audio signal, or None if the BPM cannot be computed.
    :rtype: Union[float, None]
    """

    bpm, _ = librosa.beat.beat_track(y=signal, sr=sr)
    return bpm if bpm != 0 else None


def get_pitch(signal, sr):
    """
    Computes the estimated pitch of an audio signal.

    :param signal: The audio signal.
    :type signal: np.ndarray
    :param sr: The sample rate of the audio signal.
    :type sr: int
    :return: The estimated pitch of the audio signal in logarithmic scale, or None if the pitch cannot be computed.
    :rtype: Union[float, None]
    """

    eps = 1e-8
    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("C7")

    pitch, _, _ = librosa.pyin(y=signal, sr=sr, fmin=fmin, fmax=fmax)

    if not np.isnan(pitch).all():
        mean_log_pitch = np.nanmean(np.log(pitch + eps))
    else:
        mean_log_pitch = None

    return mean_log_pitch


def get_file_info(path: Union[str, Path], extract_music_features: bool):
    """
    Loads an audio file and computes some basic information about it,
    such as pitch, BPM, onset time, duration, sample rate, and number of channels.

    :param path: The path to the audio file.
    :type path: Union[str, Path]
    :param extract_music_features: Whether to extract music features such as pitch, BPM, and onset time.
    :type extract_music_features: bool
    :return: A dictionary containing information about the audio file.
    :rtype: dict
    """

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

    return {
        "path": path,
        "pitch": pitch,
        "bpm": bpm,
        "onset": onset,
        "sample_rate": sr,
        "duration": duration,
        "channels": channels,
    }


def sync_pitch(file_to_sync: np.ndarray, sr: int, pitch_base: float, pitch: float):
    """
    Shift the pitch of an audio file to match a new pitch value.

    :param file_to_sync: The input audio file as a NumPy array.
    :type file_to_sync: np.ndarray
    :param sr: The sample rate of the input file.
    :type sr: int
    :param pitch_base: The pitch value of the original file.
    :type pitch_base: float
    :param pitch: The pitch value to synchronize the input file to.
    :type pitch: float
    :return: The synchronized audio file as a NumPy array.
    :rtype: np.ndarray
    """

    assert np.ndim(file_to_sync) == 1, "Input array has more than one dimensions"

    if any(np.isnan(x) for x in [pitch_base, pitch]):
        return file_to_sync

    steps = np.round(12 * np.log2(np.exp(pitch_base) / np.exp(pitch)), 0)

    return librosa.effects.pitch_shift(y=file_to_sync, sr=sr, n_steps=steps)


def sync_bpm(file_to_sync: np.ndarray, sr: int, bpm_base: float, bpm: float):
    """
    Stretch or compress the duration of an audio file to match a new tempo.

    :param file_to_sync: The input audio file as a NumPy array.
    :type file_to_sync: np.ndarray
    :param sr: The sample rate of the input file.
    :type sr: int
    :param bpm_base: The tempo of the original file.
    :type bpm_base: float
    :param bpm: The tempo to synchronize the input file to.
    :type bpm: float
    :return: The synchronized audio file as a NumPy array.
    :rtype: np.ndarray
    """

    assert np.ndim(file_to_sync) == 1, "Input array has more than one dimensions"

    if any(np.isnan(x) for x in [bpm_base, bpm]):
        return file_to_sync

    return librosa.effects.time_stretch(y=file_to_sync, rate=bpm_base / bpm)


def sync_onset(file_to_sync: np.ndarray, sr: int, onset_base: float, onset: float):
    """
    Sync the onset of an audio signal by adding or removing silence at the beginning.

    :param file_to_sync: The audio signal to synchronize.
    :type file_to_sync: np.ndarray
    :param sr: The sample rate of the audio signal.
    :type sr: int
    :param onset_base: The onset of the reference signal in seconds.
    :type onset_base: float
    :param onset: The onset of the signal to synchronize in seconds.
    :type onset: float
    :raises AssertionError: If the input array has more than one dimension.
    :return: The synchronized audio signal.
    :rtype: np.ndarray
    """

    assert np.ndim(file_to_sync) == 1, "Input array has more than one dimensions"

    if any(np.isnan(x) for x in [onset_base, onset]):
        return file_to_sync

    diff = int(round(abs(onset_base * sr - onset * sr), 0))

    if onset_base > onset:
        return np.pad(file_to_sync, (diff, 0), mode="constant", constant_values=0)
    else:
        return file_to_sync[diff:]


if __name__ == "__main__":
    import models

    config = parse_config("./config.yaml")
    model = models.RNN(128, 64, 3, 11)
    optimizer = init_obj(config.optimizer, optim, model.parameters())
