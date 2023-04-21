from pathlib import Path
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

import lumen_irmas.transforms as transform_module
from lumen_irmas.transforms import (
    LabelsFromTxt,
    OneHotEncode,
    ParentMultilabel,
    Preprocess,
    Transform,
)
from lumen_irmas.utils import CLASSES, get_wav_files, init_obj, init_transforms


class IRMASDataset(Dataset):
    """Dataset class for IRMAS dataset.

    :param audio_dir: Directory containing the audio files
    :type audio_dir: Union[str, Path]
    :param preprocess: Preprocessing method to apply to the audio files
    :type preprocess: Type[Preprocess]
    :param signal_augments: Signal augmentation method to apply to the audio files, defaults to None
    :type signal_augments: Optional[Union[Type[Compose], Type[Transform]]], optional
    :param transforms: Transform method to apply to the audio files, defaults to None
    :type transforms: Optional[Union[Type[Compose], Type[Transform]]], optional
    :param spec_augments: Spectrogram augmentation method to apply to the audio files, defaults to None
    :type spec_augments: Optional[Union[Type[Compose], Type[Transform]]], optional
    :param subset: Subset of the data to load (train, valid, or test), defaults to "train"
    :type subset: str, optional
    :raises AssertionError: Raises an assertion error if subset is not train, valid or test
    :raises OSError: Raises an OS error if test_songs.txt is not found in the data folder
    :return: A tuple of the preprocessed audio signal and the corresponding one-hot encoded label
    :rtype: Tuple[Tensor, Tensor]
    """

    def __init__(
        self,
        audio_dir: Union[str, Path],
        preprocess: Type[Preprocess],
        signal_augments: Optional[Union[Type[Compose], Type[Transform]]] = None,
        transforms: Optional[Union[Type[Compose], Type[Transform]]] = None,
        spec_augments: Optional[Union[Type[Compose], Type[Transform]]] = None,
        subset: str = "train",
    ):
        self.files = get_wav_files(audio_dir)
        assert subset in ["train", "valid", "test"], "Subset can only be train, valid or test"
        self.subset = subset

        if self.subset != "train":
            try:
                test_songs = np.genfromtxt("../data/test_songs.txt", dtype=str, ndmin=1, delimiter="\n")
            except OSError as e:
                print("Error: {e}")
                print("test_songs.txt not found in data/. Please generate a split before training")
                raise e

        if self.subset == "valid":
            self.files = [file for file in self.files if Path(file).stem not in test_songs]
        if self.subset == "test":
            self.files = [file for file in self.files if Path(file).stem in test_songs]

        self.preprocess = preprocess
        self.transforms = transforms
        self.signal_augments = signal_augments
        self.spec_augments = spec_augments

    def __len__(self):
        """Return the length of the dataset.

        :return: The length of the dataset
        :rtype: int
        """

        return len(self.files)

    def __getitem__(self, index):
        """Get an item from the dataset.

        :param index: The index of the item to get
        :type index: int
        :return: A tuple of the preprocessed audio signal and the corresponding one-hot encoded label
        :rtype: Tuple[Tensor, Tensor]
        """

        sample_path = self.files[index]
        signal = self.preprocess(sample_path)

        if self.subset == "train":
            target_transforms = Compose([ParentMultilabel(sep="-"), OneHotEncode(CLASSES)])
        else:
            target_transforms = Compose([LabelsFromTxt(), OneHotEncode(CLASSES)])

        label = target_transforms(sample_path)

        if self.signal_augments is not None and self.subset == "train":
            signal = self.signal_augments(signal)

        if self.transforms is not None:
            signal = self.transforms(signal)

        if self.spec_augments is not None and self.subset == "train":
            signal = self.spec_augments(signal)

        return signal, label.float()


def collate_fn(data: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Function to collate a batch of audio signals and their corresponding labels.

    :param data: A list of tuples containing the audio signals and their corresponding labels.
    :type data: List[Tuple[torch.Tensor, torch.Tensor]]

    :return: A tuple containing the batch of audio signals and their corresponding labels.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """

    features, labels = zip(*data)
    features = [item.squeeze().T for item in features]
    features = pad_sequence(features, batch_first=True)
    labels = torch.stack(labels)

    return features, labels


def get_loader(config: dict, subset: str):
    """
    Function to create a PyTorch DataLoader for a given subset of the IRMAS dataset.

    :param config: A configuration object.
    :type config: Any
    :param subset: The subset of the dataset to use. Can be "train" or "valid".
    :type subset: str

    :return: A PyTorch DataLoader for the specified subset of the dataset.
    :rtype: torch.utils.data.DataLoader
    """

    dst = IRMASDataset(
        config.train_dir if subset == "train" else config.valid_dir,
        preprocess=init_obj(config.preprocess, transform_module),
        transforms=init_obj(config.transforms, transform_module),
        signal_augments=init_transforms(config.signal_augments, transform_module),
        spec_augments=init_transforms(config.spec_augments, transform_module),
        subset=subset,
    )

    return DataLoader(
        dst,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=torch.get_num_threads() - 1,
        collate_fn=collate_fn,
    )


if __name__ == "__main__":
    from utils import parse_config

    config = parse_config("config.yaml")
    dl = get_loader(config, subset="train")
    sample, _ = next(iter(dl))
    a = 1
