from .utils import get_wav_files, CLASSES
from .transforms import LabelsFromTxt, OneHotEncode, ParentMultilabel
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms
from torch.nn.utils.rnn import pad_sequence
import src.transforms as transform_module
from .utils import init_obj, init_transforms
from pathlib import Path
import numpy as np
from typing import Union


class IRMASDataset(Dataset): 
    def __init__(self, audio_dir: Union[str, Path], preprocess, signal_augments=None,
                 transforms=None, spec_augments=None, subset: str="train"):
     
        self.files = get_wav_files(audio_dir)
        self.subset = subset

        if self.subset != "train":
            test_songs = np.genfromtxt(
                f"{audio_dir}/../../test_songs.txt", dtype=str, ndmin=1, delimiter="\n")
        if self.subset == "valid":
            self.files = [file for file in self.files if Path(
                file).stem not in test_songs]
        if self.subset == "test":
            self.files = [file for file in self.files if Path(
                file).stem in test_songs]

        self.preprocess = preprocess
        self.transforms = transforms
        self.signal_augments = signal_augments
        self.spec_augments = spec_augments

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample_path = self.files[index]
        signal = self.preprocess(sample_path)

        if self.subset == "train":
            target_transforms = torchvision.transforms.Compose([
                ParentMultilabel(sep="-"),
                OneHotEncode(CLASSES)
            ])
        else:
            target_transforms = torchvision.transforms.Compose([
                LabelsFromTxt(),
                OneHotEncode(CLASSES)
            ])

        label = target_transforms(sample_path)

        if self.signal_augments is not None:
            signal = self.signal_augments(signal)

        if self.transforms is not None:
            signal = self.transforms(signal)

        if self.spec_augments is not None:
            signal = self.spec_augments(signal)

        return signal, label.float()


def collate_fn(data):

    features, labels = zip(*data)
    features = [item.squeeze().T for item in features]
    features = pad_sequence(features, batch_first=True)
    labels = torch.stack(labels)

    return features, labels


def get_loader(config, subset: str):

    dst = IRMASDataset(
        config.train_dir if subset == "train" else config.valid_dir,
        preprocess=init_obj(
            config.preprocess, transform_module),
        transforms=init_obj(
            config.transforms, transform_module),
        signal_augments=init_transforms(
            config.signal_augments, transform_module),
        spec_augments=init_transforms(
            config.spec_augments, transform_module),
        subset=subset
    )

    return DataLoader(dst, batch_size=config.batch_size,
                      shuffle=True,
                      pin_memory=True if torch.cuda.is_available() else False,
                      num_workers=torch.get_num_threads(),
                      collate_fn=collate_fn)


if __name__ == "__main__":

    from utils import parse_config
    config = parse_config("config.yaml")
    dl = get_loader(config, subset="train")
    sample, _ = next(iter(dl))
    a = 1
