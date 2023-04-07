import os
import itertools
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed
import pandas as pd
from utils import load_raw_file, get_file_info, sync_bpm, sync_onset, sync_pitch
from transforms import LabelsFromTxt, ParentMultilabel
from typing import Union, List, Tuple


def generate_metadata(data_dir: Union[str, Path],
                      save_path: str = ".", subset: str = "train",
                      extract_music_features: bool = False,
                      n_jobs: int = -2):

    data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir

    if subset == "train":
        pattern = '(.*)__[\d]+$'
        label_extractor = ParentMultilabel()
    else:
        pattern = '(.*)-[\d]+$'
        label_extractor = LabelsFromTxt()

    sound_files = list(data_dir.glob("**/*.wav"))

    for path in tqdm(sound_files):
        output = get_file_info(path, extract_music_features)
    #output = Parallel(n_jobs=n_jobs)(delayed(get_file_info)(path, extract_music_features) for path in tqdm(sound_files))

    cols = ["path", "pitch", "bpm", "onset",
            "sample_rate", "duration", "channels"]
    df = pd.DataFrame(data=output, columns=cols)

    df["fname"] = df.path.map(lambda x: Path(x).stem)
    df["song_name"] = df.fname.str.extract(pattern)
    df["inst"] = df.path.map(lambda x: "-".join(list(label_extractor(x))))
    df["label_count"] = df.inst.map(lambda x: len(x.split("-")))
    # df = df.drop(columns="path")

    df.to_csv(f'{save_path}/metadata_{subset}.csv', index=False)

    return df


class IRMASPreprocessor:

    def __init__(self, metadata: Union[pd.DataFrame, str] = None,
                 data_dir: Union[str, Path] = None,
                 sample_rate: int = 16000):

        if metadata is not None:
            self.metadata = pd.read_csv(metadata) if isinstance(
                metadata, str) else metadata
            if data_dir is not None:
                self.metadata["path"] = self.metadata.apply(
                    lambda x: f'{data_dir}/{x.inst}/{x.fname}.wav', axis=1
                )
        else:
            assert data_dir is not None, "No metadata found. Need to provide data directory"
            self.metadata = generate_metadata(data_dir=data_dir,
                                              subset="train",
                                              extract_music_features=True)

        self.instruments = self.metadata.inst.unique()
        self.sample_rate = sample_rate

    def preprocess_and_mix(self, save_dir: str, sync: str, sr: int,
                           ordered: bool, num_track_to_mix: int, n_jobs: int = -2):

        combs = itertools.combinations(
            self.instruments, repeat=num_track_to_mix)

        if ordered:
            self.metadata = self.metadata.sort_values(by=sync)
        else:
            self.metadata = self.metadata.sample(frac=1)

        Parallel(n_jobs=n_jobs)(delayed(self._mix)
                                (insts, save_dir, sync, sr) for (insts) in tqdm(combs))
        print("Parallel preprocessing done!")

    def _mix(self, insts: Tuple[str], save_dir: str, sync: str):

        save_dir = self._create_save_dir(insts)

        insts_files_list = [self._get_filepaths(inst) for inst in insts]

        max_length = max([inst_files for inst_files in insts_files_list])
        for i, inst_files in enumerate(insts_files_list):
            if inst_files.shape[0] < max_length:
                diff = max_length - inst_files.shape[0]
                inst_files = np.pad(inst_files, (0, diff), mode="symmetric")
                insts_files_list[i] = [Path(x) for x in inst_files]

        self._mix_files_and_save(insts_files_list, save_dir, sync)

    def _get_filepaths(self, inst: str):
        metadata = self.metadata.loc[self.metadata.instr == inst]

        if metadata.empty:
            raise KeyError("Instrument not found. Please regenerate metadata!")

        files = metadata.path.to_numpy()

        return files

    def _mix_files_and_save(self, insts_files_list: List[List[Path]], save_dir: str, sync: str):

        for i in range(len(insts_files_list[0])):
            files_to_sync = [inst_files[i] for inst_files in insts_files_list]
            new_name = f"{'-'.join([file.stem for file in files_to_sync])}.wav"

            synced_file = self._sync_and_mix(files_to_sync, sync)
            sf.write(os.path.join(save_dir, new_name),
                     synced_file, samplerate=self.sample_rate)

    def _sync_and_mix(self, files_to_sync: List[Path], sync: str):

        cols = ["path", "pitch", "bpm", "onset"]
        files_metadata_df = self.metadata.loc[self.metadata.path.isin(
            files_to_sync)].set_index("path")

        num_files = files_metadata_df.shape[0]
        if num_files != len(files_to_sync):
            raise KeyError("File not found in metadata. Please regenerate")

        mean_features = files_metadata_df[cols].mean().to_dict()
        metadata_dict = files_metadata_df.to_dict("index")

        for i, (file_to_sync_path, features) in enumerate(metadata_dict.items()):

            file_to_sync, sr_sync = load_raw_file(file_to_sync_path)

            if sr_sync != 44100:
                file_to_sync = librosa.resample(
                    y=file_to_sync, orig_sr=sr_sync, target_sr=self.sample_rate)

            if sync == "bpm":
                file_to_sync = sync_bpm(
                    file_to_sync, sr_sync, bpm_base=mean_features["bpm"], bpm=features["bpm"])

            if sync == "pitch":
                file_to_sync = sync_pitch(
                    file_to_sync, sr_sync, pitch_base=mean_features["pitch"], pitch=features["pitch"])

            file_to_sync = sync_onset(
                file_to_sync, sr_sync, onset_base=mean_features["onset"], onset=features["onset"])

            file_to_sync = librosa.util.normalize(file_to_sync)

            if i == 0:
                mixed_sound = np.zeros_like(file_to_sync)

            if mixed_sound.shape[0] > file_to_sync.shape[0]:
                file_to_sync = np.resize(file_to_sync, mixed_sound.shape)
            else:
                mixed_sound = np.resize(mixed_sound, file_to_sync.shape)

            mixed_sound = mixed_sound + file_to_sync

        mixed_sound /= num_files

        return librosa.resample(y=mixed_sound, orig_sr=44100, target_sr=self.sample_rate)

    def _create_save_dir(self, insts: Union[Tuple[str], List[str]]):
        new_dir_name = '-'.join(insts)
        new_dir_path = os.path.join(self.new_dir, new_dir_name)
        os.makedirs(new_dir_path, exist_ok=True)
        return new_dir_path

    @classmethod
    def from_metadata(cls, metadata_path: str, **kwargs):
        metadata = pd.read_csv(metadata_path)
        return cls(metadata, **kwargs)


if __name__ == "__main__":

    data_dir = '/home/kpintaric/lumen-irmas/data/raw/IRMAS_Training_Data'
    generate_metadata(data_dir)
    a = 1
