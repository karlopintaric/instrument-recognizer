import itertools
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.autonotebook import tqdm

from lumen_irmas.transforms import LabelsFromTxt, ParentMultilabel
from lumen_irmas.utils import get_file_info, sync_bpm, sync_onset, sync_pitch


def generate_metadata(
    data_dir: Union[str, Path],
    save_path: str = ".",
    subset: str = "train",
    extract_music_features: bool = False,
    n_jobs: int = -2,
):
    """
    Generate metadata CSV file containing information about audio files in a directory.

    :param data_dir: Directory containing audio files.
    :type data_dir: Union[str, Path]
    :param save_path: Directory path to save metadata CSV file.
    :type save_path: str
    :param subset: Subset of the dataset (train or test), defaults to 'train'.
    :type subset: str
    :param extract_music_features: Flag to indicate whether to extract music features or not, defaults to False.
    :type extract_music_features: bool
    :param n_jobs: Number of parallel jobs to run, defaults to -2.
    :type n_jobs: int
    :raises FileNotFoundError: If the provided data directory does not exist.
    :return: DataFrame containing the metadata information.
    :rtype: pandas.DataFrame
    """

    data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir

    if subset == "train":
        pattern = r"(.*)__[\d]+$"
        label_extractor = ParentMultilabel()
    else:
        pattern = r"(.*)-[\d]+$"
        label_extractor = LabelsFromTxt()

    sound_files = list(data_dir.glob("**/*.wav"))
    output = Parallel(n_jobs=n_jobs)(delayed(get_file_info)(path, extract_music_features) for path in tqdm(sound_files))

    df = pd.DataFrame(data=output)

    df["fname"] = df.path.map(lambda x: Path(x).stem)
    df["song_name"] = df.fname.str.extract(pattern)
    df["inst"] = df.path.map(lambda x: "-".join(list(label_extractor(x))))
    df["label_count"] = df.inst.map(lambda x: len(x.split("-")))

    df.to_csv(f"{save_path}/metadata_{subset}.csv", index=False)

    return df


def create_test_split(metadata_path: str, txt_save_path: str, random_state: Optional[int] = None):
    """Create test split by generating a list of test songs and saving them to a text file.

    :param metadata_path: Path to the CSV file containing metadata of all songs
    :type metadata_path: str
    :param txt_save_path: Path to the directory where the text file containing test songs will be saved
    :type txt_save_path: str
    :param random_state: Seed value for the random number generator, defaults to None
    :type random_state: int, optional
    :raises TypeError: If metadata_path or txt_save_path is not a string or if random_state is not an integer or None
    :raises FileNotFoundError: If metadata_path does not exist
    :raises PermissionError: If the program does not have permission to write to txt_save_path
    :return: None
    :rtype: None
    """

    df = pd.read_csv(metadata_path)
    kf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=random_state)
    splits = kf.split(df.fname, df.inst, groups=df.song_name)
    _, test = list(splits)[0]

    test_songs = df.iloc[test].fname.sort_values().to_numpy()

    with open(f"{txt_save_path}/test_songs.txt", "w") as f:
        # iterate over the list of names and write each one to a new line in the file
        for song in test_songs:
            f.write(song + "\n")


class IRMASPreprocessor:
    """
    A class to preprocess IRMAS dataset metadata and create a mapping between
    file paths and their corresponding instrument labels.

    :param metadata: A pandas DataFrame or path to csv file containing metadata, defaults to None
    :type metadata: Union[pd.DataFrame, str], optional
    :param data_dir: Path to the directory containing the IRMAS dataset, defaults to None
    :type data_dir: Union[str, Path], optional
    :param sample_rate: Sample rate of the audio files, defaults to 16000
    :type sample_rate: int, optional

    :raises AssertionError: Raised when metadata is None and data_dir is also None.

    :return: An instance of IRMASPreprocessor
    :rtype: IRMASPreprocessor
    """

    def __init__(
        self, metadata: Union[pd.DataFrame, str] = None, data_dir: Union[str, Path] = None, sample_rate: int = 16000
    ):
        if metadata is not None:
            self.metadata = pd.read_csv(metadata) if isinstance(metadata, str) else metadata
            if data_dir is not None:
                self.metadata["path"] = self.metadata.apply(lambda x: f"{data_dir}/{x.inst}/{x.fname}.wav", axis=1)
        else:
            assert data_dir is not None, "No metadata found. Need to provide data directory"
            self.metadata = generate_metadata(data_dir=data_dir, subset="train", extract_music_features=True)

        self.instruments = self.metadata.inst.unique()
        self.sample_rate = sample_rate

    def preprocess_and_mix(self, save_dir: str, sync: str, ordered: bool, num_track_to_mix: int, n_jobs: int = -2):
        """
        A method to preprocess and mix audio tracks from the IRMAS dataset.

        :param save_dir: The directory to save the preprocessed and mixed tracks
        :type save_dir: str
        :param sync: The column name used to synchronize the audio tracks during mixing
        :type sync: str
        :param ordered: Whether to order the metadata by the sync column before mixing the tracks
        :type ordered: bool
        :param num_track_to_mix: The number of tracks to mix together
        :type num_track_to_mix: int
        :param n_jobs: The number of parallel jobs to run, defaults to -2
        :type n_jobs: int, optional

        :raises None

        :return: None
        :rtype: None
        """

        combs = itertools.combinations(self.instruments, r=num_track_to_mix)

        if ordered:
            self.metadata = self.metadata.sort_values(by=sync)
        else:
            self.metadata = self.metadata.sample(frac=1)

        Parallel(n_jobs=n_jobs)(delayed(self._mix)(insts, save_dir, sync) for (insts) in tqdm(combs))
        print("Parallel preprocessing done!")

    def _mix(self, insts: Tuple[str], save_dir: str, sync: str):
        """
        A private method to mix audio tracks and save them to disk.

        :param insts: A tuple of instrument labels to mix
        :type insts: Tuple[str]
        :param save_dir: The directory to save the mixed tracks
        :type save_dir: str
        :param sync: The column name used to synchronize the audio tracks during mixing
        :type sync: str

        :raises None

        :return: None
        :rtype: None
        """

        save_dir = self._create_save_dir(insts, save_dir)

        insts_files_list = [self._get_filepaths(inst) for inst in insts]

        max_length = max([inst_files.shape[0] for inst_files in insts_files_list])
        for i, inst_files in enumerate(insts_files_list):
            if inst_files.shape[0] < max_length:
                diff = max_length - inst_files.shape[0]
                inst_files = np.pad(inst_files, (0, diff), mode="symmetric")
            insts_files_list[i] = [Path(x) for x in inst_files]

        self._mix_files_and_save(insts_files_list, save_dir, sync)

    def _get_filepaths(self, inst: str):
        """
        A private method to retrieve file paths of audio tracks for a given instrument label.

        :param inst: The label of the instrument for which to retrieve the file paths
        :type inst: str

        :raises KeyError: Raised when the instrument label is not found in the metadata.

        :return: A numpy array of file paths corresponding to the instrument label.
        :rtype: numpy.ndarray
        """

        metadata = self.metadata.loc[self.metadata.inst == inst]

        if metadata.empty:
            raise KeyError("Instrument not found. Please regenerate metadata!")

        files = metadata.path.to_numpy()

        return files

    def _mix_files_and_save(self, insts_files_list: List[List[Path]], save_dir: str, sync: str):
        """
        A private method to mix audio files, synchronize them using a given column name in the metadata,
        and save the mixed file to disk.

        :param insts_files_list: A list of lists of file paths corresponding to each instrument label
        :type insts_files_list: List[List[Path]]
        :param save_dir: The directory to save the mixed tracks
        :type save_dir: str
        :param sync: The column name used to synchronize the audio tracks during mixing
        :type sync: str

        :raises None

        :return: None
        :rtype: None
        """

        for i in range(len(insts_files_list[0])):
            files_to_sync = [inst_files[i] for inst_files in insts_files_list]
            new_name = f"{'-'.join([file.stem for file in files_to_sync])}.wav"

            synced_file = self._sync_and_mix(files_to_sync, sync)
            sf.write(os.path.join(save_dir, new_name), synced_file, samplerate=self.sample_rate)

    def _sync_and_mix(self, files_to_sync: List[Path], sync: str):
        """
        Synchronize and mix audio files.

        :param files_to_sync: A list of file paths to synchronize and mix.
        :type files_to_sync: List[Path]
        :param sync: The type of synchronization to use. One of ['bpm', 'pitch', None].
        :type sync: str, optional
        :raises KeyError: If any file in files_to_sync is not found in metadata.
        :return: The synchronized and mixed audio signal.
        :rtype: numpy.ndarray
        """

        cols = ["pitch", "bpm", "onset"]
        files_metadata_df = self.metadata.loc[
            self.metadata.path.isin([str(file_path) for file_path in files_to_sync])
        ].set_index("path")

        num_files = files_metadata_df.shape[0]
        if num_files != len(files_to_sync):
            raise KeyError("File not found in metadata. Please regenerate")

        if sync is not None:
            mean_features = files_metadata_df[cols].mean().to_dict()

        metadata_dict = files_metadata_df.to_dict("index")

        for i, (file_to_sync_path, features) in enumerate(metadata_dict.items()):
            file_to_sync, sr_sync = librosa.load(file_to_sync_path, sr=None)

            if sr_sync != 44100:
                file_to_sync = librosa.resample(y=file_to_sync, orig_sr=sr_sync, target_sr=self.sample_rate)

            if sync == "bpm":
                file_to_sync = sync_bpm(file_to_sync, sr_sync, bpm_base=mean_features["bpm"], bpm=features["bpm"])

            if sync == "pitch":
                file_to_sync = sync_pitch(
                    file_to_sync, sr_sync, pitch_base=mean_features["pitch"], pitch=features["pitch"]
                )

            if sync is not None:
                file_to_sync = sync_onset(
                    file_to_sync, sr_sync, onset_base=mean_features["onset"], onset=features["onset"]
                )

            file_to_sync = librosa.util.normalize(file_to_sync)

            if i == 0:
                mixed_sound = np.zeros_like(file_to_sync)

            if mixed_sound.shape[0] > file_to_sync.shape[0]:
                file_to_sync = np.resize(file_to_sync, mixed_sound.shape)
            else:
                mixed_sound = np.resize(mixed_sound, file_to_sync.shape)

            mixed_sound += file_to_sync

        mixed_sound /= num_files

        return librosa.resample(y=mixed_sound, orig_sr=44100, target_sr=self.sample_rate)

    def _create_save_dir(self, insts: Union[Tuple[str], List[str]], save_dir: str):
        """
        Create and return a directory to save instrument-specific files.

        :param insts: A tuple or list of instrument names.
        :type insts: Union[Tuple[str], List[str]]
        :param save_dir: The path to the directory where the new directory will be created.
        :type save_dir: str
        :return: The path to the newly created directory.
        :rtype: str
        """

        new_dir_name = "-".join(insts)
        new_dir_path = os.path.join(save_dir, new_dir_name)
        os.makedirs(new_dir_path, exist_ok=True)
        return new_dir_path

    @classmethod
    def from_metadata(cls, metadata_path: str, **kwargs):
        """
        Create a new instance of the class from a metadata file.

        :param metadata_path: The path to the metadata file.
        :type metadata_path: str
        :param **kwargs: Additional keyword arguments to pass to the class constructor.
        :return: A new instance of the class.
        :rtype: cls
        """

        metadata = pd.read_csv(metadata_path)
        return cls(metadata, **kwargs)


if __name__ == "__main__":
    data_dir = "/home/kpintaric/lumen-irmas/data/raw/IRMAS_Training_Data"
    metadata_path = "/home/kpintaric/lumen-irmas/data/metadata_train.csv"
    preprocess = IRMASPreprocessor(metadata=metadata_path, data_dir=data_dir)
    preprocess.preprocess_and_mix(save_dir="data", sync="pitch", ordered=False, num_track_to_mix=3)
    a = 1
