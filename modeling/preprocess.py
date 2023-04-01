import os, itertools
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed
import pandas as pd
import warnings
from modeling.utils import extract_from_df
from modeling.transforms import LabelsFromTxt, ParentMultilabel


class IRMASPreprocessor:
    def __init__(self, base_dir, new_dir, sr, 
                 sync_pitch: bool=True, sync_bpm: bool=True, sync_onset: bool=True, 
                 ordered: bool=True, metadata: str=None, subset: str="train"):
        
        self.base_dir = Path(base_dir)
        self.new_dir = Path(new_dir)
        self.sr = sr
        self.sync_bpm = sync_bpm
        self.sync_pitch = sync_pitch
        self.sync_onset = sync_onset
        self.ordered = ordered
        self.subset = subset
        
        if metadata is None:
            try:
                self.metadata = pd.read_csv(f"metadata_{self.subset}.csv")
            except:
                warnings.warn("No metadata csv found. New metadata will be generated when you first run the preprocessor.")
                self.metadata = None
        else:
            self.metadata = pd.read_csv(metadata)

        if self.sync_pitch and not self.sync_bpm:
            self.order_by = "pitch"
        else:
            self.order_by = "bpm"
        
        self.instruments = [inst.stem for inst in self.base_dir.iterdir() if inst.is_dir()]
    
    def preprocess_and_combine_all(self, n_jobs: int=-1):

        if self.metadata is None:
            self.metadata = self.generate_metadata()
        
        combs = itertools.product(self.instruments, repeat=2)
    
        if self.ordered:
            self.metadata = self.metadata.sort_values(by=self.order_by)

        #for (instr1, instr2) in tqdm(combs):
        #    self._combine(instr1, instr2)
        
        Parallel(n_jobs=n_jobs) \
            (delayed(self._combine) \
            (instr1, instr2) for (instr1, instr2) in tqdm(combs))
        print("Parallel preprocessing done!")
    
    def _combine(self, instr1, instr2):
        
        if not self.ordered:
            self.metadata = self.metadata.sample(frac=1)
        
        save_dir = self._create_save_dir(instr1, instr2)
            
        instr1_files = self._get_filepaths(instr1)
        instr2_files = self._get_filepaths(instr2)

        if len(instr1_files) < len(instr2_files):
            diff = len(instr2_files) - len(instr1_files)
            instr1_files = np.pad(instr1_files, (0, diff), mode="symmetric")
        
        instr1_files = [Path(x) for x in instr1_files]
        instr2_files = [Path(x) for x in instr2_files]

        self._combine_files_and_save(instr1_files, instr2_files, save_dir)
    
    def _get_filepaths(self, instr):
        metadata = self.metadata.loc[self.metadata.instr==instr]

        if metadata.empty:
            raise KeyError("Instrument not found. Please regenerate metadata!")

        files = metadata.apply(lambda x: f'{self.base_dir}/{x.instr}/{x.fname}.wav', axis=1).to_numpy()
        return files

    def _combine_files_and_save(self, instr1_files, instr2_files, save_dir):
        
        for (file1, file2) in zip(instr1_files, instr2_files):
            synced_file = self._sync_and_combine(str(file1), str(file2))
            new_name = f'{file1.stem}-{file2.stem}.wav'
            sf.write(os.path.join(save_dir, new_name), synced_file, samplerate=self.sr)
    
    def _create_save_dir(self, instr1, instr2):
        new_dir_name = '-'.join([instr1, instr2])
        new_dir_path = os.path.join(self.new_dir, new_dir_name)
        os.makedirs(new_dir_path, exist_ok=True)
        return new_dir_path
    
    def _sync_pitch(self, file_to_sync, sr, pitch_base, pitch=None):
  
        if pitch is None:
            pitch = self._get_pitch(file_to_sync, sr)

        if (pitch_base == -9999) or (pitch == -9999):
            return file_to_sync

        steps = np.round(12 * np.log2( np.exp(pitch_base)/np.exp(pitch) ), 0)

        return librosa.effects.pitch_shift(y=file_to_sync, sr=sr, n_steps=steps)

    def _sync_bpm(self, file_to_sync, sr, bpm_base: float, bpm: float=None):
            
        if bpm is None:
            bpm = self._get_bpm(file_to_sync, sr)
        
        if (bpm_base==0) or (bpm==0):
            return file_to_sync

        return librosa.effects.time_stretch(y=file_to_sync, rate=bpm_base/bpm)

    def _sync_onset(self, file_to_sync, sr, onset_base, onset=None):

        if onset is None:
            onset = self._get_onset(file_to_sync, sr)
        diff = int(round(abs(onset_base*sr - onset*sr),0))
        
        if onset_base > onset:
            return np.pad(file_to_sync, (diff,), mode="constant", constant_values=0)
        else: 
            return file_to_sync[diff:]

    def _sync_and_combine(self, base_file_path, file_to_sync_path):

        base_file, sr = librosa.load(base_file_path, sr=None, mono=True)
        file_to_sync, sr_sync = librosa.load(file_to_sync_path, sr=None, mono=True)

        if sr!=sr_sync:
            file_to_sync = librosa.resample(y=file_to_sync, orig_sr=sr_sync, target_sr=sr)

        base_file_name = Path(base_file_path).stem
        file_to_sync_name = Path(file_to_sync_path).stem

        cols = ["pitch", "bpm", "onset"]
        
        base_metadata = self.metadata.loc[self.metadata.fname==base_file_name, cols]
        if not base_metadata.empty:
            pitch_base, bpm_base, onset_base = extract_from_df(base_metadata, cols)
        else:
            raise KeyError("File not found in metadata. Please regenerate")
        
        file_to_sync_metadata = self.metadata.loc[self.metadata.fname==file_to_sync_name, cols]
        if not file_to_sync_metadata.empty:
            pitch, bpm, onset = extract_from_df(file_to_sync_metadata, cols)
        else:
            raise KeyError("File not found in metadata. Please regenerate")
        
        if self.sync_bpm:
            onset = None
            file_to_sync = self._sync_bpm(file_to_sync, sr, bpm_base, bpm=bpm)

        if self.sync_pitch:
            file_to_sync = self._sync_pitch(file_to_sync, sr, pitch_base, pitch=pitch)
        
        if self.sync_onset:
            file_to_sync = self._sync_onset(file_to_sync, sr, onset_base, onset=onset)
    
        base_file = librosa.util.normalize(base_file)
        file_to_sync = librosa.util.normalize(file_to_sync)
        file_to_sync = np.resize(file_to_sync, base_file.shape)

        combined = (base_file + file_to_sync) / 2

        return librosa.resample(y=combined, orig_sr=sr, target_sr=self.sr)
    
    def generate_metadata(self, save_path: str=".", n_jobs: int=-1):

        if self.subset == "train":
            pattern = '(.*)__[\d]+$'
            label_extractor = ParentMultilabel()
        else:
            pattern = '(.*)-[\d]+$'
            label_extractor = LabelsFromTxt()

        sound_files = list(self.base_dir.glob("**/*.wav"))

        output = Parallel(n_jobs=n_jobs) \
                    (delayed(self._get_file_info) \
                    (str(path)) for path in tqdm(sound_files))
        
        cols = ["path", "pitch", "bpm", "onset", "sample_rate", "duration", "channels"]
        df = pd.DataFrame(data=output, columns=cols)

        df["fname"] = df.path.map(lambda x: Path(x).stem)
        df["song_name"] = df.fname.str.extract(pattern)
        df["instr"] = df.path.map(lambda x: "-".join(list(label_extractor(x))))
        df["label_count"] = df.instr.map(lambda x: len(x.split("-")))
        df = df.drop(columns="path")

        df.to_csv(f'{save_path}/metadata_{self.subset}.csv', index=False)
        self.metadata = df
        
        return df

    def _get_file_info(self, path: str):

        signal, sr = self._load_raw_file(str(path))
        channels = signal.shape[0]
        
        signal = librosa.to_mono(signal)
        pitch = self._get_pitch(signal, sr)
        bpm = self._get_bpm(signal, sr)
        onset = self._get_onset(signal, sr)
        duration = len(signal) / sr

        return path, pitch, bpm, onset, sr, duration, channels
    
    def _load_raw_file(self, path):
        return librosa.load(path, sr=None, mono=False)
    
    def _get_onset(self, signal, sr):
        onset = librosa.onset.onset_detect(y=signal, sr=sr, units="time")[0]
        return onset
    
    def _get_bpm(self, signal, sr):
        bpm, _ = librosa.beat.beat_track(y=signal, sr=sr)
        return bpm
    
    def _get_pitch(self, signal, sr):
        eps = 1e-6
        fmin = librosa.note_to_hz("C2")
        fmax = librosa.note_to_hz('C7')

        pitch, _, _ = librosa.pyin(y=signal, sr=sr, fmin=fmin, fmax=fmax)
        
        if not np.isnan(pitch).all():
            mean_log_pitch = np.nanmean(np.log(pitch+eps))
        else:
            mean_log_pitch = -9999
        
        return mean_log_pitch

if __name__=="__main__":
    
    data_dir = f'./data/raw/IRMAS_Training_Data'
    new_dir = f'./data/processed/pitch_sync/IRMAS_Training_Data'

    preprocessor = IRMASPreprocessor(data_dir, new_dir, sr=16000, ordered=True, sync_bpm=False, 
                                     sync_pitch=True, metadata="./metadata_train.csv",
                                     subset="train")
    #preprocessor.generate_metadata()
    preprocessor.preprocess_and_combine_all()