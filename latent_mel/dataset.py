import torch
import torch.utils.data
import torch.nn.functional as F

from librosa.core import load
from librosa.util import normalize

from pathlib import Path
import numpy as np
import random
from mel2wav.modules import Audio2Mel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

class MelDataset(torch.utils.data.Dataset):
    """
    Load a wav file and return a spectrogram.
    """

    def __init__(self, training_files, segment_length, sampling_rate, n_mel_channels, augment=True):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.audio_files = files_to_list(training_files)
        self.audio_files = [Path(training_files).parent / x for x in self.audio_files]
        self.fft = Audio2Mel(n_mel_channels=n_mel_channels).to(device)
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.augment = augment

        # Load all audio files into memory
        self.audio_data = []
        self.fft_data = []
        for audio_file in self.audio_files:
            audio, _ = self.load_wav_to_torch(audio_file)
            self.audio_data.append(audio)
            print('audio shape', audio.shape)
            with torch.no_grad():
                fft = self.fft(audio.unsqueeze(0).to(device))
                self.fft(fft)

    def __getitem__(self, index):
        # Get fft from memory
        fft = self.fft_data[index]

        # Take segment
        if fft.size(0) >= self.segment_length:
            max_audio_start = fft.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            fft = fft[audio_start : audio_start + self.segment_length]
        else:
            fft = F.pad(
                fft, (0, self.segment_length - fft.size(0)), "constant"
            ).data

        # audio = audio / 32768.0
        return fft.unsqueeze(0)

    def __len__(self):
        return len(self.audio_files)

    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data, sampling_rate = load(full_path, sr=self.sampling_rate)
        data = 0.95 * normalize(data)

        if self.augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            data = data * amplitude

        return torch.from_numpy(data).float(), sampling_rate
