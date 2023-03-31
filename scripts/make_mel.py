import sys
sys.path.append('.')

import numpy as np
from mel2wav.modules import Audio2Mel
import matplotlib.pyplot as plt
import torch
import argparse
from librosa.core import load
from librosa.util import normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

def load_audio(filename, sampling_rate=22050, augment=False):
    data, sampling_rate = load(filename, sr=sampling_rate)
    data = 0.95 * normalize(data)
    if augment:
        amplitude = np.random.uniform(low=0.3, high=1.0)
        data = data * amplitude
    # Take a segment of the data 
    data = data[:sampling_rate * 4]
    return torch.from_numpy(data).float(), sampling_rate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("--n_mel_channels", type=int, default=80)
    args = parser.parse_args()
    return args

def plot_mel(mel_spectrogram):
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time')
    plt.ylabel('Mel frequency bins')
    plt.savefig('mel_spectrogram.png', dpi=300, bbox_inches='tight')

def main():
    args = parse_args()
    data, sampling_rate = load_audio(args.filename)
    data = data.to(device)
    print('data', data.shape)
    print('sr', sampling_rate)

    data_batch = data.unsqueeze(0)
    print('data_batch', data_batch.shape)

    fft = Audio2Mel(n_mel_channels=args.n_mel_channels).to(device)
    mel = fft(data_batch).squeeze(0).cpu()

    print('mel', mel.shape)
    plot_mel(mel)



if __name__ == "__main__":
    main()
