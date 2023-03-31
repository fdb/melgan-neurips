import sys
sys.path.append('.')

import numpy as np
from mel2wav.modules import Audio2Mel
import matplotlib.pyplot as plt
from PIL import Image
import torch
import argparse
from librosa.core import load
from librosa.util import normalize
from pathlib import Path
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

def load_audio(filename, sampling_rate=44100, augment=False):
    data, sampling_rate = load(filename, sr=sampling_rate)
    data = 0.95 * normalize(data)
    if augment:
        amplitude = np.random.uniform(low=0.3, high=1.0)
        data = data * amplitude
    # Take a segment of the data (one minute)
    data = data[:sampling_rate * 60]
    return torch.from_numpy(data).float().unsqueeze(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("--output_path", default="output")
    parser.add_argument("--n_mel_channels", type=int, default=80)
    args = parser.parse_args()
    return args

def plot_mel(mel_spectrogram, output_filename):
    #print(mel_spectrogram.shape)
    mel_spectrogram = mel_spectrogram.cpu().numpy()
    #plt.figure(figsize=(10, 4))
    #plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    #plt.colorbar(format='%+2.0f dB')
    #plt.xlabel('Time')
    #plt.ylabel('Mel frequency bins')
    #plt.savefig(output_filename, dpi=300, bbox_inches='tight')

    normalized_data = (mel_spectrogram - np.min(mel_spectrogram)) / (np.max(mel_spectrogram) - np.min(mel_spectrogram)) * 255
    normalized_data = normalized_data.astype(np.uint8)
    image = Image.fromarray(normalized_data, mode='L')
    image.save(output_filename)

def main():
    args = parse_args()
    x_t = load_audio(args.filename)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    image_name = os.path.basename(args.filename)
    image_name = f'{os.path.splitext(image_name)[0]}.png'
    output_filename = output_path / image_name
    x_t = x_t.to(device)
    print('x_t', x_t.shape)

    fft = Audio2Mel(n_mel_channels=args.n_mel_channels).to(device)
    mel = fft(x_t).squeeze(0).cpu()

    print('mel', mel.shape)
    plot_mel(mel, output_filename)



if __name__ == "__main__":
    main()
