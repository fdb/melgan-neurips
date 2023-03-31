import sys
sys.path.append('.')

import numpy as np
from mel2wav.modules import Audio2Mel
from latent_mel.dataset import MelDataset
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running on {device}.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True)

    parser.add_argument("--n_mel_channels", type=int, default=80)

    parser.add_argument("--data_path", default='.', type=Path)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=8192)

    parser.add_argument("--epochs", type=int, default=3000)
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

    # Datasets
    train_set = MelDataset(
        Path(args.data_path) / "train_files.txt", args.seq_len, sampling_rate=22050, n_mel_channels=args.n_mel_channels,
    )
    test_set = MelDataset(
        Path(args.data_path) / "test_files.txt",
        22050 * 4,
        sampling_rate=22050,
        n_mel_channels=args.n_mel_channels,
        augment=False,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=1)


    steps = 0
    for epoch in range(1, args.epochs + 1):
        for iterno, x_t in enumerate(train_loader):
            x_t = x_t.to(device)
            print('x_t shape', x_t.shape)


if __name__ == "__main__":
    main()
