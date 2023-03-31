import sys

sys.path.append(".")

from mel2wav.modules import Generator, Audio2Mel
from mel2wav.utils import save_sample

import torch
import torch.nn.functional as F
from librosa.core import load
from librosa.util import normalize

import os
import numpy as np
import argparse
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_audio(filename, sampling_rate=22050, augment=False):
    data, sampling_rate = load(filename, sr=sampling_rate)
    data = 0.95 * normalize(data)
    if augment:
        amplitude = np.random.uniform(low=0.3, high=1.0)
        data = data * amplitude
    # Take a segment of the data
    data = data[: sampling_rate * 60]
    return torch.from_numpy(data).float().unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, required=True, help="Input audio WAV file"
    )
    parser.add_argument(
        "--load_path",
        default=None,
        help="Location of the model, should contain netG.pt",
    )
    parser.add_argument("--output_path", default="output")

    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=3)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Setup paths
    load_root = Path(args.load_path)
    if not load_root.exists:
        print(f"Model path {load_root} does not exist.")
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = os.path.basename(args.input_file)

    # Load Models
    netG = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).to(device)
    netG.load_state_dict(torch.load(load_root / "netG.pt"))
    fft = Audio2Mel(n_mel_channels=args.n_mel_channels).to(device)
    
    print(f'Loading {args.input_file}')
    x_t = load_audio(args.input_file)
    print("x_t", x_t.shape)
    x_t = x_t.to(device)
    with torch.no_grad():
        s_t = fft(x_t)
        x_pred_t = netG(s_t.to(device))
        print("x_pred_t", x_pred_t.shape)
        gen_audio = x_pred_t.squeeze().cpu()
        save_sample(output_path / output_file, 22500, gen_audio)
        print(f'Saved as {output_path / output_file}')


if __name__ == "__main__":
    main()
