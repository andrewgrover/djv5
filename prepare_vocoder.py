# prepare_vocoder_data.py

import torch
import torchaudio
from pathlib import Path

# Match your fine-tuning mel parameters:
SR       = 44100
N_FFT    = 2048
HOP      = 512
N_MELS   = 128

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP,
    n_mels=N_MELS,
    f_min=0.0,
    f_max=SR/2,
    power=1.0          # AMPLITUDE mel, *not* power=2.0
)

def main(root_dir="dataset", out_dir="vocoder_data"):
    root = Path(root_dir)
    out  = Path(out_dir)
    out.mkdir(exist_ok=True)

    for folder in sorted(root.iterdir()):
        if not folder.is_dir(): 
            continue
        ab_path = folder / "AB.wav"
        if not ab_path.exists():
            print(f"Skipping {folder.name}, missing AB.wav")
            continue

        # 1) Load audio
        wav, sr = torchaudio.load(ab_path)
        wav = wav.mean(dim=0, keepdim=True)            # mono
        if sr != SR:
            wav = torchaudio.transforms.Resample(sr, SR)(wav)

        # 2) Compute mel on CPU: (1,128,T)
        mel = mel_transform(wav).clamp(min=1e-5)         # amplitude mel
        # optional: cast to float32
        mel = mel.float()

        # 3) Write out
        target_folder = out / folder.name
        target_folder.mkdir(exist_ok=True)
        torch.save(mel, target_folder / "AB_mel.pt")
        torchaudio.save(target_folder / "AB.wav", wav, SR)

        print(f"Prepared {folder.name}: mel→{mel.shape}, wav→{wav.shape}")

if __name__ == "__main__":
    main()
