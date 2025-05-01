import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class TransitionDataset(Dataset):
    def __init__(
        self, root_dir,
        sample_rate=44100,   # switch to 44.1 kHz
        n_mels=128,          # match BigVGAN-v2 128-band
        n_fft=2048,          # increased FFT size
        hop_length=512       # hop length for 128-band mel
    ):
        self.root = root_dir
        # only include valid subfolders
        all_items = sorted(os.listdir(root_dir))
        self.folders = [
            d for d in all_items
            if os.path.isdir(os.path.join(root_dir, d))
            and os.path.exists(os.path.join(root_dir, d, 'A.wav'))
            and os.path.exists(os.path.join(root_dir, d, 'AB.wav'))
            and os.path.exists(os.path.join(root_dir, d, 'B.wav'))
        ]
        self.sample_rate = sample_rate
        # updated mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0,
            f_max=sample_rate/2
        )

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        path = os.path.join(self.root, folder)
        a, sr_a = torchaudio.load(os.path.join(path, 'A.wav'))
        ab, sr_ab = torchaudio.load(os.path.join(path, 'AB.wav'))
        b, sr_b = torchaudio.load(os.path.join(path, 'B.wav'))

        # mono
        a = a.mean(dim=0, keepdim=True)
        ab = ab.mean(dim=0, keepdim=True)
        b = b.mean(dim=0, keepdim=True)

        # resample
        if sr_a != self.sample_rate:
            a = torchaudio.transforms.Resample(sr_a, self.sample_rate)(a)
        if sr_ab != self.sample_rate:
            ab = torchaudio.transforms.Resample(sr_ab, self.sample_rate)(ab)
        if sr_b != self.sample_rate:
            b = torchaudio.transforms.Resample(sr_b, self.sample_rate)(b)

        # mel-spectrogram → [1, n_mels, T]
        m_a  = self.mel_transform(a).clamp(min=1e-5).log()
        m_ab = self.mel_transform(ab).clamp(min=1e-5).log()
        m_b  = self.mel_transform(b).clamp(min=1e-5).log()
        # squeeze channel → [n_mels, T]
        mel_a  = m_a.squeeze(0)
        mel_ab = m_ab.squeeze(0)
        mel_b  = m_b.squeeze(0)

        return {'mel_a': mel_a, 'mel_ab': mel_ab, 'mel_b': mel_b}

if __name__ == '__main__':
    ds = TransitionDataset('dataset')
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(dl))
    print({k: v.shape for k, v in batch.items()})