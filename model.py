import torch
import torch.nn as nn

class TransitionGenerator(nn.Module):
    def __init__(self, n_mels=128, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2 * n_mels, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, n_mels, kernel_size=5, padding=2)
        )

    def forward(self, mel_a, mel_b):
        mel_a = mel_a.squeeze(1) if mel_a.dim() == 4 else mel_a
        mel_b = mel_b.squeeze(1) if mel_b.dim() == 4 else mel_b
        x = torch.cat([mel_a, mel_b], dim=1)
        z = self.encoder(x)
        out = self.decoder(z)
        return out