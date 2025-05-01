import torch, torchaudio
from torch.utils.data import Dataset, DataLoader
import sys
# 1) Point this at wherever you cloned NVIDIA/BigVGAN:
repo_path = "/Users/andyg/Desktop/DJv5/BigVGAN"
sys.path.insert(0, repo_path)
from bigvgan import BigVGAN
from pathlib import Path

# Hyperparams
PRETRAINED = "nvidia/bigvgan_v2_44khz_128band_256x"
BATCH_SIZE  = 4
LR          = 1e-5
STEPS       = 5000
DEVICE      = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

class MelWavDataset(Dataset):
    def __init__(self, root_dir):
        self.root = Path(root_dir)
        # gather all the AB_mel.pt files
        self.pairs = list(self.root.glob("*/AB_mel.pt"))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        mel_path = self.pairs[i]                     # e.g. vocoder_data/057/AB_mel.pt
        wav_path = mel_path.parent / "AB.wav"        # point at vocoder_data/057/AB.wav

        # Load mel (shape: [1, 128, T])
        mel = torch.load(mel_path).squeeze(0)         # → [128, T]

        # Load raw audio
        wav, sr = torchaudio.load(wav_path)          # → [1, samples], sr should be 44100
        wav = wav.squeeze(0)                         # → [samples]

        return mel, wav


def multi_res_stft_loss(x, y):
    # import or implement MR-STFT + spectral convergence + mag L1
    ...

def main():
    # 1) model
    voc = BigVGAN.from_pretrained(PRETRAINED).to(DEVICE).train()

    # 2) data
    ds = MelWavDataset("vocoder_data")
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 3) optimizer & losses
    opt = torch.optim.Adam(voc.parameters(), lr=LR)
    mse = torch.nn.MSELoss()

    step = 0
    while step < STEPS:
        for mel, wav in dl:
            mel = mel.to(DEVICE)        # [B,128,T]
            wav = wav.to(DEVICE)        # [B, samples]

            # 4) forward
            pred = voc(mel)             # [B,1,samples]
            pred = pred.squeeze(1)

            # 5) compute loss
            loss_wav = mse(pred, wav)
            loss_spec = multi_res_stft_loss(pred, wav)
            loss = loss_wav + 2*loss_spec

            # 6) backward
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 500 == 0:
                print(f"Step {step}: wav {loss_wav.item():.4f} spec {loss_spec.item():.4f}")
            if step % 1000 == 0:
                torch.save(voc.state_dict(), f"vocoder_ft_step{step}.pt")
            step += 1
            if step >= STEPS:
                break

if __name__=="__main__":
    main()
