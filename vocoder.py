import sys

# 1) Point this at wherever you cloned NVIDIA/BigVGAN:
repo_path = "/Users/andyg/Desktop/DJv5/BigVGAN"
sys.path.insert(0, repo_path)
# vocoder.py

import torch
import soundfile as sf
from bigvgan import BigVGAN

def get_device():
    """Primary device for your transition model & inference."""
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

class Vocoder:
    def __init__(
        self,
        model_name: str = "nvidia/bigvgan_v2_44khz_128band_256x",
        transition_device: torch.device = None
    ):
        # 1) Transition model & preprocessing on MPS/CPU:
        self.transition_device = transition_device or get_device()
        print(f"[Vocoder] transition-device: {self.transition_device}")

        # 2) BigVGAN on CPU if MPS has channel limits:
        if self.transition_device.type == "mps":
            print("[Vocoder] MPS detected but BigVGAN-v2 is too large for MPS; using CPU for vocoder.")
            self.voc_device = torch.device("cpu")
        else:
            self.voc_device = self.transition_device

        # Load & move the BigVGAN model
        self.vocoder = BigVGAN.from_pretrained(model_name).to(self.voc_device)
        self.vocoder.eval()

    def mel_to_audio(self, mel: torch.Tensor):
        """
        mel: (batch, n_mels, time) on transition_device
        Returns: (batch, samples) NumPy on CPU
        """
        # 1) Ensure the mel is on the vocoder device (CPU)
        mel_cpu = mel.to(self.voc_device)

        # 2) Run vocoder on CPU
        with torch.no_grad():
            audio = self.vocoder(mel_cpu)

        # 3) Return NumPy on CPU
        return audio.cpu().numpy()

if __name__ == "__main__":
    # Quick test
    device = get_device()
    voc = Vocoder(transition_device=device)
    dummy_mel = torch.randn(1, 128, 100, device=device)
    wav = voc.mel_to_audio(dummy_mel)
    sf.write("test.wav", wav[0], samplerate=44100)
    print("Wrote test.wav at 44.1 kHz")
