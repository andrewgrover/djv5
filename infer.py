import torch
import torchaudio
import soundfile as sf
from model import TransitionGenerator
from vocoder import Vocoder

def get_device():
    return torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')


def load_audio(path, sample_rate=44100):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
    return wav

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', default='transition.wav')
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load transition model
    model = TransitionGenerator().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # Load and transform audio on CPU
    a = load_audio(args.a)  # (1, samples)
    b = load_audio(args.b)
    # when you define your MelSpectrogram, use amplitude directly
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        f_min=0,
        f_max=22050,
        power=1.0            # <-- amplitude, not power
    )
# then *don’t* log/exp at all—train & infer on linear mels.

    # Compute log-mels on CPU: each → (128, time)
    mel_a = mel_transform(a).clamp(min=1e-5).log()
    mel_b = mel_transform(b).clamp(min=1e-5).log()

    # Move to device + add channel dimension for your model: (1, 1, 128, time)
    mel_a = mel_a.unsqueeze(1).to(device)
    mel_b = mel_b.unsqueeze(1).to(device)

    # Predict transition log-mel
    with torch.no_grad():
        mel_ab = model(mel_a, mel_b)  # → (1, 128, time)

    # Synthesize audio via BigVGAN
    voc = Vocoder(transition_device=device)
    # exponentiate back to linear mel
    audio = voc.mel_to_audio(torch.exp(mel_ab))  # NumPy array, shape (1, samples) or (1, channels, samples)

    # Prepare and flatten the output waveform
    wav = audio[0]
    # If shape is (channels, samples), transpose to (samples, channels)
    if wav.ndim > 1:
        wav = wav.T
    # Ensure mono 1-D array if possible
    wav = wav.squeeze()

    # Write with explicit format & subtype
    sf.write(
        file=args.out,
        data=wav,
        samplerate=44100,
        format='WAV',
        subtype='PCM_16'
    )
    print(f"Saved transition to {args.out}")
