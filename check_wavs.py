# check_wavs.py
import os
import torchaudio

root = "dataset"  # or wherever your data is
for d in sorted(os.listdir(root)):
    path = os.path.join(root, d)
    if not os.path.isdir(path):
        continue
    for name in ("A.wav","AB.wav","B.wav"):
        p = os.path.join(path, name)
        try:
            _ = torchaudio.load(p)
        except Exception as e:
            print(f"✗ failed to load {p}: {e}")
