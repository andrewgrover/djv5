import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from dataset import TransitionDataset
from model import TransitionGenerator

def get_device():
    return torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')


def train(data_root='dataset', epochs=100, batch_size=8, lr=1e-4, device=None):
    device = device or get_device()
    print(f"Using device: {device}")

    ds = TransitionDataset(data_root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

    model = TransitionGenerator().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, epochs+1):
        total_loss = 0
        model.train()
        for batch in dl:
            mel_a = batch['mel_a'].to(device)
            mel_b = batch['mel_b'].to(device)
            mel_ab = batch['mel_ab'].to(device)

            pred = model(mel_a.unsqueeze(1), mel_b.unsqueeze(1))
            loss = criterion(pred, mel_ab)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(dl)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg:.4f}")
        if epoch % 10 == 0:
            ckpt = f"checkpoints/gen_epoch{epoch}.pt"
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.save(model.state_dict(), ckpt)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    train(
        data_root=args.data,
        epochs=args.epochs,
        batch_size=args.bs,
        lr=args.lr
    )