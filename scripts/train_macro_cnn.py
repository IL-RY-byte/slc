"""
Train the macro family CNN (Enhancement 1) and wear estimator (Enhancement 2).

Architecture (PLAN.md Enhancement 1):
    Macro CNN:
        Conv2D(32, 3x3, ReLU) -> MaxPool(2x2)     [64->32]
        Conv2D(64, 3x3, ReLU) -> MaxPool(2x2)     [32->16]
        Conv2D(128, 3x3, ReLU) -> MaxPool(2x2)    [16->8]
        GlobalAvgPool -> Linear(256, ReLU) -> Linear(16) [softmax at inference]
    Wear CNN (shared backbone up to GlobalAvgPool, then separate head):
        Linear(64, ReLU) -> Linear(1) [sigmoid at inference]

Usage:
    # Train on synthetic data only
    python scripts/train_macro_cnn.py --data data/cnn_train.npz

    # Train with real captured tiles mixed in
    python scripts/train_macro_cnn.py --data data/cnn_train.npz --real data/real_5k.npz

    # Quick overfitting smoke-test (should reach >99% on a tiny dataset)
    python scripts/train_macro_cnn.py --data /tmp/cnn_smoke.npz --epochs 20 --batch 32

Outputs:
    models/macro_cnn.pt         PyTorch state dict (macro head)
    models/wear_cnn.pt          PyTorch state dict (wear head)
    models/macro_cnn.onnx       ONNX for CoreML / TFLite export
    models/wear_cnn.onnx        ONNX for CoreML / TFLite export

After training, export with:
    python scripts/export_models.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split
except ImportError:
    print('PyTorch not found. Install with: pip install torch', file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TileDataset(Dataset):
    """Loads the .npz produced by generate_cnn_data.py.

    Large datasets store X in a companion _X.npy file (memory-mapped) to avoid
    loading 3+ GB into RAM at once.  Small datasets (smoke tests) may embed X
    directly in the .npz.
    """

    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        n_ok = int(data['n_ok']) if 'n_ok' in data else None

        x_path = npz_path.parent / (npz_path.stem + '_X.npy')
        if x_path.exists():
            # Large dataset: X is memory-mapped from a companion _X.npy file.
            # Each __getitem__ read is a small disk read; the full array is
            # never loaded into RAM simultaneously.
            self._X_mmap = np.load(str(x_path), mmap_mode='r')
            n = n_ok if n_ok is not None else len(self._X_mmap)
            self._X_mmap = self._X_mmap[:n]
            self._use_mmap = True
        else:
            # Small dataset (smoke test): X embedded in npz.
            self._X_arr = torch.from_numpy(data['X'].astype(np.float32)).unsqueeze(1)
            self._use_mmap = False

        n = n_ok if n_ok is not None else len(data['y_macro'])
        self.y_macro = torch.from_numpy(data['y_macro'][:n].astype(np.int64))
        self.y_wear  = torch.from_numpy(data['y_wear'][:n].astype(np.float32))

    def __len__(self) -> int:
        return len(self.y_macro)

    def __getitem__(self, idx: int):
        if self._use_mmap:
            x = torch.from_numpy(self._X_mmap[idx].copy()).unsqueeze(0)
        else:
            x = self._X_arr[idx]
        return x, self.y_macro[idx], self.y_wear[idx]


def _merge_datasets(primary: TileDataset, secondary: TileDataset) -> TileDataset:
    """Merge secondary (small real-capture set) into primary by materialising both X arrays."""
    # primary is typically the large synthetic set (mmap); secondary is small real data.
    # We materialise both into a single in-memory tensor for simplicity — the real set
    # is expected to be small (~5K samples, ~80 MB).
    def _to_tensor(ds: TileDataset) -> torch.Tensor:
        if ds._use_mmap:
            return torch.from_numpy(np.array(ds._X_mmap)).unsqueeze(1)
        return ds._X_arr

    merged_X = torch.cat([_to_tensor(primary), _to_tensor(secondary)], dim=0)
    primary._X_arr   = merged_X
    primary._use_mmap = False
    primary.y_macro  = torch.cat([primary.y_macro, secondary.y_macro], dim=0)
    primary.y_wear   = torch.cat([primary.y_wear,  secondary.y_wear],  dim=0)
    return primary


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MacroCNN(nn.Module):
    """
    Shared CNN backbone + dual-head for macro classification and wear regression.
    Input: (B, 1, 64, 64) normalised float32 height fields.
    """

    def __init__(self, n_classes: int = 16):
        super().__init__()
        # Backbone - ~500K params as specified in PLAN
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 64->32
            nn.Conv2d(32, 64, 3, padding=1),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 32->16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 16->8
            nn.AdaptiveAvgPool2d(1),           # GlobalAvgPool -> (B, 128, 1, 1)
            nn.Flatten(),                      # (B, 128)
        )
        # Macro head
        self.macro_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )
        # Wear head (~100K params for the full wear estimator per PLAN)
        self.wear_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        return self.macro_head(feat), self.wear_head(feat).squeeze(1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: MacroCNN,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    macro_weight: float,
    wear_weight: float,
) -> tuple[float, float, float]:
    """Returns (total_loss, macro_acc, wear_mae)."""
    model.train()
    total_loss = macro_correct = macro_total = wear_sum = 0.0
    for X, y_macro, y_wear in loader:
        X, y_macro, y_wear = X.to(device), y_macro.to(device), y_wear.to(device)
        optimiser.zero_grad()
        logits, wear_pred = model(X)
        loss_macro = F.cross_entropy(logits, y_macro)
        loss_wear  = F.mse_loss(wear_pred, y_wear)
        loss = macro_weight * loss_macro + wear_weight * loss_wear
        loss.backward()
        optimiser.step()
        total_loss   += loss.item() * len(X)
        macro_correct += (logits.argmax(1) == y_macro).sum().item()
        macro_total   += len(X)
        wear_sum      += (wear_pred - y_wear).abs().sum().item()
    n = macro_total
    return total_loss / n, macro_correct / n, wear_sum / n


@torch.no_grad()
def evaluate(
    model: MacroCNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (macro_acc, wear_mae)."""
    model.eval()
    correct = total = 0
    wear_sum = 0.0
    for X, y_macro, y_wear in loader:
        X, y_macro, y_wear = X.to(device), y_macro.to(device), y_wear.to(device)
        logits, wear_pred = model(X)
        correct  += (logits.argmax(1) == y_macro).sum().item()
        total    += len(X)
        wear_sum += (wear_pred - y_wear).abs().sum().item()
    return correct / total, wear_sum / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data',    type=Path, required=True,
                    help='path to synthetic .npz from generate_cnn_data.py')
    ap.add_argument('--real',    type=Path, default=None,
                    help='optional real-capture .npz to mix in (same format)')
    ap.add_argument('--out',     type=Path, default=Path('models'),
                    help='output directory for saved models')
    ap.add_argument('--epochs',  type=int,   default=50)
    ap.add_argument('--batch',   type=int,   default=256)
    ap.add_argument('--lr',      type=float, default=1e-3)
    ap.add_argument('--val-frac', type=float, default=0.1,
                    help='fraction of data held out for validation')
    ap.add_argument('--macro-weight', type=float, default=1.0,
                    help='loss weight for macro classification term')
    ap.add_argument('--wear-weight',  type=float, default=0.5,
                    help='loss weight for wear regression term')
    ap.add_argument('--no-onnx', action='store_true',
                    help='skip ONNX export (useful if onnx is not installed)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f'Loading {args.data} ...')
    dataset = TileDataset(args.data)
    if args.real is not None:
        print(f'Merging real data from {args.real} ...')
        dataset = _merge_datasets(dataset, TileDataset(args.real))
    n_total = len(dataset)
    n_val   = max(1, int(n_total * args.val_frac))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    print(f'Train  : {n_train:,}  |  Val: {n_val:,}')

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = MacroCNN(n_classes=16).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model  : {n_params:,} parameters')
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=1e-5,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_acc = 0.0
    args.out.mkdir(parents=True, exist_ok=True)
    best_ckpt = args.out / 'macro_cnn_best.pt'

    print(f'\n{"Epoch":>5}  {"Loss":>8}  {"TrainAcc":>9}  {"ValAcc":>7}  '
          f'{"WearMAE":>8}  {"LR":>8}')
    print('-' * 60)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_mae = train_one_epoch(
            model, train_loader, optimiser, device,
            args.macro_weight, args.wear_weight,
        )
        val_acc, val_mae = evaluate(model, val_loader, device)
        lr = optimiser.param_groups[0]['lr']
        scheduler.step()

        print(f'{epoch:5d}  {tr_loss:8.4f}  {tr_acc*100:8.2f}%  '
              f'{val_acc*100:6.2f}%  {val_mae:8.4f}  {lr:8.2e}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)

    print(f'\nBest val macro accuracy: {best_val_acc*100:.2f}%')
    print(f'Saved best checkpoint  : {best_ckpt}')

    # Save final weights too
    torch.save(model.state_dict(), args.out / 'macro_cnn_final.pt')

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------
    if not args.no_onnx:
        try:
            import onnx  # noqa: F401 - just check it's available
            model.load_state_dict(torch.load(best_ckpt, map_location='cpu'))
            model.eval().cpu()
            dummy = torch.zeros(1, 1, 64, 64)
            onnx_path = args.out / 'macro_cnn.onnx'
            # dynamo=False uses the legacy TorchScript exporter (stable, cross-platform)
            torch.onnx.export(
                model, dummy, str(onnx_path),
                input_names=['height_field'],
                output_names=['macro_logits', 'wear_estimate'],
                dynamic_axes={'height_field': {0: 'batch'}},
                opset_version=18,
                dynamo=False,
            )
            print(f'ONNX    : {onnx_path}')
        except ImportError:
            print('onnx not installed - skipping ONNX export '
                  '(run: pip install onnx)')
        except Exception as e:
            print(f'ONNX export failed: {e}')

    # Print per-class accuracy breakdown on the validation set
    print('\nPer-class macro accuracy on val set:')
    model.eval().to(device)
    per_class = {m: [0, 0] for m in range(16)}  # [correct, total]
    with torch.no_grad():
        for X, y_macro, _ in val_loader:
            X, y_macro = X.to(device), y_macro.to(device)
            logits, _ = model(X)
            preds = logits.argmax(1)
            for true, pred in zip(y_macro.cpu().tolist(), preds.cpu().tolist()):
                per_class[true][1] += 1
                if true == pred:
                    per_class[true][0] += 1
    macros_names = [
        'circle','triangle','hexagon','square','star_5','star_6','star_8',
        'rosette','wave','pinwheel','diamond','octagon','petal','gear',
        'spiral','cross',
    ]
    for m in range(16):
        c, t = per_class[m]
        pct = c / t * 100 if t else 0
        print(f'  macro={m:2d} ({macros_names[m]:10s}): {pct:5.1f}%  ({c}/{t})')


if __name__ == '__main__':
    main()
