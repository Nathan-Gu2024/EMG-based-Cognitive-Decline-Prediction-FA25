"""
train_fog_cnn.py

Clean LOSO CNN training with:
- Focal Loss for class imbalance
- Proper train/val/test split (dedicated val subject)
- Early stopping on val F1
- Mixed precision (GPU)
- No data leakage
- Labels already 0-indexed (0=NonFoG, 1=FoG)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# ── 1. Updated 3-Class Focal Loss ──
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()

# ── 2. Your CNN + TCN Architecture ──
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.drop(self.relu(self.bn(self.conv(x))))
        return self.relu(out + self.skip(x))

class TCNBlock(nn.Module):
    def __init__(self, channels, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        # Note: Symmetric padding (Non-causal)
        pad = (kernel - 1) * dilation // 2   
        self.conv1 = weight_norm(nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=pad))
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=pad))
        self.bn2   = nn.BatchNorm1d(channels)
        self.relu  = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        out = self.drop1(self.relu(self.bn1(self.conv1(x))))
        out = self.drop2(self.relu(self.bn2(self.conv2(out))))
        return self.relu(out + x)

class FoGCNNTCN(nn.Module):
    def __init__(self, num_classes=3, dropout_cnn=0.3, dropout_tcn=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv_blocks = nn.Sequential(
            ConvBlock(64,  64,  kernel=7, dropout=dropout_cnn),
            nn.MaxPool1d(2),
            ConvBlock(64,  128, kernel=5, dropout=dropout_cnn),
            nn.MaxPool1d(2),
            ConvBlock(128, 128, kernel=3, dropout=dropout_cnn),
        )
        self.tcn_blocks = nn.Sequential(
            TCNBlock(128, kernel=3, dilation=1,  dropout=dropout_tcn),
            TCNBlock(128, kernel=3, dilation=2,  dropout=dropout_tcn),
            TCNBlock(128, kernel=3, dilation=4,  dropout=dropout_tcn),
            TCNBlock(128, kernel=3, dilation=8,  dropout=dropout_tcn),
        )
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.conv_blocks(x)
        x = self.tcn_blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.head(x)


# ── Training & Evaluation ─────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with torch.amp.autocast('cuda'):
                out  = model(X_batch)
                loss = criterion(out, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        preds  = out.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total   += y_batch.size(0)

    return total_loss / len(loader), 100 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        preds   = model(X_batch).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.numpy())

    return np.array(all_preds), np.array(all_labels)


def metrics(y_true, y_pred):
    # Use labels [0, 1, 2] for 3 classes
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    # Calculate Macro F1 for early stopping
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    return dict(
        f1=f1_macro, 
        accuracy=acc, 
        confusion_matrix=cm,
        y_true=y_true, # Pass through for the final overall report
        y_pred=y_pred
    )
# ── LOSO Cross-Validation ─────────────────────────────────────────────────────

def loso_cv(X, y, subject_indices,
            num_classes=2, epochs=80, batch_size=256,
            lr=1e-3, device='cuda', patience=12):
    """
    Leave-One-Subject-Out CV with proper val split.
    Val set = the subject just before the test subject in the list
    (rotates so every subject gets to be val once).
    """
    # Transpose once here: (N, 384, 6) → (N, 6, 384)
    X_t = torch.tensor(X, dtype=torch.float32).transpose(1, 2)
    y_t = torch.tensor(y, dtype=torch.long)

    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    
    all_preds, all_labels = [], []
    results = {}

    n = len(subject_indices)

    for fold_idx, test_info in enumerate(subject_indices):
        test_subj = test_info["subject_id"]

        # Use the subject before test as validation (wrap around)
        val_info = subject_indices[(fold_idx - 1) % n]
        val_subj = val_info["subject_id"]

        train_infos = [s for s in subject_indices
                       if s["subject_id"] not in (test_subj, val_subj)]

        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx+1}/{n} | Test={test_subj} | Val={val_subj}")
        print(f"{'='*70}")

        test_idx  = list(range(test_info["start_idx"], test_info["end_idx"]))
        val_idx   = list(range(val_info["start_idx"],  val_info["end_idx"]))
        train_idx = []
        for s in train_infos:
            train_idx.extend(range(s["start_idx"], s["end_idx"]))

        # Report class distribution
        train_labels = y[train_idx]
        fog_n    = (train_labels == 1).sum()
        nonfog_n = (train_labels == 0).sum()
        print(f"  Train: {len(train_idx)} windows ({fog_n} FoG, {nonfog_n} NonFoG)")
        print(f"  Val  : {len(val_idx)} windows")
        print(f"  Test : {len(test_idx)} windows")

        nw = min(4, os.cpu_count() or 1)
        pin = device == 'cuda'

        train_loader = DataLoader(TensorDataset(X_t[train_idx], y_t[train_idx]),
                                  batch_size=batch_size, shuffle=True,
                                  num_workers=nw, pin_memory=pin, persistent_workers=nw>0)
        val_loader   = DataLoader(TensorDataset(X_t[val_idx],   y_t[val_idx]),
                                  batch_size=batch_size*2, shuffle=False,
                                  num_workers=nw, pin_memory=pin, persistent_workers=nw>0)
        test_loader  = DataLoader(TensorDataset(X_t[test_idx],  y_t[test_idx]),
                                  batch_size=batch_size*2, shuffle=False,
                                  num_workers=nw, pin_memory=pin, persistent_workers=nw>0)



        model = FoGCNNTCN(num_classes=3).to(device)
        
        # Calculate dynamic weights for the 3 classes
        class_counts = np.bincount(train_labels)
        weights = 1.0 / class_counts
        weights = weights / weights.sum() # Normalize
        
        # Initialize loss and move it to the GPU
        criterion = FocalLoss(alpha=weights, gamma=2.0).to(device)
        
        # Setup Optimizer and Scheduler
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-5)
        best_f1, best_state, wait = 0.0, None, 0

        for epoch in range(epochs):
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
            val_preds, val_lbls = evaluate(model, val_loader, device)
            vm = metrics(val_lbls, val_preds)
            val_f1 = vm.get('f1', vm.get('macro_f1', 0))

            scheduler.step(val_f1)

            if val_f1 > best_f1:
                best_f1, wait = val_f1, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1

            if (epoch + 1) % 10 == 0:
                # print(f"\n  → Test {test_subj}: "
                #     f"Macro F1={m['f1']*100:.1f}% "
                #     f"Acc={m['accuracy']*100:.1f}%")
                print(f"  Epoch {epoch+1:02d}/{epochs} | Tr Loss: {tr_loss:.4f} | Tr Acc: {tr_acc:.1f}% | Val F1: {val_f1*100:.1f}%")
            if wait >= patience:
                print(f"  Early stop at epoch {epoch+1}")                
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        y_pred, y_true = evaluate(model, test_loader, device)
        m = metrics(y_true, y_pred)
        results[test_subj] = m
        all_preds.extend(y_pred)
        all_labels.extend(y_true)

        print(f"\n  → Test {test_subj}: "
              f"Macro F1={m['f1']*100:.1f}% "
              f"Acc={m['accuracy']*100:.1f}%")
        
    overall = metrics(np.array(all_labels), np.array(all_preds))
    results['overall'] = overall
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='.', help="Folder containing the .npy files")
    args = parser.parse_args()

    DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}")

    # Use args.data_path here
    X = np.load(os.path.join(args.data_path, "X_windows_all_subjects.npy"))
    y = np.load(os.path.join(args.data_path, "y_windows_all_subjects.npy"))

    with open(os.path.join(args.data_path, "subject_indices.json")) as f:
        subject_indices = json.load(f)
    
    print(f"X: {X.shape}  y: {y.shape}")
    print(f"FoG (1): {(y==1).sum()} | NonFoG (0): {(y==0).sum()}")
    print(f"Subjects: {[s['subject_id'] for s in subject_indices]}")

    # Safety check — labels should already be 0/1
    assert set(np.unique(y)).issubset({0, 1, 2}), f"Unexpected labels: {np.unique(y)}"

    results = loso_cv(
        X, y,
        subject_indices=subject_indices,
        num_classes=3,
        epochs=80,
        batch_size=256,
        lr=1e-3,
        device=DEVICE,
        patience=12
    )

    print("\n" + "="*70)
    print("OVERALL 3-CLASS LOSO RESULTS")
    print("="*70)
    
    ov = results['overall']
    all_y_true = ov['y_true']
    all_y_pred = ov['y_pred']
    
    # Print the scikit-learn classification report
    print(classification_report(all_y_true, all_y_pred, 
                                labels=[0, 1, 2],
                                target_names=["Non-FoG", "Pre-FoG", "FoG"], 
                                digits=4, zero_division=0))
    # Confusion matrix (3x3)
    plt.figure(figsize=(7, 6))
    sns.heatmap(ov['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=["Non-FoG", "Pre-FoG", "FoG"], 
                yticklabels=["Non-FoG", "Pre-FoG", "FoG"])
    plt.title("LOSO Overall 3-Class Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    # Update the save path for the image
    cm_save_path = os.path.join(args.data_path, "loso_confusion_matrix_3class.png")
    plt.savefig(cm_save_path, dpi=300)
    plt.show()

    # Save final dictionary locally (exclude massive raw arrays to save space)
    save_data = {}
    for k, v in results.items():
        save_data[k] = {key: val.tolist() if isinstance(val, np.ndarray) else val 
                        for key, val in v.items() if key not in ['y_true', 'y_pred']}

    json_save_path = os.path.join(args.data_path, "loso_results_3class.json")
    with open(json_save_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nSuccessfully saved metrics to:")
    print(f"  - {cm_save_path}")
    print(f"  - {json_save_path}")