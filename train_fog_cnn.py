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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss — focuses training on hard examples.
    Better than class-weighted CE for severe imbalance.
    alpha: down-weight easy negatives (tune 0.25–0.75)
    gamma: focusing parameter (2.0 is standard)
    """
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce   = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt   = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


# ── CNN Architecture ──────────────────────────────────────────────────────────

class FoGCNN(nn.Module):
    """
    1D CNN for FoG detection.
    Input: (batch, 6, 384)
    Output: (batch, num_classes)
    """
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()

        self.conv1   = nn.Conv1d(6,   64, kernel_size=7, padding=3)
        self.bn1     = nn.BatchNorm1d(64)
        self.pool1   = nn.MaxPool1d(2)
        self.drop1   = nn.Dropout(dropout * 0.5)

        self.conv2   = nn.Conv1d(64,  128, kernel_size=5, padding=2)
        self.bn2     = nn.BatchNorm1d(128)
        self.pool2   = nn.MaxPool1d(2)
        self.drop2   = nn.Dropout(dropout * 0.75)

        self.conv3   = nn.Conv1d(128,  64, kernel_size=3, padding=1)
        self.bn3     = nn.BatchNorm1d(64)
        self.pool3   = nn.MaxPool1d(2)
        self.drop3   = nn.Dropout(dropout)

        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.relu    = nn.ReLU()

        self.fc      = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.drop1(self.pool1(self.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(self.relu(self.bn3(self.conv3(x)))))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


# ── Training & Evaluation ─────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with torch.cuda.amp.autocast():
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
    cm = confusion_matrix(y_true, y_pred)

    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1   = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
        acc  = (tp + tn) / len(y_true)
        return dict(sensitivity=sens, specificity=spec, precision=prec,
                    f1=f1, accuracy=acc, confusion_matrix=cm,
                    tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))
    else:
        rep = classification_report(y_true, y_pred, output_dict=True)
        return dict(accuracy=rep["accuracy"],
                    macro_f1=rep["macro avg"]["f1-score"],
                    confusion_matrix=cm)


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

    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

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

        model     = FoGCNN(num_classes=num_classes).to(device)
        criterion = FocalLoss(alpha=0.5, gamma=2.0)
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
                print(f"  Ep {epoch+1:3d} | "
                      f"Loss={tr_loss:.4f} Acc={tr_acc:.1f}% | "
                      f"Val F1={val_f1*100:.1f}% "
                      f"Sens={vm.get('sensitivity',0)*100:.1f}% "
                      f"Spec={vm.get('specificity',0)*100:.1f}%")

            if wait >= patience:
                print(f"  Early stop at epoch {epoch+1} (best val F1={best_f1*100:.1f}%)")
                break

        if best_state:
            model.load_state_dict(best_state)

        y_pred, y_true = evaluate(model, test_loader, device)
        m = metrics(y_true, y_pred)
        results[test_subj] = m
        all_preds.extend(y_pred)
        all_labels.extend(y_true)

        print(f"\n  → Test {test_subj}: "
              f"Sens={m['sensitivity']*100:.1f}% "
              f"Spec={m['specificity']*100:.1f}% "
              f"F1={m['f1']*100:.1f}% "
              f"Acc={m['accuracy']*100:.1f}%")

    overall = metrics(np.array(all_labels), np.array(all_preds))
    results['overall'] = overall
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}")

    X = np.load("X_windows_all_subjects.npy")
    y = np.load("y_windows_all_subjects.npy")

    with open("subject_indices.json") as f:
        subject_indices = json.load(f)

    print(f"X: {X.shape}  y: {y.shape}")
    print(f"FoG (1): {(y==1).sum()} | NonFoG (0): {(y==0).sum()}")
    print(f"Subjects: {[s['subject_id'] for s in subject_indices]}")

    # Safety check — labels should already be 0/1
    assert set(np.unique(y)).issubset({0, 1}), f"Unexpected labels: {np.unique(y)}"

    results = loso_cv(
        X, y,
        subject_indices=subject_indices,
        num_classes=2,
        epochs=80,
        batch_size=256,
        lr=1e-3,
        device=DEVICE,
        patience=12
    )

    print("\n" + "="*70)
    print("OVERALL LOSO RESULTS")
    print("="*70)
    ov = results['overall']
    print(f"Sensitivity : {ov['sensitivity']*100:.2f}%")
    print(f"Specificity : {ov['specificity']*100:.2f}%")
    print(f"Precision   : {ov['precision']*100:.2f}%")
    print(f"F1          : {ov['f1']*100:.2f}%")
    print(f"Accuracy    : {ov['accuracy']*100:.2f}%")
    print(f"TP={ov['tp']}  FP={ov['fp']}  FN={ov['fn']}  TN={ov['tn']}")

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(ov['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=["NonFoG", "FoG"], yticklabels=["NonFoG", "FoG"])
    plt.title("LOSO Overall Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("loso_confusion_matrix.png", dpi=300)
    plt.show()

    # Save results
    save = {}
    for k, v in results.items():
        save[k] = {key: val.tolist() if isinstance(val, np.ndarray) else val
                   for key, val in v.items()}
    with open("loso_results.json", "w") as f:
        json.dump(save, f, indent=2)
    print("Saved loso_results.json and loso_confusion_matrix.png")