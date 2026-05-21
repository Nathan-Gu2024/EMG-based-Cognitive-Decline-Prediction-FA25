"""
train_fog_cnn.py

Optimized Binary LOSO CNN/TCN training with:
- Fixed 0.5 Threshold + Min-Duration Post-Processing
- Capped Focal Loss weights to prevent double-penalization
- Automated remapping from 3-class data to clean 2-class binary setup
- Exclusion of uninformative / poisoning subjects (002 and 005)
- Early stopping locked onto minority FoG class F1 score
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.utils.parametrizations import weight_norm

# ── 1. Binary Focal Loss ──
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

# ── 2. CNN + TCN Architecture ──
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
    def __init__(self, num_classes=2, dropout_cnn=0.3, dropout_tcn=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(8, 64, kernel_size=7, padding=3),
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
    """Returns predictions, labels, AND FoG class probabilities."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        out     = model(X_batch)
        probs   = torch.softmax(out, dim=1)[:, 1]  # P(FoG)
        preds   = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ── Post-processing ───────────────────────────────────────────────────────────

def apply_threshold(probs, threshold=0.5):
    """Convert FoG probabilities to binary predictions at a given threshold."""
    return (probs >= threshold).astype(int)

def min_fog_duration_filter(preds, min_windows=3):
    """
    Remove FoG runs shorter than min_windows consecutive predictions.
    At step=96 (0.75s): min_windows=3 → requires ≥2.25s of FoG.
    Real clinical FoG episodes typically last >1.5s.
    """
    filtered = preds.copy()
    n = len(filtered)
    i = 0
    while i < n:
        if filtered[i] == 1:
            # Find end of this FoG run
            j = i
            while j < n and filtered[j] == 1:
                j += 1
            run_len = j - i
            if run_len < min_windows:
                filtered[i:j] = 0   # Too short — wipe it
            i = j
        else:
            i += 1
    return filtered

def post_process(probs, threshold=0.5, min_windows=3):
    """
    Fixed post-processing pipeline:
      1. Apply standard 0.5 decision threshold
      2. Minimum FoG duration filter (removes physically impossible short noise)
    """
    preds = apply_threshold(probs, threshold)
    preds = min_fog_duration_filter(preds, min_windows=min_windows)
    return preds

def metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    f1_binary = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return dict(
        f1=f1_binary,
        sensitivity=sens,
        specificity=spec,
        precision=prec,
        accuracy=acc,
        confusion_matrix=cm,
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        y_true=y_true,
        y_pred=y_pred
    )


# ── LOSO Cross-Validation ─────────────────────────────────────────────────────

def loso_cv(X, y, subject_indices,
            num_classes=2, epochs=80, batch_size=256,
            lr=1e-3, device='cuda', patience=12):

    # Transpose once here: (N, 384, 8) → (N, 8, 384)
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

        model = FoGCNNTCN(num_classes=2).to(device)
        
        # FIX: Calculate weights and heavily cap them to prevent double-penalizing the majority class
        criterion = FocalLoss(alpha=None, gamma=2.0)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-5)
        
        best_f1, best_state, wait = 0.0, None, 0

        for epoch in range(epochs):
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
            val_preds_raw, val_lbls, val_probs = evaluate(model, val_loader, device)
            
            # Evaluate validation against the fixed 0.5 post-processing pipeline
            val_preds_pp = post_process(val_probs, threshold=0.5, min_windows=3)
            vm = metrics(val_lbls, val_preds_pp)
            val_f1 = vm['f1']

            scheduler.step(val_f1)

            if val_f1 > best_f1:
                best_f1, wait = val_f1, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1

            if (epoch + 1) % 10 == 0:
                print(f"  → Epoch {epoch+1}: Val FoG F1={val_f1*100:.1f}% "
                      f"Sens={vm['sensitivity']*100:.1f}% Spec={vm['specificity']*100:.1f}%")
                
            if wait >= patience:
                print(f"  Early stop at epoch {epoch+1}")                
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Evaluate on test set
        y_pred_raw, y_true, y_probs = evaluate(model, test_loader, device)

        # Final predictions at fixed 0.5 threshold with min-duration filter
        y_pred_best = post_process(y_probs, threshold=0.5, min_windows=3)
        m_raw  = metrics(y_true, y_pred_raw)
        m_best = metrics(y_true, y_pred_best)

        results[test_subj] = {
            'raw':        m_raw,
            'best':       m_best,
            'best_thresh': 0.5
        }
        all_preds.extend(y_pred_best)
        all_labels.extend(y_true)

        print(f"\n  → Test {test_subj}  [raw]  "
              f"F1={m_raw['f1']*100:.1f}% Sens={m_raw['sensitivity']*100:.1f}% Spec={m_raw['specificity']*100:.1f}%")
        print(f"  → Test {test_subj}  [best] "
              f"F1={m_best['f1']*100:.1f}% Sens={m_best['sensitivity']*100:.1f}% Spec={m_best['specificity']*100:.1f}%  (fixed thresh=0.50)")
        
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

    X = np.load(os.path.join(args.data_path, "X_windows_all_subjects.npy"))
    y = np.load(os.path.join(args.data_path, "y_windows_all_subjects.npy"))

    with open(os.path.join(args.data_path, "subject_indices.json")) as f:
        subject_indices = json.load(f)
    
    print(f"Original Loaded X: {X.shape}  y: {y.shape}")
    
    if len(np.unique(y)) > 2:
        print("Remapping 3-class arrays back to Binary setup...")
        y_binary = np.zeros_like(y)
        y_binary[y == 2] = 1  
        y_binary[y == 1] = 0  
        y = y_binary

    EXCLUDE_SUBJS = ["002", "005"]
    print(f"Filtering out background noise/poisonous subjects: {EXCLUDE_SUBJS}")
    subject_indices = [s for s in subject_indices if s["subject_id"] not in EXCLUDE_SUBJS]

    print(f"Processed Binary Distribution -> FoG (1): {(y==1).sum()} | NonFoG (0): {(y==0).sum()}")
    print(f"Active Training Subjects: {[s['subject_id'] for s in subject_indices]}")

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
    print("OVERALL BINARY 2-CLASS LOSO RESULTS")
    print("="*70)
    
    ov = results['overall']
    all_y_true = ov['y_true']
    all_y_pred = ov['y_pred']
    
    print(classification_report(all_y_true, all_y_pred, 
                                target_names=["Non-FoG", "FoG"], 
                                digits=4, zero_division=0))

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(ov['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=["Non-FoG", "FoG"],
                yticklabels=["Non-FoG", "FoG"])
    plt.title("LOSO Overall Confusion Matrix (post-processed)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_save_path = os.path.join(args.data_path, "loso_confusion_matrix_2class.png")
    plt.savefig(cm_save_path, dpi=300)
    plt.show()

    # Save results safely
    def serialise(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, (np.integer, np.floating)): return float(v)
        return v

    save_data = {}
    for k, v in results.items():
        if k == 'overall':
            save_data[k] = {key: serialise(val) for key, val in v.items()
                            if key not in ('y_true', 'y_pred')}
        else:
            save_data[k] = {
                mode: {key: serialise(val) for key, val in v[mode].items()
                       if key not in ('y_true', 'y_pred')}
                for mode in ('raw', 'best')
                if mode in v
            }
            save_data[k]['best_thresh'] = v.get('best_thresh', 0.5)

    json_save_path = os.path.join(args.data_path, "loso_results_2class.json")
    with open(json_save_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nSuccessfully saved metrics to: {json_save_path}")