import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load your precomputed X and y here
# X: (K,384,6)
# y: (K,)   labels {1=FoG, 2=NonFoG}

# Convert labels to binary
X = np.load("X_windows.npy")
y = np.load("y_windows.npy")

print("Loaded X:", X.shape)
print("Loaded y:", y.shape)


# Convert labels → binary (1 = FoG)
y_bin = (y == 1).astype(int)

print("Class balance:", np.unique(y_bin, return_counts=True))


def extract_features(X):
    feats = []
    for w in X:
        f = []
        f += w.mean(axis=0).tolist()
        f += w.std(axis=0).tolist()
        f += np.max(w, axis=0).tolist()
        f += np.min(w, axis=0).tolist()
        feats.append(f)
    return np.array(feats)

X_feat = extract_features(X)

# Stratified split
Xtr, Xte, ytr, yte = train_test_split(
    X_feat, y_bin, test_size=0.25, stratify=y_bin, random_state=0
)

# Handle imbalance
pos = ytr.sum()
neg = len(ytr) - pos
scale = neg / max(pos, 1)

print("scale_pos_weight =", scale)

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale,
    eval_metric="logloss",
    tree_method="hist"
)

model.fit(Xtr, ytr)

yp = model.predict(Xte)
yp_prob = model.predict_proba(Xte)[:,1]

print("Confusion matrix:")
print(confusion_matrix(yte, yp))

print("\nClassification report:")
print(classification_report(yte, yp, digits=4))

print("ROC AUC:", roc_auc_score(yte, yp_prob))
