"""Skrip 02: Melatih dan mengevaluasi model Anti-Spoofing (LBP Multi-Kanal + SVM)."""

import json
import os
import sys
import cv2
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from modules.face_engine import FaceEngine, resize_max_side
from modules.spoofing_lbp import AntiSpoofChecker, extract_dual_scale_features

SPOOF_REAL_DIR = os.path.join(BASE_DIR, "data", "spoof_dataset", "real")
SPOOF_FAKE_DIR = os.path.join(BASE_DIR, "data", "spoof_dataset", "spoof")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
EXP_DIR = os.path.join(BASE_DIR, "artifacts", "experiments")
os.makedirs(EXP_DIR, exist_ok=True)


def load_folder_samples(folder_path, label, engine):
    """Membaca citra dari folder, mengekstrak ROI wajah & konteks 2.4x, lalu menghitung fitur Dual-Scale."""
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_ext)])
    features = []
    labels = []
    meta = []

    for fname in files:
        fpath = os.path.join(folder_path, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue

        resized, _ = resize_max_side(img, max_side=1280)
        faces = engine.detect(resized, score_threshold=0.55)
        if faces:
            bbox = faces[0]["bbox"]
        else:
            h_r, w_r = resized.shape[:2]
            bbox = (0, 0, w_r, h_r)

        vec, _, _ = extract_dual_scale_features(resized, bbox)
        features.append(vec)
        labels.append(label)
        meta.append((fpath, bbox))

    return features, labels, meta


def main():
    yunet_p = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
    sface_p = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
    engine = FaceEngine(yunet_p, sface_p, score_threshold=0.55)

    real_feats, real_lbls, real_meta = load_folder_samples(SPOOF_REAL_DIR, 1, engine)
    fake_feats, fake_lbls, fake_meta = load_folder_samples(SPOOF_FAKE_DIR, 0, engine)

    print(f"Sampel Real  ditemukan: {len(real_feats)} citra ({SPOOF_REAL_DIR})")
    print(f"Sampel Spoof ditemukan: {len(fake_feats)} citra ({SPOOF_FAKE_DIR})")

    if len(real_feats) < 10 or len(fake_feats) < 10:
        print(
            "\n[INFO] Dataset Anti-Spoofing belum mencukupi (minimal 10 citra 'real' dan 10 citra 'spoof').\n"
            "Silakan isi folder data/spoof_dataset/real/ dan data/spoof_dataset/spoof/ dengan:\n"
            "  1) Subset dataset LCC-FASD / NUAA, dan/atau\n"
            "  2) 30-50 foto wajah asli vs foto wajah dari layar HP Anda sendiri.\n"
            "Lalu jalankan kembali skrip 02_train_antispoofing.py ini."
        )
        return

    X = np.vstack(real_feats + fake_feats)
    y = np.array(real_lbls + fake_lbls, dtype=np.int32)
    all_meta = real_meta + fake_meta
    idx_all = np.arange(len(y))

    idx_train, idx_test, y_train, y_test = train_test_split(
        idx_all, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_test = X[idx_train], X[idx_test]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base_svm = SVC(kernel="rbf", C=3.0, gamma="scale", random_state=42)
    svm = CalibratedClassifierCV(base_svm, cv=5, ensemble=False)
    svm.fit(X_train_s, y_train)

    svm_p = os.path.join(MODELS_DIR, "anti_spoof_svm.joblib")
    scaler_p = os.path.join(MODELS_DIR, "scaler_lbp.joblib")
    joblib.dump(svm, svm_p)
    joblib.dump(scaler, scaler_p)

    y_pred_svm = svm.predict(X_test_s)
    acc_svm = accuracy_score(y_test, y_pred_svm)

    # Evaluasi Hybrid Final (MiniFASNet 2.7x+4.0x ONNX + Dual-Scale LBP/FFT SVM) pada 20% Test Split
    fas27_p = os.path.join(MODELS_DIR, "2.7_80x80_MiniFASNetV2.onnx")
    fas40_p = os.path.join(MODELS_DIR, "4_0_0_80x80_MiniFASNetV1SE.onnx")
    checker = AntiSpoofChecker(svm_p, scaler_p, prob_threshold=0.50, fas_v2_path=fas27_p, fas_v1se_path=fas40_p)

    y_prob_list = []
    for i_te in idx_test:
        fpath, bbox = all_meta[int(i_te)]
        im = cv2.imread(fpath)
        resized, _ = resize_max_side(im, max_side=1280)
        x, y0, w, h = bbox
        _, pr, _, _ = checker.predict(resized[y0:y0 + h, x:x + w], full_bgr=resized, bbox=bbox)
        y_prob_list.append(pr)

    y_prob = np.array(y_prob_list, dtype=np.float32)
    y_pred = (y_prob >= 0.50).astype(np.int32)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"\n=== Hasil Evaluasi Uji Anti-Spoofing ({checker.mode}, Test 20%) ===")
    print(f"SVM-Only Acc : {acc_svm * 100:.2f}%")
    print(f"Hybrid Acc   : {acc * 100:.2f}%")
    print(f"Precision    : {prec * 100:.2f}%")
    print(f"Recall       : {rec * 100:.2f}%")
    print(f"F1-Score     : {f1 * 100:.2f}%")
    print(f"ROC-AUC      : {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Spoof", "Real"]))

    # Simpan grafik Confusion Matrix 300 DPI
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=300)
    im_cm = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im_cm, ax=ax, fraction=0.046)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Prediksi: Spoof", "Prediksi: Real"], fontsize=9)
    ax.set_yticklabels(["Aktual: Spoof", "Aktual: Real"], fontsize=9)
    for r in range(2):
        for c in range(2):
            val = cm[r, c]
            col = "white" if val > cm.max() * 0.5 else "black"
            ax.text(c, r, str(val), ha="center", va="center", color=col, fontsize=13, fontweight="bold")
    ax.set_title(f"Confusion Matrix Anti-Spoofing Hybrid\nAkurasi Uji: {acc * 100:.1f}% | AUC: {auc:.3f}", fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR, "cm_antispoofing.png"), dpi=300)
    plt.close(fig)

    metrics = {
        "mode": checker.mode,
        "n_real": int(len(real_feats)),
        "n_spoof": int(len(fake_feats)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "svm_only_accuracy": round(float(acc_svm), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
    }
    with open(os.path.join(EXP_DIR, "antispoof_eval.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Model tersimpan di {MODELS_DIR} dan evaluasi di {EXP_DIR}")


if __name__ == "__main__":
    main()
