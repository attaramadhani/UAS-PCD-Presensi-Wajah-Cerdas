"""Skrip 03: Menjalankan eksperimen kuantitatif (LFW Identity & Adaptive Low-Light CLAHE) untuk laporan & Tab Evaluasi."""

import json
import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from modules.enhancement import apply_adaptive_enhancement, apply_gamma
from modules.face_engine import COSINE_THRESHOLD_DEFAULT, FaceEngine, resize_max_side

MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
LFW_DIR = os.path.join(BASE_DIR, "data", "raw_lfw")
WIDER_DIR = os.path.join(BASE_DIR, "data", "wider_lowlight")
EXP_DIR = os.path.join(BASE_DIR, "artifacts", "experiments")
os.makedirs(EXP_DIR, exist_ok=True)


def run_lfw_recognition_experiment(engine, lfw_data, n_pairs_each=150):
    """Menguji akurasi pengenalan SFace Cosine Similarity pada pasangan wajah LFW resmi."""
    print("\n=== Eksperimen 1: Validasi Pengenalan Wajah pada LFW ===")
    rng = np.random.default_rng(42)
    imgs = lfw_data.images
    targets = lfw_data.target

    # Konversi citra LFW RGB float/uint8 ke BGR uint8
    def to_bgr_uint8(arr):
        if arr.max() <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # Ekstrak embedding ter-cache untuk individu dengan >= 2 foto
    person_embs = {}
    detect_ok = 0
    total_checked = 0
    latencies = []

    unique_ids = np.unique(targets)
    for pid in unique_ids:
        idxs = np.where(targets == pid)[0][:6]  # maks 6 sampel per orang agar cepat
        embs = []
        for idx in idxs:
            total_checked += 1
            bgr = to_bgr_uint8(imgs[idx])
            # Perbesar sedikit (125x94 -> 250x188) agar landmark YuNet stabil
            bgr_up = cv2.resize(bgr, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            t0 = time.perf_counter()
            faces = engine.detect(bgr_up, score_threshold=0.6)
            if faces:
                detect_ok += 1
                _, emb = engine.align_and_embed(bgr_up, faces[0]["raw_row"])
                latencies.append((time.perf_counter() - t0) * 1000.0)
                embs.append(emb)
        if len(embs) >= 2:
            person_embs[int(pid)] = embs

    pids = list(person_embs.keys())
    same_scores, diff_scores = [], []

    # Pasangan identik (Positive pairs)
    for _ in range(n_pairs_each):
        pid = int(rng.choice(pids))
        e_list = person_embs[pid]
        i1, i2 = rng.choice(len(e_list), size=2, replace=False)
        same_scores.append(engine.cosine_similarity(e_list[i1], e_list[i2]))

    # Pasangan berbeda (Negative pairs)
    for _ in range(n_pairs_each):
        p1, p2 = rng.choice(pids, size=2, replace=False)
        e1 = person_embs[int(p1)][int(rng.integers(0, len(person_embs[int(p1)])))]
        e2 = person_embs[int(p2)][int(rng.integers(0, len(person_embs[int(p2)])))]
        diff_scores.append(engine.cosine_similarity(e1, e2))

    y_true = np.array([1] * len(same_scores) + [0] * len(diff_scores))
    all_scores = np.array(same_scores + diff_scores)
    y_pred = (all_scores >= COSINE_THRESHOLD_DEFAULT).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mean_ms = float(np.mean(latencies)) if latencies else 0.0

    # Grafik distribusi skor Cosine Similarity Same vs Different
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    bins = np.linspace(-0.2, 1.0, 32)
    ax.hist(diff_scores, bins=bins, alpha=0.65, color="#D62728", label="Pasangan Berbeda (Negatif)", density=True)
    ax.hist(same_scores, bins=bins, alpha=0.65, color="#2CA02C", label="Pasangan Identik (Positif)", density=True)
    ax.axvline(COSINE_THRESHOLD_DEFAULT, color="#111111", linestyle="--", linewidth=2,
               label=f"Ambang OpenCV SFace ({COSINE_THRESHOLD_DEFAULT})")
    ax.set_xlabel("Skor Kemiripan Cosine Similarity", fontsize=10)
    ax.set_ylabel("Kepadatan Distribusi", fontsize=10)
    ax.set_title(f"Verifikasi Wajah SFace pada LFW ({len(y_true)} Pasangan Uji)\nAkurasi: {acc*100:.2f}% | F1: {f1*100:.2f}% | Rerata Waktu: {mean_ms:.1f} ms/wajah", fontsize=10.5, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8.5, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR, "grafik_lfw_cosine.png"), dpi=300)
    plt.close(fig)

    res = {
        "n_individuals": int(len(pids)),
        "n_pairs_total": int(len(y_true)),
        "yunet_detection_rate": round(detect_ok / max(1, total_checked), 4),
        "cosine_threshold": COSINE_THRESHOLD_DEFAULT,
        "mean_same_score": round(float(np.mean(same_scores)), 4),
        "mean_diff_score": round(float(np.mean(diff_scores)), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "avg_latency_ms": round(mean_ms, 2),
    }
    print(json.dumps(res, indent=2))
    return res


def run_lowlight_clahe_experiment(engine, lfw_data):
    """Menguji efektivitas Adaptive CLAHE + Gamma pada citra redup/gelap terkontrol & native low-light."""
    print("\n=== Eksperimen 2: Pengaruh Adaptive Low-Light CLAHE + Gamma ===")
    imgs = lfw_data.images
    n_sample = min(60, len(imgs))

    gamma_levels = [1.0, 2.5, 3.5, 4.5]  # 1.0 = Normal; >=2.5 = Simulasi kondisi ruang gelap/redup
    rows = []

    for g_sim in gamma_levels:
        det_raw, det_enh = 0, 0
        conf_raw, conf_enh = [], []

        for idx in range(n_sample):
            arr = imgs[idx]
            if arr.max() <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
            bgr_up = cv2.resize(bgr, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

            # Terapkan penggelapan terkontrol untuk simulasi low-light
            dark_bgr = apply_gamma(bgr_up, gamma=g_sim) if g_sim != 1.0 else bgr_up
            enh_bgr, _ = apply_adaptive_enhancement(dark_bgr)

            f_raw = engine.detect(dark_bgr, score_threshold=0.65)
            f_enh = engine.detect(enh_bgr, score_threshold=0.65)

            if f_raw:
                det_raw += 1
                conf_raw.append(f_raw[0]["score"])
            if f_enh:
                det_enh += 1
                conf_enh.append(f_enh[0]["score"])

        label_kondisi = "Normal (Gamma 1.0)" if g_sim == 1.0 else f"Gelap Terkontrol (Gamma {g_sim})"
        rows.append({
            "Kondisi Pencahayaan": label_kondisi,
            "Jumlah Uji": n_sample,
            "Deteksi Tanpa CLAHE (%)": round(100.0 * det_raw / n_sample, 2),
            "Deteksi Dengan Adaptive CLAHE (%)": round(100.0 * det_enh / n_sample, 2),
            "Skor Rerata Raw": round(float(np.mean(conf_raw)) if conf_raw else 0.0, 4),
            "Skor Rerata Enhanced": round(float(np.mean(conf_enh)) if conf_enh else 0.0, 4),
        })

    # Uji tambahan bila ada foto gelap asli di data/wider_lowlight/
    valid_ext = (".jpg", ".jpeg", ".png")
    native_files = [f for f in os.listdir(WIDER_DIR) if f.lower().endswith(valid_ext)]
    native_summary = None
    if native_files:
        tot_raw, tot_enh = 0, 0
        for fn in native_files:
            im = cv2.imread(os.path.join(WIDER_DIR, fn))
            if im is None:
                continue
            im_r, _ = resize_max_side(im, 1280)
            enh, _ = apply_adaptive_enhancement(im_r)
            tot_raw += len(engine.detect(im_r, score_threshold=0.65))
            tot_enh += len(engine.detect(enh, score_threshold=0.65))
        native_summary = {"files": len(native_files), "faces_raw": tot_raw, "faces_enhanced": tot_enh}

    df_low = pd.DataFrame(rows)
    df_low.to_csv(os.path.join(EXP_DIR, "hasil_uji_lowlight.csv"), index=False)
    print(df_low.to_string(index=False))

    # Grafik perbandingan before-after Adaptive CLAHE
    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=300)
    x = np.arange(len(rows))
    w = 0.34
    ax.bar(x - w / 2, [r["Deteksi Tanpa CLAHE (%)"] for r in rows], width=w, color="#D62728", alpha=0.85, label="Tanpa Enhancement (Raw)")
    ax.bar(x + w / 2, [r["Deteksi Dengan Adaptive CLAHE (%)"] for r in rows], width=w, color="#2CA02C", alpha=0.85, label="Dengan Adaptive CLAHE + Gamma")

    for i, r in enumerate(rows):
        ax.text(i - w / 2, r["Deteksi Tanpa CLAHE (%)"] + 1.5, f"{r['Deteksi Tanpa CLAHE (%)']:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
        ax.text(i + w / 2, r["Deteksi Dengan Adaptive CLAHE (%)"] + 1.5, f"{r['Deteksi Dengan Adaptive CLAHE (%)']:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([r["Kondisi Pencahayaan"] for r in rows], fontsize=8.5, rotation=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Tingkat Keberhasilan Deteksi Wajah (%)", fontsize=9.5)
    ax.set_title("Efektivitas Adaptive Low-Light CLAHE + Gamma pada Dataset LFW", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8.5, loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR, "grafik_clahe_vs_raw.png"), dpi=300)
    plt.close(fig)

    return {"conditions": rows, "native_wider_lowlight": native_summary}


def main():
    yunet_p = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
    sface_p = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
    engine = FaceEngine(yunet_p, sface_p, score_threshold=0.65)

    lfw = fetch_lfw_people(
        data_home=LFW_DIR,
        min_faces_per_person=15,
        color=True,
        resize=1.0,
        download_if_missing=True,
    )

    lfw_res = run_lfw_recognition_experiment(engine, lfw, n_pairs_each=150)
    lowlight_res = run_lowlight_clahe_experiment(engine, lfw)

    antispoof_path = os.path.join(EXP_DIR, "antispoof_eval.json")
    antispoof_res = None
    if os.path.exists(antispoof_path):
        with open(antispoof_path, "r", encoding="utf-8") as f:
            antispoof_res = json.load(f)

    summary = {
        "lfw_recognition": lfw_res,
        "lowlight_enhancement": lowlight_res,
        "antispoofing": antispoof_res,
    }
    out_json = os.path.join(EXP_DIR, "eval_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Seluruh hasil eksperimen disimpan di: {out_json}")


if __name__ == "__main__":
    main()
