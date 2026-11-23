"""Modul Anti-Spoofing Hybrid: Multi-Scale MiniFASNet ONNX (2.7x & 4.0x) + Dual-Scale LBP/FFT/Konteks SVM."""

import os
import cv2
import joblib
import numpy as np


def _build_uniform_lbp_lut():
    """Membangun lookup table 256-kode LBP menjadi 10 bin uniform (0=non-uniform, 1..9=uniform)."""
    lut = np.zeros(256, dtype=np.uint8)
    for code in range(256):
        bits = f"{code:08b}"
        transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        if transitions <= 2:
            lut[code] = bits.count("1") + 1
        else:
            lut[code] = 0
    return lut


_UNIFORM_LUT = _build_uniform_lbp_lut()


def compute_uniform_lbp_map(gray, radius=1):
    """Menghitung peta kode LBP uniform 8-tetangga secara vektorisasi NumPy."""
    r = int(radius)
    if gray.shape[0] <= 2 * r + 2 or gray.shape[1] <= 2 * r + 2:
        return np.zeros_like(gray, dtype=np.uint8)

    center = gray[r:-r, r:-r]
    code = np.zeros_like(center, dtype=np.uint8)
    offsets = [
        (-r, -r), (-r, 0), (-r, r),
        (0, r), (r, r), (r, 0),
        (r, -r), (0, -r),
    ]
    for bit_idx, (dy, dx) in enumerate(offsets):
        y_end = gray.shape[0] - r + dy
        x_end = gray.shape[1] - r + dx
        nb = gray[r + dy:y_end, r + dx:x_end]
        code |= ((nb >= center).astype(np.uint8) << bit_idx)
    return _UNIFORM_LUT[code]


def lbp_histogram(gray, radius=1):
    """Menghitung histogram probabilitas 10 bin dari peta LBP uniform."""
    lbp_map = compute_uniform_lbp_map(gray, radius=radius)
    hist, _ = np.histogram(lbp_map, bins=10, range=(0, 10))
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist, lbp_map


def expand_bbox(img_shape, bbox, scale=2.4):
    """Memperluas kotak deteksi wajah untuk menangkap area bingkai HP / tepi layar / kertas di sekitar wajah."""
    h_img, w_img = img_shape[:2]
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    x1 = max(0, int(round(cx - w * scale / 2.0)))
    y1 = max(0, int(round(cy - h * scale / 2.0)))
    x2 = min(w_img, int(round(cx + w * scale / 2.0)))
    y2 = min(h_img, int(round(cy + h * scale / 2.0)))
    return x1, y1, x2, y2


def _crop_minifasnet_blob(img_bgr, bbox, scale):
    """Menyiapkan tensor input (1, 3, 80, 80) BGR float32 [0..255] sesuai standar Silent-Face-Anti-Spoofing."""
    src_h, src_w = img_bgr.shape[:2]
    x, y, bw, bh = bbox
    bw = max(1.0, float(bw))
    bh = max(1.0, float(bh))
    s = min((src_h - 1) / bh, min((src_w - 1) / bw, float(scale)))
    nw, nh = bw * s, bh * s
    cx, cy = x + bw / 2.0, y + bh / 2.0
    x1 = max(0, int(cx - nw / 2.0))
    y1 = max(0, int(cy - nh / 2.0))
    x2 = min(src_w - 1, int(cx + nw / 2.0))
    y2 = min(src_h - 1, int(cy + nh / 2.0))
    crop = img_bgr[y1:y2 + 1, x1:x2 + 1]
    if crop.size == 0:
        crop = img_bgr
    resized = cv2.resize(crop, (80, 80), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]


def _fft_freq_ratios(gray128):
    """Menghitung rasio energi spektrum frekuensi tinggi & menengah terhadap frekuensi rendah via 2D FFT."""
    f_shift = np.fft.fftshift(np.fft.fft2(gray128.astype(np.float32)))
    mag = np.log1p(np.abs(f_shift))
    yy, xx = np.ogrid[:128, :128]
    dist = np.sqrt((yy - 64) ** 2 + (xx - 64) ** 2)
    low = float(mag[dist < 14].mean())
    mid = float(mag[(dist >= 14) & (dist < 38)].mean())
    high = float(mag[dist >= 38].mean())
    return high / (low + 1e-6), mid / (low + 1e-6)


def extract_antispoof_features(face_bgr):
    """
    Mengekstrak fitur tekstur wajah dalam (skala 1.0x, diseragamkan 128x128 px):
    - Histogram LBP uniform Grayscale pada dua skala (R=1 dan R=2) -> 20 fitur
    - Histogram LBP uniform krominansi YCrCb (Cr, Cb) & HSV (S, V) -> 40 fitur
    - Statistik frekuensi tinggi Laplacian & fraksi pantulan spekular -> 3 fitur
    Total vektor fitur: 63 dimensi (Määttä et al., 2011; Boulkenafet et al., 2015).
    """
    if face_bgr is None or face_bgr.size == 0:
        return np.zeros(63, dtype=np.float32), None, {}

    face_std = cv2.resize(face_bgr, (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(face_std, cv2.COLOR_BGR2GRAY)
    ycrcb = cv2.cvtColor(face_std, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(face_std, cv2.COLOR_BGR2HSV)

    h_g1, lbp_map_vis = lbp_histogram(gray, radius=1)
    h_g2, _ = lbp_histogram(gray, radius=2)
    h_cr, _ = lbp_histogram(ycrcb[:, :, 1], radius=1)
    h_cb, _ = lbp_histogram(ycrcb[:, :, 2], radius=1)
    h_s, _ = lbp_histogram(hsv[:, :, 1], radius=1)
    h_v, _ = lbp_histogram(hsv[:, :, 2], radius=1)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    specular_ratio = float(np.mean((hsv[:, :, 2] > 235) & (hsv[:, :, 1] < 45)))
    non_uniform_ratio = float(h_g1[0])

    extra = np.array([lap_var / 1000.0, specular_ratio, non_uniform_ratio], dtype=np.float32)
    feat_vec = np.concatenate([h_g1, h_g2, h_cr, h_cb, h_s, h_v, extra]).astype(np.float32)

    diagnostics = {
        "lbp_hist_r1": h_g1,
        "laplacian_var": round(lap_var, 2),
        "specular_ratio": round(specular_ratio, 4),
        "non_uniform_ratio": round(non_uniform_ratio, 4),
    }
    return feat_vec, lbp_map_vis, diagnostics


def extract_dual_scale_features(full_bgr, bbox):
    """
    Mengekstrak fitur Anti-Spoofing Dual-Scale (105 dimensi):
    - Skala 1 (1.0x wajah dalam): 63 fitur LBP multi-kanal + Laplacian + Specular
    - Skala 2 (2.4x konteks sekitar wajah): 30 fitur LBP Gray/YCrCb + 12 fitur bingkai HP,
      kontras tepi layar, garis lurus Sobel/Canny, dan spektrum 2D FFT (domain frekuensi).
    """
    if full_bgr is None or full_bgr.size == 0 or bbox is None:
        return np.zeros(105, dtype=np.float32), None, {}

    h_img, w_img = full_bgr.shape[:2]
    x, y, w, h = bbox
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1_in, y1_in = min(w_img, int(x + w)), min(h_img, int(y + h))
    inner = full_bgr[y0:y1_in, x0:x1_in]
    if inner.size == 0:
        return np.zeros(105, dtype=np.float32), None, {}

    v_in, lbp_map_vis, diag = extract_antispoof_features(inner)

    cx1, cy1, cx2, cy2 = expand_bbox(full_bgr.shape, (x0, y0, max(1, x1_in - x0), max(1, y1_in - y0)), scale=2.4)
    ctx = full_bgr[cy1:cy2, cx1:cx2]
    ctx_std = cv2.resize(ctx, (128, 128), interpolation=cv2.INTER_AREA)
    gray_c = cv2.cvtColor(ctx_std, cv2.COLOR_BGR2GRAY)
    ycrcb_c = cv2.cvtColor(ctx_std, cv2.COLOR_BGR2YCrCb)
    hsv_c = cv2.cvtColor(ctx_std, cv2.COLOR_BGR2HSV)

    h_cg, _ = lbp_histogram(gray_c, radius=1)
    h_cr, _ = lbp_histogram(ycrcb_c[:, :, 1], radius=1)
    h_cb, _ = lbp_histogram(ycrcb_c[:, :, 2], radius=1)

    mask_out = np.ones((128, 128), dtype=bool)
    mask_out[30:98, 30:98] = False
    v_out = hsv_c[:, :, 2][mask_out]
    v_ctr = hsv_c[30:98, 30:98, 2]

    dark_bezel = float(np.mean(v_out < 40))
    very_dark = float(np.mean(v_out < 22))
    bright_out = float(np.mean(v_out > 220))
    lum_diff = float((np.mean(v_ctr) - np.mean(v_out)) / 255.0)
    lum_std_out = float(np.std(v_out) / 255.0)

    gx = np.abs(cv2.Sobel(gray_c, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray_c, cv2.CV_32F, 0, 1, ksize=3))
    v_lines = float(np.max(np.mean(gx[:, :32] > 45, axis=0))) + float(np.max(np.mean(gx[:, 96:] > 45, axis=0)))
    h_lines = float(np.max(np.mean(gy[:32, :] > 45, axis=1))) + float(np.max(np.mean(gy[96:, :] > 45, axis=1)))
    edges = cv2.Canny(gray_c, 60, 150)
    edge_out = float(np.mean(edges[mask_out] > 0))

    in_gray = cv2.cvtColor(cv2.resize(inner, (128, 128), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    hf_in, mf_in = _fft_freq_ratios(in_gray)
    hf_ctx, mf_ctx = _fft_freq_ratios(gray_c)

    ctx_extra = np.array([
        dark_bezel, very_dark, bright_out, lum_diff, lum_std_out,
        v_lines, h_lines, edge_out, hf_in, mf_in, hf_ctx, mf_ctx,
    ], dtype=np.float32)

    feat_vec = np.concatenate([v_in, h_cg, h_cr, h_cb, ctx_extra]).astype(np.float32)
    diag["bezel_line_score"] = round(v_lines + h_lines, 3)
    diag["dark_bezel_ratio"] = round(dark_bezel, 4)
    diag["fft_high_ratio"] = round(hf_in, 4)
    return feat_vec, lbp_map_vis, diag


class AntiSpoofChecker:
    """Klasifikator Anti-Spoofing Hybrid: Multi-Scale MiniFASNet ONNX (2.7x & 4.0x) + Dual-Scale LBP/FFT SVM."""

    def __init__(
        self,
        svm_path=None,
        scaler_path=None,
        prob_threshold=0.50,
        fas_v2_path=None,
        fas_v1se_path=None,
    ):
        self.prob_threshold = float(prob_threshold)
        self.svm = None
        self.scaler = None
        self.expected_dim = 63
        self.fas_models = []  # list of (scale, cv2.dnn.Net)

        if svm_path and scaler_path and os.path.exists(svm_path) and os.path.exists(scaler_path):
            self.svm = joblib.load(svm_path)
            self.scaler = joblib.load(scaler_path)
            self.expected_dim = int(getattr(self.scaler, "n_features_in_", 63))

        if fas_v2_path and os.path.exists(fas_v2_path):
            self.fas_models.append((2.7, cv2.dnn.readNetFromONNX(fas_v2_path)))
        if fas_v1se_path and os.path.exists(fas_v1se_path):
            self.fas_models.append((4.0, cv2.dnn.readNetFromONNX(fas_v1se_path)))

        if self.fas_models and self.svm is not None:
            self.mode = "Hybrid MiniFASNet ONNX + LBP/FFT SVM"
        elif self.fas_models:
            self.mode = "MiniFASNet ONNX (2.7x & 4.0x)"
        elif self.svm is not None:
            self.mode = "Dual-Scale LBP+FFT+Konteks SVM" if self.expected_dim == 105 else "LBP + SVM Terlatih"
        else:
            self.mode = "Heuristik Tekstur (Model SVM belum dilatih)"

    def _predict_fas(self, full_bgr, bbox):
        """Menjalankan ensemble MiniFASNet ONNX -> probabilitas [Paper, Real, Screen]."""
        probs = np.zeros(3, dtype=np.float32)
        for scale, net in self.fas_models:
            blob = _crop_minifasnet_blob(full_bgr, bbox, scale)
            net.setInput(blob)
            logits = net.forward().reshape(-1)
            exp_v = np.exp(logits - np.max(logits))
            probs += exp_v / np.sum(exp_v)
        probs /= float(len(self.fas_models))
        return float(probs[0]), float(probs[1]), float(probs[2])

    def predict(self, face_bgr, full_bgr=None, bbox=None):
        """Mengembalikan (is_real, prob_real, lbp_map, diagnostics)."""
        src_img = full_bgr if (full_bgr is not None and bbox is not None) else face_bgr
        if bbox is not None and full_bgr is not None:
            src_bbox = bbox
        else:
            h_f, w_f = face_bgr.shape[:2] if (face_bgr is not None and face_bgr.size > 0) else (0, 0)
            src_bbox = (0, 0, w_f, h_f)

        # 1. Ekstraksi fitur PCD (LBP + FFT + Bezel) & probabilitas SVM bila tersedia
        prob_svm = None
        if self.svm is not None and self.scaler is not None:
            if self.expected_dim == 105:
                feat_vec, lbp_map, diag = extract_dual_scale_features(src_img, src_bbox)
            else:
                feat_vec, lbp_map, diag = extract_antispoof_features(face_bgr)
            x_scaled = self.scaler.transform(feat_vec.reshape(1, -1))
            probs_s = self.svm.predict_proba(x_scaled)[0]
            classes = list(self.svm.classes_)
            real_idx = classes.index(1) if 1 in classes else -1
            prob_svm = float(probs_s[real_idx]) if real_idx >= 0 else float(probs_s[-1])
            diag["svm_prob_real"] = round(prob_svm, 3)
        else:
            feat_vec, lbp_map, diag = extract_dual_scale_features(src_img, src_bbox) if (full_bgr is not None and bbox is not None) else extract_antispoof_features(face_bgr)

        # 2. Inferensi Multi-Scale MiniFASNet ONNX (2.7x & 4.0x)
        if self.fas_models and src_img is not None and src_img.size > 0:
            p_paper, p_real_fas, p_screen = self._predict_fas(src_img, src_bbox)
            diag["fas_paper"] = round(p_paper, 3)
            diag["fas_real"] = round(p_real_fas, 3)
            diag["fas_screen"] = round(p_screen, 3)

            if prob_svm is not None:
                # Fusi Hybrid: bobot utama pada MiniFASNet (0.78) + PCD LBP/FFT SVM (0.22)
                # Dengan pengaman veto: jika MiniFASNet sangat yakin layar/kertas (P_Real < 0.25), jangan diloloskan
                fused = 0.78 * p_real_fas + 0.22 * prob_svm
                if p_real_fas < 0.25:
                    prob_real = min(fused, p_real_fas + 0.04)
                else:
                    prob_real = fused
            else:
                prob_real = p_real_fas

            is_real = prob_real >= self.prob_threshold
            if is_real:
                diag["attack_type"] = "Wajah Asli"
            elif p_screen >= p_paper and p_screen >= 0.35:
                diag["attack_type"] = "Layar HP"
            elif p_paper > p_screen and p_paper >= 0.35:
                diag["attack_type"] = "Foto Kertas"
            else:
                diag["attack_type"] = "Tekstur Spoof"
            diag["mode"] = self.mode
            return is_real, float(prob_real), lbp_map, diag

        # 3. Jika hanya SVM tanpa MiniFASNet
        if prob_svm is not None:
            is_real = prob_svm >= self.prob_threshold
            diag["attack_type"] = "Wajah Asli" if is_real else "Tekstur Spoof"
            diag["mode"] = self.mode
            return is_real, prob_svm, lbp_map, diag

        # 4. Fallback heuristik jika tidak ada model sama sekali
        lap_v = diag.get("laplacian_var", 100.0)
        spec_r = diag.get("specular_ratio", 0.0)
        score = 0.80
        if spec_r > 0.12:
            score -= 0.40
        if lap_v < 22.0 or lap_v > 2800.0:
            score -= 0.35
        prob_real = max(0.05, min(0.95, score))
        is_real = prob_real >= self.prob_threshold
        diag["attack_type"] = "Wajah Asli" if is_real else "Tekstur Spoof"
        diag["mode"] = self.mode
        return is_real, prob_real, lbp_map, diag
