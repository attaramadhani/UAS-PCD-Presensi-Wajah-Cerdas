"""Modul perbaikan kualitas citra adaptif untuk kondisi pencahayaan gelap & backlight."""

import cv2
import numpy as np


def analyze_lighting(img_bgr):
    """Mengukur luminansi rata-rata dan sebaran kontras pada kanal L ruang warna CIELAB."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0]
    mean_l = float(np.mean(l_chan))
    std_l = float(np.std(l_chan))

    if mean_l < 75.0:
        status = "Gelap (Underexposed)"
    elif mean_l < 105.0:
        status = "Agak Redup / Backlight"
    elif std_l < 35.0:
        status = "Kontras Rendah"
    else:
        status = "Cahaya Normal"

    return {
        "mean_l": mean_l,
        "std_l": std_l,
        "status": status,
        "l_channel": l_chan,
    }


def apply_gamma(img_bgr, gamma=0.7):
    """Menerapkan transformasi pangkat (Gamma Correction) melalui lookup table 256-level."""
    inv_gamma = float(gamma)
    lut = np.array(
        [((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(img_bgr, lut)


def apply_adaptive_enhancement(
    img_bgr,
    l_threshold=105.0,
    std_threshold=38.0,
    clip_limit=2.2,
    force=False,
):
    """
    Meningkatkan pencahayaan secara adaptif hanya jika citra terdeteksi gelap atau kontras rendah.
    Koreksi warna mempertahankan kanal a* dan b* pada CIELAB agar rona kulit alami tidak berubah.
    """
    stats = analyze_lighting(img_bgr)
    mean_l = stats["mean_l"]
    std_l = stats["std_l"]

    needs_enhance = force or (mean_l < l_threshold) or (std_l < std_threshold)
    if not needs_enhance:
        return img_bgr.copy(), {
            "applied": False,
            "before_l": round(mean_l, 2),
            "after_l": round(mean_l, 2),
            "before_std": round(std_l, 2),
            "after_std": round(std_l, 2),
            "gamma_used": 1.0,
            "status_before": stats["status"],
            "reason": "Pencahayaan sudah memadai; melewati enhancement agar detail alami terjaga.",
        }

    # Hitung nilai gamma adaptif (< 1 mencerahkan area gelap)
    if mean_l < l_threshold:
        ratio = max(0.0, min(1.0, mean_l / l_threshold))
        gamma_used = round(0.55 + 0.35 * ratio, 3)
    else:
        gamma_used = 0.88

    work = apply_gamma(img_bgr, gamma=gamma_used)

    # Terapkan CLAHE hanya pada kanal L ruang warna CIELAB
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
    l_eq = clahe.apply(l_chan)
    enhanced_bgr = cv2.cvtColor(cv2.merge((l_eq, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

    after_stats = analyze_lighting(enhanced_bgr)
    return enhanced_bgr, {
        "applied": True,
        "before_l": round(mean_l, 2),
        "after_l": round(after_stats["mean_l"], 2),
        "before_std": round(std_l, 2),
        "after_std": round(after_stats["std_l"], 2),
        "gamma_used": gamma_used,
        "status_before": stats["status"],
        "reason": f"Aktif otomatis ({stats['status']}: L={mean_l:.1f}, std={std_l:.1f}).",
    }
