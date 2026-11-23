"""Mesin deteksi dan pengenalan wajah berbasis YuNet (FaceDetectorYN) & SFace (FaceRecognizerSF)."""

import os
import cv2
import numpy as np


COSINE_THRESHOLD_DEFAULT = 0.363  # Ambang batas standar resmi OpenCV SFace


def resize_max_side(img_bgr, max_side=1280):
    """Menskala ulang citra terlalu besar agar inferensi cepat tanpa kehilangan proporsi."""
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img_bgr, 1.0
    scale = float(max_side) / float(longest)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


class FaceEngine:
    """Wrapper terpadu untuk deteksi multi-wajah YuNet, alignment 5-landmark, dan embedding SFace."""

    def __init__(self, yunet_path, sface_path, score_threshold=0.75, nms_threshold=0.3):
        if not os.path.exists(yunet_path):
            raise FileNotFoundError(f"Model YuNet ONNX tidak ditemukan: {yunet_path}")
        if not os.path.exists(sface_path):
            raise FileNotFoundError(f"Model SFace ONNX tidak ditemukan: {sface_path}")

        self.yunet_path = yunet_path
        self.sface_path = sface_path
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)

        self.detector = cv2.FaceDetectorYN.create(
            yunet_path, "", (320, 320), self.score_threshold, self.nms_threshold, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    def detect(self, img_bgr, score_threshold=None):
        """Mendeteksi seluruh wajah beserta 5 titik landmark pada citra BGR."""
        if img_bgr is None or img_bgr.size == 0:
            return []

        if score_threshold is not None and score_threshold != self.score_threshold:
            self.score_threshold = float(score_threshold)
            self.detector.setScoreThreshold(self.score_threshold)

        h, w = img_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, raw_faces = self.detector.detect(img_bgr)
        if raw_faces is None:
            return []

        results = []
        for row in raw_faces:
            x, y, bw, bh = int(round(row[0])), int(round(row[1])), int(round(row[2])), int(round(row[3]))
            # Pastikan kotak selalu berada dalam batas dimensi citra
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            if x2 <= x1 or y2 <= y1:
                continue

            landmarks = [
                (float(row[4]), float(row[5])),    # Mata kanan
                (float(row[6]), float(row[7])),    # Mata kiri
                (float(row[8]), float(row[9])),    # Ujung hidung
                (float(row[10]), float(row[11])),  # Sudut mulut kanan
                (float(row[12]), float(row[13])),  # Sudut mulut kiri
            ]
            results.append({
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "score": float(row[14]),
                "landmarks": landmarks,
                "raw_row": row,
            })

        # Urutkan deteksi dari kiri ke kanan agar penomoran wajah konsisten
        results.sort(key=lambda d: (d["bbox"][1] // 60, d["bbox"][0]))
        return results

    def align_and_embed(self, img_bgr, raw_row):
        """Meluruskan posisi wajah (5-point geometric alignment) lalu mengekstrak vektor 128D."""
        aligned_face = self.recognizer.alignCrop(img_bgr, raw_row)
        embedding = self.recognizer.feature(aligned_face)
        return aligned_face, embedding.astype(np.float32)

    def cosine_similarity(self, emb1, emb2):
        """Menghitung skor kemiripan Cosine Similarity menggunakan standar OpenCV SFace."""
        e1 = np.asarray(emb1, dtype=np.float32).reshape(1, -1)
        e2 = np.asarray(emb2, dtype=np.float32).reshape(1, -1)
        return float(self.recognizer.match(e1, e2, cv2.FaceRecognizerSF_FR_COSINE))

    def match_identity(self, query_emb, db_embeddings, db_nims, db_names, threshold=COSINE_THRESHOLD_DEFAULT):
        """Mencocokkan embedding wajah uji terhadap database mahasiswa terdaftar."""
        if db_embeddings is None or len(db_embeddings) == 0:
            return False, None, "Tidak Terdaftar", 0.0

        best_score = -1.0
        best_idx = -1
        for idx, ref_emb in enumerate(db_embeddings):
            score = self.cosine_similarity(query_emb, ref_emb)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0 and best_score >= float(threshold):
            return True, str(db_nims[best_idx]), str(db_names[best_idx]), best_score
        return False, None, "Tidak Terdaftar", max(0.0, best_score)
