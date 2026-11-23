"""Skrip 01: Mengunduh model resmi OpenCV Zoo (YuNet & SFace ONNX) serta menyiapkan cache LFW."""

import os
import sys
import urllib.request
import cv2
from sklearn.datasets import fetch_lfw_people

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
LFW_DIR = os.path.join(BASE_DIR, "data", "raw_lfw")
WIDER_DIR = os.path.join(BASE_DIR, "data", "wider_lowlight")
SPOOF_REAL_DIR = os.path.join(BASE_DIR, "data", "spoof_dataset", "real")
SPOOF_FAKE_DIR = os.path.join(BASE_DIR, "data", "spoof_dataset", "spoof")
DEMO_DIR = os.path.join(BASE_DIR, "data", "demo_kelas")

for d in [MODELS_DIR, LFW_DIR, WIDER_DIR, SPOOF_REAL_DIR, SPOOF_FAKE_DIR, DEMO_DIR]:
    os.makedirs(d, exist_ok=True)

MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
    "2.7_80x80_MiniFASNetV2.onnx": (
        "https://raw.githubusercontent.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/"
        "main/onnx/2.7_80x80_MiniFASNetV2.onnx"
    ),
    "4_0_0_80x80_MiniFASNetV1SE.onnx": (
        "https://raw.githubusercontent.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/"
        "main/onnx/4_0_0_80x80_MiniFASNetV1SE.onnx"
    ),
}


def main():
    print("=== 1. Mengunduh Model Resmi OpenCV Zoo ONNX ===")
    for fname, url in MODEL_URLS.items():
        dst = os.path.join(MODELS_DIR, fname)
        if os.path.exists(dst) and os.path.getsize(dst) > 10000:
            print(f"[SKIP] Sudah ada: {fname} ({os.path.getsize(dst):,} bytes)")
        else:
            print(f"[GET ] Mengunduh {fname} ...")
            urllib.request.urlretrieve(url, dst)
            print(f"[OK  ] Tersimpan: {dst} ({os.path.getsize(dst):,} bytes)")

    yunet_p = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
    sface_p = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
    det = cv2.FaceDetectorYN.create(yunet_p, "", (320, 320), 0.7, 0.3, 5000)
    rec = cv2.FaceRecognizerSF.create(sface_p, "")
    print(f"[OK  ] YuNet & SFace berhasil dimuat pada OpenCV {cv2.__version__}")

    print("\n=== 2. Menyiapkan Dataset Benchmark LFW (UMass Amherst) ===")
    lfw = fetch_lfw_people(
        data_home=LFW_DIR,
        min_faces_per_person=15,
        color=True,
        resize=1.0,
        download_if_missing=True,
    )
    print(
        f"[OK  ] LFW siap: {lfw.images.shape[0]} citra wajah "
        f"({lfw.images.shape[1]}x{lfw.images.shape[2]} px) dari {len(lfw.target_names)} individu"
    )
    print("\nPersiapan Tahap 1 selesai.")


if __name__ == "__main__":
    main()
