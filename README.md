# Sistem Presensi Wajah Cerdas Berbasis Pengolahan Citra Digital

Proyek Akhir Mata Kuliah **Pengolahan Citra Digital (Kelas B)** — Program Studi Sistem Informasi, Fakultas Teknik, Universitas Trunojoyo Madura (2026).

- **Dosen Pengampu:** Dr. Fitri Damayanti, S.Kom., M.Kom.
- **Tim Penyusun (Kelompok 5):**
  1. **Attala Alif Ramadhani Tri Hida** (`230441100144`) — Koordinator & Arsitek Sistem
  2. **M. Rafly Kurniawan** (`240441100086`) — Pengembang Modul PCD & Anti-Spoofing
  3. **Moh. Yoga Dwi Prasetyo** (`240441100133`) — Pengujian, Evaluasi & Dokumentasi

---

## Ringkasan Arsitektur Pipeline

Aplikasi web berbasis **Streamlit** yang berjalan sepenuhnya secara lokal pada **CPU standar (tanpa GPU/IoT/APK)** dengan 4 komponen utama Pengolahan Citra Digital:

1. **Adaptive Low-Light Enhancement (`modules/enhancement.py`)**
   - Pemeriksaan ambang luminansi pada kanal `L*` ruang warna CIELAB (`μL ≥ 105,0` dan `σL ≥ 38,0` → *bypass* otomatis agar citra terang tidak mengalami *over-enhancement*).
   - Jika redup, sistem menerapkan **Adaptive Gamma Correction** (`γ ∈ [0,55; 0,90]`) dan **CLAHE** (`clipLimit = 2,2`, `tileGridSize = 8×8`) selektif hanya pada kanal `L*` sehingga warna kulit alami (`a*`, `b*`) tetap terjaga.
2. **Deteksi Multi-Wajah & Ekstraksi Fitur (`modules/face_engine.py`)**
   - **YuNet ONNX** (`face_detection_yunet_2023mar.onnx`) mendeteksi kotak wajah + 5 titik *landmark* anatomis sekaligus meluruskan wajah menjadi `112×112` piksel (*affine alignment*).
   - **SFace ONNX** (`face_recognition_sface_2021dec.onnx`) mengekstrak *embedding* `128` dimensi ternormalisasi L2 dan mencocokkan identitas menggunakan *Cosine Similarity* (`θ = 0,363`).
3. **Pertahanan Anti-Spoofing Hybrid Pra-CLAHE (`modules/spoofing_lbp.py`)**
   - Dieksekusi **langsung pada citra mentah (*Raw*) sebelum CLAHE** agar pantulan spekular dan pola kisi layar ponsel tidak terdistorsi.
   - Menggabungkan **SVM RBF** terkalibrasi probabilitas atas **105 fitur Dual-Scale** (Uniform LBP multi-ruang warna `Gray/YCrCb/HSV`, spektrum energi `2D-FFT`, dan statistik bezel/tepi `Sobel + Canny`) bersama **Ensemble MiniFASNetV2 (`2,7×`) & MiniFASNetV1SE (`4,0×`) ONNX** + mekanisme veto (`P_FAS < 0,25`).
4. **Webcam Live Otomatis & Manajemen Sesi (`modules/live_webcam.py`, `modules/database.py`, `app.py`)**
   - Komponen kustom WebRTC `getUserMedia` di browser dengan *overlay canvas* real-time dan pemrosesan `@st.fragment` non-blocking.
   - Pembatasan ketat **maksimal 1× presensi per individu per sesi** (konfirmasi 2 frame berturut-turut → status terkunci `SUDAH ABSEN (1×)`).
   - Penyimpanan permanen ke **SQLite (`presensi.db`)**, arsip **CSV**, dan ekspor **Excel (`.xlsx`)**.

---

## Struktur Direktori Repository

> **Catatan Ukuran Dataset:** Berkas dataset latih/uji berukuran besar (`data/raw_lfw/` ~380 MB / 13.251 berkas dan `data/spoof_dataset/` ~115 MB / 523 berkas) serta database biometrik lokal (`artifacts/db/*.npz`, `presensi.db`, `students_master.csv`) sengaja dikecualikan lewat `.gitignore`. Seluruh cara mengunduh ulang, melatih model dari awal, dan mereproduksi hasil evaluasi tersedia di bagian bawah.

```text
.
├── app.py                                # Aplikasi web utama Streamlit (4 Tab)
├── jalankan_demo.bat                     # Launcher cepat Windows
├── requirements.txt                      # Dependensi Python
├── .streamlit/config.toml                # Konfigurasi tema Antarmuka
├── modules/
│   ├── enhancement.py                    # CIELAB L*, Adaptive Gamma, & CLAHE
│   ├── face_engine.py                    # YuNet 5-landmark & SFace 128-D ONNX
│   ├── spoofing_lbp.py                   # 105 fitur Dual-Scale + MiniFASNet Hybrid
│   ├── live_webcam.py                    # Komponen WebRTC Webcam Live browser
│   └── database.py                       # Enrollment NPZ/CSV & arsip SQLite/Excel
├── scripts/
│   ├── 01_download_models.py             # Unduh model ONNX publik & dataset LFW
│   ├── 02_train_antispoofing.py          # Ekstraksi fitur, training SVM, & eval Hybrid
│   └── 03_run_experiments.py             # Uji kuantitatif LFW & Low-Light CLAHE
├── artifacts/
│   ├── models/                           # Model ONNX + hasil training SVM (.joblib)
│   ├── experiments/                      # JSON, CSV, & grafik 300 DPI hasil evaluasi
│   └── db/                               # ui_labels.json (database lokal di-ignore)
├── data/
│   ├── demo_kelas/                       # Citra sampel (lena.jpg, solvay_1927.jpg)
│   ├── spoof_dataset/{real,spoof}/       # Tempat dataset latih Anti-Spoofing (diisi saat training)
│   ├── raw_lfw/                          # Cache LFW (otomatis terisi oleh script 01/03)
│   └── wider_lowlight/                   # Opsional foto gelap asli tambahan
└── docs/                                 # Laporan Akhir (PDF/DOCX), BibTeX, & PPTX Presentasi
```

---

## Cara Instalasi & Menjalankan Aplikasi

1. **Siapkan lingkungan Python (disarankan Python 3.10–3.13+)** dan pasang seluruh pustaka:
   ```bash
   pip install -r requirements.txt
   ```
2. **Jalankan aplikasi Streamlit** (bobot model ONNX dan SVM sudah siap di `artifacts/models/`):
   ```bash
   python -m streamlit run app.py
   ```
   Atau di Windows cukup klik dua kali **`jalankan_demo.bat`**.
3. Buka browser di `http://localhost:8501` (alamat `localhost` diperlukan agar izin kamera **Webcam Live** `getUserMedia` aktif penuh).

---

## Panduan Training Model Ulang & Reproduksi Evaluasi

Karena dataset mentah (`data/raw_lfw/` dan `data/spoof_dataset/`) tidak ikut disimpan ke GitHub, ikuti 3 langkah berikut apabila Anda ingin mengunduh ulang aset, melatih model Anti-Spoofing dari nol, atau mereproduksi grafik evaluasi:

### Langkah 1 — Unduh Model Pralatih ONNX & Benchmark Dataset LFW
```bash
python scripts/01_download_models.py
```
Skrip ini otomatis:
- Mengunduh 4 model ONNX ke `artifacts/models/`:
  - `face_detection_yunet_2023mar.onnx` & `face_recognition_sface_2021dec.onnx` (OpenCV Model Zoo)
  - `2.7_80x80_MiniFASNetV2.onnx` & `4_0_0_80x80_MiniFASNetV1SE.onnx` (Silent-Face-Anti-Spoofing ONNX)
- Mengunduh dan menyimpan *cache* korpus publik **Labeled Faces in the Wild (LFW)** ke `data/raw_lfw/`.

### Langkah 2 — Siapkan Dataset & Latih Model Anti-Spoofing Hybrid
1. Isi dua folder berikut dengan citra berformat `.jpg`/`.png` (minimal 10 citra per kelas; untuk performa optimal seperti evaluasi laporan digunakan **287 citra `real` + 236 citra `spoof`** resolusi *full-frame* kamera ponsel):
   - `data/spoof_dataset/real/` → foto wajah asli langsung di depan kamera.
   - `data/spoof_dataset/spoof/` → foto wajah yang dipotret dari layar ponsel/monitor atau hasil cetakan kertas.
2. Jalankan pelatihan & evaluasi:
   ```bash
   python scripts/02_train_antispoofing.py
   ```
   **Proses yang dijalankan skrip:**
   - Mendeteksi ROI wajah via YuNet (menggunakan *fallback* full-frame bila gagal) dan mengekstrak **105 fitur Dual-Scale** (`1,0×` wajah dalam + `2,4×` konteks bezel, 2D-FFT, Sobel/Canny).
   - Membagi dataset **80% latih / 20% uji** secara *stratified* (`random_state=42`).
   - Melatih `StandardScaler` + **`SVC(kernel="rbf", C=3.0, gamma="scale")`** dengan kalibrasi probabilitas **`CalibratedClassifierCV(cv=5)`**.
   - Menyimpan hasil latih ke `artifacts/models/anti_spoof_svm.joblib` & `artifacts/models/scaler_lbp.joblib`.
   - Mengevaluasi fusi Hybrid (MiniFASNet ONNX + SVM) pada 20% data uji serta menghasilkan `artifacts/experiments/cm_antispoofing.png` dan `antispoof_eval.json`.

### Langkah 3 — Jalankan Eksperimen Kuantitatif (LFW & Adaptive Low-Light)
```bash
python scripts/03_run_experiments.py
```
Skrip ini menjalankan:
- **Eksperimen 1 (Verifikasi Identitas LFW):** Menguji `300` pasangan (`150` identik, `150` berbeda) dari `96` individu pada ambang `θ = 0,363`, menghasilkan `grafik_lfw_cosine.png`.
- **Eksperimen 2 (Adaptive Low-Light CLAHE):** Menguji `60` citra LFW pada simulasi penggelapan Gamma `1,0`, `2,5`, `3,5`, dan `4,5` sebelum vs sesudah enhancement (serta foto tambahan di `data/wider_lowlight/` bila diisi), menghasilkan `hasil_uji_lowlight.csv` dan `grafik_clahe_vs_raw.png`.
- Menyimpan ringkasan seluruh metrik ke `artifacts/experiments/eval_summary.json` yang langsung dibaca oleh **Tab Evaluasi** di antarmuka Streamlit.

---

## Ringkasan Hasil Evaluasi Terukur

| Pengujian | Skenario / Sampel | Hasil Utama |
| :--- | :--- | :--- |
| **Deteksi & Verifikasi LFW (`YuNet–SFace`)** | 300 pasangan uji (96 individu, `θ = 0,363`) | Akurasi **99,33%**, Presisi **100,00%**, Recall **98,67%**, F1 **99,33%**, Latensi CPU **5,83 ms/wajah** |
| **Ketahanan Low-Light (`Adaptive CLAHE`)** | 60 citra LFW × 4 tingkat gelap | Kondisi gelap ekstrem Gamma 4,5 pulih dari **96,67% → 100,00% (+3,33%)** |
| **Anti-Spoofing Hybrid** | 523 citra *smartphone* (105 uji 20%) + uji layar HP nyata | Akurasi uji **86,67%**, Recall Real **91,38%**, F1 **88,33%**, ROC-AUC **0,9231**; serangan HP nyata diblokir (`P(Real) = 0,039–0,040`) |
| **Deteksi Multi-Wajah Serentak** | Foto grup klasik `solvay_1927.jpg` | **29/29 wajah** terdeteksi dalam 1 proses |

---

## Dokumen Laporan & Presentasi (`docs/`)
- `docs/Laporan_Final_Project_PCD_Kelompok5.pdf` — Laporan Proyek Akhir resmi (33 halaman, sitasi APA & persamaan LaTeX native Word).
- `docs/Laporan_Final_Project_PCD_Kelompok5.docx` — Sumber dokumen Microsoft Word siap sunting.
- `docs/referensi_kelompok5.bib` — Berkas referensi BibTeX (10 rujukan ilmiah).
- `docs/Presentasi_Final_Project_PCD_Kelompok5.pptx` — Salindia presentasi sidang/demo (14 slide).
