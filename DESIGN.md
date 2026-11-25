# Design System — Sistem Presensi Biometrik Wajah (PCD)

Design Read:
> Reading this as: Local Computer Vision Laboratory & Academic Attendance Instrument for lecturers and university students, in a Crisp Daylight Technical Editorial style, dial **ENERGY 2 / RHYTHM 2 / MOTION 1**.

## 1. Visual Identity & Motif
- **Character**: Instrumen laboratorium visi komputer dan presensi kelas yang presisi, tenang, dan mudah dibaca di proyektor kelas maupun layar ponsel.
- **Identity Motif**: Garis pembatas rambut tipis (`1px solid #D8DFE8`), header editorial berlatar terang dengan garis bawah tegas `#0F294A`, dan angka berjarak seragam (`tabular-nums`) untuk skor Cosine, luminansi $L^*$, FPS, dan waktu hadir.
- **Anti-Patterns Removed**: Tidak ada hero bergradasi biru-ungu, tidak ada *colored left stripe* pada seluruh kartu, tidak ada *eyebrow pill* di atas `<h1>`, tidak ada emoji dekoratif di antarmuka, tidak ada titik ber-*glow*, dan tidak ada font eksternal yang memblokir muat halaman offline.

## 2. Color Palette (2 Core + 1 Accent + Neutral Surfaces)
- **Canvas Background**: `#F5F6F8` (netral terang matte)
- **Elevated Surface**: `#FFFFFF` dengan batas `#D8DFE8`
- **Primary Core Ink**: `#111827` (rasio kontras `> 16:1` pada putih)
- **Secondary Muted Ink**: `#475569` (rasio kontras `7.2:1` pada putih, lolos WCAG AA `>= 4.5:1`)
- **Primary Brand Core (Academic Navy)**: `#0F294A`
- **Single Deliberate Accent (Focus & Active Action)**: `#1D4ED8` (dipakai khusus pada tombol aksi utama, indikator tab aktif, dan ring `:focus-visible`)
- **Functional Camera & Session States** (khusus penanda status nyata):
  - `Hadir / Aktif / Siap`: `#047857` pada `#ECFDF5` (batas `#A7F3D0`)
  - `Spoof / Ditutup / Error`: `#B91C1C` pada `#FEF2F2` (batas `#FECACA`)
  - `Peringatan / Terkunci`: `#92400E` pada `#FFFBEB` (batas `#FDE68A`)

## 3. Typography & Radius Scale
- **UI Sans**: `'Segoe UI', -apple-system, BlinkMacSystemFont, system-ui, sans-serif` (zero remote `@import`, aman offline)
- **Technical Monospace (`tabular-nums`)**: `ui-monospace, 'Cascadia Code', 'Consolas', monospace`
- **Border Radius**: `4px` (tag status/chip kompak), `6px` (tombol, input, select), `8px` (kartu, banner, panel kamera). Tidak memakai `999px` pill berlebih.

## 4. Layout Rhythm, Accessibility & Motion
- **RHYTHM 2**: Setiap tab membedakan area keputusan utama vs informasi pendukung (header datar editorial, metrik utama lebih menonjol dari metrik sekunder, panggung kamera utama berdampingan langsung dengan tabel rekap sesi).
- **Accessibility**: Satu `<h1>` semantik dalam `<header>`, `:focus-visible` `2px solid #1D4ED8` pada seluruh tombol & `<select>`, `<label>` eksplisit, target sentuh minimal `44px` pada mobile (`<= 768px`), zero horizontal overflow.
- **MOTION 1**: Transisi hover `120ms ease` saja dan menghormati `@media (prefers-reduced-motion: reduce)`. Zero animasi berulang/pulsing.
