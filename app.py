"""Aplikasi Web Live Demo: Sistem Presensi Wajah Cerdas (Adaptive Low-Light CLAHE + YuNet-SFace + Anti-Spoofing LBP)."""

import hashlib
import html
import json
import os
import socket
import sys
import time
from datetime import datetime, timedelta
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.database import StudentDatabase
from modules.enhancement import analyze_lighting, apply_adaptive_enhancement
from modules.face_engine import COSINE_THRESHOLD_DEFAULT, FaceEngine, resize_max_side
from modules.live_webcam import (
    build_overlay_boxes,
    decode_data_url_to_bgr,
    render_live_webcam,
)
from modules.spoofing_lbp import AntiSpoofChecker

MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
DB_DIR = os.path.join(BASE_DIR, "artifacts", "db")
LOGS_DIR = os.path.join(BASE_DIR, "artifacts", "attendance_logs")
EXP_DIR = os.path.join(BASE_DIR, "artifacts", "experiments")
DEMO_DIR = os.path.join(BASE_DIR, "data", "demo_kelas")
SPOOF_REAL_DIR = os.path.join(BASE_DIR, "data", "spoof_dataset", "real")
SPOOF_FAKE_DIR = os.path.join(BASE_DIR, "data", "spoof_dataset", "spoof")
LABEL_CONFIG_PATH = os.path.join(DB_DIR, "ui_labels.json")

LABEL_PRESETS = {
    "Umum / Netral": {
        "badge": "Presensi Wajah Umum",
        "subjek": "Peserta",
        "id_label": "ID Peserta",
        "group_label": "Kategori",
        "absent_label": "Belum Hadir",
        "session_default": "Sesi_01",
    },
    "Perkuliahan / Sekolah": {
        "badge": "Presensi Kelas Perkuliahan",
        "subjek": "Mahasiswa",
        "id_label": "NIM",
        "group_label": "Kelas",
        "absent_label": "Alpa",
        "session_default": "Pertemuan_14",
    },
    "Kantor / Pegawai": {
        "badge": "Presensi Pegawai & Kantor",
        "subjek": "Pegawai",
        "id_label": "NIP",
        "group_label": "Divisi",
        "absent_label": "Tidak Hadir",
        "session_default": "Absen_Harian",
    },
    "Rapat / Event / Organisasi": {
        "badge": "Presensi Rapat & Kegiatan",
        "subjek": "Anggota",
        "id_label": "Kode Anggota",
        "group_label": "Bagian / Tim",
        "absent_label": "Belum Hadir",
        "session_default": "Rapat_01",
    },
    "Kustom (Ketik Sendiri)": None,
}


st.set_page_config(
    page_title="Sistem Presensi Biometrik Wajah | PCD",
    layout="wide",
    initial_sidebar_state="auto",
)


def inject_custom_css():
    """Menyuntikkan styling editorial daylight terstruktur sesuai DESIGN.md."""
    st.markdown(
        """
        <style>
        :root {
            --bg-canvas: #F5F6F8;
            --bg-surface: #FFFFFF;
            --bg-subtle: #F8FAFC;
            --border-hairline: #D8DFE8;
            --ink-brand: #0F294A;
            --ink-primary: #111827;
            --ink-muted: #475569;
            --accent-action: #1D4ED8;
            --accent-action-hover: #1E40AF;
            --status-ok-fg: #047857;
            --status-ok-bg: #ECFDF5;
            --status-ok-border: #A7F3D0;
            --status-warn-fg: #92400E;
            --status-warn-bg: #FFFBEB;
            --status-warn-border: #FDE68A;
            --status-err-fg: #B91C1C;
            --status-err-bg: #FEF2F2;
            --status-err-border: #FECACA;
            --font-sans: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
            --font-mono: ui-monospace, 'Cascadia Mono', 'SFMono-Regular', Consolas, monospace;
        }

        html, body, [class*="css"] {
            font-family: var(--font-sans);
            color: var(--ink-primary);
        }

        .stApp {
            background-color: var(--bg-canvas);
        }

        /* Ruang kerja utama & kebersihan chrome default */
        .block-container {
            padding-top: 3rem !important;
            padding-bottom: 2.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 1380px !important;
        }
        #MainMenu, footer, .stAppDeployButton, [data-testid="stAppDeployButton"], [data-testid="stDeployButton"] {
            display: none !important;
            visibility: hidden;
        }
        header[data-testid="stHeader"] {
            background: var(--bg-canvas) !important;
            border-bottom: 1px solid var(--border-hairline);
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
            box-shadow: none !important;
        }

        /* Focus indicator yang jelas untuk aksesibilitas keyboard */
        button:focus-visible,
        [role="tab"]:focus-visible,
        input:focus-visible,
        select:focus-visible,
        textarea:focus-visible,
        a:focus-visible {
            outline: 2px solid var(--accent-action) !important;
            outline-offset: 2px !important;
        }

        /* Editorial Daylight Masthead */
        .app-masthead {
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            border-bottom: 2px solid var(--ink-brand);
            border-radius: 8px;
            padding: 1.05rem 1.35rem;
            margin-bottom: 1rem;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: flex-end;
            gap: 1rem 1.5rem;
        }
        .masthead-title {
            font-size: 1.55rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1.2 !important;
            color: var(--ink-brand) !important;
        }
        .masthead-subtitle {
            font-size: 0.88rem;
            color: var(--ink-muted);
            margin: 0.3rem 0 0 0;
            line-height: 1.45;
        }
        .masthead-telemetry {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.5rem 1rem;
            padding-top: 0.15rem;
        }
        .telemetry-item {
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
            font-size: 0.80rem;
            color: var(--ink-primary);
            padding-left: 0.85rem;
            border-left: 1px solid var(--border-hairline);
        }
        .telemetry-item:first-child {
            padding-left: 0;
            border-left: none;
        }
        .telemetry-label {
            color: var(--ink-muted);
            font-weight: 500;
        }
        .telemetry-val {
            font-family: var(--font-mono);
            font-variant-numeric: tabular-nums;
            font-weight: 600;
            color: var(--ink-brand);
        }
        .dot-status {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            display: inline-block;
            flex-shrink: 0;
        }
        .dot-green {
            background: var(--status-ok-fg);
        }
        .dot-amber {
            background: #B45309;
        }

        /* Tab Bar ber-garis bawah tajam */
        .stTabs [data-baseweb="tab-list"],
        .stTabs [role="tablist"] {
            gap: 0.25rem;
            background: var(--bg-surface);
            padding: 0.3rem 0.4rem 0 0.4rem;
            border-radius: 8px 8px 0 0;
            border: 1px solid var(--border-hairline);
            margin-bottom: 0.9rem;
        }
        .stTabs [data-baseweb="tab"],
        .stTabs [role="tab"] {
            border-radius: 6px 6px 0 0 !important;
            padding: 0.55rem 1rem !important;
            min-height: 44px !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            color: var(--ink-muted) !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            transition: background-color 0.15s ease, color 0.15s ease;
        }
        .stTabs [data-baseweb="tab"]:hover,
        .stTabs [role="tab"]:hover {
            background: var(--bg-subtle) !important;
            color: var(--ink-primary) !important;
        }
        .stTabs [aria-selected="true"],
        .stTabs [role="tab"][aria-selected="true"] {
            background: #EFF6FF !important;
            color: var(--accent-action) !important;
            font-weight: 700 !important;
            border-bottom: 2px solid var(--accent-action) !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
        }

        /* Tombol min-height konsisten */
        .stButton > button,
        .stDownloadButton > button {
            min-height: 44px;
            font-weight: 600;
            border-radius: 6px !important;
        }

        /* Metric KPI Cards standar Streamlit (tanpa strip warna kiri) */
        div[data-testid="stMetric"] {
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            border-radius: 8px;
            padding: 0.85rem 1rem !important;
        }
        div[data-testid="stMetric"] label {
            font-size: 0.76rem !important;
            font-weight: 600 !important;
            color: var(--ink-muted) !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink-brand) !important;
            font-variant-numeric: tabular-nums;
        }

        /* Kartu Kontainer Streamlit dengan Hairline Border */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--bg-surface) !important;
            border-radius: 8px !important;
            border: 1px solid var(--border-hairline) !important;
            box-shadow: none !important;
        }

        /* Dropzone Uploader */
        section[data-testid="stFileUploaderDropzone"] {
            background: var(--bg-subtle) !important;
            border: 1px dashed #94A3B8 !important;
            border-radius: 8px !important;
        }
        section[data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--accent-action) !important;
            background: #EFF6FF !important;
        }

        /* Sidebar Cards & LAN Box */
        .sidebar-box {
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.75rem;
        }
        .sidebar-box-title {
            font-size: 0.73rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--ink-muted);
            margin-bottom: 0.45rem;
        }
        .status-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.82rem;
            padding: 0.25rem 0;
            color: var(--ink-primary);
            border-bottom: 1px solid #F1F5F9;
        }
        .status-row:last-child {
            border-bottom: none;
        }
        .badge-ok {
            background: var(--status-ok-bg);
            color: var(--status-ok-fg);
            border: 1px solid var(--status-ok-border);
            padding: 0.12rem 0.5rem;
            border-radius: 4px;
            font-weight: 700;
            font-size: 0.72rem;
        }
        .badge-warn {
            background: var(--status-warn-bg);
            color: var(--status-warn-fg);
            border: 1px solid var(--status-warn-border);
            padding: 0.12rem 0.5rem;
            border-radius: 4px;
            font-weight: 700;
            font-size: 0.72rem;
        }
        .lan-url {
            font-family: var(--font-mono);
            font-size: 0.83rem;
            font-weight: 600;
            background: var(--ink-brand);
            color: #F8FAFC;
            padding: 0.5rem 0.7rem;
            border-radius: 6px;
            display: block;
            text-align: center;
            margin: 0.45rem 0;
            word-break: break-all;
        }

        /* Section Title, Info Strip & Empty Card */
        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--ink-brand);
            margin: 0 0 0.2rem 0;
        }
        .section-sub {
            font-size: 0.84rem;
            color: var(--ink-muted);
            margin: 0 0 0.75rem 0;
        }
        .strip-pill {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.5rem;
            background: var(--bg-subtle);
            border: 1px solid var(--border-hairline);
            color: var(--ink-primary);
            border-radius: 6px;
            padding: 0.55rem 0.85rem;
            font-size: 0.83rem;
            font-weight: 500;
            margin: 0.45rem 0 0.85rem 0;
        }
        .empty-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            border-radius: 8px;
            padding: 1.65rem 1.25rem;
            text-align: center;
            color: var(--ink-muted);
        }
        .empty-card-title {
            font-size: 0.96rem;
            font-weight: 700;
            color: var(--ink-brand);
            margin-bottom: 0.35rem;
        }

        /* Asymmetric Metric Strip (Hero Lead Readout + Compact Telemetry Readouts) */
        .kpi-grid {
            display: grid;
            width: 100%;
            box-sizing: border-box;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 0.65rem;
            margin: 0.45rem 0 0.85rem 0;
        }
        .kpi-grid--asymmetric {
            grid-template-columns: 1.45fr repeat(auto-fit, minmax(0, 1fr));
        }
        .kpi-grid--asymmetric.kpi-grid--n3 {
            grid-template-columns: 1.45fr 1fr 1fr;
        }
        .kpi-grid--asymmetric.kpi-grid--n4 {
            grid-template-columns: 1.45fr 1fr 1fr 1fr;
        }
        .kpi-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            border-radius: 8px;
            padding: 0.78rem 0.95rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 100%;
        }
        .kpi-card--lead {
            background: var(--bg-subtle);
            border-top: 2px solid var(--kpi-accent, var(--ink-brand));
        }
        .kpi-label {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--ink-muted);
            line-height: 1.3;
            margin-bottom: 0.25rem;
        }
        .kpi-value {
            font-family: var(--font-sans);
            font-variant-numeric: tabular-nums;
            font-size: clamp(1.05rem, 1.4vw, 1.32rem);
            font-weight: 700;
            color: var(--ink-brand);
            line-height: 1.25;
            white-space: normal;
            word-break: break-word;
        }
        .kpi-card--lead .kpi-value {
            font-size: clamp(1.18rem, 1.6vw, 1.48rem);
        }
        .kpi-sub {
            font-size: 0.76rem;
            font-weight: 500;
            color: var(--ink-muted);
            margin-top: 0.22rem;
        }

        /* Status Bar Sesi Aktif / Terkunci (Hairline Flat, Tanpa Gradient) */
        .session-open-bar {
            background: var(--status-ok-bg);
            border: 1px solid var(--status-ok-border);
            border-radius: 8px;
            padding: 0.8rem 1.05rem;
            color: #064E3B;
            margin-bottom: 0.85rem;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 0.6rem;
        }
        .session-locked-banner {
            background: var(--status-warn-bg);
            border: 1px solid var(--status-warn-border);
            border-radius: 8px;
            padding: 0.85rem 1.1rem;
            color: var(--status-warn-fg);
            margin-bottom: 0.85rem;
        }
        .sess-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 0.42rem;
        }
        .sess-chip {
            background: var(--bg-surface);
            border: 1px solid var(--status-ok-border);
            color: var(--status-ok-fg);
            border-radius: 4px;
            padding: 0.14rem 0.5rem;
            font-size: 0.75rem;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
        }
        .strip-chip {
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            color: var(--ink-brand);
            border-radius: 4px;
            padding: 0.16rem 0.5rem;
            font-size: 0.78rem;
            font-variant-numeric: tabular-nums;
        }
        .desktop-spacer {
            height: 1.5rem;
        }
        .desktop-spacer-sm {
            height: 0.35rem;
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                transition: none !important;
                animation: none !important;
            }
        }

        /* Responsif Mobile (<= 768px) */
        @media (max-width: 768px) {
            .block-container {
                padding-top: 3.35rem !important;
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
                padding-bottom: 2rem !important;
            }
            .desktop-spacer, .desktop-spacer-sm {
                display: none !important;
            }
            .app-masthead {
                padding: 0.85rem 0.95rem !important;
                gap: 0.6rem;
                margin-bottom: 0.75rem;
            }
            .masthead-title {
                font-size: 1.2rem !important;
            }
            .masthead-subtitle {
                font-size: 0.76rem;
                margin-top: 0.2rem;
            }
            .masthead-telemetry {
                gap: 0.35rem 0.65rem;
                width: 100%;
                border-top: 1px solid var(--border-hairline);
                padding-top: 0.45rem;
            }
            .telemetry-item {
                font-size: 0.72rem;
                padding-left: 0;
                border-left: none;
            }
            .stTabs [data-baseweb="tab-list"],
            .stTabs [role="tablist"] {
                display: grid !important;
                grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
                gap: 0.3rem !important;
                padding: 0.3rem !important;
                overflow: visible !important;
                border-radius: 8px !important;
            }
            .stTabs [data-baseweb="tab"],
            .stTabs [role="tab"] {
                padding: 0.5rem 0.4rem !important;
                min-height: 44px !important;
                height: auto !important;
                font-size: 0.82rem !important;
                justify-content: center !important;
                width: 100% !important;
                margin: 0 !important;
                white-space: nowrap !important;
                border-radius: 6px !important;
            }
            .kpi-grid,
            .kpi-grid--asymmetric,
            .kpi-grid--asymmetric.kpi-grid--n3,
            .kpi-grid--asymmetric.kpi-grid--n4 {
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.5rem;
            }
            .kpi-card--lead {
                grid-column: span 1;
            }
            .kpi-grid--asymmetric.kpi-grid--n3 .kpi-card--lead {
                grid-column: 1 / -1;
            }
            .kpi-card {
                padding: 0.65rem 0.75rem;
            }
            .kpi-label {
                font-size: 0.68rem;
            }
            .kpi-value {
                font-size: 1.02rem;
            }
            .kpi-sub {
                font-size: 0.70rem;
            }
            .strip-pill {
                font-size: 0.76rem;
                padding: 0.45rem 0.65rem;
                gap: 0.35rem;
            }
            .session-open-bar {
                padding: 0.75rem 0.85rem;
                margin-bottom: 0.45rem;
            }
            .session-locked-banner {
                padding: 0.75rem 0.85rem;
                font-size: 0.85rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_ui_labels():
    """Memuat konfigurasi label konteks UI tersimpan atau memakai preset netral."""
    default_cfg = {"preset": "Umum / Netral", **LABEL_PRESETS["Umum / Netral"]}
    if os.path.exists(LABEL_CONFIG_PATH):
        try:
            with open(LABEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            default_cfg.update({k: str(v) for k, v in saved.items() if k in default_cfg and str(v).strip()})
        except Exception:
            pass
    return default_cfg


def save_ui_labels(cfg):
    """Menyimpan konfigurasi label kustom agar konsisten lintas perangkat & refresh."""
    os.makedirs(DB_DIR, exist_ok=True)
    with open(LABEL_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def init_label_state():
    """Menginisialisasi session_state untuk label kustom dan sinkronisasi preset."""
    if "lbl_preset" not in st.session_state:
        cfg = load_ui_labels()
        st.session_state["lbl_preset"] = cfg["preset"] if cfg["preset"] in LABEL_PRESETS else "Umum / Netral"
        st.session_state["lbl_badge"] = cfg["badge"]
        st.session_state["lbl_subjek"] = cfg["subjek"]
        st.session_state["lbl_id"] = cfg["id_label"]
        st.session_state["lbl_group"] = cfg["group_label"]
        st.session_state["lbl_absent"] = cfg["absent_label"]
        st.session_state["lbl_session"] = cfg["session_default"]
        st.session_state["active_session_name"] = cfg["session_default"]


def _current_label_cfg():
    return {
        "preset": st.session_state.get("lbl_preset", "Umum / Netral"),
        "badge": st.session_state.get("lbl_badge", "Presensi Wajah Umum").strip() or "Presensi Wajah Umum",
        "subjek": st.session_state.get("lbl_subjek", "Peserta").strip() or "Peserta",
        "id_label": st.session_state.get("lbl_id", "ID Peserta").strip() or "ID Peserta",
        "group_label": st.session_state.get("lbl_group", "Kategori").strip() or "Kategori",
        "absent_label": st.session_state.get("lbl_absent", "Belum Hadir").strip() or "Belum Hadir",
        "session_default": st.session_state.get("lbl_session", "Sesi_01").strip() or "Sesi_01",
    }


def _on_preset_change():
    preset_name = st.session_state.get("lbl_preset", "Umum / Netral")
    values = LABEL_PRESETS.get(preset_name)
    if values is not None:
        st.session_state["lbl_badge"] = values["badge"]
        st.session_state["lbl_subjek"] = values["subjek"]
        st.session_state["lbl_id"] = values["id_label"]
        st.session_state["lbl_group"] = values["group_label"]
        st.session_state["lbl_absent"] = values["absent_label"]
        st.session_state["lbl_session"] = values["session_default"]
        st.session_state["active_session_name"] = values["session_default"]
    save_ui_labels(_current_label_cfg())


def _on_custom_label_change():
    st.session_state["lbl_preset"] = "Kustom (Ketik Sendiri)"
    save_ui_labels(_current_label_cfg())


def render_hero_banner(n_registered: int, antispoof_ready: bool, cfg: dict, spoof_tag: str = "Hybrid ONNX+SVM"):
    """Menampilkan masthead editorial aplikasi dan telemetri kesiapan pipeline."""
    spoof_dot = "dot-green" if antispoof_ready else "dot-amber"
    spoof_lbl = spoof_tag if antispoof_ready else "Heuristik"
    st.markdown(
        f"""
        <header class="app-masthead">
            <div>
                <h1 class="masthead-title">Sistem Presensi Biometrik Wajah</h1>
                <p class="masthead-subtitle">
                    Konteks {html.escape(cfg["badge"])} · Adaptive CLAHE, YuNet 5-Landmark, SFace 128-D, Anti-Spoofing Hybrid
                </p>
            </div>
            <div class="masthead-telemetry">
                <div class="telemetry-item">
                    <span class="dot-status dot-green"></span>
                    <span class="telemetry-label">Detektor</span>
                    <span class="telemetry-val">YuNet + SFace</span>
                </div>
                <div class="telemetry-item">
                    <span class="dot-status {spoof_dot}"></span>
                    <span class="telemetry-label">Liveness</span>
                    <span class="telemetry-val">{html.escape(spoof_lbl)}</span>
                </div>
                <div class="telemetry-item">
                    <span class="telemetry-label">Terdaftar</span>
                    <span class="telemetry-val">{n_registered} {html.escape(cfg["subjek"])}</span>
                </div>
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards_html(items):
    """Membangun markup HTML readout metrik asimetris tanpa pemotongan elipsis."""
    cards_html = []
    n_items = len(items)
    grid_cls = f"kpi-grid kpi-grid--asymmetric kpi-grid--n{n_items}" if n_items >= 3 else "kpi-grid"
    for idx, it in enumerate(items):
        accent = html.escape(str(it.get("accent", "#0F294A")))
        label = html.escape(str(it["label"]))
        value = html.escape(str(it["value"]))
        sub = it.get("sub")
        sub_html = f'<div class="kpi-sub">{html.escape(str(sub))}</div>' if sub else ""
        card_cls = "kpi-card kpi-card--lead" if (idx == 0 and len(items) >= 3) else "kpi-card"
        cards_html.append(
            f'<div class="{card_cls}" style="--kpi-accent:{accent};">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'{sub_html}'
            f'</div>'
        )
    return f'<div class="{grid_cls}">{"".join(cards_html)}</div>'


def render_kpi_cards(items):
    """Merender readout metrik responsif tanpa pemotongan elipsis pada label maupun nilai."""
    st.markdown(render_kpi_cards_html(items), unsafe_allow_html=True)


def get_local_ip():
    """Mendapatkan alamat IP lokal agar HP di hotspot yang sama bisa membuka aplikasi."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def decode_uploaded_image(uploaded_file):
    """Membaca foto unggahan/kamera termasuk memperbaiki rotasi EXIF kamera HP."""
    uploaded_file.seek(0)
    pil_img = Image.open(uploaded_file)
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    rgb_arr = np.array(pil_img, dtype=np.uint8)
    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    return bgr_arr


def _get_model_cache_key():
    """Menghasilkan kunci cache berdasarkan mtime file model agar pembaruan model langsung termuat otomatis."""
    names = [
        "face_detection_yunet_2023mar.onnx",
        "face_recognition_sface_2021dec.onnx",
        "anti_spoof_svm.joblib",
        "scaler_lbp.joblib",
        "2.7_80x80_MiniFASNetV2.onnx",
        "4_0_0_80x80_MiniFASNetV1SE.onnx",
    ]
    stamps = []
    for n in names:
        p = os.path.join(MODELS_DIR, n)
        stamps.append(round(os.path.getmtime(p), 2) if os.path.exists(p) else 0.0)
    return tuple(stamps)


@st.cache_resource
def load_pipeline(cache_key):
    _ = cache_key
    yunet_p = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
    sface_p = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
    svm_p = os.path.join(MODELS_DIR, "anti_spoof_svm.joblib")
    scaler_p = os.path.join(MODELS_DIR, "scaler_lbp.joblib")
    fas27_p = os.path.join(MODELS_DIR, "2.7_80x80_MiniFASNetV2.onnx")
    fas40_p = os.path.join(MODELS_DIR, "4_0_0_80x80_MiniFASNetV1SE.onnx")

    engine = FaceEngine(yunet_p, sface_p, score_threshold=0.70)
    antispoof = AntiSpoofChecker(
        svm_p, scaler_p, prob_threshold=0.50, fas_v2_path=fas27_p, fas_v1se_path=fas40_p
    )
    return engine, antispoof


def get_db():
    return StudentDatabase(DB_DIR, LOGS_DIR)


def draw_result_annotations(img_bgr, face_results):
    """Menggambar bounding box modern proporsional (Hijau=Hadir, Merah=Spoof, Kuning=Tidak Terdaftar)."""
    vis = img_bgr.copy()
    h_img, w_img = vis.shape[:2]
    is_crowd = len(face_results) > 8
    scale_base = max(h_img, w_img) / 900.0
    box_thick = max(2, int(round((1.8 if is_crowd else 2.4) * scale_base)))
    font_scale = 0.36 if is_crowd else max(0.45, min(0.75, 0.55 * scale_base))
    font_thick = 1 if is_crowd else max(1, int(round(1.6 * scale_base)))

    for item in face_results:
        x, y, w, h = item["bbox"]
        status = item["status"]
        if status == "Hadir":
            color = (129, 185, 16)   # Emerald BGR (#10B981)
            label = f"{item['nama']} ({item['score']:.2f})"
        elif status == "Sudah Absen":
            color = (199, 132, 2)    # Sky Blue BGR (#0284C7) — Batas 1x per sesi
            label = "Sudah" if is_crowd else f"SUDAH ABSEN: {item['nama']}"
        elif status == "Spoof":
            color = (68, 68, 239)    # Rose/Red BGR (#EF4444)
            atk = item.get("attack_type", "SPOOF").upper()
            label = "SPOOF" if is_crowd else f"SPOOF: {atk} ({item['prob_real']:.2f})"
        else:
            color = (11, 158, 245)   # Amber BGR (#F59E0B)
            label = "Asing" if is_crowd else f"Tidak Terdaftar ({item['score']:.2f})"

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, box_thick, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        y_top = max(0, y - th - 12)
        cv2.rectangle(vis, (x, y_top), (min(w_img, x + tw + 10), y), color, -1)
        text_col = (255, 255, 255) if status != "Tidak Terdaftar" else (15, 23, 42)
        cv2.putText(
            vis,
            label,
            (x + 5, max(th + 3, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_col,
            font_thick,
            cv2.LINE_AA,
        )
    return vis


def render_histogram_l(raw_bgr, enh_bgr):
    """Membuat grafik modern distribusi luminansi CIELAB L* sebelum vs sesudah enhancement."""
    l_raw = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    l_enh = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    mean_raw = float(l_raw.mean())
    mean_enh = float(l_enh.mean())
    is_same = bool(np.array_equal(l_raw, l_enh))

    fig, ax = plt.subplots(figsize=(6.6, 2.9), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    bins = np.linspace(0, 255, 49)
    # Isi lembut + garis tegas agar kedua kurva tetap terlihat jelas meski saat berimpit
    ax.hist(l_enh.ravel(), bins=bins, alpha=0.35, color="#10B981", zorder=2)
    ax.hist(
        l_enh.ravel(),
        bins=bins,
        histtype="step",
        color="#059669",
        linewidth=1.9,
        label=f"Setelah Adaptive CLAHE (L*={mean_enh:.1f})",
        zorder=3,
    )
    ax.hist(l_raw.ravel(), bins=bins, alpha=0.28, color="#F43F5E", zorder=2)
    ax.hist(
        l_raw.ravel(),
        bins=bins,
        histtype="step",
        color="#E11D48",
        linewidth=2.0,
        linestyle="--",
        label=f"Asli / Raw (L*={mean_raw:.1f})",
        zorder=4,
    )

    ax.axvline(mean_raw, color="#E11D48", linestyle=":", linewidth=1.4, alpha=0.85, zorder=5)
    if not is_same:
        ax.axvline(mean_enh, color="#059669", linestyle="-.", linewidth=1.4, alpha=0.85, zorder=5)

    title_txt = (
        "Distribusi Luminansi L* (Cahaya Normal: Asli & Sesudah Berimpit Identik)"
        if is_same
        else "Distribusi Luminansi L* Sebelum vs Sesudah Adaptive CLAHE"
    )
    ax.set_xlabel("Intensitas Luminansi L* (0 – 255)", fontsize=8.5, color="#334155")
    ax.set_ylabel("Jumlah Piksel", fontsize=8.5, color="#334155")
    ax.set_title(title_txt, fontsize=9.2, fontweight="bold", color="#0F172A")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.35, color="#CBD5E1")
    ax.legend(fontsize=7.8, frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", loc="upper right")
    plt.tight_layout()
    return fig


def render_lbp_histogram(lbp_hist):
    """Membuat grafik batang modern histogram LBP uniform 10 bin."""
    fig, ax = plt.subplots(figsize=(5.2, 2.7), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    bins = np.arange(10)
    ax.bar(bins, lbp_hist, color="#2563EB", edgecolor="#1E40AF", linewidth=0.5, alpha=0.88)
    ax.set_xticks(bins)
    ax.set_xticklabels([f"B{i}" for i in bins], fontsize=7.5, color="#334155")
    ax.set_ylabel("Proporsi Piksel", fontsize=8.5, color="#334155")
    ax.set_title("Distribusi Kode Uniform LBP (10 Bin)", fontsize=9.5, fontweight="bold", color="#0F172A")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35, color="#CBD5E1")
    plt.tight_layout()
    return fig


def list_demo_samples():
    """Mengembalikan daftar file sampel uji bawaan di data/demo_kelas/."""
    if not os.path.exists(DEMO_DIR):
        return []
    valid = (".jpg", ".jpeg", ".png")
    return sorted([f for f in os.listdir(DEMO_DIR) if f.lower().endswith(valid)])


def process_attendance_frame(
    raw_bgr,
    engine,
    antispoof,
    db_embs,
    db_nims,
    db_names,
    use_clahe,
    use_spoof,
    cos_thr,
    det_thr,
    spoof_thr=0.45,
    already_present_ids=None,
):
    """Menjalankan pipeline PCD lengkap (Enhancement -> Deteksi YuNet -> Anti-Spoofing Hybrid -> Matching SFace)."""
    raw_bgr, _ = resize_max_side(raw_bgr, max_side=1280)
    locked_ids = set(already_present_ids) if already_present_ids else set()

    if use_clahe:
        proc_bgr, enh_info = apply_adaptive_enhancement(raw_bgr)
    else:
        st_raw = analyze_lighting(raw_bgr)
        proc_bgr = raw_bgr.copy()
        enh_info = {
            "applied": False,
            "before_l": round(st_raw["mean_l"], 2),
            "after_l": round(st_raw["mean_l"], 2),
            "gamma_used": 1.0,
            "status_before": st_raw["status"],
            "reason": "Adaptive Enhancement dinonaktifkan manual.",
        }

    faces = engine.detect(proc_bgr, score_threshold=det_thr)
    face_results = []
    recognized_records = []
    first_debug = None

    for idx, fdict in enumerate(faces):
        x, y, w, h = fdict["bbox"]
        face_roi_raw = raw_bgr[y:y + h, x:x + w]
        aligned_face, emb = engine.align_and_embed(proc_bgr, fdict["raw_row"])

        _, prob_real, lbp_map, spoof_diag = antispoof.predict(
            face_roi_raw, full_bgr=raw_bgr, bbox=fdict["bbox"]
        )
        is_real = bool(prob_real >= spoof_thr)
        if use_spoof and not is_real:
            face_results.append({
                "bbox": fdict["bbox"],
                "status": "Spoof",
                "nim": "-",
                "nama": "SPOOF",
                "score": 0.0,
                "prob_real": prob_real,
                "attack_type": spoof_diag.get("attack_type", "Spoof"),
            })
        else:
            matched, nim, nama, sim_score = engine.match_identity(
                emb, db_embs, db_nims, db_names, threshold=cos_thr
            )
            if matched and nim in locked_ids:
                status = "Sudah Absen"
            elif matched:
                status = "Hadir"
            else:
                status = "Tidak Terdaftar"

            rec_item = {
                "bbox": fdict["bbox"],
                "status": status,
                "nim": nim if matched else "-",
                "nama": nama,
                "score": sim_score,
                "prob_real": prob_real,
            }
            face_results.append(rec_item)
            if matched and status == "Hadir":
                recognized_records.append(rec_item)

        if idx == 0:
            first_debug = {
                "raw_bgr": raw_bgr,
                "proc_bgr": proc_bgr,
                "enh_info": enh_info,
                "aligned_face": aligned_face,
                "landmarks": fdict["landmarks"],
                "bbox": fdict["bbox"],
                "lbp_map": lbp_map,
                "spoof_diag": spoof_diag,
                "prob_real": prob_real,
                "is_real": is_real,
            }

    if first_debug is None:
        first_debug = {
            "raw_bgr": raw_bgr,
            "proc_bgr": proc_bgr,
            "enh_info": enh_info,
            "aligned_face": None,
            "landmarks": [],
            "bbox": None,
            "lbp_map": None,
            "spoof_diag": {},
            "prob_real": 0.0,
            "is_real": False,
        }

    return proc_bgr, enh_info, face_results, recognized_records, first_debug


def reset_live_state(sess):
    """Mereset state pemindaian Webcam Live tanpa menghapus daftar hadir sesi aktif."""
    sess["live_streak_map"] = {}
    sess["live_last_seq"] = 0
    sess["live_last_spoof_ts"] = 0.0
    sess["live_boxes"] = []
    sess["live_light_badge"] = ""
    sess["live_event"] = (
        "info",
        "Webcam aktif. Arahkan wajah ke kamera untuk presensi otomatis (batas 1× per orang).",
    )
    sess["live_hud_text"] = ""
    sess["live_hud_kind"] = "info"
    sess["live_kpi"] = {"faces": 0, "sudah": 0, "spoof": 0, "unk": 0}
    sess["live_just_recorded"] = False
    sess["live_toast_msg"] = ""


def init_attendance_session(default_name="Sesi_01"):
    """Menginisialisasi state manajemen sesi presensi (Tutup -> Buka -> Akumulasi -> Tutup & Arsip)."""
    if "att_session" not in st.session_state:
        st.session_state["att_session"] = {
            "is_open": False,
            "session_name": default_name,
            "opened_at": None,
            "expires_at": None,
            "duration_min": 15,
            "present_map": {},        # {id: {"nim": id, "nama": ..., "score": ..., "time": ...}} -> maks 1x/sesi
            "manual_status_map": {},  # {id: {"status": "Hadir (Manual)"|"Izin"|"Sakit", "score": 0.0, "time": ...}}
            "spoof_logs": [],         # [{"Waktu": ..., "Skor Real": ..., "Status": "Diblokir (...)"}]
            "captures_count": 0,
            "last_frame_sig": None,
            "last_frame_new_ids": set(),
            "last_closed_name": None,
            "last_closed_df": None,
            "last_closed_file": None,
            "last_closed_spoofs": 0,
            "force_stop_live": False,
        }
        reset_live_state(st.session_state["att_session"])
    sess = st.session_state["att_session"]
    sess.setdefault("manual_status_map", {})
    sess.setdefault("last_frame_new_ids", set())
    sess.setdefault("force_stop_live", False)
    if "live_streak_map" not in sess:
        reset_live_state(sess)
    return sess


def count_session_present(sess):
    """Menghitung total ID unik berstatus Hadir (otomatis + koreksi manual) pada sesi aktif."""
    ids = set(sess["present_map"].keys())
    for rec_id, info in sess.get("manual_status_map", {}).items():
        if str(info.get("status", "")).startswith("Hadir"):
            ids.add(rec_id)
        else:
            ids.discard(rec_id)
    return len(ids)


def record_frame_into_session(sess, frame_sig, recognized_records, face_results, force=False):
    """Mencatat hasil deteksi foto ke sesi aktif dengan batas ketat maksimal 1x hadir per orang per sesi."""
    if not force and sess.get("last_frame_sig") == frame_sig:
        return []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sess["last_frame_sig"] = frame_sig
    sess["captures_count"] = int(sess.get("captures_count", 0)) + 1

    newly_added_ids = set()
    newly_added_names = []
    for rec in recognized_records:
        rec_id = rec.get("nim")
        if not rec_id or rec_id == "-":
            continue
        # Batasan 1x absen per orang per sesi: jika sudah hadir atau sudah ada status manual, jangan catat ulang
        if rec_id in sess["present_map"] or rec_id in sess.get("manual_status_map", {}):
            continue
        score = float(rec.get("score", 0.0))
        sess["present_map"][rec_id] = {
            "nim": rec_id,
            "nama": rec.get("nama", ""),
            "score": score,
            "time": now_str,
        }
        newly_added_ids.add(rec_id)
        newly_added_names.append(rec.get("nama", rec_id))

    sess["last_frame_new_ids"] = newly_added_ids

    for r in face_results:
        if r.get("status") == "Spoof":
            atk = r.get("attack_type", "Layar / Spoof")
            sess["spoof_logs"].append({
                "Waktu": now_str,
                "Skor Real": round(float(r.get("prob_real", 0.0)), 3),
                "Status": f"Diblokir ({atk})",
            })
    return newly_added_names


def handle_live_webcam_frame(
    payload,
    sess,
    engine,
    antispoof,
    db_embs,
    db_nims,
    db_names,
    use_clahe,
    use_spoof,
    cos_thr,
    det_thr,
    spoof_thr,
    confirm_frames=2,
    subjek_lbl="Peserta",
    save_debug=True,
):
    """Memproses 1 frame dari browser Webcam Live dengan batas ketat 1x hadir per orang per sesi."""
    if not isinstance(payload, dict):
        return []
    seq = int(payload.get("seq", 0))
    if seq <= 0 or seq == int(sess.get("live_last_seq", 0)):
        return []

    raw_bgr = decode_data_url_to_bgr(payload.get("data_url", ""))
    sess["live_last_seq"] = seq
    if raw_bgr is None:
        return []

    locked_ids = set(sess["present_map"].keys()) | set(sess.get("manual_status_map", {}).keys())
    proc_bgr, enh_info, face_results, recognized_records, first_debug = process_attendance_frame(
        raw_bgr,
        engine,
        antispoof,
        db_embs,
        db_nims,
        db_names,
        use_clahe,
        use_spoof,
        cos_thr,
        det_thr,
        spoof_thr,
        already_present_ids=locked_ids,
    )

    if save_debug and face_results:
        st.session_state["last_debug"] = first_debug

    streak_map = sess.setdefault("live_streak_map", {})
    seen_new_ids = set()
    newly_recorded = []
    confirm_n = max(1, int(confirm_frames))

    for rec in recognized_records:
        rid = rec.get("nim")
        if not rid or rid == "-" or rid in locked_ids:
            continue
        seen_new_ids.add(rid)
        streak_map[rid] = int(streak_map.get(rid, 0)) + 1
        if streak_map[rid] >= confirm_n:
            now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sess["present_map"][rid] = {
                "nim": rid,
                "nama": rec.get("nama", ""),
                "score": float(rec.get("score", 0.0)),
                "time": now_s,
            }
            sess["captures_count"] = int(sess.get("captures_count", 0)) + 1
            locked_ids.add(rid)
            streak_map.pop(rid, None)
            newly_recorded.append((rid, rec.get("nama", rid), now_s))

    # Reset streak untuk wajah yang keluar dari frame
    for old_id in list(streak_map.keys()):
        if old_id not in seen_new_ids:
            streak_map.pop(old_id, None)

    already_seen = [r["nama"] for r in face_results if r["status"] == "Sudah Absen"]
    spoof_hits = [r for r in face_results if r["status"] == "Spoof"]
    n_sudah = len(already_seen)
    n_spoof_lv = len(spoof_hits)
    n_unk_lv = sum(1 for r in face_results if r["status"] == "Tidak Terdaftar")

    sess["live_kpi"] = {
        "faces": len(face_results),
        "sudah": n_sudah,
        "spoof": n_spoof_lv,
        "unk": n_unk_lv,
    }
    sess["live_boxes"] = build_overlay_boxes(
        face_results,
        proc_bgr.shape,
        streak_map=streak_map,
        confirm_frames=confirm_n,
    )

    if enh_info.get("applied"):
        sess["live_light_badge"] = f"CLAHE · L* {enh_info['before_l']}→{enh_info['after_l']}"
    else:
        sess["live_light_badge"] = f"Normal · L* {enh_info['before_l']}"

    if newly_recorded:
        rid, rnama, rtime = newly_recorded[0]
        msg = f"**{rnama}** (`{rid}`) otomatis tercatat **Hadir** pukul `{rtime[-8:]}` (terkunci maks. 1×/sesi)."
        sess["live_event"] = ("success", msg)
        sess["live_hud_text"] = f"{rnama} tercatat Hadir ({rtime[-8:]}) · Batas 1×/sesi"
        sess["live_hud_kind"] = "success"
        sess["live_just_recorded"] = True
        sess["live_toast_msg"] = f"{rnama} otomatis tercatat Hadir ({rtime[-8:]})"
    elif spoof_hits:
        atk_name = spoof_hits[0].get("attack_type", "Layar HP")
        pr = float(spoof_hits[0].get("prob_real", 0.0))
        sess["live_event"] = (
            "error",
            f"**SPOOF ({atk_name})** terdeteksi dan diblokir (Skor Real: `{pr:.2f}`).",
        )
        sess["live_hud_text"] = f"SPOOF ({atk_name}) diblokir · Real {pr:.2f}"
        sess["live_hud_kind"] = "error"
        now_mono = time.monotonic()
        if now_mono - float(sess.get("live_last_spoof_ts", 0.0)) >= 4.0:
            sess["live_last_spoof_ts"] = now_mono
            sess["spoof_logs"].append({
                "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Skor Real": round(pr, 3),
                "Status": f"Diblokir ({atk_name})",
            })
    elif already_seen:
        names_str = ", ".join(already_seen)
        sess["live_event"] = (
            "info",
            f"**{names_str}** sudah tercatat hadir pada sesi ini (batasan maks. 1× per orang).",
        )
        sess["live_hud_text"] = f"{names_str} sudah absen (maks. 1×/sesi)"
        sess["live_hud_kind"] = "info"
    elif len(db_nims) > 0 and count_session_present(sess) >= len(db_nims):
        sess["live_event"] = (
            "success",
            f"Seluruh {subjek_lbl.lower()} terdaftar (**{len(db_nims)}/{len(db_nims)}**) sudah tercatat Hadir. Matikan webcam lalu tutup sesi.",
        )
        sess["live_hud_text"] = f"Semua {subjek_lbl.lower()} ({len(db_nims)}/{len(db_nims)}) sudah Hadir"
        sess["live_hud_kind"] = "success"
    elif seen_new_ids:
        first_rid = next(iter(seen_new_ids))
        cur_s = int(streak_map.get(first_rid, 1))
        sess["live_event"] = (
            "info",
            f"Wajah dikenali. Tahan sebentar untuk konfirmasi otomatis ({cur_s}/{confirm_n})...",
        )
        sess["live_hud_text"] = f"Memverifikasi wajah ({cur_s}/{confirm_n})..."
        sess["live_hud_kind"] = "info"
    elif n_unk_lv > 0:
        sess["live_event"] = (
            "warning",
            f"Wajah terdeteksi tetapi **belum terdaftar** di database {subjek_lbl.lower()}.",
        )
        sess["live_hud_text"] = "Wajah belum terdaftar di database"
        sess["live_hud_kind"] = "warn"
    else:
        sess["live_hud_text"] = ""

    return newly_recorded


def close_attendance_session(sess, db, id_lbl, group_lbl, absent_lbl):
    """Menutup sesi aktif, membangun rekap final, dan menyimpan ke SQLite presensi.db + arsip CSV."""
    sess["force_stop_live"] = True
    reset_live_state(sess)
    records_list = list(sess["present_map"].values())
    df_final = db.build_attendance_table(
        records_list,
        id_col=id_lbl,
        label_col=group_lbl,
        absent_label=absent_lbl,
        manual_status_map=sess.get("manual_status_map", {}),
    )
    saved_path = None
    csv_name = ""
    closed_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    opened_str = sess["opened_at"].strftime("%Y-%m-%d %H:%M:%S") if sess.get("opened_at") else closed_str

    if not df_final.empty:
        saved_path = db.save_attendance_log(df_final, session_name=sess["session_name"])
        csv_name = os.path.basename(saved_path)

    db.save_session_to_db(
        session_name=sess["session_name"],
        opened_at=opened_str,
        closed_at=closed_str,
        attendance_df=df_final,
        spoof_blocked=len(sess["spoof_logs"]),
        csv_filename=csv_name,
    )

    sess["last_closed_name"] = sess["session_name"]
    sess["last_closed_df"] = df_final
    sess["last_closed_file"] = csv_name or None
    sess["last_closed_spoofs"] = len(sess["spoof_logs"])
    sess["is_open"] = False
    sess["opened_at"] = None
    sess["expires_at"] = None
    sess["last_frame_sig"] = None
    sess["last_frame_new_ids"] = set()
    sess["manual_status_map"] = {}


def list_saved_logs():
    """Mengembalikan daftar file CSV log presensi tersimpan (terbaru di atas)."""
    if not os.path.exists(LOGS_DIR):
        return []
    files = [f for f in os.listdir(LOGS_DIR) if f.lower().endswith(".csv")]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(LOGS_DIR, f)), reverse=True)
    return files


def main():
    inject_custom_css()
    init_label_state()
    cfg = _current_label_cfg()
    sess = init_attendance_session(default_name=cfg["session_default"])

    try:
        engine, antispoof = load_pipeline(_get_model_cache_key())
    except Exception as exc:
        st.error(f"Gagal memuat model ONNX: {exc}\nJalankan skrip `scripts/01_download_models.py` terlebih dahulu.")
        return

    db = get_db()
    db_embs, db_nims, db_names = db.load()

    # =========================================================================
    # SIDEBAR: STATUS PIPELINE, PARAMETER PCD, LABEL KUSTOM, & AKSES BROWSER HP
    # =========================================================================
    with st.sidebar:
        st.markdown("### Panel Kontrol & Kalibrasi")
        st.caption("Atur konteks presensi, label kustom, parameter deteksi, dan anti-spoofing secara langsung.")

        if antispoof.fas_models and antispoof.svm is not None:
            spoof_badge = '<span class="badge-ok">Hybrid ONNX+SVM</span>'
        elif antispoof.fas_models:
            spoof_badge = '<span class="badge-ok">MiniFASNet ONNX</span>'
        elif antispoof.svm is not None:
            spoof_badge = '<span class="badge-ok">SVM Aktif</span>'
        else:
            spoof_badge = '<span class="badge-warn">Heuristik</span>'
        st.markdown(
            f"""
            <div class="sidebar-box">
                <div class="sidebar-box-title">Status Pipeline ONNX</div>
                <div class="status-row"><span>YuNet + SFace</span><span class="badge-ok">Siap</span></div>
                <div class="status-row"><span>Anti-Spoofing</span>{spoof_badge}</div>
                <div class="status-row"><span>Terdaftar</span><strong>{len(db_nims)} {html.escape(cfg["subjek"].lower())}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown("**Konteks & Label Kustom**")
            st.selectbox(
                "Mode Presensi",
                options=list(LABEL_PRESETS.keys()),
                key="lbl_preset",
                on_change=_on_preset_change,
                help="Pilih konteks cepat atau ubah langsung label di bawah sesuai kebutuhan.",
            )
            with st.expander("Input Label Tersendiri"):
                st.text_input("Judul / Konteks Banner", key="lbl_badge", on_change=_on_custom_label_change)
                st.text_input("Label Subjek (Orang)", key="lbl_subjek", on_change=_on_custom_label_change)
                st.text_input("Label Nomor ID", key="lbl_id", on_change=_on_custom_label_change)
                st.text_input("Label Grup / Kategori", key="lbl_group", on_change=_on_custom_label_change)
                st.text_input("Label Belum Hadir", key="lbl_absent", on_change=_on_custom_label_change)
                st.text_input("Nama Sesi Bawaan", key="lbl_session", on_change=_on_custom_label_change)

        cfg = _current_label_cfg()

        with st.container(border=True):
            st.markdown("**Parameter Algoritma PCD**")
            use_clahe = st.toggle("Adaptive Low-Light CLAHE", value=True, help="Otomatis memperbaiki ruang gelap/backlight di kanal CIELAB L*.")
            use_spoof = st.toggle("Filter Anti-Spoofing LBP", value=True, help="Menolak foto wajah dari layar HP/kertas.")
            cos_thr = st.slider("Ambang Cosine SFace", 0.25, 0.65, float(COSINE_THRESHOLD_DEFAULT), 0.01)
            det_thr = st.slider("Ambang Keyakinan YuNet", 0.45, 0.90, 0.70, 0.05)
            spoof_thr = st.slider(
                "Ambang Anti-Spoofing (Real)",
                0.15,
                0.85,
                0.45,
                0.05,
                help="Skor minimal agar wajah dianggap ASLI. Turunkan jika wajah nyata di kamera Anda masih dianggap SPOOF.",
            )

        local_ip = get_local_ip()
        st.markdown(
            f"""
            <div class="sidebar-box">
                <div class="sidebar-box-title">Akses Browser HP Operator</div>
                <div style="font-size:0.79rem; color:#475569; line-height:1.4;">
                    Hubungkan HP operator/kiosk ke Wi-Fi/hotspot yang sama dengan laptop, lalu buka URL ini (tanpa APK):
                </div>
                <span class="lan-url">http://{local_ip}:8501</span>
                <div style="font-size:0.74rem; color:#475569;">
                    Kamera terkunci di sisi Operator/Kiosk untuk mencegah presensi palsu jarak jauh.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    spoof_tag = "Hybrid ONNX+SVM" if (antispoof.fas_models and antispoof.svm is not None) else ("MiniFASNet ONNX" if antispoof.fas_models else "SVM Aktif")
    render_hero_banner(
        n_registered=len(db_nims),
        antispoof_ready=(bool(antispoof.fas_models) or antispoof.svm is not None),
        cfg=cfg,
        spoof_tag=spoof_tag,
    )

    # =========================================================================
    # NAVIGASI 4 TAB UTAMA
    # =========================================================================
    subjek_lbl = cfg["subjek"]
    id_lbl = cfg["id_label"]
    group_lbl = cfg["group_label"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "Presensi",
        f"{subjek_lbl}",
        "Lab PCD",
        "Evaluasi",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: AMBIL PRESENSI (ALUR BUKA SESI -> AKUMULASI -> TUTUP SESI)
    # -------------------------------------------------------------------------
    with tab1:
        # Cek otomatis apakah durasi sesi aktif telah habis
        if sess["is_open"] and sess.get("expires_at") is not None:
            if datetime.now() >= sess["expires_at"]:
                close_attendance_session(sess, db, id_lbl, group_lbl, cfg["absent_label"])
                st.warning(f"Waktu sesi **{sess['last_closed_name']}** telah habis. Sesi ditutup otomatis dan rekap disimpan.")

        if not sess["is_open"]:
            # =================================================================
            # KEADAAN A: SESI BELUM DIBUKA (KAMERA & UPLOAD TERKUNCI)
            # =================================================================
            st.markdown(
                f"""
                <div class="session-locked-banner">
                    <strong>Status: Sesi Presensi Belum Dibuka</strong><br>
                    <span style="font-size:0.86rem;">
                        Kamera dan pengunggahan foto dikunci untuk mencegah presensi di luar jadwal.
                        Operator harus membuka sesi terlebih dahulu sebelum merekam kehadiran {html.escape(subjek_lbl.lower())}.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.container(border=True):
                st.markdown('<p class="section-title">Buka Sesi Presensi Baru</p>', unsafe_allow_html=True)
                st.markdown('<p class="section-sub">Tentukan nama sesi dan batas waktu presensi, lalu klik tombol buka sesi.</p>', unsafe_allow_html=True)

                c_s1, c_s2, c_s3 = st.columns([1.4, 1.0, 1.0])
                with c_s1:
                    new_sess_name = st.text_input("Nama Sesi / Kegiatan", key="active_session_name")
                with c_s2:
                    dur_map = {
                        "Tanpa Batas Waktu": 0,
                        "10 Menit": 10,
                        "15 Menit": 15,
                        "30 Menit": 30,
                        "60 Menit": 60,
                    }
                    dur_choice = st.selectbox("Durasi Presensi", options=list(dur_map.keys()), index=2)
                with c_s3:
                    st.markdown("<div class='desktop-spacer'></div>", unsafe_allow_html=True)
                    if st.button("Buka Sesi Presensi", type="primary", width="stretch"):
                        clean_name = (new_sess_name or cfg["session_default"]).strip() or "Sesi_01"
                        now_dt = datetime.now()
                        mins = dur_map[dur_choice]
                        sess["is_open"] = True
                        sess["session_name"] = clean_name
                        sess["opened_at"] = now_dt
                        sess["duration_min"] = mins
                        sess["expires_at"] = (now_dt + timedelta(minutes=mins)) if mins > 0 else None
                        sess["present_map"] = {}
                        sess["manual_status_map"] = {}
                        sess["spoof_logs"] = []
                        sess["captures_count"] = 0
                        sess["last_frame_sig"] = None
                        sess["last_frame_new_ids"] = set()
                        sess["force_stop_live"] = True
                        reset_live_state(sess)
                        st.rerun()

            if sess.get("last_closed_df") is not None and not sess["last_closed_df"].empty:
                with st.container(border=True):
                    st.markdown(f'<p class="section-title">Rekap Final Sesi Terakhir: {html.escape(str(sess["last_closed_name"]))}</p>', unsafe_allow_html=True)
                    sub_info = (
                        f"Tersimpan di **SQLite (`presensi.db`)** & CSV (`{sess['last_closed_file']}`)"
                        if sess.get("last_closed_file")
                        else "Tersimpan di **SQLite (`presensi.db`)**."
                    )
                    if sess.get("last_closed_spoofs", 0) > 0:
                        sub_info += f" · {sess['last_closed_spoofs']} percobaan spoof diblokir"
                    st.caption(sub_info)
                    st.dataframe(sess["last_closed_df"], width="stretch", hide_index=True)
                    dl1, dl2 = st.columns(2)
                    with dl1:
                        csv_final = sess["last_closed_df"].to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Unduh Rekap CSV",
                            data=csv_final,
                            file_name=sess.get("last_closed_file") or f"presensi_{sess['last_closed_name']}.csv",
                            mime="text/csv",
                            width="stretch",
                        )
                    with dl2:
                        xlsx_final = db.export_excel_bytes(sess["last_closed_df"], sheet_name=sess["last_closed_name"])
                        st.download_button(
                            "Unduh Rekap Excel (.xlsx)",
                            data=xlsx_final,
                            file_name=f"presensi_{sess['last_closed_name']}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            width="stretch",
                        )

            df_sessions = db.list_sessions_db()
            saved_csvs = list_saved_logs()
            tracked_csvs = set(df_sessions["File CSV"].dropna().astype(str).tolist()) if not df_sessions.empty else set()
            orphan_csvs = [f for f in saved_csvs if f not in tracked_csvs]
            total_archives = len(df_sessions) + len(orphan_csvs)

            with st.expander(f"Riwayat & Kelola Presensi ({total_archives} arsip)", expanded=False):
                if total_archives == 0:
                    st.caption("Belum ada arsip sesi presensi yang tersimpan.")
                else:
                    if not df_sessions.empty:
                        st.markdown("**Arsip Database SQLite (`presensi.db`)**")
                        st.dataframe(df_sessions, width="stretch", hide_index=True)

                        sess_options = df_sessions["ID Sesi"].tolist()
                        sess_labels = {
                            int(r["ID Sesi"]): f"#{int(r['ID Sesi'])} | {r['Nama Sesi']} ({r['Ditutup']} · Hadir {r['Hadir']}/{r['Total']})"
                            for _, r in df_sessions.iterrows()
                        }
                        pick_sid = st.selectbox(
                            "Pilih sesi untuk dilihat / diunduh / dihapus",
                            options=sess_options,
                            format_func=lambda sid: sess_labels.get(int(sid), str(sid)),
                        )
                        if pick_sid is not None:
                            df_rec = db.get_session_records_db(pick_sid, id_col=id_lbl, label_col=group_lbl)
                            if not df_rec.empty:
                                st.dataframe(df_rec, width="stretch", hide_index=True)

                            act_c1, act_c2, act_c3 = st.columns(3)
                            s_row = df_sessions[df_sessions["ID Sesi"] == pick_sid].iloc[0]
                            s_name = str(s_row["Nama Sesi"])
                            s_csv = str(s_row["File CSV"]) or f"presensi_{s_name}.csv"
                            with act_c1:
                                st.download_button(
                                    "Unduh CSV",
                                    data=df_rec.to_csv(index=False).encode("utf-8"),
                                    file_name=s_csv,
                                    mime="text/csv",
                                    width="stretch",
                                    key=f"dl_csv_{pick_sid}",
                                )
                            with act_c2:
                                st.download_button(
                                    "Unduh Excel (.xlsx)",
                                    data=db.export_excel_bytes(df_rec, sheet_name=s_name),
                                    file_name=f"presensi_{s_name}_id{pick_sid}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    width="stretch",
                                    key=f"dl_xlsx_{pick_sid}",
                                )
                            with act_c3:
                                if st.button("Hapus Sesi Ini", width="stretch", key=f"del_sid_{pick_sid}"):
                                    db.delete_session(session_id=pick_sid)
                                    if sess.get("last_closed_file") == s_row["File CSV"]:
                                        sess["last_closed_df"] = None
                                        sess["last_closed_file"] = None
                                    st.success(f"Sesi #{pick_sid} ({s_name}) dan arsip CSV-nya telah dihapus.")
                                    st.rerun()

                    if orphan_csvs:
                        if not df_sessions.empty:
                            st.divider()
                        st.markdown("**Arsip CSV Tambahan**")
                        pick_csv = st.selectbox("Pilih file CSV", options=orphan_csvs)
                        if pick_csv:
                            csv_p = os.path.join(LOGS_DIR, pick_csv)
                            df_leg = pd.read_csv(csv_p)
                            st.dataframe(df_leg, width="stretch", hide_index=True)
                            lc1, lc2, lc3 = st.columns(3)
                            with lc1:
                                st.download_button(
                                    "Unduh CSV",
                                    data=df_leg.to_csv(index=False).encode("utf-8"),
                                    file_name=pick_csv,
                                    mime="text/csv",
                                    width="stretch",
                                    key="dl_leg_csv",
                                )
                            with lc2:
                                st.download_button(
                                    "Unduh Excel (.xlsx)",
                                    data=db.export_excel_bytes(df_leg, sheet_name="Rekap_CSV"),
                                    file_name=pick_csv.replace(".csv", ".xlsx"),
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    width="stretch",
                                    key="dl_leg_xlsx",
                                )
                            with lc3:
                                if st.button("Hapus File CSV Ini", width="stretch", key="del_leg_csv"):
                                    db.delete_session(csv_filename=pick_csv)
                                    st.success(f"File {pick_csv} dihapus.")
                                    st.rerun()

                    st.divider()
                    confirm_wipe = st.checkbox("Konfirmasi: hapus seluruh riwayat presensi (SQLite & semua CSV)", key="confirm_wipe_logs")
                    if st.button("Hapus Semua Riwayat Presensi", disabled=not confirm_wipe, width="stretch"):
                        db.delete_all_sessions()
                        sess["last_closed_df"] = None
                        sess["last_closed_file"] = None
                        st.success("Seluruh riwayat presensi telah dibersihkan.")
                        st.rerun()

        else:
            # =================================================================
            # KEADAAN B: SESI SEDANG DIBUKA (WEBCAM LIVE & MULTI-FOTO AKUMULATIF)
            # =================================================================
            if sess.pop("force_stop_live", False):
                st.session_state["live_cam_running"] = False
            st.session_state.setdefault("live_confirm_frames", 2)
            st.session_state.setdefault("live_interval_ms", 500)
            st.session_state.setdefault("live_mirror", True)

            session_name = sess["session_name"]
            opened_txt = sess["opened_at"].strftime("%H:%M:%S") if sess.get("opened_at") else "-"
            if sess.get("expires_at") is not None:
                rem_sec = max(0, int((sess["expires_at"] - datetime.now()).total_seconds()))
                rem_txt = f"Sisa {rem_sec // 60}m {rem_sec % 60:02d}d (tutup {sess['expires_at'].strftime('%H:%M:%S')})"
            else:
                rem_txt = "Tanpa batas waktu"

            col_bar, col_close = st.columns([3.0, 1.25])
            tot_hadir_sess = count_session_present(sess)
            with col_bar:
                st.markdown(
                    f"""
                    <div class="session-open-bar">
                        <div>
                            <strong>Sesi Aktif: {html.escape(session_name)}</strong>
                            <div class="sess-chips">
                                <span class="sess-chip">Mulai {opened_txt}</span>
                                <span class="sess-chip">{html.escape(rem_txt)}</span>
                                <span class="sess-chip">Hadir {tot_hadir_sess}/{len(db_nims)}</span>
                                <span class="sess-chip">Spoof {len(sess['spoof_logs'])}</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_close:
                st.markdown("<div class='desktop-spacer-sm'></div>", unsafe_allow_html=True)
                if st.button("Tutup & Simpan", type="secondary", width="stretch"):
                    close_attendance_session(sess, db, id_lbl, group_lbl, cfg["absent_label"])
                    st.rerun()

            with st.container(border=True):
                demo_files = list_demo_samples()
                source_options = ["Webcam Live", "Upload / HP"]
                if demo_files:
                    source_options.append("Sampel")
                input_mode = st.radio("Sumber Citra Presensi", source_options, horizontal=True)

                raw_bgr = None
                frame_sig = None
                live_on = False
                confirm_frames = int(st.session_state.get("live_confirm_frames", 2))
                interval_ms = int(st.session_state.get("live_interval_ms", 500))
                mirror_cam = bool(st.session_state.get("live_mirror", True))

                if input_mode == "Webcam Live":
                    lc_top1, lc_top2 = st.columns([1.6, 1.4])
                    with lc_top1:
                        live_on = st.toggle(
                            "Nyalakan Webcam Live",
                            key="live_cam_running",
                            help="Langsung membuka webcam browser secara live tanpa tombol foto. Setiap orang dibatasi maksimal 1× absen per sesi.",
                        )
                    with lc_top2:
                        st.caption("Rekam otomatis · Maks. 1× absen per orang dalam 1 sesi.")

                    with st.expander("Pengaturan Kamera Live & Mode Foto Manual", expanded=False):
                        set1, set2, set3 = st.columns(3)
                        with set1:
                            confirm_frames = st.slider(
                                "Konfirmasi Stabil (frame)",
                                1,
                                3,
                                key="live_confirm_frames",
                                help="1 = instan, 2 = tahan ~1 detik (disarankan agar tidak salah rekam frame buram).",
                            )
                        with set2:
                            interval_ms = st.slider(
                                "Jeda Pemindaian (ms)",
                                350,
                                1200,
                                step=50,
                                key="live_interval_ms",
                                help="Interval pengiriman frame ke AI untuk deteksi & anti-spoofing.",
                            )
                        with set3:
                            st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
                            mirror_cam = st.checkbox("Mirror (Cermin)", key="live_mirror")

                        if not live_on:
                            st.caption("Jika membuka web dari HP lewat alamat IP Wi-Fi, gunakan tab **Upload / HP** atau tombol foto manual berikut:")
                            cam_file = st.camera_input("Mode Cadangan: Ambil 1 Foto Webcam Browser")
                            if cam_file is not None:
                                raw_bytes = cam_file.getvalue()
                                frame_sig = f"cam:{hashlib.md5(raw_bytes).hexdigest()}"
                                raw_bgr = decode_uploaded_image(cam_file)
                        else:
                            st.caption("Mode foto manual disembunyikan sementara agar tidak bentrok dengan Webcam Live yang sedang aktif.")

                    if not live_on and sess.get("live_last_seq", 0) > 0:
                        reset_live_state(sess)

                elif input_mode == "Upload / HP":
                    st.session_state["live_cam_running"] = False
                    if sess.get("live_last_seq", 0) > 0:
                        reset_live_state(sess)
                    up_file = st.file_uploader("Unggah Foto / Buka Kamera HP (JPG / PNG)", type=["jpg", "jpeg", "png"])
                    if up_file is not None:
                        raw_bytes = up_file.getvalue()
                        frame_sig = f"up:{hashlib.md5(raw_bytes).hexdigest()}"
                        raw_bgr = decode_uploaded_image(up_file)
                else:
                    st.session_state["live_cam_running"] = False
                    if sess.get("live_last_seq", 0) > 0:
                        reset_live_state(sess)
                    sample_labels = {
                        f: ("Portrait Tunggal: Lena" if "lena" in f.lower() else ("Foto Grup Ramai: Solvay 1927 (29 Wajah)" if "solvay" in f.lower() else f))
                        for f in demo_files
                    }
                    chosen_sample = st.selectbox("Pilih Citra Uji Bawaan", demo_files, format_func=lambda k: sample_labels.get(k, k))
                    if chosen_sample:
                        frame_sig = f"demo:{chosen_sample}"
                        raw_bgr = cv2.imread(os.path.join(DEMO_DIR, chosen_sample))

            # -----------------------------------------------------------------
            # ALUR WEBCAM LIVE OTOMATIS BERBASIS BROWSER (NON-BLOCKING FRAGMENT)
            # -----------------------------------------------------------------
            if input_mode == "Webcam Live" and live_on:
                comp_key = "live_webcam_scanner_v2"

                @st.fragment
                def _render_live_webcam_panel():
                    if sess.get("expires_at") is not None and datetime.now() >= sess["expires_at"]:
                        close_attendance_session(sess, db, id_lbl, group_lbl, cfg["absent_label"])
                        st.rerun()

                    def _on_live_frame():
                        comp_val = st.session_state.get(comp_key)
                        if isinstance(comp_val, dict):
                            frame_payload = comp_val.get("frame")
                        else:
                            frame_payload = getattr(comp_val, "frame", None)
                        if isinstance(frame_payload, dict) and frame_payload.get("data_url"):
                            handle_live_webcam_frame(
                                frame_payload,
                                sess=sess,
                                engine=engine,
                                antispoof=antispoof,
                                db_embs=db_embs,
                                db_nims=db_nims,
                                db_names=db_names,
                                use_clahe=use_clahe,
                                use_spoof=use_spoof,
                                cos_thr=cos_thr,
                                det_thr=det_thr,
                                spoof_thr=spoof_thr,
                                confirm_frames=confirm_frames,
                                subjek_lbl=subjek_lbl,
                            )

                    ev_kind, ev_msg = sess.get(
                        "live_event",
                        ("info", "Webcam aktif. Arahkan wajah ke kamera untuk presensi otomatis (batas 1× per orang)."),
                    )
                    if ev_kind == "success":
                        st.success(ev_msg)
                    elif ev_kind == "error":
                        st.error(ev_msg)
                    elif ev_kind == "warning":
                        st.warning(ev_msg)
                    else:
                        st.info(ev_msg)

                    lk = sess.get("live_kpi", {"faces": 0, "sudah": 0, "spoof": 0, "unk": 0})
                    render_kpi_cards([
                        {"label": "Hadir (Total Sesi)", "value": f"{count_session_present(sess)} / {len(db_nims)}", "sub": "Maks. 1× per orang", "accent": "#0F294A"},
                        {"label": "Wajah di Kamera", "value": str(lk.get("faces", 0)), "sub": f"Sudah absen: {lk.get('sudah', 0)}", "accent": "#1D4ED8"},
                        {"label": "Ditolak (Spoof)", "value": str(lk.get("spoof", 0)), "sub": f"Total sesi: {len(sess['spoof_logs'])}", "accent": "#B91C1C"},
                        {"label": "Tidak Terdaftar", "value": str(lk.get("unk", 0)), "sub": "Belum Enroll", "accent": "#92400E"},
                    ])

                    col_lv_img, col_lv_tbl = st.columns([1.3, 1.05], gap="medium")
                    with col_lv_img:
                        with st.container(border=True):
                            st.markdown('<p class="section-title">Live Webcam Presensi Otomatis</p>', unsafe_allow_html=True)
                            st.markdown(
                                '<p class="section-sub">Arahkan wajah ke kamera agar kehadiran otomatis tercatat tanpa menekan tombol foto.</p>',
                                unsafe_allow_html=True,
                            )
                            overlay_payload = {
                                "instance_id": f"live_{session_name}",
                                "active": True,
                                "ack_seq": int(sess.get("live_last_seq", 0)),
                                "boxes": sess.get("live_boxes", []),
                                "mirror": bool(mirror_cam),
                                "interval_ms": int(interval_ms),
                                "light_badge": str(sess.get("live_light_badge", "")),
                                "hud_text": str(sess.get("live_hud_text", "")),
                                "hud_kind": str(sess.get("live_hud_kind", "info")),
                            }
                            comp_res = render_live_webcam(
                                key=comp_key,
                                data=overlay_payload,
                                on_frame_change=_on_live_frame,
                            )

                    with col_lv_tbl:
                        with st.container(border=True):
                            st.markdown('<p class="section-title">Rekap Sesi Aktif (Real-Time)</p>', unsafe_allow_html=True)
                            st.caption("Matikan toggle **Nyalakan Webcam Live** di atas kapan saja untuk berhenti atau mengoreksi data.")
                            if len(db_nims) == 0:
                                st.info(f"Database {subjek_lbl.lower()} masih kosong. Daftarkan wajah di tab **{subjek_lbl}**.")
                            else:
                                df_live = db.build_attendance_table(
                                    list(sess["present_map"].values()),
                                    id_col=id_lbl,
                                    label_col=group_lbl,
                                    absent_label=cfg["absent_label"],
                                    manual_status_map=sess.get("manual_status_map", {}),
                                )
                                st.dataframe(df_live, width="stretch", hide_index=True)

                            if sess["spoof_logs"]:
                                st.caption(f"Log Spoof Diblokir ({len(sess['spoof_logs'])} kejadian):")
                                st.dataframe(pd.DataFrame(sess["spoof_logs"][-4:]), width="stretch", hide_index=True)

                    # Cadangan jika trigger payload tiba namun belum diproses lewat callback
                    res_frame = getattr(comp_res, "frame", None)
                    if (
                        isinstance(res_frame, dict)
                        and res_frame.get("data_url")
                        and int(res_frame.get("seq", 0)) != int(sess.get("live_last_seq", 0))
                    ):
                        handle_live_webcam_frame(
                            res_frame,
                            sess=sess,
                            engine=engine,
                            antispoof=antispoof,
                            db_embs=db_embs,
                            db_nims=db_nims,
                            db_names=db_names,
                            use_clahe=use_clahe,
                            use_spoof=use_spoof,
                            cos_thr=cos_thr,
                            det_thr=det_thr,
                            spoof_thr=spoof_thr,
                            confirm_frames=confirm_frames,
                            subjek_lbl=subjek_lbl,
                        )
                        if sess.pop("live_just_recorded", False):
                            if sess.get("live_toast_msg"):
                                st.toast(sess["live_toast_msg"])
                            st.rerun()
                        else:
                            st.rerun(scope="fragment")

                    if sess.pop("live_just_recorded", False):
                        if sess.get("live_toast_msg"):
                            st.toast(sess["live_toast_msg"])
                        st.rerun()

                _render_live_webcam_panel()

            elif raw_bgr is None:
                st.markdown(
                    """
                    <div class="empty-card">
                        <div class="empty-card-title">Sesi Terbuka · Siap Merekam Kehadiran</div>
                        <div style="font-size:0.88rem; max-width:640px; margin:0 auto; line-height:1.5;">
                            Aktifkan <strong>Nyalakan Webcam Live</strong> di atas untuk absen otomatis real-time (maksimal <strong>1× per orang</strong>),
                            atau pilih <strong>Upload / HP</strong> untuk mengunggah foto dari perangkat.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Tampilkan rekap sementara yang sudah terkumpul saat webcam dimatikan
                if len(db_nims) > 0 and (sess["present_map"] or sess.get("manual_status_map") or sess["spoof_logs"]):
                    with st.container(border=True):
                        st.markdown('<p class="section-title">Rekap Sesi Aktif Saat Ini</p>', unsafe_allow_html=True)
                        df_idle = db.build_attendance_table(
                            list(sess["present_map"].values()),
                            id_col=id_lbl,
                            label_col=group_lbl,
                            absent_label=cfg["absent_label"],
                            manual_status_map=sess.get("manual_status_map", {}),
                        )
                        st.dataframe(df_idle, width="stretch", hide_index=True)
                        with st.container(horizontal=True, gap="small"):
                            if st.button("Reset Sesi", key="rst_idle"):
                                sess["present_map"] = {}
                                sess["manual_status_map"] = {}
                                sess["spoof_logs"] = []
                                sess["captures_count"] = 0
                                sess["last_frame_sig"] = None
                                sess["last_frame_new_ids"] = set()
                                reset_live_state(sess)
                                st.rerun()
                            st.download_button(
                                "Unduh CSV",
                                data=df_idle.to_csv(index=False).encode("utf-8"),
                                file_name=f"presensi_{session_name}.csv",
                                mime="text/csv",
                                key="dl_csv_idle",
                            )
                            st.download_button(
                                "Unduh Excel",
                                data=db.export_excel_bytes(df_idle, sheet_name=session_name),
                                file_name=f"presensi_{session_name}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="dl_xl_idle",
                            )
                        if sess["spoof_logs"]:
                            st.error(f"{len(sess['spoof_logs'])} percobaan Spoof diblokir pada sesi ini:")
                            st.dataframe(pd.DataFrame(sess["spoof_logs"][-5:]), width="stretch", hide_index=True)
            else:
                # Untuk foto statis baru, ID yang sudah tercatat dari foto sebelumnya ditandai Biru ("Sudah Absen")
                all_locked = set(sess["present_map"].keys()) | set(sess.get("manual_status_map", {}).keys())
                if sess.get("last_frame_sig") == frame_sig:
                    prev_locked = all_locked - set(sess.get("last_frame_new_ids", set()))
                else:
                    prev_locked = all_locked

                proc_bgr, enh_info, face_results, recognized_records, first_debug = process_attendance_frame(
                    raw_bgr,
                    engine,
                    antispoof,
                    db_embs,
                    db_nims,
                    db_names,
                    use_clahe,
                    use_spoof,
                    cos_thr,
                    det_thr,
                    spoof_thr,
                    already_present_ids=prev_locked,
                )
                st.session_state["last_debug"] = first_debug
                record_frame_into_session(sess, frame_sig, recognized_records, face_results, force=False)

                annotated_bgr = draw_result_annotations(proc_bgr, face_results)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                n_hadir_now = sum(1 for r in face_results if r["status"] in ("Hadir", "Sudah Absen"))
                n_sudah_now = sum(1 for r in face_results if r["status"] == "Sudah Absen")
                n_spoof_now = sum(1 for r in face_results if r["status"] == "Spoof")
                n_unk_now = sum(1 for r in face_results if r["status"] == "Tidak Terdaftar")

                render_kpi_cards([
                    {"label": "Hadir (Total Sesi)", "value": f"{count_session_present(sess)} / {len(db_nims)}", "sub": f"Foto ini: {n_hadir_now} cocok", "accent": "#0F294A"},
                    {"label": "Wajah di Foto Ini", "value": str(len(face_results)), "sub": f"Sudah absen: {n_sudah_now}", "accent": "#1D4ED8"},
                    {"label": "Ditolak (Spoof)", "value": str(n_spoof_now), "sub": f"Total sesi: {len(sess['spoof_logs'])}", "accent": "#B91C1C"},
                    {"label": "Tidak Terdaftar", "value": str(n_unk_now), "sub": "Belum Enroll", "accent": "#92400E"},
                ])

                if n_sudah_now > 0:
                    sudah_names = [r["nama"] for r in face_results if r["status"] == "Sudah Absen"]
                    st.info(f"**{', '.join(sudah_names)}** sudah tercatat hadir sebelumnya pada sesi ini (dibatasi maksimal **1× absen per orang**).")

                light_tag = "CLAHE Aktif" if enh_info["applied"] else "Pencahayaan Normal"
                st.markdown(
                    f"""
                    <div class="strip-pill">
                        <span class="strip-chip"><strong>{light_tag}: {html.escape(str(enh_info['status_before']))}</strong></span>
                        <span class="strip-chip">L* <strong>{enh_info['before_l']} → {enh_info['after_l']}</strong> (γ={enh_info['gamma_used']})</span>
                        <span>{html.escape(str(enh_info['reason']))}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_img, col_tbl = st.columns([1.3, 1.05], gap="medium")
                with col_img:
                    with st.container(border=True):
                        st.markdown('<p class="section-title">Visualisasi Deteksi & Pengenalan</p>', unsafe_allow_html=True)
                        st.markdown('<p class="section-sub">Hijau = Hadir · Biru = Sudah Absen (1×) · Merah = Spoof · Kuning = Belum Terdaftar</p>', unsafe_allow_html=True)
                        st.image(annotated_rgb, width="stretch")

                with col_tbl:
                    with st.container(border=True):
                        st.markdown('<p class="section-title">Rekap Sesi Aktif (Terakumulasi)</p>', unsafe_allow_html=True)
                        st.markdown(
                            f'<p class="section-sub">Sesi: <strong>{html.escape(session_name)}</strong> · '
                            f'Foto diproses: <strong>{sess["captures_count"]}</strong></p>',
                            unsafe_allow_html=True,
                        )

                        if len(db_nims) == 0:
                            st.info(
                                f"Database {subjek_lbl.lower()} masih kosong. Daftarkan 1–3 foto wajah di tab "
                                f"**Database {subjek_lbl}** untuk mulai mencocokkan nama & {id_lbl}."
                            )
                        else:
                            df_att = db.build_attendance_table(
                                list(sess["present_map"].values()),
                                id_col=id_lbl,
                                label_col=group_lbl,
                                absent_label=cfg["absent_label"],
                                manual_status_map=sess.get("manual_status_map", {}),
                            )
                            st.dataframe(df_att, width="stretch", hide_index=True)

                            with st.container(horizontal=True, gap="small"):
                                if st.button("Catat Ulang", help="Terapkan ulang hasil foto ini ke sesi aktif (mis. setelah menggeser slider)."):
                                    record_frame_into_session(sess, frame_sig, recognized_records, face_results, force=True)
                                    st.rerun()
                                if st.button("Reset", help="Kosongkan sementara daftar hadir pada sesi aktif ini."):
                                    sess["present_map"] = {}
                                    sess["manual_status_map"] = {}
                                    sess["spoof_logs"] = []
                                    sess["captures_count"] = 0
                                    sess["last_frame_sig"] = None
                                    st.rerun()
                                csv_bytes = df_att.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "Unduh CSV",
                                    data=csv_bytes,
                                    file_name=f"presensi_{session_name}.csv",
                                    mime="text/csv",
                                )
                                xlsx_bytes = db.export_excel_bytes(df_att, sheet_name=session_name)
                                st.download_button(
                                    "Unduh Excel",
                                    data=xlsx_bytes,
                                    file_name=f"presensi_{session_name}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                )

                            with st.expander(f"Koreksi / Hapus Kehadiran Per {subjek_lbl}"):
                                master_opts = db.list_students(id_col=id_lbl, label_col=group_lbl)
                                opt_ids = master_opts[id_lbl].tolist()
                                opt_lbls = {
                                    str(r[id_lbl]): f"{r[id_lbl]} | {r['Nama']}"
                                    for _, r in master_opts.iterrows()
                                }
                                kc1, kc2 = st.columns(2)
                                with kc1:
                                    pick_person = st.selectbox(
                                        f"Pilih {subjek_lbl}",
                                        options=opt_ids,
                                        format_func=lambda k: opt_lbls.get(str(k), str(k)),
                                        key="kor_person",
                                    )
                                with kc2:
                                    kor_action = st.selectbox(
                                        "Status / Aksi",
                                        options=[
                                            "Batalkan / Hapus Kehadiran",
                                            "Hadir (Manual)",
                                            "Izin",
                                            "Sakit",
                                        ],
                                        key="kor_action",
                                    )
                                if st.button("Terapkan Koreksi", width="stretch", key="btn_apply_kor"):
                                    now_k = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    if kor_action.startswith("Batalkan"):
                                        sess["present_map"].pop(pick_person, None)
                                        sess["manual_status_map"].pop(pick_person, None)
                                    elif kor_action.startswith("Hadir"):
                                        sess["manual_status_map"][pick_person] = {"status": "Hadir (Manual)", "score": 0.0, "time": now_k}
                                    elif kor_action.startswith("Izin"):
                                        sess["present_map"].pop(pick_person, None)
                                        sess["manual_status_map"][pick_person] = {"status": "Izin", "score": 0.0, "time": now_k}
                                    else:
                                        sess["present_map"].pop(pick_person, None)
                                        sess["manual_status_map"][pick_person] = {"status": "Sakit", "score": 0.0, "time": now_k}
                                    st.rerun()

                        if sess["spoof_logs"]:
                            st.error(f"{len(sess['spoof_logs'])} percobaan Spoof/Layar HP diblokir pada sesi ini:")
                            st.dataframe(pd.DataFrame(sess["spoof_logs"][-5:]), width="stretch", hide_index=True)

    # -------------------------------------------------------------------------
    # TAB 2: DATABASE SUBJEK (ONE-SHOT ENROLLMENT + LABEL KUSTOM PER ORANG)
    # -------------------------------------------------------------------------
    with tab2:
        col_form, col_list = st.columns([1.0, 1.15], gap="medium")

        with col_form:
            with st.container(border=True):
                st.markdown('<p class="section-title">Pendaftaran Wajah Baru (One-Shot Enrollment)</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<p class="section-sub">Tidak perlu training ulang jaringan saraf. Cukup unggah 1–3 foto wajah menghadap kamera per {html.escape(subjek_lbl.lower())}.</p>',
                    unsafe_allow_html=True,
                )

                with st.form("form_enroll", clear_on_submit=True, border=False):
                    nim_in = st.text_input(id_lbl, placeholder="Masukkan nomor ID (contoh: ID-001)")
                    nama_in = st.text_input("Nama Lengkap", placeholder="Masukkan nama lengkap")
                    group_in = st.text_input(
                        f"{group_lbl} / Label Tersendiri (Opsional)",
                        placeholder="Contoh: Divisi Operasional / Unit B / Tamu",
                    )
                    enroll_photos = st.file_uploader(
                        "Unggah 1–3 Foto Referensi Wajah", type=["jpg", "jpeg", "png"], accept_multiple_files=True
                    )
                    submitted = st.form_submit_button(f"Daftarkan {subjek_lbl} ke Database", type="primary", width="stretch")

                if submitted:
                    if not nim_in.strip() or not nama_in.strip() or not enroll_photos:
                        st.error(f"Mohon lengkapi {id_lbl}, Nama, dan minimal 1 foto wajah.")
                    else:
                        embs_new = []
                        for upf in enroll_photos:
                            bgr_in = decode_uploaded_image(upf)
                            bgr_in, _ = resize_max_side(bgr_in, max_side=1024)
                            enh_in, _ = apply_adaptive_enhancement(bgr_in)
                            det_faces = engine.detect(enh_in, score_threshold=0.6)
                            if det_faces:
                                _, emb_v = engine.align_and_embed(enh_in, det_faces[0]["raw_row"])
                                embs_new.append(emb_v)

                        if not embs_new:
                            st.error("Wajah tidak terdeteksi pada foto yang diunggah. Gunakan foto menghadap ke kamera.")
                        else:
                            mean_emb = np.mean(np.vstack(embs_new), axis=0)
                            act_status = db.enroll(nim_in, nama_in, mean_emb, label=group_in)
                            st.success(f"{subjek_lbl} {nama_in} ({nim_in}) berhasil {act_status} menggunakan {len(embs_new)} foto!")
                            st.rerun()

            with st.expander("Sampel Anti-Spoofing (Opsional)"):
                st.caption("Simpan foto langsung (real) vs foto layar HP (spoof) ke `data/spoof_dataset/` untuk melatih model SVM lokal.")
                sp_label = st.selectbox("Kategori Sampel", ["real (Wajah Langsung)", "spoof (Foto Layar HP)"])
                sp_files = st.file_uploader("Unggah Foto Sampel Spoof/Real", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="spoof_up")
                if st.button("Simpan ke Dataset Spoofing", width="stretch") and sp_files:
                    target_dir = SPOOF_REAL_DIR if sp_label.startswith("real") else SPOOF_FAKE_DIR
                    os.makedirs(target_dir, exist_ok=True)
                    existing_n = len(os.listdir(target_dir))
                    saved_cnt = 0
                    for idx_s, uf in enumerate(sp_files):
                        im_s = decode_uploaded_image(uf)
                        out_sp = os.path.join(target_dir, f"sample_{existing_n + idx_s + 1:03d}.jpg")
                        cv2.imwrite(out_sp, im_s)
                        saved_cnt += 1
                    st.success(f"{saved_cnt} sampel tersimpan. Jalankan `python scripts/02_train_antispoofing.py` untuk melatih SVM.")

        with col_list:
            with st.container(border=True):
                st.markdown(f'<p class="section-title">Daftar {html.escape(subjek_lbl)} Terdaftar</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<p class="section-sub">Setiap {html.escape(id_lbl)} disimpan bersama label "{html.escape(group_lbl)}" dan vektor embedding 128-D ter-normalisasi L2.</p>',
                    unsafe_allow_html=True,
                )

                df_master = db.list_students(id_col=id_lbl, label_col=group_lbl)
                if df_master.empty:
                    st.info(f"Belum ada {subjek_lbl.lower()} yang terdaftar di database.")
                else:
                    st.dataframe(df_master, width="stretch", hide_index=True)
                    d_col1, d_col2 = st.columns([1.8, 1.2])
                    with d_col1:
                        del_nim = st.selectbox(f"Pilih {id_lbl} untuk dihapus", options=df_master[id_lbl].tolist(), label_visibility="collapsed")
                    with d_col2:
                        if st.button(f"Hapus {id_lbl}", width="stretch"):
                            if db.delete(del_nim):
                                st.success(f"{id_lbl} {del_nim} dihapus.")
                                st.rerun()

    # -------------------------------------------------------------------------
    # TAB 3: LAB VISUAL PCD (PIPELINE BEFORE - AFTER)
    # -------------------------------------------------------------------------
    with tab3:
        dbg = st.session_state.get("last_debug")
        if dbg is None:
            st.markdown(
                """
                <div class="empty-card">
                    <div class="empty-card-title">Laboratorium Visual Tahap PCD</div>
                    <div style="font-size:0.9rem; max-width:600px; margin:0 auto; line-height:1.5;">
                        Proses minimal satu foto pada tab <strong>Presensi</strong> (dapat memakai <em>Sampel Uji Bawaan</em>)
                        untuk memeriksa transformasi <strong>Adaptive CLAHE + Gamma</strong>, <strong>5-Landmark Alignment</strong>,
                        dan <strong>Peta Tekstur Mikro LBP</strong> secara transparan.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            raw_rgb = cv2.cvtColor(dbg["raw_bgr"], cv2.COLOR_BGR2RGB)
            proc_rgb = cv2.cvtColor(dbg["proc_bgr"], cv2.COLOR_BGR2RGB)
            e_inf = dbg["enh_info"]
            sd = dbg["spoof_diag"]

            delta_l = e_inf["after_l"] - e_inf["before_l"]
            render_kpi_cards([
                {"label": "Luminansi Akhir (L*)", "value": f"{e_inf['after_l']}", "sub": f"{delta_l:+.2f} ΔL*", "accent": "#047857"},
                {"label": "Luminansi Awal (L*)", "value": f"{e_inf['before_l']}", "sub": str(e_inf["status_before"]), "accent": "#475569"},
                {"label": "Gamma Adaptif", "value": f"{e_inf['gamma_used']}", "sub": "Koreksi Otomatis", "accent": "#1D4ED8"},
                {"label": "Ketajaman Laplacian", "value": f"{sd.get('laplacian_var', '-')}", "sub": f"Spekular {sd.get('specular_ratio', '-')}", "accent": "#0F294A"},
            ])

            c1, c2 = st.columns(2, gap="medium")
            with c1:
                with st.container(border=True):
                    st.markdown('<p class="section-title">1. Citra Asli Sebelum Enhancement</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="section-sub">Status cahaya awal: {html.escape(str(e_inf["status_before"]))}</p>', unsafe_allow_html=True)
                    st.image(raw_rgb, width="stretch")
            with c2:
                with st.container(border=True):
                    st.markdown('<p class="section-title">2. Setelah Adaptive CLAHE + Gamma</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="section-sub">{html.escape(str(e_inf["reason"]))}</p>', unsafe_allow_html=True)
                    st.image(proc_rgb, width="stretch")

            with st.container(border=True):
                fig_hist = render_histogram_l(dbg["raw_bgr"], dbg["proc_bgr"])
                st.pyplot(fig_hist, width="stretch")
                plt.close(fig_hist)

            if dbg["aligned_face"] is not None:
                c3, c4, c5 = st.columns([1.0, 1.0, 1.4], gap="medium")
                with c3:
                    with st.container(border=True):
                        st.markdown('<p class="section-title">3. Alignment Geometri</p>', unsafe_allow_html=True)
                        st.markdown('<p class="section-sub">Crop wajah diluruskan 112×112 px</p>', unsafe_allow_html=True)
                        aligned_rgb = cv2.cvtColor(dbg["aligned_face"], cv2.COLOR_BGR2RGB)
                        st.image(aligned_rgb, width="stretch")
                with c4:
                    with st.container(border=True):
                        st.markdown('<p class="section-title">4. Peta Tekstur LBP</p>', unsafe_allow_html=True)
                        st.markdown('<p class="section-sub">Kode Uniform LBP radius 1 (8 tetangga)</p>', unsafe_allow_html=True)
                        if dbg["lbp_map"] is not None:
                            lbp_vis = (dbg["lbp_map"].astype(np.float32) / 9.0 * 255.0).astype(np.uint8)
                            lbp_col = cv2.applyColorMap(lbp_vis, cv2.COLORMAP_VIRIDIS)
                            st.image(cv2.cvtColor(lbp_col, cv2.COLOR_BGR2RGB), width="stretch")
                with c5:
                    with st.container(border=True):
                        st.markdown('<p class="section-title">5. Diagnostik Anti-Spoofing</p>', unsafe_allow_html=True)
                        status_txt = "ASLI (Real)" if dbg["is_real"] else f"SPOOF ({sd.get('attack_type', 'Layar')})"
                        fas_info = ""
                        if "fas_screen" in sd:
                            fas_info = (
                                f" · MiniFASNet [Real=<code>{sd['fas_real']}</code>, "
                                f"Layar=<code>{sd['fas_screen']}</code>, Kertas=<code>{sd['fas_paper']}</code>]"
                            )
                        st.markdown(
                            f'<p class="section-sub">Status: <strong>{html.escape(status_txt)}</strong> '
                            f'(Skor Fusi = <code>{dbg["prob_real"]:.3f}</code>{fas_info}) · '
                            f'SVM = <code>{sd.get("svm_prob_real", "-")}</code> · '
                            f'Bezel = <code>{sd.get("bezel_line_score", "-")}</code> · '
                            f'FFT = <code>{sd.get("fft_high_ratio", "-")}</code></p>',
                            unsafe_allow_html=True,
                        )
                        if "lbp_hist_r1" in sd:
                            fig_lbp = render_lbp_histogram(sd["lbp_hist_r1"])
                            st.pyplot(fig_lbp, width="stretch")
                            plt.close(fig_lbp)

    # -------------------------------------------------------------------------
    # TAB 4: DASHBOARD EVALUASI & BUKTI LAPORAN
    # -------------------------------------------------------------------------
    with tab4:
        summary_p = os.path.join(EXP_DIR, "eval_summary.json")

        if not os.path.exists(summary_p):
            st.warning(
                "Hasil eksperimen belum ditemukan (`artifacts/experiments/eval_summary.json`).\n"
                "Jalankan `python scripts/03_run_experiments.py` untuk mengisi grafik dan metrik LFW/Low-Light."
            )
        else:
            with open(summary_p, "r", encoding="utf-8") as f:
                summary = json.load(f)

            lfw_m = summary.get("lfw_recognition", {})
            st.markdown('<p class="section-title">A. Validasi Identitas SFace pada Dataset Resmi LFW (UMass Amherst)</p>', unsafe_allow_html=True)
            st.markdown('<p class="section-sub">Evaluasi otomatis pada 300 pasangan uji (positif & negatif) dari 96 individu.</p>', unsafe_allow_html=True)

            render_kpi_cards([
                {"label": "Akurasi Verifikasi", "value": f"{lfw_m.get('accuracy', 0)*100:.2f}%", "sub": f"{lfw_m.get('n_pairs_total', 300)} pasangan uji", "accent": "#0F294A"},
                {"label": "F1-Score", "value": f"{lfw_m.get('f1_score', 0)*100:.2f}%", "sub": f"Prec {lfw_m.get('precision', 0)*100:.1f}% · Rec {lfw_m.get('recall', 0)*100:.1f}%", "accent": "#1D4ED8"},
                {"label": "Cosine (Identik vs Beda)", "value": f"{lfw_m.get('mean_same_score', 0):.3f} vs {lfw_m.get('mean_diff_score', 0):.3f}", "sub": f"Ambang {lfw_m.get('cosine_threshold', 0.363)}", "accent": "#047857"},
                {"label": "Rerata Latensi CPU", "value": f"{lfw_m.get('avg_latency_ms', 0):.1f} ms", "sub": "Per wajah (tanpa GPU)", "accent": "#475569"},
            ])

            g_lfw = os.path.join(EXP_DIR, "grafik_lfw_cosine.png")
            g_clahe = os.path.join(EXP_DIR, "grafik_clahe_vs_raw.png")
            g_cm = os.path.join(EXP_DIR, "cm_antispoofing.png")
            low_csv = os.path.join(EXP_DIR, "hasil_uji_lowlight.csv")

            col_g1, col_g2 = st.columns(2, gap="medium")
            with col_g1:
                if os.path.exists(g_lfw):
                    with st.container(border=True):
                        st.markdown('<p class="section-title">Distribusi Cosine Similarity (LFW)</p>', unsafe_allow_html=True)
                        st.image(g_lfw, width="stretch")
            with col_g2:
                if os.path.exists(g_clahe):
                    with st.container(border=True):
                        st.markdown('<p class="section-title">Efektivitas Adaptive Low-Light CLAHE</p>', unsafe_allow_html=True)
                        st.image(g_clahe, width="stretch")

            if os.path.exists(low_csv):
                with st.container(border=True):
                    st.markdown('<p class="section-title">B. Perbandingan Deteksi Tanpa vs Dengan Adaptive CLAHE</p>', unsafe_allow_html=True)
                    df_low = pd.read_csv(low_csv)
                    df_low["Kondisi Pencahayaan"] = df_low["Kondisi Pencahayaan"].str.replace("Gelap Terkontrol ", "Gelap ", regex=False)
                    st.dataframe(
                        df_low,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Kondisi Pencahayaan": st.column_config.TextColumn("Kondisi", width="medium"),
                            "Jumlah Uji": st.column_config.NumberColumn("N", format="%d", width="small"),
                            "Deteksi Tanpa CLAHE (%)": st.column_config.ProgressColumn(
                                "Raw (%)",
                                format="%.1f%%",
                                min_value=0,
                                max_value=100,
                                width="small",
                            ),
                            "Deteksi Dengan Adaptive CLAHE (%)": st.column_config.ProgressColumn(
                                "CLAHE (%)",
                                format="%.1f%%",
                                min_value=0,
                                max_value=100,
                                width="small",
                            ),
                            "Skor Rerata Raw": st.column_config.NumberColumn("Raw Score", format="%.2f", width="small"),
                            "Skor Rerata Enhanced": st.column_config.NumberColumn("CLAHE Score", format="%.2f", width="small"),
                        },
                    )

            if summary.get("antispoofing") and os.path.exists(g_cm):
                with st.container(border=True):
                    st.markdown('<p class="section-title">C. Hasil Evaluasi Anti-Spoofing (MiniFASNet ONNX + 105-D SVM)</p>', unsafe_allow_html=True)
                    sp_m = summary["antispoofing"]
                    c_sp1, c_sp2 = st.columns([1.0, 1.2], gap="medium")
                    with c_sp1:
                        render_kpi_cards([
                            {"label": "Akurasi Anti-Spoof", "value": f"{sp_m['accuracy']*100:.2f}%", "sub": f"{sp_m['n_real']} Real + {sp_m['n_spoof']} Spoof", "accent": "#0F294A"},
                            {"label": "ROC-AUC", "value": f"{sp_m['roc_auc']:.4f}", "sub": "Validasi Hybrid Anti-Spoof", "accent": "#1D4ED8"},
                        ])
                    with c_sp2:
                        st.image(g_cm, caption="Confusion Matrix Evaluasi Anti-Spoofing", width="stretch")


if __name__ == "__main__":
    main()
