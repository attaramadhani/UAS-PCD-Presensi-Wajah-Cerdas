"""Komponen Webcam Live berbasis Browser (WebRTC/getUserMedia) untuk Streamlit v2.

Menghindari masalah driver OpenCV DirectShow/MSMF (layar hitam / error -1072875772)
dengan mengambil video langsung di browser pengguna (30 FPS mulus) sambil mengirim
frame periodik ke Python untuk deteksi YuNet, Anti-Spoofing Hybrid, pengenalan SFace,
dan penguncian presensi 1x per orang per sesi.
"""

from __future__ import annotations

import base64
from typing import Any, Callable

import cv2
import numpy as np
import streamlit as st

_LIVE_WEBCAM_HTML = """
<div class="cam-shell">
  <div class="cam-stage">
    <video class="cam-video" autoplay playsinline muted></video>
    <canvas class="cam-overlay"></canvas>
    <div class="cam-hud-top">
      <span class="cam-live-badge"><span class="cam-dot"></span><span class="cam-live-txt">LIVE</span></span>
      <span class="cam-light-badge" style="display:none;"></span>
    </div>
    <div class="cam-hud-bottom" style="display:none;"></div>
    <div class="cam-msg-box">
      <div class="cam-msg-icon">🎥</div>
      <div class="cam-msg-title">Menghubungkan Webcam Browser...</div>
      <div class="cam-msg-desc">Klik <strong>Izinkan / Allow</strong> jika browser meminta izin akses kamera.</div>
      <button type="button" class="cam-retry-btn" style="display:none;">Coba Hubungkan Ulang</button>
    </div>
  </div>
  <div class="cam-toolbar">
    <div class="cam-legend">
      <span class="leg"><i style="background:#10B981"></i>Hadir</span>
      <span class="leg"><i style="background:#2563EB"></i>Sudah Absen (1×)</span>
      <span class="leg"><i style="background:#EF4444"></i>Spoof</span>
      <span class="leg"><i style="background:#F59E0B"></i>Asing</span>
    </div>
    <div class="cam-controls">
      <select class="cam-select" title="Pilih Kamera" style="display:none;"></select>
      <span class="cam-fps">Menyiapkan...</span>
    </div>
  </div>
</div>
"""

_LIVE_WEBCAM_CSS = """
:host {
  display: block;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.cam-shell {
  width: 100%;
  border-radius: 14px;
  overflow: hidden;
  background: #0F172A;
  border: 1px solid #CBD5E1;
  box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08);
}
.cam-stage {
  position: relative;
  width: 100%;
  min-height: 240px;
  background: #0B1220;
  overflow: hidden;
}
.cam-stage.is-live {
  min-height: 0;
}
.cam-video {
  width: 100%;
  height: auto;
  display: block;
  background: #0B1220;
}
.cam-video.mirrored {
  transform: scaleX(-1);
}
.cam-overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}
.cam-hud-top {
  position: absolute;
  top: 10px;
  left: 10px;
  right: 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  pointer-events: none;
  flex-wrap: wrap;
}
.cam-live-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(15, 23, 42, 0.78);
  color: #F8FAFC;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.05em;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.16);
}
.cam-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #10B981;
  box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.28);
}
.cam-dot.scanning {
  background: #38BDF8;
  box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.35);
}
.cam-dot.error {
  background: #EF4444;
  box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.32);
}
.cam-light-badge {
  background: rgba(15, 23, 42, 0.78);
  color: #E2E8F0;
  font-size: 11px;
  font-weight: 600;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.14);
}
.cam-hud-bottom {
  position: absolute;
  left: 10px;
  right: 10px;
  bottom: 10px;
  padding: 8px 12px;
  border-radius: 10px;
  font-size: 12px;
  font-weight: 600;
  line-height: 1.35;
  color: #FFFFFF;
  background: rgba(15, 23, 42, 0.84);
  border: 1px solid rgba(255, 255, 255, 0.15);
  pointer-events: none;
}
.cam-hud-bottom.kind-success {
  background: rgba(6, 95, 70, 0.90);
  border-color: rgba(16, 185, 129, 0.55);
}
.cam-hud-bottom.kind-error {
  background: rgba(153, 27, 27, 0.92);
  border-color: rgba(248, 113, 113, 0.55);
}
.cam-hud-bottom.kind-warn {
  background: rgba(146, 64, 14, 0.92);
  border-color: rgba(251, 191, 36, 0.55);
}
.cam-msg-box {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 20px;
  background: rgba(15, 23, 42, 0.94);
  color: #F8FAFC;
  gap: 8px;
}
.cam-msg-icon {
  font-size: 32px;
  line-height: 1;
}
.cam-msg-title {
  font-size: 15px;
  font-weight: 700;
  color: #F8FAFC;
}
.cam-msg-desc {
  font-size: 12.5px;
  color: #CBD5E1;
  max-width: 440px;
  line-height: 1.5;
}
.cam-retry-btn {
  margin-top: 6px;
  padding: 7px 14px;
  border-radius: 8px;
  border: none;
  background: #2563EB;
  color: #FFFFFF;
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
}
.cam-retry-btn:hover {
  background: #1D4ED8;
}
.cam-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
  padding: 8px 12px;
  background: #F8FAFC;
  border-top: 1px solid #E2E8F0;
}
.cam-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  font-size: 11.5px;
  font-weight: 600;
  color: #334155;
}
.cam-legend .leg {
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.cam-legend i {
  display: inline-block;
  width: 9px;
  height: 9px;
  border-radius: 3px;
}
.cam-controls {
  display: flex;
  align-items: center;
  gap: 8px;
}
.cam-select {
  font-size: 11.5px;
  padding: 3px 7px;
  border-radius: 6px;
  border: 1px solid #CBD5E1;
  background: #FFFFFF;
  color: #0F172A;
  max-width: 170px;
}
.cam-fps {
  font-size: 11.5px;
  font-weight: 600;
  color: #475569;
  font-variant-numeric: tabular-nums;
}
"""

_LIVE_WEBCAM_JS = """
const POOL_KEY = "__presensiLiveCamPool_v2";
const UNLOAD_FLAG = "__presensiLiveCamUnloadBound_v2";

function getPool() {
  if (!window[POOL_KEY]) {
    window[POOL_KEY] = {};
  }
  return window[POOL_KEY];
}

function hasLiveStream(stream) {
  return !!(
    stream &&
    typeof stream.getVideoTracks === "function" &&
    stream.getVideoTracks().some((t) => t.readyState === "live")
  );
}

function stopTracksOnly(ctrl) {
  if (ctrl && ctrl.stream) {
    try {
      ctrl.stream.getTracks().forEach((t) => t.stop());
    } catch (_) {}
    ctrl.stream = null;
  }
}

function stopController(instKey) {
  const pool = getPool();
  const ctrl = pool[instKey];
  if (!ctrl) return;
  if (ctrl.stopTimer) {
    clearTimeout(ctrl.stopTimer);
    ctrl.stopTimer = null;
  }
  if (ctrl.tickTimer) {
    clearInterval(ctrl.tickTimer);
    ctrl.tickTimer = null;
  }
  stopTracksOnly(ctrl);
  delete pool[instKey];
}

if (!window[UNLOAD_FLAG]) {
  window[UNLOAD_FLAG] = true;
  window.addEventListener("pagehide", () => {
    const pool = getPool();
    Object.keys(pool).forEach((k) => stopController(k));
  });
}

function sampleMeanBrightness(ctx, w, h) {
  try {
    const data = ctx.getImageData(0, 0, w, h).data;
    let sum = 0;
    let cnt = 0;
    const step = 16 * 4;
    for (let i = 0; i < data.length; i += step) {
      sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      cnt += 1;
    }
    return cnt > 0 ? sum / cnt : 255;
  } catch (_) {
    return 255;
  }
}

function renderMessage(ctrl, icon, title, desc, showRetry) {
  if (!ctrl.els) return;
  const { stage, msgBox, msgIcon, msgTitle, msgDesc, retryBtn, dotEl, liveTxt } = ctrl.els;
  if (stage) stage.classList.remove("is-live");
  msgBox.style.display = "flex";
  msgIcon.textContent = icon;
  msgTitle.textContent = title;
  msgDesc.textContent = desc;
  retryBtn.style.display = showRetry ? "inline-block" : "none";
  if (dotEl) {
    dotEl.classList.remove("scanning");
    dotEl.classList.add("error");
  }
  if (liveTxt) liveTxt.textContent = "OFFLINE";
}

function hideMessage(ctrl) {
  if (!ctrl.els) return;
  const { stage, msgBox, dotEl, liveTxt } = ctrl.els;
  if (stage) stage.classList.add("is-live");
  msgBox.style.display = "none";
  if (dotEl) dotEl.classList.remove("error");
  if (liveTxt) liveTxt.textContent = "LIVE · AUTO";
}

function updateHud(ctrl) {
  if (!ctrl.els) return;
  const { lightBadgeEl, hudBottom, fpsEl, dotEl } = ctrl.els;
  if (lightBadgeEl) {
    if (ctrl.lightBadge) {
      lightBadgeEl.textContent = ctrl.lightBadge;
      lightBadgeEl.style.display = "inline-block";
    } else {
      lightBadgeEl.style.display = "none";
    }
  }
  if (hudBottom) {
    let text = ctrl.hudText || "";
    let kind = ctrl.hudKind || "info";
    if (ctrl.darkStreak >= 3) {
      text = "⚠️ Gambar kamera sangat gelap/hitam — pastikan penutup kamera (privacy shutter) laptop terbuka.";
      kind = "warn";
    }
    if (text) {
      hudBottom.textContent = text;
      hudBottom.className = "cam-hud-bottom kind-" + kind;
      hudBottom.style.display = "block";
    } else {
      hudBottom.style.display = "none";
    }
  }
  if (dotEl && !dotEl.classList.contains("error")) {
    dotEl.classList.toggle("scanning", !!ctrl.inFlight);
  }
  if (fpsEl) {
    if (ctrl.fatalError) {
      fpsEl.textContent = "Kamera tidak aktif";
    } else if (!hasLiveStream(ctrl.stream)) {
      fpsEl.textContent = "Menghubungkan...";
    } else {
      const nBox = (ctrl.boxes || []).length;
      const scanLbl = ctrl.seq > 0 ? `Scan #${ctrl.seq}` : "Siap";
      fpsEl.textContent = `${scanLbl} · ${nBox} wajah`;
    }
  }
}

async function populateDevices(ctrl) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  if (!ctrl.els || !ctrl.els.camSelect) return;
  try {
    const devs = await navigator.mediaDevices.enumerateDevices();
    const cams = devs.filter((d) => d.kind === "videoinput");
    const sel = ctrl.els.camSelect;
    if (cams.length <= 1) {
      sel.style.display = "none";
      return;
    }
    const activeTrack = ctrl.stream && ctrl.stream.getVideoTracks()[0];
    const activeId = (activeTrack && activeTrack.getSettings && activeTrack.getSettings().deviceId) || ctrl.deviceId || "";
    sel.innerHTML = "";
    cams.forEach((cam, idx) => {
      const opt = document.createElement("option");
      opt.value = cam.deviceId;
      opt.textContent = cam.label || `Kamera ${idx + 1}`;
      if (cam.deviceId === activeId) opt.selected = true;
      sel.appendChild(opt);
    });
    sel.style.display = "inline-block";
  } catch (_) {}
}

async function startCamera(ctrl) {
  if (ctrl.starting) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    ctrl.fatalError = "insecure";
    ctrl.fatalInfo = {
      icon: "📱",
      title: "Kamera Live Membutuhkan localhost atau HTTPS",
      desc: "Browser memblokir live webcam pada alamat IP HTTP non-localhost. Jika membuka dari HP, gunakan tab 'Upload / HP'. Untuk Webcam Live otomatis, buka di laptop via http://localhost:8501.",
      showRetry: false,
    };
    renderMessage(ctrl, ctrl.fatalInfo.icon, ctrl.fatalInfo.title, ctrl.fatalInfo.desc, ctrl.fatalInfo.showRetry);
    updateHud(ctrl);
    return;
  }

  ctrl.starting = true;
  ctrl.fatalError = null;
  ctrl.fatalInfo = null;
  renderMessage(
    ctrl,
    "🎥",
    "Menghubungkan Webcam Browser...",
    "Izinkan akses kamera saat browser menampilkan permintaan izin.",
    false
  );

  const videoConstraints = {
    width: { ideal: 640 },
    height: { ideal: 480 },
  };
  if (ctrl.deviceId) {
    videoConstraints.deviceId = { exact: ctrl.deviceId };
  } else {
    videoConstraints.facingMode = "user";
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: videoConstraints,
      audio: false,
    });
    if (ctrl.disposed) {
      stream.getTracks().forEach((t) => t.stop());
      ctrl.starting = false;
      return;
    }
    ctrl.stream = stream;
    ctrl.starting = false;
    ctrl.darkStreak = 0;

    if (ctrl.els && ctrl.els.video) {
      ctrl.els.video.srcObject = stream;
      await ctrl.els.video.play().catch(() => {});
    }
    hideMessage(ctrl);
    await populateDevices(ctrl);
    updateHud(ctrl);
  } catch (err) {
    ctrl.starting = false;
    ctrl.fatalError = (err && err.name) || "CameraError";
    let title = "Gagal Membuka Webcam Browser";
    let desc = (err && err.message) || "Periksa izin kamera di browser Anda.";
    if (ctrl.fatalError === "NotAllowedError" || ctrl.fatalError === "PermissionDeniedError") {
      title = "Izin Kamera Ditolak Browser";
      desc = "Klik ikon kamera/gembok di samping alamat URL browser, pilih 'Izinkan / Allow' untuk Kamera, lalu klik Coba Hubungkan Ulang.";
    } else if (ctrl.fatalError === "NotFoundError" || ctrl.fatalError === "DevicesNotFoundError") {
      title = "Webcam Tidak Ditemukan";
      desc = "Pastikan laptop memiliki webcam aktif atau kamera USB terpasang, lalu coba lagi.";
    } else if (ctrl.fatalError === "NotReadableError" || ctrl.fatalError === "TrackStartError") {
      title = "Webcam Sedang Dipakai Aplikasi Lain";
      desc = "Tutup Zoom, Google Meet, OBS, atau aplikasi Camera bawaan Windows yang sedang memakai kamera, lalu klik Coba Hubungkan Ulang.";
    }
    ctrl.fatalInfo = { icon: "⚠️", title, desc, showRetry: true };
    renderMessage(ctrl, "⚠️", title, desc, true);
    updateHud(ctrl);
  }
}

function captureTick(ctrl) {
  if (!ctrl || ctrl.disposed || !hasLiveStream(ctrl.stream)) return;
  const video = ctrl.els && ctrl.els.video;
  if (!video || !video.isConnected || video.readyState < 2 || video.videoWidth <= 0) return;
  if (typeof ctrl.sendFrame !== "function") return;

  const now = Date.now();
  if (ctrl.inFlight && now - ctrl.inFlightAt < 3800) return;
  if (now - ctrl.lastSentAt < ctrl.intervalMs) return;

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const maxW = 640;
  const scale = vw > maxW ? maxW / vw : 1.0;
  const cw = Math.max(160, Math.round(vw * scale));
  const ch = Math.max(120, Math.round(vh * scale));

  if (!ctrl.capCanvas) {
    ctrl.capCanvas = document.createElement("canvas");
  }
  ctrl.capCanvas.width = cw;
  ctrl.capCanvas.height = ch;
  const cctx = ctrl.capCanvas.getContext("2d", { willReadFrequently: true });
  if (!cctx) return;

  cctx.save();
  if (ctrl.mirror) {
    cctx.translate(cw, 0);
    cctx.scale(-1, 1);
  }
  cctx.drawImage(video, 0, 0, cw, ch);
  cctx.restore();

  const meanLuma = sampleMeanBrightness(cctx, cw, ch);
  if (meanLuma < 3.5) {
    ctrl.darkStreak = (ctrl.darkStreak || 0) + 1;
  } else {
    ctrl.darkStreak = 0;
  }

  let dataUrl = "";
  try {
    dataUrl = ctrl.capCanvas.toDataURL("image/jpeg", 0.78);
  } catch (_) {
    return;
  }

  ctrl.seq = (ctrl.seq || 0) + 1;
  ctrl.lastSentSeq = ctrl.seq;
  ctrl.inFlight = true;
  ctrl.inFlightAt = now;
  ctrl.lastSentAt = now;
  updateHud(ctrl);

  ctrl.sendFrame({
    seq: ctrl.seq,
    w: cw,
    h: ch,
    data_url: dataUrl,
  });
}

function drawRoundedRect(ctx, x, y, w, h, r) {
  const rad = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rad, y);
  ctx.arcTo(x + w, y, x + w, y + h, rad);
  ctx.arcTo(x + w, y + h, x, y + h, rad);
  ctx.arcTo(x, y + h, x, y, rad);
  ctx.arcTo(x, y, x + w, y, rad);
  ctx.closePath();
}

function drawBoxesOnCanvas(canvas, boxes) {
  if (!canvas || !canvas.isConnected) return;
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const targetW = Math.round(rect.width * dpr);
  const targetH = Math.round(rect.height * dpr);
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const list = Array.isArray(boxes) ? boxes : [];
  for (const b of list) {
    const bx = Math.max(0, Number(b.x || 0) * rect.width);
    const by = Math.max(0, Number(b.y || 0) * rect.height);
    const bw = Math.max(12, Number(b.w || 0) * rect.width);
    const bh = Math.max(12, Number(b.h || 0) * rect.height);
    const color = String(b.color || "#10B981");
    const label = String(b.label || "");
    const sub = String(b.sub || "");

    // Kotak utama halus
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.4;
    drawRoundedRect(ctx, bx, by, bw, bh, 8);
    ctx.stroke();

    // Aksen siku tegas di 4 sudut
    const corner = Math.max(10, Math.min(20, bw * 0.18, bh * 0.18));
    ctx.lineWidth = 4;
    ctx.beginPath();
    // Kiri atas
    ctx.moveTo(bx, by + corner); ctx.lineTo(bx, by); ctx.lineTo(bx + corner, by);
    // Kanan atas
    ctx.moveTo(bx + bw - corner, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + corner);
    // Kiri bawah
    ctx.moveTo(bx, by + bh - corner); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + corner, by + bh);
    // Kanan bawah
    ctx.moveTo(bx + bw - corner, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - corner);
    ctx.stroke();

    // Pil label di atas kotak
    if (label) {
      ctx.font = "700 12px Inter, -apple-system, Segoe UI, sans-serif";
      const mainW = ctx.measureText(label).width;
      ctx.font = "500 10.5px Inter, -apple-system, Segoe UI, sans-serif";
      const subW = sub ? ctx.measureText(sub).width : 0;
      const pillW = Math.min(rect.width - 12, Math.max(mainW, subW) + 18);
      const pillH = sub ? 34 : 22;
      let px = bx;
      if (px + pillW > rect.width - 6) px = Math.max(6, rect.width - pillW - 6);
      let py = by - pillH - 6;
      if (py < 38) py = Math.min(rect.height - pillH - 6, by + bh + 6);

      ctx.fillStyle = color;
      drawRoundedRect(ctx, px, py, pillW, pillH, 7);
      ctx.fill();

      ctx.fillStyle = "#FFFFFF";
      ctx.textBaseline = "top";
      ctx.font = "700 12px Inter, -apple-system, Segoe UI, sans-serif";
      ctx.fillText(label, px + 9, py + 4, pillW - 16);
      if (sub) {
        ctx.font = "500 10.5px Inter, -apple-system, Segoe UI, sans-serif";
        ctx.fillStyle = "rgba(255,255,255,0.92)";
        ctx.fillText(sub, px + 9, py + 19, pillW - 16);
      }
    }
  }
  ctx.restore();
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const cfg = data && typeof data === "object" ? data : {};
  const instKey = String(cfg.instance_id || "presensi_live_default");
  const pool = getPool();

  if (cfg.active === false) {
    stopController(instKey);
    return () => {};
  }

  let ctrl = pool[instKey];
  if (!ctrl) {
    ctrl = {
      stream: null,
      starting: false,
      disposed: false,
      deviceId: "",
      seq: 0,
      lastSentSeq: 0,
      inFlight: false,
      inFlightAt: 0,
      lastSentAt: 0,
      darkStreak: 0,
      fatalError: null,
      boxes: [],
      mirror: true,
      intervalMs: 500,
      lightBadge: "",
      hudText: "",
      hudKind: "info",
      stopTimer: null,
      tickTimer: null,
      capCanvas: null,
      els: null,
      sendFrame: null,
    };
    pool[instKey] = ctrl;
  }

  // Batalkan jadwal berhenti bila ini hanya re-render pembaruan data/box
  if (ctrl.stopTimer) {
    clearTimeout(ctrl.stopTimer);
    ctrl.stopTimer = null;
  }
  ctrl.disposed = false;

  // Perbarui konfigurasi & overlay terbaru dari Python
  ctrl.mirror = cfg.mirror !== false;
  ctrl.intervalMs = Math.max(280, Number(cfg.interval_ms || 500));
  ctrl.boxes = Array.isArray(cfg.boxes) ? cfg.boxes : [];
  ctrl.lightBadge = String(cfg.light_badge || "");
  ctrl.hudText = String(cfg.hud_text || "");
  ctrl.hudKind = String(cfg.hud_kind || "info");

  const ackSeq = Number(cfg.ack_seq || 0);
  if (ackSeq >= ctrl.lastSentSeq) {
    ctrl.inFlight = false;
  }
  ctrl.sendFrame = (payload) => setTriggerValue("frame", payload);

  const els = {
    stage: parentElement.querySelector(".cam-stage"),
    video: parentElement.querySelector(".cam-video"),
    overlay: parentElement.querySelector(".cam-overlay"),
    dotEl: parentElement.querySelector(".cam-dot"),
    liveTxt: parentElement.querySelector(".cam-live-txt"),
    lightBadgeEl: parentElement.querySelector(".cam-light-badge"),
    hudBottom: parentElement.querySelector(".cam-hud-bottom"),
    msgBox: parentElement.querySelector(".cam-msg-box"),
    msgIcon: parentElement.querySelector(".cam-msg-icon"),
    msgTitle: parentElement.querySelector(".cam-msg-title"),
    msgDesc: parentElement.querySelector(".cam-msg-desc"),
    retryBtn: parentElement.querySelector(".cam-retry-btn"),
    camSelect: parentElement.querySelector(".cam-select"),
    fpsEl: parentElement.querySelector(".cam-fps"),
  };
  ctrl.els = els;

  if (els.video) {
    els.video.classList.toggle("mirrored", ctrl.mirror);
  }
  if (els.retryBtn) {
    els.retryBtn.onclick = () => {
      ctrl.fatalError = null;
      ctrl.fatalInfo = null;
      stopTracksOnly(ctrl);
      startCamera(ctrl);
    };
  }
  if (els.camSelect) {
    els.camSelect.onchange = () => {
      ctrl.deviceId = els.camSelect.value || "";
      ctrl.fatalError = null;
      ctrl.fatalInfo = null;
      stopTracksOnly(ctrl);
      startCamera(ctrl);
    };
  }

  if (hasLiveStream(ctrl.stream)) {
    if (els.video && els.video.srcObject !== ctrl.stream) {
      els.video.srcObject = ctrl.stream;
    }
    if (els.video && els.video.paused) {
      els.video.play().catch(() => {});
    }
    hideMessage(ctrl);
    populateDevices(ctrl);
  } else if (ctrl.fatalError && ctrl.fatalInfo) {
    renderMessage(
      ctrl,
      ctrl.fatalInfo.icon,
      ctrl.fatalInfo.title,
      ctrl.fatalInfo.desc,
      ctrl.fatalInfo.showRetry
    );
  } else if (!ctrl.starting) {
    startCamera(ctrl);
  }

  if (!ctrl.tickTimer) {
    ctrl.tickTimer = setInterval(() => captureTick(ctrl), 140);
  }
  updateHud(ctrl);

  let rafActive = true;
  let rafId = 0;
  function renderLoop() {
    if (!rafActive) return;
    if (els.overlay) {
      drawBoxesOnCanvas(els.overlay, ctrl.boxes);
    }
    rafId = requestAnimationFrame(renderLoop);
  }
  rafId = requestAnimationFrame(renderLoop);

  return () => {
    rafActive = false;
    if (rafId) cancelAnimationFrame(rafId);
    ctrl.stopTimer = setTimeout(() => {
      ctrl.disposed = true;
      stopController(instKey);
    }, 900);
  };
}
"""

# Daftarkan SEKALI per proses Python (module-level cache via sys.modules)
_LIVE_WEBCAM_COMPONENT = st.components.v2.component(
    "presensi_live_webcam_v2",
    html=_LIVE_WEBCAM_HTML,
    css=_LIVE_WEBCAM_CSS,
    js=_LIVE_WEBCAM_JS,
    isolate_styles=True,
)


def decode_data_url_to_bgr(data_url: str) -> np.ndarray | None:
    """Mengubah string data URL JPEG/PNG base64 dari browser menjadi array BGR OpenCV."""
    if not data_url or not isinstance(data_url, str):
        return None
    try:
        encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
        raw_bytes = base64.b64decode(encoded)
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def build_overlay_boxes(
    face_results: list[dict[str, Any]],
    frame_shape: tuple[int, ...],
    streak_map: dict[str, int] | None = None,
    confirm_frames: int = 2,
) -> list[dict[str, Any]]:
    """Menyusun daftar bounding box ternormalisasi (0..1) untuk digambar pada canvas live."""
    if not face_results or len(frame_shape) < 2:
        return []
    fh, fw = int(frame_shape[0]), int(frame_shape[1])
    if fw <= 0 or fh <= 0:
        return []

    streaks = streak_map or {}
    boxes: list[dict[str, Any]] = []
    for r in face_results:
        x, y, w, h = r["bbox"]
        status = r.get("status", "Tidak Terdaftar")
        nim = str(r.get("nim", "-"))
        nama = str(r.get("nama", ""))
        score = float(r.get("score", 0.0))
        prob_real = float(r.get("prob_real", 0.0))

        if status == "Spoof":
            atk = str(r.get("attack_type", "Layar HP")).upper()
            color = "#EF4444"
            label = f"SPOOF: {atk}"
            sub = f"Diblokir · Real {prob_real:.2f}"
        elif status == "Sudah Absen":
            color = "#2563EB"
            label = f"SUDAH ABSEN (1x): {nama}"
            sub = "Terkunci maks. 1x per sesi"
        elif status == "Hadir":
            cur_streak = int(streaks.get(nim, 0))
            if cur_streak > 0 and cur_streak < int(confirm_frames):
                color = "#0EA5E9"
                label = f"MEMVERIFIKASI ({cur_streak}/{confirm_frames}): {nama}"
                sub = f"Tahan sebentar · Skor {score:.2f}"
            else:
                color = "#10B981"
                label = f"HADIR: {nama}"
                sub = f"Otomatis tercatat · Skor {score:.2f}"
        else:
            color = "#F59E0B"
            label = "TIDAK TERDAFTAR"
            sub = f"Belum enroll · Real {prob_real:.2f}"

        boxes.append({
            "x": round(max(0.0, float(x) / fw), 4),
            "y": round(max(0.0, float(y) / fh), 4),
            "w": round(min(1.0, float(w) / fw), 4),
            "h": round(min(1.0, float(h) / fh), 4),
            "color": color,
            "label": label,
            "sub": sub,
        })
    return boxes


def render_live_webcam(
    key: str,
    data: dict[str, Any],
    on_frame_change: Callable[[], None],
):
    """Merender komponen Webcam Live interaktif dan mengembalikan ComponentResult."""
    return _LIVE_WEBCAM_COMPONENT(
        key=key,
        data=data,
        on_frame_change=on_frame_change,
        width="stretch",
        height="content",
    )
