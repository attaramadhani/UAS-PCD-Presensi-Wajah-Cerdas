"""Modul manajemen database embedding wajah (One-Shot Enrollment) & penyimpanan presensi (SQLite, CSV, Excel)."""

from contextlib import contextmanager
import io
import os
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd


class StudentDatabase:
    """Mengelola penyimpanan face_embeddings.npz, SQLite presensi.db, arsip CSV, dan ekspor Excel (.xlsx)."""

    def __init__(self, db_dir, logs_dir):
        self.db_dir = db_dir
        self.logs_dir = logs_dir
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.emb_path = os.path.join(self.db_dir, "face_embeddings.npz")
        self.master_csv = os.path.join(self.db_dir, "students_master.csv")
        self.sqlite_path = os.path.join(self.db_dir, "presensi.db")
        self._init_sqlite()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.sqlite_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_sqlite(self):
        """Membuat skema tabel SQLite (attendance_sessions & attendance_records) jika belum ada."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attendance_sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_name TEXT NOT NULL,
                    opened_at TEXT,
                    closed_at TEXT NOT NULL,
                    total_registered INTEGER DEFAULT 0,
                    total_present INTEGER DEFAULT 0,
                    spoof_blocked INTEGER DEFAULT 0,
                    csv_filename TEXT DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attendance_records (
                    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    subject_id TEXT NOT NULL,
                    nama TEXT NOT NULL,
                    label TEXT DEFAULT '-',
                    status TEXT NOT NULL,
                    cosine_score REAL DEFAULT 0.0,
                    waktu_presensi TEXT DEFAULT '-',
                    FOREIGN KEY(session_id) REFERENCES attendance_sessions(session_id) ON DELETE CASCADE
                )
                """
            )

    def _load_full(self):
        """Memuat embeddings (N, 128), ID, Nama, dan Label per peserta secara internal."""
        if not os.path.exists(self.emb_path):
            empty_f = np.empty((0, 128), dtype=np.float32)
            empty_s = np.array([], dtype=str)
            return empty_f, empty_s, empty_s, empty_s

        data = np.load(self.emb_path, allow_pickle=False)
        embeddings = data["embeddings"].astype(np.float32)
        ids = data["nims"].astype(str)
        names = data["names"].astype(str)
        if "labels" in data.files:
            labels = data["labels"].astype(str)
        else:
            labels = np.array(["-"] * len(ids), dtype=str)
        return embeddings, ids, names, labels

    def load(self):
        """Memuat seluruh vektor embedding (N, 128), ID, dan Nama terdaftar."""
        embeddings, ids, names, _ = self._load_full()
        return embeddings, ids, names

    def save(self, embeddings, nims, names, labels=None):
        """Menyimpan embeddings dan memperbarui file master CSV."""
        if labels is None:
            labels = ["-" for _ in nims]

        np.savez(
            self.emb_path,
            embeddings=np.asarray(embeddings, dtype=np.float32),
            nims=np.asarray(nims, dtype=str),
            names=np.asarray(names, dtype=str),
            labels=np.asarray(labels, dtype=str),
        )
        df_master = pd.DataFrame({
            "ID": [str(x) for x in nims],
            "Nama": [str(x) for x in names],
            "Label": [str(x).strip() or "-" for x in labels],
        }).drop_duplicates(subset=["ID"]).sort_values("ID").reset_index(drop=True)
        df_master.to_csv(self.master_csv, index=False)

    def enroll(self, nim, nama, embedding, label="-"):
        """Mendaftarkan subjek baru atau memperbarui embedding (dirata-ratakan) jika ID sudah ada."""
        id_clean = str(nim).strip()
        nama_clean = str(nama).strip()
        label_clean = str(label).strip() or "-"

        emb_vec = np.asarray(embedding, dtype=np.float32).reshape(128)
        norm = np.linalg.norm(emb_vec)
        if norm > 0:
            emb_vec = emb_vec / norm

        embeddings, ids, names, labels = self._load_full()
        if len(ids) > 0 and id_clean in ids:
            idx = int(np.where(ids == id_clean)[0][0])
            avg_vec = (embeddings[idx] + emb_vec) / 2.0
            avg_norm = np.linalg.norm(avg_vec)
            embeddings[idx] = avg_vec / avg_norm if avg_norm > 0 else avg_vec
            names[idx] = nama_clean
            labels[idx] = label_clean
            status = "diperbarui"
        else:
            embeddings = np.vstack([embeddings, emb_vec]) if len(embeddings) > 0 else emb_vec.reshape(1, 128)
            ids = np.append(ids, id_clean)
            names = np.append(names, nama_clean)
            labels = np.append(labels, label_clean)
            status = "ditambahkan"

        self.save(embeddings, ids, names, labels)
        return status

    def delete(self, nim):
        """Menghapus subjek terdaftar berdasarkan ID dari database."""
        id_clean = str(nim).strip()
        embeddings, ids, names, labels = self._load_full()
        if len(ids) == 0 or id_clean not in ids:
            return False

        keep_mask = ids != id_clean
        self.save(embeddings[keep_mask], ids[keep_mask], names[keep_mask], labels[keep_mask])
        return True

    def list_students(self, id_col="ID", label_col="Label"):
        """Mengembalikan DataFrame daftar subjek terdaftar dengan nama kolom dapat dikustomisasi."""
        if os.path.exists(self.master_csv):
            df = pd.read_csv(self.master_csv, dtype=str).fillna("-")
            if "NIM" in df.columns and "ID" not in df.columns:
                df = df.rename(columns={"NIM": "ID"})
            if "Label" not in df.columns:
                df["Label"] = "-"
        else:
            _, ids, names, labels = self._load_full()
            df = pd.DataFrame({"ID": ids, "Nama": names, "Label": labels})

        return df[["ID", "Nama", "Label"]].rename(columns={"ID": id_col, "Label": label_col})

    def build_attendance_table(
        self,
        recognized_records,
        id_col="ID",
        label_col="Label",
        absent_label="Belum Hadir",
        manual_status_map=None,
    ):
        """Menyusun tabel rekap kehadiran seluruh subjek terdaftar termasuk koreksi status manual (Hadir/Izin/Sakit)."""
        master_df = self.list_students(id_col="ID", label_col="Label")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        manual_map = manual_status_map or {}

        present_map = {}
        present_time = {}
        for rec in recognized_records:
            rec_id = rec.get("nim")
            score = float(rec.get("score", 0.0))
            rec_time = rec.get("time") or now_str
            if rec_id:
                if rec_id not in present_time:
                    present_time[rec_id] = rec_time
                if rec_id not in present_map or score > present_map[rec_id]:
                    present_map[rec_id] = score

        rows = []
        for _, mrow in master_df.iterrows():
            rec_id = str(mrow["ID"])
            nama = str(mrow["Nama"])
            lbl = str(mrow["Label"])

            if rec_id in manual_map:
                m_entry = manual_map[rec_id]
                rows.append({
                    id_col: rec_id,
                    "Nama": nama,
                    label_col: lbl,
                    "Status": str(m_entry.get("status", absent_label)),
                    "Cosine Score": round(float(m_entry.get("score", 0.0)), 4),
                    "Waktu Presensi": str(m_entry.get("time", now_str)),
                })
            elif rec_id in present_map:
                rows.append({
                    id_col: rec_id,
                    "Nama": nama,
                    label_col: lbl,
                    "Status": "Hadir",
                    "Cosine Score": round(present_map[rec_id], 4),
                    "Waktu Presensi": present_time[rec_id],
                })
            else:
                rows.append({
                    id_col: rec_id,
                    "Nama": nama,
                    label_col: lbl,
                    "Status": absent_label,
                    "Cosine Score": 0.0,
                    "Waktu Presensi": "-",
                })
        return pd.DataFrame(rows)

    def save_attendance_log(self, attendance_df, session_name="Sesi"):
        """Menyimpan rekap presensi ke folder attendance_logs sebagai file CSV berstempel waktu."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_session = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in session_name)
        fname = f"presensi_{safe_session}_{ts}.csv"
        out_path = os.path.join(self.logs_dir, fname)
        attendance_df.to_csv(out_path, index=False)
        return out_path

    def save_session_to_db(
        self,
        session_name,
        opened_at,
        closed_at,
        attendance_df,
        spoof_blocked=0,
        csv_filename="",
    ):
        """Menyimpan metadata sesi dan seluruh baris rekap kehadiran ke database SQLite presensi.db."""
        if attendance_df is None or attendance_df.empty:
            total_reg = 0
            total_hadir = 0
        else:
            total_reg = int(len(attendance_df))
            total_hadir = int(attendance_df["Status"].astype(str).str.startswith("Hadir").sum())

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO attendance_sessions
                    (session_name, opened_at, closed_at, total_registered, total_present, spoof_blocked, csv_filename)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(session_name),
                    str(opened_at or "-"),
                    str(closed_at),
                    total_reg,
                    total_hadir,
                    int(spoof_blocked),
                    os.path.basename(str(csv_filename)) if csv_filename else "",
                ),
            )
            session_id = int(cur.lastrowid)

            if attendance_df is not None and not attendance_df.empty:
                cols = list(attendance_df.columns)
                id_c = cols[0]
                nama_c = "Nama" if "Nama" in cols else cols[1]
                lbl_c = cols[2] if len(cols) > 2 and cols[2] not in ("Status", "Cosine Score", "Waktu Presensi") else None

                records_rows = []
                for _, row in attendance_df.iterrows():
                    records_rows.append((
                        session_id,
                        str(row.get(id_c, "-")),
                        str(row.get(nama_c, "-")),
                        str(row.get(lbl_c, "-")) if lbl_c else "-",
                        str(row.get("Status", "-")),
                        float(row.get("Cosine Score", 0.0)),
                        str(row.get("Waktu Presensi", "-")),
                    ))
                conn.executemany(
                    """
                    INSERT INTO attendance_records
                        (session_id, subject_id, nama, label, status, cosine_score, waktu_presensi)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    records_rows,
                )
        return session_id

    def list_sessions_db(self):
        """Mengembalikan daftar seluruh sesi presensi yang tersimpan di SQLite (terbaru di atas)."""
        with self._connect() as conn:
            df = pd.read_sql_query(
                """
                SELECT session_id AS "ID Sesi",
                       session_name AS "Nama Sesi",
                       opened_at AS "Dibuka",
                       closed_at AS "Ditutup",
                       total_present AS "Hadir",
                       total_registered AS "Total",
                       spoof_blocked AS "Spoof",
                       csv_filename AS "File CSV"
                FROM attendance_sessions
                ORDER BY session_id DESC
                """,
                conn,
            )
        return df

    def get_session_records_db(self, session_id, id_col="ID", label_col="Label"):
        """Mengembalikan tabel detail kehadiran dari satu sesi tersimpan di SQLite."""
        with self._connect() as conn:
            df = pd.read_sql_query(
                """
                SELECT subject_id, nama, label, status, cosine_score, waktu_presensi
                FROM attendance_records
                WHERE session_id = ?
                ORDER BY subject_id ASC
                """,
                conn,
                params=(int(session_id),),
            )
        return df.rename(columns={
            "subject_id": id_col,
            "nama": "Nama",
            "label": label_col,
            "status": "Status",
            "cosine_score": "Cosine Score",
            "waktu_presensi": "Waktu Presensi",
        })

    def delete_session(self, session_id=None, csv_filename=None):
        """Menghapus sesi presensi dari SQLite (cascade records) beserta arsip CSV terkait."""
        deleted = False
        target_csv = os.path.basename(csv_filename) if csv_filename else ""

        if session_id is not None:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT csv_filename FROM attendance_sessions WHERE session_id = ?",
                    (int(session_id),),
                ).fetchone()
                if row is not None:
                    if not target_csv and row[0]:
                        target_csv = os.path.basename(str(row[0]))
                    conn.execute("DELETE FROM attendance_sessions WHERE session_id = ?", (int(session_id),))
                    deleted = True
        elif target_csv:
            with self._connect() as conn:
                cur = conn.execute("DELETE FROM attendance_sessions WHERE csv_filename = ?", (target_csv,))
                if cur.rowcount > 0:
                    deleted = True

        if target_csv:
            csv_path = os.path.join(self.logs_dir, target_csv)
            if os.path.exists(csv_path) and os.path.isfile(csv_path):
                os.remove(csv_path)
                deleted = True
        return deleted

    def delete_all_sessions(self):
        """Menghapus seluruh riwayat sesi presensi di SQLite dan seluruh arsip CSV di attendance_logs."""
        count = 0
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM attendance_sessions")
            count = int(cur.rowcount)
        if os.path.exists(self.logs_dir):
            for fn in os.listdir(self.logs_dir):
                if fn.lower().endswith(".csv"):
                    fp = os.path.join(self.logs_dir, fn)
                    if os.path.isfile(fp):
                        os.remove(fp)
                        count = max(count, 1)
        return count

    @staticmethod
    def export_excel_bytes(df, sheet_name="Rekap_Presensi"):
        """Mengonversi DataFrame presensi ke file Excel (.xlsx) rapi dengan lebar kolom otomatis."""
        buf = io.BytesIO()
        safe_sheet = "".join(c if c.isalnum() or c in ("_", " ") else "_" for c in str(sheet_name))[:31] or "Rekap"
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=safe_sheet)
            ws = writer.sheets[safe_sheet]
            for col_idx, col_name in enumerate(df.columns, 1):
                max_len = max(
                    len(str(col_name)),
                    int(df[col_name].astype(str).map(len).max()) if not df.empty else 0,
                )
                col_letter = ws.cell(row=1, column=col_idx).column_letter
                ws.column_dimensions[col_letter].width = min(42, max(12, max_len + 4))
        buf.seek(0)
        return buf.getvalue()


ParticipantDatabase = StudentDatabase
