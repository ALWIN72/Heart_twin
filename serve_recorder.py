"""
serve_recorder.py

HTTPS server for the browser heart recorder, with:
  - GET  /                serves recorder/index.html (+ assets)
  - POST /upload          writes a recording into data/raw/Digital Heart Twin/...
  - POST /train_twin      trains the real PersonalHeartTwin for a subject (background thread)
  - GET  /twin_status     reports whether a twin exists / training state / recording count
  - POST /twin_score      scores an uploaded recording with the REAL trained models
                          (placement -> harmonize -> personal-twin KL anomaly score)
  - GET  /dashboard_data  returns all scores + alerts for a subject (for mobile dashboard tab)

Phone browsers expose motion sensors only over HTTPS, so a throwaway
self-signed certificate is generated automatically. The phone will warn about
it -> Advanced -> proceed.

Usage:  python serve_recorder.py [--port 8443]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import ipaddress
import json
import re
import secrets
import socket
import ssl
import tempfile
import threading
import time
import uuid
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PROJECT_ROOT = Path(__file__).resolve().parent
RECORDER_DIR = PROJECT_ROOT / "recorder"
DATA_RAW = PROJECT_ROOT
CKPT_DIR = PROJECT_ROOT / "checkpoints"
ALLOWED_FILES = {"scg.csv", "Uncalibrated_scg.csv", "gyro.csv"}

# subject -> "training" | "done" | "error: ..."   (in-memory training job state)
JOBS: dict[str, str] = {}

# SQLite Database & Authentication Integration
from src.db import (
    auth_register as _auth_register,
    auth_login as _auth_login,
    get_user_by_token as _get_user_by_token,
    auth_logout as _auth_logout,
    auth_update_profile as _auth_update_profile,
    list_patient_profiles as _list_patient_profiles,
    record_upload,
    record_twin_score,
    record_heartprint,
    record_twin_training,
)



def _san(v, width: int) -> str:
    s = re.sub(r"[^0-9A-Za-z]", "", str(v))[:8]
    if s.isdigit():
        s = s.zfill(width)
    return s or ("0" * width)


def _twin_path(subject: str) -> Path:
    return CKPT_DIR / f"twin_user_{subject}.pt"


def _count_recordings(subject: str) -> int:
    d = DATA_RAW / "MSCardio" / f"Subject_{subject}"
    return len(list(d.glob("Recording_*"))) if d.exists() else 0


def _do_train(subject: str):
    """Background: train the real per-user twin from uploaded recordings."""
    try:
        JOBS[subject] = "training"
        from src.config import ModelConfig
        from src.data_loader import build_multimodal_manifest
        from src.training.train_personal_twin import train_user_twin, load_foundation_encoder
        cfg = ModelConfig()
        manifest = build_multimodal_manifest(raw_data_path=str(DATA_RAW))
        foundation = load_foundation_encoder(cfg)
        twin = train_user_twin(subject, manifest, foundation, n_baseline_recordings=5, epochs=25)
        JOBS[subject] = "done" if twin is not None else "error: need >= 6 recordings for this subject"
        print(f"[train_twin] subject {subject}: {JOBS[subject]}")
        record_twin_training(subject, checkpoint_path=str(_twin_path(subject)), status=JOBS[subject], count=_count_recordings(subject))
    except Exception as e:  # noqa: BLE001
        JOBS[subject] = f"error: {e}"
        print(f"[train_twin] subject {subject} FAILED: {e}")


def _score(subject: str, recording: str) -> dict:
    """Score one uploaded recording with the real trained models."""
    from src.data_loader import build_multimodal_manifest
    from src.dataset import SCGWindowDataset
    from src.utils.anomaly_scorer import HealthMonitor
    manifest = build_multimodal_manifest(raw_data_path=str(DATA_RAW))
    df = manifest[(manifest["subject_id"].astype(str) == subject) &
                  (manifest["recording_id"].astype(str) == recording)]
    if df.empty:
        return {"ok": False, "error": f"recording {recording} for subject {subject} not found"}
    ds = SCGWindowDataset(df)
    if len(ds) == 0:
        return {"ok": False, "error": "no analyzable window in that recording"}
    x, _ = ds[0]
    mon = HealthMonitor.for_user(subject)
    res = mon.analyze_recording(x)
    if "error" in res:
        if "No trained PersonalHeartTwin found" in res["error"]:
            if _count_recordings(subject) >= 6:
                if JOBS.get(subject) != "training":
                    threading.Thread(target=_do_train, args=(subject,), daemon=True).start()
                return {"ok": False, "error": "New user detected. Training started automatically in the background. Please wait a few minutes and try again."}
            else:
                return {"ok": False, "error": "New user detected. Please upload at least 6 recordings to train your model."}
        return {"ok": False, "error": res["error"]}
    return {"ok": True, "score": res.get("score"), "placement": res.get("placement"),
            "alert": res.get("alert") or res.get("status", "Scored against your trained twin.")}


def _dashboard_data(subject: str) -> dict:
    """Return all per-recording scores for a subject — powers the mobile Dashboard tab."""
    try:
        from src.config import TwinConfig
        from src.data_loader import build_multimodal_manifest
        from src.dataset import SCGWindowDataset
        from src.utils.anomaly_scorer import HealthMonitor

        if not _twin_path(subject).exists():
            if _count_recordings(subject) >= 6 and JOBS.get(subject) != "training":
                threading.Thread(target=_do_train, args=(subject,), daemon=True).start()
                return {"ok": False, "error": "New user detected. Training started automatically in the background. Please refresh in a few minutes.",
                        "trained": False, "recordings": _count_recordings(subject)}
            
            return {"ok": False, "error": "No trained twin for this subject yet. Upload at least 6 recordings to auto-train.",
                    "trained": False, "recordings": _count_recordings(subject)}

        manifest = build_multimodal_manifest(raw_data_path=str(DATA_RAW))
        sub_df = manifest[manifest["subject_id"].astype(str) == subject]
        if sub_df.empty:
            return {"ok": False, "error": "No recordings found on disk for this subject.",
                    "trained": True, "recordings": 0}

        sub_df = sub_df.sort_values("recording_id")
        ds = SCGWindowDataset(sub_df)
        cfg = TwinConfig()
        mon = HealthMonitor.for_user(subject)
        mon.reset_history()

        results = []
        unique_recs = set()
        for i in range(len(ds)):
            x, meta = ds[i]
            res = mon.analyze_recording(x)
            alert = res.get("alert") or res.get("status", "")
            rec_id = str(meta.get("recording_id", i + 1))
            win_idx = int(meta.get("window_idx", 0)) + 1
            unique_recs.add(rec_id)
            results.append({
                "recording": i + 1,
                "rec_id": rec_id,
                "window": win_idx,
                "score": round(float(res.get("score", 0)), 4),
                "alert": alert,
                "placement": res.get("placement", ""),
                "flagged": float(res.get("score", 0)) > cfg.anomaly_threshold,
            })

        return {
            "ok": True,
            "subject": subject,
            "trained": True,
            "recordings": len(results),
            "raw_recordings": len(unique_recs),
            "threshold": round(float(cfg.anomaly_threshold), 4),
            "job": JOBS.get(subject),
            "results": results,
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def _heartprint_engine():
    """Lazily load HeartPrintEngine singleton."""
    global _HEARTPRINT_ENGINE
    if "_HEARTPRINT_ENGINE" not in globals() or _HEARTPRINT_ENGINE is None:
        from src.heartprint import HeartPrintEngine
        _HEARTPRINT_ENGINE = HeartPrintEngine(checkpoint_dir=CKPT_DIR)
    return _HEARTPRINT_ENGINE


def _heartprint_status(subject: str) -> dict:
    """Return HeartPrint enrollment state and puzzle piece layout for subject."""
    try:
        engine = _heartprint_engine()
        state = engine.get_state(subject)
        pieces_info = [
            {
                "piece_id": p.piece_id,
                "index": p.index,
                "grid_row": p.grid_row,
                "grid_col": p.grid_col,
                "center_x": p.center_x,
                "center_y": p.center_y,
                "svg_path": p.svg_path,
                "established": state.piece_states.get(p.piece_id, {}).get("established", False),
                "stability": state.piece_states.get(p.piece_id, {}).get("stability", 0.0),
                "observations": state.piece_states.get(p.piece_id, {}).get("observations", 0),
            }
            for p in engine.puzzle.pieces
        ]
        return {
            "ok": True,
            "subject": subject,
            "state": state.to_dict(),
            "puzzle_pieces": pieces_info,
            "total_pieces": engine.cfg.total_pieces,
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def _heartprint_enroll(subject: str, recording: str, detected_bpm: float | None = None) -> dict:
    """Process a recording through the real HeartPrint progressive enrollment pipeline."""
    try:
        engine = _heartprint_engine()
        res = engine.process_saved_recording(
            subject_id=subject, recording_id=recording, raw_root=DATA_RAW
        )
        if detected_bpm:
            res["detected_bpm"] = detected_bpm
        return res
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def _heartprint_reset(subject: str) -> dict:
    """Reset HeartPrint enrollment for a subject."""
    try:
        engine = _heartprint_engine()
        state = engine.reset_enrollment(subject)
        return {"ok": True, "subject": subject, "state": state.to_dict()}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def _find_subject_dir(subject: str) -> Path | None:
    for candidate in [
        f"Subject_{subject}",
        f"Subject_{str(subject).zfill(4)}",
        f"Subject_{str(subject).zfill(3)}",
        f"Subject_{str(subject).lstrip('0')}",
    ]:
        d = DATA_RAW / "MSCardio" / candidate
        if d.exists() and d.is_dir():
            return d
    return None


def _subject_recordings(subject: str) -> dict:
    d = _find_subject_dir(subject)
    if not d:
        return {"ok": True, "subject": subject, "found": False, "recordings": []}
    recs = []
    for rdir in sorted(d.glob("Recording_*")):
        rname = rdir.name
        rid = rname.replace("Recording_", "")
        files = [f.name for f in rdir.iterdir() if f.is_file()]
        recs.append({
            "id": rid,
            "name": rname,
            "files": files,
            "has_scg": any(x in files for x in ("scg.csv", "Uncalibrated_scg.csv", "uncalibrated_scg.csv")),
        })
    return {"ok": True, "subject": subject, "found": True, "count": len(recs), "recordings": recs}


def _load_recording_data(subject: str, recording: str) -> dict:
    d = _find_subject_dir(subject)
    if not d:
        return {"ok": False, "error": f"Subject {subject} directory not found"}
    candidates = [
        d / f"Recording_{recording}",
        d / f"Recording_{recording.zfill(2)}",
        d / f"Recording_{recording.zfill(3)}",
    ]
    rdir = next((c for c in candidates if c.exists() and c.is_dir()), None)
    if not rdir:
        return {"ok": False, "error": f"Recording {recording} not found in {d.name}"}

    csv_file = None
    for name in ["scg.csv", "Uncalibrated_scg.csv", "uncalibrated_scg.csv"]:
        if (rdir / name).exists():
            csv_file = rdir / name
            break
    if not csv_file:
        return {"ok": False, "error": "No SCG CSV file found in recording directory"}

    rows = []
    with open(csv_file, "r", encoding="utf-8") as f:
        header = [h.strip() for h in f.readline().strip().split(",")]
        t_idx = 0
        if "seconds_elapsed" in header:
            t_idx = header.index("seconds_elapsed")
        elif "timestamp" in header:
            t_idx = header.index("timestamp")
        x_idx = header.index("x") if "x" in header else 1
        y_idx = header.index("y") if "y" in header else 2
        z_idx = header.index("z") if "z" in header else 3

        t0 = None
        for line in f:
            parts = line.strip().split(",")
            if len(parts) > max(t_idx, x_idx, y_idx, z_idx):
                try:
                    raw_t = float(parts[t_idx])
                    if t0 is None:
                        t0 = raw_t
                    t_val = (raw_t - t0) * 1e-9 if raw_t > 1e12 else (raw_t - t0 if raw_t > 1e6 else raw_t)
                    x_val = float(parts[x_idx])
                    y_val = float(parts[y_idx])
                    z_val = float(parts[z_idx])
                    rows.append([round(t_val, 4), round(x_val, 5), round(y_val, 5), round(z_val, 5)])
                    if len(rows) >= 6000:
                        break
                except ValueError:
                    continue

    return {
        "ok": True,
        "subject": subject,
        "recording": recording,
        "total_samples": len(rows),
        "data": rows,
    }


def _delete_recording(subject: str, recording: str) -> dict:
    d = _find_subject_dir(subject)
    if not d:
        return {"ok": False, "error": f"Subject {subject} directory not found"}
    import shutil
    candidates = [
        d / f"Recording_{recording}",
        d / f"Recording_{recording.zfill(2)}",
        d / f"Recording_{recording.zfill(3)}",
    ]
    rdir = next((c for c in candidates if c.exists() and c.is_dir()), None)
    if rdir:
        shutil.rmtree(rdir, ignore_errors=True)
        return {"ok": True, "deleted": True, "path": str(rdir)}
    return {"ok": True, "deleted": False, "note": "Recording directory not found"}


class RecorderHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".svg": "image/svg+xml",
        ".wasm": "application/wasm",
    }

    def guess_type(self, path):
        p = str(path).lower()
        if p.endswith(".js") or p.endswith(".mjs"):
            return "text/javascript; charset=utf-8"
        if p.endswith(".css"):
            return "text/css; charset=utf-8"
        if p.endswith(".json"):
            return "application/json; charset=utf-8"
        if p.endswith(".svg"):
            return "image/svg+xml"
        return super().guess_type(path)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Clear-Site-Data", '"cache"')
        super().end_headers()

    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._set_cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _get_auth_token(self) -> str:
        # Check Authorization header
        auth_hdr = self.headers.get("Authorization", "")
        if auth_hdr.startswith("Bearer "):
            return auth_hdr[7:].strip()
        # Check Cookie header
        cookies = self.headers.get("Cookie", "")
        for cookie in cookies.split(";"):
            if "=" in cookie:
                k, v = cookie.strip().split("=", 1)
                if k.strip() == "dht_token":
                    return v.strip()
        # Check query string
        p = urlparse(self.path)
        qs = parse_qs(p.query)
        return qs.get("token", [""])[0]

    def do_GET(self):
        p = urlparse(self.path)
        qs = parse_qs(p.query)

        if p.path in ("/app", "/m", "/home", "/live"):
            self.path = "/index.html"
            return super().do_GET()

        if p.path == "/auth/me":
            token = self._get_auth_token()
            user = _get_user_by_token(token)
            if user:
                self._json(200, {"ok": True, "user": user})
            else:
                self._json(401, {"ok": False, "error": "Not authenticated"})
            return

        if p.path == "/auth/profiles":
            self._json(200, _list_patient_profiles())
            return

        if p.path == "/twin_status":
            subj = _san(qs.get("subject", ["100"])[0], 3)
            self._json(200, {"subject": subj, "trained": _twin_path(subj).exists(),
                             "recordings": _count_recordings(subj), "job": JOBS.get(subj)})
            return
        if p.path == "/dashboard_data":
            subj = _san(qs.get("subject", ["100"])[0], 3)
            self._json(200, _dashboard_data(subj))
            return
        if p.path == "/heartprint_status":
            subj = _san(qs.get("subject", ["100"])[0], 3)
            self._json(200, _heartprint_status(subj))
            return
        if p.path == "/subject_recordings":
            subj = qs.get("subject", ["100"])[0]
            self._json(200, _subject_recordings(subj))
            return
        if p.path == "/load_recording_data":
            subj = qs.get("subject", ["100"])[0]
            rec = qs.get("recording", ["01"])[0]
            self._json(200, _load_recording_data(subj, rec))
            return
        if p.path in ("/auth", "/heartprint"):
            self.path = "/auth.html"
            return super().do_GET()
        return super().do_GET()

    def do_POST(self):
        path = self.path.rstrip("/")
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            data = {}

        if path == "/auth/register":
            res = _auth_register(data)
            self._json(200 if res.get("ok") else 400, res)

        elif path == "/auth/login":
            res = _auth_login(data)
            self._json(200 if res.get("ok") else 401, res)

        elif path == "/auth/logout":
            token = self._get_auth_token()
            self._json(200, _auth_logout(token))

        elif path == "/auth/update_profile":
            token = self._get_auth_token()
            res = _auth_update_profile(token, data)
            self._json(200 if res.get("ok") else 400, res)

        elif path == "/upload":
            try:
                subj = _san(data.get("subject", "100"), 3)
                rec = _san(data.get("recording", "01"), 2)
                base = DATA_RAW / "MSCardio" / f"Subject_{subj}" / f"Recording_{rec}"
                base.mkdir(parents=True, exist_ok=True)
                written = []
                for name, content in (data.get("files") or {}).items():
                    if name in ALLOWED_FILES and isinstance(content, str):
                        (base / name).write_text(content, encoding="utf-8")
                        written.append(name)
                meta = data.get("metadata")
                if meta:
                    mp = base.parent / "general_metadata.json"
                    if not mp.exists():
                        mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                print(f"[upload] Subject_{subj}/Recording_{rec}: {', '.join(written) or '(nothing)'}")
                user_obj = _get_user_by_token(self._get_auth_token())
                user_id = user_obj["id"] if user_obj else None
                record_upload(subj, rec, written, storage_path=str(base), metadata=meta, user_id=user_id)
                self._json(200, {"ok": True, "subject": subj, "recording": rec, "written": written})
            except Exception as e:  # noqa: BLE001
                self._json(500, {"ok": False, "error": str(e)})

        elif path == "/train_twin":
            subj = _san(data.get("subject", "100"), 3)
            if JOBS.get(subj) == "training":
                self._json(200, {"ok": True, "started": False, "subject": subj, "note": "already training"})
                return
            threading.Thread(target=_do_train, args=(subj,), daemon=True).start()
            self._json(200, {"ok": True, "started": True, "subject": subj})

        elif path == "/twin_score":
            subj = _san(data.get("subject", "100"), 3)
            rec = _san(data.get("recording", "01"), 2)
            try:
                res = _score(subj, rec)
                if res.get("ok"):
                    record_twin_score(subj, rec, float(res.get("score", 0)), res.get("placement", ""), res.get("alert", ""))
                self._json(200, res)
            except Exception as e:  # noqa: BLE001
                self._json(500, {"ok": False, "error": str(e)})

        elif path == "/heartprint_enroll":
            subj = _san(data.get("subject", "100"), 3)
            rec = _san(data.get("recording", "01"), 2)
            bpm = data.get("detected_bpm", None)
            try:
                res = _heartprint_enroll(subj, rec, detected_bpm=bpm)
                if res.get("ok"):
                    record_heartprint(subj, rec, bpm=bpm, quality=res.get("learning_score"))
                self._json(200, res)
            except Exception as e:  # noqa: BLE001
                self._json(500, {"ok": False, "error": str(e)})

        elif path == "/heartprint_reset":
            subj = _san(data.get("subject", "100"), 3)
            try:
                self._json(200, _heartprint_reset(subj))
            except Exception as e:  # noqa: BLE001
                self._json(500, {"ok": False, "error": str(e)})

        elif path == "/delete_recording":
            subj = data.get("subject", "100")
            rec = data.get("recording", "01")
            try:
                self._json(200, _delete_recording(subj, rec))
            except Exception as e:  # noqa: BLE001
                self._json(500, {"ok": False, "error": str(e)})

        else:
            self.send_error(404, "not found")


def get_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def make_self_signed_cert(ip: str) -> tuple[str, str]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, ip)])
    alt = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    try:
        alt.append(x509.IPAddress(ipaddress.ip_address(ip)))
    except ValueError:
        pass
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.SubjectAlternativeName(alt), critical=False)
            .sign(key, hashes.SHA256()))
    cf = tempfile.NamedTemporaryFile(prefix="digital_heart_twin_cert_", suffix=".pem", delete=False)
    kf = tempfile.NamedTemporaryFile(prefix="digital_heart_twin_key_", suffix=".pem", delete=False)
    cf.write(cert.public_bytes(serialization.Encoding.PEM))
    kf.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                               serialization.NoEncryption()))
    cf.close(); kf.close()
    return cf.name, kf.name


def main():
    import os
    default_port = int(os.environ.get("PORT", 9443))
    is_cloud = bool(os.environ.get("RENDER") or os.environ.get("PORT"))
    ap = argparse.ArgumentParser(description="Serve the Digital Heart Twin recorder (+ real-twin endpoints).")
    ap.add_argument("--port", type=int, default=default_port)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--http", action="store_true", default=is_cloud, help="Serve plain HTTP (for cloud proxies like Render that terminate SSL)")
    args = ap.parse_args()
    if not (RECORDER_DIR / "index.html").exists():
        raise SystemExit(f"recorder/index.html not found under {RECORDER_DIR}. Run from the project root.")

    httpd = ThreadingHTTPServer((args.host, args.port), partial(RecorderHandler, directory=str(RECORDER_DIR)))
    cert_path, key_path = None, None

    if not args.http:
        ip = get_lan_ip()
        cert_path, key_path = make_self_signed_cert(ip)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        url = f"https://{ip}:{args.port}"
    else:
        url = f"http://0.0.0.0:{args.port}"

    print("=" * 60)
    print("  Digital Heart Twin Heart Recorder — Server (real-twin enabled)")
    print("=" * 60)
    print(f"  Listening on:                {url}")
    print(f"  Uploads -> {DATA_RAW / 'MSCardio'}")
    print("  Endpoints: /upload /train_twin /twin_status /twin_score /dashboard_data")
    if not args.http:
        print("  Self-signed cert warning is expected on phone -> Advanced -> proceed.")
    else:
        print("  Running behind cloud SSL proxy (Render / Railway).")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve_recorder] stopped.")
    finally:
        httpd.server_close()
        if cert_path:
            Path(cert_path).unlink(missing_ok=True)
        if key_path:
            Path(key_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
