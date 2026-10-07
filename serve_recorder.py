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
import ipaddress
import json
import re
import socket
import ssl
import tempfile
import threading
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
        twin = train_user_twin(subject, manifest, foundation, n_baseline_recordings=5, epochs=100)
        JOBS[subject] = "done" if twin is not None else "error: need >= 6 recordings for this subject"
        print(f"[train_twin] subject {subject}: {JOBS[subject]}")
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
        for i in range(len(ds)):
            x, meta = ds[i]
            res = mon.analyze_recording(x)
            alert = res.get("alert") or res.get("status", "")
            results.append({
                "recording": i + 1,
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


class RecorderHandler(SimpleHTTPRequestHandler):
    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/twin_status":
            subj = _san(parse_qs(p.query).get("subject", ["100"])[0], 3)
            self._json(200, {"subject": subj, "trained": _twin_path(subj).exists(),
                             "recordings": _count_recordings(subj), "job": JOBS.get(subj)})
            return
        if p.path == "/dashboard_data":
            subj = _san(parse_qs(p.query).get("subject", ["100"])[0], 3)
            self._json(200, _dashboard_data(subj))
            return
        if p.path == "/heartprint_status":
            subj = _san(parse_qs(p.query).get("subject", ["100"])[0], 3)
            self._json(200, _heartprint_status(subj))
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

        if path == "/upload":
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
                self._json(200, _score(subj, rec))
            except Exception as e:  # noqa: BLE001
                self._json(500, {"ok": False, "error": str(e)})

        elif path == "/heartprint_enroll":
            subj = _san(data.get("subject", "100"), 3)
            rec = _san(data.get("recording", "01"), 2)
            bpm = data.get("detected_bpm", None)
            try:
                self._json(200, _heartprint_enroll(subj, rec, detected_bpm=bpm))
            except Exception as e:  # noqa: BLE001
                self._json(500, {"ok": False, "error": str(e)})

        elif path == "/heartprint_reset":
            subj = _san(data.get("subject", "100"), 3)
            try:
                self._json(200, _heartprint_reset(subj))
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
    from cryptography.x509.oid import NameOID
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
    ap = argparse.ArgumentParser(description="Serve the Digital Heart Twin recorder over HTTPS (+ real-twin endpoints).")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    args = ap.parse_args()
    if not (RECORDER_DIR / "index.html").exists():
        raise SystemExit(f"recorder/index.html not found under {RECORDER_DIR}. Run from the project root.")
    ip = get_lan_ip()
    cert_path, key_path = make_self_signed_cert(ip)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
    httpd = ThreadingHTTPServer((args.host, args.port), partial(RecorderHandler, directory=str(RECORDER_DIR)))
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    url = f"https://{ip}:{args.port}"
    print("=" * 60)
    print("  Digital Heart Twin Heart Recorder — HTTPS server (real-twin enabled)")
    print("=" * 60)
    print(f"  On your phone (same Wi-Fi):  {url}")
    print(f"  On this PC:                  https://localhost:{args.port}")
    print(f"  Uploads -> {DATA_RAW / 'MSCardio'}")
    print("  Endpoints: /upload /train_twin /twin_status /twin_score /dashboard_data")
    print("  Self-signed cert warning is expected -> Advanced -> proceed. Ctrl+C to stop.")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve_recorder] stopped.")
    finally:
        httpd.server_close()
        Path(cert_path).unlink(missing_ok=True)
        Path(key_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
