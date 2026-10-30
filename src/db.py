"""
src/db.py - Robust SQLite Database & Authentication Engine for Digital Heart Twin

Features:
- Persistent storage for user profiles, credentials, sessions, and telemetry recordings.
- Automatic, collision-free assignment of Subject IDs (starts at 1001 for new users).
- Users never need to know, guess, or type research Subject IDs.
- Secure salted SHA-256 password hashing and 30-day session tokens ("Remember Me").
- Safe auto-migration from legacy users_db.json if present.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "heart_twin.db"

def get_db_path() -> Path:
    env_path = os.environ.get("HEART_TWIN_DB")
    if env_path:
        p = Path(env_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    if os.environ.get("VERCEL") or not os.access(str(PROJECT_ROOT), os.W_OK):
        return Path("/tmp/heart_twin.db")
    return DEFAULT_DB_PATH

def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    p = db_path or get_db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=20.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(db_path: Path | None = None):
    """Initialize SQLite database tables and run migration from users_db.json if needed."""
    p = db_path or get_db_path()
    conn = get_connection(p)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                subject_id TEXT UNIQUE NOT NULL,
                age INTEGER,
                gender TEXT,
                created_at TEXT NOT NULL,
                notes TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                subject_id TEXT NOT NULL,
                recording_id TEXT NOT NULL,
                files TEXT,
                sample_count INTEGER DEFAULT 0,
                detected_bpm REAL,
                quality_score REAL,
                anomaly_score REAL,
                placement TEXT,
                alert TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                storage_path TEXT,
                UNIQUE(subject_id, recording_id)
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS twins (
                subject_id TEXT PRIMARY KEY,
                user_id TEXT,
                checkpoint_path TEXT,
                status TEXT DEFAULT 'ready',
                recordings_count INTEGER DEFAULT 0,
                trained_at TEXT
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_subject_id ON users(subject_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_recordings_subject ON recordings(subject_id);")

    # Migrate legacy users if needed
    _migrate_legacy_users(conn)
    conn.close()

def _hash_pw(password: str, salt: str) -> str:
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()

def _migrate_legacy_users(conn: sqlite3.Connection):
    """Migrate accounts from users_db.json into SQLite if table is currently empty."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] > 0:
        return

    legacy_file = PROJECT_ROOT / "users_db.json"
    if not legacy_file.exists():
        # Insert default demo accounts
        salt = secrets.token_hex(8)
        demo_hash = _hash_pw("demo123", salt)
        with conn:
            conn.execute("""
                INSERT OR IGNORE INTO users 
                (id, username, email, name, password_hash, salt, subject_id, age, gender, created_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "u_100", "demo", "demo@hearttwin.ai", "Demo Patient",
                demo_hash, salt, "100", 38, "Other",
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "Default demo patient profile"
            ))
            conn.execute("""
                INSERT OR IGNORE INTO users 
                (id, username, email, name, password_hash, salt, subject_id, age, gender, created_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "u_0004", "patient_0004", "patient0004@hearttwin.ai", "Alex Taylor",
                demo_hash, salt, "0004", 45, "Male",
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "Verified telemetry baseline Subject 0004"
            ))
        return

    try:
        data = json.loads(legacy_file.read_text(encoding="utf-8"))
        users = data.get("users", {})
        with conn:
            for k, u in users.items():
                user_id = u.get("id") or f"u_{uuid.uuid4().hex[:8]}"
                username = u.get("username", k)
                email = u.get("email") or f"{username}@local"
                name = u.get("name", username)
                pw_hash = u.get("password_hash")
                salt = u.get("salt") or secrets.token_hex(8)
                if not pw_hash:
                    pw_hash = _hash_pw("demo123", salt)
                subj = str(u.get("subject_id", "100"))
                # If alwinvar2003 or duplicate 100, appoint 1001
                cur.execute("SELECT id FROM users WHERE subject_id = ?", (subj,))
                if cur.fetchone():
                    subj = allocate_next_subject_id(conn)
                age = u.get("age")
                gender = u.get("gender", "")
                created = u.get("created_at") or datetime.datetime.now(datetime.timezone.utc).isoformat()
                notes = u.get("notes", "")

                conn.execute("""
                    INSERT OR IGNORE INTO users 
                    (id, username, email, name, password_hash, salt, subject_id, age, gender, created_at, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_id, username, email, name, pw_hash, salt, subj, age, gender, created, notes))
    except Exception as e:
        print(f"[db] Warning: legacy user migration error: {e}")

def allocate_next_subject_id(conn: sqlite3.Connection | None = None) -> str:
    """
    Allocates the next unique Subject ID without repetition or collision.
    - Scans both the SQLite database and the filesystem (MSCardio/Subject_*)
    - Always assigns non-colliding IDs starting at 1001 for newly registered patients.
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cur = conn.cursor()
        cur.execute("SELECT subject_id FROM users")
        taken_ids = {str(row[0]).strip().lstrip("0") for row in cur.fetchall()}

        # Also inspect existing folders in MSCardio
        mscardio_dir = PROJECT_ROOT / "MSCardio"
        if mscardio_dir.exists():
            for p in mscardio_dir.glob("Subject_*"):
                folder_id = p.name.replace("Subject_", "").strip()
                taken_ids.add(folder_id)
                taken_ids.add(folder_id.lstrip("0"))

        cand = 1001
        while str(cand) in taken_ids or f"{cand:04d}" in taken_ids:
            cand += 1

        return str(cand)
    finally:
        if should_close:
            conn.close()

def auth_register(data: dict) -> dict:
    """
    Registers a new user and automatically appoints a guaranteed unique Subject ID.
    The user NEVER has to provide a Subject ID.
    """
    raw_user = str(data.get("username", "")).strip().lower()
    raw_email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    name = str(data.get("name", "")).strip()
    age = data.get("age")
    gender = str(data.get("gender", "")).strip()

    if "@" in raw_user and not raw_email:
        raw_email = raw_user
        raw_user = raw_user.split("@")[0]

    if not raw_user and not raw_email:
        return {"ok": False, "error": "Username or email is required."}
    if not password or len(password) < 4:
        return {"ok": False, "error": "Password must be at least 4 characters long."}

    username = raw_user or (raw_email.split("@")[0] if raw_email else "patient")
    email = raw_email or f"{username}@hearttwin.local"
    name = name or username.capitalize()

    conn = get_connection()
    try:
        with conn:
            cur = conn.cursor()
            cur.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
            if cur.fetchone():
                return {"ok": False, "error": "An account with this username or email already exists."}

            # Appoint a clean, unique subject ID
            subject_id = allocate_next_subject_id(conn)
            salt = secrets.token_hex(8)
            pw_hash = _hash_pw(password, salt)
            user_id = "u_" + uuid.uuid4().hex[:8]
            created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

            conn.execute("""
                INSERT INTO users (id, username, email, name, password_hash, salt, subject_id, age, gender, created_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, username, email, name, pw_hash, salt, subject_id, age, gender, created_at, ""))

            # Generate 30-day session token ("Remember Me")
            token = secrets.token_urlsafe(32)
            now_ts = time.time()
            exp_ts = now_ts + (86400 * 30)

            conn.execute("""
                INSERT INTO sessions (token, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
            """, (token, user_id, now_ts, exp_ts))

        safe_user = {
            "id": user_id,
            "username": username,
            "email": email,
            "name": name,
            "subject_id": subject_id,
            "age": age,
            "gender": gender,
            "created_at": created_at
        }
        return {"ok": True, "token": token, "user": safe_user}
    except Exception as e:
        return {"ok": False, "error": f"Registration failed: {str(e)}"}
    finally:
        conn.close()

def auth_login(data: dict) -> dict:
    """
    Authenticates a user via Username or Email + Password.
    Returns their profile including their permanently assigned Subject ID.
    """
    ident = str(data.get("identifier", data.get("username", data.get("email", "")))).strip().lower()
    password = str(data.get("password", ""))

    if not ident or not password:
        return {"ok": False, "error": "Username/email and password are required."}

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, username, email, name, password_hash, salt, subject_id, age, gender, created_at, notes
            FROM users WHERE username = ? OR email = ?
        """, (ident, ident))
        row = cur.fetchone()

        if not row:
            return {"ok": False, "error": "Invalid username/email or password."}

        user = dict(row)
        if _hash_pw(password, user["salt"]) != user["password_hash"]:
            return {"ok": False, "error": "Invalid username/email or password."}

        # Create session token
        token = secrets.token_urlsafe(32)
        now_ts = time.time()
        exp_ts = now_ts + (86400 * 30)

        with conn:
            conn.execute("""
                INSERT INTO sessions (token, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
            """, (token, user["id"], now_ts, exp_ts))

        safe_user = {k: v for k, v in user.items() if k not in ("password_hash", "salt")}
        return {"ok": True, "token": token, "user": safe_user}
    finally:
        conn.close()

def get_user_by_token(token: str) -> dict | None:
    """Validates session token and returns the current user profile."""
    if not token:
        return None
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT s.expires_at, u.id, u.username, u.email, u.name, u.subject_id, u.age, u.gender, u.created_at, u.notes
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ?
        """, (token,))
        row = cur.fetchone()
        if not row:
            return None

        if float(row["expires_at"]) < time.time():
            with conn:
                conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            return None

        return {
            "id": row["id"],
            "username": row["username"],
            "email": row["email"],
            "name": row["name"],
            "subject_id": row["subject_id"],
            "age": row["age"],
            "gender": row["gender"],
            "created_at": row["created_at"],
            "notes": row["notes"]
        }
    finally:
        conn.close()

def auth_logout(token: str) -> dict:
    if token:
        conn = get_connection()
        try:
            with conn:
                conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        finally:
            conn.close()
    return {"ok": True}

def auth_update_profile(token: str, data: dict) -> dict:
    """Updates profile attributes (name, age, gender, notes) while keeping assigned Subject ID immutable."""
    user = get_user_by_token(token)
    if not user:
        return {"ok": False, "error": "Unauthorized or session expired"}

    name = str(data.get("name", user["name"])).strip() or user["name"]
    age = data.get("age", user["age"])
    gender = str(data.get("gender", user["gender"])).strip()
    notes = str(data.get("notes", user["notes"])).strip()

    conn = get_connection()
    try:
        with conn:
            conn.execute("""
                UPDATE users SET name = ?, age = ?, gender = ?, notes = ?
                WHERE id = ?
            """, (name, age, gender, notes, user["id"]))

        user["name"] = name
        user["age"] = age
        user["gender"] = gender
        user["notes"] = notes
        return {"ok": True, "user": user}
    finally:
        conn.close()

def list_patient_profiles() -> dict:
    """List profiles for clinical and presentation selector."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, username, name, subject_id, gender, age FROM users ORDER BY subject_id ASC")
        rows = cur.fetchall()
        return {"ok": True, "profiles": [dict(r) for r in rows]}
    finally:
        conn.close()

def record_upload(subject_id: str, recording_id: str, files: list[str],
                  storage_path: str = "", metadata: dict | None = None, user_id: str | None = None):
    """Logs an uploaded recording into the SQLite recordings database."""
    conn = get_connection()
    try:
        with conn:
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            files_json = json.dumps(files)
            meta_json = json.dumps(metadata or {})
            conn.execute("""
                INSERT INTO recordings (user_id, subject_id, recording_id, files, created_at, storage_path, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(subject_id, recording_id) DO UPDATE SET
                    files = excluded.files,
                    storage_path = excluded.storage_path,
                    metadata = excluded.metadata
            """, (user_id, str(subject_id), str(recording_id), files_json, now_iso, storage_path, meta_json))
    finally:
        conn.close()

def record_twin_score(subject_id: str, recording_id: str, score: float, placement: str, alert: str):
    """Updates a recording entry with the latest ML scoring results."""
    conn = get_connection()
    try:
        with conn:
            conn.execute("""
                UPDATE recordings SET anomaly_score = ?, placement = ?, alert = ?
                WHERE subject_id = ? AND recording_id = ?
            """, (score, placement, alert, str(subject_id), str(recording_id)))
    finally:
        conn.close()

def record_heartprint(subject_id: str, recording_id: str, bpm: float | None = None, quality: float | None = None):
    """Updates heart rate and signal quality metrics for a recording."""
    conn = get_connection()
    try:
        with conn:
            conn.execute("""
                UPDATE recordings SET detected_bpm = COALESCE(?, detected_bpm), quality_score = COALESCE(?, quality_score)
                WHERE subject_id = ? AND recording_id = ?
            """, (bpm, quality, str(subject_id), str(recording_id)))
    finally:
        conn.close()

def record_twin_training(subject_id: str, user_id: str | None = None,
                         checkpoint_path: str = "", status: str = "done", count: int = 0):
    """Records completion of a PersonalHeartTwin training run."""
    conn = get_connection()
    try:
        with conn:
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            conn.execute("""
                INSERT INTO twins (subject_id, user_id, checkpoint_path, status, recordings_count, trained_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(subject_id) DO UPDATE SET
                    status = excluded.status,
                    recordings_count = excluded.recordings_count,
                    checkpoint_path = excluded.checkpoint_path,
                    trained_at = excluded.trained_at
            """, (str(subject_id), user_id, checkpoint_path, status, count, now_iso))
    finally:
        conn.close()

# Auto-initialize database on import
init_db()
