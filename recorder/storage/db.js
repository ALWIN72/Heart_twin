/**
 * storage/db.js - Modern IndexedDB Storage Layer for Digital Heart Twin
 * User-scoped, offline-first, replaces volatile localStorage.
 */

const DB_NAME = 'DigitalHeartTwinDB';
const DB_VERSION = 1;

let _dbInstance = null;

export async function openDB() {
  if (_dbInstance) return _dbInstance;

  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);

    req.onupgradeneeded = (e) => {
      const db = e.target.result;

      // 1. Recordings store
      if (!db.objectStoreNames.contains('recordings')) {
        const recStore = db.createObjectStore('recordings', { keyPath: 'id' });
        recStore.createIndex('subject_id', 'subject_id', { unique: false });
        recStore.createIndex('timestamp', 'timestamp', { unique: false });
        recStore.createIndex('context', 'context', { unique: false });
      }

      // 2. Baselines store (per user & context)
      if (!db.objectStoreNames.contains('baselines')) {
        db.createObjectStore('baselines', { keyPath: 'key' });
      }

      // 3. User & Settings store
      if (!db.objectStoreNames.contains('settings')) {
        db.createObjectStore('settings', { keyPath: 'key' });
      }
    };

    req.onsuccess = (e) => {
      _dbInstance = e.target.result;
      resolve(_dbInstance);
    };

    req.onerror = (e) => {
      console.error('[IndexedDB] Failed to open database:', e.target.error);
      reject(e.target.error);
    };
  });
}

/** Save a new cardiac telemetry recording */
export async function saveRecording(rec) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('recordings', 'readwrite');
    const store = tx.objectStore('recordings');
    const item = {
      ...rec,
      id: rec.id || `rec_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
      timestamp: rec.timestamp || new Date().toISOString(),
    };
    const req = store.put(item);
    req.onsuccess = () => resolve(item);
    req.onerror = (e) => reject(e.target.error);
  });
}

/** Get recordings for a specific subject, sorted newest first */
export async function getRecordings(subjectId, limit = 50) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('recordings', 'readonly');
    const store = tx.objectStore('recordings');
    const index = store.index('timestamp');
    const results = [];

    const req = index.openCursor(null, 'prev');
    req.onsuccess = (e) => {
      const cursor = e.target.result;
      if (cursor && results.length < limit) {
        if (!subjectId || cursor.value.subject_id === String(subjectId)) {
          results.push(cursor.value);
        }
        cursor.continue();
      } else {
        resolve(results);
      }
    };
    req.onerror = (e) => reject(e.target.error);
  });
}

/** Get a single recording by ID */
export async function getRecordingById(id) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('recordings', 'readonly');
    const req = tx.objectStore('recordings').get(id);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = (e) => reject(e.target.error);
  });
}

/** Delete a recording by ID */
export async function deleteRecording(id) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('recordings', 'readwrite');
    const req = tx.objectStore('recordings').delete(id);
    req.onsuccess = () => resolve(true);
    req.onerror = (e) => reject(e.target.error);
  });
}

/** Save or update personal baseline metrics */
export async function saveBaseline(subjectId, context = 'Resting', baselineData) {
  const db = await openDB();
  const key = `baseline_${subjectId}_${context}`;
  return new Promise((resolve, reject) => {
    const tx = db.transaction('baselines', 'readwrite');
    const req = tx.objectStore('baselines').put({
      key,
      subjectId: String(subjectId),
      context,
      data: baselineData,
      updatedAt: new Date().toISOString(),
    });
    req.onsuccess = () => resolve(true);
    req.onerror = (e) => reject(e.target.error);
  });
}

/** Retrieve personal baseline metrics */
export async function getBaseline(subjectId, context = 'Resting') {
  const db = await openDB();
  const key = `baseline_${subjectId}_${context}`;
  return new Promise((resolve, reject) => {
    const tx = db.transaction('baselines', 'readonly');
    const req = tx.objectStore('baselines').get(key);
    req.onsuccess = () => resolve(req.result ? req.result.data : null);
    req.onerror = (e) => reject(e.target.error);
  });
}

/** Key-Value Settings & User Profile */
export async function setSetting(key, value) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('settings', 'readwrite');
    const req = tx.objectStore('settings').put({ key, value });
    req.onsuccess = () => resolve(true);
    req.onerror = (e) => reject(e.target.error);
  });
}

export async function getSetting(key, defaultValue = null) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('settings', 'readonly');
    const req = tx.objectStore('settings').get(key);
    req.onsuccess = () => resolve(req.result ? req.result.value : defaultValue);
    req.onerror = () => resolve(defaultValue);
  });
}

/** Clear all data for current user (Data & Privacy) */
export async function clearAllUserData(subjectId) {
  const db = await openDB();
  const recs = await getRecordings(subjectId, 1000);
  const tx = db.transaction(['recordings', 'baselines'], 'readwrite');
  const recStore = tx.objectStore('recordings');
  for (const r of recs) {
    recStore.delete(r.id);
  }
  const baseStore = tx.objectStore('baselines');
  baseStore.delete(`baseline_${subjectId}_Resting`);
  baseStore.delete(`baseline_${subjectId}_After activity`);
  baseStore.delete(`baseline_${subjectId}_Other`);
  return new Promise((resolve) => {
    tx.oncomplete = () => resolve(true);
  });
}

