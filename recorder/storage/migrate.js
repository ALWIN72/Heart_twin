/**
 * storage/migrate.js - One-time migration from legacy localStorage to IndexedDB
 */
import { saveRecording, saveBaseline, setSetting, getSetting } from './db.js';

export async function runMigrationIfNeeded(currentSubjectId) {
  const migrated = await getSetting('dht_migrated_v2', false);
  if (migrated) return;

  try {
    // 1. Check legacy recordings in localStorage
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith('digital_heart_twin_hist_')) {
        const sid = key.replace('digital_heart_twin_hist_', '');
        // Do not migrate demo dummy data
        if (sid === '100' || sid === '0004') continue;

        try {
          const items = JSON.parse(localStorage.getItem(key) || '[]');
          for (const item of items) {
            await saveRecording({
              subject_id: sid,
              timestamp: item.ts || new Date().toISOString(),
              context: item.context || 'Resting',
              analysis: {
                bpm: item.bpm,
                sdnn: item.sdnn,
                resp: item.resp,
                quality: item.quality,
                dev: item.dev,
              },
              quality: item.quality || 'Good',
              status: item.dev ? 'Slightly different' : 'Normal for you',
            });
          }
        } catch (_) {}
      }

      if (key && key.startsWith('digital_heart_twin_base_')) {
        const parts = key.replace('digital_heart_twin_base_', '').split('_');
        const sid = parts[0];
        const ctx = parts.slice(1).join('_') || 'Resting';
        if (sid === '100') continue;
        try {
          const bData = JSON.parse(localStorage.getItem(key) || 'null');
          if (bData) {
            await saveBaseline(sid, ctx, bData);
          }
        } catch (_) {}
      }
    }

    await setSetting('dht_migrated_v2', true);
  } catch (err) {
    console.warn('[Migration] Note: legacy storage migration skipped:', err);
  }
}

