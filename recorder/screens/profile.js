/**
 * screens/profile.js - Profile & Settings ("Me" tab)
 * Warm orange glass aesthetic with Patient Medical ID, Twin model training trigger,
 * CSV export, and researcher mode toggle.
 */

import { getTwinStatus, trainTwin, logoutUser } from '../api/client.js';
import { clearAllUserData, getSetting, setSetting, getRecordings } from '../storage/db.js';
import { showToast } from '../ui/toast.js';
import { navigateTo } from '../ui/nav.js';
import { icons } from '../ui/icons.js';

export async function renderProfile(container, { user, onLogout } = {}) {
  const subjectId = user ? user.subject_id : '1001';
  const name = user ? (user.name || user.username) : 'Amelia Vance';
  const username = user ? user.username : 'amelia';

  let modelStatusText = 'Personal Twin Not Trained';
  let isTrained = false;
  try {
    const twinRes = await getTwinStatus(subjectId);
    if (twinRes && twinRes.trained) {
      modelStatusText = 'Personal Twin Model Ready';
      isTrained = true;
    } else if (twinRes && twinRes.job === 'training') {
      modelStatusText = 'Training in progress...';
    }
  } catch (_) {
    modelStatusText = 'Local baseline active';
  }

  const isResearcherMode = await getSetting('researcher_mode', false);

  container.innerHTML = `
    <div style="padding: 24px 20px 104px 20px; display:flex; flex-direction:column; gap:18px;">
      
      <!-- Top Title -->
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <h2 class="screen-title">Profile &amp; Settings</h2>
        <div style="font-size:12px; color:var(--accent-orange); font-weight:700; background:rgba(255,90,31,0.1); padding:4px 10px; border-radius:var(--r-pill);">
          Subject #${subjectId}
        </div>
      </div>

      <!-- 1. Patient Profile Card -->
      <div style="background:#FFFFFF; border-radius:28px; padding:20px; box-shadow:var(--shadow-card); border:1px solid var(--border-subtle); display:flex; align-items:center; gap:16px;">
        <div style="width:56px; height:56px; border-radius:50%; background:var(--accent-orange); color:#FFFFFF; display:flex; align-items:center; justify-content:center; font-size:22px; font-weight:700; box-shadow:0 6px 16px rgba(255,90,31,0.3);">
          ${(name || 'A')[0].toUpperCase()}
        </div>
        <div style="flex:1;">
          <div style="font-size:18px; font-weight:700; color:var(--ink);">${name}</div>
          <div style="font-size:13px; color:var(--muted); margin-top:2px;">@${username}</div>
        </div>
      </div>

      <!-- 2. Personal Heart Twin Model Card -->
      <div style="background:#FFFFFF; border-radius:28px; padding:20px; box-shadow:var(--shadow-card); border:1px solid var(--border-subtle); display:flex; flex-direction:column; gap:12px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div style="font-size:16px; font-weight:700; color:var(--ink);">Cardiac Digital Twin</div>
          <span style="font-size:12px; font-weight:600; padding:4px 10px; border-radius:var(--r-pill); background:${isTrained ? 'rgba(16,185,129,0.12)' : 'rgba(245,158,11,0.12)'}; color:${isTrained ? '#059669' : '#D97706'};">
            ${isTrained ? 'Trained' : 'Untrained'}
          </span>
        </div>
        <div style="font-size:13px; color:var(--muted); line-height:1.4;">
          ${modelStatusText}. Collect at least 5 resting recordings to train your custom autoencoder neural network.
        </div>
        <button class="btn-orange-pill" id="profile-train-btn" style="padding:12px 20px; font-size:14px; margin-top:4px;">
          ${icons.activity(16)}
          <span>Train My Personal Model</span>
        </button>
      </div>

      <!-- 3. Clinical & Doctor Visit -->
      <div style="background:#FFFFFF; border-radius:28px; padding:20px; box-shadow:var(--shadow-card); border:1px solid var(--border-subtle); display:flex; flex-direction:column; gap:14px;">
        <div style="font-size:16px; font-weight:700; color:var(--ink);">Doctor Visit &amp; Data</div>
        <button id="profile-goto-share-btn" style="display:flex; align-items:center; justify-content:space-between; background:var(--bg-sunken); border:none; padding:14px 18px; border-radius:18px; cursor:pointer; text-align:left;">
          <div style="display:flex; align-items:center; gap:12px; color:var(--ink); font-weight:600; font-size:14px;">
            ${icons.trends(18)}
            <span>Generate 30-Day Doctor Summary</span>
          </div>
          <span style="color:var(--muted); font-size:18px;">›</span>
        </button>

        <button id="profile-export-csv-btn" style="display:flex; align-items:center; justify-content:space-between; background:var(--bg-sunken); border:none; padding:14px 18px; border-radius:18px; cursor:pointer; text-align:left;">
          <div style="display:flex; align-items:center; gap:12px; color:var(--ink); font-weight:600; font-size:14px;">
            ${icons.play(16)}
            <span>Export Recordings (CSV)</span>
          </div>
          <span style="color:var(--muted); font-size:18px;">›</span>
        </button>
      </div>

      <!-- 4. Researcher Mode Toggle -->
      <div style="background:#FFFFFF; border-radius:28px; padding:18px 20px; box-shadow:var(--shadow-card); border:1px solid var(--border-subtle); display:flex; align-items:center; justify-content:space-between;">
        <div>
          <div style="font-size:15px; font-weight:700; color:var(--ink);">Researcher Mode</div>
          <div style="font-size:12px; color:var(--muted); margin-top:2px;">Display raw sensor Hz, KL anomaly &amp; placement confidence</div>
        </div>
        <input type="checkbox" id="researcher-toggle" ${isResearcherMode ? 'checked' : ''} style="width:20px; height:20px; accent-color:var(--accent-orange); cursor:pointer;" />
      </div>

      <!-- 5. Danger Zone & Logout -->
      <div style="display:flex; flex-direction:column; gap:10px; margin-top:8px;">
        <button id="profile-logout-btn" style="padding:14px; border-radius:var(--r-btn); background:transparent; border:1px solid var(--border-strong); color:var(--ink); font-size:14px; font-weight:600; cursor:pointer;">
          Sign Out
        </button>
      </div>

      <div style="text-align:center; font-size:12px; color:var(--muted); margin-top:8px;">
        Digital Heart Twin v2.0 • Research Prototype<br/>
        Not a diagnostic medical device
      </div>

    </div>
  `;

  // Handlers
  document.getElementById('profile-goto-share-btn')?.addEventListener('click', () => {
    navigateTo('share');
  });

  document.getElementById('profile-train-btn')?.addEventListener('click', async () => {
    const btn = document.getElementById('profile-train-btn');
    btn.innerHTML = `<span class="heart-pulse">⏳</span> <span>Training in Background...</span>`;
    btn.disabled = true;
    try {
      const res = await trainTwin(subjectId);
      if (res && res.ok) {
        showToast('Personal twin training started on server!', 'normal');
        let attempts = 0;
        const pollTimer = setInterval(async () => {
          attempts++;
          try {
            const status = await getTwinStatus(subjectId);
            if (status && status.trained) {
              clearInterval(pollTimer);
              showToast('Personal Heart Twin Model is now Trained & Ready!', 'normal');
              renderProfile(container, { user, onLogout });
            } else if (attempts >= 10) {
              clearInterval(pollTimer);
              renderProfile(container, { user, onLogout });
            }
          } catch (_) {
            if (attempts >= 10) clearInterval(pollTimer);
          }
        }, 2000);
      } else {
        showToast(res?.error || 'Training requested on server.', 'info');
        renderProfile(container, { user, onLogout });
      }
    } catch (_) {
      showToast('Model training triggered on server.', 'info');
      renderProfile(container, { user, onLogout });
    }
  });

  document.getElementById('profile-export-csv-btn')?.addEventListener('click', async () => {
    const recs = await getRecordings(subjectId, 100);
    if (!recs || recs.length === 0) {
      showToast('No recordings found to export.', 'info');
      return;
    }
    let csv = 'timestamp,date,context,bpm,sdnn,rmssd,quality,verdict\n';
    recs.forEach((r) => {
      csv += `${r.timestamp},"${new Date(r.timestamp).toISOString()}","${r.context || ''}",${r.hr || ''},${r.sdnn || ''},${r.rmssd || ''},${r.qualityScore || ''},"${r.verdict || ''}"\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `heart_twin_recordings_${subjectId}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Exported CSV successfully.', 'normal');
  });

  document.getElementById('researcher-toggle')?.addEventListener('change', async (e) => {
    await setSetting('researcher_mode', e.target.checked);
    showToast(`Researcher mode ${e.target.checked ? 'enabled' : 'disabled'}.`, 'normal');
  });

  document.getElementById('profile-logout-btn')?.addEventListener('click', async () => {
    await logoutUser();
    await setSetting('current_user', null);
    if (typeof onLogout === 'function') onLogout();
    navigateTo('auth');
  });
}

export const renderMe = renderProfile;

