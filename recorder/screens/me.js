/**
 * screens/me.js - Profile, Model Training, Privacy, Help & Researcher Mode
 */

import { getTwinStatus, trainTwin, logoutUser } from '../api/client.js';
import { clearAllUserData, getSetting, setSetting, getRecordings } from '../storage/db.js';
import { openSheet, closeSheet } from '../ui/sheet.js';
import { showToast } from '../ui/toast.js';
import { navigateTo } from '../ui/nav.js';

export async function renderMe(container, { user, onLogout } = {}) {
  const subjectId = user ? user.subject_id : '1001';
  const name = user ? (user.name || user.username) : 'Patient';
  const email = user ? (user.email || `${user.username}@local`) : 'patient@hearttwin.ai';

  // Check personal twin model status
  let modelStatusText = 'Checking…';
  let modelBadgeClass = 'badge-neutral';
  try {
    const twinRes = await getTwinStatus(subjectId);
    if (twinRes.trained) {
      modelStatusText = 'Ready & Established';
      modelBadgeClass = 'badge-normal';
    } else if (twinRes.job === 'training') {
      modelStatusText = 'Training in progress…';
      modelBadgeClass = 'badge-slight';
    } else {
      modelStatusText = 'Not trained yet';
      modelBadgeClass = 'badge-neutral';
    }
  } catch (_) {
    modelStatusText = 'Local baseline active';
  }

  const isResearcherMode = await getSetting('researcher_mode', false);

  container.innerHTML = `
    <div style="display:flex; flex-direction:column; gap: var(--sp-4);">
      <h2>Profile &amp; Settings</h2>

      <!-- 1. Patient Profile Card -->
      <div class="card">
        <div style="display:flex; align-items:center; gap:12px;">
          <div style="width:48px; height:48px; border-radius:var(--r-pill); background:var(--primary); color:#FFFFFF; display:flex; align-items:center; justify-content:center; font-size:20px; font-weight:700;">
            ${(name || 'P')[0].toUpperCase()}
          </div>
          <div style="flex:1;">
            <div style="font-size:16px; font-weight:700; color:var(--text-main);">${name}</div>
            <div style="font-size:12px; color:var(--text-muted);">${email}</div>
          </div>
          <span class="badge badge-neutral" style="font-family:var(--font-mono); font-size:12px;">ID #${subjectId}</span>
        </div>
        <button class="btn btn-secondary btn-block" id="me-logout-btn" style="margin-top:6px; min-height:40px; font-size:13px;">
          Sign Out of Account
        </button>
      </div>

      <!-- 2. Personal Digital Twin Model Card -->
      <div class="card">
        <div class="card-header">
          <span class="card-title" style="font-size:14px;">Personal Digital Twin Model</span>
          <span class="badge ${modelBadgeClass}" id="me-model-badge">${modelStatusText}</span>
        </div>
        <p style="font-size:12px; color:var(--text-muted);">
          Your personal VAE neural model learns your specific sternal motion geometry to detect anomalies relative to yourself.
        </p>
        <button class="btn btn-primary btn-block" id="me-train-model-btn" style="margin-top:4px;">
          ⚡ Train / Retrain My Model
        </button>
      </div>

      <!-- 3. Doctor Visit Summary Action -->
      <div class="card" style="border-left:4px solid var(--primary);">
        <div class="card-header">
          <span class="card-title" style="font-size:14px;">Doctor Visit Summary</span>
          <span class="badge badge-normal">PDF Ready</span>
        </div>
        <p style="font-size:12px; color:var(--text-muted);">
          Generate a 1-page clinical printout of your resting heart rate and rhythm consistency over the last 30 days.
        </p>
        <button class="btn btn-secondary btn-block" id="me-open-share-btn">
          Open 30-Day Report →
        </button>
      </div>

      <!-- 4. Data & Privacy -->
      <div class="card">
        <span class="card-title" style="font-size:14px;">Data &amp; Privacy</span>
        <div style="display:flex; flex-direction:column; gap:8px; margin-top:4px;">
          <button class="btn btn-secondary btn-block" id="me-export-data-btn">
            💾 Export All Recordings (CSV)
          </button>
          <button class="btn btn-ghost btn-block" id="me-delete-data-btn" style="color:var(--status-diff-text);">
            🗑️ Delete All My Stored Data
          </button>
        </div>
      </div>

      <!-- 5. Researcher Mode Switch -->
      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div>
            <span class="card-title" style="font-size:14px;">Researcher Mode</span>
            <p style="font-size:12px; color:var(--text-muted);">Reveal raw KL scores, frequency spectra, and ML thresholds.</p>
          </div>
          <label class="switch" style="position:relative; display:inline-block; width:44px; height:24px;">
            <input type="checkbox" id="researcher-mode-toggle" ${isResearcherMode ? 'checked' : ''} style="opacity:0; width:0; height:0;">
            <span class="slider" style="position:absolute; cursor:pointer; inset:0; background-color:${isResearcherMode ? 'var(--primary)' : 'var(--border-strong)'}; border-radius:24px; transition:0.2s;"></span>
          </label>
        </div>
      </div>

      <!-- 6. About & Disclaimer -->
      <div style="text-align:center; padding: 12px var(--sp-4);">
        <p style="font-size:12px; color:var(--text-muted);">
          Digital Heart Twin · Version 2.0 (Mobile PWA)<br/>
          Investigational Prototype · Not a certified diagnostic medical device.
        </p>
      </div>
    </div>
  `;

  // Attach Handlers
  const logoutBtn = container.querySelector('#me-logout-btn');
  const trainBtn = container.querySelector('#me-train-model-btn');
  const shareBtn = container.querySelector('#me-open-share-btn');
  const exportBtn = container.querySelector('#me-export-data-btn');
  const deleteBtn = container.querySelector('#me-delete-data-btn');
  const researcherToggle = container.querySelector('#researcher-mode-toggle');

  if (logoutBtn) {
    logoutBtn.onclick = async () => {
      await logoutUser();
      if (typeof onLogout === 'function') onLogout();
      showToast('Signed out successfully', 'info');
      navigateTo('onboarding');
    };
  }

  if (trainBtn) {
    trainBtn.onclick = async () => {
      trainBtn.disabled = true;
      trainBtn.textContent = 'Starting training…';
      try {
        const res = await trainTwin(subjectId);
        if (res.ok) {
          showToast('Model training started in background! Check back in 1-2 minutes.', 'normal');
          container.querySelector('#me-model-badge').textContent = 'Training in progress…';
          container.querySelector('#me-model-badge').className = 'badge badge-slight';
        } else {
          showToast(res.error || 'Need at least 5-6 recordings to train', 'slight');
        }
      } catch (err) {
        showToast('Error communicating with ML server', 'diff');
      } finally {
        trainBtn.disabled = false;
        trainBtn.textContent = '⚡ Train / Retrain My Model';
      }
    };
  }

  if (shareBtn) shareBtn.onclick = () => navigateTo('share');

  if (exportBtn) {
    exportBtn.onclick = async () => {
      const recs = await getRecordings(subjectId, 500);
      if (recs.length === 0) {
        showToast('No recordings to export', 'info');
        return;
      }
      let csv = 'timestamp,context,bpm,sdnn,rmssd,pnn50,quality,verdict,ratio,anomaly_score\n';
      recs.forEach((r) => {
        csv += `${r.timestamp},${r.context || 'Resting'},${r.bpm || ''},${r.sdnn || ''},${r.rmssd || ''},${r.pnn50 || ''},${r.quality || ''},"${r.verdict || ''}",${r.ratio || ''},${r.score || ''}\n`;
      });
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `heart_twin_recordings_${subjectId}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      showToast('Exported CSV file', 'normal');
    };
  }

  if (deleteBtn) {
    deleteBtn.onclick = () => {
      openSheet(`
        <div style="text-align:center; padding:12px 0;">
          <span style="font-size:32px;">⚠️</span>
          <h3 style="margin-top:8px;">Delete All Stored Data?</h3>
          <p style="font-size:13px; color:var(--text-muted); margin: 8px 0 16px;">
            This will permanently remove all your local cardiac recordings and baseline metrics from this device.
          </p>
          <div style="display:flex; flex-direction:column; gap:8px;">
            <button class="btn btn-block" id="confirm-delete-btn" style="background:var(--status-diff); color:#FFF;">
              Yes, Delete Everything
            </button>
            <button class="btn btn-secondary btn-block" id="cancel-delete-btn">
              Cancel
            </button>
          </div>
        </div>
      `, 'Confirm Deletion');

      setTimeout(() => {
        const confirmBtn = document.getElementById('confirm-delete-btn');
        const cancelBtn = document.getElementById('cancel-delete-btn');
        if (cancelBtn) cancelBtn.onclick = closeSheet;
        if (confirmBtn) {
          confirmBtn.onclick = async () => {
            await clearAllUserData(subjectId);
            closeSheet();
            showToast('All local data cleared', 'info');
            renderMe(container, { user, onLogout });
          };
        }
      }, 50);
    };
  }

  if (researcherToggle) {
    researcherToggle.onchange = async () => {
      await setSetting('researcher_mode', researcherToggle.checked);
      const slider = container.querySelector('.slider');
      if (slider) slider.style.backgroundColor = researcherToggle.checked ? 'var(--primary)' : 'var(--border-strong)';
      showToast(researcherToggle.checked ? 'Researcher Mode Enabled' : 'Researcher Mode Disabled', 'info');
    };
  }
}

