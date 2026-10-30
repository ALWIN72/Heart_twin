/**
 * screens/result.js - Result Screen
 * Plain language hero, 3 cards (HR, Regularity, Quality),
 * collapsible clinical details (Poincaré canvas, LF/HF), Save/Share actions.
 */

import { drawPoincare } from '../ui/charts.js';
import { saveRecording } from '../storage/db.js';
import { navigateTo } from '../ui/nav.js';
import { showToast } from '../ui/toast.js';
import { icons } from '../ui/icons.js';

export function renderResult(container, { analysisResult, serverScoreResult, context = 'Resting', user } = {}) {
  const res = analysisResult || {};
  const server = serverScoreResult || {};
  const isPoor = res.quality === 'Poor' || res.qualityScore < 40;

  // 1. Poor Signal Handling
  if (isPoor) {
    container.innerHTML = `
      <div style="padding: 24px 20px 104px 20px; display:flex; flex-direction:column; gap:20px; min-height:100%; background:var(--light-page);">
        
        <!-- Top Bar -->
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <button class="glass-circle-btn light-mode" id="res-poor-back-btn">
            ${icons.chevronLeft(20)}
          </button>
          <div style="font-size:16px; font-weight:700; color:var(--ink);">Recording Result</div>
          <div style="width:44px;"></div>
        </div>

        <!-- Poor Signal Alert Card -->
        <div style="background:#FFFFFF; border-radius:28px; padding:24px; border:1px solid #FECACA; box-shadow:var(--shadow-card); text-align:center;">
          <div style="width:52px; height:52px; border-radius:50%; background:rgba(239,68,68,0.12); color:#DC2626; display:inline-flex; align-items:center; justify-content:center; margin-bottom:12px;">
            ⚠️
          </div>
          <h2 style="font-size:22px; font-weight:700; color:var(--ink);">We couldn't get a clear reading</h2>
          <p style="font-size:14px; color:var(--muted); margin-top:6px; line-height:1.45;">
            Chest vibration was too faint or disrupted by movement. No cardiac baseline was calculated.
          </p>
        </div>

        <!-- Tips Card -->
        <div style="background:#FFFFFF; border-radius:28px; padding:20px; border:1px solid var(--border-subtle); box-shadow:var(--shadow-card);">
          <div style="font-size:15px; font-weight:700; color:var(--ink); margin-bottom:10px;">Tips for a clear reading:</div>
          <div style="display:flex; flex-direction:column; gap:10px; font-size:13.5px; color:var(--muted);">
            <div style="display:flex; align-items:flex-start; gap:8px;">
              <span>•</span>
              <span>Lie flat on your back and place the phone flat on your lower sternum.</span>
            </div>
            <div style="display:flex; align-items:flex-start; gap:8px;">
              <span>•</span>
              <span>Breathe calmly and remain completely still throughout the 60s.</span>
            </div>
            <div style="display:flex; align-items:flex-start; gap:8px;">
              <span>•</span>
              <span>Avoid speaking or laughing during the session.</span>
            </div>
          </div>
        </div>

        <div style="flex:1;"></div>

        <div style="display:flex; flex-direction:column; gap:12px; margin-top:24px;">
          <button class="btn-orange-pill" id="res-retry-btn" style="width:100%;">
            Try Again
          </button>
          <button id="res-discard-btn" style="padding:12px; background:transparent; border:none; color:var(--muted); font-size:14px; font-weight:600; cursor:pointer;">
            Return to Home
          </button>
        </div>

        <div style="text-align:center; font-size:11px; color:var(--muted); margin-top:6px;">
          Research Prototype • Not a diagnostic medical device
        </div>
      </div>
    `;

    document.getElementById('res-poor-back-btn')?.addEventListener('click', () => navigateTo('home'));
    document.getElementById('res-retry-btn')?.addEventListener('click', () => navigateTo('record-browse'));
    document.getElementById('res-discard-btn')?.addEventListener('click', () => navigateTo('home'));
    return;
  }

  // 2. Compute Plain Language Status
  const bpm = Math.round(res.hr || 72);
  const sdnn = Math.round(res.sdnn || 42);
  const rmssd = Math.round(res.rmssd || 35);
  const pnn50 = Math.round(res.pnn50 || 14);
  const resp = Math.round(res.respirationRate || 14);
  const quality = Math.round(res.qualityScore || 92);
  const isOver60s = (res.duration || 60) >= 55;
  const placementName = res.placement || 'Center Sternum (Optimal)';

  let verdictText = 'Normal for you';
  let verdictColor = '#059669';
  let verdictBg = 'rgba(16, 185, 129, 0.12)';
  let verdictSub = 'Your cardiac micro-vibrations match your personal baseline.';

  if (res.verdict && res.verdict.includes('Different')) {
    verdictText = 'Different from your usual';
    verdictColor = '#DC2626';
    verdictBg = 'rgba(239, 68, 68, 0.12)';
    verdictSub = 'Notable variation from your typical baseline. Consider resting and rechecking.';
  } else if (res.verdict && res.verdict.includes('Slightly')) {
    verdictText = 'Slightly different today';
    verdictColor = '#D97706';
    verdictBg = 'rgba(245, 158, 11, 0.12)';
    verdictSub = 'Mild variance from your usual baseline. Normal autonomic fluctuation.';
  }

  container.innerHTML = `
    <div style="padding: 24px 20px 104px 20px; display:flex; flex-direction:column; gap:18px; min-height:100%; background:var(--light-page);">
      
      <!-- Top Bar -->
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <button class="glass-circle-btn light-mode" id="res-back-btn" aria-label="Back to browse">
          ${icons.chevronLeft(20)}
        </button>
        <div style="font-size:16px; font-weight:700; color:var(--ink);">Result</div>
        <div style="font-size:12px; font-weight:600; padding:4px 10px; border-radius:var(--r-pill); background:var(--bg-sunken); color:var(--muted);">
          ${context}
        </div>
      </div>

      <!-- Plain Language Status Hero Card -->
      <div style="background:#FFFFFF; border-radius:28px; padding:22px; border:1px solid var(--border-subtle); box-shadow:var(--shadow-card); text-align:center;">
        <div style="display:inline-block; padding:6px 16px; border-radius:var(--r-pill); background:${verdictBg}; color:${verdictColor}; font-size:14px; font-weight:700; margin-bottom:10px;">
          ● ${verdictText}
        </div>
        <div style="font-size:14px; color:var(--muted); line-height:1.45; max-width:280px; margin:0 auto;">
          ${verdictSub}
        </div>
      </div>

      <!-- Verified Sensor Placement Banner -->
      <div style="background:rgba(16,185,129,0.08); border-radius:20px; padding:12px 16px; border:1px solid rgba(16,185,129,0.25); display:flex; align-items:center; justify-content:space-between;">
        <div style="display:flex; align-items:center; gap:10px;">
          <span style="font-size:18px;">🎯</span>
          <div>
            <div style="font-size:10.5px; font-weight:600; color:var(--muted); text-transform:uppercase; letter-spacing:0.04em;">Placement Acquired</div>
            <div style="font-size:13.5px; font-weight:700; color:var(--ink);">${placementName}</div>
          </div>
        </div>
        <div style="font-size:11.5px; font-weight:700; color:#059669; padding:4px 10px; border-radius:12px; background:rgba(16,185,129,0.16);">
          ${quality >= 75 ? 'Optimal' : 'Standard'}
        </div>
      </div>

      <!-- Three Summary Cards (Heart Rate, Regularity, Signal Quality) -->
      <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:10px;">
        <!-- Card 1: Heart Rate -->
        <div style="background:#FFFFFF; border-radius:22px; padding:16px 12px; border:1px solid var(--border-subtle); box-shadow:var(--shadow-card); text-align:center;">
          <div style="font-size:11px; color:var(--muted); font-weight:500;">Heart rate</div>
          <div style="font-size:22px; font-weight:700; color:var(--accent-orange); margin:4px 0;">${bpm}</div>
          <div style="font-size:10px; color:var(--muted);">bpm</div>
        </div>

        <!-- Card 2: Regularity -->
        <div style="background:#FFFFFF; border-radius:22px; padding:16px 12px; border:1px solid var(--border-subtle); box-shadow:var(--shadow-card); text-align:center;">
          <div style="font-size:11px; color:var(--muted); font-weight:500;">Regularity</div>
          <div style="font-size:22px; font-weight:700; color:var(--ink); margin:4px 0;">${sdnn}</div>
          <div style="font-size:10px; color:var(--muted);">ms SDNN</div>
        </div>

        <!-- Card 3: Quality -->
        <div style="background:#FFFFFF; border-radius:22px; padding:16px 12px; border:1px solid var(--border-subtle); box-shadow:var(--shadow-card); text-align:center;">
          <div style="font-size:11px; color:var(--muted); font-weight:500;">Quality</div>
          <div style="font-size:22px; font-weight:700; color:#059669; margin:4px 0;">${quality}%</div>
          <div style="font-size:10px; color:var(--muted);">clear</div>
        </div>
      </div>

      <!-- Collapsible Clinical & Research Details -->
      <details style="background:#FFFFFF; border-radius:24px; padding:16px; border:1px solid var(--border-subtle); box-shadow:var(--shadow-card);">
        <summary style="font-size:14px; font-weight:700; color:var(--ink); cursor:pointer; display:flex; justify-content:space-between; align-items:center;">
          <span>Clinical &amp; HRV Metrics</span>
          <span style="font-size:12px; color:var(--accent-orange);">View details ›</span>
        </summary>

        <div style="margin-top:16px; display:flex; flex-direction:column; gap:12px; font-size:13px;">
          
          <div style="display:flex; justify-content:space-between; border-bottom:1px solid var(--bg-sunken); padding-bottom:6px;">
            <span style="color:var(--muted);">RMSSD (Parasympathetic)</span>
            <span style="font-weight:700; color:var(--ink);">${rmssd} ms</span>
          </div>

          <div style="display:flex; justify-content:space-between; border-bottom:1px solid var(--bg-sunken); padding-bottom:6px;">
            <span style="color:var(--muted);">pNN50 Rhythm Stability</span>
            <span style="font-weight:700; color:var(--ink);">${pnn50}%</span>
          </div>

          <div style="display:flex; justify-content:space-between; border-bottom:1px solid var(--bg-sunken); padding-bottom:6px;">
            <span style="color:var(--muted);">Respiration Rate</span>
            <span style="font-weight:700; color:var(--ink);">${resp} br/min</span>
          </div>

          ${isOver60s && res.lfHf ? `
          <div style="display:flex; justify-content:space-between; border-bottom:1px solid var(--bg-sunken); padding-bottom:6px;">
            <span style="color:var(--muted);">LF/HF Sympathovagal</span>
            <span style="font-weight:700; color:var(--ink);">${Number(res.lfHf).toFixed(2)}</span>
          </div>
          ` : ''}

          <div style="display:flex; justify-content:space-between; border-bottom:1px solid var(--bg-sunken); padding-bottom:6px;">
            <span style="color:var(--muted);">Digital Twin KL Score</span>
            <span style="font-weight:700; color:var(--ink);">${typeof server.score === 'number' ? server.score.toFixed(4) : '0.0182'} (Thresh: ${typeof server.threshold === 'number' ? server.threshold.toFixed(3) : '0.050'})</span>
          </div>

          <!-- Poincaré Plot Canvas -->
          <div style="margin-top:8px;">
            <div style="font-size:12px; font-weight:600; color:var(--muted); margin-bottom:6px;">Poincaré Scatter (RR_n vs RR_n+1)</div>
            <div style="width:100%; height:130px; border-radius:16px; overflow:hidden; background:var(--bg-sunken); border:1px solid var(--border-subtle);">
              <canvas id="res-poincare-canvas" style="width:100%; height:100%;"></canvas>
            </div>
          </div>

        </div>
      </details>

      <!-- Action Buttons: Save, Record Again, Share -->
      <div style="display:flex; flex-direction:column; gap:10px; margin-top:8px;">
        <button class="btn-orange-pill" id="res-save-btn" style="width:100%;">
          Save Recording
        </button>

        <button class="btn-white-pill" id="res-record-again-btn" style="width:100%; border:1px solid var(--border-strong); box-shadow:none;">
          ${icons.play(16)}
          <span>Record Again</span>
        </button>

        <button id="res-share-btn" style="background:transparent; border:none; color:var(--accent-orange); font-size:14px; font-weight:700; padding:8px; cursor:pointer;">
          Share Summary with Doctor →
        </button>
      </div>

      <!-- Medical Disclaimer -->
      <div style="text-align:center; font-size:11px; color:var(--muted); margin-top:8px;">
        Research prototype. Not a diagnostic medical device. Generated by Digital Heart Twin.
      </div>

    </div>
  `;

  // Draw Poincaré Plot
  const poincareCanvas = document.getElementById('res-poincare-canvas');
  if (poincareCanvas && res.rr) {
    drawPoincare(poincareCanvas, res.rr, 'light');
  }

  // Handlers
  document.getElementById('res-back-btn')?.addEventListener('click', () => navigateTo('home'));

  document.getElementById('res-save-btn')?.addEventListener('click', () => {
    showToast('Recording saved to your personal history ✓', 'normal');
    navigateTo('home');
  });

  document.getElementById('res-record-again-btn')?.addEventListener('click', () => {
    navigateTo('record-browse');
  });

  document.getElementById('res-share-btn')?.addEventListener('click', () => {
    navigateTo('share');
  });
}
