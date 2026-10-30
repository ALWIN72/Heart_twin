/**
 * screens/share.js - 30-Day Clinical Doctor Visit Summary Screen (Print/PDF)
 * Clean, printable 1-page summary of last 30 days.
 */

import { getRecordings } from '../storage/db.js';
import { drawTrendsHR } from '../ui/charts.js';
import { navigateTo } from '../ui/nav.js';
import { icons } from '../ui/icons.js';

export async function renderShare(container, { user } = {}) {
  const subjectId = user ? user.subject_id : '1001';
  const name = user ? (user.name || user.username) : 'Amelia Vance';

  const allRecordings = await getRecordings(subjectId, 100);
  const now = Date.now();
  const thirtyDays = allRecordings.filter((r) => {
    const t = new Date(r.timestamp).getTime();
    return (now - t) <= 30 * 86400 * 1000;
  });

  const restingOnly = thirtyDays.filter((r) => r.context === 'Resting');
  const bpms = restingOnly.map((r) => r.bpm || r.hr).filter(Boolean);
  
  // Median BPM
  let medianBpm = '-';
  if (bpms.length > 0) {
    const sorted = [...bpms].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    medianBpm = sorted.length % 2 !== 0 ? Math.round(sorted[mid]) : Math.round((sorted[mid - 1] + sorted[mid]) / 2);
  }

  // HRV Range (SDNN min - max)
  const sdnns = restingOnly.map((r) => r.sdnn).filter(Boolean);
  let hrvRange = '-';
  if (sdnns.length > 0) {
    const minSdnn = Math.round(Math.min(...sdnns));
    const maxSdnn = Math.round(Math.max(...sdnns));
    hrvRange = `${minSdnn} – ${maxSdnn} ms`;
  }

  // % Days Normal
  let normalCount = 0;
  thirtyDays.forEach((r) => {
    if (!r.verdict || (!r.verdict.includes('Different') && !r.verdict.includes('Slightly'))) {
      normalCount++;
    }
  });
  const normalPercent = thirtyDays.length > 0 ? Math.round((normalCount / thirtyDays.length) * 100) : 100;

  const todayStr = new Date().toLocaleDateString([], { month: 'long', day: 'numeric', year: 'numeric' });

  container.innerHTML = `
    <div style="padding: 24px 20px 40px 20px; display:flex; flex-direction:column; gap:18px; min-height:100%; background:var(--light-page);">
      
      <!-- Screen Header (Hidden on Print) -->
      <div class="no-print" style="display:flex; justify-content:space-between; align-items:center;">
        <button class="glass-circle-btn light-mode" id="share-back-btn" aria-label="Go back">
          ${icons.chevronLeft(20)}
        </button>
        <div style="font-size:16px; font-weight:700; color:var(--ink);">Doctor Visit Summary</div>
        <button class="btn-orange-pill" id="share-print-btn" style="padding:8px 16px; font-size:13px;">
          Print / PDF
        </button>
      </div>

      <!-- Printable 1-Page Document Container -->
      <div id="print-sheet" style="background:#FFFFFF; border-radius:28px; padding:24px; border:1px solid var(--border-subtle); box-shadow:var(--shadow-card); display:flex; flex-direction:column; gap:16px;">
        
        <!-- Document Title & Date -->
        <div style="display:flex; justify-content:space-between; align-items:flex-start; border-bottom:1.5px solid var(--bg-sunken); padding-bottom:14px;">
          <div>
            <div style="display:flex; align-items:center; gap:8px;">
              <span style="font-size:18px; color:var(--accent-orange);">♥</span>
              <h1 style="font-size:18px; font-weight:700; color:var(--ink); letter-spacing:-0.02em;">Digital Heart Twin — 30-Day Summary</h1>
            </div>
            <div style="font-size:12px; color:var(--muted); margin-top:2px;">
              Smartphone Seismocardiography (SCG) Rhythm Telemetry
            </div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:12px; font-weight:700; color:var(--ink);">${todayStr}</div>
            <div style="font-size:11px; color:var(--muted);">Subject #${subjectId}</div>
          </div>
        </div>

        <!-- Patient Info Card -->
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; padding:12px 16px; background:var(--bg-sunken); border-radius:18px; font-size:13px;">
          <div>
            <span style="color:var(--muted); font-size:11px;">Patient Name:</span>
            <div style="font-weight:700; color:var(--ink);">${name}</div>
          </div>
          <div>
            <span style="color:var(--muted); font-size:11px;">Monitoring Window:</span>
            <div style="font-weight:700; color:var(--ink);">Last 30 Days (${thirtyDays.length} readings)</div>
          </div>
        </div>

        <!-- 30-Day Metrics Grid -->
        <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:10px;">
          <div style="background:var(--bg-sunken); border-radius:16px; padding:12px; text-align:center;">
            <div style="font-size:11px; color:var(--muted);">Median Resting HR</div>
            <div style="font-size:20px; font-weight:700; color:var(--accent-orange); margin:4px 0;">${medianBpm}</div>
            <div style="font-size:10px; color:var(--muted);">BPM</div>
          </div>

          <div style="background:var(--bg-sunken); border-radius:16px; padding:12px; text-align:center;">
            <div style="font-size:11px; color:var(--muted);">HRV Span (SDNN)</div>
            <div style="font-size:15px; font-weight:700; color:var(--ink); margin:6px 0;">${hrvRange}</div>
            <div style="font-size:10px; color:var(--muted);">variance range</div>
          </div>

          <div style="background:var(--bg-sunken); border-radius:16px; padding:12px; text-align:center;">
            <div style="font-size:11px; color:var(--muted);">Rhythm Concordance</div>
            <div style="font-size:20px; font-weight:700; color:#059669; margin:4px 0;">${normalPercent}%</div>
            <div style="font-size:10px; color:var(--muted);">within baseline</div>
          </div>
        </div>

        <!-- Mini 30-Day Trend Chart -->
        <div>
          <div style="font-size:13px; font-weight:700; color:var(--ink); margin-bottom:8px;">Heart Rate Trajectory</div>
          <div style="height:120px; width:100%; border-radius:16px; background:var(--bg-sunken); padding:8px;">
            <canvas id="share-chart-canvas" style="width:100%; height:100%;"></canvas>
          </div>
        </div>

        <!-- Clinical Notes Box -->
        <div style="font-size:12px; color:var(--muted); line-height:1.45; background:rgba(255,90,31,0.05); border-left:3px solid var(--accent-orange); padding:10px 14px; border-radius:8px;">
          <strong>Clinical Interpretation:</strong> Cardiac mechanical motion (seismocardiography) records aortic valve opening (AO) and mitral valve closure (MC) micro-accelerations. Comparison is performed relative to the patient's individual trained digital twin baseline.
        </div>

        <!-- Disclaimer Footer -->
        <div style="border-top:1px solid var(--border-subtle); padding-top:12px; font-size:10.5px; color:var(--muted); text-align:center; line-height:1.4;">
          Research prototype. Not an FDA-cleared diagnostic medical device.<br/>
          Generated by Digital Heart Twin on ${todayStr}.
        </div>

      </div>

    </div>
  `;

  // Draw chart
  const canvas = document.getElementById('share-chart-canvas');
  if (canvas) {
    drawTrendsHR(canvas, [...thirtyDays].reverse());
  }

  // Handlers
  document.getElementById('share-back-btn')?.addEventListener('click', () => {
    navigateTo('trends');
  });

  document.getElementById('share-print-btn')?.addEventListener('click', () => {
    window.print();
  });
}
