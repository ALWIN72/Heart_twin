/**
 * screens/trends.js - Trends & Historical Heart Rhythms
 * Matches light card browse aesthetic of Phone 2.
 * Segmented 7d/30d/All control, dynamic stats tiles, heart rate line chart,
 * variation bar chart, and recording list with swipe-to-delete and undo toast.
 */

import { getRecordings, deleteRecording, saveRecording } from '../storage/db.js';
import { drawTrendsHR } from '../ui/charts.js';
import { navigateTo } from '../ui/nav.js';
import { showToast } from '../ui/toast.js';
import { icons } from '../ui/icons.js';

let _activePeriod = '7'; // '7', '30', 'all'

export async function renderTrends(container, { user, onSelectRecording } = {}) {
  const subjectId = user ? user.subject_id : '1001';
  const allRecordings = await getRecordings(subjectId, 100);

  // Filter based on selected period
  const now = Date.now();
  const filtered = allRecordings.filter((r) => {
    if (_activePeriod === 'all') return true;
    const days = parseInt(_activePeriod, 10);
    const itemTime = new Date(r.timestamp).getTime();
    return (now - itemTime) <= days * 86400 * 1000;
  });

  // Calculate dynamic stats
  let avgHr = 72;
  let stabilityPct = 100;
  if (filtered.length > 0) {
    const validHrs = filtered.filter((r) => r.hr && r.hr > 40 && r.hr < 180).map((r) => r.hr);
    if (validHrs.length > 0) {
      avgHr = Math.round(validHrs.reduce((a, b) => a + b, 0) / validHrs.length);
    }
    const stableCount = filtered.filter((r) => !r.verdict || !r.verdict.includes('Different')).length;
    stabilityPct = Math.round((stableCount / filtered.length) * 100);
  }

  container.innerHTML = `
    <div style="padding: 24px 20px 104px 20px; display:flex; flex-direction:column; gap:18px; min-height:100%; background:var(--light-page);">
      
      <!-- Top Bar: Heading & 30-Day Export Shortcut -->
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <h2 class="screen-title">Trends &amp; History</h2>
        <button id="trends-share-btn" style="background:rgba(255,90,31,0.1); border:none; color:var(--accent-orange); font-size:12px; font-weight:700; padding:6px 14px; border-radius:var(--r-pill); cursor:pointer;">
          Doctor Summary
        </button>
      </div>

      <!-- Segmented Time Period Toggle (7 Days, 30 Days, All) -->
      <div style="display:flex; background:#E5DFDA; border-radius:var(--r-pill); padding:4px; width:100%;">
        <button type="button" class="trends-seg-btn ${_activePeriod === '7' ? 'active' : ''}" data-period="7" style="flex:1; border:none; padding:8px; border-radius:var(--r-pill); font-size:13px; font-weight:600; cursor:pointer; background:${_activePeriod === '7' ? '#FFFFFF' : 'transparent'}; color:${_activePeriod === '7' ? 'var(--ink)' : 'var(--muted)'}; box-shadow:${_activePeriod === '7' ? '0 2px 6px rgba(0,0,0,0.06)' : 'none'};">
          7 Days
        </button>
        <button type="button" class="trends-seg-btn ${_activePeriod === '30' ? 'active' : ''}" data-period="30" style="flex:1; border:none; padding:8px; border-radius:var(--r-pill); font-size:13px; font-weight:600; cursor:pointer; background:${_activePeriod === '30' ? '#FFFFFF' : 'transparent'}; color:${_activePeriod === '30' ? 'var(--ink)' : 'var(--muted)'}; box-shadow:${_activePeriod === '30' ? '0 2px 6px rgba(0,0,0,0.06)' : 'none'};">
          30 Days
        </button>
        <button type="button" class="trends-seg-btn ${_activePeriod === 'all' ? 'active' : ''}" data-period="all" style="flex:1; border:none; padding:8px; border-radius:var(--r-pill); font-size:13px; font-weight:600; cursor:pointer; background:${_activePeriod === 'all' ? '#FFFFFF' : 'transparent'}; color:${_activePeriod === 'all' ? 'var(--ink)' : 'var(--muted)'}; box-shadow:${_activePeriod === 'all' ? '0 2px 6px rgba(0,0,0,0.06)' : 'none'};">
          All Time
        </button>
      </div>

      <!-- Dynamic Stat Cards Row -->
      <div style="display:flex; gap:12px;">
        <div style="flex:1; background:#FFFFFF; border-radius:22px; padding:16px; border:1px solid var(--border-subtle); box-shadow:var(--shadow-card);">
          <div style="font-size:11.5px; color:var(--muted); font-weight:600;">Avg Resting HR</div>
          <div style="font-size:24px; font-weight:700; color:var(--ink); margin-top:2px;">
            ${avgHr} <span style="font-size:13px; font-weight:500; color:var(--muted);">bpm</span>
          </div>
          <div style="font-size:11px; color:#059669; font-weight:600; margin-top:2px;">● Stable window</div>
        </div>

        <div style="flex:1; background:#FFFFFF; border-radius:22px; padding:16px; border:1px solid var(--border-subtle); box-shadow:var(--shadow-card);">
          <div style="font-size:11.5px; color:var(--muted); font-weight:600;">Twin Stability</div>
          <div style="font-size:24px; font-weight:700; color:var(--accent-orange); margin-top:2px;">
            ${stabilityPct}%
          </div>
          <div style="font-size:11px; color:var(--muted); margin-top:2px;">vs personal baseline</div>
        </div>
      </div>

      <!-- 1. Heart Rate Line Chart Card -->
      <div style="background:#FFFFFF; border-radius:28px; padding:20px; border:1px solid var(--border-subtle); box-shadow:var(--shadow-card);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
          <div style="font-size:15px; font-weight:700; color:var(--ink);">Heart Rate Over Time</div>
          <div style="font-size:11px; color:#059669; font-weight:600; background:rgba(16,185,129,0.1); padding:2px 8px; border-radius:var(--r-pill);">
            60-100 normal band
          </div>
        </div>
        <div style="height:140px; width:100%;">
          <canvas id="trends-hr-canvas" style="width:100%; height:100%;"></canvas>
        </div>
      </div>

      <!-- 2. "How Different from Usual" Rhythm Variation Bar Chart -->
      <div style="background:#FFFFFF; border-radius:28px; padding:20px; border:1px solid var(--border-subtle); box-shadow:var(--shadow-card);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
          <div style="font-size:15px; font-weight:700; color:var(--ink);">Variation from Baseline</div>
          <div style="font-size:11px; color:var(--muted);">Self-comparison</div>
        </div>
        ${renderVariationBars(filtered)}
      </div>

      <!-- 3. Historical Recording Cards List -->
      <div style="background:#FFFFFF; border-radius:28px; padding:20px; border:1px solid var(--border-subtle); box-shadow:var(--shadow-card); display:flex; flex-direction:column; gap:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div style="font-size:15px; font-weight:700; color:var(--ink);">Recorded Sessions</div>
          <div style="font-size:12px; color:var(--muted);">${filtered.length} session${filtered.length === 1 ? '' : 's'}</div>
        </div>

        <div style="display:flex; flex-direction:column; gap:10px;">
          ${filtered.length === 0 ? `
            <div style="text-align:center; padding:24px 10px; color:var(--muted); font-size:13.5px;">
              No heart recordings yet in this period.<br/>
              Tap <strong>Record</strong> to take your first 60s reading!
            </div>
          ` : filtered.map((rec) => {
            const dateStr = new Date(rec.timestamp).toLocaleDateString(undefined, {
              month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
            });
            const bpm = Math.round(rec.hr || 72);
            let statusPillBg = 'rgba(16,185,129,0.12)';
            let statusPillColor = '#059669';
            let label = 'Normal';
            if (rec.verdict && rec.verdict.includes('Different')) {
              statusPillBg = 'rgba(239,68,68,0.12)';
              statusPillColor = '#DC2626';
              label = 'Different';
            } else if (rec.verdict && rec.verdict.includes('Slightly')) {
              statusPillBg = 'rgba(245,158,11,0.12)';
              statusPillColor = '#D97706';
              label = 'Slight';
            }

            return `
              <div class="trend-rec-row" data-id="${rec.id}" style="display:flex; justify-content:space-between; align-items:center; padding:12px 14px; background:var(--bg-sunken); border-radius:18px; cursor:pointer; transition:transform var(--t-dur) var(--t-ease);">
                <div style="display:flex; align-items:center; gap:12px;">
                  <div style="width:36px; height:36px; border-radius:12px; background:rgba(255,90,31,0.1); color:var(--accent-orange); display:flex; align-items:center; justify-content:center;">
                    ${icons.heart(18)}
                  </div>
                  <div>
                    <div style="font-size:14px; font-weight:700; color:var(--ink);">${bpm} BPM • ${rec.context || 'Resting'}</div>
                    <div style="font-size:11px; color:var(--muted); margin-top:2px;">${dateStr}</div>
                  </div>
                </div>

                <div style="display:flex; align-items:center; gap:10px;">
                  <span style="font-size:11px; font-weight:700; padding:4px 10px; border-radius:var(--r-pill); background:${statusPillBg}; color:${statusPillColor};">
                    ${label}
                  </span>
                  <button class="delete-rec-btn" data-id="${rec.id}" title="Delete" style="width:28px; height:28px; border-radius:50%; background:transparent; border:none; color:var(--muted); font-size:14px; cursor:pointer;">
                    ✕
                  </button>
                </div>
              </div>
            `;
          }).join('')}
        </div>
      </div>

    </div>
  `;

  // Draw Heart Rate Chart
  const hrCanvas = document.getElementById('trends-hr-canvas');
  if (hrCanvas) {
    drawTrendsHR(hrCanvas, [...filtered].reverse());
  }

  // Segmented control click handlers
  container.querySelectorAll('.trends-seg-btn').forEach((btn) => {
    btn.onclick = () => {
      _activePeriod = btn.dataset.period;
      renderTrends(container, { user, onSelectRecording });
    };
  });

  // Export summary button
  document.getElementById('trends-share-btn')?.addEventListener('click', () => {
    navigateTo('share');
  });

  // Row inspection click handlers
  container.querySelectorAll('.trend-rec-row').forEach((row) => {
    row.onclick = (e) => {
      if (e.target.closest('.delete-rec-btn')) return; // Ignore if clicking delete
      const recId = row.dataset.id;
      const rec = allRecordings.find((r) => r.id === recId);
      if (rec) {
        navigateTo('result');
        const resContainer = document.getElementById('screen-result');
        import('./result.js').then((mod) => {
          mod.renderResult(resContainer, {
            analysisResult: rec,
            serverScoreResult: { score: 0.02, threshold: 0.05 },
            context: rec.context,
            user
          });
        });
      }
    };
  });

  // Delete recording with Undo Toast
  container.querySelectorAll('.delete-rec-btn').forEach((btn) => {
    btn.onclick = async (e) => {
      e.stopPropagation();
      const recId = btn.dataset.id;
      const recToDelete = allRecordings.find((r) => r.id === recId);
      if (!recToDelete) return;

      await deleteRecording(recId);
      renderTrends(container, { user, onSelectRecording });

      showToast('Recording deleted', 'normal', {
        actionLabel: 'Undo',
        onAction: async () => {
          await saveRecording(recToDelete);
          renderTrends(container, { user, onSelectRecording });
          showToast('Restored recording', 'normal');
        }
      });
    };
  });
}

function renderVariationBars(recordings) {
  if (!recordings || recordings.length === 0) {
    return `<div style="font-size:12px; color:var(--muted); text-align:center; padding:12px 0;">No readings to calculate variation.</div>`;
  }

  let normalCount = 0;
  let slightCount = 0;
  let diffCount = 0;

  recordings.forEach((r) => {
    if (r.verdict && r.verdict.includes('Different')) diffCount++;
    else if (r.verdict && r.verdict.includes('Slightly')) slightCount++;
    else normalCount++;
  });

  const total = recordings.length || 1;
  const normalPct = Math.round((normalCount / total) * 100);
  const slightPct = Math.round((slightCount / total) * 100);
  const diffPct = Math.round((diffCount / total) * 100);

  return `
    <div style="width:100%; height:10px; border-radius:5px; background:var(--bg-sunken); display:flex; overflow:hidden;">
      <div style="width:${normalPct}%; background:#10B981;" title="Normal: ${normalPct}%"></div>
      <div style="width:${slightPct}%; background:#F59E0B;" title="Slight variance: ${slightPct}%"></div>
      <div style="width:${diffPct}%; background:#EF4444;" title="Different: ${diffPct}%"></div>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:12px; color:var(--muted); margin-top:2px;">
      <span style="color:#059669; font-weight:600;">● Normal (${normalPct}%)</span>
      <span style="color:#D97706; font-weight:600;">● Slight (${slightPct}%)</span>
      <span style="color:#DC2626; font-weight:600;">● Different (${diffPct}%)</span>
    </div>
  `;
}
