/**
 * ui/charts.js - High-Contrast Accessible Canvas Charts
 * Renders Poincaré, Live Waveform, and Trends charts with screen-reader text alternatives.
 */

export function fitCanvas(canvas) {
  if (!canvas) return null;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 300;
  const h = canvas.clientHeight || 150;
  if (w < 10 || h < 10) return null;

  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

/** Draw Calm White Live Waveform (for Phone 3 Full-Bleed Photo Player) */
export function drawCalmWhiteWaveform(canvas, buffer) {
  const f = fitCanvas(canvas);
  if (!f) return;
  const { ctx, w, h } = f;

  // Transparent clear so the warm portrait photo is visible behind
  ctx.clearRect(0, 0, w, h);

  if (!buffer || buffer.length < 2) return;

  const N = buffer.length;
  const midY = h / 2;
  const scale = h * 0.42;

  // Soft glow
  ctx.shadowColor = 'rgba(255, 255, 255, 0.6)';
  ctx.shadowBlur = 6;

  ctx.beginPath();
  ctx.strokeStyle = '#FFFFFF';
  ctx.lineWidth = 2.4;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (let i = 0; i < N; i++) {
    const x = (i / (N - 1)) * w;
    const y = midY - buffer[i] * scale;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Reset shadow
  ctx.shadowBlur = 0;
}

/** Draw Live Oscilloscope Waveform */
export function drawLiveWaveform(canvas, buffer, theme = 'dark') {
  const f = fitCanvas(canvas);
  if (!f) return;
  const { ctx, w, h } = f;

  // Background Grid
  ctx.fillStyle = theme === 'dark' ? '#0B0F19' : '#F8FAFC';
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = theme === 'dark' ? 'rgba(255, 255, 255, 0.04)' : 'rgba(0, 0, 0, 0.04)';
  ctx.lineWidth = 1;
  for (let x = 16; x < w; x += 16) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 16; y < h; y += 16) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  if (!buffer || buffer.length < 2) return;

  const N = buffer.length;
  ctx.beginPath();
  ctx.strokeStyle = '#FF5A1F'; // Accent orange
  ctx.lineWidth = 2.2;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  const midY = h / 2;
  const scale = h * 0.42;

  for (let i = 0; i < N; i++) {
    const x = (i / (N - 1)) * w;
    const y = midY - buffer[i] * scale;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

/** Draw Poincaré Plot (RR_n vs RR_n+1) */
export function drawPoincare(canvas, rr, theme = 'light') {
  const f = fitCanvas(canvas);
  if (!f) return;
  const { ctx, w, h } = f;

  ctx.fillStyle = theme === 'dark' ? '#1F1A17' : '#FFFFFF';
  ctx.fillRect(0, 0, w, h);

  if (!rr || rr.length < 4) {
    ctx.fillStyle = theme === 'dark' ? '#8A817A' : '#8A817A';
    ctx.font = '12px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('Not enough beats detected', w / 2, h / 2);
    return;
  }

  // Identity line (x = y)
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h);
  ctx.lineTo(w, 0);
  ctx.stroke();

  const minRR = 400; // ms
  const maxRR = 1400; // ms
  const span = maxRR - minRR;

  ctx.fillStyle = '#FF5A1F';
  for (let i = 0; i < rr.length - 1; i++) {
    const x = ((rr[i] - minRR) / span) * w;
    const y = h - ((rr[i + 1] - minRR) / span) * h;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

/** Draw Trends Heart Rate Line Chart */
export function drawTrendsHR(canvas, dataPoints) {
  const f = fitCanvas(canvas);
  if (!f) return;
  const { ctx, w, h } = f;

  ctx.clearRect(0, 0, w, h);

  if (!dataPoints || dataPoints.length === 0) {
    ctx.fillStyle = '#8A817A';
    ctx.font = '13px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('No recordings in this timeframe', w / 2, h / 2);
    return;
  }

  const padding = { top: 20, right: 16, bottom: 24, left: 36 };
  const chartW = w - padding.left - padding.right;
  const chartH = h - padding.top - padding.bottom;

  const bpms = dataPoints.map((d) => d.bpm || d.hr || 70);
  const minBPM = Math.max(40, Math.min(...bpms) - 8);
  const maxBPM = Math.min(160, Math.max(...bpms) + 8);
  const span = maxBPM - minBPM || 1;

  // Normal band (60 to 100 bpm) shaded area
  const y60 = padding.top + chartH - ((60 - minBPM) / span) * chartH;
  const y100 = padding.top + chartH - ((100 - minBPM) / span) * chartH;
  ctx.fillStyle = 'rgba(16, 185, 129, 0.08)';
  ctx.fillRect(padding.left, Math.max(padding.top, y100), chartW, Math.min(chartH, y60 - y100));

  // Horizontal Grid Lines & Y-Axis Labels
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.06)';
  ctx.fillStyle = '#8A817A';
  ctx.font = '10px ui-monospace, monospace';
  ctx.textAlign = 'right';

  const steps = 3;
  for (let i = 0; i <= steps; i++) {
    const yVal = Math.round(minBPM + (span / steps) * i);
    const y = padding.top + chartH - (i / steps) * chartH;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(w - padding.right, y);
    ctx.stroke();
    ctx.fillText(`${yVal}`, padding.left - 6, y + 3);
  }

  // Draw Line
  ctx.beginPath();
  ctx.strokeStyle = '#FF5A1F';
  ctx.lineWidth = 2.5;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  const pts = [];
  dataPoints.forEach((d, i) => {
    const val = d.bpm || d.hr || 70;
    const x = padding.left + (i / Math.max(1, dataPoints.length - 1)) * chartW;
    const y = padding.top + chartH - ((val - minBPM) / span) * chartH;
    pts.push({ x, y, val });
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Draw Dots
  pts.forEach((pt) => {
    ctx.beginPath();
    ctx.fillStyle = '#FF5A1F';
    ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#FFFFFF';
    ctx.arc(pt.x, pt.y, 1.8, 0, Math.PI * 2);
    ctx.fill();
  });
}
