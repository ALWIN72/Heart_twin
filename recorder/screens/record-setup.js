/**
 * screens/record-setup.js - Step 1: Get Ready & Placement Guide
 */

import { navigateTo } from '../ui/nav.js';

let _selectedContext = 'Resting';
let _selectedDuration = 60; // 60s default for optimal ML training

export function renderRecordSetup(container, { onStartRecording } = {}) {
  container.innerHTML = `
    <div style="display:flex; flex-direction:column; gap: var(--sp-4);">
      <!-- Screen Header -->
      <div style="display:flex; align-items:center; justify-content:space-between;">
        <div>
          <h2>Get Ready to Record</h2>
          <p style="font-size:13px; color:var(--text-muted);">Step 1 of 2: Phone placement</p>
        </div>
        <button class="btn btn-ghost" id="setup-calm-btn" style="font-size:13px; color:var(--primary); padding:4px 8px;">
          🧘 Calm down first
        </button>
      </div>

      <!-- Placement Visual Graphic -->
      <div class="card" style="align-items:center; text-align:center; padding: var(--sp-6) var(--sp-4);">
        <div style="width:140px; height:180px; position:relative; margin-bottom:12px;">
          <!-- Torso Outline SVG -->
          <svg viewBox="0 0 100 130" width="100%" height="100%">
            <path d="M 25 15 C 35 12, 65 12, 75 15 C 88 28, 95 65, 85 125 L 15 125 C 5 65, 12 28, 25 15 Z" 
                  fill="var(--bg-sunken)" stroke="var(--border-strong)" stroke-width="2.5" />
            <!-- Sternum line -->
            <line x1="50" y1="25" x2="50" y2="75" stroke="var(--border-subtle)" stroke-width="2" stroke-dasharray="3,3" />
            <!-- Phone Graphic placed on lower breastbone -->
            <rect x="36" y="44" width="28" height="46" rx="5" fill="var(--primary)" stroke="#FFFFFF" stroke-width="1.8" />
            <circle cx="50" cy="50" r="1.5" fill="#FFFFFF" opacity="0.8" />
            <!-- Gentle pulse ring -->
            <circle cx="50" cy="67" r="14" fill="none" stroke="var(--primary-glow)" stroke-width="2">
              <animate attributeName="r" values="10;22;10" dur="2s" repeatCount="indefinite" />
              <animate attributeName="opacity" values="0.8;0;0.8" dur="2s" repeatCount="indefinite" />
            </circle>
          </svg>
        </div>
        <strong style="font-size:15px; color:var(--text-main);">Place phone flat on your lower breastbone</strong>
        <p style="font-size:13px; color:var(--text-muted); max-width:280px; margin-top:4px;">
          Lie down comfortably or recline still. Rest the phone screen-up against your lower sternum.
        </p>
      </div>

      <!-- Activity Context Selector -->
      <div class="card">
        <label style="font-size:13px; font-weight:600; color:var(--text-main);">Activity Context</label>
        <p style="font-size:12px; color:var(--text-muted); margin-bottom:4px;">
          Results are compared only against your matching state.
        </p>
        <div class="chip-group" role="radiogroup" aria-label="Recording context">
          <button type="button" class="chip-btn active" data-context="Resting" role="radio" aria-checked="true">
            🛏️ Resting
          </button>
          <button type="button" class="chip-btn" data-context="After activity" role="radio" aria-checked="false">
            🏃 After activity
          </button>
          <button type="button" class="chip-btn" data-context="Other" role="radio" aria-checked="false">
            ✨ Other
          </button>
        </div>
      </div>

      <!-- Duration Selector -->
      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <label style="font-size:13px; font-weight:600; color:var(--text-main);">Recording Duration</label>
          <span class="badge badge-normal" id="duration-badge">Optimal for ML</span>
        </div>
        <div class="segmented-control" role="tablist">
          <button type="button" class="segmented-option active" data-duration="60" role="tab" aria-selected="true">
            60 Seconds (Recommended)
          </button>
          <button type="button" class="segmented-option" data-duration="30" role="tab" aria-selected="false">
            30 Seconds (Quick)
          </button>
        </div>
        <p style="font-size:12px; color:var(--text-muted); margin-top:4px;" id="duration-tip">
          60 seconds captures steady heart rate variability and builds your digital twin model with high fidelity.
        </p>
      </div>

      <!-- Primary Action -->
      <button class="btn btn-primary btn-block btn-start-record" id="start-recording-btn">
        Start 60s Recording →
      </button>
    </div>
  `;

  // Context Selection
  const chips = container.querySelectorAll('.chip-btn');
  chips.forEach((chip) => {
    chip.onclick = () => {
      chips.forEach((c) => {
        c.classList.remove('active');
        c.setAttribute('aria-checked', 'false');
      });
      chip.classList.add('active');
      chip.setAttribute('aria-checked', 'true');
      _selectedContext = chip.dataset.context;
    };
  });

  // Duration Selection
  const durOptions = container.querySelectorAll('.segmented-option');
  const durBadge = container.querySelector('#duration-badge');
  const durTip = container.querySelector('#duration-tip');
  const startBtn = container.querySelector('#start-recording-btn');

  durOptions.forEach((opt) => {
    opt.onclick = () => {
      durOptions.forEach((o) => {
        o.classList.remove('active');
        o.setAttribute('aria-selected', 'false');
      });
      opt.classList.add('active');
      opt.setAttribute('aria-selected', 'true');
      _selectedDuration = parseInt(opt.dataset.duration, 10);

      if (_selectedDuration === 60) {
        durBadge.textContent = 'Optimal for ML';
        durBadge.className = 'badge badge-normal';
        durTip.textContent = '60 seconds captures steady heart rate variability and builds your digital twin model with high fidelity.';
        startBtn.textContent = 'Start 60s Recording →';
      } else {
        durBadge.textContent = 'Quick Check';
        durBadge.className = 'badge badge-neutral';
        durTip.textContent = '30 seconds gives a fast heart rate check. Note: 60s is recommended for personal model training.';
        startBtn.textContent = 'Start 30s Recording →';
      }
    };
  });

  // Calm Down Link
  const calmBtn = container.querySelector('#setup-calm-btn');
  if (calmBtn) {
    calmBtn.onclick = () => navigateTo('calm');
  }

  // Start Button Click
  if (startBtn) {
    startBtn.onclick = () => {
      if ('vibrate' in navigator) navigator.vibrate(80);
      if (typeof onStartRecording === 'function') {
        onStartRecording({
          context: _selectedContext,
          duration: _selectedDuration,
        });
      }
    };
  }
}

