/**
 * screens/calm.js - Guided 5.5 Breaths/min Resonant Breathing Screen
 * Warm orange glass meditation aesthetic. Inhale 4s, Hold 2s, Exhale 4s, Rest 1s.
 * Rich dynamic multi-halo breathing animation, subtle audio chime, and tactile cues.
 */

import { icons } from '../ui/icons.js';
import { navigateTo } from '../ui/nav.js';

let _calmTimer = null;
let _breathInterval = null;
let _phaseTimer = null;
let _audioCtx = null;
let _soundEnabled = true;

function playChime(freq = 432, duration = 1.4) {
  if (!_soundEnabled) return;
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    if (!_audioCtx || _audioCtx.state === 'closed') {
      _audioCtx = new AudioCtx();
    }
    if (_audioCtx.state === 'suspended') {
      _audioCtx.resume().catch(() => {});
    }
    const osc = _audioCtx.createOscillator();
    const gain = _audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, _audioCtx.currentTime);
    gain.gain.setValueAtTime(0.06, _audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, _audioCtx.currentTime + duration);
    osc.connect(gain);
    gain.connect(_audioCtx.destination);
    osc.start();
    osc.stop(_audioCtx.currentTime + duration);
  } catch (_) {}
}

export function renderCalm(container, { onDone, user } = {}) {
  stopCalm();

  let secondsRemaining = 120; // 2 minutes
  let currentCycle = 1;
  const totalCycles = Math.ceil(secondsRemaining / 11);

  // 5.5 breaths/min cycle: Inhale 4s, Hold 2s, Exhale 4s, Rest 1s
  const PHASES = [
    { text: 'Inhale slowly', duration: 4000, scale: 1.38, glowScale: 1.45, freq: 432, cue: 'in' },
    { text: 'Hold gently', duration: 2000, scale: 1.38, glowScale: 1.45, freq: null, cue: 'hold' },
    { text: 'Exhale completely', duration: 4000, scale: 0.85, glowScale: 0.9, freq: 320, cue: 'out' },
    { text: 'Rest', duration: 1000, scale: 0.85, glowScale: 0.9, freq: null, cue: 'rest' }
  ];

  container.innerHTML = `
    <div style="width:100%; height:100%; padding: 18px 22px 112px 22px; display:flex; flex-direction:column; justify-content:space-between; align-items:center; background:linear-gradient(180deg, #3A0F07 0%, #6E1F0E 40%, #B83B09 75%, #FF5A1F 100%); color:#FFFFFF; text-align:center; overflow-y:auto;">
      
      <!-- Top Bar: Back, Title & Sound Toggle -->
      <div style="width:100%; display:flex; justify-content:space-between; align-items:center; z-index:10; flex-shrink:0;">
        <button class="glass-circle-btn" id="calm-back-btn" aria-label="Go back">
          ${icons.chevronLeft(20)}
        </button>
        <div>
          <div style="font-size:17px; font-weight:700;">Resonant Calm</div>
          <div style="font-size:12px; color:rgba(255,255,255,0.7); font-weight:500;" id="calm-cycle-tracker">Cycle 1 of ${totalCycles}</div>
        </div>
        <button class="glass-circle-btn" id="calm-sound-btn" aria-label="Toggle chime sound">
          🔔
        </button>
      </div>

      <!-- Breathing Circle Visualization -->
      <div style="display:flex; flex-direction:column; align-items:center; gap:14px; z-index:10; margin: 4px 0;">
        
        <div style="position:relative; width:200px; height:200px; display:flex; align-items:center; justify-content:center;">
          <!-- Ambient Radiant Glow -->
          <div id="calm-glow" style="position:absolute; width:154px; height:154px; border-radius:50%; background:radial-gradient(circle, rgba(255,160,122,0.55) 0%, rgba(255,90,31,0.2) 60%, transparent 85%); filter:blur(20px); transform:scale(0.9); transition:transform 4s cubic-bezier(0.4, 0, 0.2, 1);"></div>
          
          <!-- Outer Pulsing Border Rings -->
          <div style="position:absolute; width:194px; height:194px; border-radius:50%; border:1px solid rgba(255,255,255,0.18);"></div>
          <div style="position:absolute; width:160px; height:160px; border-radius:50%; border:1px dashed rgba(255,255,255,0.25);"></div>

          <!-- Central Resonant Breath Circle -->
          <div id="calm-circle" style="width:124px; height:124px; border-radius:50%; background:rgba(255,255,255,0.22); backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px); border:1.5px solid rgba(255,255,255,0.45); display:flex; flex-direction:column; align-items:center; justify-content:center; box-shadow:0 16px 40px rgba(0,0,0,0.3); transform:scale(0.85); transition:transform 4s cubic-bezier(0.4, 0, 0.2, 1);">
            <div id="calm-phase-text" style="font-size:15px; font-weight:700; color:#FFFFFF; letter-spacing:-0.01em;">Inhale slowly</div>
            <div id="calm-timer-display" style="font-size:19px; font-weight:700; color:rgba(255,255,255,0.95); margin-top:3px;">4</div>
          </div>
        </div>

        <!-- Guidance Card -->
        <div style="background:rgba(255,255,255,0.14); backdrop-filter:blur(14px); border:1px solid rgba(255,255,255,0.22); border-radius:20px; padding:11px 16px; max-width:320px;">
          <div style="font-size:14px; font-weight:700;">5.5 Breaths Per Minute</div>
          <div style="font-size:12.5px; color:rgba(255,255,255,0.88); margin-top:3px; line-height:1.36;">
            Harmonizes heart rate variability (HRV) and baroreflex rhythm before recording your baseline.
          </div>
        </div>

      </div>

      <!-- Bottom Session Time & Controls (Positioned cleanly above the 5-tab navigation bar) -->
      <div style="width:100%; display:flex; flex-direction:column; gap:8px; z-index:10; flex-shrink:0;">
        <div style="font-size:13px; color:rgba(255,255,255,0.85); font-weight:600;" id="calm-total-left">
          2:00 remaining
        </div>
        <div style="display:flex; align-items:center; gap:10px; width:100%;">
          <button class="btn-white-pill" id="calm-record-now-btn" style="flex:1; justify-content:center; padding:13px 16px;">
            ${icons.play(16)}
            <span>Record Your Heart Now</span>
          </button>
        </div>
      </div>

    </div>
  `;

  const circle = document.getElementById('calm-circle');
  const glow = document.getElementById('calm-glow');
  const phaseTxt = document.getElementById('calm-phase-text');
  const timerDsp = document.getElementById('calm-timer-display');
  const totalLeft = document.getElementById('calm-total-left');
  const cycleTracker = document.getElementById('calm-cycle-tracker');
  const soundBtn = document.getElementById('calm-sound-btn');

  // Sound toggle
  soundBtn?.addEventListener('click', () => {
    _soundEnabled = !_soundEnabled;
    soundBtn.textContent = _soundEnabled ? '🔔' : '🔕';
  });

  document.getElementById('calm-back-btn')?.addEventListener('click', () => {
    stopCalm();
    navigateTo('record-browse');
  });

  document.getElementById('calm-record-now-btn')?.addEventListener('click', async () => {
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
      try { await DeviceMotionEvent.requestPermission(); } catch (_) {}
    }
    stopCalm();
    navigateTo('record-player');
    const playerContainer = document.getElementById('screen-record-player');
    import('./record-player.js?v=32').then((mod) => {
      mod.renderRecordPlayer(playerContainer, {
        context: 'Resting',
        duration: 60,
        user
      });
    });
  });

  // Breathing Phase Machine
  let phaseIndex = 0;
  let phaseSec = 4;

  function runPhase() {
    if (!document.getElementById('screen-calm')?.classList.contains('active')) {
      stopCalm();
      return;
    }
    const p = PHASES[phaseIndex];
    if (!circle || !glow || !phaseTxt) return;

    phaseTxt.textContent = p.text;
    circle.style.transition = `transform ${p.duration}ms cubic-bezier(0.4, 0, 0.2, 1)`;
    circle.style.transform = `scale(${p.scale})`;
    glow.style.transition = `transform ${p.duration}ms cubic-bezier(0.4, 0, 0.2, 1)`;
    glow.style.transform = `scale(${p.glowScale})`;

    if (p.freq) playChime(p.freq);

    phaseSec = Math.round(p.duration / 1000);
    if (timerDsp) timerDsp.textContent = phaseSec;

    // Countdown within phase
    if (_phaseTimer) {
      clearInterval(_phaseTimer);
      _phaseTimer = null;
    }
    _phaseTimer = setInterval(() => {
      phaseSec--;
      if (timerDsp && phaseSec >= 1) timerDsp.textContent = phaseSec;
      if (phaseSec <= 0 && _phaseTimer) {
        clearInterval(_phaseTimer);
        _phaseTimer = null;
      }
    }, 1000);

    _breathInterval = setTimeout(() => {
      if (_phaseTimer) {
        clearInterval(_phaseTimer);
        _phaseTimer = null;
      }
      phaseIndex = (phaseIndex + 1) % PHASES.length;
      if (phaseIndex === 0) {
        currentCycle++;
        if (cycleTracker) cycleTracker.textContent = `Cycle ${currentCycle} of ${totalCycles}`;
      }
      runPhase();
    }, p.duration);
  }

  // Start breathing cycle
  runPhase();

  // Overall session countdown
  _calmTimer = setInterval(() => {
    secondsRemaining--;
    const m = Math.floor(secondsRemaining / 60);
    const s = secondsRemaining % 60;
    if (totalLeft) totalLeft.textContent = `${m}:${s < 10 ? '0' : ''}${s} remaining`;

    if (secondsRemaining <= 0) {
      stopCalm();
      if (phaseTxt) phaseTxt.textContent = 'Session Complete';
      if (circle) circle.style.transform = 'scale(1.1)';
      playChime(528, 2.0); // Warm completion chime
    }
  }, 1000);
}

export function stopCalm() {
  if (_calmTimer) {
    clearInterval(_calmTimer);
    _calmTimer = null;
  }
  if (_breathInterval) {
    clearTimeout(_breathInterval);
    _breathInterval = null;
  }
  if (_phaseTimer) {
    clearInterval(_phaseTimer);
    _phaseTimer = null;
  }
  if (_audioCtx && _audioCtx.state !== 'closed') {
    try { _audioCtx.close(); } catch (_) {}
    _audioCtx = null;
  }
}
