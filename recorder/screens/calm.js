/**
 * screens/calm.js - Guided 5.5 Breaths/min Resonant Breathing Screen
 * Warm orange glass meditation aesthetic. Inhale 4s, Hold 2s, Exhale 4s, Rest 1s.
 * Rich dynamic multi-halo breathing animation, subtle audio chime, and tactile cues.
 */

import { icons } from '../ui/icons.js';
import { navigateTo } from '../ui/nav.js';

let _calmTimer = null;
let _breathInterval = null;
let _soundEnabled = true;
let _isPaused = false;

function playChime(freq = 432, duration = 1.4) {
  if (!_soundEnabled) return;
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, ctx.currentTime);
    gain.gain.setValueAtTime(0.06, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + duration);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + duration);
  } catch (_) {}
}

export function renderCalm(container, { onDone } = {}) {
  stopCalm();
  _isPaused = false;

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
    <div style="width:100%; height:100%; padding: 24px 24px 34px 24px; display:flex; flex-direction:column; justify-content:space-between; align-items:center; background:linear-gradient(180deg, #3A0F07 0%, #6E1F0E 40%, #B83B09 75%, #FF5A1F 100%); color:#FFFFFF; text-align:center;">
      
      <!-- Top Bar: Back, Title & Sound Toggle -->
      <div style="width:100%; display:flex; justify-content:space-between; align-items:center; z-index:10;">
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
      <div style="display:flex; flex-direction:column; align-items:center; gap:22px; z-index:10;">
        
        <div style="position:relative; width:240px; height:240px; display:flex; align-items:center; justify-content:center;">
          <!-- Ambient Radiant Glow -->
          <div id="calm-glow" style="position:absolute; width:180px; height:180px; border-radius:50%; background:radial-gradient(circle, rgba(255,160,122,0.55) 0%, rgba(255,90,31,0.2) 60%, transparent 85%); filter:blur(22px); transform:scale(0.9); transition:transform 4s cubic-bezier(0.4, 0, 0.2, 1);"></div>
          
          <!-- Outer Pulsing Border Rings -->
          <div style="position:absolute; width:230px; height:230px; border-radius:50%; border:1px solid rgba(255,255,255,0.18);"></div>
          <div style="position:absolute; width:190px; height:190px; border-radius:50%; border:1px dashed rgba(255,255,255,0.25);"></div>

          <!-- Central Resonant Breath Circle -->
          <div id="calm-circle" style="width:140px; height:140px; border-radius:50%; background:rgba(255,255,255,0.22); backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px); border:1.5px solid rgba(255,255,255,0.45); display:flex; flex-direction:column; align-items:center; justify-content:center; box-shadow:0 16px 40px rgba(0,0,0,0.3); transform:scale(0.85); transition:transform 4s cubic-bezier(0.4, 0, 0.2, 1); cursor:pointer;">
            <div id="calm-phase-text" style="font-size:16px; font-weight:700; color:#FFFFFF; letter-spacing:-0.01em;">Inhale slowly</div>
            <div id="calm-timer-display" style="font-size:20px; font-weight:700; color:rgba(255,255,255,0.95); margin-top:4px;">4</div>
          </div>
        </div>

        <!-- Guidance Card -->
        <div style="background:rgba(255,255,255,0.14); backdrop-filter:blur(14px); border:1px solid rgba(255,255,255,0.22); border-radius:24px; padding:14px 20px; max-width:320px;">
          <div style="font-size:14.5px; font-weight:700;">5.5 Breaths Per Minute</div>
          <div style="font-size:13px; color:rgba(255,255,255,0.88); margin-top:4px; line-height:1.42;">
            Harmonizes heart rate variability (HRV) and baroreflex rhythm before recording your baseline.
          </div>
        </div>

      </div>

      <!-- Bottom Session Time & Controls -->
      <div style="width:100%; display:flex; flex-direction:column; gap:12px; z-index:10;">
        <div style="font-size:13px; color:rgba(255,255,255,0.75); font-weight:500;" id="calm-total-left">
          2:00 remaining
        </div>
        <div style="display:flex; gap:10px; width:100%;">
          <button class="glass-circle-btn" id="calm-pause-btn" style="width:52px; height:52px; flex-shrink:0; font-size:20px;" aria-label="Pause breathing">
            ⏸
          </button>
          <button class="btn-white-pill" id="calm-record-now-btn" style="flex:1; justify-content:center;">
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

  // Pause / Resume toggle
  const pauseBtn = document.getElementById('calm-pause-btn');
  pauseBtn?.addEventListener('click', () => {
    _isPaused = !_isPaused;
    pauseBtn.textContent = _isPaused ? '▶️' : '⏸';
    pauseBtn.setAttribute('aria-label', _isPaused ? 'Resume breathing' : 'Pause breathing');

    if (_isPaused) {
      // Stop breathing cycle and session timer
      if (_breathInterval) {
        clearTimeout(_breathInterval);
        _breathInterval = null;
      }
      if (_calmTimer) {
        clearInterval(_calmTimer);
        _calmTimer = null;
      }
      if (phaseTxt) phaseTxt.textContent = 'Paused';
      if (circle) {
        circle.style.transition = 'transform 0.4s ease';
        circle.style.transform = 'scale(1.0)';
      }
      if (glow) {
        glow.style.transition = 'transform 0.4s ease';
        glow.style.transform = 'scale(1.0)';
      }
    } else {
      // Resume breathing and session timer
      runPhase();
      _calmTimer = setInterval(() => {
        secondsRemaining--;
        const m = Math.floor(secondsRemaining / 60);
        const s = secondsRemaining % 60;
        if (totalLeft) totalLeft.textContent = `${m}:${s < 10 ? '0' : ''}${s} remaining`;
        if (secondsRemaining <= 0) {
          stopCalm();
          if (phaseTxt) phaseTxt.textContent = 'Session Complete';
          if (circle) circle.style.transform = 'scale(1.1)';
          playChime(528, 2.0);
        }
      }, 1000);
    }
  });

  document.getElementById('calm-back-btn')?.addEventListener('click', () => {
    stopCalm();
    navigateTo('record-browse');
  });

  document.getElementById('calm-record-now-btn')?.addEventListener('click', () => {
    stopCalm();
    navigateTo('record-player');
    const playerContainer = document.getElementById('screen-record-player');
    import('./record-player.js').then((mod) => {
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
    const pTimer = setInterval(() => {
      phaseSec--;
      if (timerDsp && phaseSec >= 1) timerDsp.textContent = phaseSec;
      if (phaseSec <= 0) clearInterval(pTimer);
    }, 1000);

    _breathInterval = setTimeout(() => {
      clearInterval(pTimer);
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

function stopCalm() {
  _isPaused = false;
  if (_calmTimer) {
    clearInterval(_calmTimer);
    _calmTimer = null;
  }
  if (_breathInterval) {
    clearTimeout(_breathInterval);
    _breathInterval = null;
  }
}
