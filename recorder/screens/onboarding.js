/**
 * screens/onboarding.js - 3-Card Warm Orange Glass Onboarding
 * Matches aesthetic of Phone 1 & 3 reference.
 */

import { IMAGES } from '../config/images.js';
import { icons } from '../ui/icons.js';
import { setSetting } from '../storage/db.js';
import { navigateTo } from '../ui/nav.js';
import { showToast } from '../ui/toast.js';

let _currentStep = 1;

export function renderOnboarding(container, { onComplete, onUserAuthenticated } = {}) {
  _currentStep = 1;

  container.innerHTML = `
    <div class="onboarding-wrap">
      <!-- Top Dots Indicator -->
      <div class="dots-indicator" role="tablist" aria-label="Onboarding Progress">
        <div class="dot-pill active" id="dot-1"></div>
        <div class="dot-pill" id="dot-2"></div>
        <div class="dot-pill" id="dot-3"></div>
      </div>

      <!-- Card 1: Value Proposition -->
      <div class="onboarding-card active" id="onboard-card-1">
        <div style="width:72px; height:72px; border-radius:50%; background:rgba(255,255,255,0.18); backdrop-filter:blur(16px); display:flex; align-items:center; justify-content:center; border:1px solid rgba(255,255,255,0.3); margin-bottom:8px;">
          ${icons.heart(36)}
        </div>
        <h1 class="hero-title" style="font-size:36px; text-align:center;">Your Heart, in Your Pocket</h1>
        <p style="font-size:16px; color:rgba(255,255,255,0.85); line-height:1.5; max-width:320px;">
          Understand your normal cardiac rhythm from subtle chest vibrations recorded by your smartphone. No external straps or sensors required.
        </p>
        <div style="flex:1;"></div>
        <div style="display:flex; flex-direction:column; gap:12px; width:100%; margin-top:24px;">
          <button class="btn-white-pill" id="btn-next-1" style="width:100%;">
            Get Started
          </button>
          <button type="button" id="btn-goto-signin" style="background:transparent; border:none; color:rgba(255,255,255,0.85); font-size:14px; font-weight:600; padding:10px; cursor:pointer;">
            Already have an account? Sign In
          </button>
        </div>
      </div>

      <!-- Card 2: Research Disclaimer -->
      <div class="onboarding-card" id="onboard-card-2">
        <div style="width:72px; height:72px; border-radius:50%; background:rgba(255,255,255,0.18); backdrop-filter:blur(16px); display:flex; align-items:center; justify-content:center; border:1px solid rgba(255,255,255,0.3); margin-bottom:8px;">
          ${icons.activity(34)}
        </div>
        <h1 class="hero-title" style="font-size:32px; text-align:center;">A Research Tool, Not a Doctor</h1>
        <div style="background:rgba(255,255,255,0.14); backdrop-filter:blur(16px); border:1px solid rgba(255,255,255,0.22); border-radius:24px; padding:16px; text-align:left; font-size:13.5px; color:rgba(255,255,255,0.9); line-height:1.45;">
          <p style="margin-bottom:8px; font-weight:700;">Important notice:</p>
          <p style="margin-bottom:8px;">
            Digital Heart Twin is an investigational research prototype for personal autonomic rhythm tracking. It is not an FDA-cleared diagnostic device.
          </p>
          <p>
            If you ever experience acute chest discomfort, shortness of breath, dizziness or pressure, seek emergency medical care immediately.
          </p>
        </div>
        <div style="flex:1;"></div>
        <div style="display:flex; flex-direction:column; gap:12px; width:100%; margin-top:24px;">
          <button class="btn-white-pill" id="btn-next-2" style="width:100%;">
            I Understand &amp; Agree
          </button>
        </div>
      </div>

      <!-- Card 3: Placement & Motion Sensors -->
      <div class="onboarding-card" id="onboard-card-3">
        <div style="width:180px; height:120px; border-radius:20px; overflow:hidden; background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.2); display:flex; align-items:center; justify-content:center;">
          <img src="${IMAGES.sternumGuide}" alt="Phone placed flat on breastbone" style="width:100%; height:100%; object-fit:contain;" />
        </div>
        <h1 class="hero-title" style="font-size:30px; text-align:center;">How to Record</h1>
        <p style="font-size:15px; color:rgba(255,255,255,0.85); line-height:1.45; max-width:320px;">
          Place your smartphone flat on your lower breastbone (sternum). Breathe calmly and stay still for 60 seconds.
        </p>

        <!-- Sensor Permission Status Box -->
        <div id="sensor-status-box" style="width:100%; background:rgba(0,0,0,0.25); border-radius:18px; padding:12px 16px; font-size:13px; text-align:left; color:#FFFFFF;">
          <div style="display:flex; align-items:center; gap:8px;">
            <span id="sensor-icon">📱</span>
            <span id="sensor-text" style="font-weight:600;">Motion Sensor Access Required</span>
          </div>
          <div id="sensor-help-text" style="font-size:12px; color:rgba(255,255,255,0.7); margin-top:4px;">
            Enables micro-vibration seismocardiography recording.
          </div>
        </div>

        <div style="flex:1;"></div>
        <div style="display:flex; flex-direction:column; gap:12px; width:100%; margin-top:20px;">
          <button class="btn-white-pill" id="btn-request-sensor" style="width:100%;">
            Allow Motion Sensors
          </button>
          <button class="btn-orange-pill hidden" id="btn-finish-onboarding" style="width:100%;">
            Continue to Account →
          </button>
        </div>
      </div>
    </div>
  `;

  // Navigation between cards
  const dot1 = document.getElementById('dot-1');
  const dot2 = document.getElementById('dot-2');
  const dot3 = document.getElementById('dot-3');

  function showCard(step) {
    _currentStep = step;
    document.getElementById('onboard-card-1').classList.toggle('active', step === 1);
    document.getElementById('onboard-card-2').classList.toggle('active', step === 2);
    document.getElementById('onboard-card-3').classList.toggle('active', step === 3);

    dot1.classList.toggle('active', step === 1);
    dot2.classList.toggle('active', step === 2);
    dot3.classList.toggle('active', step === 3);
  }

  document.getElementById('btn-next-1')?.addEventListener('click', () => showCard(2));
  document.getElementById('btn-goto-signin')?.addEventListener('click', () => navigateTo('auth'));
  document.getElementById('btn-next-2')?.addEventListener('click', () => showCard(3));

  // Step 3: Motion Sensor Permission
  const sensorBtn = document.getElementById('btn-request-sensor');
  const finishBtn = document.getElementById('btn-finish-onboarding');
  const sensorText = document.getElementById('sensor-text');
  const sensorHelp = document.getElementById('sensor-help-text');

  sensorBtn?.addEventListener('click', async () => {
    sensorBtn.textContent = 'Checking sensors...';

    // 1. iOS 13+ DeviceMotionEvent permission
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
      try {
        const response = await DeviceMotionEvent.requestPermission();
        if (response === 'granted') {
          handleSensorGranted();
          return;
        } else {
          handleSensorDenied();
          return;
        }
      } catch (err) {
        console.warn('DeviceMotionEvent permission error:', err);
        handleSensorDenied();
        return;
      }
    }

    // 2. Generic Sensors or standard DeviceMotionEvent (Android/Desktop)
    if ('DeviceMotionEvent' in window || 'LinearAccelerationSensor' in window) {
      handleSensorGranted();
    } else {
      handleSensorDenied();
    }
  });

  function handleSensorGranted() {
    sensorText.textContent = 'Motion Sensors Ready ✓';
    sensorHelp.textContent = 'High-frequency accelerometer enabled.';
    sensorBtn.classList.add('hidden');
    finishBtn.classList.remove('hidden');
    if (navigator.vibrate) {
      navigator.vibrate(50);
    }
  }

  function handleSensorDenied() {
    sensorText.textContent = 'Sensor Access Blocked';
    sensorHelp.innerHTML = `
      <strong>iPhone:</strong> Go to Settings &gt; Safari &gt; Motion &amp; Orientation Access.<br/>
      <strong>Android:</strong> Allow site sensor permission in Chrome settings.
    `;
    sensorBtn.textContent = 'Retry Sensor Access';
    finishBtn.classList.remove('hidden');
    finishBtn.textContent = 'Continue Anyway (Demo Mode) →';
  }

  finishBtn?.addEventListener('click', async () => {
    await setSetting('onboarding_completed', true);
    if (typeof onComplete === 'function') {
      onComplete();
    } else {
      navigateTo('auth');
    }
  });
}
