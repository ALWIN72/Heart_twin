/**
 * screens/home.js - Dynamic Home Screen (Phone 1 from Reference Screenshot)
 * Full-bleed rust canyon texture, glass pill status, dynamic time-of-day greeting,
 * ambient live cardiac pulse strip, white pill play button, and filter chips.
 */

import { IMAGES } from '../config/images.js';
import { icons } from '../ui/icons.js';
import { getRecordings } from '../storage/db.js';
import { navigateTo } from '../ui/nav.js';
import { showToast } from '../ui/toast.js';

let _ambientRaf = null;

export async function renderHome(container, { user, onStartRecording, onOpenMenu } = {}) {
  if (_ambientRaf) {
    cancelAnimationFrame(_ambientRaf);
    _ambientRaf = null;
  }

  const subjectId = user ? user.subject_id : '1001';
  const rawName = user ? (user.name || user.username) : 'Amelia';
  const firstName = rawName.split(' ')[0];

  // Dynamic Time-of-Day Context
  const hour = new Date().getHours();
  let timeGreeting = 'Good to see you again';
  let heroTitleText = 'Morning<br/>Heart Check';
  let heroDesc = 'Lie still for 60 seconds and see how your heart compares with your usual.';
  if (hour >= 12 && hour < 17) {
    timeGreeting = 'Good afternoon';
    heroTitleText = 'Midday<br/>Heart Check';
    heroDesc = 'Take a quiet minute to check your cardiac stability and autonomic tone.';
  } else if (hour >= 17 && hour < 22) {
    timeGreeting = 'Good evening';
    heroTitleText = 'Evening<br/>Heart Check';
    heroDesc = 'Wind down and see how today\'s activity influenced your heart rhythm.';
  } else if (hour >= 22 || hour < 5) {
    timeGreeting = 'Restful night';
    heroTitleText = 'Night<br/>Resting Check';
    heroDesc = 'Rest flat and assess your overnight cardiac recovery baseline.';
  }

  // Fetch recent recordings from IndexedDB
  let latestRec = null;
  let totalRecCount = 0;
  try {
    const recs = await getRecordings(subjectId, 10);
    totalRecCount = recs.length;
    if (recs.length > 0) {
      latestRec = recs[0];
    }
  } catch (e) {
    console.warn('Could not read recordings:', e);
  }

  // Determine Status Pill
  let statusText = 'Normal for you';
  let statusDotClass = 'normal';
  let statusExplanation = 'Your resting cardiac rhythm and heart rate variability closely match your baseline model.';

  if (totalRecCount < 5) {
    statusText = `Setup: ${totalRecCount} of 5 recordings`;
    statusDotClass = 'slight';
    statusExplanation = `You have completed ${totalRecCount} of 5 calibration recordings. Complete 5 resting recordings to train your custom Personal Heart Twin!`;
  } else if (latestRec && latestRec.verdict) {
    if (latestRec.verdict.includes('Different')) {
      statusText = 'Different today';
      statusDotClass = 'diff';
      statusExplanation = 'Your recent cardiac rhythm shows noticeable variation from your baseline. Consider resting and checking in later.';
    } else if (latestRec.verdict.includes('Slightly')) {
      statusText = 'Slightly different';
      statusDotClass = 'slight';
      statusExplanation = 'Mild autonomic variation detected compared to your typical baseline. Within expected daily fluctuations.';
    } else {
      statusText = 'Normal for you';
      statusDotClass = 'normal';
      statusExplanation = 'Your cardiac motion and HRV are steady and consistent with your personal digital twin profile.';
    }
  }

  // Last check sub-line
  let lastCheckText = 'Last check: 72 bpm, regular';
  let liveHeartRateDisplay = 72;
  if (latestRec && latestRec.hr) {
    const bpm = Math.round(latestRec.hr);
    liveHeartRateDisplay = bpm;
    const reg = (latestRec.sdnn && latestRec.sdnn > 25) ? 'regular' : 'mild variance';
    lastCheckText = `Last check: ${bpm} bpm, ${reg}`;
  }

  container.innerHTML = `
    <div class="home-screen-wrap" style="width:100%; height:100%; position:relative; display:flex; flex-direction:column; justify-content:space-between; padding: 24px 24px 104px 24px; background: #4A150A linear-gradient(180deg, #4A150A 0%, #7A2412 35%, #C2410C 65%, #FF5A1F 100%) center/cover no-repeat; overflow:hidden;">
      
      <!-- Top Bar: Logo & 44px Glass Hamburger Circle -->
      <div style="display:flex; justify-content:space-between; align-items:center; z-index:10;">
        <!-- Top Left Heart Pulse Logo with rhythmic cardiac animation -->
        <div style="display:flex; align-items:center; gap:8px; color:#FFFFFF;">
          <div style="width:38px; height:38px; border-radius:12px; background:rgba(255,255,255,0.18); backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px); border:1px solid rgba(255,255,255,0.25); display:flex; align-items:center; justify-content:center;">
            <div class="heart-pulse" style="color:#FFFFFF;">
              ${icons.heart(20)}
            </div>
          </div>
        </div>

        <!-- Top Right Glass Hamburger Button -->
        <button class="glass-circle-btn" id="home-menu-btn" aria-label="Open menu and settings">
          ${icons.menu(20)}
        </button>
      </div>

      <!-- Center & Hero Content -->
      <div style="display:flex; flex-direction:column; gap:14px; margin-top:14px; z-index:10;">
        
        <!-- Greeting: "Hi, Amelia!" -->
        <div>
          <div style="font-size:28px; line-height:1.2; color:#FFFFFF;">
            <span style="font-weight:700;">Hi, </span><span style="font-weight:300;">${firstName}!</span>
          </div>
          <div style="font-size:14px; color:rgba(255,255,255,0.85); font-weight:400; margin-top:3px;">
            ${timeGreeting}
          </div>
        </div>

        <!-- Interactive Glass Status Pill (Tapping opens explanation via global bottom sheet) -->
        <div class="glass-status-pill" id="home-status-pill" title="Tap to learn more about your status">
          <span class="status-dot ${statusDotClass}"></span>
          <span>${statusText}</span>
          <span style="font-size:11px; opacity:0.7; margin-left:2px;">ⓘ</span>
        </div>

        <!-- Hero Title (48px, 2 Lines) -->
        <h1 class="hero-title" style="margin-top:2px;">
          ${heroTitleText}
        </h1>

        <!-- Hero Subtitle -->
        <p style="font-size:15px; color:rgba(255,255,255,0.88); line-height:1.42; max-width:310px;">
          ${heroDesc}
        </p>

        <!-- Dynamic Ambient Cardiac Pulse Strip -->
        <div style="height:32px; width:100%; max-width:280px; position:relative; opacity:0.75; margin-top:2px;">
          <canvas id="home-ambient-pulse-canvas" style="width:100%; height:100%;"></canvas>
        </div>

        <!-- White Pill Button: "Start Recording" -->
        <div style="margin-top:6px;">
          <button class="btn-white-pill" id="home-start-record-btn">
            ${icons.play(16)}
            <span>Start Recording</span>
          </button>

          <!-- Small Glass Sub-line -->
          <div style="font-size:12px; color:rgba(255,255,255,0.7); margin-top:10px; margin-left:8px; font-weight:500;">
            ${lastCheckText}
          </div>
        </div>

      </div>

      <!-- Horizontal Filter Chips Row (All, Check-in, Calm, Rest, Activity) -->
      <div style="z-index:10; margin-top:12px;">
        <div class="chip-row">
          <button class="filter-chip active" data-context="All">All</button>
          <button class="filter-chip" data-context="Resting">Check-in</button>
          <button class="filter-chip" data-context="Calm">Calm</button>
          <button class="filter-chip" data-context="Resting">Rest</button>
          <button class="filter-chip" data-context="After activity">Activity</button>
        </div>
      </div>

    </div>

    <!-- Slide-over Drawer Menu for Hamburger Button -->
    <div class="drawer-backdrop" id="home-drawer-backdrop"></div>
    <div class="slide-menu" id="home-slide-menu">
      <div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:28px;">
          <div style="font-size:18px; font-weight:700;">Settings &amp; Menu</div>
          <button class="glass-circle-btn" id="drawer-close-btn" style="width:36px; height:36px;">
            ${icons.close(16)}
          </button>
        </div>

        <div style="background:rgba(255,255,255,0.12); border-radius:18px; padding:14px; margin-bottom:20px;">
          <div style="font-size:12px; color:rgba(255,255,255,0.7);">Signed in as:</div>
          <div style="font-size:16px; font-weight:700; margin-top:2px;">${user ? (user.name || user.username) : 'Amelia Vance'}</div>
          <div style="font-size:12px; color:var(--accent-orange); margin-top:2px;">Subject ID: ${user ? user.subject_id : '1001'}</div>
        </div>

        <div style="display:flex; flex-direction:column; gap:12px;">
          <button class="drawer-link-btn" id="menu-help-btn" style="background:transparent; border:none; color:#FFFFFF; display:flex; align-items:center; gap:12px; padding:10px 0; font-size:15px; cursor:pointer; text-align:left;">
            ${icons.heart(18)}
            <span>Phone Placement Guide</span>
          </button>
          <button class="drawer-link-btn" id="menu-trends-btn" style="background:transparent; border:none; color:#FFFFFF; display:flex; align-items:center; gap:12px; padding:10px 0; font-size:15px; cursor:pointer; text-align:left;">
            ${icons.trends(18)}
            <span>View 30-Day Trends</span>
          </button>
          <button class="drawer-link-btn" id="menu-doctor-btn" style="background:transparent; border:none; color:#FFFFFF; display:flex; align-items:center; gap:12px; padding:10px 0; font-size:15px; cursor:pointer; text-align:left;">
            ${icons.activity(18)}
            <span>Share Doctor Summary</span>
          </button>
        </div>
      </div>

      <div>
        <button class="btn-orange-pill" id="menu-logout-btn" style="width:100%; justify-content:center; padding:12px;">
          Sign Out
        </button>
      </div>
    </div>
  `;

  // Apply rich canyon vector background
  const wrap = container.querySelector('.home-screen-wrap');
  if (wrap && IMAGES.homeCanyon) {
    try {
      wrap.style.backgroundImage = `url("${IMAGES.homeCanyon}")`;
    } catch (_) {}
  }

  // Draw Dynamic Ambient Pulse Line Canvas
  const pulseCanvas = document.getElementById('home-ambient-pulse-canvas');
  if (pulseCanvas) {
    const dpr = window.devicePixelRatio || 1;
    pulseCanvas.width = 280 * dpr;
    pulseCanvas.height = 32 * dpr;
    const ctx = pulseCanvas.getContext('2d');
    ctx.scale(dpr, dpr);

    let t = 0;
    function animatePulse() {
      t += 0.035;
      ctx.clearRect(0, 0, 280, 32);
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.65)';
      ctx.lineWidth = 1.6;
      ctx.lineCap = 'round';

      for (let x = 0; x < 280; x += 2) {
        const phase = (x / 70 - t) % 1;
        let y = 16;
        if (phase > 0 && phase < 0.25) {
          y = 16 - Math.sin((phase / 0.25) * Math.PI) * 12;
        } else if (phase >= 0.25 && phase < 0.45) {
          y = 16 + Math.sin(((phase - 0.25) / 0.2) * Math.PI) * 7;
        }
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      _ambientRaf = requestAnimationFrame(animatePulse);
    }
    _ambientRaf = requestAnimationFrame(animatePulse);
  }

  // Attach event handlers
  document.getElementById('home-start-record-btn')?.addEventListener('click', () => {
    navigateTo('record-browse');
  });

  // Status pill click -> uses Global High-Z-Index Bottom Sheet (zero overlap!)
  const statusPill = document.getElementById('home-status-pill');
  statusPill?.addEventListener('click', () => {
    if (typeof window.openBottomSheet === 'function') {
      window.openBottomSheet(`
        <div style="text-align:center; padding: 4px 6px 12px 6px;">
          <div style="display:inline-flex; padding:8px 16px; border-radius:var(--r-pill); background:var(--bg-sunken); margin-bottom:12px; font-size:13px; font-weight:700; color:var(--ink);">
            <span class="status-dot ${statusDotClass}" style="margin-right:6px; margin-top:4px;"></span>
            ${statusText}
          </div>
          <h3 style="font-size:21px; font-weight:700; color:var(--ink); margin-bottom:8px;">Your Cardiac Baseline</h3>
          <p style="font-size:14px; color:var(--text-main); line-height:1.5;">${statusExplanation}</p>
          <div style="margin-top:18px; padding:14px; border-radius:18px; background:var(--bg-sunken); text-align:left; font-size:12.5px; color:var(--muted); line-height:1.45;">
            <strong>How it works:</strong> The Digital Heart Twin measures subtle micro-vibrations of your chest wall (seismocardiography). Each recording is calibrated against your own baseline model rather than generic population averages.
          </div>
          <button class="btn-orange-pill" style="width:100%; margin-top:20px; justify-content:center; padding:14px;" onclick="window.closeBottomSheet && window.closeBottomSheet()">
            Got it
          </button>
        </div>
      `);
    }
  });

  // Filter chips click
  container.querySelectorAll('.filter-chip').forEach((chip) => {
    chip.addEventListener('click', (e) => {
      container.querySelectorAll('.filter-chip').forEach((c) => c.classList.remove('active'));
      chip.classList.add('active');
      const ctx = chip.dataset.context;
      if (ctx === 'Calm') {
        navigateTo('calm');
      } else {
        navigateTo('record-browse');
      }
    });
  });

  // Drawer handlers
  const drawerBackdrop = document.getElementById('home-drawer-backdrop');
  const slideMenu = document.getElementById('home-slide-menu');
  const menuBtn = document.getElementById('home-menu-btn');
  const closeBtn = document.getElementById('drawer-close-btn');

  function openDrawer() {
    drawerBackdrop?.classList.add('active');
    slideMenu?.classList.add('active');
  }

  function closeDrawer() {
    drawerBackdrop?.classList.remove('active');
    slideMenu?.classList.remove('active');
  }

  menuBtn?.addEventListener('click', openDrawer);
  closeBtn?.addEventListener('click', closeDrawer);
  drawerBackdrop?.addEventListener('click', closeDrawer);

  document.getElementById('menu-help-btn')?.addEventListener('click', () => {
    closeDrawer();
    navigateTo('onboarding');
  });

  document.getElementById('menu-trends-btn')?.addEventListener('click', () => {
    closeDrawer();
    navigateTo('trends');
  });

  document.getElementById('menu-doctor-btn')?.addEventListener('click', () => {
    closeDrawer();
    navigateTo('share');
  });

  document.getElementById('menu-logout-btn')?.addEventListener('click', async () => {
    closeDrawer();
    const { setSetting } = await import('../storage/db.js');
    await setSetting('current_user', null);
    navigateTo('auth');
  });
}
