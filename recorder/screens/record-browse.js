/**
 * screens/record-browse.js - Record Browse Screen (Phone 2 from Reference Screenshot)
 * Light page background (#F6F4F2), two-line headline with orange accent,
 * scrollable pill chips, 3 gradient cards with play buttons, setup card,
 * and high-z-index non-overlapping "Get Ready" placement modal.
 */

import { IMAGES } from '../config/images.js';
import { icons } from '../ui/icons.js';
import { getRecordings } from '../storage/db.js';
import { navigateTo } from '../ui/nav.js';
import { showToast } from '../ui/toast.js';

let _selectedContext = 'Resting';
let _selectedDuration = 60; // default 60s

export async function renderRecordBrowse(container, { user, onStartSession } = {}) {
  const subjectId = user ? user.subject_id : '1001';

  // Count existing recordings for setup progress
  let recCount = 0;
  try {
    const recs = await getRecordings(subjectId, 20);
    recCount = recs.length;
  } catch (_) {}

  const setupNeeded = recCount < 5;
  const progressPercent = Math.min(100, Math.round((recCount / 5) * 100));

  container.innerHTML = `
    <div style="padding: 24px 20px 104px 20px; display:flex; flex-direction:column; gap:20px; min-height:100%; background:var(--light-page);">
      
      <!-- Top Bar: 44px Circular Back & Filter/Sliders Buttons -->
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <button class="glass-circle-btn light-mode" id="browse-back-btn" aria-label="Go back to Home">
          ${icons.chevronLeft(20)}
        </button>
        <button class="glass-circle-btn light-mode" id="browse-filter-btn" aria-label="Filters and Duration">
          ${icons.sliders(20)}
        </button>
      </div>

      <!-- Heading (Phone 2 Match) -->
      <div>
        <h2 style="font-size:32px; font-weight:700; line-height:1.15; letter-spacing:-0.02em; color:var(--ink);">
          What's your<br/>
          <span style="color:var(--accent-orange);">heart need</span> today?
        </h2>
      </div>

      <!-- Horizontal Pill Chips Row (Light Mode) -->
      <div class="chip-row chip-row-light" id="browse-filter-chips">
        <button class="filter-chip active" data-filter="all">All</button>
        <button class="filter-chip" data-filter="first">First time</button>
        <button class="filter-chip" data-filter="resting">Resting</button>
        <button class="filter-chip" data-filter="activity">After activity</button>
        <button class="filter-chip" data-filter="calm">Calm down</button>
      </div>

      <!-- Card Stack (3 Stacked 134px Cards + Setup Card) -->
      <div id="browse-cards-container" style="display:flex; flex-direction:column; gap:16px;">
        
        <!-- Setup Card (if < 5 recordings) -->
        ${setupNeeded ? `
        <div class="setup-card" id="card-setup-progress" data-category="first" style="cursor:pointer;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
              <div style="font-size:16px; font-weight:700; color:var(--ink);">Set up your personal model</div>
              <div style="font-size:13px; color:var(--muted); margin-top:2px;">${recCount} of 5 resting recordings completed</div>
            </div>
            <div style="font-size:14px; font-weight:700; color:var(--accent-orange);">${progressPercent}%</div>
          </div>
          <div class="progress-bar-track">
            <div class="progress-bar-fill" style="width:${progressPercent}%;"></div>
          </div>
        </div>
        ` : ''}

        <!-- Card 1: Morning Check (Orange Gradient) -->
        <div class="browse-card" id="card-morning-check" data-category="resting" style="background:${IMAGES.cards.morning.bg};">
          <div class="browse-card-info">
            <div class="browse-card-title">Morning Check</div>
            <div class="browse-card-sub">1 min • Resting baseline</div>
          </div>
          <button class="browse-card-play-btn" aria-label="Start Morning Check">
            ${icons.play(16)}
          </button>
        </div>

        <!-- Card 2: After Workout (Peach-Tan Gradient) -->
        <div class="browse-card" id="card-workout-check" data-category="activity" style="background:${IMAGES.cards.workout.bg};">
          <div class="browse-card-info">
            <div class="browse-card-title">After Workout</div>
            <div class="browse-card-sub">1 min • After activity</div>
          </div>
          <button class="browse-card-play-btn" aria-label="Start After Workout Check">
            ${icons.play(16)}
          </button>
        </div>

        <!-- Card 3: Calm Down (Blue-Grey Gradient) -->
        <div class="browse-card" id="card-calm-check" data-category="calm" style="background:${IMAGES.cards.calm.bg};">
          <div class="browse-card-info">
            <div class="browse-card-title">Calm Down</div>
            <div class="browse-card-sub">2 min • Guided breathing</div>
          </div>
          <button class="browse-card-play-btn" aria-label="Start Calm Down Session">
            ${icons.play(16)}
          </button>
        </div>

      </div>

    </div>
  `;

  // Back button
  document.getElementById('browse-back-btn')?.addEventListener('click', () => {
    navigateTo('home');
  });

  // Filter Chips Interactive Toggle with smooth animations
  const filterButtons = container.querySelectorAll('#browse-filter-chips .filter-chip');
  const allCards = container.querySelectorAll('#browse-cards-container > div');

  filterButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      filterButtons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');

      const filter = btn.dataset.filter;
      allCards.forEach((card) => {
        const cat = card.dataset.category;
        if (filter === 'all' || cat === filter) {
          card.style.display = 'flex';
          card.style.opacity = '1';
          card.style.transform = 'translateY(0)';
        } else {
          card.style.display = 'none';
        }
      });
    });
  });

  // Reusable Get Ready Sheet (Rendered via Global High-Z-Index Sheet, Zero Overlap!)
  function openGetReadyModal(title, defaultCtx) {
    _selectedContext = defaultCtx;

    const sheetHtml = `
      <div style="display:flex; flex-direction:column; gap:16px;">
        <div style="text-align:center;">
          <h3 style="font-size:22px; font-weight:700; color:var(--ink);">${title}</h3>
          <p style="font-size:13.5px; color:var(--muted); margin-top:4px;">Lie down flat or sit still. Rest phone on your lower breastbone.</p>
        </div>

        <!-- Sternum Placement Guide -->
        <div style="width:100%; height:130px; border-radius:20px; background:var(--bg-sunken); border:1px solid var(--border-subtle); display:flex; align-items:center; justify-content:center; overflow:hidden;">
          <img src="${IMAGES.sternumGuide}" alt="Placement on sternum" style="width:100%; height:100%; object-fit:contain;" />
        </div>

        <!-- Activity Context Selector -->
        <div>
          <label style="font-size:13px; font-weight:700; color:var(--ink); display:block; margin-bottom:8px;">Activity Context</label>
          <div class="chip-row chip-row-light" id="ready-context-chips">
            <button type="button" class="filter-chip ${_selectedContext === 'Resting' ? 'active' : ''}" data-val="Resting">Resting</button>
            <button type="button" class="filter-chip ${_selectedContext === 'After activity' ? 'active' : ''}" data-val="After activity">After activity</button>
            <button type="button" class="filter-chip ${_selectedContext === 'Other' ? 'active' : ''}" data-val="Other">Other</button>
          </div>
        </div>

        <!-- Recording Duration Selector -->
        <div>
          <label style="font-size:13px; font-weight:700; color:var(--ink); display:block; margin-bottom:8px;">Recording Duration</label>
          <div style="display:flex; gap:10px;">
            <button type="button" class="duration-opt-btn ${_selectedDuration === 60 ? 'active' : ''}" id="btn-dur-60" data-sec="60" style="flex:1; padding:12px; border-radius:var(--r-inner); border:${_selectedDuration === 60 ? '1.5px solid var(--accent-orange)' : '1px solid var(--border-strong)'}; background:${_selectedDuration === 60 ? 'rgba(255,90,31,0.08)' : 'var(--bg-sunken)'}; color:${_selectedDuration === 60 ? 'var(--accent-orange)' : 'var(--muted)'}; font-weight:700; font-size:13.5px; cursor:pointer;">
              60s (Recommended)
            </button>
            <button type="button" class="duration-opt-btn ${_selectedDuration === 30 ? 'active' : ''}" id="btn-dur-30" data-sec="30" style="flex:1; padding:12px; border-radius:var(--r-inner); border:${_selectedDuration === 30 ? '1.5px solid var(--accent-orange)' : '1px solid var(--border-strong)'}; background:${_selectedDuration === 30 ? 'rgba(255,90,31,0.08)' : 'var(--bg-sunken)'}; color:${_selectedDuration === 30 ? 'var(--accent-orange)' : 'var(--muted)'}; font-weight:700; font-size:13.5px; cursor:pointer;">
              30s (Quick check)
            </button>
          </div>
        </div>

        <!-- Primary Start Button -->
        <button class="btn-orange-pill" id="ready-start-action-btn" style="width:100%; margin-top:6px; font-size:16px; justify-content:center; padding:14px;">
          ${icons.play(18)}
          <span>Start Recording</span>
        </button>

        <!-- Cancel Link -->
        <button type="button" id="ready-cancel-action-btn" style="background:transparent; border:none; color:var(--muted); font-size:14px; font-weight:600; cursor:pointer; padding:8px;" onclick="window.closeBottomSheet && window.closeBottomSheet()">
          Cancel
        </button>
      </div>
    `;

    if (typeof window.openBottomSheet === 'function') {
      window.openBottomSheet(sheetHtml, (content) => {
        // Context chips
        content.querySelectorAll('#ready-context-chips .filter-chip').forEach((btn) => {
          btn.onclick = () => {
            content.querySelectorAll('#ready-context-chips .filter-chip').forEach((b) => b.classList.remove('active'));
            btn.classList.add('active');
            _selectedContext = btn.dataset.val;
          };
        });

        // Duration buttons
        const b60 = content.querySelector('#btn-dur-60');
        const b30 = content.querySelector('#btn-dur-30');
        if (b60 && b30) {
          b60.onclick = () => {
            _selectedDuration = 60;
            b60.style.border = '1.5px solid var(--accent-orange)';
            b60.style.background = 'rgba(255,90,31,0.08)';
            b60.style.color = 'var(--accent-orange)';
            b30.style.border = '1px solid var(--border-strong)';
            b30.style.background = 'var(--bg-sunken)';
            b30.style.color = 'var(--muted)';
          };
          b30.onclick = () => {
            _selectedDuration = 30;
            b30.style.border = '1.5px solid var(--accent-orange)';
            b30.style.background = 'rgba(255,90,31,0.08)';
            b30.style.color = 'var(--accent-orange)';
            b60.style.border = '1px solid var(--border-strong)';
            b60.style.background = 'var(--bg-sunken)';
            b60.style.color = 'var(--muted)';
          };
        }

        // Launch Recording
        content.querySelector('#ready-start-action-btn')?.addEventListener('click', async () => {
          // Request iOS Motion Sensor Permission on user interaction
          if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
            try {
              const perm = await DeviceMotionEvent.requestPermission();
              console.log('[iOS Motion Permission]:', perm);
            } catch (err) {
              console.warn('[iOS Motion Permission error]:', err);
            }
          }

          window.closeBottomSheet();
          navigateTo('record-player');
          const playerContainer = document.getElementById('screen-record-player');
          import('./record-player.js?v=12').then((mod) => {
            mod.renderRecordPlayer(playerContainer, {
              context: _selectedContext,
              duration: _selectedDuration,
              user
            });
          });
        });
      });
    }
  }

  // Card click triggers
  document.getElementById('card-morning-check')?.addEventListener('click', () => {
    openGetReadyModal('Morning Heart Check', 'Resting');
  });

  document.getElementById('card-workout-check')?.addEventListener('click', () => {
    openGetReadyModal('After Workout Check', 'After activity');
  });

  document.getElementById('card-setup-progress')?.addEventListener('click', () => {
    openGetReadyModal('Personal Calibration Recording', 'Resting');
  });

  document.getElementById('card-calm-check')?.addEventListener('click', () => {
    navigateTo('calm');
  });

  // Top Sliders button opens duration/context options
  document.getElementById('browse-filter-btn')?.addEventListener('click', () => {
    openGetReadyModal('Select Duration & Context', _selectedContext);
  });
}
