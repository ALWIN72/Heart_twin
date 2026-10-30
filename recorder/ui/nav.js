/**
 * ui/nav.js - Navigation Controller & 5-Tab Bar Manager
 * Warm orange glass meditation app navigation
 */

let _currentScreen = 'home';
let _historyStack = [];
let _globalOnScreenChange = null;

// 5 Main Tabs
export const TAB_SCREENS = new Set(['home', 'record-browse', 'calm', 'trends', 'profile']);

// Screens that use Glass Navbar (dark background) vs Light Navbar (light background)
export const GLASS_SCREENS = new Set(['home']);

export function setOnScreenChange(fn) {
  _globalOnScreenChange = fn;
}

export function initNavigation({ onScreenChange } = {}) {
  if (typeof onScreenChange === 'function') {
    _globalOnScreenChange = onScreenChange;
  }

  const navButtons = document.querySelectorAll('.nav-item-btn');

  navButtons.forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const target = btn.dataset.target;
      if (target) {
        navigateTo(target, { isTab: true });
      }
    });
  });

  // Global back button listener
  document.querySelectorAll('[data-action="go-back"]').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      goBack();
    });
  });
}

export function getCurrentScreen() {
  return _currentScreen;
}

export function navigateTo(screenId, { isTab = false, clearHistory = false, onScreenChange } = {}) {
  const prevScreen = _currentScreen;

  if (clearHistory) {
    _historyStack = [];
  } else if (!isTab && prevScreen && prevScreen !== screenId) {
    _historyStack.push(prevScreen);
  }

  _currentScreen = screenId;

  // Toggle active class on all screen containers
  const allScreens = document.querySelectorAll('.app-screen');
  allScreens.forEach((el) => {
    el.classList.remove('active');
  });

  const targetEl = document.getElementById(`screen-${screenId}`);
  if (targetEl) {
    targetEl.classList.add('active');
    targetEl.scrollTop = 0;
  }

  // Update Bottom Nav state & appearance
  const bottomNav = document.getElementById('floating-bottom-nav');
  const navButtons = document.querySelectorAll('.nav-item-btn');

  if (bottomNav) {
    if (TAB_SCREENS.has(screenId)) {
      bottomNav.classList.remove('nav-hidden');

      // Toggle Glass vs Light theme on the floating nav bar
      if (GLASS_SCREENS.has(screenId)) {
        bottomNav.classList.add('nav-glass');
        bottomNav.classList.remove('nav-light');
      } else {
        bottomNav.classList.add('nav-light');
        bottomNav.classList.remove('nav-glass');
      }

      // Highlight active tab
      navButtons.forEach((btn) => {
        if (btn.dataset.target === screenId) {
          btn.classList.add('active');
          btn.setAttribute('aria-selected', 'true');
        } else {
          btn.classList.remove('active');
          btn.setAttribute('aria-selected', 'false');
        }
      });
    } else {
      // Hide bottom nav during player, onboarding, auth, or full-bleed share
      bottomNav.classList.add('nav-hidden');
    }
  }

  const cb = onScreenChange || _globalOnScreenChange;
  if (typeof cb === 'function') {
    cb(screenId);
  }
}

export function goBack({ onScreenChange } = {}) {
  const cb = onScreenChange || _globalOnScreenChange;
  if (_historyStack.length > 0) {
    const prev = _historyStack.pop();
    navigateTo(prev, { isTab: TAB_SCREENS.has(prev), onScreenChange: cb });
  } else {
    navigateTo('home', { isTab: true, onScreenChange: cb });
  }
}
