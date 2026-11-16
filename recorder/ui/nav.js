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

  // Tab buttons are wired via the single window.navGo entry point (see index.html),
  // which delegates to navigateTo(). No second listener here to avoid double-loading.

  // Android/browser hardware back button -> in-app back
  window.addEventListener('popstate', () => {
    goBack();
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
  isTab = isTab || TAB_SCREENS.has(screenId);

  // Always dismiss transient overlays when changing screens
  if (typeof window !== 'undefined') {
    window.closeBottomSheet && document.getElementById('global-bottom-sheet')?.classList.contains('active') && window.closeBottomSheet();
    document.getElementById('home-slide-menu')?.classList.remove('active');
    document.getElementById('home-drawer-backdrop')?.classList.remove('active');
    window._currentScreenId = screenId;
  }

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

  // Give the browser/Android back button something to pop for in-app screens
  if (!isTab && typeof history !== 'undefined' && history.pushState) {
    try { history.pushState({ screen: screenId }, ''); } catch (_) {}
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
