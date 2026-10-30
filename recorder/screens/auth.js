/**
 * screens/auth.js - Patient Sign In & Registration Screen
 * Warm orange glass aesthetic with automatic non-colliding Subject ID allocation.
 */

import { loginUser, registerUser } from '../api/client.js';
import { setSetting } from '../storage/db.js';
import { navigateTo } from '../ui/nav.js';
import { showToast } from '../ui/toast.js';
import { icons } from '../ui/icons.js';

let _activeMode = 'login'; // 'login' or 'register'

export function renderAuth(container, { initialMode = 'login', onSuccess } = {}) {
  _activeMode = initialMode;

  container.innerHTML = `
    <div style="width:100%; min-height:100%; padding: 28px 24px 34px 24px; display:flex; flex-direction:column; justify-content:center; align-items:center; background:var(--light-page);">
      <!-- Brand Logo & Header -->
      <div style="text-align:center; margin-bottom:24px;">
        <div style="width:56px; height:56px; border-radius:20px; background:var(--accent-orange); color:#FFFFFF; display:inline-flex; align-items:center; justify-content:center; box-shadow:0 8px 24px rgba(255,90,31,0.35); margin-bottom:12px;">
          ${icons.heart(28)}
        </div>
        <h1 style="font-size:26px; font-weight:700; color:var(--ink); letter-spacing:-0.02em;" id="auth-main-title">
          ${_activeMode === 'login' ? 'Welcome Back' : 'Create Patient Profile'}
        </h1>
        <p style="font-size:14px; color:var(--muted); margin-top:4px;" id="auth-main-sub">
          ${_activeMode === 'login' ? 'Sign in to access your heart telemetry & digital twin' : 'No Subject ID needed — one is automatically assigned'}
        </p>
      </div>

      <!-- Segmented Toggle (Sign In vs Create Account) -->
      <div style="display:flex; background:#E5DFDA; border-radius:var(--r-pill); padding:4px; width:100%; max-width:340px; margin-bottom:20px;">
        <button type="button" class="auth-toggle-btn ${_activeMode === 'login' ? 'active' : ''}" id="tab-auth-login" style="flex:1; border:none; padding:10px 16px; border-radius:var(--r-pill); font-size:14px; font-weight:600; cursor:pointer; transition:all var(--t-dur) var(--t-ease); background:${_activeMode === 'login' ? '#FFFFFF' : 'transparent'}; color:${_activeMode === 'login' ? 'var(--ink)' : 'var(--muted)'}; box-shadow:${_activeMode === 'login' ? '0 2px 8px rgba(0,0,0,0.08)' : 'none'};">
          Sign In
        </button>
        <button type="button" class="auth-toggle-btn ${_activeMode === 'register' ? 'active' : ''}" id="tab-auth-register" style="flex:1; border:none; padding:10px 16px; border-radius:var(--r-pill); font-size:14px; font-weight:600; cursor:pointer; transition:all var(--t-dur) var(--t-ease); background:${_activeMode === 'register' ? '#FFFFFF' : 'transparent'}; color:${_activeMode === 'register' ? 'var(--ink)' : 'var(--muted)'}; box-shadow:${_activeMode === 'register' ? '0 2px 8px rgba(0,0,0,0.08)' : 'none'};">
          New Patient
        </button>
      </div>

      <!-- Form Card -->
      <div style="width:100%; max-width:340px; background:#FFFFFF; border-radius:28px; padding:24px; box-shadow:var(--shadow-card); border:1px solid var(--border-subtle);">
        
        <!-- 1. SIGN IN FORM -->
        <form id="auth-login-form" style="display:${_activeMode === 'login' ? 'flex' : 'none'}; flex-direction:column; gap:14px;">
          <div>
            <label style="font-size:13px; font-weight:600; color:var(--ink); display:block; margin-bottom:6px;">Username or Email</label>
            <input type="text" id="login-ident" required autocomplete="username" placeholder="e.g. sarah or sarah@email.com" 
                   style="width:100%; height:48px; padding:0 14px; border-radius:14px; border:1px solid var(--border-strong); background:var(--bg-sunken); font-size:15px; color:var(--ink);" />
          </div>

          <div>
            <label style="font-size:13px; font-weight:600; color:var(--ink); display:block; margin-bottom:6px;">Password</label>
            <input type="password" id="login-password" required autocomplete="current-password" placeholder="Enter password" 
                   style="width:100%; height:48px; padding:0 14px; border-radius:14px; border:1px solid var(--border-strong); background:var(--bg-sunken); font-size:15px; color:var(--ink);" />
          </div>

          <div id="login-error-msg" style="color:var(--status-diff-text); font-size:13px; min-height:18px;"></div>

          <button type="submit" class="btn-orange-pill" id="login-submit-btn" style="width:100%; margin-top:4px;">
            Sign In →
          </button>
        </form>

        <!-- 2. REGISTER FORM -->
        <form id="auth-register-form" style="display:${_activeMode === 'register' ? 'flex' : 'none'}; flex-direction:column; gap:14px;">
          <div>
            <label style="font-size:13px; font-weight:600; color:var(--ink); display:block; margin-bottom:6px;">Your Name</label>
            <input type="text" id="reg-name" required placeholder="e.g. Amelia Vance" 
                   style="width:100%; height:48px; padding:0 14px; border-radius:14px; border:1px solid var(--border-strong); background:var(--bg-sunken); font-size:15px; color:var(--ink);" />
          </div>

          <div>
            <label style="font-size:13px; font-weight:600; color:var(--ink); display:block; margin-bottom:6px;">Username</label>
            <input type="text" id="reg-username" required placeholder="e.g. amelia" 
                   style="width:100%; height:48px; padding:0 14px; border-radius:14px; border:1px solid var(--border-strong); background:var(--bg-sunken); font-size:15px; color:var(--ink);" />
          </div>

          <div>
            <label style="font-size:13px; font-weight:600; color:var(--ink); display:block; margin-bottom:6px;">Password</label>
            <input type="password" id="reg-password" required minlength="4" placeholder="Min 4 characters" 
                   style="width:100%; height:48px; padding:0 14px; border-radius:14px; border:1px solid var(--border-strong); background:var(--bg-sunken); font-size:15px; color:var(--ink);" />
          </div>

          <div id="register-error-msg" style="color:var(--status-diff-text); font-size:13px; min-height:18px;"></div>

          <button type="submit" class="btn-orange-pill" id="register-submit-btn" style="width:100%; margin-top:4px;">
            Create Profile &amp; Start →
          </button>
        </form>

      </div>
    </div>
  `;

  const tabLogin = document.getElementById('tab-auth-login');
  const tabRegister = document.getElementById('tab-auth-register');
  const formLogin = document.getElementById('auth-login-form');
  const formRegister = document.getElementById('auth-register-form');
  const titleEl = document.getElementById('auth-main-title');
  const subEl = document.getElementById('auth-main-sub');

  function switchMode(mode) {
    _activeMode = mode;
    if (mode === 'login') {
      tabLogin.style.background = '#FFFFFF';
      tabLogin.style.color = 'var(--ink)';
      tabLogin.style.boxShadow = '0 2px 8px rgba(0,0,0,0.08)';
      tabRegister.style.background = 'transparent';
      tabRegister.style.color = 'var(--muted)';
      tabRegister.style.boxShadow = 'none';

      formLogin.style.display = 'flex';
      formRegister.style.display = 'none';
      titleEl.textContent = 'Welcome Back';
      subEl.textContent = 'Sign in to access your heart telemetry & digital twin';
    } else {
      tabRegister.style.background = '#FFFFFF';
      tabRegister.style.color = 'var(--ink)';
      tabRegister.style.boxShadow = '0 2px 8px rgba(0,0,0,0.08)';
      tabLogin.style.background = 'transparent';
      tabLogin.style.color = 'var(--muted)';
      tabLogin.style.boxShadow = 'none';

      formRegister.style.display = 'flex';
      formLogin.style.display = 'none';
      titleEl.textContent = 'Create Patient Profile';
      subEl.textContent = 'No Subject ID needed — one is automatically assigned';
    }
  }

  tabLogin.onclick = () => switchMode('login');
  tabRegister.onclick = () => switchMode('register');

  // Submit Sign In
  formLogin.onsubmit = async (e) => {
    e.preventDefault();
    const ident = document.getElementById('login-ident').value.trim();
    const pass = document.getElementById('login-password').value;
    const btn = document.getElementById('login-submit-btn');
    const errEl = document.getElementById('login-error-msg');

    btn.textContent = 'Signing in...';
    btn.disabled = true;
    errEl.textContent = '';

    try {
      const res = await loginUser(ident, pass);
      if (res && res.ok && res.user) {
        await setSetting('current_user', res.user);
        showToast(`Welcome back, ${res.user.name || res.user.username}!`, 'normal');
        if (typeof onSuccess === 'function') onSuccess(res.user);
      } else {
        errEl.textContent = res.error || 'Invalid username or password.';
      }
    } catch (err) {
      errEl.textContent = 'Network or server connection error.';
    } finally {
      btn.textContent = 'Sign In →';
      btn.disabled = false;
    }
  };

  // Submit Registration
  formRegister.onsubmit = async (e) => {
    e.preventDefault();
    const name = document.getElementById('reg-name').value.trim();
    const username = document.getElementById('reg-username').value.trim();
    const pass = document.getElementById('reg-password').value;
    const btn = document.getElementById('register-submit-btn');
    const errEl = document.getElementById('register-error-msg');

    btn.textContent = 'Creating account...';
    btn.disabled = true;
    errEl.textContent = '';

    try {
      const res = await registerUser({ username, name, password: pass });
      if (res && res.ok && res.user) {
        await setSetting('current_user', res.user);
        showToast(`Account created! Subject ID: ${res.user.subject_id}`, 'normal');
        if (typeof onSuccess === 'function') onSuccess(res.user);
      } else {
        errEl.textContent = res.error || 'Username already exists.';
      }
    } catch (err) {
      errEl.textContent = 'Error connecting to registration server.';
    } finally {
      btn.textContent = 'Create Profile & Start →';
      btn.disabled = false;
    }
  };
}
