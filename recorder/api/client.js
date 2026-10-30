/**
 * api/client.js - API client for serve_recorder.py
 * Handles authentication, uploads, real-time ML scoring and twin status.
 */

let _authToken = localStorage.getItem('dht_auth_token') || '';

export function setAuthToken(token) {
  _authToken = token || '';
  if (token) {
    localStorage.setItem('dht_auth_token', token);
  } else {
    localStorage.removeItem('dht_auth_token');
  }
}

export function getAuthToken() {
  return _authToken;
}

function getHeaders() {
  const headers = { 'Content-Type': 'application/json' };
  if (_authToken) {
    headers['Authorization'] = `Bearer ${_authToken}`;
  }
  return headers;
}

export async function registerUser({ name, username, email, password, age, gender }) {
  const res = await fetch('/auth/register', {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify({ name, username, email, password, age, gender }),
  });
  const data = await res.json();
  if (data.ok && data.token) {
    setAuthToken(data.token);
  }
  return data;
}

export async function loginUser(arg1, arg2) {
  let identifier, password;
  if (typeof arg1 === 'object' && arg1 !== null) {
    identifier = arg1.identifier;
    password = arg1.password;
  } else {
    identifier = arg1;
    password = arg2;
  }
  const res = await fetch('/auth/login', {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify({ identifier, password }),
  });
  const data = await res.json();
  if (data.ok && data.token) {
    setAuthToken(data.token);
  }
  return data;
}

export async function getCurrentUser() {
  if (!_authToken) return null;
  try {
    const res = await fetch('/auth/me', { headers: getHeaders() });
    if (!res.ok) return null;
    const data = await res.json();
    return data.ok ? data.user : null;
  } catch (_) {
    return null;
  }
}

export async function logoutUser() {
  try {
    await fetch('/auth/logout', { method: 'POST', headers: getHeaders() });
  } catch (_) {}
  setAuthToken('');
}

export async function updateProfile(data) {
  const res = await fetch('/auth/update_profile', {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify(data),
  });
  return res.json();
}

export async function uploadRecording({ subject, recording, files, metadata }) {
  const res = await fetch('/upload', {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify({ subject, recording, files, metadata }),
  });
  return res.json();
}

export async function scoreRecording({ subject, recording }) {
  const res = await fetch('/twin_score', {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify({ subject, recording }),
  });
  return res.json();
}

export async function getTwinStatus(subject) {
  const res = await fetch(`/twin_status?subject=${encodeURIComponent(subject)}`, {
    headers: getHeaders(),
  });
  return res.json();
}

export async function trainTwin(subject) {
  const res = await fetch('/train_twin', {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify({ subject }),
  });
  return res.json();
}

export async function getDashboardData(subject) {
  const res = await fetch(`/dashboard_data?subject=${encodeURIComponent(subject)}`, {
    headers: getHeaders(),
  });
  return res.json();
}

export async function getHeartprintStatus(subject) {
  const res = await fetch(`/heartprint_status?subject=${encodeURIComponent(subject)}`, {
    headers: getHeaders(),
  });
  return res.json();
}

export async function enrollHeartprint({ subject, recording, detected_bpm }) {
  const res = await fetch('/heartprint_enroll', {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify({ subject, recording, detected_bpm }),
  });
  return res.json();
}

