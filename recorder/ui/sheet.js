/**
 * ui/sheet.js - Accessible Bottom Sheet Modal
 */

let _activeSheet = null;

export function openSheet(contentHtml, title = '') {
  closeSheet();

  const backdrop = document.createElement('div');
  backdrop.className = 'sheet-backdrop';
  backdrop.id = 'active-sheet-backdrop';

  const container = document.createElement('div');
  container.className = 'sheet-container';
  container.id = 'active-sheet-container';
  container.setAttribute('role', 'dialog');
  container.setAttribute('aria-modal', 'true');
  if (title) container.setAttribute('aria-label', title);

  container.innerHTML = `
    <div class="sheet-handle"></div>
    ${title ? `<div class="card-header"><h3 class="card-title">${title}</h3><button class="header-action-btn" id="sheet-close-btn" aria-label="Close sheet">✕</button></div>` : ''}
    <div class="sheet-body">${contentHtml}</div>
  `;

  document.body.appendChild(backdrop);
  document.body.appendChild(container);

  // Trigger reflow for slide up animation
  void container.offsetWidth;
  backdrop.classList.add('active');
  container.classList.add('active');

  const closeBtn = container.querySelector('#sheet-close-btn');
  if (closeBtn) closeBtn.onclick = closeSheet;

  backdrop.onclick = closeSheet;

  const onKeyDown = (e) => {
    if (e.key === 'Escape') {
      closeSheet();
      window.removeEventListener('keydown', onKeyDown);
    }
  };
  window.addEventListener('keydown', onKeyDown);

  _activeSheet = { backdrop, container, onKeyDown };
  return container;
}

export function closeSheet() {
  if (!_activeSheet) return;
  const { backdrop, container, onKeyDown } = _activeSheet;
  window.removeEventListener('keydown', onKeyDown);

  backdrop.classList.remove('active');
  container.classList.remove('active');

  setTimeout(() => {
    if (backdrop.parentNode) backdrop.parentNode.removeChild(backdrop);
    if (container.parentNode) container.parentNode.removeChild(container);
  }, 260);

  _activeSheet = null;
}

