/**
 * ui/toast.js - Accessible Toast System with Action/Undo Support
 */

let toastTimeout = null;

export function showToast(message, type = 'normal', actionOrOptions = null, onActionFn = null) {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container';
    container.setAttribute('role', 'status');
    container.setAttribute('aria-live', 'polite');
    document.body.appendChild(container);
  }

  container.innerHTML = '';
  if (toastTimeout) clearTimeout(toastTimeout);

  let actionText = null;
  let onAction = null;

  if (typeof actionOrOptions === 'string') {
    actionText = actionOrOptions;
    onAction = onActionFn;
  } else if (actionOrOptions && typeof actionOrOptions === 'object') {
    actionText = actionOrOptions.actionLabel || actionOrOptions.text;
    onAction = actionOrOptions.onAction;
  }

  const toast = document.createElement('div');
  toast.className = 'toast-msg';

  const textSpan = document.createElement('span');
  textSpan.textContent = message;
  toast.appendChild(textSpan);

  if (actionText && typeof onAction === 'function') {
    const actionBtn = document.createElement('button');
    actionBtn.style.background = 'transparent';
    actionBtn.style.border = 'none';
    actionBtn.style.color = 'var(--accent-orange)';
    actionBtn.style.fontWeight = '700';
    actionBtn.style.cursor = 'pointer';
    actionBtn.style.padding = '2px 8px';
    actionBtn.style.fontSize = '13px';
    actionBtn.textContent = actionText;
    actionBtn.onclick = () => {
      onAction();
      container.innerHTML = '';
    };
    toast.appendChild(actionBtn);
  }

  container.appendChild(toast);

  toastTimeout = setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 200);
  }, 3500);
}
