/**
 * sw.js - Service Worker Cache Purge & Live Passthrough
 * Clears any stale caches and unregisters to guarantee live code delivery.
 */

self.addEventListener('install', (e) => {
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.map((k) => caches.delete(k))))
      .then(() => self.registration.unregister())
      .then(() => self.clients.matchAll())
      .then((clients) => {
        clients.forEach((client) => {
          if (client.url && 'navigate' in client) {
            client.navigate(client.url);
          }
        });
      })
  );
  self.clients.claim();
});

self.addEventListener('fetch', (e) => {
  // Always fetch fresh from network without caching
  e.respondWith(
    fetch(e.request).catch(() => caches.match(e.request))
  );
});
