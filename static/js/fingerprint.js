/**
 * fingerprint.js
 * Collects stable, low-entropy-but-combined client-side signals to build
 * a device fingerprint. This runs in the browser because none of these
 * signals (canvas rendering, WebGL renderer string, font list, screen
 * geometry) are visible to the server from headers alone.
 *
 * IMPORTANT: this fingerprint is a *behavioural/hardware* signal, not a
 * secret. It is meant to detect "this looks like a different device than
 * before", not to authenticate on its own. It is combined with TOTP and
 * SIM-binding checks server-side.
 */

(function () {
  function getCanvasFingerprint() {
    try {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = 220;
      canvas.height = 30;

      ctx.textBaseline = 'top';
      ctx.font = "14px 'Arial'";
      ctx.fillStyle = '#f60';
      ctx.fillRect(0, 0, 100, 20);
      ctx.fillStyle = '#069';
      ctx.fillText('SecureAuth-fp-check-😀', 2, 2);
      ctx.fillStyle = 'rgba(102, 200, 0, 0.7)';
      ctx.fillText('SecureAuth-fp-check-😀', 4, 8);

      return canvas.toDataURL();
    } catch (e) {
      return 'canvas-unavailable';
    }
  }

  function getWebGLFingerprint() {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      if (!gl) return 'webgl-unavailable';

      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
      if (!debugInfo) return 'webgl-no-debug-info';

      const vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
      const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
      return `${vendor}~${renderer}`;
    } catch (e) {
      return 'webgl-error';
    }
  }

  function getFontFingerprint() {
    // Detect a fixed list of candidate fonts by measuring text width
    // against baseline fonts. No installed-font enumeration API exists,
    // so this is the standard width-comparison technique.
    const baseFonts = ['monospace', 'sans-serif', 'serif'];
    const testFonts = [
      'Arial', 'Verdana', 'Times New Roman', 'Courier New', 'Georgia',
      'Comic Sans MS', 'Trebuchet MS', 'Impact', 'Tahoma'
    ];
    const testString = 'mmmmmmmmmmlli';
    const testSize = '72px';
    const span = document.createElement('span');
    span.style.fontSize = testSize;
    span.style.position = 'absolute';
    span.style.left = '-9999px';
    span.innerHTML = testString;
    document.body.appendChild(span);

    const baseWidths = {};
    baseFonts.forEach((bf) => {
      span.style.fontFamily = bf;
      baseWidths[bf] = span.offsetWidth;
    });

    const detected = [];
    testFonts.forEach((font) => {
      let isDetected = false;
      for (const bf of baseFonts) {
        span.style.fontFamily = `'${font}', ${bf}`;
        if (span.offsetWidth !== baseWidths[bf]) {
          isDetected = true;
          break;
        }
      }
      if (isDetected) detected.push(font);
    });

    document.body.removeChild(span);
    return detected.join(',');
  }

  function getScreenFingerprint() {
    return [
      window.screen.width,
      window.screen.height,
      window.screen.colorDepth,
      window.devicePixelRatio || 1
    ].join('x');
  }

  function getTimezoneFingerprint() {
    try {
      return Intl.DateTimeFormat().resolvedOptions().timeZone || String(new Date().getTimezoneOffset());
    } catch (e) {
      return String(new Date().getTimezoneOffset());
    }
  }

  async function sha256Hex(message) {
    const msgBuffer = new TextEncoder().encode(message);
    const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
  }

  /**
   * Build the full fingerprint payload. Returns a promise resolving to
   * { hash, raw } where raw is the underlying component breakdown (useful
   * for debugging / showing "why did this device look different").
   */
  window.collectDeviceFingerprint = async function () {
    const components = {
      canvas: getCanvasFingerprint(),
      webgl: getWebGLFingerprint(),
      fonts: getFontFingerprint(),
      screen: getScreenFingerprint(),
      timezone: getTimezoneFingerprint(),
      platform: navigator.platform || 'unknown',
      hardwareConcurrency: navigator.hardwareConcurrency || 0,
      languages: (navigator.languages || [navigator.language || '']).join(',')
    };

    const raw = JSON.stringify(components);
    const hash = await sha256Hex(raw);

    return { hash, components };
  };
})();