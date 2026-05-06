// ONNX Runtime Web bridge for Flutter.
// Loaded after ort.min.js; exposes window._slcOrt for Dart JS interop.

(function () {
  ort.env.wasm.wasmPaths = '';  // serve WASM from same directory

  // ── macro CNN ────────────────────────────────────────────────────────────
  let _cnnSession = null;
  let _cnnInitPromise = null;

  async function _initCnn() {
    if (_cnnSession) return;
    _cnnSession = await ort.InferenceSession.create(
      'assets/assets/models/macro_cnn.onnx',
      { executionProviders: ['wasm'] }
    );
  }

  // Returns { macroClass: int, macroConf: float, wear: float }
  async function runInference(float32Array) {
    if (!_cnnInitPromise) _cnnInitPromise = _initCnn();
    await _cnnInitPromise;

    const tensor = new ort.Tensor('float32', float32Array, [1, 1, 64, 64]);
    const results = await _cnnSession.run({ 'height_field': tensor });

    const logits = results['macro_logits'].data;
    const wearArr = results['wear_estimate'].data;

    const maxLogit = Math.max(...logits);
    const exps = logits.map(v => Math.exp(v - maxLogit));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    let maxVal = -Infinity, maxIdx = 0;
    for (let i = 0; i < 16; i++) {
      if (exps[i] > maxVal) { maxVal = exps[i]; maxIdx = i; }
    }
    return {
      macroClass: maxIdx,
      macroConf:  maxVal / sumExp,
      wear:       Math.max(0, Math.min(1, wearArr[0])),
    };
  }

  // ── MiDaS depth ──────────────────────────────────────────────────────────
  let _midasSession = null;
  let _midasInitPromise = null;

  async function _initMidas() {
    if (_midasSession) return;
    _midasSession = await ort.InferenceSession.create(
      'assets/assets/models/midas_small.onnx',
      { executionProviders: ['wasm'] }
    );
    console.log('[slcOrt] MiDaS session ready');
  }

  // jpegBytes: Uint8Array of JPEG image bytes from camera
  // Returns Float32Array of length 128*128 — a 128×128 height field in mm.
  async function runMidas(jpegBytes, tileX, tileY, tileW, tileH) {
    if (!_midasInitPromise) _midasInitPromise = _initMidas();
    await _midasInitPromise;

    // ── decode JPEG → ImageData via OffscreenCanvas ─────────────────────
    const blob  = new Blob([jpegBytes], { type: 'image/jpeg' });
    const bmp   = await createImageBitmap(blob);
    const cvs   = new OffscreenCanvas(256, 256);
    const ctx   = cvs.getContext('2d');

    // Crop to tile bounding box, then resize to 256×256
    const sx = tileX - tileW / 2, sy = tileY - tileH / 2;
    ctx.drawImage(bmp, sx, sy, tileW, tileH, 0, 0, 256, 256);
    const imgData = ctx.getImageData(0, 0, 256, 256);
    bmp.close();

    // ── ImageNet normalisation → NCHW float32 ───────────────────────────
    const mean = [0.485, 0.456, 0.406];
    const std  = [0.229, 0.224, 0.225];
    const inp  = new Float32Array(3 * 256 * 256);
    const px   = imgData.data;
    for (let i = 0; i < 256 * 256; i++) {
      const r = px[i*4] / 255, g = px[i*4+1] / 255, b = px[i*4+2] / 255;
      inp[i]               = (r - mean[0]) / std[0];
      inp[i + 256*256]     = (g - mean[1]) / std[1];
      inp[i + 2*256*256]   = (b - mean[2]) / std[2];
    }

    const tensor  = new ort.Tensor('float32', inp, [1, 3, 256, 256]);
    const results = await _midasSession.run({ 'input': tensor });
    const depth   = results['output'].data; // Float32Array [1,1,256,256]

    // ── Crop inner 128×128 and convert to ~mm scale ──────────────────────
    // MiDaS outputs inverse relative depth; we normalise to [1.5 .. 8.0] mm
    // (the codec's height range) using the tile region statistics.
    const inner = new Float32Array(128 * 128);
    const off   = 64;  // (256-128)/2
    let minV = Infinity, maxV = -Infinity;
    for (let j = 0; j < 128; j++) {
      for (let i = 0; i < 128; i++) {
        const v = depth[(j + off) * 256 + (i + off)];
        inner[j * 128 + i] = v;
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }
    }
    const range = maxV - minV || 1;
    for (let k = 0; k < inner.length; k++) {
      inner[k] = 1.5 + ((inner[k] - minV) / range) * 6.5;
    }
    return inner;  // Float32Array(128*128) in mm
  }

  // ── YOLOv8-nano-seg tile detector ────────────────────────────────────────
  // ── Optional YOLOv8 ONNX upgrade ─────────────────────────────────────────
  // If tile_detector.onnx is present, it takes priority.
  // Otherwise falls back to the Hough circle detector below.

  let _yoloSession = null;
  let _yoloFailed  = false;

  (async () => {
    try {
      _yoloSession = await ort.InferenceSession.create(
        'assets/assets/models/tile_detector.onnx',
        { executionProviders: ['wasm'] }
      );
      console.log('[slcOrt] YOLOv8 tile detector ready');
    } catch (_) {
      _yoloFailed = true;  // model not deployed yet — Hough fallback will be used
    }
  })();

  // ── Hough circle detector (runs when YOLO model not loaded) ───────────────
  // Uses gradient-direction voting on a 320×240 working image.
  // Typical latency: ~8 ms on a modern phone browser.

  async function _houghCircleDetect(jpegBytes, imgW, imgH) {
    const blob = new Blob([jpegBytes], { type: 'image/jpeg' });
    const bmp  = await createImageBitmap(blob);

    // Work at 1/4 resolution for speed (portrait or landscape)
    const W = 320, H = Math.round(320 * imgH / imgW) || 240;
    const cvs = new OffscreenCanvas(W, H);
    const ctx = cvs.getContext('2d');
    ctx.drawImage(bmp, 0, 0, W, H);
    bmp.close();
    const px = ctx.getImageData(0, 0, W, H).data;

    // Grayscale
    const g = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++)
      g[i] = (px[i*4]*77 + px[i*4+1]*150 + px[i*4+2]*29) * (1/255/256);

    // Sobel gradients
    const gx = new Float32Array(W * H);
    const gy = new Float32Array(W * H);
    for (let y = 1; y < H-1; y++) {
      for (let x = 1; x < W-1; x++) {
        const p = y*W+x;
        gx[p] = -g[p-W-1] - 2*g[p-1] - g[p+W-1] + g[p-W+1] + 2*g[p+1] + g[p+W+1];
        gy[p] = -g[p-W-1] - 2*g[p-W]  - g[p-W+1] + g[p+W-1] + 2*g[p+W] + g[p+W+1];
      }
    }

    // Gradient magnitude threshold (keep top 20%)
    const mag = new Float32Array(W * H);
    let maxM = 0;
    for (let i = 0; i < W * H; i++) {
      mag[i] = Math.sqrt(gx[i]*gx[i] + gy[i]*gy[i]);
      if (mag[i] > maxM) maxM = mag[i];
    }
    if (maxM < 0.01) return null;
    const thresh = maxM * 0.18;

    // Expected tile radius range (15%–42% of shorter working dimension)
    const minD  = Math.min(W, H);
    const rMin  = Math.max(10, Math.floor(minD * 0.075));
    const rMax  = Math.ceil(minD * 0.21);
    const rStep = Math.max(1, Math.floor((rMax - rMin) / 12));
    const radii = [];
    for (let r = rMin; r <= rMax; r += rStep) radii.push(r);

    // Accumulator (half-res x/y to speed up)
    const aW = Math.ceil(W/2), aH = Math.ceil(H/2);
    const nR = radii.length;
    const acc = new Int32Array(aW * aH * nR);

    for (let y = 1; y < H-1; y++) {
      for (let x = 1; x < W-1; x++) {
        const p = y*W+x;
        if (mag[p] < thresh) continue;
        const a = Math.atan2(gy[p], gx[p]);
        const ca = Math.cos(a), sa = Math.sin(a);
        for (let ri = 0; ri < nR; ri++) {
          const r = radii[ri];
          for (const s of [1, -1]) {
            const cx = Math.round(x + r*ca*s);
            const cy = Math.round(y + r*sa*s);
            if (cx < 0 || cx >= W || cy < 0 || cy >= H) continue;
            acc[ri * aH * aW + Math.floor(cy/2) * aW + Math.floor(cx/2)]++;
          }
        }
      }
    }

    // Find peak
    let bestV = 0, bestCx = 0, bestCy = 0, bestR = rMin;
    for (let ri = 0; ri < nR; ri++) {
      const r = radii[ri];
      const minVotes = Math.floor(Math.PI * r * 0.25);  // ≥25% of circumference
      const base = ri * aH * aW;
      for (let ay = 0; ay < aH; ay++) {
        for (let ax = 0; ax < aW; ax++) {
          const v = acc[base + ay * aW + ax];
          if (v > bestV && v >= minVotes) {
            bestV = v; bestCx = ax*2+1; bestCy = ay*2+1; bestR = r;
          }
        }
      }
    }

    if (bestV === 0) return null;

    const conf = Math.min(0.92, bestV / (Math.PI * bestR * 0.5));
    if (conf < 0.15) return null;

    // Scale back to original frame coords
    const sx = imgW / W, sy = imgH / H;
    return { cx: bestCx*sx, cy: bestCy*sy, w: bestR*2*sx, h: bestR*2*sy, conf };
  }

  // ── Public: tile detection entry point ───────────────────────────────────
  // jpegBytes : Uint8Array — full camera JPEG frame
  // imgW, imgH : original frame dimensions
  // Returns { cx, cy, w, h, conf } or null.
  async function runYolo(jpegBytes, imgW, imgH) {
    // Use YOLO model if loaded; otherwise fall back to Hough circles
    if (_yoloSession) {
      try {
        return await _runYoloOnnx(jpegBytes, imgW, imgH);
      } catch (_) { /* fall through to Hough */ }
    }
    return _houghCircleDetect(jpegBytes, imgW, imgH);
  }

  async function _runYoloOnnx(jpegBytes, imgW, imgH) {
    const blob = new Blob([jpegBytes], { type: 'image/jpeg' });
    const bmp  = await createImageBitmap(blob);
    const cvs  = new OffscreenCanvas(640, 640);
    const ctx  = cvs.getContext('2d');
    ctx.drawImage(bmp, 0, 0, 640, 640);
    bmp.close();
    const px  = ctx.getImageData(0, 0, 640, 640).data;
    const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
    const inp  = new Float32Array(3 * 640 * 640);
    for (let i = 0; i < 640*640; i++) {
      inp[i]            = (px[i*4]   /255 - mean[0]) / std[0];
      inp[i + 640*640]  = (px[i*4+1] /255 - mean[1]) / std[1];
      inp[i+2*640*640]  = (px[i*4+2] /255 - mean[2]) / std[2];
    }
    const out0 = (await _yoloSession.run(
      { images: new ort.Tensor('float32', inp, [1, 3, 640, 640]) }
    ))['output0'].data;
    const n = 8400;
    let bestConf = 0.30, bestBox = null;
    for (let i = 0; i < n; i++) {
      const conf = out0[4*n + i];
      if (conf > bestConf) {
        bestConf = conf;
        bestBox = {
          cx: out0[i]       / 640 * imgW,
          cy: out0[n+i]     / 640 * imgH,
          w:  out0[2*n+i]   / 640 * imgW,
          h:  out0[3*n+i]   / 640 * imgH,
          conf,
        };
      }
    }
    return bestBox;
  }

  window._slcOrt = { runInference, runMidas, runYolo };
})();
