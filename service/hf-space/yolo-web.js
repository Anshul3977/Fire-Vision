// Browser-side YOLOv8 inference with onnxruntime-web (WASM, runs on the visitor's CPU).
// Same pipeline as the Python service: letterbox 640 -> forward -> decode -> class-aware NMS.
// Supports the axis-aligned detect head (1, 4+nc, N) and the OBB head (1, 4+nc+1, N).
export async function loadModel(url, names, { obb = false, size = 640 } = {}) {
  const ort = window.ort;
  ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(url, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  return { session, names, obb, size, loadMs: performance.now() - t0 };
}

function letterbox(src, w, h, size) {
  const scale = Math.min(size / w, size / h), nw = Math.round(w * scale), nh = Math.round(h * scale);
  const px = Math.floor((size - nw) / 2), py = Math.floor((size - nh) / 2);
  const c = new OffscreenCanvas(size, size), g = c.getContext("2d", { willReadFrequently: true });
  g.fillStyle = "rgb(114,114,114)"; g.fillRect(0, 0, size, size); g.drawImage(src, px, py, nw, nh);
  const px4 = g.getImageData(0, 0, size, size).data, n = size * size, out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) { out[i] = px4[i * 4] / 255; out[n + i] = px4[i * 4 + 1] / 255; out[2 * n + i] = px4[i * 4 + 2] / 255; }
  return { data: out, scale, px, py };
}

const iou = (a, b) => {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]), x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9);
};

export async function detect(model, src, w, h, { conf = 0.25, iouThres = 0.45, maxDet = 100 } = {}) {
  const ort = window.ort, t0 = performance.now();
  const lb = letterbox(src, w, h, model.size);
  const t1 = performance.now();
  const feeds = { [model.session.inputNames[0]]: new ort.Tensor("float32", lb.data, [1, 3, model.size, model.size]) };
  const out = (await model.session.run(feeds))[model.session.outputNames[0]];
  const t2 = performance.now();
  const [, ch, n] = out.dims, d = out.data, nc = Object.keys(model.names).length;
  const cand = [];
  for (let i = 0; i < n; i++) {
    let best = 0, cls = 0;
    for (let c = 0; c < nc; c++) { const s = d[(4 + c) * n + i]; if (s > best) { best = s; cls = c; } }
    if (best < conf) continue;
    const cx = d[i], cy = d[n + i], bw = d[2 * n + i], bh = d[3 * n + i];
    const angle = model.obb ? d[(4 + nc) * n + i] : 0;
    // For OBB, NMS uses the rotated box's axis-aligned hull (close enough for display).
    const ex = Math.abs(bw * Math.cos(angle)) / 2 + Math.abs(bh * Math.sin(angle)) / 2;
    const ey = Math.abs(bw * Math.sin(angle)) / 2 + Math.abs(bh * Math.cos(angle)) / 2;
    cand.push({ score: best, cls, cx, cy, bw, bh, angle, hull: [cx - ex, cy - ey, cx + ex, cy + ey] });
  }
  cand.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const c of cand.slice(0, 1000)) {
    if (keep.length >= maxDet) break;
    if (keep.every((k) => k.cls !== c.cls || iou(k.hull, c.hull) <= iouThres)) keep.push(c);
  }
  const map = (x, y) => [(x - lb.px) / lb.scale, (y - lb.py) / lb.scale];
  const dets = keep.map((k) => {
    const [x1, y1] = map(k.hull[0], k.hull[1]), [x2, y2] = map(k.hull[2], k.hull[3]);
    const cos = Math.cos(k.angle), sin = Math.sin(k.angle), poly = [];
    for (const [dx, dy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const ox = (dx * k.bw) / 2, oy = (dy * k.bh) / 2;
      poly.push(map(k.cx + ox * cos - oy * sin, k.cy + ox * sin + oy * cos));
    }
    return { class_id: k.cls, label: model.names[k.cls] ?? String(k.cls), confidence: +k.score.toFixed(3),
             box: { x1: Math.max(0, x1), y1: Math.max(0, y1), x2: Math.min(w, x2), y2: Math.min(h, y2) }, polygon: poly };
  });
  const t3 = performance.now();
  return { detections: dets, timings: { preprocess_ms: +(t1 - t0).toFixed(1), inference_ms: +(t2 - t1).toFixed(1), postprocess_ms: +(t3 - t2).toFixed(1) } };
}

// Greedy same-class IoU tracker (mirrors app/tracker.py in the API repo).
export class IoUTracker {
  constructor({ iouThres = 0.3, maxAge = 5, minHits = 2 } = {}) { Object.assign(this, { iouThres, maxAge, minHits, tracks: [], next: 1, confirmed: new Map() }); }
  update(dets) {
    const pairs = [];
    this.tracks.forEach((t, ti) => dets.forEach((d, di) => {
      if (t.cls !== d.class_id) return;
      const v = iou(t.box, [d.box.x1, d.box.y1, d.box.x2, d.box.y2]); if (v >= this.iouThres) pairs.push([v, ti, di]);
    }));
    pairs.sort((a, b) => b[0] - a[0]);
    const usedT = new Set(), assigned = new Map();
    for (const [, ti, di] of pairs) { if (usedT.has(ti) || assigned.has(di)) continue; usedT.add(ti); assigned.set(di, this.tracks[ti]); }
    this.tracks.forEach((t, ti) => { if (!usedT.has(ti)) t.age++; });
    const out = dets.map((d, di) => {
      let t = assigned.get(di); const box = [d.box.x1, d.box.y1, d.box.x2, d.box.y2];
      if (!t) { t = { id: this.next++, cls: d.class_id, label: d.label, box, hits: 1, age: 0 }; this.tracks.push(t); }
      else Object.assign(t, { box, hits: t.hits + 1, age: 0 });
      if (t.hits >= this.minHits) this.confirmed.set(t.id, t.label);
      return { ...d, track_id: t.id };
    });
    this.tracks = this.tracks.filter((t) => t.age <= this.maxAge);
    return out;
  }
  summary() { const c = {}; for (const l of this.confirmed.values()) c[l] = (c[l] || 0) + 1; return c; }
}
