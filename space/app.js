/*
 * Browser port of the `rmn` inference pipeline.
 *
 * YuNet locates faces, ResMaskingNet classifies each crop. Every preprocessing
 * step mirrors the Python package exactly -- grayscale conversion, the 1.1x
 * square box expansion, OpenCV's bilinear pixel-center mapping, and the plain
 * /255 scaling that `transforms.ToTensor()` applies -- so the browser and
 * `pip install rmn` agree on the same image.
 */

/* global ort */

/* Every index into an array in this file is a loop counter or a value derived
   from model output shapes, never a user-supplied key, so the object-injection
   rule only produces false positives on the pixel and tensor loops below. */
/* eslint-disable security/detect-object-injection */

// overridable so the pipeline can be exercised against local files in tests
const MODEL_BASE =
  window.RMN_MODEL_BASE ||
  "https://huggingface.co/phamquiluan/ResidualMaskingNetwork/resolve/main/";
const CLASSIFIER_URL = MODEL_BASE + "onnx/resmasking_int8.onnx";
const DETECTOR_URL = MODEL_BASE + "face_detection_yunet_2023mar.onnx";

const EMOTIONS = [
  "angry",
  "disgust",
  "fear",
  "happy",
  "sad",
  "surprise",
  "neutral",
];

const DET_SIZE = 640;
const STRIDES = [8, 16, 32];
const SCORE_THRESHOLD = 0.5;
const NMS_THRESHOLD = 0.3;
const FACE_SIZE = 224;

let detectorSession = null;
let classifierSession = null;

/* ---------- model loading ---------- */

async function fetchWithProgress(url, onProgress) {
  const cache = await caches.open("rmn-models-v1");
  const cached = await cache.match(url);
  if (cached) {
    onProgress(1);
    return cached.arrayBuffer();
  }

  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} fetching ${url}`);

  const total = Number(response.headers.get("content-length")) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) onProgress(received / total);
  }

  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }

  // store a copy so repeat visits skip the download entirely
  await cache.put(url, new Response(buffer.slice()));
  return buffer.buffer;
}

async function loadModels(onStatus) {
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";
  // WASM threads need cross-origin isolation, which static hosting does not
  // provide; single-threaded runs a face in well under a second, so take the
  // extra threads only if the headers ever appear
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 1)
    : 1;
  ort.env.wasm.simd = true;

  onStatus("Loading face detector...", 0);
  const detectorBytes = await fetchWithProgress(DETECTOR_URL, (p) =>
    onStatus("Loading face detector...", p * 0.02),
  );
  detectorSession = await ort.InferenceSession.create(detectorBytes, {
    executionProviders: ["wasm"],
  });

  onStatus(
    "Downloading emotion model (132 MB, cached after first visit)...",
    0.02,
  );
  const classifierBytes = await fetchWithProgress(CLASSIFIER_URL, (p) =>
    onStatus(
      `Downloading emotion model (132 MB, cached after first visit)... ${Math.round(p * 100)}%`,
      0.02 + p * 0.93,
    ),
  );

  onStatus("Initialising inference session...", 0.96);
  classifierSession = await ort.InferenceSession.create(classifierBytes, {
    executionProviders: ["wasm"],
  });

  onStatus("Ready.", 1);
}

/* ---------- image helpers (ported from OpenCV semantics) ---------- */

/** cv2.COLOR_BGR2GRAY / COLOR_RGB2GRAY weights. */
function toGrayscale(rgba, width, height) {
  const gray = new Float32Array(width * height);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
  }
  return gray;
}

/**
 * Bilinear resize using OpenCV's pixel-center mapping,
 * src = (dst + 0.5) * scale - 0.5, which differs from a naive dst * scale
 * and shifts the sampled crop by half a pixel if you get it wrong.
 */
function resizeBilinear(src, srcW, srcH, dstW, dstH) {
  const dst = new Float32Array(dstW * dstH);
  const scaleX = srcW / dstW;
  const scaleY = srcH / dstH;

  for (let y = 0; y < dstH; y++) {
    let fy = (y + 0.5) * scaleY - 0.5;
    if (fy < 0) fy = 0;
    const y0 = Math.min(Math.floor(fy), srcH - 1);
    const y1 = Math.min(y0 + 1, srcH - 1);
    const wy = fy - y0;

    for (let x = 0; x < dstW; x++) {
      let fx = (x + 0.5) * scaleX - 0.5;
      if (fx < 0) fx = 0;
      const x0 = Math.min(Math.floor(fx), srcW - 1);
      const x1 = Math.min(x0 + 1, srcW - 1);
      const wx = fx - x0;

      const top = src[y0 * srcW + x0] * (1 - wx) + src[y0 * srcW + x1] * wx;
      const bottom = src[y1 * srcW + x0] * (1 - wx) + src[y1 * srcW + x1] * wx;
      dst[y * dstW + x] = top * (1 - wy) + bottom * wy;
    }
  }
  return dst;
}

/* ---------- face detection ---------- */

function nonMaxSuppression(boxes, scores, threshold) {
  const order = scores
    .map((score, index) => index)
    .sort((a, b) => scores[b] - scores[a]);
  const keep = [];

  while (order.length) {
    const current = order.shift();
    keep.push(current);
    const [ax, ay, aw, ah] = boxes[current];

    for (let i = order.length - 1; i >= 0; i--) {
      const [bx, by, bw, bh] = boxes[order[i]];
      const overlapW = Math.max(
        0,
        Math.min(ax + aw, bx + bw) - Math.max(ax, bx),
      );
      const overlapH = Math.max(
        0,
        Math.min(ay + ah, by + bh) - Math.max(ay, by),
      );
      const intersection = overlapW * overlapH;
      const iou = intersection / (aw * ah + bw * bh - intersection);
      if (iou > threshold) order.splice(i, 1);
    }
  }
  return keep;
}

/**
 * Letterbox into DET_SIZE so the model sees an undistorted image; the ONNX has
 * a fixed 640x640 input, and a plain square resize would stretch tall or wide
 * photos and shift the boxes.
 */
function letterbox(canvas) {
  const scale = Math.min(DET_SIZE / canvas.width, DET_SIZE / canvas.height);
  const target = document.createElement("canvas");
  target.width = DET_SIZE;
  target.height = DET_SIZE;
  const context = target.getContext("2d", { willReadFrequently: true });
  context.fillStyle = "black";
  context.fillRect(0, 0, DET_SIZE, DET_SIZE);
  context.drawImage(
    canvas,
    0,
    0,
    Math.round(canvas.width * scale),
    Math.round(canvas.height * scale),
  );
  return { canvas: target, scale };
}

async function detectFaces(sourceCanvas) {
  const { canvas, scale } = letterbox(sourceCanvas);
  const { data } = canvas
    .getContext("2d", { willReadFrequently: true })
    .getImageData(0, 0, DET_SIZE, DET_SIZE);

  // YuNet consumes raw BGR values in [0, 255], no mean subtraction
  const input = new Float32Array(3 * DET_SIZE * DET_SIZE);
  const plane = DET_SIZE * DET_SIZE;
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    input[i] = data[p + 2];
    input[plane + i] = data[p + 1];
    input[2 * plane + i] = data[p];
  }

  const outputs = await detectorSession.run({
    input: new ort.Tensor("float32", input, [1, 3, DET_SIZE, DET_SIZE]),
  });

  const boxes = [];
  const scores = [];
  for (const stride of STRIDES) {
    const cls = outputs[`cls_${stride}`].data;
    const obj = outputs[`obj_${stride}`].data;
    const bbox = outputs[`bbox_${stride}`].data;
    const cols = DET_SIZE / stride;

    for (let i = 0; i < cls.length; i++) {
      const clsScore = Math.min(Math.max(cls[i], 0), 1);
      const objScore = Math.min(Math.max(obj[i], 0), 1);
      const score = Math.sqrt(clsScore * objScore);
      if (score < SCORE_THRESHOLD) continue;

      const col = i % cols;
      const row = Math.floor(i / cols);
      const cx = (col + bbox[i * 4]) * stride;
      const cy = (row + bbox[i * 4 + 1]) * stride;
      const w = Math.exp(bbox[i * 4 + 2]) * stride;
      const h = Math.exp(bbox[i * 4 + 3]) * stride;
      boxes.push([cx - w / 2, cy - h / 2, w, h]);
      scores.push(score);
    }
  }

  return nonMaxSuppression(boxes, scores, NMS_THRESHOLD).map((index) => {
    const [x, y, w, h] = boxes[index];
    return {
      xmin: x / scale,
      ymin: y / scale,
      xmax: (x + w) / scale,
      ymax: (y + h) / scale,
      score: scores[index],
    };
  });
}

/* ---------- classification ---------- */

/** Port of rmn's convert_to_square: a 1.1x expanded square around the box. */
function convertToSquare(xmin, ymin, xmax, ymax) {
  const centerX = Math.floor((xmin + xmax) / 2);
  const centerY = Math.floor((ymin + ymax) / 2);
  let length = Math.floor(Math.floor((xmax - xmin + (ymax - ymin)) / 2) / 2);
  length *= 1.1;
  return {
    xmin: Math.trunc(centerX - length),
    ymin: Math.trunc(centerY - length),
    xmax: Math.trunc(centerX + length),
    ymax: Math.trunc(centerY + length),
  };
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

async function classifyCrop(gray, width, height, box) {
  // numpy slicing truncates at the array edge rather than padding, so clamp
  // the same way the Python does
  const x0 = Math.max(box.xmin, 0);
  const y0 = Math.max(box.ymin, 0);
  const x1 = Math.min(box.xmax, width);
  const y1 = Math.min(box.ymax, height);
  const cropW = x1 - x0;
  const cropH = y1 - y0;
  if (cropW < 10 || cropH < 10) return null;

  const crop = new Float32Array(cropW * cropH);
  for (let y = 0; y < cropH; y++) {
    for (let x = 0; x < cropW; x++) {
      crop[y * cropW + x] = gray[(y0 + y) * width + (x0 + x)];
    }
  }

  const resized = resizeBilinear(crop, cropW, cropH, FACE_SIZE, FACE_SIZE);
  const plane = FACE_SIZE * FACE_SIZE;
  const input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    const value = resized[i] / 255;
    input[i] = value;
    input[plane + i] = value;
    input[2 * plane + i] = value;
  }

  const outputs = await classifierSession.run({
    input: new ort.Tensor("float32", input, [1, 3, FACE_SIZE, FACE_SIZE]),
  });
  const logits = Array.from(Object.values(outputs)[0].data);
  const probabilities = softmax(logits);
  const best = probabilities.indexOf(Math.max(...probabilities));

  return {
    label: EMOTIONS[best],
    probability: probabilities[best],
    probabilities,
  };
}

/* ---------- top level ---------- */

async function analyse(sourceCanvas) {
  const context = sourceCanvas.getContext("2d", { willReadFrequently: true });
  const { data } = context.getImageData(
    0,
    0,
    sourceCanvas.width,
    sourceCanvas.height,
  );
  const gray = toGrayscale(data, sourceCanvas.width, sourceCanvas.height);

  let faces = await detectFaces(sourceCanvas);
  let usedFallback = false;

  if (!faces.length) {
    faces = [
      {
        xmin: 0,
        ymin: 0,
        xmax: sourceCanvas.width,
        ymax: sourceCanvas.height,
        score: null,
      },
    ];
    usedFallback = true;
  }

  const results = [];
  for (const face of faces) {
    const box = usedFallback
      ? face
      : convertToSquare(face.xmin, face.ymin, face.xmax, face.ymax);
    const prediction = await classifyCrop(
      gray,
      sourceCanvas.width,
      sourceCanvas.height,
      box,
    );
    if (prediction) results.push({ ...box, ...prediction });
  }

  return { results, usedFallback };
}

window.rmn = { loadModels, analyse, EMOTIONS };
