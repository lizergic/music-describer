// PURE music analysis — ports music_describer/analyzers/*.py + llm/prompt.py.
// Consumes a `features` object (produced by audio.js); imports nothing browser-only,
// so the whole file runs under `node --test`.
//
// features = {
//   sr, hop, duration,
//   rms:      Float32Array(nFrames),        // per-frame RMS energy
//   onset:    Float32Array(nFrames),        // onset-strength envelope (>=0)
//   chroma:   Array<Array(12)>,             // per-frame chroma (nFrames x 12)
//   mfcc:     Array<Array(13)>,             // per-frame MFCC   (nFrames x 13)
//   centroid: Float32Array(nFrames),        // spectral centroid in Hz
//   rolloff:  Float32Array(nFrames),        // spectral rolloff in Hz
// }

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

// Krumhansl-Kessler key profiles
const MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

// Mode profiles (binary scale degrees relative to root)
const MODE_PROFILES = {
  ionian: [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
  dorian: [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
  phrygian: [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
  lydian: [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
  mixolydian: [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
  aeolian: [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
  locrian: [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
};

// ---------- math helpers (numpy/scipy parity) ----------
const arr = (a) => (Array.isArray(a) ? a : Array.from(a));
const sum = (a) => a.reduce((s, v) => s + v, 0);
const mean = (a) => (a.length ? sum(a) / a.length : 0);
function variance(a) { const m = mean(a); return a.length ? a.reduce((s, v) => s + (v - m) * (v - m), 0) / a.length : 0; }
const std = (a) => Math.sqrt(variance(a));
const round = (v, d) => { const f = 10 ** d; return Math.round(v * f) / f; };

// numpy.percentile, linear interpolation (default)
function percentile(a, p) {
  if (!a.length) return 0;
  const s = arr(a).slice().sort((x, y) => x - y);
  const idx = (p / 100) * (s.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (idx - lo);
}

// numpy.roll(a, shift): out[(i+shift) mod n] = a[i]
function npRoll(a, shift) {
  const n = a.length, out = new Array(n);
  for (let i = 0; i < n; i++) out[((i + shift) % n + n) % n] = a[i];
  return out;
}

// numpy.corrcoef(x, y)[0,1] — Pearson; NaN when a vector is constant (matches numpy).
function pearson(x, y) {
  const n = x.length, mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { const dx = x[i] - mx, dy = y[i] - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
  const d = Math.sqrt(sxx * syy);
  return d > 0 ? sxy / d : NaN;
}

// autocorrelation of x for lags 0..maxLag (librosa.autocorrelate, truncated to needed lags)
function autocorr(x, maxLag) {
  maxLag = Math.min(maxLag, x.length - 1);
  const out = new Float64Array(Math.max(0, maxLag) + 1);
  for (let k = 0; k <= maxLag; k++) { let s = 0; for (let t = 0; t + k < x.length; t++) s += x[t] * x[t + k]; out[k] = s; }
  return out;
}

const framesToTime = (frame, sr, hop) => (frame * hop) / sr;

// ---------- rhythm (rhythm.py) ----------
export function rhythmAnalyzer(f) {
  const onset = arr(f.onset);
  const tempo = estimateTempo(onset, f.sr, f.hop);
  const beatStrength = mean(onset);
  const beats = pickBeats(onset, f.sr, f.hop, tempo);
  const timeSignature = estimateTimeSignature(onset, tempo, f.sr, f.hop);
  const feel = estimateFeel(onset, beats);
  return {
    tempo: round(tempo, 1),
    time_signature: timeSignature,
    feel,
    beat_strength: round(beatStrength, 3),
  };
}

// librosa.feature.tempo: argmax over BPM of log1p(1e6 * tempogram) + log-normal prior.
// ponytail: global autocorrelation instead of a windowed per-frame tempogram, and no DP
// beat tracker — a steady-tempo approximation of librosa.beat.beat_track. The log1p +
// additive prior below is what fixes octave selection; upgrade to a windowed tempogram
// only if tempo-varying tracks read an octave off.
function estimateTempo(onset, sr, hop) {
  if (onset.length < 2) return 0;
  const minBpm = 30, maxBpm = 300, startBpm = 120, stdBpm = 1.0;
  const maxLag = Math.min(onset.length - 1, Math.ceil((60 * sr) / (hop * minBpm)));
  const ac = autocorr(onset, maxLag);
  let acMax = 0;
  for (let k = 1; k <= maxLag; k++) acMax = Math.max(acMax, ac[k]);
  if (acMax <= 0) return 0;
  let best = -Infinity, bestBpm = 0;
  for (let lag = 1; lag <= maxLag; lag++) {
    const bpm = (60 * sr) / (hop * lag);
    if (bpm < minBpm || bpm > maxBpm) continue;
    const logp = Math.log2(bpm / startBpm);
    const score = Math.log1p(1e6 * (ac[lag] / acMax)) - 0.5 * (logp / stdBpm) ** 2;
    if (score > best) { best = score; bestBpm = bpm; }
  }
  return bestBpm;
}

// Approximate beat positions by peak-picking the onset envelope (we have no DP beat
// tracker). Used only by feel detection, which needs inter-beat intervals.
// ponytail: O(n^2) min-distance enforcement; fine for track-length envelopes.
function pickBeats(onset, sr, hop, tempo) {
  if (tempo <= 0) return [];
  const fpb = (60 / tempo) * (sr / hop);
  const minDist = Math.max(1, Math.round(fpb * 0.5));
  const m = mean(onset);
  const cands = [];
  for (let i = 1; i < onset.length - 1; i++) {
    if (onset[i] > onset[i - 1] && onset[i] >= onset[i + 1] && onset[i] > m) cands.push(i);
  }
  cands.sort((a, b) => onset[b] - onset[a]);
  const taken = [];
  for (const c of cands) if (taken.every((t) => Math.abs(t - c) >= minDist)) taken.push(c);
  taken.sort((a, b) => a - b);
  return taken;
}

function estimateTimeSignature(onset, tempo, sr, hop) {
  if (tempo <= 0) return '4/4';
  const fpb = (60 / tempo) * (sr / hop);
  if (fpb < 1) return '4/4';
  // librosa.autocorrelate(onset, max_size=len//2): lags >= len//2 are absent → score 0
  const maxSize = Math.floor(onset.length / 2);
  const ac = autocorr(onset, Math.min(onset.length - 1, maxSize - 1));
  if (ac.length < 4) return '4/4';
  if (ac[0] > 0) for (let i = 0; i < ac.length; i++) ac[i] /= ac[0];
  const at = (mult) => { const lag = Math.round(fpb * mult); return lag < maxSize && lag < ac.length ? ac[lag] : 0; };
  const opts = [['3/4', at(3)], ['4/4', at(4)], ['5/4', at(5)], ['7/8', at(7)]];
  opts.sort((a, b) => b[1] - a[1]);
  return opts[0][0];
}

function estimateFeel(onset, beats) {
  if (beats.length < 4) return 'straight';
  const intervals = [];
  for (let i = 1; i < beats.length; i++) intervals.push(beats[i] - beats[i - 1]);
  if (intervals.length < 2) return 'straight';
  const even = intervals.filter((_, i) => i % 2 === 0);
  const odd = intervals.filter((_, i) => i % 2 === 1);
  const minLen = Math.min(even.length, odd.length);
  if (minLen > 0) {
    const ratio = mean(even.slice(0, minLen)) / (mean(odd.slice(0, minLen)) + 1e-10);
    if (ratio > 1.3 || ratio < 0.7) return 'swung';
  }
  const meanStrength = mean(onset);
  if (meanStrength > percentile(onset, 75)) return 'driving';
  const cv = std(onset) / (meanStrength + 1e-10);
  if (cv < 0.5) return 'laid-back';
  return 'straight';
}

// ---------- harmony (harmony.py) ----------
export function harmonyAnalyzer(f) {
  const chromaMean = new Array(12).fill(0);
  for (const fr of f.chroma) for (let b = 0; b < 12; b++) chromaMean[b] += fr[b];
  const n = f.chroma.length || 1;
  for (let b = 0; b < 12; b++) chromaMean[b] /= n;

  const { key, scale } = detectKey(chromaMean);
  return {
    key,
    scale,
    mode: detectMode(chromaMean, key),
    harmonic_complexity: harmonicComplexity(f.chroma),
  };
}

function detectKey(chromaMean) {
  let best = -2, key = 'C', scale = 'major';
  for (let i = 0; i < 12; i++) {
    const rotated = npRoll(chromaMean, -i);
    const maj = pearson(rotated, MAJOR_PROFILE);
    const min = pearson(rotated, MINOR_PROFILE);
    if (!Number.isNaN(maj) && maj > best) { best = maj; key = NOTE_NAMES[i]; scale = 'major'; }
    if (!Number.isNaN(min) && min > best) { best = min; key = NOTE_NAMES[i]; scale = 'minor'; }
  }
  return { key, scale };
}

function detectMode(chromaMean, key) {
  const root = NOTE_NAMES.indexOf(key);
  let rotated = npRoll(chromaMean, -root);
  const total = sum(rotated);
  if (total > 0) rotated = rotated.map((v) => v / total);
  let bestMode = 'ionian', bestScore = -1;
  for (const [name, profile] of Object.entries(MODE_PROFILES)) {
    const psum = sum(profile);
    let score = 0;
    for (let i = 0; i < 12; i++) score += rotated[i] * (profile[i] / psum);
    if (score > bestScore) { bestScore = score; bestMode = name; }
  }
  return bestMode;
}

function harmonicComplexity(chroma) {
  let total = 0;
  for (const fr of chroma) {
    const s = sum(fr) + 1e-10;
    let e = 0;
    for (let b = 0; b < 12; b++) { const p = fr[b] / s; e += p * Math.log2(p + 1e-10); }
    total += -e;
  }
  const avg = total / chroma.length; // empty → NaN → falls through to 'complex', like Python
  if (avg < 2.0) return 'simple';
  if (avg < 3.0) return 'moderate';
  return 'complex';
}

// ---------- timbre (timbre.py) ----------
export function timbreAnalyzer(f) {
  const sr = f.sr;
  const brightness = mean(arr(f.centroid)) / (sr / 2);
  const warmth = 1.0 - brightness;
  return {
    brightness: round(brightness, 3),
    warmth: round(warmth, 3),
    instrumentation_hints: guessInstrumentation(brightness, f.rolloff, f.mfcc, sr),
    has_vocals: detectVocals(f.mfcc, f.centroid, sr),
    tonal_quality: tonalQuality(brightness),
  };
}

function tonalQuality(brightness) {
  if (brightness > 0.35) return 'bright';
  if (brightness > 0.2) return 'neutral';
  if (brightness > 0.1) return 'warm';
  return 'dark';
}

function colVar(matrix, col) { return variance(matrix.map((r) => r[col])); }

function guessInstrumentation(brightness, rolloff, mfcc, sr) {
  const hints = [];
  const rolloffMean = mean(arr(rolloff)) / (sr / 2);
  if (brightness > 0.3 && rolloffMean > 0.5) hints.push('bright synths or cymbals');
  if (brightness < 0.15) hints.push('bass-heavy');
  // ponytail: MFCC magnitudes differ between Meyda and librosa, so the variance
  // thresholds (50, 10, and 20 in vocals) are calibration knobs vs the Python output.
  const ncoef = mfcc[0] ? mfcc[0].length : 13;
  let mfccVar = 0;
  for (let c = 0; c < ncoef; c++) mfccVar += colVar(mfcc, c);
  mfccVar /= ncoef || 1;
  if (mfccVar > 50) hints.push('varied instrumentation');
  else if (mfccVar < 10) hints.push('sparse or uniform timbre');
  if (brightness >= 0.15 && brightness <= 0.3) hints.push('mid-range instruments (guitar, piano, or vocals)');
  if (hints.length === 0) hints.push('mixed instrumentation');
  return hints;
}

function detectVocals(mfcc, centroid, sr) {
  let lowerVar = 0;
  for (let c = 1; c < 5; c++) lowerVar += colVar(mfcc, c);
  lowerVar /= 4;
  const centroidHz = mean(arr(centroid));
  return lowerVar > 20 && centroidHz > 300 && centroidHz < 3000;
}

// ---------- energy (energy.py) ----------
export function energyAnalyzer(f) {
  const rms = arr(f.rms);
  return {
    overall_energy: round(mean(rms), 4),
    dynamic_range: round(Math.max(...rms) - Math.min(...rms), 4),
    energy_arc: detectArc(rms),
    notable_moments: detectNotableMoments(rms, f.sr, f.hop),
  };
}

function detectArc(rms) {
  const n = rms.length;
  if (n < 4) return 'flat';
  const third = Math.floor(n / 3);
  const start = mean(rms.slice(0, third));
  const mid = mean(rms.slice(third, 2 * third));
  const end = mean(rms.slice(2 * third));
  const maxE = Math.max(start, mid, end);
  if (maxE === 0) return 'flat';
  const startToMid = (mid - start) / (maxE + 1e-10);
  const midToEnd = (end - mid) / (maxE + 1e-10);
  const t = 0.15;
  if (startToMid > t && midToEnd > t) return 'building';
  if (startToMid < -t && midToEnd < -t) return 'fading';
  if (mid > start * 1.2 && mid > end * 1.2) return 'peaks in middle';
  if (startToMid > t && midToEnd < -t) return 'builds then fades';
  if (Math.abs(startToMid) < t && Math.abs(midToEnd) < t) return 'flat';
  return 'dynamic';
}

function detectNotableMoments(rms, sr, hop) {
  const moments = [];
  const n = rms.length;
  if (n < 10) return moments;
  const meanRms = mean(rms), stdRms = std(rms);
  const thresholdDrop = -(meanRms + 2 * stdRms);
  const thresholdSpike = meanRms + 2 * stdRms;
  for (let i = 0; i < n - 1; i++) {
    const diff = rms[i + 1] - rms[i];
    const time = round(framesToTime(i, sr, hop), 1);
    if (diff < thresholdDrop) moments.push({ time, type: 'breakdown' });
    else if (diff > thresholdSpike) moments.push({ time, type: 'climax' });
  }
  let peakFrame = 0;
  for (let i = 1; i < n; i++) if (rms[i] > rms[peakFrame]) peakFrame = i;
  const peakTimeRaw = framesToTime(peakFrame, sr, hop); // dedup against the unrounded time, like Python
  if (rms[peakFrame] > meanRms + stdRms &&
      !moments.some((m) => Math.abs(m.time - peakTimeRaw) < 1.0 && m.type === 'climax')) {
    moments.push({ time: round(peakTimeRaw, 1), type: 'peak' });
  }
  moments.sort((a, b) => a.time - b.time);
  return moments;
}

// ---------- structure (structure.py) ----------
export function structureAnalyzer(f) {
  const duration = f.duration;
  const boundaryFrames = findBoundaries(arr(f.rms), f.chroma, f.mfcc);
  const boundaryTimes = boundaryFrames.map((fr) => framesToTime(fr, f.sr, f.hop));
  const all = [0.0, ...boundaryTimes, duration];
  const uniqSorted = [...new Set(all)].sort((a, b) => a - b);
  const sections = labelSections(uniqSorted, arr(f.rms), f.sr, f.hop);
  return { sections, form_summary: sections.map((s) => s.label).join(' -> ') };
}

// librosa.util.normalize(M, norm=inf, axis=0): divide each column by its max-abs.
function normalizeColumns(matrix) {
  const rows = matrix.length, cols = matrix[0] ? matrix[0].length : 0;
  const out = matrix.map((r) => r.slice());
  for (let j = 0; j < cols; j++) {
    let m = 0;
    for (let i = 0; i < rows; i++) m = Math.max(m, Math.abs(matrix[i][j]));
    if (m > 0) for (let i = 0; i < rows; i++) out[i][j] /= m;
  }
  return out;
}

function findBoundaries(rms, chroma, mfcc) {
  const nFrames = Math.min(rms.length, chroma.length, mfcc.length);
  if (nFrames < 1) return [];
  // Build the feature matrix as rows: 1 rms + 12 chroma + 13 mfcc, columns = frames.
  const rows = [];
  rows.push(rms.slice(0, nFrames));
  const nChroma = chroma[0] ? chroma[0].length : 0;
  for (let b = 0; b < nChroma; b++) rows.push(chroma.slice(0, nFrames).map((fr) => fr[b]));
  const nMfcc = mfcc[0] ? mfcc[0].length : 0;
  for (let c = 0; c < nMfcc; c++) rows.push(mfcc.slice(0, nFrames).map((fr) => fr[c]));

  const norm = normalizeColumns(rows);
  // novelty = sum over rows of |diff across columns|
  const diff = new Array(Math.max(0, nFrames - 1)).fill(0);
  for (let r = 0; r < norm.length; r++) {
    for (let j = 0; j < nFrames - 1; j++) diff[j] += Math.abs(norm[r][j + 1] - norm[r][j]);
  }
  if (diff.length < 40) return [];

  const kernelSize = Math.min(20, Math.floor(diff.length / 4));
  const smooth = kernelSize > 0 ? movingAverageSame(diff, kernelSize) : diff;
  const threshold = mean(smooth) + 1.0 * std(smooth);
  const minDistance = Math.max(40, Math.floor(nFrames / 10));
  return findPeaks(smooth, threshold, minDistance);
}

// numpy.convolve(diff, ones(k)/k, mode='same')
function movingAverageSame(x, k) {
  const n = x.length, out = new Array(n).fill(0);
  const half = Math.floor(k / 2);
  for (let i = 0; i < n; i++) {
    let s = 0;
    // 'same' centers the kernel: output[i] = mean of x[i-half .. i-half+k-1]
    for (let j = 0; j < k; j++) { const idx = i - half + j; if (idx >= 0 && idx < n) s += x[idx]; }
    out[i] = s / k;
  }
  return out;
}

// scipy.signal.find_peaks(values, height, distance): plateau-aware local maxima above
// height (flat tops report their midpoint), then thin by distance keeping the tallest —
// ties broken toward the higher index, matching scipy's reversed-argsort order.
function findPeaks(values, height, distance) {
  const n = values.length;
  const peaks = [];
  let i = 1;
  while (i < n - 1) {
    if (values[i] > values[i - 1]) {
      let k = i;
      while (k < n - 1 && values[k + 1] === values[i]) k++; // extend a flat top
      if (k < n - 1 && values[k + 1] < values[i]) {
        const mid = Math.floor((i + k) / 2);
        if (values[mid] >= height) peaks.push(mid);
      }
      i = k + 1;
    } else {
      i++;
    }
  }
  if (distance <= 1) return peaks;
  const byHeight = peaks.slice().sort((a, b) => (values[b] - values[a]) || (b - a));
  const keep = new Set(peaks);
  for (const p of byHeight) {
    if (!keep.has(p)) continue;
    for (const q of peaks) if (q !== p && keep.has(q) && Math.abs(q - p) < distance) keep.delete(q);
  }
  return peaks.filter((p) => keep.has(p));
}

function labelSections(boundaryTimes, rms, sr, hop) {
  const sections = [];
  const nSections = boundaryTimes.length - 1;
  const avgEnergy = mean(rms);
  for (let i = 0; i < nSections; i++) {
    const start = boundaryTimes[i], end = boundaryTimes[i + 1];
    // librosa.time_to_frames floors, not rounds
    const startFrame = Math.floor((start * sr) / hop);
    const endFrame = Math.min(Math.floor((end * sr) / hop), rms.length);
    const energy = endFrame > startFrame ? mean(rms.slice(startFrame, endFrame)) : 0;
    sections.push({
      label: guessLabel(i, nSections, energy, end - start, avgEnergy),
      start_time: round(start, 1),
      end_time: round(end, 1),
    });
  }
  return sections;
}

function guessLabel(index, total, energy, duration, avgEnergy) {
  if (index === 0 && (duration < 15 || energy < avgEnergy * 0.7)) return 'intro';
  if (index === total - 1 && (energy < avgEnergy * 0.7 || duration < 15)) return 'outro';
  if (energy > avgEnergy * 1.2) return 'chorus';
  if (energy < avgEnergy * 0.6 && index > 0 && index < total - 1) return 'bridge';
  return 'verse';
}

// ---------- orchestration (mirrors music_describer.analyze ANALYZERS order) ----------
const ANALYZERS = {
  rhythm: rhythmAnalyzer,
  harmony: harmonyAnalyzer,
  timbre: timbreAnalyzer,
  structure: structureAnalyzer,
  energy: energyAnalyzer,
};

export function runAnalysis(features, selected) {
  const names = selected && selected.length ? selected : Object.keys(ANALYZERS);
  const out = {};
  for (const n of names) if (ANALYZERS[n]) out[n] = ANALYZERS[n](features);
  return out;
}

// ---------- prompt (llm/prompt.py, verbatim) ----------
export const SYSTEM_PROMPT =
  'You are a music analysis assistant. Given structured audio analysis data, ' +
  'write a natural, musician-level description of the song.\n\n' +
  'Describe:\n' +
  '- Overall genre and style\n' +
  '- Mood and emotional character\n' +
  '- Key, scale/mode, and harmonic qualities\n' +
  '- Tempo and rhythmic feel (including time signature if notable)\n' +
  '- Instrumentation and tonal qualities\n' +
  '- Song structure and how it flows between sections\n' +
  '- Dynamic arc and notable energy moments\n\n' +
  'Write 2-3 paragraphs of natural prose. Be specific but accessible to musicians. ' +
  'Do not describe lyrics. Do not describe production or mixing details. ' +
  'Do not list the raw numbers — interpret them musically. ' +
  'Do not mention the analysis process or that you received structured data.';

export const FIELD_REFERENCE = {
  rhythm: {
    tempo: 'Beats per minute',
    time_signature: 'Meter (e.g. 4/4, 3/4, 7/8)',
    feel: 'Rhythmic character: straight, swung, driving, or laid-back',
    beat_strength: 'Average onset strength (higher = more percussive)',
  },
  harmony: {
    key: 'Musical key (e.g. C, F#, Bb)',
    scale: 'Major or minor',
    mode: 'Specific mode (ionian, dorian, phrygian, lydian, mixolydian, aeolian, locrian)',
    harmonic_complexity: 'simple / moderate / complex based on chroma entropy',
  },
  timbre: {
    brightness: '0-1 normalized spectral centroid (higher = brighter/trebly)',
    warmth: '1 - brightness',
    instrumentation_hints: 'Rough guesses about instruments present',
    has_vocals: 'Whether vocal-like formant patterns were detected',
    tonal_quality: 'bright / neutral / warm / dark',
  },
  structure: {
    sections: 'Detected sections with labels, start and end times',
    form_summary: 'Section flow as a string (e.g. intro -> verse -> chorus -> outro)',
  },
  energy: {
    overall_energy: 'Mean RMS energy (higher = louder overall)',
    dynamic_range: 'Difference between loudest and quietest moments',
    energy_arc: 'Overall trajectory: building, fading, flat, peaks in middle, etc.',
    notable_moments: 'Detected breakdowns, climaxes, and energy peaks with timestamps',
  },
};

export function buildUserPrompt(analysis) {
  return (
    'Here is the structured analysis of an audio track:\n\n' +
    JSON.stringify(analysis, null, 2) +
    '\n\nField reference (what each field means):\n\n' +
    JSON.stringify(FIELD_REFERENCE, null, 2) +
    '\n\nWrite a musician-level description of this track.'
  );
}
