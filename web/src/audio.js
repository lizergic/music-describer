import Meyda from 'meyda';

// librosa.load defaults: mono, sr=22050. STFT defaults: n_fft=2048, hop=512, hann window.
const TARGET_SR = 22050;
const BUFFER = 2048;
const HOP = 512;

// Browser only: decode any audio file to a mono Float32 waveform resampled to 22050
// (OfflineAudioContext downmixes to mono and resamples — parity with librosa.load).
export async function decodeToWaveform(file) {
  const arrayBuf = await file.arrayBuffer();
  const AC = window.AudioContext || window.webkitAudioContext;
  const ac = new AC();
  let decoded;
  try {
    decoded = await ac.decodeAudioData(arrayBuf);
  } finally {
    if (ac.close) ac.close();
  }
  const frames = Math.max(1, Math.ceil(decoded.duration * TARGET_SR));
  const off = new OfflineAudioContext(1, frames, TARGET_SR);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return { waveform: rendered.getChannelData(0), sr: TARGET_SR };
}

// numpy 'reflect' padding (no edge repeat): map any index back into [0, n).
function reflectIndex(i, n) {
  if (n <= 1) return 0;
  const period = 2 * (n - 1);
  const m = ((i % period) + period) % period;
  return m < n ? m : period - m;
}

// librosa.feature.spectral_rolloff: lowest freq below which roll_percent of the magnitude
// energy lies. Meyda hardcodes 0.99 with no knob, so we compute it at librosa's 0.85.
function rolloffHz(amp, binToHz, rollPercent) {
  if (!amp || !amp.length) return 0;
  let total = 0;
  for (let k = 0; k < amp.length; k++) total += amp[k];
  const threshold = rollPercent * total;
  let cum = 0;
  for (let k = 0; k < amp.length; k++) { cum += amp[k]; if (cum >= threshold) return k * binToHz; }
  return (amp.length - 1) * binToHz;
}

// Isomorphic (no DOM): frame the waveform and run Meyda per frame, producing the
// `features` object consumed by analyze.js. Meyda is pure JS, so this runs under node too.
export function extractFeatures(waveform, sr) {
  Meyda.bufferSize = BUFFER;
  Meyda.sampleRate = sr;
  Meyda.numberOfMFCCCoefficients = 13;
  Meyda.windowingFunction = 'hanning';

  // librosa default center=True: reflect-pad by n_fft/2 so frame i is centered at i*hop,
  // matching librosa frame counts and timestamps.
  const len = waveform.length;
  const pad = BUFFER / 2;
  const padded = new Float32Array(len + 2 * pad);
  for (let i = 0; i < padded.length; i++) padded[i] = waveform[reflectIndex(i - pad, len)];

  const want = ['spectralCentroid', 'mfcc', 'chroma', 'amplitudeSpectrum'];
  const binToHz = sr / BUFFER; // Meyda centroid is a spectrum-bin index, not Hz
  const rms = [], onset = [], chroma = [], mfcc = [], centroid = [], rolloff = [];
  let prevSpec = null;

  const nFrames = len > 0 ? Math.floor(len / HOP) + 1 : 0;
  for (let i = 0; i < nFrames; i++) {
    const start = i * HOP;
    const frame = padded.subarray(start, start + BUFFER);
    const r = Meyda.extract(want, frame);
    // RMS on the raw (un-windowed) frame, matching librosa.feature.rms
    let sq = 0;
    for (let k = 0; k < BUFFER; k++) sq += frame[k] * frame[k];
    rms.push(Math.sqrt(sq / BUFFER));
    centroid.push((r.spectralCentroid || 0) * binToHz);
    rolloff.push(rolloffHz(r.amplitudeSpectrum, binToHz, 0.85));
    mfcc.push(r.mfcc);
    chroma.push(r.chroma);
    // onset strength ≈ summed positive spectral flux (librosa.onset.onset_strength proxy)
    const spec = r.amplitudeSpectrum;
    let flux = 0;
    if (prevSpec) for (let k = 0; k < spec.length; k++) { const d = spec[k] - prevSpec[k]; if (d > 0) flux += d; }
    onset.push(flux);
    prevSpec = spec;
  }

  return {
    sr, hop: HOP, duration: len / sr,
    rms: Float32Array.from(rms),
    onset: Float32Array.from(onset),
    chroma, mfcc,
    centroid: Float32Array.from(centroid),
    rolloff: Float32Array.from(rolloff),
  };
}
