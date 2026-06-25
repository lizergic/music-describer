import test from 'node:test';
import assert from 'node:assert/strict';
import {
  rhythmAnalyzer, harmonyAnalyzer, timbreAnalyzer, structureAnalyzer, energyAnalyzer,
  runAnalysis, buildUserPrompt, SYSTEM_PROMPT,
} from '../src/analyze.js';

const SR = 22050, HOP = 512;
const MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

function features(over = {}) {
  const n = over.n ?? 60;
  return {
    sr: SR, hop: HOP, duration: (n * HOP) / SR,
    rms: over.rms ?? new Float32Array(n).fill(0.2),
    onset: over.onset ?? new Float32Array(n).fill(0.1),
    chroma: over.chroma ?? Array.from({ length: n }, () => MAJOR.slice()),
    mfcc: over.mfcc ?? Array.from({ length: n }, () => new Array(13).fill(0)),
    centroid: over.centroid ?? new Float32Array(n).fill(2000),
    rolloff: over.rolloff ?? new Float32Array(n).fill(5000),
  };
}

test('harmony: chroma matching the major profile reads as C major', () => {
  const out = harmonyAnalyzer(features({ chroma: Array.from({ length: 60 }, () => MAJOR.slice()) }));
  assert.equal(out.key, 'C');
  assert.equal(out.scale, 'major');
  assert.ok(['simple', 'moderate', 'complex'].includes(out.harmonic_complexity));
});

test('harmony: minor profile at root C reads as C minor', () => {
  const out = harmonyAnalyzer(features({ chroma: Array.from({ length: 60 }, () => MINOR.slice()) }));
  assert.equal(out.key, 'C');
  assert.equal(out.scale, 'minor');
});

test('energy: rising RMS across thirds reads as building', () => {
  const rms = new Float32Array(30);
  for (let i = 0; i < 10; i++) rms[i] = 0.1;
  for (let i = 10; i < 20; i++) rms[i] = 0.3;
  for (let i = 20; i < 30; i++) rms[i] = 0.6;
  const out = energyAnalyzer(features({ n: 30, rms }));
  assert.equal(out.energy_arc, 'building');
  assert.equal(out.overall_energy, 0.3333); // (1+3+6)/30 rounded to 4dp
  assert.equal(out.dynamic_range, 0.5);     // 0.6 - 0.1
});

test('timbre: bright centroid → bright, no formant variation → no vocals', () => {
  const out = timbreAnalyzer(features({
    centroid: new Float32Array(60).fill(0.4 * (SR / 2)),  // brightness 0.4
    rolloff: new Float32Array(60).fill(0.6 * (SR / 2)),
    mfcc: Array.from({ length: 60 }, () => new Array(13).fill(1)), // zero variance
  }));
  assert.equal(out.tonal_quality, 'bright');
  assert.equal(out.brightness, 0.4);
  assert.equal(out.has_vocals, false);
  assert.ok(out.instrumentation_hints.includes('bright synths or cymbals'));
});

test('rhythm: periodic onset spikes recover a plausible tempo + valid meter', () => {
  const onset = new Float32Array(660);
  for (let i = 0; i < 660; i += 22) onset[i] = 1; // ~117 BPM at sr/hop
  const out = rhythmAnalyzer(features({ n: 660, onset }));
  assert.ok(out.tempo > 108 && out.tempo < 126, `tempo ${out.tempo}`);
  assert.ok(['3/4', '4/4', '5/4', '7/8'].includes(out.time_signature));
  assert.ok(['straight', 'swung', 'driving', 'laid-back'].includes(out.feel));
});

test('structure: returns sections spanning 0..duration with a form summary', () => {
  const out = structureAnalyzer(features({ n: 50 }));
  assert.ok(Array.isArray(out.sections) && out.sections.length >= 1);
  assert.equal(out.sections[0].start_time, 0);
  const dur = features({ n: 50 }).duration;
  assert.equal(out.sections[out.sections.length - 1].end_time, Math.round(dur * 10) / 10); // end_time rounded to 0.1
  assert.equal(typeof out.form_summary, 'string');
});

test('runAnalysis returns the five analyzers in canonical order', () => {
  const out = runAnalysis(features());
  assert.deepEqual(Object.keys(out), ['rhythm', 'harmony', 'timbre', 'structure', 'energy']);
});

test('runAnalysis honors a selected subset', () => {
  const out = runAnalysis(features(), ['harmony', 'energy']);
  assert.deepEqual(Object.keys(out), ['harmony', 'energy']);
});

test('buildUserPrompt embeds the analysis JSON and field reference', () => {
  const p = buildUserPrompt({ rhythm: { tempo: 120 } });
  assert.ok(p.includes('"tempo": 120'));
  assert.ok(p.includes('Field reference'));
  assert.ok(SYSTEM_PROMPT.includes('music analysis assistant'));
});
