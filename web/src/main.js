import '@fontsource-variable/space-grotesk';
import '@fontsource-variable/jetbrains-mono';
import { decodeToWaveform, extractFeatures } from './audio.js';
import { runAnalysis, SYSTEM_PROMPT, buildUserPrompt } from './analyze.js';
import { describe, DEFAULT_MODELS } from './llm.js';

const $ = (id) => document.getElementById(id);
const state = { key: '', provider: 'claude', model: DEFAULT_MODELS.claude, file: null };

let descText = '';
let analysisText = '';

// ---------- inputs ----------
$('model').value = state.model;
updateEngineTag();

$('key').oninput = (e) => { state.key = e.target.value.trim(); refresh(); };
$('provider').onchange = (e) => {
  state.provider = e.target.value;
  state.model = DEFAULT_MODELS[state.provider];
  $('model').value = state.model;
  updateEngineTag();
};
$('model').oninput = (e) => { state.model = e.target.value.trim(); updateEngineTag(); };

// ---------- file: click + drag-drop ----------
const drop = $('drop');
$('file').onchange = (e) => loadFile(e.target.files[0]);
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('drag'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('drag'); }));
drop.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) loadFile(f); });

const AUDIO_EXT = /\.(mp3|wav|flac|ogg|oga|m4a|aac|opus)$/i;
function loadFile(file) {
  if (!file || !(file.type.startsWith('audio/') || AUDIO_EXT.test(file.name))) return;
  state.file = file;
  const a = $('preview');
  a.src = URL.createObjectURL(file);
  a.hidden = false;
  $('drophint').hidden = true;
  setShot('live', 'loaded');
  refresh();
}

function refresh() { $('describe').disabled = !(state.key && state.file); }
function updateEngineTag() { $('enginetag').innerHTML = `engine <b>${state.provider} · ${escapeHtml(state.model)}</b>`; }

// ---------- status ----------
function setStatus(kind, text) {
  $('errslot').dataset.state = kind;
  $('errtext').textContent = text;
  const ps = $('pstatus');
  ps.className = 'status' + (kind === 'busy' ? ' busy' : kind === 'done' ? ' live' : '');
  ps.querySelector('.txt').textContent =
    kind === 'busy' ? 'working' : kind === 'done' ? 'done' : kind === 'err' ? 'error' : 'idle';
}
function setShot(cls, text) {
  const s = $('shotstatus');
  s.className = 'status' + (cls ? ' ' + cls : '');
  s.querySelector('.txt').textContent = text;
}

// ---------- pipeline ----------
$('describe').onclick = run;

async function run() {
  const btn = $('describe');
  btn.disabled = true; btn.classList.add('busy');
  try {
    setStatus('busy', 'decoding audio…');
    const { waveform, sr } = await decodeToWaveform(state.file);

    setStatus('busy', 'analyzing on your device…');
    setShot('busy', 'analyzing');
    const features = extractFeatures(waveform, sr);
    const analysis = runAnalysis(features);
    showAnalysis(analysis);
    showReadout(analysis);
    setShot('live', 'analyzed');

    setStatus('busy', `calling ${state.provider}…`);
    const text = await describe({
      provider: state.provider, model: state.model, key: state.key,
      systemPrompt: SYSTEM_PROMPT, userPrompt: buildUserPrompt(analysis),
    });
    showDescription(text);
    setStatus('done', 'done — description ready to copy');
  } catch (err) {
    setStatus('err', err.message || String(err));
  } finally {
    btn.classList.remove('busy');
    btn.disabled = !(state.key && state.file);
  }
}

// ---------- render ----------
function showDescription(text) {
  descText = text;
  const paras = text.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
  const wrap = $('blocks');
  wrap.replaceChildren();
  paras.forEach((p) => wrap.append(el('p', 'para', p)));
  $('promptcopy').disabled = false;
}

function showAnalysis(analysis) {
  analysisText = JSON.stringify(analysis, null, 2);
  $('analysis').textContent = analysisText;
  $('analysiscopy').disabled = false;
}

function showReadout(a) {
  const chips = [
    ['tempo', a.rhythm ? `${a.rhythm.tempo} BPM` : '—'],
    ['key', a.harmony ? `${a.harmony.key} ${a.harmony.scale}` : '—'],
    ['feel', a.rhythm?.feel || '—'],
    ['energy', a.energy?.energy_arc || '—'],
    ['voice', a.timbre ? (a.timbre.has_vocals ? 'vocals' : 'instrumental') : '—'],
  ];
  const out = $('readout');
  out.replaceChildren();
  for (const [k, v] of chips) {
    const chip = el('div', 'chip');
    chip.append(el('span', 'k', k), el('span', 'v', String(v)));
    out.append(chip);
  }
  out.hidden = false;
}

// ---------- copy ----------
wireCopy('promptcopy', () => descText);
wireCopy('analysiscopy', () => analysisText);

function wireCopy(id, getText) {
  const btn = $(id);
  const label = btn.querySelector('.lbl');
  const original = label.textContent;
  btn.addEventListener('click', async () => {
    const text = getText();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      btn.classList.add('ok'); label.textContent = 'copied';
      setTimeout(() => { btn.classList.remove('ok'); label.textContent = original; }, 1400);
    } catch {
      label.textContent = 'copy failed';
      setTimeout(() => { label.textContent = original; }, 1400);
    }
  });
}

// ---------- keyboard ----------
document.addEventListener('keydown', (e) => {
  if (e.target.matches('input, textarea, select')) return;
  if (e.key === 'd' && !$('describe').disabled) { e.preventDefault(); run(); }
  if (e.key === 'c' && descText) { e.preventDefault(); $('promptcopy').click(); }
});

// ---------- utils ----------
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ---------- github stars (graceful; button still works as a link if this fails) ----------
fetch('https://api.github.com/repos/lizergic/music-describer')
  .then((r) => (r.ok ? r.json() : null))
  .then((d) => { const node = document.getElementById('repoStars'); if (node && d && typeof d.stargazers_count === 'number') node.textContent = '★ ' + d.stargazers_count.toLocaleString(); })
  .catch(() => {});
