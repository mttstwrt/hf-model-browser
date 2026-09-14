import { search, loadRepo } from './hf.js';
import { estimate, command, iniSection, STATUS, fitColor, GB } from './fit.js';

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);

/* ---------- formatting ---------- */
const fmt = (b) => {
  const g = b / GB;
  return (g >= 100 ? g.toFixed(0) : g >= 10 ? g.toFixed(1) : g.toFixed(2)) + ' GB';
};
const fmtShort = (b) =>
  b < GB ? Math.round(b / 2 ** 20) + ' MB' : (b / GB >= 10 ? (b / GB).toFixed(0) : (b / GB).toFixed(1)) + ' GB';
const count = (n) =>
  n >= 1e6 ? (n / 1e6).toFixed(1).replace(/\.0$/, '') + 'M' : n >= 1e3 ? Math.round(n / 1e3) + 'k' : String(n ?? 0);
const ago = (iso) => {
  const days = (Date.now() - new Date(iso)) / 864e5;
  if (days < 1) return 'today';
  for (const [n, unit] of [[365, 'y'], [30, 'mo'], [7, 'w'], [1, 'd']])
    if (days >= n) return `${Math.floor(days / n)}${unit} ago`;
};
const tokens = (n) => (n >= 1024 ? Math.round(n / 1024) + 'K' : n);

/* ---------- hardware ---------- */
const DEFAULTS = {
  vram: 8, ram: 16, ctx: 8192, cacheType: 'q8_0',
  // Desktop compositors hold onto rather more VRAM on Windows than elsewhere.
  reserved: /Win/i.test(navigator.userAgent) ? 1.5 : 1,
};
let hw = { ...DEFAULTS, ...JSON.parse(localStorage.getItem('hw') || '{}') };
const inBytes = () => ({ ...hw, vram: hw.vram * GB, ram: hw.ram * GB, reserved: hw.reserved * GB });

const FIELDS = { vram: 'vram', ram: 'ram', ctx: 'ctx', reserved: 'reserved', cache: 'cacheType' };

function readHardware() {
  for (const [id, key] of Object.entries(FIELDS)) {
    const el = $('#' + id);
    const v = el.type === 'number' ? parseFloat(el.value) : el.value;
    if (el.type !== 'number' || Number.isFinite(v)) hw[key] = v;
  }
  localStorage.setItem('hw', JSON.stringify(hw));
  $('#fa-caveat').hidden = hw.cacheType === 'f16';
  schedule();
}

function showHardware() {
  for (const [id, key] of Object.entries(FIELDS)) $('#' + id).value = hw[key];
  $('#fa-caveat').hidden = hw.cacheType === 'f16';
}

/* ---------- state ---------- */
const state = { q: '', sort: 'trendingScore', fitsOnly: false, models: [], limit: 20, open: null, status: null };
const repos = new Map();         // model id -> live record from loadRepo
let focusAfterRender = null;

/** Start loading a repo's files and architecture, re-rendering as each lands. */
function ensure(id) {
  if (!repos.has(id)) repos.set(id, loadRepo(id, schedule));
  return repos.get(id);
}

/** Per-quant fit rows for a repo, or null until its file list has arrived. */
function rowsFor(id) {
  const d = repos.get(id);
  if (!d?.files) return null;
  const b = inBytes();
  return d.files.map((q) => ({ q, fit: d.shape ? estimate(q.size, d.shape, b) : null }));
}

/* ---------- search ---------- */
// Our own orderings reshuffle whatever the query returned, so they all share
// one server-side ordering.
const serverSort = (sort) => (sort.startsWith('~') ? 'downloads' : sort);

let requestId = 0;
async function runSearch() {
  const mine = ++requestId;
  state.status = 'Searching…';
  render();
  try {
    const models = await search({ q: state.q, sort: serverSort(state.sort), limit: state.limit });
    if (mine !== requestId) return;
    state.models = models;
    state.status = models.length ? null : 'No GGUF models match that search.';
  } catch (e) {
    if (mine !== requestId) return;
    state.status = { error: e.message };
  }
  render();
}

/* ---------- rendering ---------- */
// Coalesce the burst of re-renders as repo data lands. A timer rather than an
// animation frame, so a backgrounded tab still catches up on its results.
let pending = 0;
const schedule = () => { clearTimeout(pending); pending = setTimeout(render, 30); };

function ordered() {
  const list = [...state.models];
  const byFit = (m) => {
    const rows = rowsFor(m.id);
    if (!rows) return -1;                                   // still loading: park at the end
    const full = rows.filter((r) => r.fit?.status === 'full');
    return full.length ? Math.max(...full.map((r) => r.q.size)) : 0;
  };
  if (state.sort === '~fit') list.sort((a, b) => byFit(b) - byFit(a));
  if (state.sort === '~size') list.sort((a, b) => (rowsFor(a.id)?.[0]?.q.size ?? Infinity) - (rowsFor(b.id)?.[0]?.q.size ?? Infinity));
  if (state.sort === '~creator') list.sort((a, b) => a.id.toLowerCase().localeCompare(b.id.toLowerCase()));
  if (!state.fitsOnly) return list;
  return list.filter((m) => {
    const rows = rowsFor(m.id);
    return !rows || rows.some((r) => r.fit?.status === 'full');
  });
}

function render() {
  const el = $('#results');
  if (state.status) {
    const e = state.status.error;
    el.innerHTML = `<p class="status${e ? ' err' : ''}">${esc(e || state.status)}</p>`;
    $('#more').hidden = true;
    return;
  }
  // Client-side ordering needs every repo measured, not just the visible ones.
  if (state.sort.startsWith('~') || state.fitsOnly) state.models.forEach((m) => ensure(m.id));

  const list = ordered();
  el.innerHTML = list.length ? list.map(card).join('') : `<p class="status">Nothing here fits your GPU. Try a lower context length, a coarser KV cache, or turn the filter off.</p>`;
  $('#more').hidden = state.models.length < state.limit;

  for (const node of el.querySelectorAll('[data-id]')) if (!repos.has(node.dataset.id)) observer.observe(node);
  if (focusAfterRender) {
    el.querySelector(`[data-key="${CSS.escape(focusAfterRender)}"]`)?.focus({ preventScroll: true });
    focusAfterRender = null;
    // A card's panel opens below its quants, which can leave it off-screen.
    const panel = el.querySelector('.detail');
    if (panel && panel.getBoundingClientRect().bottom > innerHeight)
      panel.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
}

// Only fetch a repo's file list once its card is near the viewport.
const observer = new IntersectionObserver(
  (entries) => entries.forEach((e) => { if (e.isIntersecting) { observer.unobserve(e.target); ensure(e.target.dataset.id); } }),
  { rootMargin: '400px' },
);

function card(m) {
  const [org, name] = m.id.split('/');
  const d = repos.get(m.id);
  const rows = rowsFor(m.id);
  const stats = [
    m.downloads != null ? `${count(m.downloads)} downloads` : '',
    m.likes ? `${count(m.likes)} likes` : '',
    m.lastModified ? ago(m.lastModified) : '',
  ].filter(Boolean).join(' · ');

  let body;
  if (!d || (d.loading && !rows)) body = '<div class="skeleton"></div>';
  else if (!rows?.length) body = `<p class="note err">${esc(d.error || 'no GGUF files in this repo')}</p>`;
  else body = `<div class="quants">${rows.map((r) => chip(m.id, r)).join('')}</div>`
    + (d.loading ? `<p class="note">Reading the model’s architecture…</p>`
      : d.headerError ? `<p class="note err">No fit estimate: ${esc(d.headerError)}. Sizes are still accurate.</p>` : '');

  const open = rows?.find((r) => key(m.id, r) === state.open);

  return `<article class="card" data-id="${esc(m.id)}">
    <div class="card-head">
      <h2><a href="https://huggingface.co/${esc(m.id)}" target="_blank" rel="noopener"><span class="org">${esc(org)}/</span>${esc(name)}</a></h2>
      <div class="stats">${esc(stats)}</div>
    </div>
    ${d?.shape ? `<div class="arch"><span class="tag">${esc(d.shape.arch)}</span> · ${d.shape.nLayer} layers · ${Math.round(d.shape.nHeadKv)} KV heads${d.shape.trainCtx ? ` · ${tokens(d.shape.trainCtx)} trained context` : ''}</div>` : ''}
    ${body}
    ${open ? detail(m.id, open) : ''}
  </article>`;
}

const key = (id, r) => `${id}::${r.q.label}`;

function chip(id, r) {
  const { q, fit } = r;
  const on = key(id, r) === state.open;
  const s = fit ? STATUS[fit.status] : { icon: '?', word: 'Fit unknown' };
  return `<button class="chip${fit ? '' : ' unknown'}" data-key="${esc(key(id, r))}" aria-expanded="${on}"
    style="--fit:${fit ? fitColor(fit) : 'var(--muted)'}" title="${esc(s.word)}">
    <span class="dot" aria-hidden="true">${s.icon}</span><span class="name">${esc(q.label)}</span>
    <span class="size">${fmtShort(q.size)}${q.parts > 1 ? ` · ${q.parts} parts` : ''}</span>
  </button>`;
}

function detail(id, r) {
  const { q, fit } = r;
  if (!fit) return `<div class="detail"><p class="note">No architecture metadata for this repo, so there is no fit estimate — only the ${fmt(q.size)} download size.</p></div>`;

  const warnings = [];
  if (fit.status === 'no')
    warnings.push(['bad', `Needs about ${fmt(fit.onCpu)} of system RAM for the layers that don’t fit in VRAM, and you have ${fmt(hw.ram * GB)}. Expect an out-of-memory failure or heavy swapping.`]);
  else if (fit.status === 'partial')
    warnings.push(['', `${fit.nLayer - fit.ngl} of ${fit.nLayer} layers run on the CPU. Generation will be several times slower than a full GPU fit.`]);
  else if (fit.status === 'cpu')
    warnings.push(['', `Nothing fits in VRAM at this context length, so the whole model runs on the CPU.`]);
  if (fit.overCtx)
    warnings.push(['', `${hw.ctx} tokens is beyond this model’s trained context — llama.cpp will need RoPE scaling, and quality usually drops past the trained length.`]);

  const bytes = inBytes();
  const vramPct = (b) => Math.min(100, (b / (hw.vram * GB || 1)) * 100);

  return `<div class="detail" style="--fit:${fitColor(fit)}">
    <p class="verdict"><span class="dot" aria-hidden="true">${STATUS[fit.status].icon}</span>
      ${STATUS[fit.status].word}
      <span class="sub">— ${fit.ngl} of ${fit.nLayer} layers</span></p>

    ${hw.vram > 0 ? `<div class="bar">
      <i style="width:${vramPct(fit.onGpu)}%"></i>
      <u style="width:${vramPct(hw.reserved * GB)}%"></u>
    </div>
    <p class="bar-label"><b>${fmt(fit.onGpu)}</b> of ${fmt(hw.vram * GB)} VRAM
      — ${fmt(fit.weights * fit.gpuFraction)} weights, ${fmt(fit.kv * fit.gpuFraction)} KV cache, plus 12% headroom
      (${fmt(hw.reserved * GB)} reserved for the OS)</p>` : ''}

    ${fit.onCpu > 0 ? `<div class="bar" style="--fill:var(--muted)">
      <i style="width:${Math.min(100, (fit.onCpu / (hw.ram * GB || 1)) * 100)}%"></i></div>
      <p class="bar-label"><b>${fmt(fit.onCpu)}</b> of ${fmt(hw.ram * GB)} system RAM for the remaining ${fit.nLayer - fit.ngl} layers</p>` : ''}

    ${warnings.map(([cls, text]) => `<p class="warn ${cls}">${esc(text)}</p>`).join('')}

    ${block('Download and run', command(id, q, fit, bytes), 'llama.cpp has no download-only flag: this fetches the model if it isn’t cached, then serves it. Interrupt it once the download finishes if you only want the file on disk for router use.')}
    ${block('models.ini section', iniSection(id, q, fit, bytes, fmt), 'For llama.cpp’s router (--models-preset). It only overrides settings for a model already in the router’s cache — it downloads nothing, and a section name that doesn’t match the cached model id creates a phantom entry instead of overriding one. The numbers in the comments are a snapshot: regenerate the section if your hardware settings change.')}
  </div>`;
}

const block = (title, text, note) => `<div class="out">
  <div class="out-head"><h3>${esc(title)}</h3><button class="copy">Copy</button><p>${esc(note)}</p></div>
  <pre>${esc(text).replace(/^;.*/gm, (c) => `<span class="c">${c}</span>`)}</pre>
</div>`;

/* ---------- events ---------- */
$('#hardware').addEventListener('input', readHardware);
$('#hardware').addEventListener('submit', (e) => e.preventDefault());

let typing;
$('#q').addEventListener('input', (e) => {
  state.q = e.target.value.trim();
  clearTimeout(typing);
  typing = setTimeout(runSearch, 320);
});
$('#sort').addEventListener('change', (e) => {
  const before = serverSort(state.sort);
  state.sort = e.target.value;
  serverSort(state.sort) === before ? render() : runSearch();
});
$('#fits-only').addEventListener('change', (e) => { state.fitsOnly = e.target.checked; render(); });
$('#more').addEventListener('click', () => { state.limit += 20; runSearch(); });

$('#results').addEventListener('click', (e) => {
  const chip = e.target.closest('.chip');
  if (chip) {
    state.open = state.open === chip.dataset.key ? null : chip.dataset.key;
    focusAfterRender = chip.dataset.key;
    return render();
  }
  const copy = e.target.closest('.copy');
  if (copy) {
    navigator.clipboard.writeText(copy.closest('.out').querySelector('pre').textContent).then(() => {
      copy.textContent = 'Copied';
      copy.classList.add('done');
      setTimeout(() => { copy.textContent = 'Copy'; copy.classList.remove('done'); }, 1400);
    }, () => {
      // Clipboard access can be refused; select the text so it can still be copied.
      getSelection().selectAllChildren(copy.closest('.out').querySelector('pre'));
      copy.textContent = 'Selected — copy it';
    });
  }
});

showHardware();
runSearch();
