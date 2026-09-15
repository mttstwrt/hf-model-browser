import { search, loadRepo } from './hf.js';
import { estimate, command, iniSection, STATUS, fitColor, GB } from './fit.js';
import { lineage, TIERS } from './lineage.js';

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
  vram: 8, ram: 16, ctx: 8192, cacheType: 'q8_0', unified: false,
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
  const wasUnified = hw.unified;
  hw.unified = $('#unified').checked;
  // Switching to one pool changes what the VRAM box means. A discrete card's
  // VRAM is rarely a sane wiring limit for a whole machine, so on that one
  // transition seed it with the ~75% macOS allows — unless the value already
  // looks like a share of the pool, in which case it was chosen deliberately.
  if (hw.unified && !wasUnified && hw.vram < hw.ram * 0.6) {
    hw.vram = Math.round(hw.ram * 0.75 * 2) / 2;
    $('#vram').value = hw.vram;
  }
  localStorage.setItem('hw', JSON.stringify(hw));
  syncHardwareUi();
  schedule();
}

function showHardware() {
  for (const [id, key] of Object.entries(FIELDS)) $('#' + id).value = hw[key];
  $('#unified').checked = hw.unified;
  syncHardwareUi();
}

/** Labels and notes that depend on the hardware values rather than the results. */
function syncHardwareUi() {
  $('#fa-caveat').hidden = hw.cacheType === 'f16';
  $('#unified-note').hidden = !hw.unified;
  $('#vram-label').textContent = hw.unified ? 'GPU limit' : 'GPU VRAM';
  $('#ram-label').textContent = hw.unified ? 'Total memory' : 'System RAM';
  $('#ctx-echo').textContent = tokens(hw.ctx);
}

/* ---------- state ---------- */
const state = {
  q: '', sort: 'trendingScore',
  lineage: 'any', minDownloads: 0, ctxOnly: false, fitsOnly: false,
  models: [],                    // the pool the API returned, before filtering
  shown: 20,                     // how many survivors to put on the page
  open: null, status: null,
};
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

/* ---------- filtering ---------- */
// Filters the search response can answer on its own, without measuring a repo.
const cheapFilters = () => state.lineage !== 'any' || state.minDownloads > 0 || state.ctxOnly;

const lineageOf = (m) => (m._lineage ??= lineage(m));

/**
 * The context the model was trained for. The GGUF header is authoritative but
 * arrives only once a card has been measured; the search response carries Hugging
 * Face's own reading of the repo, which is there immediately and right nearly
 * always — it misses only when the file it sampled was a projector rather than
 * the model. Undefined when neither knows.
 */
const trainedCtx = (m) => repos.get(m.id)?.shape?.trainCtx || m.gguf?.context_length || null;

function passesFilters(m) {
  if (state.minDownloads > 0 && (m.downloads ?? 0) < state.minDownloads) return false;
  const tiers = TIERS[state.lineage];
  if (tiers && !tiers.has(lineageOf(m).tier)) return false;
  // An unknown trained context is let through rather than hidden: it usually
  // means Hugging Face sampled a sidecar file, not that the model is short.
  if (state.ctxOnly) {
    const t = trainedCtx(m);
    if (t && t < hw.ctx) return false;
  }
  return true;
}

/* ---------- search ---------- */
// Our own orderings reshuffle whatever the query returned, so they all share
// one server-side ordering.
const serverSort = (sort) => (sort.startsWith('~') ? 'downloads' : sort);

// Asking for `gguf` metadata costs roughly 2 KB gzipped per model, so the first
// page stays small enough to paint quickly. Only a filter strict enough to eat
// a whole page sends us back, and then round trips cost more than bytes do.
const PAGE = () => (state.models.length ? 100 : 30);

let requestId = 0;

/** Append the next page to the pool. Returns false if the request was superseded. */
async function fetchPage(mine) {
  const want = PAGE();
  try {
    const got = await search({ q: state.q, sort: serverSort(state.sort), limit: want, skip: state.models.length });
    if (mine !== requestId) return false;
    state.models = state.models.concat(got);
    state.exhausted = got.length < want;
    state.status = state.models.length ? null : 'No GGUF models match that search.';
    return true;
  } catch (e) {
    if (mine === requestId) state.status = { error: e.message };
    return false;
  }
}

// How deep to walk unprompted. "Official releases with a million downloads"
// matches on the order of one result in fifty, and without a ceiling a pair of
// filter changes will quietly pull a couple of megabytes chasing a full page.
// The bound is on the pool rather than on iterations, so it holds however many
// times a filter is toggled; asking for more is what the Load more button does.
const CEILING = 400;
const REACHED_CEILING = () => !state.exhausted && state.models.length >= state.ceiling;

/**
 * Filtering happens here rather than at the API, so a strict filter can leave a
 * page with two survivors out of a hundred. Keep pulling until the page is full,
 * the results run out, or the ceiling is hit.
 */
async function fill(mine) {
  while (mine === requestId && !state.exhausted && state.models.length < state.ceiling) {
    if (state.models.filter(passesFilters).length >= state.shown) return;
    render();                                   // show what is in hand meanwhile
    if (!await fetchPage(mine)) return;
  }
}

async function runSearch() {
  const mine = ++requestId;
  state.models = [];
  state.exhausted = false;
  state.ceiling = CEILING;
  state.status = 'Searching…';
  render();
  if (await fetchPage(mine)) await fill(mine);
  if (mine === requestId) render();
}

/** Grow the page in place, without discarding what has already been measured. */
async function loadMore() {
  const mine = requestId;
  state.shown += 20;
  state.ceiling = state.models.length + CEILING;
  if (!state.exhausted) await fetchPage(mine);
  await fill(mine);
  if (mine === requestId) render();
}

/* ---------- rendering ---------- */
// Coalesce the burst of re-renders as repo data lands. A timer rather than an
// animation frame, so a backgrounded tab still catches up on its results.
let pending = 0;
const schedule = () => { clearTimeout(pending); pending = setTimeout(render, 30); };

function ordered() {
  const eligible = state.models.filter(passesFilters);

  // Ordering by fit, and filtering by it, both need every candidate measured —
  // two requests apiece. Bound that to a window a little deeper than the page
  // being shown, instead of measuring the whole pool.
  const needsRepos = state.sort === '~fit' || state.sort === '~size' || state.fitsOnly;
  let list = eligible;
  if (needsRepos) {
    list = eligible.slice(0, state.shown * 2);
    list.forEach((m) => ensure(m.id));
  }

  const byFit = (m) => {
    const rows = rowsFor(m.id);
    if (!rows) return -1;                                   // still loading: park at the end
    const full = rows.filter((r) => r.fit?.status === 'full');
    return full.length ? Math.max(...full.map((r) => r.q.size)) : 0;
  };
  if (state.sort === '~fit') list.sort((a, b) => byFit(b) - byFit(a));
  if (state.sort === '~size') list.sort((a, b) => (rowsFor(a.id)?.[0]?.q.size ?? Infinity) - (rowsFor(b.id)?.[0]?.q.size ?? Infinity));
  if (state.sort === '~creator') list.sort((a, b) => a.id.toLowerCase().localeCompare(b.id.toLowerCase()));
  if (state.fitsOnly) list = list.filter((m) => {
    const rows = rowsFor(m.id);
    return !rows || rows.some((r) => r.fit?.status === 'full');
  });

  return { list: list.slice(0, state.shown), matched: list.length };
}

/** "12 of 80 searched" — so a thin page reads as a strict filter, not an empty index. */
function showCount(matched) {
  const el = $('#count');
  const pool = state.models.length;
  const trimmed = pool - matched;
  el.hidden = !pool || (!cheapFilters() && !state.fitsOnly);
  el.textContent = trimmed <= 0
    ? `all ${pool} searched match`
    : `${matched} of ${pool} searched match — ${trimmed} filtered out`
      + (REACHED_CEILING() ? '; stopped looking here' : '');
}

function render() {
  const el = $('#results');
  if (state.status) {
    const e = state.status.error;
    el.innerHTML = `<p class="status${e ? ' err' : ''}">${esc(e || state.status)}</p>`;
    $('#more').hidden = true;
    $('#count').hidden = true;
    return;
  }

  const { list, matched } = ordered();
  showCount(matched);
  el.innerHTML = list.length ? list.map(card).join('') : `<p class="status">Nothing here matches. Loosen the source filter, lower the download or context thresholds, or try a coarser KV cache.</p>`;
  $('#more').hidden = !!state.exhausted;

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

  const lin = lineageOf(m);
  const badge = lin.tier === 'official' ? `<span class="badge ok" title="${esc(lin.why)}">official</span>`
    : lin.tier === 'derivative' ? `<span class="badge alt" title="${esc(lin.why)}">finetune</span>` : '';

  // The architecture numbers wait on the header read, but the trained context
  // comes with the search, so the line is worth drawing before the rest lands.
  const tctx = trainedCtx(m);
  const short = tctx && tctx < hw.ctx;
  const arch = [
    d?.shape ? `<span class="tag">${esc(d.shape.arch)}</span>` : m.gguf?.architecture ? `<span class="tag">${esc(m.gguf.architecture)}</span>` : '',
    d?.shape ? `${d.shape.nLayer} layers` : '',
    d?.shape ? `${Math.round(d.shape.nHeadKv)} KV heads` : '',
    tctx ? `<span class="${short ? 'short' : ''}">${tokens(tctx)} trained context${short ? ` — short of your ${tokens(hw.ctx)}` : ''}</span>`
      // Kept in a filtered list rather than dropped: a missing context usually
      // means the repo's metadata was read off a projector, not that the model
      // is short. Saying so beats letting it sit there looking unexplained.
      : `<span class="unknown-ctx">trained context unknown</span>`,
  ].filter(Boolean);

  return `<article class="card" data-id="${esc(m.id)}">
    <div class="card-head">
      <h2><a href="https://huggingface.co/${esc(m.id)}" target="_blank" rel="noopener"><span class="org">${esc(org)}/</span>${esc(name)}</a>${badge}</h2>
      <div class="stats">${esc(stats)}</div>
    </div>
    ${arch.length ? `<div class="arch">${arch.join(' · ')}</div>` : ''}
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
    warnings.push(['bad', hw.unified
      ? `Needs about ${fmt(fit.onGpu + fit.onCpu)} in total and only ${fmt(fit.pool)} of your ${fmt(hw.ram * GB)} is usable once the OS reserve is taken out. Expect an out-of-memory failure or heavy swapping.`
      : `Needs about ${fmt(fit.onCpu)} of system RAM for the layers that don’t fit in VRAM, and you have ${fmt(hw.ram * GB)}. Expect an out-of-memory failure or heavy swapping.`]);
  else if (fit.status === 'partial')
    warnings.push(['', `${fit.nLayer - fit.ngl} of ${fit.nLayer} layers run on the CPU. Generation will be several times slower than a full GPU fit.`]);
  else if (fit.status === 'cpu')
    warnings.push(['', `Nothing fits in VRAM at this context length, so the whole model runs on the CPU.`]);
  if (fit.overCtx)
    warnings.push(['', `${hw.ctx} tokens is beyond this model’s trained context — llama.cpp will need RoPE scaling, and quality usually drops past the trained length.`]);

  const bytes = inBytes();
  const pct = (b, of) => Math.min(100, (b / (of || 1)) * 100);
  const vramPct = (b) => pct(b, hw.vram * GB);

  // Two pools or one. On unified memory the split between GPU and CPU is a
  // matter of which layers are wired down, not of which chip's memory they sit
  // in, so drawing two bars would invite adding them up as separate budgets.
  const memory = fit.unified
    ? `<div class="bar">
        <i style="width:${pct(fit.onGpu + fit.onCpu, fit.pool)}%"></i>
      </div>
      <p class="bar-label"><b>${fmt(fit.onGpu + fit.onCpu)}</b> of ${fmt(fit.pool)} usable unified memory
        — ${fmt(fit.weights)} weights, ${fmt(fit.kv)} KV cache, plus 12% headroom.
        ${fit.ngl > 0 ? `${fmt(fit.onGpu)} of it wired to the GPU` : 'None of it wired to the GPU'}${fit.onCpu > 0 ? `, ${fmt(fit.onCpu)} left to the CPU` : ''}
        (GPU capped at ${fmt(hw.vram * GB)}, ${fmt(hw.reserved * GB)} reserved for the OS out of ${fmt(hw.ram * GB)} total)</p>`
    : `${hw.vram > 0 ? `<div class="bar">
        <i style="width:${vramPct(fit.onGpu)}%"></i>
        <u style="width:${vramPct(hw.reserved * GB)}%"></u>
      </div>
      <p class="bar-label"><b>${fmt(fit.onGpu)}</b> of ${fmt(hw.vram * GB)} VRAM
        — ${fmt(fit.weights * fit.gpuFraction)} weights, ${fmt(fit.kv * fit.gpuFraction)} KV cache, plus 12% headroom
        (${fmt(hw.reserved * GB)} reserved for the OS)</p>` : ''}

      ${fit.onCpu > 0 ? `<div class="bar" style="--fill:var(--muted)">
        <i style="width:${pct(fit.onCpu, hw.ram * GB)}%"></i></div>
        <p class="bar-label"><b>${fmt(fit.onCpu)}</b> of ${fmt(hw.ram * GB)} system RAM for the remaining ${fit.nLayer - fit.ngl} layers</p>` : ''}`;

  return `<div class="detail" style="--fit:${fitColor(fit)}">
    <p class="verdict"><span class="dot" aria-hidden="true">${STATUS[fit.status].icon}</span>
      ${STATUS[fit.status].word}
      <span class="sub">— ${fit.ngl} of ${fit.nLayer} layers</span></p>

    ${memory}

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
$('#ctx-only').addEventListener('change', (e) => { state.ctxOnly = e.target.checked; refilter(); });
$('#lineage').addEventListener('change', (e) => { state.lineage = e.target.value; refilter(); });
// Its own timer, not the search box's: typing in one must not cancel the other's
// pending work.
let dlTyping;
$('#min-downloads').addEventListener('input', (e) => {
  state.minDownloads = Math.max(0, parseFloat(e.target.value) || 0);
  clearTimeout(dlTyping);
  dlTyping = setTimeout(refilter, 320);
});
$('#more').addEventListener('click', loadMore);

/** Re-render on what is already loaded, then top the page up if it came out thin. */
function refilter() {
  render();
  fill(requestId).then(() => render());
}

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
