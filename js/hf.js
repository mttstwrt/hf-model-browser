// Hugging Face API access. Everything here runs unauthenticated from the
// browser; both endpoints send permissive CORS headers.

import { parseHeader, shape, Truncated } from './gguf.js';

const API = 'https://huggingface.co/api';
// `tags` carries the base_model lineage; `gguf` carries the trained context
// length. Both let the list be filtered the moment it arrives, rather than
// waiting on a header read per repo. `gguf` also drags in each repo's chat
// template, which is most of the response weight — about 2 KB gzipped per
// model — so the caller keeps the page size in hand instead of always asking
// for the maximum.
const EXPAND = ['lastModified', 'downloads', 'likes', 'pipeline_tag', 'tags', 'gguf'];

async function json(url) {
  const r = await fetch(url);
  if (r.status === 429) throw new Error('Hugging Face rate limit reached — wait a minute and try again.');
  if (!r.ok) throw new Error(`Hugging Face returned ${r.status}`);
  return r.json();
}

export function search({ q, sort, limit = 20, skip = 0 }) {
  const p = new URLSearchParams({ filter: 'gguf', sort, direction: '-1', limit });
  if (skip) p.set('skip', skip);
  if (q) p.set('search', q);
  for (const f of EXPAND) p.append('expand[]', f);
  return json(`${API}/models?${p}`);
}

// Quant tag as llama.cpp's `-hf repo:TAG` expects it, e.g. Q4_K_M, IQ3_XS, UD-Q4_K_XL.
const QUANT = /(?:^|[-._])((?:UD-)?(?:IQ|Q)\d+(?:_[A-Za-z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?)(?=[-._]|$)/gi;
const SHARD = /-(\d{5})-of-(\d{5})$/;

function quantOf(stem) {
  const m = [...stem.matchAll(QUANT)];
  return m.length ? m[m.length - 1][1].toUpperCase() : null;
}

/** Collapse a repo's file tree into one entry per quant, summing sharded files. */
export function quants(tree) {
  const out = new Map();
  for (const f of tree) {
    if (f.type !== 'file' || !f.path.endsWith('.gguf')) continue;
    // Projectors, importance matrices and multi-token-prediction heads ship
    // alongside the model in the same repo, but are not models themselves.
    const base = f.path.split('/').pop();
    if (/imatrix/i.test(base) || /^(mmproj|mtp)[-_.]/i.test(base)) continue;
    const stem = f.path.slice(0, -5).replace(SHARD, '');
    const quant = quantOf(stem);
    const key = quant || stem;
    const e = out.get(key) || { quant, label: quant || stem, file: f.path, size: 0, parts: 0 };
    e.size += f.size;
    e.parts++;
    if (f.path < e.file) e.file = f.path;               // shard 00001 carries the header
    out.set(key, e);
  }
  return [...out.values()].sort((a, b) => a.size - b.size);
}

/**
 * Read architecture metadata from the start of a GGUF file over HTTP range
 * requests, growing the range only if the metadata block runs past it.
 * Resolves to null for a GGUF file that carries no architecture.
 */
async function readShape(repo, path) {
  const url = `https://huggingface.co/${repo}/resolve/main/${path.split('/').map(encodeURIComponent).join('/')}`;
  for (const bytes of [4096, 65536, 1048576]) {
    const r = await fetch(url, { headers: { Range: `bytes=0-${bytes - 1}` } });
    if (!r.ok) throw new Error(r.status === 401 || r.status === 403 ? 'the repo is gated' : `HTTP ${r.status}`);
    try {
      return shape(parseHeader(await r.arrayBuffer()));
    } catch (e) {
      if (!(e instanceof Truncated)) throw e;
    }
  }
  throw new Error('the header is unexpectedly large');
}

/**
 * Load one repo in two stages, mutating and re-announcing a single record:
 * the file list lands first so sizes can be shown right away, the architecture
 * follows. Returns that record synchronously.
 */
export function loadRepo(id, notify) {
  const rec = { loading: true };
  (async () => {
    try {
      rec.files = quants(await json(`${API}/models/${id}/tree/main?recursive=1`));
      if (!rec.files.length) rec.error = 'no GGUF files in this repo';
    } catch (e) {
      rec.error = e.message;
    }
    notify();

    // Every quant in a repo shares one architecture, so a single header read
    // covers them all. A ranged read costs the same whatever the file's size,
    // so start from the largest — that is the one most certainly the model
    // itself rather than a small sidecar sitting next to it.
    for (const f of (rec.files || []).slice(-3).reverse()) {
      try {
        if ((rec.shape = await readShape(id, f.file))) break;
      } catch (e) {
        rec.headerError = e.message;
        break;
      }
    }
    if (!rec.shape) rec.headerError ??= 'no architecture metadata in the file';
    rec.loading = false;
    notify();
  })();
  return rec;
}
