# HuggingFace Model Finder — Project Scope

A GitHub-hosted, client-side site for finding llama.cpp-compatible GGUF models on
Hugging Face by hardware fit, with copy-paste output for both standalone use and
llama.cpp's router mode.

## Goal

Given a user's system RAM, GPU VRAM, desired context length, and KV cache
precision, search Hugging Face live (no static list), show whether a model/quant
fits, and generate ready-to-use llama.cpp commands and config.

## Architecture

**Static site, no backend (pending one open item below).** Two data calls, both
made directly from the browser:

- `GET https://huggingface.co/api/models?search=...&filter=gguf&sort=...` for
  search, sort, and metadata (creator, file size, last-modified).
- A ranged `GET` against the `resolve/main/*.gguf` CDN URL to read just the GGUF
  header (a few KB) — gives architecture metadata (`n_layer`, `n_embd`, `n_head`,
  `n_head_kv`) without downloading the model. This is the same technique HF's own
  model-card UI uses to show quantization info.

**Open item — resolved.** Both calls work CORS-unauthenticated from a plain
static page, so no proxy is needed. The API reflects the requesting origin in
`access-control-allow-origin`; the `resolve/main/*.gguf` URL answers the
`Range` preflight with `access-control-allow-headers: range` and redirects to a
CDN that returns `206` with `access-control-allow-origin: *`. 4 KB of the file
was enough to reach the architecture metadata in every model tested.

**Scope boundary:** GGUF models only (what llama.cpp consumes). No other
formats/runtimes.

## Core Features

### 1. Search, filter, sort

Sort by creator, file size, recency — all directly available from the API
response. **Granularity is per file, not per repo** — a repo with ten quant
variants shows ten rows, each with its own fit indicator, so "Q4_K_M fits, Q8_0
doesn't" is visible at a glance. This also makes "sort by size" mean something
concrete (file size), rather than an ambiguous parameter-count label.

### 2. Fit indicators

**Inputs, all up front:** system RAM, GPU VRAM, desired context length, KV cache
precision (dropdown: `f16`, `q8_0`, `q4_0` — default `q8_0`; same value applied
to both K and V to avoid a known llama.cpp incompatibility between mismatched
cache types), and a reserved-VRAM field for OS/desktop overhead (default
1–1.5GB, guessed from OS — lower baseline for Linux, higher for Windows —
always user-editable, not worth modeling more precisely than that).

**Computation**, per quant file, from GGUF header metadata + file size:

```
weights per layer   ≈ file_size / n_layer
kv cache per token    = 2 × n_layer × n_head_kv × head_dim × bytes_per_elem(cache_type)
total kv cache         = kv cache per token × context_length

fits fully (green):   weights + total_kv_cache + ~10-15% headroom + reserved_OS_VRAM ≤ VRAM
offload gradient:      fraction of layers moved to CPU to bring the rest under VRAM,
                        0% (green) → 100% (red), plus a check that system RAM can
                        hold the offloaded layers
```

This is a simplified re-implementation of what llama.cpp's own `-ngl`/`--fit`
logic does internally — expect it to be a ballpark estimate, not exact (real
usage often needs a bit more headroom than the naive math shows). Label it as
an estimate in the UI.

**Flash-attention caveat:** quantized KV cache types (`q8_0`, `q4_0`) require
`--flash-attn on` to function at all — without it, a prebuilt binary lacking the
right CUDA kernels silently falls back to CPU attention (25-45x slower, no error
message). The generated command auto-appends `--flash-attn on` whenever cache
type ≠ `f16`. The UI additionally surfaces a caveat next to the KV precision
selector: "requires a build with flash-attention support" — since not every
prebuilt llama.cpp binary has the right kernels compiled in, and that's not
something the tool can detect from the browser.

### 3. Generated output

Two artifacts per model/quant, both fully computed from the fit numbers above:

**a. Single-model command** — doubles as both the download step and a standalone
launch. (llama.cpp has no dedicated "download only" flag; this command is
interrupted after the download finishes if the model is just being prefetched
for router use.)

```
llama-server -hf <org>/<repo>:<QUANT> -ngl <N> --ctx-size <C> \
  --cache-type-k <T> --cache-type-v <T> [--flash-attn on]
```

**b. Router `models.ini` section** — for llama.cpp's router mode
(`--models-preset`). Always fully explicit (no attempt to diff against a user's
own `[*]` global defaults), with comments explaining the computed values, not
just the field names:

```ini
[<org>/<repo>:<QUANT>]
alias        = <short-name>
; 50 of 64 layers on GPU, 14 offloaded to system RAM
; computed for 24GB VRAM (1.5GB reserved for OS), 8K context, q8_0 KV cache
n-gpu-layers = 50
cache-type-k = q8_0
cache-type-v = q8_0
```

Note: a `models.ini` section only overrides settings for a model already
present in the router's cache — it does not download anything itself. The
section name must exactly match the model ID the router derives from the
cache; a name that doesn't match creates a phantom entry rather than
overriding anything.

**Known limitation, accepted:** comments bake in the computed numbers at
generation time — a snapshot, not a live computation. If hardware or the
reserved-VRAM estimate changes later, the comment goes stale until the section
is regenerated.

### 4. Version detection (Phase 2 / stretch — deprioritized)

Flag when a newer version of a model may exist from the same org (e.g. Qwen
3.6→3.8). Hugging Face has no structural "supersedes" relationship for
independent releases — only for direct derivatives (finetunes, quants) of a
given repo. Approach would be regex-based: parse org + family name + version
token, compare against other same-org repos, and surface as "possible newer
version, verify manually" — never an authoritative claim, since a false
positive here actively misleads. Explicitly out of scope for v1; revisit once
real-world naming conventions are visible in actual search results.

## Suggested phasing

- **Phase 1:** search/sort/filter, fit indicators (file-level), both generated
  command outputs. The fit calculation is the hard part; both output formats
  are cheap once it exists.
- **Phase 2 (stretch):** version detection.

## Open items

- [x] Confirm HF API + ranged CDN fetch work CORS-unauthenticated from a static
      page (see Architecture) — the one assumption the whole "no backend" plan
      rests on. Confirmed; Phase 1 is built against it.
