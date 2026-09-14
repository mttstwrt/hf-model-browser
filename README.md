# GGUF Model Finder

A static, client-side page for finding llama.cpp-compatible GGUF models on
Hugging Face by whether they fit your hardware — and for generating the command
to run them.

Enter your VRAM, system RAM, context length and KV cache precision once. Every
quant of every search result is then shown with a fit indicator: green when the
whole model fits in VRAM, amber through red as layers spill onto the CPU, and a
cross when it will not fit at all. Click a quant for the arithmetic behind that
verdict, a `llama-server` command, and a `models.ini` section for llama.cpp's
router.

## Running it

There is no build step and no dependencies. Serve the directory over HTTP — the
page uses ES modules, so `file://` will not work:

```
python3 -m http.server 8000
```

To publish, push the repository and turn on GitHub Pages for the branch root.

## How it works

Two calls, both made straight from the browser with no API key and no backend:

- `GET /api/models?filter=gguf&search=…` for search, sort and repo metadata, and
  `/api/models/<id>/tree/main` for per-file sizes.
- A ranged `GET` against `resolve/main/<file>.gguf` that reads the first 4 KB of
  the file. That is enough for the GGUF metadata block, which carries the layer
  count, KV head count and head dimension the fit calculation needs. Every quant
  in a repo shares one architecture, so this happens once per repo.

Both endpoints send permissive CORS headers, including on the CDN redirect that
the ranged request follows.

`js/gguf.js` parses the header, `js/fit.js` holds the sizing arithmetic and
command generation, `js/hf.js` wraps the API, and `js/app.js` is the UI.

## Accuracy

The fit numbers are estimates, not llama.cpp's own sizing. Weights are assumed
to spread evenly across layers, the KV cache is computed as
`2 × layers × KV heads × head dim × context` at the selected precision, and 12%
is added for compute buffers and allocator slack. Treat a result near the edge
of your VRAM as "probably" rather than "definitely". Mixture-of-experts and MLA
architectures are the least accurate, since their cache does not follow that
formula.

Quantised KV cache types (`q8_0`, `q4_0`) require `--flash-attn on`, which the
generated command appends automatically. A prebuilt llama.cpp binary without
flash-attention kernels falls back to CPU attention silently, which is 25–45×
slower — something the page cannot detect for you.
