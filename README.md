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

## Filters

Hugging Face's index is mostly conversions and remixes of a much smaller set of
actual releases, so the filter row narrows a search four ways:

- **Source** — `No finetunes or merges` drops anything whose weights were
  altered (abliterations, merges, roleplay tunes); `Official releases only`
  further restricts to models from a known lab and straight conversions of them.
  See "Telling releases from remixes" below.
- **Min downloads** — a blunt popularity floor. Worth knowing that it does not
  do what "show me the real releases" wants: abliterated and uncensored remixes
  of popular models routinely clear a million downloads, while a genuine
  first-party release of a small model may sit in the low hundreds of thousands.
  It is there for narrowing a broad search, not for judging provenance.
- **Only models trained for N context** — compares your context setting against
  the length the model was actually trained for, so asking for 1M tokens stops
  surfacing 256K models. A model whose trained length is unknown is kept rather
  than hidden, and labelled as such: a missing value usually means Hugging Face
  read the repo's metadata off a projector file, not that the model is short.
- **Only what fits on GPU** — as before.

These are applied in the browser, not by the API, so a strict filter has to read
deeper into the results. The page pulls more as needed, stops after about 400
models, and says so rather than walking the index.

Filter settings persist alongside the hardware ones, and the controls are the
single source of truth — they are re-read rather than tracked in a parallel
variable. Browsers restore form values on reload and on session restore without
firing `change`, so state seeded from defaults and updated only by events will
quietly disagree with what the page shows: a dropdown reading "Official releases
only" above a list that was never filtered.

## Unified memory

Tick **Unified memory** for Apple Silicon, Ryzen AI Max, Jetson and anything
else where the GPU reads the same DRAM as the CPU. The two fields change meaning:
**Total memory** is everything the machine has, and **GPU limit** is how much of
it the GPU may wire down — macOS allows roughly 75% by default, adjustable with
`sudo sysctl iogpu.wired_limit_mb=…`.

This is a correctness fix, not a relabelling. With two separate pools the
estimate checks the offloaded layers against system RAM *in addition to* VRAM;
on one pool that double-counts. A 70 GB model on a 64 GB Mac with a 48 GB wiring
limit reads as "partial offload — 17 GB on the CPU" under the discrete
arithmetic, because it adds 48 GB of VRAM to 64 GB of RAM. Ticking the box
counts the pool once and correctly reports that it will not fit.

## Running it

There is no build step and no dependencies. Serve the directory over HTTP — the
page uses ES modules, so `file://` will not work:

```
python3 -m http.server 8000
```

That server sends no cache headers, so Chrome will happily keep serving an ES
module it already has and your edits will appear to do nothing. Reload with
cache bypassed (`Ctrl`/`Cmd`+`Shift`+`R`) after changing a file under `js/`, or
serve with caching off:

```
python3 -c "import http.server as h; \
  H=type('H',(h.SimpleHTTPRequestHandler,),{'end_headers':lambda s:(s.send_header('Cache-Control','no-store'),h.SimpleHTTPRequestHandler.end_headers(s))}); \
  h.test(HandlerClass=H, port=8000)"
```

GitHub Pages sends ETags and revalidates, so this only bites in local
development.

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

The search asks for `expand[]=tags` and `expand[]=gguf`, which carry the
base-model lineage and the trained context length. That means the source and
context filters work the moment the list arrives, instead of waiting on a header
read per repo. `gguf` also drags in each repo's chat template, which is most of
the response weight — roughly 2 KB gzipped per model — so the first page is kept
small and deeper pages are fetched only when a filter needs them.

`js/gguf.js` parses the header, `js/fit.js` holds the sizing arithmetic and
command generation, `js/hf.js` wraps the API, `js/lineage.js` classifies
provenance, and `js/app.js` is the UI.

## Telling releases from remixes

A GGUF repo is almost never where a model came from — it is someone's conversion
of weights published elsewhere. Hugging Face records that as
`base_model:<relation>:<org>/<name>` tags, but the relation alone will not
separate releases from remixes: uploaders routinely declare an abliterated
finetune as `quantized`, so an abliteration arrives tagged exactly like a
straight conversion. `js/lineage.js` combines three signals instead:

- **Words in the repo name.** `abliterated`, `uncensored`, `merge`, `heretic`
  and friends only appear because someone changed the weights. Words the labs
  themselves use — `instruct`, `distill`, `coder`, `qat` — are deliberately
  excluded. When a repo converts its *own* upstream, the upstream's name is
  read too, since self-published finetunes carry none of the giveaways downstream.
- **Tokens the name adds to its upstream's.** `unsloth/Qwen3.8-27B-GGUF` adds
  nothing to `Qwen/Qwen3.8-27B`, so it is a repackaging;
  `huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF` adds two words, so it is not.
- **Which org the upstream belongs to** — judged on the base model, not the
  converter, since a trusted quantiser will happily convert someone's finetune.

Checked against ~670 real repos, this splits them roughly 53% official, 32%
community, 16% derivative, with the abliterated/uncensored tail landing where
it should. It is a heuristic and will misjudge edge cases — an unusual
quantisation scheme in the name reads as a modification, for instance.

`OFFICIAL_ORGS` in `js/lineage.js` is the one part that ages: a new lab shows up
as `community` until it is added to that list. The two looser tiers do not
depend on it.

## Accuracy

The fit numbers are estimates, not llama.cpp's own sizing. Weights are assumed
to spread evenly across layers, the KV cache is computed as
`2 × layers × KV heads × head dim × context` at the selected precision, and 12%
is added for compute buffers and allocator slack. On unified memory the OS
reserve comes off the single pool once, and the GPU limit caps how much of that
pool can be wired down rather than adding a second budget to it. Treat a result near the edge
of your VRAM as "probably" rather than "definitely". Mixture-of-experts and MLA
architectures are the least accurate, since their cache does not follow that
formula.

Quantised KV cache types (`q8_0`, `q4_0`) require `--flash-attn on`, which the
generated command appends automatically. A prebuilt llama.cpp binary without
flash-attention kernels falls back to CPU attention silently, which is 25–45×
slower — something the page cannot detect for you.
