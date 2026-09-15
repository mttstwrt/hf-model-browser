// Estimates how a given quant file lands on the user's hardware, and turns that
// estimate into llama.cpp invocations. A simplified stand-in for llama.cpp's own
// -ngl sizing: close enough to choose a quant by, never exact.

export const GB = 1024 ** 3;

// Bytes per cached element. Quantised types store 32 values per block plus an
// fp16 scale, so they cost slightly more than the nominal bits suggest.
export const CACHE_TYPES = { f16: 2, q8_0: 34 / 32, q4_0: 18 / 32 };

// Real runs need a little more than the naive sum: compute buffers, allocator
// slack, fragmentation. 12% is a middle-of-the-road allowance.
const HEADROOM = 1.12;

/**
 * @param size  quant file size in bytes
 * @param shape architecture numbers from the GGUF header
 * @param hw    { vram, ram, reserved } in bytes, { ctx, cacheType, unified }
 */
export function estimate(size, shape, hw) {
  const { nLayer, nHeadKv, headDim } = shape;
  const kvPerLayer = 2 * nHeadKv * headDim * CACHE_TYPES[hw.cacheType] * hw.ctx;
  const weightPerLayer = size / nLayer;              // embeddings smeared across layers
  const perLayer = (weightPerLayer + kvPerLayer) * HEADROOM;

  // On a unified-memory machine there is one pool, not two: an offloaded layer
  // lands in the very same DRAM the GPU is reading from, so RAM and VRAM cannot
  // be spent twice. `vram` there is a ceiling on how much of the pool the GPU is
  // allowed to wire down, not a separate budget, and the OS reserve comes off
  // the pool once rather than off each side.
  const pool = Math.max(0, hw.ram - hw.reserved);
  const vram = hw.unified ? Math.min(hw.vram, pool) : Math.max(0, hw.vram - hw.reserved);
  const ngl = Math.max(0, Math.min(nLayer, Math.floor(vram / perLayer)));

  const onGpu = ngl * perLayer;
  const onCpu = (nLayer - ngl) * perLayer;
  const ramOk = hw.unified ? onGpu + onCpu <= pool : onCpu <= hw.ram;

  return {
    ngl, nLayer, onGpu, onCpu, vram, ramOk, pool,
    unified: !!hw.unified,
    weights: size,
    kv: kvPerLayer * nLayer,
    gpuFraction: ngl / nLayer,
    overCtx: shape.trainCtx > 0 && hw.ctx > shape.trainCtx,
    status: !ramOk ? 'no' : ngl === nLayer ? 'full' : ngl === 0 ? 'cpu' : 'partial',
  };
}

export const STATUS = {
  full:    { icon: '●', word: 'Fits on GPU' },
  partial: { icon: '◐', word: 'Partial offload' },
  cpu:     { icon: '○', word: 'CPU only' },
  no:      { icon: '✕', word: "Won't fit" },
};

/**
 * Fitting entirely in VRAM is a cliff, not a slope — a handful of CPU layers
 * already costs most of the speed — so full fit gets its own colour and the
 * gradient only grades the partial range, amber down to red.
 */
export const fitColor = (f) =>
  f.status === 'full' ? 'var(--ok)'
  : f.status === 'cpu' ? 'var(--muted)'
  : f.status === 'no' ? 'var(--bad)'
  : `hsl(${Math.round(6 + 36 * f.gpuFraction)} 88% 52%)`;

const ref = (repo, q) => (q.quant ? `-hf ${repo}:${q.quant}` : `-hf ${repo} --hf-file ${q.file}`);
const flashAttn = (hw) => hw.cacheType !== 'f16';
// -ngl above the layer count also puts the output layer on the GPU.
const ngl = (f) => (f.status === 'full' ? 99 : f.ngl);

/** Downloads the model if absent, then serves it. Also the download step for router use. */
export function command(repo, q, f, hw) {
  const parts = [
    `llama-server ${ref(repo, q)}`,
    `-ngl ${ngl(f)}`,
    `--ctx-size ${hw.ctx}`,
    `--cache-type-k ${hw.cacheType} --cache-type-v ${hw.cacheType}`,
  ];
  if (flashAttn(hw)) parts.push('--flash-attn on');
  return parts.join(' \\\n  ');
}

/** A fully explicit models.ini section for llama.cpp's router (--models-preset). */
export function iniSection(repo, q, f, hw, fmt) {
  const alias = (repo.split('/')[1] || repo).replace(/[-_.]?gguf$/i, '').toLowerCase()
    + (q.quant ? '-' + q.quant.toLowerCase() : '');
  const rows = [
    ['alias', alias],
    ['ctx-size', hw.ctx],
    ['n-gpu-layers', ngl(f)],
    ['cache-type-k', hw.cacheType],
    ['cache-type-v', hw.cacheType],
  ];
  if (flashAttn(hw)) rows.push(['flash-attn', 'on']);
  const pad = Math.max(...rows.map(([k]) => k.length));

  return [
    `[${repo}${q.quant ? ':' + q.quant : ''}]`,
    f.ngl === f.nLayer
      ? `; all ${f.nLayer} layers on GPU — n-gpu-layers 99 puts the output layer there too`
      : `; ${f.ngl} of ${f.nLayer} layers on GPU, ${f.nLayer - f.ngl} offloaded to system RAM`,
    `; computed for ${hw.unified ? `${fmt(hw.ram)} unified memory, GPU capped at ${fmt(hw.vram)} (${fmt(hw.reserved)} reserved for the OS)`
      : hw.vram ? `${fmt(hw.vram)} VRAM (${fmt(hw.reserved)} reserved for the OS)` : 'no GPU'},`
      + ` ${hw.ctx} context, ${hw.cacheType} KV cache`,
    hw.vram
      ? `; estimated ${fmt(f.onGpu)} VRAM in use` + (f.onCpu > 0 ? `, plus ${fmt(f.onCpu)} of system RAM` : '')
      : `; estimated ${fmt(f.onCpu)} of system RAM in use`,
    ...rows.map(([k, v]) => `${k.padEnd(pad)} = ${v}`),
  ].join('\n');
}
