// Minimal GGUF header reader. Parses only the metadata key/value block, which
// sits at the very start of the file, so a few KB of a ranged GET is enough.

const DECODER = new TextDecoder();

const T = { U8:0, I8:1, U16:2, I16:3, U32:4, I32:5, F32:6, BOOL:7, STRING:8, ARRAY:9, U64:10, I64:11, F64:12 };

/** Thrown when the buffer ends mid-value: caller should refetch a larger range. */
export class Truncated extends Error {}

class Reader {
  constructor(buf) { this.v = new DataView(buf); this.p = 0; }
  need(n) { if (this.p + n > this.v.byteLength) throw new Truncated('need more bytes'); }
  u32() { this.need(4); const x = this.v.getUint32(this.p, true); this.p += 4; return x; }
  u64() { this.need(8); const x = this.v.getBigUint64(this.p, true); this.p += 8; return Number(x); }
  str() { const n = this.u64(); this.need(n); const s = DECODER.decode(new Uint8Array(this.v.buffer, this.v.byteOffset + this.p, n)); this.p += n; return s; }

  value(type) {
    switch (type) {
      case T.U8: case T.BOOL: this.need(1); this.p += 1; return this.v.getUint8(this.p - 1);
      case T.I8: this.need(1); this.p += 1; return this.v.getInt8(this.p - 1);
      case T.U16: this.need(2); this.p += 2; return this.v.getUint16(this.p - 2, true);
      case T.I16: this.need(2); this.p += 2; return this.v.getInt16(this.p - 2, true);
      case T.U32: this.need(4); this.p += 4; return this.v.getUint32(this.p - 4, true);
      case T.I32: this.need(4); this.p += 4; return this.v.getInt32(this.p - 4, true);
      case T.F32: this.need(4); this.p += 4; return this.v.getFloat32(this.p - 4, true);
      case T.U64: case T.I64: return this.u64();
      case T.F64: this.need(8); this.p += 8; return this.v.getFloat64(this.p - 8, true);
      case T.STRING: return this.str();
      case T.ARRAY: {
        const et = this.u32(), n = this.u64();
        // Arrays are only ever tokenizer payloads here, so skip rather than collect.
        const width = { [T.U8]: 1, [T.I8]: 1, [T.BOOL]: 1, [T.U16]: 2, [T.I16]: 2, [T.U32]: 4,
                        [T.I32]: 4, [T.F32]: 4, [T.U64]: 8, [T.I64]: 8, [T.F64]: 8 }[et];
        if (!width) { for (let i = 0; i < n; i++) this.value(et); return null; }  // strings: must walk
        // A few hybrid architectures store head_count_kv per layer, so keep
        // short numeric arrays; long ones are tokenizer payloads worth skipping.
        if (n > 1024) { this.need(n * width); this.p += n * width; return null; }
        const a = []; for (let i = 0; i < n; i++) a.push(this.value(et)); return a;
      }
      default: throw new Error('unknown GGUF value type ' + type);
    }
  }
}

/**
 * Parse GGUF metadata from a (possibly truncated) leading chunk of the file.
 * Stops at the first `tokenizer.*` key: everything we need is written before
 * that boundary, and past it lie multi-megabyte token arrays. Throws Truncated
 * if the chunk ended first, so the caller can refetch a larger range.
 */
export function parseHeader(buf) {
  const r = new Reader(buf);
  if (r.u32() !== 0x46554747) throw new Error('not a GGUF file');
  const version = r.u32();
  if (version < 2 || version > 3) throw new Error('unsupported GGUF version ' + version);
  r.u64();                       // tensor count
  const kvCount = r.u64();
  const kv = {};
  for (let i = 0; i < kvCount; i++) {
    const key = r.str();
    // Writers put the architecture block before the tokenizer, so this is
    // normally where we stop. If one does not, keep going rather than give up.
    if (key.startsWith('tokenizer.') && shape(kv)) return kv;
    kv[key] = r.value(r.u32());
  }
  return kv;
}

/**
 * The numbers the fit calculation needs, or null if this file does not carry
 * them — imatrix sidecars and projectors are GGUF files too, but not models.
 */
export function shape(kv) {
  const a = kv['general.architecture'];
  const g = (k) => kv[`${a}.${k}`];
  const num = (x) => (Array.isArray(x) ? x.reduce((s, v) => s + v, 0) / x.length : x);
  const nHead = num(g('attention.head_count'));
  const nEmbd = num(g('embedding_length'));
  const s = {
    arch: a,
    nLayer: num(g('block_count')),
    nHead,
    nHeadKv: num(g('attention.head_count_kv')) ?? nHead,
    headDim: num(g('attention.key_length')) ?? (nEmbd && nHead ? nEmbd / nHead : undefined),
    trainCtx: num(g('context_length')),
  };
  return s.arch && s.nLayer && s.nHeadKv && s.headDim ? s : null;
}

