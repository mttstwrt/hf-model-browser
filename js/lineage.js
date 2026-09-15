// Sorting the mainline releases from the remixes.
//
// A GGUF repo is almost never the model's origin — it is someone's conversion of
// a base model published elsewhere in safetensors. Hugging Face records that link
// as `base_model:<relation>:<org>/<name>` tags, so the question "is this a
// mainline release?" is really "what sits upstream of it, and did this repo
// change the weights on the way down?".
//
// The relation tag alone will not answer that: uploaders routinely declare an
// abliterated finetune as `quantized`, so huihui-ai's abliterations arrive
// tagged exactly like unsloth's straight conversions. Three signals together do
// better — the words in the repo name, the relation, and which org the upstream
// belongs to.

/**
 * Orgs that publish base models rather than conversions of someone else's.
 * Used only by the strictest filter tier, and the one part of this file that
 * ages: a new lab shows up as `community` until it is added here.
 */
const OFFICIAL_ORGS = new Set([
  'qwen', 'google', 'meta-llama', 'nvidia', 'ibm-granite', 'deepseek-ai', 'allenai',
  'minimaxai', 'huggingfacetb', 'zai-org', 'tiiuae', 'microsoft', 'moonshotai',
  'coherelabs', 'cohereforai', 'mistralai', 'jinaai', 'nomic-ai', 'kwaipilot',
  'tencent', 'liquidai', '01-ai', 'rinna', 'openbmb', 'internlm', 'baidu',
  'bytedance-seed', 'snowflake', 'intfloat', 'sentence-transformers', 'mixedbread-ai',
  'openai', 'stabilityai', 'baai', 'thudm', 'salesforce', 'eleutherai', 'bigcode',
  'ai21labs', 'upstage', 'sakanaai', 'inclusionai', 'xiaomimimo', 'lgai-exaone',
  'pleias', 'motif-technologies', 'servicenow', 'jetbrains', 'apple', 'facebook',
  'bigscience', 'perplexity-ai', 'agentica-org', 'open-thoughts', 'nousresearch',
]);

/**
 * Words that only ever appear in a repo name because someone altered the
 * weights. Deliberately excludes words the labs themselves use — instruct,
 * chat, coder, distill, base, thinking, qat — which describe official variants.
 */
const DERIVATIVE_MARKERS = new Set([
  'abliterated', 'abliteration', 'uncensored', 'heretic', 'nsfw', 'erp', 'roleplay',
  'rp', 'noromaid', 'venice', 'turbo', 'defiant', 'obliterated', 'aggressive',
  'humanlike', 'efficientthink', 'neo', 'lewd', 'smut', 'horny', 'degenerate',
  'merge', 'merged', 'lora', 'dpo', 'orpo', 'sft', 'finetune', 'finetuned',
  'tuned', 'slerp', 'ties', 'dare', 'passthrough', 'frankenmerge', 'selfmerge',
  'storywriter', 'creative', 'rpmax', 'magnum', 'lumimaid', 'stheno', 'euryale',
  'nymeria', 'unhinged', 'evil', 'forbidden', 'jailbreak', 'jailbroken',
  'unfiltered', 'unalign', 'unaligned', 'unlocked', 'unrestricted', 'amoral',
  'toxic', 'waifu', 'kink', 'decensored', 'deabliterated',
]);

/** Packaging and quantisation-scheme tokens: noise when comparing two names. */
const NOISE = new Set([
  'gguf', 'ggufs', 'ggml', 'hf', 'imatrix', 'imat', 'i1', 'mtp',
  'quantized', 'quant', 'quants', 'qat',
  'f16', 'f32', 'bf16', 'fp16', 'fp8', 'fp4', 'nvfp4', 'mxfp4', 'afp4',
  'int4', 'int8', 'awq', 'gptq', 'exl2', 'gsq', 'rco',
  'q2', 'q3', 'q4', 'q5', 'q6', 'q8', 'iq1', 'iq2', 'iq3', 'iq4',
  'ud', 'k', 'm', 's', 'xs', 'xl', 'xxl', 'l', 'v1', 'v2',
]);

/** Relations that state outright that the weights were changed. */
const CHANGED_WEIGHTS = new Set(['finetune', 'merge', 'adapter']);

const tokens = (name) =>
  name.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t && !NOISE.has(t));

const split = (id) => {
  const i = id.indexOf('/');
  return i < 0 ? ['', id] : [id.slice(0, i), id.slice(i + 1)];
};

/** The upstream this repo declares, preferring the tag that names a relation. */
function upstream(tags = []) {
  let rel = null, base = null;
  for (const t of tags) {
    if (!t.startsWith('base_model:')) continue;
    const rest = t.slice('base_model:'.length);
    const c = rest.indexOf(':');
    if (c < 0) base ??= rest;
    else { rel = rest.slice(0, c); base = rest.slice(c + 1); }
  }
  return { rel, base };
}

/**
 * Classify a search result as one of three tiers, with a short reason for the
 * UI to show on hover:
 *
 *   official    a lab's own release, or a straight conversion of one
 *   community   an independent model, or a conversion of one
 *   derivative  the weights were altered — finetune, merge, abliteration
 */
export function lineage(model) {
  const [org, name] = split(model.id);
  const { rel, base } = upstream(model.tags);
  const [bOrg, bName] = base ? split(base) : ['', ''];
  const sameOrg = bOrg && bOrg.toLowerCase() === org.toLowerCase();

  // A repo that requantises its own upstream carries none of the giveaway words
  // in its own name — DavidAU's conversions of DavidAU's finetunes look clean —
  // so judge the upstream's name too whenever the org is quantising itself.
  const judged = sameOrg ? [...tokens(name), ...tokens(bName)] : tokens(name);
  const marker = judged.find((t) => DERIVATIVE_MARKERS.has(t));
  if (marker) return { tier: 'derivative', why: `name says “${marker}”` };
  if (CHANGED_WEIGHTS.has(rel)) return { tier: 'derivative', why: `tagged as a ${rel}` };

  if (base) {
    // Tokens this repo's name adds to its upstream's. None means a plain
    // repackaging; extras mean someone did something the upstream did not.
    const upstreamTokens = new Set(tokens(bName));
    const extra = tokens(name).filter((t) => !upstreamTokens.has(t));
    if (!OFFICIAL_ORGS.has(bOrg.toLowerCase()))
      return { tier: 'community', why: `converted from ${bOrg}` };
    return extra.length
      ? { tier: 'community', why: `${bOrg} model, modified: ${extra.join(', ')}` }
      : { tier: 'official', why: `straight conversion of ${base}` };
  }

  return OFFICIAL_ORGS.has(org.toLowerCase())
    ? { tier: 'official', why: `published by ${org}` }
    : { tier: 'community', why: 'no upstream declared' };
}

/** Which tiers each filter setting admits. */
export const TIERS = {
  any: null,                                        // no filtering
  vanilla: new Set(['official', 'community']),      // anything with unaltered weights
  official: new Set(['official']),
};
