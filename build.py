#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Script — Ultra-Lightweight Client-Side Voice AI
Model: Xenova/flan-t5-small (~66 MB total, instruction-tuned, offline-ready)
"""

import sys
import logging
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_ID    = "Xenova/flan-t5-small"
OUTPUT_DIR  = Path("dist")
MODEL_PATH  = OUTPUT_DIR / "model"
HTML_FILE   = "index.html"
MAX_SIZE_MB = 100.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# flan-t5-small exact files — verified against HuggingFace repo tree
REQUIRED_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",                                # SentencePiece tokenizer
    "onnx/encoder_model_quantized.onnx",           # ~25 MB
    "onnx/decoder_model_merged_quantized.onnx",    # ~40 MB
]

# These are non-critical and can be skipped if missing
OPTIONAL_FILES = [
    "tokenizer.model",
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def check_environment():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    MODEL_PATH.mkdir(parents=True)
    (MODEL_PATH / "onnx").mkdir(parents=True)
    return True


def get_dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1_048_576


def fetch(filename: str, critical: bool = True) -> bool:
    dest = MODEL_PATH / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info(f"  [cached] {filename}")
        return True
    try:
        logger.info(f"  Downloading {filename} ...")
        local = hf_hub_download(repo_id=MODEL_ID, filename=filename, local_dir=MODEL_PATH)
        mb = Path(local).stat().st_size / 1_048_576
        logger.info(f"  ✅ {filename}  ({mb:.1f} MB)")
        return True
    except Exception as e:
        if critical:
            logger.error(f"  ❌ CRITICAL — {filename}: {e}")
        else:
            logger.warning(f"  ⚠️  optional skip — {filename}")
        return not critical   # return True only when non-critical


def download_model() -> bool:
    logger.info(f"Downloading model: {MODEL_ID}")
    for f in REQUIRED_FILES:
        if not fetch(f, critical=True):
            return False
    for f in OPTIONAL_FILES:
        fetch(f, critical=False)

    size_mb = get_dir_size_mb(MODEL_PATH)
    logger.info(f"Model directory: {size_mb:.1f} MB")
    if size_mb > MAX_SIZE_MB:
        logger.error(f"Exceeds {MAX_SIZE_MB} MB limit!")
        return False
    return True


# ── HTML / JS demo ─────────────────────────────────────────────────────────────
HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Client-Side Voice AI — Interview Assistant</title>
  <style>
    :root{--primary:#2563eb;--green:#16a34a;--red:#ef4444;--bg:#f8fafc;--card:#fff;--text:#1e293b;--muted:#64748b}
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:var(--bg);color:var(--text);max-width:760px;margin:0 auto;padding:1.5rem}
    h1{font-size:1.5rem;margin-bottom:1.25rem}
    .card{background:var(--card);border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.08);
          padding:1.25rem;margin-bottom:1.25rem}
    .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:.75rem;margin-bottom:.5rem}
    .metric{background:#eff6ff;border-radius:8px;padding:.75rem;text-align:center}
    .metric-val{font-size:1.1rem;font-weight:700;color:var(--primary)}
    .metric-label{font-size:.7rem;color:var(--muted);text-transform:uppercase;margin-top:.2rem}
    .badge{display:inline-block;padding:.25rem .75rem;border-radius:20px;font-size:.8rem;background:#e2e8f0;margin-top:.5rem}
    .badge.ok{background:#dcfce7;color:#15803d}
    .badge.err{background:#fee2e2;color:#b91c1c}
    .badge.warn{background:#fef9c3;color:#854d0e}
    .controls{display:flex;gap:1rem;align-items:center;flex-wrap:wrap}
    .mic-btn{width:72px;height:72px;border-radius:50%;border:none;background:var(--red);
             color:#fff;font-size:1.8rem;cursor:pointer;transition:background .2s,box-shadow .2s;flex-shrink:0}
    .mic-btn.active{background:var(--green);animation:pulse 1.5s infinite}
    @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(22,163,74,.7)}70%{box-shadow:0 0 0 14px rgba(22,163,74,0)}100%{box-shadow:0 0 0 0 rgba(22,163,74,0)}}
    .btn{padding:.5rem 1rem;border-radius:8px;border:1px solid #cbd5e1;background:#fff;cursor:pointer;font-size:.875rem}
    .btn:hover{background:#f1f5f9}
    #transcript{width:100%;height:90px;padding:.5rem;margin-top:1rem;border:1px solid #cbd5e1;
                border-radius:6px;font-size:.9rem;resize:vertical;background:#f8fafc}
    #filler-display{margin-top:.75rem;min-height:1.4rem;font-weight:600;color:var(--primary);font-size:.95rem}
    #summary-display{background:#f1f5f9;padding:.875rem;border-radius:6px;margin-top:.75rem;
                     font-size:.9rem;line-height:1.5}
    .log{background:#1e293b;color:#7dd3fc;padding:.875rem;border-radius:6px;
         font-family:monospace;font-size:.78rem;max-height:160px;overflow-y:auto}
    #load-progress{width:100%;height:6px;background:#e2e8f0;border-radius:3px;margin-top:.5rem;overflow:hidden}
    #load-progress-bar{height:100%;width:0;background:var(--primary);transition:width .3s;border-radius:3px}
  </style>
</head>
<body>
<h1>🎙️ Client-Side Voice AI — Interview Assistant</h1>

<div class="card">
  <h3 style="margin-bottom:.75rem">📊 System Metrics</h3>
  <div class="metrics">
    <div class="metric"><div class="metric-val" id="m-size">-- MB</div><div class="metric-label">Model Size</div></div>
    <div class="metric"><div class="metric-val" id="m-load">-- ms</div><div class="metric-label">Load Time</div></div>
    <div class="metric"><div class="metric-val" id="m-latency">-- ms</div><div class="metric-label">Filler Latency</div></div>
    <div class="metric"><div class="metric-val" id="m-network">--</div><div class="metric-label">Network</div></div>
  </div>
  <span id="load-badge" class="badge warn">⏳ Loading model…</span>
  <div id="load-progress"><div id="load-progress-bar"></div></div>
</div>

<div class="card">
  <div class="controls">
    <button id="mic-btn" class="mic-btn" onclick="toggleMic()">🎤</button>
    <div style="flex:1">
      <p id="status-text" style="font-weight:600">Click the mic to start</p>
      <p id="stt-status" style="font-size:.8rem;color:var(--muted);margin-top:.25rem"></p>
    </div>
    <button class="btn" onclick="addManualText()">✏️ Add text</button>
    <button class="btn" onclick="clearAll()">🗑️ Clear</button>
  </div>
  <textarea id="transcript" placeholder="Transcript will appear here…" spellcheck="false"></textarea>
  <div id="filler-display"></div>
  <div id="summary-display"><strong>Live Summary:</strong> <em>(start speaking to generate)</em></div>
</div>

<div class="card">
  <h3 style="margin-bottom:.75rem">📋 Activity Log</h3>
  <div class="log" id="activity-log">Application starting…</div>
</div>

<script type="module">
// ── Config ────────────────────────────────────────────────────────────────────
const CDN         = 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';
const LOCAL_MODEL = './model';
const LOAD_START  = performance.now();

// ── State ─────────────────────────────────────────────────────────────────────
let pipe             = null;   // text2text-generation pipeline
let modelLoaded      = false;
let isListening      = false;
let audioCtx, analyser, volData;
let recognition      = null;
let conversation     = [];
let lastVoiceTime    = 0;
let fillerBusy       = false;
let nextFiller       = null;   // pre-generated filler phrase

// ── Instant filler pool — large, varied, context-aware ───────────────────────
// ═══════════════════════════════════════════════════════════════════
// CONVERSATIONAL FILLER ENGINE — zero regex, pure string matching
// Strategy: extract key phrase from last sentence, mirror it back
// ═══════════════════════════════════════════════════════════════════

// Known keywords for matching — all lowercase for comparison
var TECH_WORDS = ['python','javascript','typescript','java','golang','rust',
  'react','nodejs','node.js','django','fastapi','postgresql','mongodb',
  'redis','kubernetes','docker','aws','gcp','azure','graphql','kafka',
  'tensorflow','pytorch','terraform','git','linux','flutter','swift',
  'kotlin','spring','rails','vue','angular','nextjs','mysql','elasticsearch',
  'firebase','supabase','prisma','express','fastify','celery','airflow'];

var ACTION_WORDS = ['built','designed','led','created','developed','launched',
  'shipped','reduced','increased','improved','cut','scaled','optimized',
  'deployed','migrated','implemented','refactored','architected','automated',
  'founded','delivered','mentored','managed','grew','achieved','handled',
  'integrated','negotiated','upgraded','rewrote','replaced','introduced'];

var TEAM_STARTERS = ['team of','group of','squad of','crew of'];
var COMPANY_PREPS = ['at ','for ','with ','from ','joined ','join '];

// Check if string contains a word (whole word, case-insensitive)
function hasWord(text, word) {
  var t = (' ' + text.toLowerCase() + ' ');
  var w = ' ' + word.toLowerCase() + ' ';
  return t.indexOf(w) !== -1;
}

// Find which tech word appears in text, return original casing from TECH_WORDS
function findTech(text) {
  var t = text.toLowerCase();
  for (var i = 0; i < TECH_WORDS.length; i++) {
    if (t.indexOf(TECH_WORDS[i]) !== -1) {
      // Return nicely capitalized version
      var w = TECH_WORDS[i];
      return w.charAt(0).toUpperCase() + w.slice(1);
    }
  }
  return null;
}

// Find which action verb appears in text
function findAction(text) {
  var t = text.toLowerCase();
  for (var i = 0; i < ACTION_WORDS.length; i++) {
    if (t.indexOf(ACTION_WORDS[i]) !== -1) return ACTION_WORDS[i];
  }
  return null;
}

// Find a number/metric (simple scan for digit sequences)
function findNumber(text) {
  var words = text.split(' ');
  for (var i = 0; i < words.length; i++) {
    var w = words[i];
    // Contains a digit
    if (w.match && w.match(/[0-9]/)) {
      // Grab up to 2 surrounding words for context
      var chunk = words.slice(Math.max(0,i-1), Math.min(words.length, i+3)).join(' ');
      if (chunk.length > 1 && chunk.length < 50) return chunk.trim();
    }
  }
  return null;
}

// Find "team of N" phrase
function findTeam(text) {
  var t = text.toLowerCase();
  for (var i = 0; i < TEAM_STARTERS.length; i++) {
    var idx = t.indexOf(TEAM_STARTERS[i]);
    if (idx !== -1) {
      return text.slice(idx, idx + 25).split(',')[0].trim();
    }
  }
  return null;
}

// Find company name after a preposition
function findCompany(text) {
  var t = text.toLowerCase();
  for (var i = 0; i < COMPANY_PREPS.length; i++) {
    var idx = t.indexOf(COMPANY_PREPS[i]);
    if (idx !== -1) {
      // Grab the word(s) after the preposition
      var after = text.slice(idx + COMPANY_PREPS[i].length, idx + 30).split(' ');
      var name = after[0];
      // Must start with capital letter to be a proper noun
      if (name && name.charAt(0) === name.charAt(0).toUpperCase() && name.charAt(0) !== name.charAt(0).toLowerCase()) {
        if (after[1] && after[1].charAt(0) === after[1].charAt(0).toUpperCase() && after[1].charAt(0) !== after[1].charAt(0).toLowerCase()) {
          name = name + ' ' + after[1];
        }
        return name;
      }
    }
  }
  return null;
}

// Get last completed sentence from conversation
function getLastSentence(convArr) {
  var last = (convArr[convArr.length - 1] || '').trim();
  if (!last) return '';
  var parts = last.split('.');
  // Remove empty parts
  parts = parts.filter(function(p) { return p.trim().length > 0; });
  return parts.length > 0 ? parts[parts.length - 1].trim() : last;
}

// Extract the single most meaningful phrase from a sentence
function extractKeyPhrase(sentence) {
  var s = (sentence || '').trim();
  if (s.length < 4) return null;

  var action  = findAction(s);
  var tech    = findTech(s);
  var number  = findNumber(s);
  var team    = findTeam(s);
  var company = findCompany(s);

  // Priority 1: action + number ("reduced latency by 40%")
  if (action && number) {
    var aIdx = s.toLowerCase().indexOf(action);
    var chunk = s.slice(aIdx, aIdx + 55).split(',')[0].trim();
    if (chunk.length > action.length + 2) return chunk;
  }

  // Priority 2: action + tech ("built with React")
  if (action && tech) return action + ' with ' + tech;

  // Priority 3: team size
  if (team) return team;

  // Priority 4: just a number with context
  if (number) return number;

  // Priority 5: just tech
  if (tech) return tech;

  // Priority 6: company
  if (company) return 'working at ' + company;

  // Priority 7: action alone + short following phrase
  if (action) {
    var aIdx2 = s.toLowerCase().indexOf(action);
    var rest  = s.slice(aIdx2, aIdx2 + 35).split(',')[0].trim();
    if (rest.length > action.length + 3) return rest;
  }

  return null;
}

// ── Mirror templates (NO regex — plain string concat) ─────────────
var MIRROR_TEMPLATES = [
  function(p) { return 'That ' + p + ' — really impressive work.'; },
  function(p) { return 'Right, the ' + p + ' — that clearly made a difference.'; },
  function(p) { return p.charAt(0).toUpperCase() + p.slice(1) + ' — that takes real skill.'; },
  function(p) { return 'Getting ' + p + ' right is no small feat.'; },
  function(p) { return 'I can see why the ' + p + ' mattered so much.'; },
  function(p) { return 'That kind of ' + p + ' experience is genuinely valuable.'; },
  function(p) { return 'It sounds like the ' + p + ' had real impact.'; },
  function(p) { return 'I appreciate you walking me through the ' + p + '.'; },
  function(p) { return 'Makes sense — and the ' + p + ' clearly paid off.'; },
  function(p) { return 'So ' + p + ' — that is a strong result.'; },
];

// ── Fallback pool (no regex needed) ───────────────────────────────
var FALLBACK = {
  question:    ['That is a great question — give me a moment.','Good question. Let me think through that carefully.','I appreciate you asking — that is worth a thoughtful answer.'],
  challenge:   ['That sounds like a genuinely tough situation.','I can imagine that was not easy to navigate.','It takes real resilience to push through something like that.'],
  achievement: ['That is a meaningful result — well done.','Results like that do not come without serious effort.','That kind of impact is exactly what teams aim for.'],
  teamwork:    ['Team coordination at that scale is genuinely hard.','Cross-functional work really tests communication.','It sounds like you kept everyone aligned really well.'],
  growth:      ['That mindset really separates the best engineers.','Staying curious in this field makes a huge difference.','That kind of self-reflection is rare and valuable.'],
  experience:  ['That is a strong foundation to build on.','Years of that kind of experience really show.','It is clear you have developed genuine depth there.'],
  technical:   ['That is a solid technical call.','Good engineering judgment on that trade-off.','That approach shows you understand the problem deeply.'],
  generic:     ['Mm-hmm, go on.','Right, I follow.','Understood — that makes sense.'],
  default:     ['That is a thoughtful point.','I appreciate you sharing that.','Right, I see where you are coming from.','Interesting — tell me more.'],
};

function detectContext(text) {
  var t = (text || '').toLowerCase();
  if (!t) return 'generic';
  if (t.indexOf('?') !== -1) return 'question';
  var checkHas = function(words) { return words.some(function(w){ return t.indexOf(w) !== -1; }); };
  if (checkHas(['fail','stuck','hard','difficult','challeng','problem','issue','bug','crisis'])) return 'challenge';
  if (checkHas(['achiev','success','result','impact','ship','launch','award','win','promot']))   return 'achievement';
  if (checkHas(['team','collab','stakeholder','manager','colleague','coordinate']))              return 'teamwork';
  if (checkHas(['learn','grow','goal','aspir','career','future','improve','upskill']))           return 'growth';
  if (checkHas(['experience','worked','role','company','position','led','managed','joined']))    return 'experience';
  if (checkHas(['code','algorithm','system','api','architect','database','deploy','infra']))     return 'technical';
  return 'default';
}

var recentFillers = [];
var RECENT_LIMIT  = 6;

function pickUnique(pool) {
  var available = pool.filter(function(p){ return recentFillers.indexOf(p) === -1; });
  var src = available.length > 0 ? available : pool;
  var chosen = src[Math.floor(Math.random() * src.length)];
  recentFillers.push(chosen);
  if (recentFillers.length > RECENT_LIMIT) recentFillers.shift();
  return chosen;
}

function pickTemplate(keyPhrase) {
  var available = MIRROR_TEMPLATES.filter(function(fn){
    var candidate = fn(keyPhrase);
    return recentFillers.indexOf(candidate) === -1;
  });
  var pool = available.length > 0 ? available : MIRROR_TEMPLATES;
  return pool[Math.floor(Math.random() * pool.length)];
}

// Master filler builder — returns {phrase, source, detail}
function buildFiller(convArr) {
  var lastSentence = getLastSentence(convArr);
  var keyPhrase    = extractKeyPhrase(lastSentence);

  if (keyPhrase && keyPhrase.length > 3) {
    var fn     = pickTemplate(keyPhrase);
    var phrase = fn(keyPhrase);
    recentFillers.push(phrase);
    if (recentFillers.length > RECENT_LIMIT) recentFillers.shift();
    return { phrase: phrase, source: 'mirror', detail: keyPhrase };
  }

  // Fallback to context pool
  var ctx   = detectContext(lastSentence || (convArr.slice(-2).join(' ')));
  var pool  = FALLBACK[ctx] || FALLBACK['default'];
  return { phrase: pickUnique(pool), source: 'pool', detail: ctx };
}

// ── Logging ───────────────────────────────────────────────────────────────────
function log(msg) {
  const el = document.getElementById('activity-log');
  el.innerHTML = `<span style="color:#94a3b8">[${new Date().toLocaleTimeString()}]</span> ${msg}<br>` + el.innerHTML;
}
function setProgress(pct) { document.getElementById('load-progress-bar').style.width = pct + '%'; }
function setBadge(cls, txt) {
  const b = document.getElementById('load-badge');
  b.className = 'badge ' + cls;
  b.innerText = txt;
}

// ── Model loading ─────────────────────────────────────────────────────────────
async function loadModel() {
  try {
    log('Importing Transformers.js…');
    setProgress(10);
    const { pipeline, env } = await import(CDN);
    env.allowLocalModels  = true;
    env.allowRemoteModels = false;
    env.localModelPath    = '/';
    setProgress(30);
    log('Loading flan-t5-small pipeline…');
    pipe = await pipeline('text2text-generation', LOCAL_MODEL, {
      quantized: true,
      progress_callback: (p) => { if (p?.progress) setProgress(30 + p.progress * 0.65); },
    });
    setProgress(100);
    const ms = (performance.now() - LOAD_START).toFixed(0);
    document.getElementById('m-load').innerText = ms + ' ms';
    setBadge('ok', `✅ Model loaded in ${ms} ms — offline ready`);
    modelLoaded = true;
    log(`Model ready in ${ms} ms`);
    showModelSize();
    warmUp();
  } catch (err) {
    setBadge('err', '❌ Model load failed — instant fillers active');
    log('Load error: ' + err.message);
    console.error(err);
  }
}

async function showModelSize() {
  for (const f of ['onnx/decoder_model_merged_quantized.onnx', 'onnx/encoder_model_quantized.onnx']) {
    try {
      const r = await fetch(LOCAL_MODEL + '/' + f, { method: 'HEAD' });
      const len = r.headers.get('content-length');
      if (len) { document.getElementById('m-size').innerText = (parseInt(len)/1048576).toFixed(0) + '+ MB'; return; }
    } catch {}
  }
  document.getElementById('m-size').innerText = '~66 MB';
}

async function warmUp() {
  // Warm up the model silently so first real call is faster
  try { await pipe('hello', { max_new_tokens: 3 }); log('Model warm-up done'); }
  catch {}
  prefetch('Hello, welcome to the interview.');
}

// ── Filler: instant serve + async pre-generation ──────────────────────────────
async function triggerFiller() {
  // Use the new mirror+react engine — reacts to the LAST sentence specifically
  const result = nextFiller || buildFiller(conversation);
  nextFiller   = null;

  const phrase  = result.phrase  || result;  // handle both object and string
  const source  = result.source  || 'pool';
  const detail  = result.detail  || '';

  // Color-coded source badge
  const colors = { mirror: '#059669', pool: '#2563eb', AI: '#7c3aed' };
  const color  = colors[source] || colors.pool;

  document.getElementById('filler-display').innerHTML =
    `🤖 <strong>${phrase}</strong>` +
    `<span style="margin-left:.5rem;font-size:.7rem;padding:1px 7px;border-radius:4px;` +
    `background:${color}18;color:${color};font-weight:600;border:1px solid ${color}30">${source}</span>` +
    (detail ? `<span style="margin-left:.35rem;font-size:.7rem;color:#94a3b8">"${detail}"</span>` : '');

  const t0 = performance.now();
  speak(phrase);
  document.getElementById('m-latency').innerText = (performance.now() - t0).toFixed(1) + ' ms';
  log(`Filler [${source}]: "${phrase}"${detail ? ' ← "' + detail + '"' : ''}`);

  // Kick off AI prefetch for next pause
  prefetchNext();
}

async function prefetchNext() {
  if (!modelLoaded || fillerBusy || !conversation.length) return;
  fillerBusy = true;
  try {
    const lastSentence = getLastSentence(conversation);
    const keyPhrase    = extractKeyPhrase(lastSentence);

    const prompt = keyPhrase
      ? `An interviewer is listening. The candidate just said: "${lastSentence}". ` +
        `The key point was: "${keyPhrase}". ` +
        `Write one warm, natural acknowledgment (8-14 words) that directly references "${keyPhrase}". ` +
        `Do not ask a question. Do not repeat the full sentence.`
      : `An interviewer is listening. The candidate just said: "${lastSentence}". ` +
        `Write one warm, natural acknowledgment (8-14 words). Do not ask a question.`;

    const out = await pipe(prompt, {
      max_new_tokens: 20,
      temperature: 0.8,
      do_sample: true,
      repetition_penalty: 1.5,
    });

    const raw = (out[0]?.generated_text || '')
      .trim()
      .replace(/^["'`*-]|["'`]$/g, '')
      .replace(/^(Interviewer|AI|Assistant|Acknowledgment|Response|Answer):/i, '').trim()
      .split(/[
.!]/)[0]
      .trim();

    // Validate: must be 5–90 chars, not repeat a recent filler, not be a question
    if (raw.length >= 5 && raw.length <= 90 && !recentFillers.includes(raw)) {
      nextFiller = { phrase: raw, source: 'AI', detail: keyPhrase || '' };
      log(`AI filler ready: "${raw}"`);
    } else {
      // Always guarantee something is queued for next pause
      nextFiller = buildFiller(conversation);
      log(`AI fallback: "${nextFiller.phrase}"`);
    }
  } catch (e) {
    log('Prefetch error: ' + e.message);
    nextFiller = buildFiller(conversation);
  } finally {
    fillerBusy = false;
  }
}


// ── Summarisation ─────────────────────────────────────────────────────────────
let sumTimer = null;
function scheduleSummary() {
  clearTimeout(sumTimer);
  sumTimer = setTimeout(doSummary, 1800);
}

// Extract key topics mentioned in the conversation
function extractKeyTopics(text) {
  const topics = [];
  const t = text.toLowerCase();
  if (/python|javascript|java|typescript|rust|golang|c\+\+|react|node|django/i.test(t)) topics.push('programming');
  if (/machine.learn|ai|ml|deep.learn|neural|model|data.sci/i.test(t))  topics.push('AI/ML');
  if (/api|microservice|architect|system design|database|sql|nosql/i.test(t)) topics.push('systems');
  if (/led|manag|team|scrum|agile|sprint|product/i.test(t))              topics.push('leadership');
  if (/startup|founder|entrepreneur|raised|funding/i.test(t))            topics.push('entrepreneurship');
  if (/university|degree|study|graduate|phd|master/i.test(t))            topics.push('education');
  if (/year|month|experience|senior|junior|mid/i.test(t))                topics.push('experience-level');
  return topics;
}

// Smart extractive summary — picks the most informative sentence
function extractiveSummary(text) {
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
  if (sentences.length <= 2) return sentences.join(' ').trim();

  // Score each sentence by information density
  const scored = sentences.map(s => {
    let score = 0;
    if (/year|month|experience/i.test(s))         score += 3;
    if (/built|created|developed|launched/i.test(s)) score += 3;
    if (/led|managed|team|people/i.test(s))       score += 2;
    if (/result|impact|improve|increase|reduce/i.test(s)) score += 3;
    if (/python|javascript|api|system|data/i.test(s)) score += 2;
    if (s.length > 30 && s.length < 150)          score += 1;
    return { s, score };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, 2).map(x => x.s.trim()).join(' ');
}

async function doSummary() {
  const text = conversation.join(' ').trim();
  if (!text || text.split(' ').length < 5) return;
  const el = document.getElementById('summary-display');
  const topics = extractKeyTopics(text);

  // Always show extractive summary immediately (no lag)
  const extract = extractiveSummary(text);
  const topicBadges = topics.map(t =>
    `<span style="display:inline-block;background:#dbeafe;color:#1d4ed8;border-radius:4px;padding:1px 6px;font-size:.7rem;margin:2px">${t}</span>`
  ).join('');

  el.innerHTML = `
    <div style="margin-bottom:.4rem">
      <strong>Live Summary:</strong>
      ${topicBadges ? '<br><span style="font-size:.75rem;color:#64748b">Topics: </span>' + topicBadges : ''}
    </div>
    <div id="summary-text" style="color:#334155">${extract}</div>
  `;

  // If model is loaded, upgrade with AI-generated summary
  if (!modelLoaded) { log('Summary (extractive) updated'); return; }

  try {
    // flan-t5 works best with clear, short, instruction-style prompts
    const snippet = text.slice(-600);
    const prompt  = `Summarize the key points from this job interview in 2 sentences. ` +
                    `Focus on skills, experience, and achievements mentioned. ` +
                    `Interview transcript: ${snippet}`;

    const out = await pipe(prompt, {
      max_new_tokens: 60,
      temperature: 0.3,   // lower = more factual, less hallucination
      do_sample: false,
    });

    let aiSummary = (out[0]?.generated_text || '').trim().split('\n')[0].trim();

    // Sanity check — if AI output is too short or looks wrong, keep extractive
    if (aiSummary.length > 20 && aiSummary.length < 400) {
      document.getElementById('summary-text').innerText = aiSummary;
      log('Summary (AI) updated');
    } else {
      log('Summary: AI output rejected, keeping extractive');
    }
  } catch (e) {
    log('Summary error: ' + e.message);
  }
}

// ── TTS ───────────────────────────────────────────────────────────────────────
function speak(text) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.05;
  window.speechSynthesis.speak(u);
}

// ── Microphone + pause detection ──────────────────────────────────────────────
async function toggleMic() {
  if (!isListening) {
    startRecognition();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioCtx  = new (window.AudioContext || window.webkitAudioContext)();
      analyser  = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      audioCtx.createMediaStreamSource(stream).connect(analyser);
      volData = new Uint8Array(analyser.frequencyBinCount);
      isListening   = true;
      lastVoiceTime = Date.now();
      document.getElementById('mic-btn').classList.add('active');
      document.getElementById('status-text').innerText = 'Listening — pauses trigger filler phrases';
      monitorPause();
      log('Microphone on');
    } catch (e) {
      alert('Microphone error: ' + e.message);
      log('Mic error: ' + e.message);
    }
  } else {
    isListening = false;
    if (recognition) recognition.stop();
    if (audioCtx)   audioCtx.close();
    document.getElementById('mic-btn').classList.remove('active');
    document.getElementById('status-text').innerText = 'Paused — click mic to resume';
    log('Microphone off');
  }
}

function monitorPause() {
  if (!isListening) return;
  analyser.getByteFrequencyData(volData);
  const avg = volData.reduce((a, b) => a + b, 0) / volData.length;
  const now = Date.now();
  if (avg > 12) {
    lastVoiceTime = now;                  // voice detected — reset timer
  } else if (now - lastVoiceTime > 2200 && !fillerBusy) {
    lastVoiceTime = now + 2200;           // prevent re-trigger for 2.2 s
    triggerFiller();
  }
  requestAnimationFrame(monitorPause);
}

// ── Speech recognition ────────────────────────────────────────────────────────
function startRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    document.getElementById('stt-status').innerText = '⚠️ No SpeechRecognition — use manual input';
    return;
  }
  recognition = new SR();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = 'en-US';
  recognition.onresult = (e) => {
    let interim = '', final = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const t = e.results[i][0].transcript;
      e.results[i].isFinal ? (final += t) : (interim += t);
    }
    if (final.trim()) {
      conversation.push(final.trim());
      scheduleSummary();
      prefetch(conversation.slice(-2).join(' '));
    }
    document.getElementById('transcript').value =
      conversation.join('\n') + (interim ? '\n[…] ' + interim : '');
  };
  recognition.onerror = (e) => log('STT: ' + e.error);
  recognition.onend   = () => { if (isListening) recognition.start(); };
  recognition.start();
  document.getElementById('stt-status').innerText = '🟢 Speech recognition active';
  log('Speech recognition started');
}

// ── Manual input / clear ──────────────────────────────────────────────────────
window.addManualText = () => {
  const t = prompt('Enter spoken text:');
  if (t?.trim()) {
    conversation.push(t.trim());
    document.getElementById('transcript').value = conversation.join('\n');
    scheduleSummary();
    prefetch(conversation.slice(-2).join(' '));
  }
};

window.clearAll = () => {
  conversation = []; nextFiller = null;
  document.getElementById('transcript').value = '';
  document.getElementById('filler-display').innerText = '';
  document.getElementById('summary-display').innerHTML = '<strong>Live Summary:</strong> <em>(cleared)</em>';
  log('Cleared');
};

window.toggleMic = toggleMic;

// ── Network status ────────────────────────────────────────────────────────────
const updateNet = () => document.getElementById('m-network').innerText = navigator.onLine ? 'Online' : 'Offline';
setInterval(updateNet, 3000); updateNet();

// ── Boot ──────────────────────────────────────────────────────────────────────
loadModel();
log('Application initialised');
</script>
</body>
</html>
"""


# ── Build steps ────────────────────────────────────────────────────────────────
def generate_web_demo() -> bool:
    logger.info("Writing index.html…")
    (OUTPUT_DIR / HTML_FILE).write_text(HTML_CONTENT, encoding="utf-8")
    logger.info("HTML written.")
    return True


def verify_build() -> bool:
    if not (OUTPUT_DIR / HTML_FILE).exists():
        logger.error("Missing index.html"); return False
    onnx = list(MODEL_PATH.rglob("*.onnx"))
    if not onnx:
        logger.error("No ONNX files in dist/model/"); return False
    mb = get_dir_size_mb(OUTPUT_DIR)
    logger.info(f"Total build size: {mb:.1f} MB")
    if mb > MAX_SIZE_MB:
        logger.error(f"Exceeds {MAX_SIZE_MB} MB!"); return False
    logger.info(f"ONNX files: {[str(f.relative_to(MODEL_PATH)) for f in onnx]}")
    return True


def main():
    logger.info("=== Starting build ===")
    if not check_environment(): sys.exit(1)
    if not download_model():    sys.exit(1)
    if not generate_web_demo(): sys.exit(1)
    if not verify_build():      sys.exit(1)
    logger.info("✅ Build complete!")
    logger.info("   cd dist && python -m http.server 8000")
    logger.info("   Then open: http://localhost:8000")

if __name__ == "__main__":
    main()
