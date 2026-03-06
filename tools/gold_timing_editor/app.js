const state = {
  doc: null,
  secondsToPx: 90,
  editMode: "word",
  selected: null,
  selectedLine: null,
  drag: null,
  undoStack: [],
  lastActiveLineIndex: -1,
  pendingSeekScroll: false,
  audioAnchors: [],
  audioAnalysisToken: 0,
  audioAnalysisReady: false,
  sessionStats: null,
  autoScrollEnabled: true,
  tapPassMode: false,
  tapCursor: null,
};
const MIN_WORD_DURATION = 0.1;
const SNAP_SECONDS = 0.05;
const KEYBOARD_FINE_STEP = SNAP_SECONDS;
const KEYBOARD_COARSE_STEP = 0.2;
const SNAP_MAX_DISTANCE_SEC = 0.2;

const els = {
  status: document.getElementById("status"),
  timingPath: document.getElementById("timingPath"),
  loadTimingBtn: document.getElementById("loadTimingBtn"),
  audioPath: document.getElementById("audioPath"),
  loadAudioBtn: document.getElementById("loadAudioBtn"),
  savePath: document.getElementById("savePath"),
  saveBtn: document.getElementById("saveBtn"),
  playPauseBtn: document.getElementById("playPauseBtn"),
  startTapPassBtn: document.getElementById("startTapPassBtn"),
  tapNextBtn: document.getElementById("tapNextBtn"),
  wordModeBtn: document.getElementById("wordModeBtn"),
  lineModeBtn: document.getElementById("lineModeBtn"),
  tapModeToggle: document.getElementById("tapModeToggle"),
  autoscrollToggle: document.getElementById("autoscrollToggle"),
  zoomRange: document.getElementById("zoomRange"),
  playbackInfo: document.getElementById("playbackInfo"),
  selectionInfo: document.getElementById("selectionInfo"),
  telemetryInfo: document.getElementById("telemetryInfo"),
  timeline: document.getElementById("timeline"),
  audio: document.getElementById("audio"),
};

function snap(v) {
  return Math.round((Math.round(v / SNAP_SECONDS) * SNAP_SECONDS) * 1000) / 1000;
}

function isTextEditable(el) {
  return (
    el instanceof HTMLElement &&
    (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)
  );
}

function blurEditableFocus() {
  const active = document.activeElement;
  if (isTextEditable(active)) {
    active.blur();
  }
}

function setStatus(msg, isError = false) {
  els.status.textContent = msg;
  els.status.style.color = isError ? "#a92323" : "#5e6c5f";
}

function resetSessionStats() {
  state.sessionStats = {
    sessionStartMs: Date.now(),
    editCount: 0,
    undoCount: 0,
    snapCount: 0,
    jumpCount: 0,
    nudgeFineCount: 0,
    nudgeCoarseCount: 0,
    dragCount: 0,
  };
}

function stats() {
  if (!state.sessionStats) {
    resetSessionStats();
  }
  return state.sessionStats;
}

function recordEdit(kind) {
  const s = stats();
  s.editCount += 1;
  if (kind === "snap") s.snapCount += 1;
  if (kind === "jump") s.jumpCount += 1;
  if (kind === "nudge-fine") s.nudgeFineCount += 1;
  if (kind === "nudge-coarse") s.nudgeCoarseCount += 1;
  if (kind === "drag") s.dragCount += 1;
}

function recordUndo() {
  stats().undoCount += 1;
}

function updateTelemetryInfo() {
  if (!els.telemetryInfo) return;
  const s = stats();
  const elapsedMin = Math.max((Date.now() - s.sessionStartMs) / 60000, 1 / 60);
  const epm = s.editCount / elapsedMin;
  const undoDenom = Math.max(1, s.editCount + s.undoCount);
  const undoPct = (100 * s.undoCount) / undoDenom;
  const snapPct = s.editCount ? (100 * s.snapCount) / s.editCount : 0;
  const jumpPct = s.editCount ? (100 * s.jumpCount) / s.editCount : 0;
  els.telemetryInfo.textContent =
    `session: edits=${s.editCount} | epm=${epm.toFixed(1)} | undo=${undoPct.toFixed(0)}%` +
    ` | snap=${snapPct.toFixed(0)}% | jump=${jumpPct.toFixed(0)}%`;
}

function resetUndo() {
  state.undoStack = [];
}

function snapshotForUndo() {
  if (!state.doc) return;
  state.undoStack.push(JSON.stringify(state.doc));
  if (state.undoStack.length > 100) {
    state.undoStack.shift();
  }
}

function undo() {
  if (!state.undoStack.length) return;
  state.doc = JSON.parse(state.undoStack.pop());
  recordUndo();
  updateLineBounds();
  render();
  setStatus("Undo applied.");
}

function lineWordCount(doc) {
  return doc.lines.reduce((acc, line) => acc + line.words.length, 0);
}

function findPrevWord(li, wi) {
  if (li === 0 && wi === 0) return null;
  if (wi > 0) return state.doc.lines[li].words[wi - 1];
  const prevLine = state.doc.lines[li - 1];
  return prevLine.words[prevLine.words.length - 1];
}

function findNextWordRef(li, wi) {
  if (!state.doc?.lines?.length) return null;
  const line = state.doc.lines[li];
  if (!line) return null;
  if (wi + 1 < line.words.length) return { li, wi: wi + 1 };
  if (li + 1 >= state.doc.lines.length) return null;
  if (!state.doc.lines[li + 1].words.length) return null;
  return { li: li + 1, wi: 0 };
}

function firstWordRef() {
  if (!state.doc?.lines?.length) return null;
  for (let li = 0; li < state.doc.lines.length; li += 1) {
    if (state.doc.lines[li].words.length) return { li, wi: 0 };
  }
  return null;
}

function setTapPassMode(enabled) {
  state.tapPassMode = Boolean(enabled);
  if (els.tapModeToggle) {
    els.tapModeToggle.checked = state.tapPassMode;
  }
  if (!state.tapCursor) {
    state.tapCursor = firstWordRef();
  }
}

function stampWordStartToCurrentTime(li, wi) {
  if (!state.doc || Number.isNaN(els.audio.currentTime)) return false;
  const word = state.doc.lines?.[li]?.words?.[wi];
  if (!word) return false;
  const duration = Math.max(word.end - word.start, MIN_WORD_DURATION);
  setWordTiming(li, wi, els.audio.currentTime, els.audio.currentTime + duration);
  return true;
}

function stampTapCursorWordToCurrentTime() {
  if (!state.doc) return false;
  if (!state.tapCursor) {
    state.tapCursor = firstWordRef();
  }
  if (!state.tapCursor) return false;

  const { li, wi } = state.tapCursor;
  if (!stampWordStartToCurrentTime(li, wi)) return false;

  const nextRef = findNextWordRef(li, wi);
  state.tapCursor = nextRef;
  if (nextRef) {
    selectWord(nextRef.li, nextRef.wi);
  } else {
    selectWord(li, wi);
  }
  return true;
}

function startTapPassFromBeginning() {
  if (!state.doc) return false;
  const first = firstWordRef();
  if (!first) return false;
  setTapPassMode(true);
  state.tapCursor = first;
  selectWord(first.li, first.wi);
  state.pendingSeekScroll = state.autoScrollEnabled;
  els.audio.currentTime = 0;
  els.playbackInfo.textContent = `t=${els.audio.currentTime.toFixed(1)}s`;
  if (els.audio.paused) {
    els.audio.play();
  }
  return true;
}

function updateLineBounds() {
  for (let i = 0; i < state.doc.lines.length; i += 1) {
    const words = state.doc.lines[i].words;
    state.doc.lines[i].line_index = i + 1;
    state.doc.lines[i].start = words[0].start;
    state.doc.lines[i].end = words[words.length - 1].end;
    state.doc.lines[i].text = words.map((w) => w.text).join(" ");
    for (let j = 0; j < words.length; j += 1) {
      words[j].word_index = j + 1;
    }
  }
}

function selectWord(li, wi) {
  state.selectedLine = li;
  state.selected = { li, wi };
  render();
}

function selectLine(li) {
  state.selectedLine = li;
  state.selected = null;
  render();
}

function setEditMode(mode) {
  state.editMode = mode === "line" ? "line" : "word";
  els.wordModeBtn.classList.toggle("active", state.editMode === "word");
  els.lineModeBtn.classList.toggle("active", state.editMode === "line");
  document.body.classList.toggle("line-edit-mode", state.editMode === "line");
  if (state.editMode === "word" && state.selectedLine != null && !state.selected) {
    state.selected = { li: state.selectedLine, wi: 0 };
  }
  if (state.editMode === "line" && state.selected?.li != null) {
    state.selectedLine = state.selected.li;
    state.selected = null;
  }
  render();
}

function shiftLine(li, delta) {
  const line = state.doc.lines[li];
  if (!line || !line.words.length) return;
  const effectiveDelta = snap(delta);
  for (const w of line.words) {
    w.start = snap(w.start + effectiveDelta);
    w.end = snap(w.end + effectiveDelta);
  }
  updateLineBounds();
}

function snapshotAllLineWords() {
  return state.doc.lines.map((line) =>
    line.words.map((w) => ({ start: w.start, end: w.end }))
  );
}

function restoreAllLineWords(snapshot) {
  for (let li = 0; li < state.doc.lines.length; li += 1) {
    const words = state.doc.lines[li].words;
    const savedWords = snapshot[li] || [];
    for (let wi = 0; wi < words.length; wi += 1) {
      const sw = savedWords[wi];
      if (!sw) continue;
      words[wi].start = sw.start;
      words[wi].end = sw.end;
    }
  }
  updateLineBounds();
}

function enforceNonOverlapCascade(anchorLi) {
  const lines = state.doc.lines;
  if (!lines.length) return;

  for (let li = anchorLi + 1; li < lines.length; li += 1) {
    const overlap = lines[li - 1].end - lines[li].start;
    if (overlap > 0) {
      shiftLine(li, overlap);
    }
  }

  for (let li = anchorLi - 1; li >= 0; li -= 1) {
    const overlap = lines[li].end - lines[li + 1].start;
    if (overlap > 0) {
      shiftLine(li, -overlap);
    }
  }

  let minStart = Number.POSITIVE_INFINITY;
  for (const line of lines) {
    if (line.words.length && line.start < minStart) {
      minStart = line.start;
    }
  }
  if (minStart < 0) {
    const fixDelta = -minStart;
    for (let li = 0; li < lines.length; li += 1) {
      shiftLine(li, fixDelta);
    }
  }
  updateLineBounds();
}

function startLineDrag(li, ev, pxPerSec = state.secondsToPx) {
  const line = state.doc.lines[li];
  if (!line || !line.words?.length) return;
  ev.preventDefault();
  ev.stopPropagation();
  blurEditableFocus();
  snapshotForUndo();
  selectLine(li);
  state.drag = {
    mode: "line-move",
    li,
    startX: ev.clientX,
    moved: false,
    pxPerSec,
    originalAll: snapshotAllLineWords(),
  };
}

function _allWordRefs() {
  const refs = [];
  for (let lineIdx = 0; lineIdx < state.doc.lines.length; lineIdx += 1) {
    const words = state.doc.lines[lineIdx].words;
    for (let wordIdx = 0; wordIdx < words.length; wordIdx += 1) {
      refs.push({ li: lineIdx, wi: wordIdx, word: words[wordIdx] });
    }
  }
  return refs;
}

function setWordTiming(li, wi, nextStart, nextEnd) {
  const refs = _allWordRefs();
  const editedIndex = refs.findIndex((r) => r.li === li && r.wi === wi);
  if (editedIndex < 0) return;

  const starts = refs.map((r) => snap(r.word.start));
  const ends = refs.map((r) => snap(r.word.end));
  const durations = refs.map((r) =>
    Math.max(snap(r.word.end - r.word.start), MIN_WORD_DURATION)
  );

  let start = snap(nextStart);
  let end = snap(nextEnd);
  if (start < 0) start = 0;
  if (end < start + MIN_WORD_DURATION) end = start + MIN_WORD_DURATION;
  starts[editedIndex] = start;
  ends[editedIndex] = end;
  durations[editedIndex] = Math.max(snap(end - start), MIN_WORD_DURATION);

  // Cascade backward across song words.
  for (let j = editedIndex - 1; j >= 0; j -= 1) {
    if (ends[j] > starts[j + 1]) {
      ends[j] = starts[j + 1];
      starts[j] = ends[j] - durations[j];
    } else if (ends[j] < starts[j] + MIN_WORD_DURATION) {
      ends[j] = starts[j] + MIN_WORD_DURATION;
    }
  }

  // Shift entire song forward if any word fell below zero.
  if (starts[0] < 0) {
    const shift = -starts[0];
    for (let j = 0; j < refs.length; j += 1) {
      starts[j] += shift;
      ends[j] += shift;
    }
  }

  // Cascade forward across song words.
  for (let j = editedIndex + 1; j < refs.length; j += 1) {
    if (starts[j] < ends[j - 1]) {
      starts[j] = ends[j - 1];
      ends[j] = starts[j] + durations[j];
    } else if (ends[j] < starts[j] + MIN_WORD_DURATION) {
      ends[j] = starts[j] + MIN_WORD_DURATION;
    }
  }

  // Final defensive pass.
  for (let j = 0; j < refs.length; j += 1) {
    starts[j] = snap(Math.max(starts[j], 0));
    ends[j] = snap(Math.max(ends[j], starts[j] + MIN_WORD_DURATION));
    if (j > 0 && starts[j] < ends[j - 1]) {
      const push = ends[j - 1] - starts[j];
      starts[j] = snap(starts[j] + push);
      ends[j] = snap(ends[j] + push);
    }
  }

  for (let j = 0; j < refs.length; j += 1) {
    refs[j].word.start = snap(starts[j]);
    refs[j].word.end = snap(ends[j]);
  }
  updateLineBounds();
}

function moveWord(li, wi, delta) {
  const w = state.doc.lines[li].words[wi];
  setWordTiming(li, wi, w.start + delta, w.end + delta);
}

function adjustStart(li, wi, delta) {
  const w = state.doc.lines[li].words[wi];
  setWordTiming(li, wi, w.start + delta, w.end);
}

function adjustEnd(li, wi, delta) {
  const w = state.doc.lines[li].words[wi];
  setWordTiming(li, wi, w.start, w.end + delta);
}

function getLineBounds(line) {
  let minStart = Number.POSITIVE_INFINITY;
  let maxEnd = Number.NEGATIVE_INFINITY;
  for (const w of line.words) {
    if (w.start < minStart) minStart = w.start;
    if (w.end > maxEnd) maxEnd = w.end;
  }
  return { start: minStart, end: maxEnd };
}

function getLineViewWindow(line) {
  const bounds = getLineBounds(line);
  const start = Math.max(0, bounds.start - 1.0);
  const end = Math.max(start + 2.0, bounds.end + 1.0);
  return { start, end, duration: end - start };
}

function nearestAudioAnchor(timeSec, maxDistanceSec = SNAP_MAX_DISTANCE_SEC) {
  if (!state.audioAnchors.length) return null;
  let best = null;
  let bestDist = Number.POSITIVE_INFINITY;
  for (const anchor of state.audioAnchors) {
    const dist = Math.abs(anchor - timeSec);
    if (dist < bestDist) {
      bestDist = dist;
      best = anchor;
    }
  }
  if (best == null || bestDist > maxDistanceSec) return null;
  return best;
}

function directionalAudioAnchor(timeSec, direction) {
  if (!state.audioAnchors.length) return null;
  const eps = 0.001;
  if (direction > 0) {
    for (const anchor of state.audioAnchors) {
      if (anchor > timeSec + eps) return anchor;
    }
    return null;
  }
  for (let i = state.audioAnchors.length - 1; i >= 0; i -= 1) {
    const anchor = state.audioAnchors[i];
    if (anchor < timeSec - eps) return anchor;
  }
  return null;
}

function snapSelectedToAudioAnchor(mode) {
  if (!state.doc) return false;
  if (state.editMode === "line") {
    if (state.selectedLine == null) return false;
    const line = state.doc.lines[state.selectedLine];
    const target = nearestAudioAnchor(line.start);
    if (target == null) return false;
    shiftLine(state.selectedLine, target - line.start);
    enforceNonOverlapCascade(state.selectedLine);
    return true;
  }

  if (!state.selected) return false;
  const { li, wi } = state.selected;
  const word = state.doc.lines[li].words[wi];
  if (!word) return false;

  if (mode === "start") {
    const target = nearestAudioAnchor(word.start);
    if (target == null) return false;
    setWordTiming(li, wi, target, word.end);
    return true;
  }
  if (mode === "end") {
    const target = nearestAudioAnchor(word.end);
    if (target == null) return false;
    setWordTiming(li, wi, word.start, target);
    return true;
  }

  const target = nearestAudioAnchor(word.start);
  if (target == null) return false;
  const duration = Math.max(word.end - word.start, MIN_WORD_DURATION);
  setWordTiming(li, wi, target, target + duration);
  return true;
}

function jumpSelectedToAudioAnchor(direction, mode) {
  if (!state.doc) return false;
  if (state.editMode === "line") {
    if (state.selectedLine == null) return false;
    const line = state.doc.lines[state.selectedLine];
    const target = directionalAudioAnchor(line.start, direction);
    if (target == null) return false;
    shiftLine(state.selectedLine, target - line.start);
    enforceNonOverlapCascade(state.selectedLine);
    return true;
  }

  if (!state.selected) return false;
  const { li, wi } = state.selected;
  const word = state.doc.lines[li].words[wi];
  if (!word) return false;

  if (mode === "start") {
    const target = directionalAudioAnchor(word.start, direction);
    if (target == null) return false;
    setWordTiming(li, wi, target, word.end);
    return true;
  }
  if (mode === "end") {
    const target = directionalAudioAnchor(word.end, direction);
    if (target == null) return false;
    setWordTiming(li, wi, word.start, target);
    return true;
  }

  const target = directionalAudioAnchor(word.start, direction);
  if (target == null) return false;
  const duration = Math.max(word.end - word.start, MIN_WORD_DURATION);
  setWordTiming(li, wi, target, target + duration);
  return true;
}

function buildAudioAnchors(buffer) {
  const channel = buffer.getChannelData(0);
  if (!channel || !channel.length) return [];

  const sampleRate = buffer.sampleRate;
  const frame = 1024;
  const hop = 256;
  const envelope = [];
  for (let start = 0; start + frame < channel.length; start += hop) {
    let energy = 0;
    for (let i = start; i < start + frame; i += 1) {
      const s = channel[i];
      energy += s * s;
    }
    envelope.push(Math.sqrt(energy / frame));
  }
  if (envelope.length < 3) return [];

  const flux = new Array(envelope.length).fill(0);
  let fluxSum = 0;
  for (let i = 1; i < envelope.length; i += 1) {
    const val = Math.max(0, envelope[i] - envelope[i - 1]);
    flux[i] = val;
    fluxSum += val;
  }
  const fluxMean = fluxSum / Math.max(1, flux.length - 1);
  const threshold = Math.max(0.006, fluxMean * 2.25);

  const anchors = [];
  const minGapFrames = Math.max(1, Math.round(0.05 * sampleRate / hop));
  let lastFrame = -minGapFrames;
  for (let i = 1; i < flux.length - 1; i += 1) {
    if (i - lastFrame < minGapFrames) continue;
    const cur = flux[i];
    if (cur < threshold) continue;
    if (cur >= flux[i - 1] && cur >= flux[i + 1]) {
      anchors.push(snap((i * hop) / sampleRate));
      lastFrame = i;
    }
  }
  return anchors;
}

async function analyzeAudioForAnchors() {
  if (!els.audio.src) return;
  const token = state.audioAnalysisToken + 1;
  state.audioAnalysisToken = token;
  state.audioAnalysisReady = false;
  state.audioAnchors = [];
  setStatus("Analyzing audio onsets for snap anchors...");
  try {
    const res = await fetch(els.audio.src);
    if (!res.ok) throw new Error(`Failed to read audio (${res.status})`);
    const data = await res.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const buffer = await audioCtx.decodeAudioData(data);
    const anchors = buildAudioAnchors(buffer);
    await audioCtx.close();
    if (token !== state.audioAnalysisToken) return;
    state.audioAnchors = anchors;
    state.audioAnalysisReady = true;
    if (anchors.length) {
      setStatus(`Audio anchors ready (${anchors.length} candidate onsets).`);
    } else {
      setStatus("No strong audio onsets detected; snap-to-audio disabled.");
    }
    render();
  } catch (err) {
    if (token !== state.audioAnalysisToken) return;
    state.audioAnchors = [];
    state.audioAnalysisReady = false;
    setStatus(`Audio analysis unavailable: ${String(err.message || err)}`, true);
  }
}

function buildTrack(line, li, viewWindow, pxPerSec) {
  const track = document.createElement("div");
  track.className = "track";
  track.style.width = `${Math.max(viewWindow.duration * pxPerSec, 300)}px`;

  function appendWordBlock(w, wordLi, wordWi, isContext) {
    const blockLeftSec = Math.max(w.start, viewWindow.start);
    const blockRightSec = Math.min(w.end, viewWindow.end);
    if (blockRightSec <= blockLeftSec) return;

    const word = document.createElement("div");
    word.className = isContext ? "word word-context" : "word";
    if (!isContext && state.selected && state.selected.li === wordLi && state.selected.wi === wordWi) {
      word.classList.add("selected");
    }
    word.style.left = `${(blockLeftSec - viewWindow.start) * pxPerSec}px`;
    word.style.width = `${Math.max((blockRightSec - blockLeftSec) * pxPerSec, 8)}px`;
    word.textContent = w.text;
    word.title = `${w.text}  [${w.start.toFixed(1)} - ${w.end.toFixed(1)}]`;

    if (!isContext && state.editMode === "word") {
      word.addEventListener("mousedown", (ev) => {
        ev.preventDefault();
        snapshotForUndo();
        selectWord(wordLi, wordWi);
        state.drag = {
          mode: "move",
          li: wordLi,
          wi: wordWi,
          startX: ev.clientX,
          moved: false,
          pxPerSec,
          startStart: w.start,
          startEnd: w.end,
        };
      });

      const left = document.createElement("div");
      left.className = "handle left";
      left.addEventListener("mousedown", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        snapshotForUndo();
        selectWord(wordLi, wordWi);
        state.drag = {
          mode: "resize-left",
          li: wordLi,
          wi: wordWi,
          startX: ev.clientX,
          moved: false,
          pxPerSec,
          startStart: w.start,
        };
      });

      const right = document.createElement("div");
      right.className = "handle right";
      right.addEventListener("mousedown", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        snapshotForUndo();
        selectWord(wordLi, wordWi);
        state.drag = {
          mode: "resize-right",
          li: wordLi,
          wi: wordWi,
          startX: ev.clientX,
          moved: false,
          pxPerSec,
          startEnd: w.end,
        };
      });

      word.appendChild(left);
      word.appendChild(right);
    }
    track.appendChild(word);
  }

  for (let wi = 0; wi < line.words.length; wi += 1) {
    appendWordBlock(line.words[wi], li, wi, false);
  }

  if (state.editMode === "line") {
    const bounds = getLineBounds(line);
    const box = document.createElement("div");
    box.className = "line-drag-box";
    if (state.selectedLine === li) {
      box.classList.add("selected");
    }
    box.style.left = `${(bounds.start - viewWindow.start) * pxPerSec}px`;
    box.style.width = `${Math.max((bounds.end - bounds.start) * pxPerSec, 18)}px`;
    box.title = `Line ${li + 1}: drag to shift entire line`;
    box.addEventListener("mousedown", (ev) => startLineDrag(li, ev, pxPerSec));
    track.appendChild(box);
  }

  return track;
}

function buildLineRuler(line, viewWindow, pxPerSec) {
  const width = Math.max(viewWindow.duration * pxPerSec, 300);
  const ruler = document.createElement("div");
  ruler.className = "line-ruler";
  ruler.style.width = `${width}px`;

  const progress = document.createElement("div");
  progress.className = "ruler-progress";
  ruler.appendChild(progress);

  // Place ticks at absolute integer seconds
  const firstTick = Math.ceil(viewWindow.start);
  const lastTick = Math.floor(viewWindow.start + viewWindow.duration);

  for (let t = firstTick; t <= lastTick; t += 1) {
    const offsetPx = (t - viewWindow.start) * pxPerSec;

    const major = document.createElement("div");
    major.className = "tick major";
    major.style.left = `${offsetPx}px`;
    ruler.appendChild(major);

    const label = document.createElement("div");
    label.className = "tick-label";
    label.style.left = `${offsetPx}px`;
    label.textContent = `${t.toFixed(0)}s`;
    ruler.appendChild(label);

    // Minor tick at half-second
    const minorT = t + 0.5;
    if (minorT < lastTick) {
      const minorPx = (minorT - viewWindow.start) * pxPerSec;
      const minor = document.createElement("div");
      minor.className = "tick minor";
      minor.style.left = `${minorPx}px`;
      ruler.appendChild(minor);
    }
  }

  if (state.audioAnalysisReady && state.audioAnchors.length) {
    for (const anchor of state.audioAnchors) {
      if (anchor < viewWindow.start || anchor > viewWindow.end) continue;
      const anchorPx = (anchor - viewWindow.start) * pxPerSec;
      const mark = document.createElement("div");
      mark.className = "tick audio-anchor";
      mark.style.left = `${anchorPx}px`;
      ruler.appendChild(mark);
    }
  }

  const playhead = document.createElement("div");
  playhead.className = "ruler-playhead";
  ruler.appendChild(playhead);

  return ruler;
}

function buildLineLane(line, li, row) {
  const viewWindow = getLineViewWindow(line);
  const laneMinWidth = Math.max(800, els.timeline.clientWidth - 360);
  const pxPerSec = Math.max(
    state.secondsToPx,
    laneMinWidth / Math.max(viewWindow.duration, 0.1)
  );
  
  // Store metadata on the ROW for consistent access
  row.dataset.start = viewWindow.start;
  row.dataset.duration = viewWindow.duration;
  row.dataset.pxPerSec = pxPerSec;

  const lane = document.createElement("div");
  lane.className = "line-lane";
  lane.appendChild(buildLineRuler(line, viewWindow, pxPerSec));
  lane.appendChild(buildTrack(line, li, viewWindow, pxPerSec));
  return lane;
}

function buildGapLane(gapStart, gapEnd) {
  const viewWindow = {
    start: Math.max(0, gapStart - 1.0),
    end: gapEnd + 1.0,
    duration: Math.max(2.0, gapEnd - gapStart + 2.0),
  };
  const laneMinWidth = Math.max(800, els.timeline.clientWidth - 360);
  const pxPerSec = Math.max(
    state.secondsToPx,
    laneMinWidth / Math.max(viewWindow.duration, 0.1)
  );
  const width = Math.max(viewWindow.duration * pxPerSec, 300);
  const whole = Math.ceil(viewWindow.duration);

  const lane = document.createElement("div");
  lane.className = "line-lane";

  const ruler = document.createElement("div");
  ruler.className = "line-ruler gap-ruler";
  ruler.style.width = `${width}px`;
  for (let t = 0; t <= whole; t += 1) {
    const leftPx = t * pxPerSec;
    const major = document.createElement("div");
    major.className = "tick major";
    major.style.left = `${leftPx}px`;
    ruler.appendChild(major);

    const label = document.createElement("div");
    label.className = "tick-label";
    label.style.left = `${leftPx}px`;
    label.textContent = `${(viewWindow.start + t).toFixed(1)}s`;
    ruler.appendChild(label);
  }

  const playhead = document.createElement("div");
  playhead.className = "ruler-playhead";
  ruler.appendChild(playhead);

  const track = document.createElement("div");
  track.className = "track gap-track";
  track.style.width = `${width}px`;

  const bar = document.createElement("div");
  bar.className = "gap-bar";
  bar.style.left = `${(gapStart - viewWindow.start) * pxPerSec}px`;
  bar.style.width = `${Math.max((gapEnd - gapStart) * pxPerSec, 8)}px`;
  track.appendChild(bar);

  lane.appendChild(ruler);
  lane.appendChild(track);
  return { lane, pxPerSec };
}

function updatePlaybackVisuals() {
  if (!state.doc) return;
  const currentTime = els.audio.currentTime;
  els.playbackInfo.textContent = `t=${currentTime.toFixed(1)}s`;

  const currentActiveLineIndex = currentLineIndexForTime(currentTime);
  const allRows = els.timeline.querySelectorAll(".lyric-line, .gap-row");
  
  allRows.forEach((row) => {
    const type = row.dataset.type;
    const vStart = parseFloat(row.dataset.start);
    const vDuration = parseFloat(row.dataset.duration);
    const pxPerSec = parseFloat(row.dataset.pxPerSec);

    const rel = currentTime - vStart;
    const clampedRel = Math.max(0, Math.min(rel, vDuration));
    
    const playhead = row.querySelector(".ruler-playhead");
    if (playhead) playhead.style.left = `${clampedRel * pxPerSec}px`;

    const progress = row.querySelector(".ruler-progress");
    if (progress) progress.style.width = `${clampedRel * pxPerSec}px`;

    if (type === "lyric") {
      const li = parseInt(row.dataset.li);
      const line = state.doc.lines[li];
      const isPlayingNow = currentTime >= line.start && currentTime <= line.end;
      const isCurrentLine = li === currentActiveLineIndex;
      row.classList.toggle("active-playing", isCurrentLine);

      // Update Word Classes
      const wordEls = row.querySelectorAll(".word:not(.word-context)");
      for (let wi = 0; wi < line.words.length; wi++) {
        const w = line.words[wi];
        const wEl = wordEls[wi];
        if (!wEl) continue;
        const isSung = currentTime >= w.end;
        const isPlaying = isPlayingNow && currentTime >= w.start && currentTime <= w.end;
        if (wEl.classList.contains("sung") !== isSung) wEl.classList.toggle("sung", isSung);
        if (wEl.classList.contains("playing") !== isPlaying) wEl.classList.toggle("playing", isPlaying);
      }
    }
  });

  // Follow playback/seek by keeping the active row near the top of the viewport.
  if (state.autoScrollEnabled && currentActiveLineIndex !== -1) {
    const activeChanged = currentActiveLineIndex !== state.lastActiveLineIndex;
    const shouldFollow = activeChanged || state.pendingSeekScroll;
    state.lastActiveLineIndex = currentActiveLineIndex;
    if (!state.drag) {
      const activeRow = els.timeline.querySelector(
        `.lyric-line[data-li="${currentActiveLineIndex}"]`
      );
      if (activeRow) {
        pinRowToPlaybackWindowTop(activeRow, {
          behavior:
            shouldFollow && !(state.pendingSeekScroll || els.audio.paused)
              ? "smooth"
              : "auto",
          force: shouldFollow,
        });
      }
      state.pendingSeekScroll = false;
    }
  } else {
    state.lastActiveLineIndex = -1;
    if (state.pendingSeekScroll) {
      state.pendingSeekScroll = false;
    }
  }
}

function currentLineIndexForTime(t) {
  if (!state.doc || !state.doc.lines || !state.doc.lines.length) return -1;
  const lines = state.doc.lines;
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (t >= line.start && t <= line.end) return i;
    if (t < line.start) return i;
  }
  return lines.length - 1;
}

function playbackViewportTopOffsetPx() {
  const sticky = document.querySelector(".sticky-controls");
  if (!sticky) return 12;
  const r = sticky.getBoundingClientRect();
  return Math.max(0, r.bottom) + 8;
}

function pinRowToPlaybackWindowTop(
  row,
  { behavior = "smooth", force = false } = {}
) {
  if (!state.autoScrollEnabled) return;
  const rowRect = row.getBoundingClientRect();
  const viewportTop = playbackViewportTopOffsetPx();
  const delta = rowRect.top - viewportTop;
  // Keep the current lyric line pinned near the top playback window.
  if (!force && Math.abs(delta) <= 16) {
    return;
  }
  const targetTop = window.scrollY + delta;
  window.scrollTo({
    top: Math.max(0, targetTop),
    behavior,
  });
}

function render() {
  els.timeline.innerHTML = "";
  if (!state.doc) {
    els.selectionInfo.textContent = "No word selected";
    return;
  }

  for (let li = 0; li < state.doc.lines.length; li += 1) {
    const line = state.doc.lines[li];
    const row = document.createElement("div");
    row.className = "line-row lyric-line";
    row.dataset.type = "lyric";
    row.dataset.li = li;
    if (state.selectedLine === li) {
      row.classList.add("selected-line");
    }

    const label = document.createElement("div");
    label.className = "line-text";
    label.textContent = `${String(li + 1).padStart(3, "0")}: ${line.text}`;
    label.addEventListener("click", () => {
      if (state.editMode === "line") selectLine(li);
    });

    row.appendChild(label);
    row.appendChild(buildLineLane(line, li, row));
    els.timeline.appendChild(row);

    if (li + 1 < state.doc.lines.length) {
      const nextLine = state.doc.lines[li + 1];
      const gapStart = line.end;
      const gapEnd = nextLine.start;
      const gapDuration = gapEnd - gapStart;
      if (gapDuration > 1.0) {
        const gapRow = document.createElement("div");
        gapRow.className = "gap-row";
        gapRow.dataset.type = "gap";
        gapRow.dataset.start = Math.max(0, gapStart - 1.0);
        gapRow.dataset.duration = Math.max(2.0, gapEnd - gapStart + 2.0);

        const gapLabel = document.createElement("div");
        gapLabel.className = "line-text gap-text";
        gapLabel.textContent = `Gap: ${gapDuration.toFixed(1)}s instrumental`;

        gapRow.appendChild(gapLabel);
        const gapLaneData = buildGapLane(gapStart, gapEnd);
        gapRow.dataset.pxPerSec = gapLaneData.pxPerSec;
        gapRow.appendChild(gapLaneData.lane);
        els.timeline.appendChild(gapRow);
      }
    }
  }

  if (state.editMode === "line" && state.selectedLine != null) {
    const line = state.doc.lines[state.selectedLine];
    els.selectionInfo.textContent = `Selected line ${state.selectedLine + 1}: start=${line.start.toFixed(1)}s`;
  } else if (state.selected) {
    const w = state.doc.lines[state.selected.li].words[state.selected.wi];
    els.selectionInfo.textContent = `Selected: "${w.text}"  [${w.start.toFixed(1)}s - ${w.end.toFixed(1)}s]`;
  } else {
    els.selectionInfo.textContent = "No word selected";
  }
  updateTelemetryInfo();
  
  // Update visuals immediately after full render
  updatePlaybackVisuals();
}

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || `Request failed: ${res.status}`);
  }
  return data;
}

async function loadTimingPath(path) {
  const data = await postJson("/api/load", { path });
  state.doc = data.document;
  state.tapCursor = firstWordRef();
  resetUndo();
  resetSessionStats();
  updateLineBounds();
  render();
  setStatus(`Loaded ${state.doc.lines.length} lines / ${lineWordCount(state.doc)} words.`);
  if (!els.savePath.value.trim()) {
    const guess = path.replace(/\.json$/i, ".gold.json");
    els.savePath.value = guess;
  }
}

function loadAudioPath(path) {
  if (!path) {
    setStatus("Audio path is required", true);
    return;
  }
  if (path.startsWith("/") || path.includes(":\\")) {
    els.audio.src = `/api/audio?path=${encodeURIComponent(path)}`;
  } else {
    els.audio.src = `/${path}`;
  }
  state.audioAnalysisReady = false;
  state.audioAnchors = [];
  if (state.doc) state.doc.audio_path = path;
  setStatus("Audio loaded. Waiting for metadata...");
}

els.loadTimingBtn.addEventListener("click", async () => {
  try {
    const path = els.timingPath.value.trim();
    if (!path) throw new Error("Timing path is required");
    await loadTimingPath(path);
  } catch (err) {
    setStatus(String(err.message || err), true);
  }
});

els.loadAudioBtn.addEventListener("click", () => {
  loadAudioPath(els.audioPath.value.trim());
});

els.saveBtn.addEventListener("click", async () => {
  try {
    if (!state.doc) throw new Error("Nothing loaded");
    const path = els.savePath.value.trim();
    if (!path) throw new Error("Save path is required");
    updateLineBounds();
    const payload = {
      path,
      document: state.doc,
    };
    const data = await postJson("/api/save", payload);
    setStatus(`Saved ${data.word_count} words to ${data.saved}`);
  } catch (err) {
    setStatus(String(err.message || err), true);
  }
});

els.playPauseBtn.addEventListener("click", () => {
  if (els.audio.paused) {
    els.audio.play();
  } else {
    els.audio.pause();
  }
});

els.startTapPassBtn.addEventListener("click", () => {
  if (!state.doc) return;
  const started = startTapPassFromBeginning();
  if (!started) {
    setStatus("Unable to start tap pass (no words loaded).", true);
    return;
  }
  setStatus("Tap pass started from 0:00. Press T on each word onset.");
  render();
});

els.tapNextBtn.addEventListener("click", () => {
  if (!state.doc) return;
  snapshotForUndo();
  const stamped = stampTapCursorWordToCurrentTime();
  if (!stamped) {
    state.undoStack.pop();
    setStatus("Tap pass is complete (no remaining words).");
    return;
  }
  recordEdit("nudge-fine");
  setStatus("Stamped tap-pass cursor word and advanced.");
  render();
});

els.wordModeBtn.addEventListener("click", () => setEditMode("word"));
els.lineModeBtn.addEventListener("click", () => setEditMode("line"));

els.audio.addEventListener("play", () => {
  els.playPauseBtn.textContent = "Pause";
});

els.audio.addEventListener("pause", () => {
  els.playPauseBtn.textContent = "Play";
});

els.audio.addEventListener("error", (e) => {
  console.error("Audio error:", e);
  setStatus(`Audio Error: ${els.audio.error?.message || "Unknown error"}`, true);
});

els.audio.addEventListener("loadedmetadata", () => {
  analyzeAudioForAnchors();
});

els.audio.addEventListener("timeupdate", () => {
  updatePlaybackVisuals();
});

els.audio.addEventListener("seeking", () => {
  state.pendingSeekScroll = state.autoScrollEnabled;
});

els.zoomRange.addEventListener("input", () => {
  state.secondsToPx = Number(els.zoomRange.value);
  render();
});

els.autoscrollToggle.addEventListener("change", () => {
  state.autoScrollEnabled = Boolean(els.autoscrollToggle.checked);
  state.pendingSeekScroll = false;
  if (state.autoScrollEnabled) {
    setStatus("Autoscroll enabled.");
  } else {
    setStatus("Autoscroll disabled.");
  }
});

els.tapModeToggle.addEventListener("change", () => {
  setTapPassMode(Boolean(els.tapModeToggle.checked));
  if (state.tapPassMode && !state.tapCursor) {
    state.tapCursor = firstWordRef();
  }
  setStatus(state.tapPassMode ? "Tap pass mode enabled." : "Tap pass mode disabled.");
  render();
});

document.addEventListener("mousemove", (ev) => {
  if (!state.drag || !state.doc) return;
  const dx = ev.clientX - state.drag.startX;
  if (Math.abs(dx) >= 2) {
    state.drag.moved = true;
  }
  const dt = snap(dx / (state.drag.pxPerSec || state.secondsToPx));
  const { li, wi } = state.drag;

  if (state.drag.mode === "move") {
    setWordTiming(li, wi, state.drag.startStart + dt, state.drag.startEnd + dt);
  } else if (state.drag.mode === "resize-left") {
    const end = state.doc.lines[li].words[wi].end;
    setWordTiming(li, wi, state.drag.startStart + dt, end);
  } else if (state.drag.mode === "resize-right") {
    const start = state.doc.lines[li].words[wi].start;
    setWordTiming(li, wi, start, state.drag.startEnd + dt);
  } else if (state.drag.mode === "line-move") {
    restoreAllLineWords(state.drag.originalAll);
    shiftLine(state.drag.li, dt);
    enforceNonOverlapCascade(state.drag.li);
  }

  render();
});

document.addEventListener("mouseup", () => {
  if (
    state.drag &&
    !state.drag.moved &&
    state.editMode === "word" &&
    typeof state.drag.li === "number" &&
    typeof state.drag.wi === "number"
  ) {
    const word = state.doc?.lines?.[state.drag.li]?.words?.[state.drag.wi];
    if (word && typeof word.start === "number") {
      state.pendingSeekScroll = state.autoScrollEnabled;
      els.audio.currentTime = Math.max(0, word.start);
      els.playbackInfo.textContent = `t=${els.audio.currentTime.toFixed(1)}s`;
    }
  }
  if (state.drag?.moved) {
    recordEdit("drag");
  }
  state.drag = null;
});

document.addEventListener("keydown", (ev) => {
  if (!state.doc) return;
  const target = ev.target;
  if (isTextEditable(target)) {
    return;
  }
  const key = ev.key;

  if (key === " ") {
    ev.preventDefault();
    if (els.audio.paused) {
      els.audio.play();
    } else {
      els.audio.pause();
    }
    return;
  }

  if (key === "ArrowLeft" || key === "ArrowRight") {
    ev.preventDefault();
    snapshotForUndo();
    const step = ev.ctrlKey || ev.metaKey ? KEYBOARD_COARSE_STEP : KEYBOARD_FINE_STEP;
    const delta = key === "ArrowRight" ? step : -step;

    if (state.editMode === "line") {
      if (state.selectedLine == null) return;
      shiftLine(state.selectedLine, delta);
      enforceNonOverlapCascade(state.selectedLine);
    } else {
      if (!state.selected) return;
      const { li, wi } = state.selected;
      if (ev.shiftKey) {
        adjustEnd(li, wi, delta);
      } else if (ev.altKey) {
        adjustStart(li, wi, delta);
      } else {
        moveWord(li, wi, delta);
      }
    }
    recordEdit(step === KEYBOARD_COARSE_STEP ? "nudge-coarse" : "nudge-fine");

    render();
    return;
  }

  const lower = key.toLowerCase();
  if (lower === "a" || lower === "s" || lower === "e") {
    ev.preventDefault();
    snapshotForUndo();
    const mode = lower === "s" ? "start" : lower === "e" ? "end" : "all";
    const snapped = snapSelectedToAudioAnchor(mode);
    if (!snapped) {
      state.undoStack.pop();
      setStatus(
        state.audioAnalysisReady
          ? "No nearby audio anchor found for snap."
          : "Audio anchors are not ready yet."
      );
      return;
    }
    recordEdit("snap");
    setStatus("Snapped to nearest audio anchor.");
    render();
    return;
  }

  if (lower === "t") {
    ev.preventDefault();
    snapshotForUndo();
    const stamped = stampTapCursorWordToCurrentTime();
    if (!stamped) {
      state.undoStack.pop();
      setStatus("Tap pass is complete (no remaining words).");
      return;
    }
    recordEdit("nudge-fine");
    setStatus("Stamped tap-pass cursor word and advanced.");
    render();
    return;
  }

  if (lower === "r") {
    ev.preventDefault();
    const started = startTapPassFromBeginning();
    if (!started) {
      setStatus("Unable to start tap pass (no words loaded).", true);
      return;
    }
    setStatus("Tap pass restarted from 0:00. Press T on each word onset.");
    render();
    return;
  }

  if (key === "[" || key === "]") {
    ev.preventDefault();
    snapshotForUndo();
    const direction = key === "]" ? 1 : -1;
    const mode = ev.shiftKey ? "end" : ev.altKey ? "start" : "all";
    const jumped = jumpSelectedToAudioAnchor(direction, mode);
    if (!jumped) {
      state.undoStack.pop();
      setStatus(
        state.audioAnalysisReady
          ? "No further audio anchor in that direction."
          : "Audio anchors are not ready yet."
      );
      return;
    }
    recordEdit("jump");
    setStatus(`Jumped to ${direction > 0 ? "next" : "previous"} audio anchor.`);
    render();
  }
});

document.addEventListener("keydown", (ev) => {
  const target = ev.target;
  if (isTextEditable(target)) {
    return;
  }
  if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "z") {
    ev.preventDefault();
    undo();
  }
});

async function applyUrlParams() {
  const params = new URLSearchParams(window.location.search);
  const timing =
    params.get("timing") ||
    params.get("timingFile") ||
    params.get("timing_path");
  const audio =
    params.get("audio") || params.get("audioFile") || params.get("audio_path");
  const save =
    params.get("save") || params.get("gold") || params.get("goldFile");
  const autoscroll = params.get("autoscroll");
  const tapMode = params.get("tapmode");

  if (timing) els.timingPath.value = timing;
  if (audio) els.audioPath.value = audio;
  if (save) els.savePath.value = save;
  if (autoscroll != null) {
    const enabled = !["0", "false", "off"].includes(autoscroll.toLowerCase());
    state.autoScrollEnabled = enabled;
    els.autoscrollToggle.checked = enabled;
  }
  if (tapMode != null) {
    const enabled = !["0", "false", "off"].includes(tapMode.toLowerCase());
    setTapPassMode(enabled);
  }

  if (audio) {
    loadAudioPath(audio);
  }
  if (timing) {
    try {
      await loadTimingPath(timing);
    } catch (err) {
      setStatus(String(err.message || err), true);
    }
  }
}

function startAnimationLoop() {
  function frame() {
    updatePlaybackVisuals();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

setStatus("Load timing + audio to start.");
resetSessionStats();
setEditMode("word");
setTapPassMode(false);
els.timeline.addEventListener("mousedown", blurEditableFocus);
applyUrlParams();
startAnimationLoop();
