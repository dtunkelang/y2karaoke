const state = {
  doc: null,
  secondsToPx: 90,
  editMode: "word",
  selected: null,
  selectedLine: null,
  drag: null,
  undoStack: [],
};
const MIN_WORD_DURATION = 0.1;

const els = {
  status: document.getElementById("status"),
  timingPath: document.getElementById("timingPath"),
  loadTimingBtn: document.getElementById("loadTimingBtn"),
  audioPath: document.getElementById("audioPath"),
  loadAudioBtn: document.getElementById("loadAudioBtn"),
  savePath: document.getElementById("savePath"),
  saveBtn: document.getElementById("saveBtn"),
  playPauseBtn: document.getElementById("playPauseBtn"),
  wordModeBtn: document.getElementById("wordModeBtn"),
  lineModeBtn: document.getElementById("lineModeBtn"),
  zoomRange: document.getElementById("zoomRange"),
  playbackInfo: document.getElementById("playbackInfo"),
  selectionInfo: document.getElementById("selectionInfo"),
  timeline: document.getElementById("timeline"),
  audio: document.getElementById("audio"),
};

function snap(v) {
  return Math.round(v * 10) / 10;
}

function setStatus(msg, isError = false) {
  els.status.textContent = msg;
  els.status.style.color = isError ? "#a92323" : "#5e6c5f";
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

function findNextWord(li, wi) {
  const line = state.doc.lines[li];
  if (wi + 1 < line.words.length) return line.words[wi + 1];
  if (li + 1 >= state.doc.lines.length) return null;
  return state.doc.lines[li + 1].words[0];
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

function buildTrack(line, li, viewWindow, pxPerSec) {
  const track = document.createElement("div");
  track.className = "track";
  track.style.width = `${Math.max(viewWindow.duration * pxPerSec, 300)}px`;

  const lineBounds = getLineBounds(line);
  const allRefs = _allWordRefs();
  const contextRefs = allRefs.filter((ref) => {
    if (ref.li === li) return false;
    const ws = ref.word.start;
    const we = ref.word.end;
    return we > viewWindow.start && ws < viewWindow.end;
  });

  function appendWordBlock(w, wordLi, wordWi, opts = {}) {
    const {
      isContext = false,
      clipStart = viewWindow.start,
      clipEnd = viewWindow.end,
    } = opts;
    const blockLeftSec = Math.max(w.start, viewWindow.start, clipStart);
    const blockRightSec = Math.min(w.end, viewWindow.end, clipEnd);
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
    if (!isContext) {
      if (els.audio.currentTime >= w.end) {
        word.classList.add("sung");
      } else if (els.audio.currentTime >= w.start && els.audio.currentTime <= w.end) {
        word.classList.add("playing");
      }
    }

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

  const contextZones = [
    { start: viewWindow.start, end: lineBounds.start },
    { start: lineBounds.end, end: viewWindow.end },
  ].filter((z) => z.end > z.start);

  for (const ref of contextRefs) {
    for (const zone of contextZones) {
      if (ref.word.end <= zone.start || ref.word.start >= zone.end) {
        continue;
      }
      appendWordBlock(ref.word, ref.li, ref.wi, {
        isContext: true,
        clipStart: zone.start,
        clipEnd: zone.end,
      });
    }
  }

  for (let wi = 0; wi < line.words.length; wi += 1) {
    const w = line.words[wi];
    appendWordBlock(w, li, wi, { isContext: false });
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
  const bounds = getLineBounds(line);
  const width = Math.max(viewWindow.duration * pxPerSec, 300);
  const whole = Math.ceil(viewWindow.duration);

  const ruler = document.createElement("div");
  ruler.className = "line-ruler";
  ruler.style.width = `${width}px`;

  const rel = els.audio.currentTime - viewWindow.start;
  const clampedRel = Math.max(0, Math.min(rel, viewWindow.duration));
  const progress = document.createElement("div");
  progress.className = "ruler-progress";
  progress.style.width = `${clampedRel * pxPerSec}px`;
  ruler.appendChild(progress);

  if (els.audio.currentTime >= bounds.start && els.audio.currentTime <= bounds.end) {
    ruler.classList.add("active");
  }

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

    if (t < whole) {
      const minor = document.createElement("div");
      minor.className = "tick minor";
      minor.style.left = `${(t + 0.5) * pxPerSec}px`;
      ruler.appendChild(minor);
    }
  }

  const playhead = document.createElement("div");
  playhead.className = "ruler-playhead";
  playhead.style.left = `${clampedRel * pxPerSec}px`;
  ruler.appendChild(playhead);

  return ruler;
}

function buildLineLane(line, li) {
  const viewWindow = getLineViewWindow(line);
  const laneMinWidth = Math.max(800, els.timeline.clientWidth - 360);
  const pxPerSec = Math.max(
    state.secondsToPx,
    laneMinWidth / Math.max(viewWindow.duration, 0.1)
  );
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
  const rel = Math.max(0, Math.min(els.audio.currentTime - viewWindow.start, viewWindow.duration));
  playhead.style.left = `${rel * pxPerSec}px`;
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
  return lane;
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
    row.className = "line-row";
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
    row.appendChild(buildLineLane(line, li));
    els.timeline.appendChild(row);

    if (li + 1 < state.doc.lines.length) {
      const nextLine = state.doc.lines[li + 1];
      const gapStart = line.end;
      const gapEnd = nextLine.start;
      const gapDuration = gapEnd - gapStart;
      if (gapDuration > 1.0) {
        const gapRow = document.createElement("div");
        gapRow.className = "line-row gap-row";

        const gapLabel = document.createElement("div");
        gapLabel.className = "line-text gap-text";
        gapLabel.textContent = `Gap: ${gapDuration.toFixed(1)}s instrumental`;

        gapRow.appendChild(gapLabel);
        gapRow.appendChild(buildGapLane(gapStart, gapEnd));
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
  resetUndo();
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
  els.audio.src = `/api/audio?path=${encodeURIComponent(path)}`;
  if (state.doc) state.doc.audio_path = path;
  setStatus("Audio loaded.");
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

els.wordModeBtn.addEventListener("click", () => setEditMode("word"));
els.lineModeBtn.addEventListener("click", () => setEditMode("line"));

els.audio.addEventListener("play", () => {
  els.playPauseBtn.textContent = "Pause";
});

els.audio.addEventListener("pause", () => {
  els.playPauseBtn.textContent = "Play";
});

els.audio.addEventListener("timeupdate", () => {
  els.playbackInfo.textContent = `t=${els.audio.currentTime.toFixed(1)}s`;
  render();
});

els.zoomRange.addEventListener("input", () => {
  state.secondsToPx = Number(els.zoomRange.value);
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
      els.audio.currentTime = Math.max(0, word.start);
      els.playbackInfo.textContent = `t=${els.audio.currentTime.toFixed(1)}s`;
    }
  }
  state.drag = null;
});

document.addEventListener("keydown", (ev) => {
  if (!state.doc) return;
  if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;

  ev.preventDefault();
  snapshotForUndo();
  const delta = ev.key === "ArrowRight" ? 0.1 : -0.1;

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

  render();
});

document.addEventListener("keydown", (ev) => {
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

  if (timing) els.timingPath.value = timing;
  if (audio) els.audioPath.value = audio;
  if (save) els.savePath.value = save;

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

setStatus("Load timing + audio to start.");
setEditMode("word");
applyUrlParams();
