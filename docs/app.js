import { NTupleAI, isTerminal, planMove } from './ai.js';

const boardEl = document.getElementById('board');
const gridBg = document.getElementById('gridBg');
const tileLayer = document.getElementById('tileLayer');
const scoreEl = document.getElementById('score');
const bestEl = document.getElementById('best');
const statusEl = document.getElementById('status');
const perfEl = document.getElementById('perf');
const modelInfoEl = document.getElementById('modelInfo');
const overlayEl = document.getElementById('overlay');
const overlayTextEl = document.getElementById('overlayText');
const btnAuto = document.getElementById('btnAuto');
const btnStep = document.getElementById('btnStep');
const btnReset = document.getElementById('btnReset');
const btnRetry = document.getElementById('btnRetry');
const speedEl = document.getElementById('speed');
const deepEl = document.getElementById('deep');

const KEY_TO_DIR = {
    ArrowLeft: 0,
    ArrowRight: 1,
    ArrowUp: 2,
    ArrowDown: 3,
    a: 0,
    d: 1,
    w: 2,
    s: 3,
};

/** Tiles live 92 ms in the "dying" state while they slide into a merge. */
const SLIDE_MS = 92;

let ai = null;
let grid = new Array(16).fill(null);
let score = 0;
let best = Number(localStorage.getItem('ntuple2048-best') || 0);
let running = false;
let timer = null;
let thinkMs = 0;
let thinkCount = 0;

for (let i = 0; i < 16; i++) {
    const cell = document.createElement('div');
    cell.className = 'grid-cell';
    gridBg.appendChild(cell);
}

// Tile geometry is measured from the background grid instead of being derived
// independently, otherwise sub-pixel rounding of `clamp()` inside `calc()`
// makes the two layers drift apart (the error accumulates per column/row).
const geo = { cell: 0, gap: 0, ox: 0, oy: 0 };

function measure() {
    const cells = gridBg.children;
    if (cells.length < 2) return;
    const first = cells[0].getBoundingClientRect();
    const second = cells[1].getBoundingClientRect();
    const layer = tileLayer.getBoundingClientRect();
    if (first.width < 1) return;
    geo.cell = first.width;
    geo.gap = Math.max(0, second.left - first.right);
    geo.ox = first.left - layer.left;
    geo.oy = first.top - layer.top;
}

function applyLayout() {
    if (geo.cell < 1) return;
    tileLayer.style.setProperty('--cell', `${geo.cell}px`);
    tileLayer.style.setProperty('--tile-gap', `${geo.gap}px`);
    tileLayer.style.setProperty('--ox', `${geo.ox}px`);
    tileLayer.style.setProperty('--oy', `${geo.oy}px`);
}

function relayout() {
    measure();
    applyLayout();
}

// ------------------------------------------------------------------ tiles

function makeTile(value, row, col, extra = {}) {
    return { value, row, col, el: null, inner: null, ...extra };
}

function attach(tile) {
    if (!tile.el) {
        const el = document.createElement('div');
        el.className = 'tile';
        if (tile.isNew) el.classList.add('is-new');
        if (tile.isMerged) el.classList.add('is-merged');
        const inner = document.createElement('div');
        inner.className = 'inner';
        el.appendChild(inner);
        el.addEventListener('animationend', () => {
            el.classList.remove('is-new', 'is-merged');
        });
        tile.el = el;
        tile.inner = inner;
        tileLayer.appendChild(el);
    }
    tile.el.style.setProperty('--row', tile.row);
    tile.el.style.setProperty('--col', tile.col);
    tile.el.dataset.v = String(tile.value);
    const text = String(2 ** tile.value);
    if (tile.inner.textContent !== text) tile.inner.textContent = text;
}

function render(dying = []) {
    for (const tile of grid) {
        if (tile && !tile.pending) attach(tile);
    }
    for (const tile of dying) attach(tile);

    // Safety net: drop any DOM node that no longer belongs to a live tile.
    const live = new Set();
    for (const tile of grid) if (tile && tile.el) live.add(tile.el);
    for (const tile of dying) if (tile && tile.el) live.add(tile.el);
    for (const node of Array.from(tileLayer.children)) {
        if (!live.has(node)) node.remove();
    }
}

function cellsFromGrid() {
    const cells = new Uint8Array(16);
    for (let i = 0; i < 16; i++) if (grid[i]) cells[i] = grid[i].value;
    return cells;
}

function spawnTile() {
    const empty = [];
    for (let i = 0; i < 16; i++) if (!grid[i]) empty.push(i);
    if (!empty.length) return -1;
    const position = empty[Math.floor(Math.random() * empty.length)];
    const tile = makeTile(Math.random() < 0.9 ? 1 : 2, Math.floor(position / 4), position % 4, {
        isNew: true,
    });
    grid[position] = tile;
    render();
    return position;
}

// ------------------------------------------------------------------- move

function moveTiles(dir) {
    const plan = planMove(cellsFromGrid(), dir);
    if (!plan.moved) return null;

    // `plan` indexes the *original* board, so all lookups must go through a
    // snapshot.  Reading `grid` after applying the slides would pick up tiles
    // that already moved in, leaving the real tiles orphaned forever.
    const before = grid;
    const next = before.slice();
    const dying = [];
    const created = [];

    for (const { from, to } of plan.moves) {
        const tile = before[from];
        next[from] = null;
        next[to] = tile;
        tile.row = Math.floor(to / 4);
        tile.col = to % 4;
    }

    for (const merge of plan.merges) {
        const a = before[merge.a];
        const b = before[merge.b];
        const row = Math.floor(merge.to / 4);
        const col = merge.to % 4;

        if (a) {
            a.row = row;
            a.col = col;
            dying.push(a);
        }
        if (b) {
            b.row = row;
            b.col = col;
            dying.push(b);
        }

        // `merge.a` may already hold a tile that slid in from `plan.moves`
        // (e.g. [1,1,2,0] LEFT: the 2 slides onto index 1 while index 1's 1
        // merges away).  Only clear the slot when it still points at the
        // original tile, otherwise the slid-in one becomes an orphan.
        if (next[merge.a] === a) next[merge.a] = null;
        if (next[merge.b] === b) next[merge.b] = null;

        const merged = makeTile(merge.value, row, col, { pending: true });
        next[merge.to] = merged;
        created.push(merged);
    }

    grid = next;
    render(dying);

    setTimeout(() => {
        for (const tile of dying) {
            if (tile.el) {
                tile.el.remove();
                tile.el = null;
                tile.inner = null;
            }
        }
        for (const tile of created) {
            tile.pending = false;
            tile.isMerged = true;
        }
        render();
    }, SLIDE_MS);

    score += plan.gain;
    return plan.cells;
}

// -------------------------------------------------------------------- UI

function updateScores() {
    scoreEl.textContent = String(score);
    if (score > best) {
        best = score;
        localStorage.setItem('ntuple2048-best', String(best));
    }
    bestEl.textContent = String(best);
}

function updatePerf() {
    perfEl.textContent = thinkCount
        ? `${(thinkMs / thinkCount).toFixed(2)} ms/步 · ${thinkCount} 次决策`
        : '';
}

function stopAuto() {
    running = false;
    clearTimeout(timer);
    timer = null;
    btnAuto.textContent = 'AI 自动';
    btnAuto.classList.remove('running');
}

function endGame(text) {
    stopAuto();
    overlayTextEl.textContent = text;
    overlayEl.classList.remove('hidden');
}

function reset() {
    stopAuto();
    overlayEl.classList.add('hidden');
    tileLayer.innerHTML = '';
    grid = new Array(16).fill(null);
    score = 0;
    thinkMs = 0;
    thinkCount = 0;
    spawnTile();
    spawnTile();
    updateScores();
    updatePerf();
}

function step(dir) {
    if (!moveTiles(dir)) return false;
    spawnTile();
    updateScores();
    if (isTerminal(cellsFromGrid(), ai.lut)) {
        endGame(`无法移动 · ${score} 分`);
        return false;
    }
    return true;
}

function chooseMove() {
    const cells = cellsFromGrid();
    return deepEl.checked ? ai.bestMoveDeep(cells, 2) : ai.bestMove(cells);
}

function delayMs() {
    // slider 0..100 -> 240ms..8ms per move
    return Math.round(240 - (Number(speedEl.value) / 100) * 232);
}

function aiTick() {
    if (!running || !ai) return;
    const started = performance.now();
    const dir = chooseMove();
    thinkMs += performance.now() - started;
    thinkCount += 1;
    updatePerf();

    if (dir < 0) {
        endGame(`无法移动 · ${score} 分`);
        return;
    }
    const alive = step(dir);
    if (alive && running) timer = setTimeout(aiTick, delayMs());
}

function startAuto() {
    if (!ai) return;
    running = true;
    btnAuto.textContent = '暂停';
    btnAuto.classList.add('running');
    overlayEl.classList.add('hidden');
    timer = setTimeout(aiTick, 30);
}

btnAuto.addEventListener('click', () => {
    if (running) stopAuto();
    else startAuto();
});

btnStep.addEventListener('click', () => {
    if (!ai || running) return;
    const dir = chooseMove();
    if (dir >= 0) step(dir);
});

btnReset.addEventListener('click', reset);
btnRetry.addEventListener('click', reset);

document.addEventListener('keydown', (event) => {
    const dir = KEY_TO_DIR[event.key];
    if (dir === undefined) {
        if (event.key === ' ') {
            event.preventDefault();
            btnAuto.click();
        }
        return;
    }
    event.preventDefault();
    if (!ai || running) return;
    step(dir);
});

let touchStart = null;
boardEl.addEventListener(
    'touchstart',
    (event) => {
        const touch = event.changedTouches[0];
        touchStart = { x: touch.clientX, y: touch.clientY };
    },
    { passive: true },
);
boardEl.addEventListener(
    'touchend',
    (event) => {
        if (!touchStart || !ai || running) return;
        const touch = event.changedTouches[0];
        const dx = touch.clientX - touchStart.x;
        const dy = touch.clientY - touchStart.y;
        touchStart = null;
        if (Math.max(Math.abs(dx), Math.abs(dy)) < 24) return;
        const dir = Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 1 : 0) : (dy > 0 ? 3 : 2);
        step(dir);
    },
    { passive: true },
);

// ------------------------------------------------------------------- boot

async function boot() {
    relayout();
    reset();
    try {
        const started = performance.now();
        ai = await NTupleAI.load('model.bin.gz');
        const seconds = (performance.now() - started) / 1000;
        modelInfoEl.textContent = `base ${ai.model.base} · ${ai.model.n6}×6-tuple + ${ai.model.n4}×4-tuple`;
        statusEl.textContent = `模型已加载（${seconds.toFixed(1)}s）`;
        btnAuto.disabled = false;
        btnStep.disabled = false;
        relayout();
        reset();
    } catch (error) {
        console.error(error);
        statusEl.textContent = `模型加载失败：${error.message}`;
        modelInfoEl.textContent = 'no model';
    }
}

// Debug hook: lets the headless regression test compare the logical board with
// what is actually in the DOM (catching orphaned tiles that never get removed).
window.__ntuple2048 = {
    state: () => ({
        logical: grid.filter(Boolean).map((t) => ({ row: t.row, col: t.col, value: t.value })),
        dom: tileLayer.childElementCount,
        score,
    }),
};

window.addEventListener('resize', () => relayout());

boot();
