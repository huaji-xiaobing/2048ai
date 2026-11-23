import { NTupleAI, applyMove, isTerminal, newBoard, spawn } from './ai.js';

const boardEl = document.getElementById('board');
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

let ai = null;
let cells = newBoard();
let score = 0;
let best = Number(localStorage.getItem('ntuple2048-best') || 0);
let running = false;
let timer = null;
let thinkMs = 0;
let thinkCount = 0;

const cellEls = [];
for (let i = 0; i < 16; i++) {
    const el = document.createElement('div');
    el.className = 'cell';
    boardEl.appendChild(el);
    cellEls.push(el);
}

function render(popped = -1) {
    for (let i = 0; i < 16; i++) {
        const value = cells[i];
        const el = cellEls[i];
        const cls = value ? `cell v${value}` : 'cell';
        if (el.className !== cls) el.className = cls;
        const text = value ? String(2 ** value) : '';
        if (el.textContent !== text) el.textContent = text;
    }
    if (popped >= 0) {
        const el = cellEls[popped];
        el.classList.remove('pop');
        void el.offsetWidth;
        el.classList.add('pop');
        setTimeout(() => el.classList.remove('pop'), 200);
    }
}

function updateScores() {
    scoreEl.textContent = String(score);
    if (score > best) {
        best = score;
        localStorage.setItem('ntuple2048-best', String(best));
    }
    bestEl.textContent = String(best);
}

function updatePerf() {
    if (!thinkCount) {
        perfEl.textContent = '';
        return;
    }
    perfEl.textContent = `${(thinkMs / thinkCount).toFixed(2)} ms/步 · ${thinkCount} 次决策`;
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
    cells = newBoard();
    score = 0;
    thinkMs = 0;
    thinkCount = 0;
    updateScores();
    updatePerf();
    render();
    if (ai) statusEl.textContent = '就绪';
}

function step(dir) {
    const result = applyMove(cells, dir, ai.lut);
    if (!result.changed) return false;
    cells = result.cells;
    score += result.gain;
    const popped = spawn(cells);
    render(popped);
    updateScores();
    if (isTerminal(cells, ai.lut)) {
        endGame(`无法移动 · ${score} 分`);
        return false;
    }
    return true;
}

function chooseMove() {
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
boardEl.addEventListener('touchstart', (event) => {
    const touch = event.changedTouches[0];
    touchStart = { x: touch.clientX, y: touch.clientY };
}, { passive: true });

boardEl.addEventListener('touchend', (event) => {
    if (!touchStart || !ai || running) return;
    const touch = event.changedTouches[0];
    const dx = touch.clientX - touchStart.x;
    const dy = touch.clientY - touchStart.y;
    touchStart = null;
    if (Math.max(Math.abs(dx), Math.abs(dy)) < 24) return;
    const dir = Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 1 : 0) : (dy > 0 ? 3 : 2);
    step(dir);
}, { passive: true });

async function boot() {
    updateScores();
    render();
    try {
        const started = performance.now();
        ai = await NTupleAI.load('model.bin.gz');
        const seconds = (performance.now() - started) / 1000;
        modelInfoEl.textContent = `base ${ai.model.base} · ${ai.model.n6}×6-tuple + ${ai.model.n4}×4-tuple`;
        statusEl.textContent = `模型已加载（${seconds.toFixed(1)}s）`;
        btnAuto.disabled = false;
        btnStep.disabled = false;
        reset();
    } catch (error) {
        console.error(error);
        statusEl.textContent = `模型加载失败：${error.message}`;
        modelInfoEl.textContent = 'no model';
    }
}

boot();
