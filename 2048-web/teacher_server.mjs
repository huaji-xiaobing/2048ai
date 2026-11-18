// Expert trajectory generator built on the original web demo (ai_core.wasm).
//
// Usage:
//   node teacher_server.mjs --games 100 --seed 0 --speed 1.0 --out traj.bin
//
// The policy is *exactly* the one shipped in worker.js: `CoreAILogic`
// (iterative deepening + L3Manager EGTB probes) from `expert_core.mjs`, which is
// an untouched extraction of that file.  `--speed` maps to the demo's speed
// slider: `time_limit_ratio = 10 ** ((100 - slider) / 100)`, so 1.0 is the
// default slider position, <1 is faster/weaker, >1 is slower/stronger.
//
// Output file layout (little endian):
//   magic  "T2048V1\0"           8 bytes
//   u32    n_states
//   u32    depth        (0: policy-managed)
//   u64    seed
//   per state: 16 x u8 (cell exponents, cell 0 first) + 1 x u8 (action 0..3)
//
// The action maps to the trainer's LEFT/RIGHT/UP/DOWN = 0/1/2/3 order; the wasm
// API uses 1..4 in the same order, so we subtract one.

import createAICore from './ai_core.js';
import { CoreAILogic, setCore } from './expert_core.mjs';
import fs from 'node:fs';

function parseArgs(argv) {
    const out = {
        games: 20, seed: 0, speed: 1.0, out: 'traj.bin',
        spawn4: 0.1, maxSteps: 400000,
    };
    for (let i = 0; i < argv.length; i += 2) {
        const key = argv[i].replace(/^--/, '');
        const val = argv[i + 1];
        out[key] = key === 'out' ? val : Number(val);
    }
    return out;
}

const opts = parseArgs(process.argv.slice(2));
const core = await createAICore();
setCore(core);

const encode = (cells) => {
    let r = 0n;
    for (let i = 0; i < 16; i++) r |= BigInt(cells[i]) << BigInt((15 - i) * 4);
    return r;
};

const decodeInto = (board, cells) => {
    const b = BigInt(board);
    for (let i = 0; i < 16; i++) cells[i] = Number((b >> BigInt((15 - i) * 4)) & 0xFn);
    return cells;
};

function mulberry32(a) {
    return function () {
        a |= 0; a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function spawn(cells, rnd) {
    const empty = [];
    for (let i = 0; i < 16; i++) if (cells[i] === 0) empty.push(i);
    if (!empty.length) return false;
    cells[empty[Math.floor(rnd() * empty.length)]] = rnd() < 0.9 ? 1 : 2;
    return true;
}

const player = new core.AIPlayer(0n);
player.spawn_rate4 = opts.spawn4;

const logic = new CoreAILogic();
logic.time_limit_ratio = opts.speed;

const boards = [];
const actions = [];
const counts = new Array(16).fill(0);
const cells = new Array(16).fill(0);
const nextCells = new Array(16).fill(0);

for (let g = 0; g < opts.games; g++) {
    const rnd = mulberry32(opts.seed + g * 7919);
    cells.fill(0);
    spawn(cells, rnd);
    spawn(cells, rnd);

    let steps = 0;
    while (steps < opts.maxSteps) {
        counts.fill(0);
        for (let i = 0; i < 16; i++) counts[cells[i]]++;
        player.reset_board(encode(cells));
        const op = logic.calculate_step(player, cells, counts);
        if (!op || op < 1 || op > 4) break;

        decodeInto(core.move_board(encode(cells), op), nextCells);
        let changed = false;
        for (let i = 0; i < 16; i++) if (nextCells[i] !== cells[i]) { changed = true; break; }
        if (!changed) break;

        // record (state, action) before applying the move
        for (let i = 0; i < 16; i++) boards.push(cells[i]);
        actions.push(op - 1);

        cells.splice(0, 16, ...nextCells);
        steps++;
        spawn(cells, rnd);
    }
    if (process.env.TEACHER_VERBOSE) {
        console.error(`game ${g}: states=${steps} maxTile=${2 ** Math.max(...cells)}`);
    }
}

const n = actions.length;
const header = Buffer.alloc(24);
header.write('T2048V1\0', 0, 'latin1');
header.writeUInt32LE(n, 8);
header.writeUInt32LE(0, 12);
header.writeBigUInt64LE(BigInt(opts.seed), 16);
const body = Buffer.alloc(n * 17);
for (let i = 0; i < n; i++) {
    for (let c = 0; c < 16; c++) body.writeUInt8(boards[i * 16 + c], i * 17 + c);
    body.writeUInt8(actions[i], i * 17 + 16);
}
fs.writeFileSync(opts.out, Buffer.concat([header, body]));
console.log(JSON.stringify({ ok: true, states: n, games: opts.games, speed: opts.speed, out: opts.out }));
