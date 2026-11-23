// Minimal, dependency-free inference for an n-tuple 2048 value table.
//
// The network scores *afterstates*: for a board we evaluate the four possible
// slides and pick the one with the highest value.  The table is the same one
// the Python trainer produces, quantised to int16 per pattern.

export const LEFT = 0;
export const RIGHT = 1;
export const UP = 2;
export const DOWN = 3;
export const DIRECTIONS = [LEFT, RIGHT, UP, DOWN];

export const BOARD_SIZE = 4;
export const CELLS = BOARD_SIZE * BOARD_SIZE;

const LINES = (() => {
    const out = [[], [], [], []];
    for (let k = 0; k < 4; k++) {
        out[LEFT].push([k * 4 + 0, k * 4 + 1, k * 4 + 2, k * 4 + 3]);
        out[RIGHT].push([k * 4 + 3, k * 4 + 2, k * 4 + 1, k * 4 + 0]);
        out[UP].push([0 * 4 + k, 1 * 4 + k, 2 * 4 + k, 3 * 4 + k]);
        out[DOWN].push([3 * 4 + k, 2 * 4 + k, 1 * 4 + k, 0 * 4 + k]);
    }
    return out;
})();

/** Pre-computes the 65536-entry line table (slide + merge + score). */
export function buildMoveLUT() {
    const move = new Uint16Array(65536);
    const gain = new Uint32Array(65536);
    for (let code = 0; code < 65536; code++) {
        const line = [code & 15, (code >> 4) & 15, (code >> 8) & 15, (code >> 12) & 15];
        const vals = line.filter((v) => v !== 0);
        const out = [];
        let g = 0;
        let i = 0;
        while (i < vals.length) {
            if (i + 1 < vals.length && vals[i] === vals[i + 1]) {
                const merged = Math.min(vals[i] + 1, 15);
                out.push(merged);
                g += 1 << merged;
                i += 2;
            } else {
                out.push(vals[i]);
                i += 1;
            }
        }
        while (out.length < 4) out.push(0);
        let nc = 0;
        for (let t = 0; t < 4; t++) nc |= out[t] << (4 * t);
        move[code] = nc;
        gain[code] = g;
    }
    return { move, gain };
}

/** Applies `dir` to a 16-cell Uint8Array, returns `{cells, gain, changed}`. */
export function applyMove(cells, dir, lut) {
    const out = new Uint8Array(CELLS);
    let gain = 0;
    for (let li = 0; li < 4; li++) {
        const line = LINES[dir][li];
        const code =
            cells[line[0]] | (cells[line[1]] << 4) | (cells[line[2]] << 8) | (cells[line[3]] << 12);
        const nc = lut.move[code];
        gain += lut.gain[code];
        out[line[0]] = nc & 15;
        out[line[1]] = (nc >> 4) & 15;
        out[line[2]] = (nc >> 8) & 15;
        out[line[3]] = (nc >> 12) & 15;
    }
    let changed = false;
    for (let i = 0; i < CELLS; i++) {
        if (out[i] !== cells[i]) {
            changed = true;
            break;
        }
    }
    return { cells: out, gain, changed };
}

export function emptyCells(cells) {
    const list = [];
    for (let i = 0; i < CELLS; i++) if (cells[i] === 0) list.push(i);
    return list;
}

export function spawn(cells, rng = Math.random) {
    const empty = emptyCells(cells);
    if (!empty.length) return -1;
    const pos = empty[Math.floor(rng() * empty.length)];
    cells[pos] = rng() < 0.9 ? 1 : 2;
    return pos;
}

export function newBoard(rng = Math.random) {
    const cells = new Uint8Array(CELLS);
    spawn(cells, rng);
    spawn(cells, rng);
    return cells;
}

export function isTerminal(cells, lut) {
    return !DIRECTIONS.some((d) => applyMove(cells, d, lut).changed);
}

function defaultPatterns6() {
    const patterns = [];
    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 2; c++) {
            const p = [];
            for (let dr = 0; dr < 2; dr++) for (let dc = 0; dc < 3; dc++) p.push((r + dr) * 4 + c + dc);
            patterns.push(p);
        }
    }
    for (let r = 0; r < 2; r++) {
        for (let c = 0; c < 3; c++) {
            const p = [];
            for (let dr = 0; dr < 3; dr++) for (let dc = 0; dc < 2; dc++) p.push((r + dr) * 4 + c + dc);
            patterns.push(p);
        }
    }
    return patterns;
}

function defaultPatterns4() {
    const patterns = [];
    for (let r = 0; r < 4; r++) patterns.push([r * 4, r * 4 + 1, r * 4 + 2, r * 4 + 3]);
    for (let c = 0; c < 4; c++) patterns.push([c, c + 4, c + 8, c + 12]);
    return patterns;
}

async function gunzipIfNeeded(url, buffer) {
    if (!url.endsWith('.gz')) return new Uint8Array(buffer);
    if (typeof DecompressionStream === 'undefined') {
        throw new Error('this browser lacks DecompressionStream; serve the uncompressed .bin');
    }
    const stream = new Response(buffer).body.pipeThrough(new DecompressionStream('gzip'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
}

export class NTupleAI {
    constructor(model) {
        this.model = model;
        this.lut = model.lut;
    }

    /** Loads and parses `model.bin` / `model.bin.gz`. */
    static async load(url) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`failed to fetch ${url}: ${response.status}`);
        const bytes = await gunzipIfNeeded(url, await response.arrayBuffer());
        return NTupleAI.fromBytes(bytes);
    }

    /** Same as :meth:`load` but from an already decompressed byte array. */
    static fromBytes(bytes) {
        const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
        const view = new DataView(buffer);

        const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 8));
        if (magic !== 'N2048V1\0') throw new Error(`bad model magic: ${JSON.stringify(magic)}`);

        const base = view.getUint32(8, true);
        const n6 = view.getUint32(12, true);
        const n4 = view.getUint32(16, true);
        const len6 = view.getUint32(20, true);
        const len4 = view.getUint32(24, true);
        const dtype = view.getUint32(28, true);
        if (dtype !== 0 && dtype !== 2) throw new Error(`unsupported dtype code ${dtype}`);
        if (len6 !== base ** 6 || len4 !== base ** 4) throw new Error('inconsistent model header');
        const Typed = dtype === 0 ? Int16Array : Int8Array;
        const width = dtype === 0 ? 2 : 1;

        let offset = 32;
        const scale6 = new Float32Array(buffer, offset, n6);
        offset += n6 * 4;
        const scale4 = new Float32Array(buffer, offset, n4);
        offset += n4 * 4;
        const W6 = new Typed(buffer, offset, n6 * len6);
        offset += n6 * len6 * width;
        const W4 = new Typed(buffer, offset, n4 * len4);

        const patterns6 = defaultPatterns6();
        const patterns4 = defaultPatterns4();
        const pow6 = new Float64Array(6);
        const pow4 = new Float64Array(4);
        for (let t = 0; t < 6; t++) pow6[t] = base ** t;
        for (let t = 0; t < 4; t++) pow4[t] = base ** t;

        return new NTupleAI({
            base,
            n6,
            n4,
            len6,
            len4,
            scale6,
            scale4,
            W6,
            W4,
            patterns6,
            patterns4,
            pow6,
            pow4,
            maxExp: base - 1,
            lut: buildMoveLUT(),
        });
    }

    /** Value of one afterstate. */
    evaluate(cells) {
        const m = this.model;
        const cap = m.maxExp;
        let total = 0;
        for (let i = 0; i < m.n6; i++) {
            const pattern = m.patterns6[i];
            let code = 0;
            for (let t = 0; t < 6; t++) {
                const v = cells[pattern[t]];
                code += (v > cap ? cap : v) * m.pow6[t];
            }
            total += m.W6[i * m.len6 + code] * m.scale6[i];
        }
        for (let i = 0; i < m.n4; i++) {
            const pattern = m.patterns4[i];
            let code = 0;
            for (let t = 0; t < 4; t++) {
                const v = cells[pattern[t]];
                code += (v > cap ? cap : v) * m.pow4[t];
            }
            total += m.W4[i * m.len4 + code] * m.scale4[i];
        }
        return total;
    }

    /** Greedy action; returns -1 when the board is terminal. */
    bestMove(cells) {
        let bestDir = -1;
        let bestValue = -Infinity;
        const lut = this.model.lut;
        for (const dir of DIRECTIONS) {
            const moved = applyMove(cells, dir, lut);
            if (!moved.changed) continue;
            const value = this.evaluate(moved.cells);
            if (value > bestValue) {
                bestValue = value;
                bestDir = dir;
            }
        }
        return bestDir;
    }

    /**
     * Expectimax lookahead over the value table.
     *
     * depth=1 is the plain greedy rule.  depth=2 expands the spawn layer
     * exactly (0.9/0.1 over every empty cell) and bootstraps on `V`:
     *
     *     Q(s, a) = r_a + gamma * E_spawn[ max_b V(afterstate(spawn(s_a), b)) ]
     *
     * ~80x the compute of depth=1 (about 2 ms/step here) for a large gain in
     * both mean score and worst-case survival.
     */
    bestMoveDeep(cells, depth = 2, gamma = 0.999) {
        const lut = this.model.lut;
        const moves = [];
        for (const dir of DIRECTIONS) {
            const moved = applyMove(cells, dir, lut);
            if (moved.changed) moves.push({ dir, cells: moved.cells, gain: moved.gain });
        }
        if (!moves.length) return -1;

        if (depth <= 1) {
            let bestDir = -1;
            let bestValue = -Infinity;
            for (const move of moves) {
                const value = this.evaluate(move.cells);
                if (value > bestValue) {
                    bestValue = value;
                    bestDir = move.dir;
                }
            }
            return bestDir;
        }

        const q = new Float64Array(moves.length);
        for (let i = 0; i < moves.length; i++) q[i] = moves[i].gain;

        for (let i = 0; i < moves.length; i++) {
            const after = moves[i].cells;
            const empties = [];
            for (let p = 0; p < CELLS; p++) if (after[p] === 0) empties.push(p);
            if (!empties.length) continue;
            const share = 1 / empties.length;
            let expectation = 0;
            for (const position of empties) {
                for (let t = 0; t < 2; t++) {
                    const tile = t === 0 ? 1 : 2;
                    const probability = t === 0 ? 0.9 : 0.1;
                    const spawned = Uint8Array.from(after);
                    spawned[position] = tile;
                    let bestNext = 0;
                    for (const next of DIRECTIONS) {
                        const moved = applyMove(spawned, next, lut);
                        if (!moved.changed) continue;
                        const value = this.evaluate(moved.cells);
                        if (value > bestNext) bestNext = value;
                    }
                    expectation += probability * share * bestNext;
                }
            }
            q[i] += gamma * expectation;
        }

        let bestDir = -1;
        let bestQ = -Infinity;
        for (let i = 0; i < moves.length; i++) {
            if (q[i] > bestQ) {
                bestQ = q[i];
                bestDir = moves[i].dir;
            }
        }
        return bestDir;
    }

    /** Values of the four afterstates, `null` for illegal moves. */
    actionValues(cells) {
        const lut = this.model.lut;
        return DIRECTIONS.map((dir) => {
            const moved = applyMove(cells, dir, lut);
            return moved.changed ? this.evaluate(moved.cells) : null;
        });
    }
}
