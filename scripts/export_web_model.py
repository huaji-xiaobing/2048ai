#!/usr/bin/env python3
"""Export an n-tuple value table to the compact binary used by the web demo.

Layout (little endian, then gzipped)::

    magic   8 bytes  b"N2048V1\\0"
    u32     base
    u32     n6, n4            number of 6-tuples / 4-tuples
    u32     len6, len4        base**6, base**4
    u32     dtype            0 = int16, 2 = int8
    f32 * n6                  per-table scale for the 6-tuples
    f32 * n4                  per-table scale for the 4-tuples
    i16 * n6 * len6           quantised W6 (row major)
    i16 * n4 * len4           quantised W4

Per-table symmetric quantisation keeps the relative ordering inside one table
exact to ~max/32767, which is two orders of magnitude finer than the spread of
afterstate values, so the argmax is preserved.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from ntuple2048.network import NTupleNetwork

MAGIC = b"N2048V1\x00"
INT16_MAX = 32767.0
INT8_MAX = 127.0


def quantise(W: np.ndarray, bits: int = 16) -> tuple[np.ndarray, np.ndarray]:
    peak_value = INT8_MAX if bits == 8 else INT16_MAX
    peak = np.abs(W).max(axis=1).astype(np.float32)
    peak[peak <= 0] = 1.0
    scale = peak / peak_value
    q = np.rint(W / scale[:, None]).astype(np.int8 if bits == 8 else np.int16)
    return q, scale


def export(network: NTupleNetwork, out_path: Path, compress: bool = True, bits: int = 16) -> dict:
    q6, s6 = quantise(network.W6, bits)
    q4, s4 = quantise(network.W4, bits)

    header = bytearray()
    header += MAGIC
    header += struct.pack(
        "<6I",
        network.base,
        network.n6,
        network.n4,
        network.l6,
        network.l4,
        0 if bits == 16 else 2,
    )
    header += s6.astype("<f4").tobytes()
    header += s4.astype("<f4").tobytes()

    item = "<i2" if bits == 16 else "i1"
    raw = bytes(header) + q6.astype(item).tobytes() + q4.astype(item).tobytes()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        import gzip

        with gzip.open(out_path, "wb", compresslevel=9) as handle:
            handle.write(raw)
    else:
        out_path.write_bytes(raw)

    return {
        "path": str(out_path),
        "raw_bytes": len(raw),
        "file_bytes": out_path.stat().st_size,
        "base": network.base,
        "patterns6": [list(p) for p in network.patterns6],
        "patterns4": [list(p) for p in network.patterns4],
        "scales6": [float(x) for x in s6],
        "scales4": [float(x) for x in s4],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/web_model.npz")
    parser.add_argument("--out", default="docs/model.bin.gz")
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument("--bits", type=int, choices=(8, 16), default=16)
    args = parser.parse_args()

    network = NTupleNetwork.load(args.model)
    info = export(network, Path(args.out), compress=not args.no_compress, bits=args.bits)
    print(
        f"{info['path']}: {info['file_bytes'] / 1e6:.2f} MB "
        f"(raw {info['raw_bytes'] / 1e6:.1f} MB), base={info['base']}"
    )
    print("scales6:", [round(x, 3) for x in info["scales6"]][:4], "...")
    print(json.dumps({"patterns6": len(info["patterns6"]), "patterns4": len(info["patterns4"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
