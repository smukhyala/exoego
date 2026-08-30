"""Synchronize the World Context ego/exo pair by audio onset correlation.

The two GoPros started at different times (exo 910 s, ego 1130 s), so every
ego/exo comparison depends on recovering the offset first.

A first attempt correlated log-RMS envelopes and failed outright (peak 0.195,
below the 99.9th percentile of the correlogram). The reason is the room: a
workshop is full of continuous broadband noise — compressors, fans — which
dominates an energy envelope and is *not* informative about timing. What the two
mics genuinely share is transients: tool strikes, dropped parts, impacts.

So we correlate a spectral-flux ONSET envelope instead, which is what onset
detection uses precisely because it discards steady-state energy.

Usage:  python -m src.sync_audio
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

SCRATCH = Path("/private/tmp/claude-501/-Users-evan-Projects-exoego/"
               "a0c6b8fb-f59d-4405-8f14-34aa52a4c40a/scratchpad")
EXO = SCRATCH / "GX010104-exo-C7459.wav"
EGO = SCRATCH / "GX014991-ego-C2920.wav"
SR = 8000
HOP = 80           # -> 100 Hz feature rate
WIN = 512
FPS = SR / HOP


def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return a


def onset_envelope(x: np.ndarray) -> np.ndarray:
    """Half-wave-rectified spectral flux — energy ONSETS, not energy."""
    n_frames = 1 + (len(x) - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(WIN)[None, :]
    S = np.abs(np.fft.rfft(frames, axis=1))
    S = np.log1p(S * 100.0)
    flux = np.diff(S, axis=0)
    flux = np.maximum(flux, 0.0).sum(axis=1)
    # remove slow drift so correlation is driven by transients
    k = int(FPS * 2)
    kern = np.ones(k) / k
    base = np.convolve(flux, kern, mode="same")
    o = flux - base
    return (o - o.mean()) / (o.std() + 1e-9)


def xcorr(a: np.ndarray, b: np.ndarray):
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    c = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    c = np.concatenate([c[-(len(b) - 1):], c[:len(a)]])
    lags = np.arange(-(len(b) - 1), len(a))
    # normalize by the number of overlapping samples at each lag
    ov = np.minimum(np.minimum(len(a), len(b)),
                    np.minimum(len(a) - np.maximum(lags, 0),
                               len(b) + np.minimum(lags, 0)))
    ov = np.maximum(ov, 1)
    c = c / ov
    valid = ov > FPS * 60  # need >=60 s of overlap to be meaningful
    c[~valid] = 0.0
    return lags, c


def main() -> int:
    exo, ego = read_wav(EXO), read_wav(EGO)
    print(f"exo {len(exo)/SR:7.1f} s   ego {len(ego)/SR:7.1f} s\n")

    a, b = onset_envelope(exo), onset_envelope(ego)
    print(f"onset envelopes: exo {len(a)} frames, ego {len(b)} frames @ {FPS:.0f} Hz")

    lags, c = xcorr(a, b)
    i = int(np.argmax(c))
    lag, peak = int(lags[i]), float(c[i])
    off = lag / FPS

    nz = c[c != 0]
    p999 = float(np.percentile(np.abs(nz), 99.9))
    ratio = peak / (p999 + 1e-12)

    print("\nONSET CROSS-CORRELATION")
    print(f"  best lag       : {off:+.3f} s")
    print(f"  peak           : {peak:.4f}")
    print(f"  99.9th pct |c| : {p999:.4f}")
    print(f"  peak / p99.9   : {ratio:.2f}x")

    order = np.argsort(c)[::-1][:5]
    print("\n  top 5 candidate offsets:")
    for j in order:
        print(f"    {lags[j]/FPS:+9.2f} s   c={c[j]:.4f}")

    if off >= 0:
        print(f"\n  ego t=0 corresponds to exo t={off:.2f} s")
    else:
        print(f"\n  exo t=0 corresponds to ego t={-off:.2f} s")

    overlap = min(len(exo) / SR - max(0.0, off), len(ego) / SR - max(0.0, -off))
    print(f"  usable overlap : {overlap:.1f} s ({overlap/60:.1f} min, "
          f"{overlap*29.97:.0f} frame pairs)")

    good = peak > 0 and ratio > 1.5
    print(f"\n  verdict: {'TRUSTWORTHY' if good else 'STILL WEAK — verify visually'}")

    out = Path("results/sync.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "method": "spectral_flux_onset_xcorr",
        "offset_s_exo_of_ego_t0": off, "peak": peak, "peak_over_p999": ratio,
        "overlap_s": overlap, "trustworthy": bool(good),
        "top5_s": [float(lags[j] / FPS) for j in order],
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
