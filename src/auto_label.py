"""Auto-label the World Context ego/exo pair with a vision model, without
deciding the experiment in advance.

The experiment this feeds asks which VIEW better predicts the label. So a label
generated from one view would settle that question by construction — exo-derived
labels are trivially predictable from exo, and vice versa. Showing the model both
views at once does not fix it either: whichever view the model leans on leaks
into the labels.

So the model labels each window TWICE and never sees both views at once:

    pass A — ego frames only
    pass B — exo frames only

Only windows where the two passes AGREE are kept. A label that both views
independently produce cannot favour either one, which makes the kept set
view-neutral by construction.

The discarded windows are not waste — the DISAGREEMENT RATE is itself a
measurement of the view gap on this footage, and it needs no ground truth at all.

Usage:
    python -m src.auto_label [--model gemini-3.7-flash] [--workers 8] [--limit N]
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import re
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

FRAMES = pathlib.Path("/private/tmp/claude-501/-Users-evan-Projects-exoego/"
                      "a0c6b8fb-f59d-4405-8f14-34aa52a4c40a/scratchpad/lab")
KEYFILE = pathlib.Path.home() / ".concentrate_key"
ENDPOINT = "https://api.concentrate.ai/v1/responses"
OUT = pathlib.Path("results/auto_labels.json")
LABELS_OUT = pathlib.Path("results/exoego_labels.json")

WINDOW = 5.0
N_WINDOWS = 179
OFFSET_S = 11.16

VERBS = ["pick up", "put down", "attach / fit", "detach / remove", "screw / tighten",
         "insert", "position / align", "connect wire", "inspect / check", "idle / walk"]
NOUNS = ["frame / chassis", "seat", "battery", "wheel / tyre", "handlebar", "mirror",
         "body panel", "footboard", "screw / bolt", "screwdriver", "wrench",
         "wire / harness", "connector", "carton / box", "sticker / label",
         "cable tie", "other", "none"]

PROMPT = """You are labelling 5 seconds of footage from a scooter assembly line.
The three images are consecutive seconds from ONE camera.

Choose exactly one VERB (what the worker is doing) and one NOUN (what they are
handling) from these closed lists. Use the exact strings.

VERBS: {verbs}
NOUNS: {nouns}

If the worker is not manipulating anything, use "idle / walk" and "none".
Answer with ONLY a JSON object, no prose:
{{"verb": "<one verb>", "noun": "<one noun>"}}"""

_print_lock = threading.Lock()


def api_key() -> str:
    if not KEYFILE.exists():
        sys.exit(f"missing {KEYFILE} — write the Concentrate key there (chmod 600)")
    return KEYFILE.read_text().strip()


def b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def call(model: str, key: str, images: list[pathlib.Path], retries: int = 3):
    content = [{"type": "input_text",
                "text": PROMPT.format(verbs=" | ".join(VERBS), nouns=" | ".join(NOUNS))}]
    for p in images:
        content.append({"type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{b64(p)}"})
    body = {"model": model, "input": [{"role": "user", "content": content}]}
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 # Cloudflare rejects urllib's default UA with a 403/1010
                 "User-Agent": "curl/8.7.1", "Accept": "*/*"})
    last = None
    for _ in range(retries):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            for item in r.get("output", []):
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            return parse(c["text"])
            last = "no output_text"
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
        except Exception as e:  # timeouts, transient DNS, malformed json
            last = type(e).__name__
    return {"verb": None, "noun": None, "error": last}


def parse(text: str) -> dict:
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return {"verb": None, "noun": None, "error": "unparseable"}
    try:
        d = json.loads(m.group(0))
    except Exception:
        return {"verb": None, "noun": None, "error": "bad json"}
    v, n = d.get("verb"), d.get("noun")
    return {"verb": v if v in VERBS else None,
            "noun": n if n in NOUNS else None}


def frames_for(view: str, w: int) -> list[pathlib.Path]:
    """Three frames spanning window w — enough to read motion, cheap to send."""
    base = int(w * WINDOW)
    out = []
    for off in (1, 2, 4):
        p = FRAMES / view / f"{base + off + 1:05d}.jpg"
        if p.exists():
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.7-flash")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    key = api_key()
    n = args.limit or N_WINDOWS
    jobs = [(v, w) for w in range(n) for v in ("ego", "exo")]
    print(f"model={args.model}  windows={n}  calls={len(jobs)}  workers={args.workers}")

    results: dict[tuple[str, int], dict] = {}
    done = [0]

    def work(job):
        view, w = job
        imgs = frames_for(view, w)
        r = {"verb": None, "noun": None, "error": "no frames"} if not imgs \
            else call(args.model, key, imgs)
        results[job] = r
        with _print_lock:
            done[0] += 1
            if done[0] % 40 == 0:
                print(f"  {done[0]}/{len(jobs)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, jobs))

    rows, agree, both_ok, errs = [], 0, 0, 0
    for w in range(n):
        a, b = results.get(("ego", w), {}), results.get(("exo", w), {})
        if a.get("error") or b.get("error"):
            errs += 1
        ok = a.get("verb") and a.get("noun") and b.get("verb") and b.get("noun")
        if ok:
            both_ok += 1
        v_ok = ok and a["verb"] == b["verb"]
        n_ok = ok and a["noun"] == b["noun"]
        if v_ok and n_ok:
            agree += 1
        rows.append({"window": w,
                     "t_start": round(w * WINDOW, 2), "t_end": round((w + 1) * WINDOW, 2),
                     "ego": {"verb": a.get("verb"), "noun": a.get("noun")},
                     "exo": {"verb": b.get("verb"), "noun": b.get("noun")},
                     "verb_agree": bool(v_ok), "noun_agree": bool(n_ok)})

    v_agree = sum(r["verb_agree"] for r in rows)
    n_agree = sum(r["noun_agree"] for r in rows)
    print("\n" + "=" * 66)
    print("INDEPENDENT LABELLING FROM EACH VIEW")
    print("=" * 66)
    print(f"  windows              : {n}")
    print(f"  both views answered  : {both_ok}   (errors on {errs})")
    if both_ok:
        print(f"  VERB agreement       : {v_agree}/{both_ok} = {v_agree/both_ok:6.1%}")
        print(f"  NOUN agreement       : {n_agree}/{both_ok} = {n_agree/both_ok:6.1%}")
        print(f"  BOTH agree           : {agree}/{both_ok} = {agree/both_ok:6.1%}")
    print("\n  Disagreement is a view-gap measurement in its own right: it is how")
    print("  often the two cameras support different readings of the same moment,")
    print("  and it needs no ground truth.")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(
        {"model": args.model, "window_s": WINDOW, "offset_s": OFFSET_S,
         "verbs": VERBS, "nouns": NOUNS, "n_windows": n,
         "verb_agreement": v_agree / both_ok if both_ok else None,
         "noun_agreement": n_agree / both_ok if both_ok else None,
         "rows": rows}, indent=2))

    kept = [{"window": r["window"], "t_start": r["t_start"], "t_end": r["t_end"],
             "verb": r["ego"]["verb"], "noun": r["ego"]["noun"], "unclear": False}
            for r in rows if r["verb_agree"] and r["noun_agree"]]
    LABELS_OUT.write_text(json.dumps(
        {"source": "world_context_ego_exo_pair", "labeller": f"auto:{args.model}",
         "selection": "kept only windows where ego-only and exo-only passes agreed",
         "window_s": WINDOW, "offset_s": OFFSET_S,
         "verbs": VERBS, "nouns": NOUNS, "n_windows": n, "labels": kept}, indent=2))

    print(f"\n  kept {len(kept)} view-neutral windows -> {LABELS_OUT}")
    print(f"  full per-view record -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
