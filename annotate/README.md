# ExoEgo Annotator

Label 5-second windows of the synchronized World Context ego/exo pair with a
verb (what is being done) and a noun (what is being handled).

## Run it

    cd annotate && python3 serve.py

Then open http://localhost:8000

**Do not use `python3 -m http.server`.** Its handler ignores the HTTP `Range`
header and answers every request with 200 and the whole file, so a `<video>`
element cannot seek — to reach t=300s the browser must first download all 65 MB.
The symptom is that the first window plays and every window after it hangs. It is
also HTTP/1.0 and single-threaded, so the two videos serialize behind one
connection. `serve.py` fixes both: 206 Partial Content with a correct
Content-Range, on a threading HTTP/1.1 server.

## Why this exists

Assembly101 could not settle whether the ego/exo difference is about VIEWPOINT
or about CAMERA QUALITY: its ego cameras are 636x480 monochrome while its exo
cameras are 1920x1080 colour, so exo won on object recognition for reasons that
have nothing to do with where the camera sits.

The World Context pair removes that confound. Both cameras are GoPros at
1920x1080 colour, same session, same lighting. Whatever difference we measure
here is viewpoint.

## Labelling rule

**Use BOTH views to decide the label.** Each view is evaluated separately
afterwards, so the ground truth must not favour either one. If an object is only
legible in one view, still label it — that asymmetry is exactly what the
experiment is meant to detect. Use `unclear` only when both views leave you
genuinely unsure.

## Keys

    1-9, 0     verb
    a-z        noun
    left/right window
    space      replay current window
    u          unclear

Progress saves in the browser automatically. **Export JSON** writes
`exoego_labels.json`.

## Coverage

179 windows over 898.8 s. Two people splitting it (one takes windows 0-89, the
other 90-178) should finish in well under an hour. Merge the two exported JSONs
before running the eval.

## Regenerating the media

The clips are gitignored. To rebuild them from the source MP4s, with the
+11.16 s offset baked in so both start aligned at t=0:

    ffmpeg -ss 0     -t 898.8 -i data/wc_exo/GX014991-ego-C2920.MP4 \
      -vf scale=640:-2 -c:v libx264 -preset veryfast -crf 30 -an \
      -movflags +faststart annotate/media/ego.mp4

    ffmpeg -ss 11.16 -t 898.8 -i data/wc_exo/GX010104-exo-C7459.MP4 \
      -vf scale=640:-2 -c:v libx264 -preset veryfast -crf 30 -an \
      -movflags +faststart annotate/media/exo.mp4
