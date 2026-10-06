# Modality probe fixtures

Binary fixtures used by the A2A agent validator's modality probes. Each
file is loaded at import time via `_b64(name)` in `verify_a2a_modalities.py`
and exposed as a `*_B64` constant. The same constants get assembled into
`*_PROBE_PARTS` lists for the inline-bytes path, and the raw bytes are also
uploaded to the object store for the URI path.

| File | Probe modality | What it contains | Why |
|---|---|---|---|
| `red.png`      | `image/png`           | 8×8 solid red square                | smallest valid red png (75 B)   |
| `red.jpg`      | `image/jpeg`          | 8×8 solid red square                | tests separate JPEG decode path |
| `red.gif`      | `image/gif`           | 8×8 solid red square                | tests separate GIF decode path  |
| `clip.wav`     | `audio/wav`           | spoken `"Say the word: banana"`     | tests PCM decode + transcription|
| `clip.mp3`     | `audio/mpeg`          | spoken `"Say the word: telephone"`  | tests MP3 decode + transcription|
| `clip.m4a`     | `audio/mp4`           | spoken `"Say the word: bicycle"`    | tests AAC decode + transcription|
| `clip.ogg`     | `audio/ogg`           | spoken `"Say the word: rainbow"`    | tests Opus decode + transcription|
| `document.pdf` | `application/pdf`     | one-page instructional doc, target `ELEPHANT` | tests PDF text extraction |
| `clip.mp4`     | `video/mp4`           | 3 s clip showing `GIRAFFE` on white | tests video-frame OCR; URI-delivered |

## Design choices that surface as gotchas

### Distinct target words per audio format
Each audio file says a *different* word — the validator uses
`AUDIO_WAV_EXPECTED = "banana"`, `AUDIO_MP3_EXPECTED = "telephone"`, etc.
If every probe expected the same word, a model could pass by outputting
that word from any cue (filename, bias, the prompt). Distinct expected
values per format means the model has to actually decode each clip.

### Neutral filenames
All fixtures are named `clip.<ext>` / `red.<ext>` / `document.pdf` rather
than `banana.<ext>` / `red-square.png` etc. A leaky filename lets the model
answer from the path the wrapper materializes (`@/app/inputs/<task>/X`)
without ever decoding the bytes. We caught this when an early probe with
`banana.mp3` was "passing" via `ls banana.mp3` + filename inference.

### Audio encoding details
- **16 kHz mono is the format Gemini accepts reliably.** An earlier 8 kHz
  WAV failed even though `audio/wav` is in the gemini-cli MIME table.
- **`Alex` voice transcribes more reliably than the default `say` voice.**
  With the default voice the model occasionally heard "banana" as "apple".
- **3× volume boost + 200 ms silence padding raise transcription
  accuracy from ~70% to ~100%.** Without them the model sometimes guessed
  unrelated words ("visual", "red") when it couldn't lock onto the
  utterance.

### PDF embeds the target word inside an instructional paragraph
`document.pdf` doesn't just show `ELEPHANT` alone — it shows a short
paragraph naming `ELEPHANT` as the target word. Bare word alone tripped
letter-truncation edge cases (the model occasionally reading "ELEPHANT"
as "ANT" or "PHANT", probably from letter-spacing artifacts in the PDF
text extractor).

### Video shows a static frame, not motion
`clip.mp4` is a static frame on a white background (480×360 @ 15 fps, h.264
CRF 28, no audio track). It tests video-frame OCR, not temporal
understanding. The static design plus on-screen instructional context plus
the larger 480×360 frame size were each needed: smaller / barer clips
truncated to "FFE" probabilistically.

### Why video is URI-only
The video fixture is delivered via `FileWithUri`, not inline
`FileWithBytes` like the others: the validator uploads it to the object
store and the agent receives an HTTPS URL for it. At 8.9 KB it would fit
inline, but the validator deliberately exercises the wrapper's
URI-materialization code path with this probe — it's the only fixture that
does so for video, and one of two (alongside the URI-delivered PNG probe)
that exercise URI delivery at all.

## Regenerating fixtures

These are *not* runtime dependencies. They're only needed if you want to
re-create or modify a fixture. Install with `pip install Pillow fpdf2`.
You'll also need `ffmpeg` and the `flac` CLI on PATH (`brew install
ffmpeg flac` on macOS).

### Audio (`clip.wav`, `clip.mp3`, `clip.m4a`, `clip.ogg`)

```bash
FF_FILTER="silenceremove=start_periods=1:start_duration=0:start_threshold=-50dB:detection=peak,areverse,silenceremove=start_periods=1:start_duration=0:start_threshold=-50dB:detection=peak,areverse,volume=3.0,adelay=200|200,apad=pad_dur=0.2"

# Per-format: substitute the target word, re-encode in native format.
say -v Alex -r 150 "Say the word: Banana" -o /tmp/x.aiff
ffmpeg -y -i /tmp/x.aiff -ac 1 -ar 16000 -map_metadata -1 -fflags +bitexact \
  -af "$FF_FILTER" -acodec pcm_s16le fixtures/clip.wav

# MP3 / M4A / OGG: encode from the freshly-trimmed WAV
ffmpeg -y -i fixtures/clip.wav -ac 1 -ar 16000 -b:a 64k -map_metadata -1 \
  -fflags +bitexact -write_xing 1 fixtures/clip.mp3
ffmpeg -y -i fixtures/clip.wav -c:a aac -b:a 64k -ac 1 -ar 16000 \
  -map_metadata -1 -fflags +bitexact fixtures/clip.m4a
ffmpeg -y -i fixtures/clip.wav -c:a libopus -b:a 32k -ac 1 -ar 16000 \
  -map_metadata -1 -fflags +bitexact fixtures/clip.ogg
```

If updating the target word for a format, also update the corresponding
`AUDIO_<FMT>_EXPECTED` constant in `verify_a2a_modalities.py`.

### PDF (`document.pdf`)

```python
# requires: pip install fpdf2
from fpdf import FPDF
pdf = FPDF(orientation='P', unit='pt', format='A4')
pdf.set_compression(True); pdf.set_creator(''); pdf.set_producer('')
pdf.add_page()
pdf.set_font('Helvetica', '', 36)
pdf.set_xy(50, 100)
pdf.multi_cell(500, 50,
    'A2A Modality Probe Document\n\n'
    'This page contains exactly one target word that the reader should report.\n\n'
    'The target word is: ELEPHANT\n\n'
    'Reply with only the word above.',
    align='L')
pdf.output('fixtures/document.pdf')
```

### Video (`clip.mp4`)

```python
# requires: pip install Pillow; brew install ffmpeg
from PIL import Image, ImageDraw, ImageFont
img = Image.new('RGB', (480, 360), 'white')
draw = ImageDraw.Draw(img)
big   = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 80)
small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
draw.text((30, 30), "A2A Video Probe", font=small, fill='black')
draw.text((30, 60), "Target word:",    font=small, fill='black')
bbox = draw.textbbox((0, 0), "GIRAFFE", font=big); w = bbox[2] - bbox[0]
draw.text(((480 - w) / 2, 150), "GIRAFFE", font=big, fill='black')
draw.text((30, 280), "Reply with only the target word.", font=small, fill='black')
img.save('/tmp/frame.png')
```
```bash
ffmpeg -y -loop 1 -i /tmp/frame.png -t 3 -r 15 \
  -c:v libx264 -preset slow -crf 28 -pix_fmt yuv420p -movflags +faststart \
  fixtures/clip.mp4
```

### Images (`red.png`, `red.jpg`, `red.gif`)

These are static and committed once. To regenerate, encode an 8×8 solid
red square via Pillow or any image tool — the existing files are about
as small as their formats allow.

## Listening / previewing fixtures

```bash
# Audio (afplay handles wav/mp3/m4a; ffplay for ogg)
afplay fixtures/clip.wav
ffplay -nodisp -autoexit fixtures/clip.ogg

# PDF / video
qlmanage -p fixtures/document.pdf
open fixtures/clip.mp4

# Or open the whole directory in Finder and hit space for QuickLook
open fixtures/
```
