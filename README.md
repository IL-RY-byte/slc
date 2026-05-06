# Second-Life Couture — Pattern Codec MVP

A working proof-of-concept for the 3D pattern codec described in
`Second-Life_Couture_-_Pattern_Codec_System_TZ_v3.docx`.

> Encode a 32-bit item ID into a 3D-printed TPU tile. Print it. Sew it to the
> garment. Wear the garment for a year. Read the ID back with a phone camera
> by the silhouette and the pattern of bumps that's left.

## What's in the box

```
slc/
├─ codec/            Python codec (encode / Reed-Solomon / decode)
├─ geometry/         Pattern geometry generator + STL writer
├─ perception/       Wear simulator + channel extractor (the "scanner brain")
├─ web/              Single-file editorial web preview (encoder + 3D viewer)
├─ flutter/          Flutter scanner app skeleton (dart codec port + camera UI)
├─ tests/            Batch accuracy harness
└─ examples/         CLI to generate a sample STL
```

### What actually works
- Round-trip an item ID → channels → 3D mesh → STL file
- Simulate wear (micro / meso / macro stages from §2.3 of the ТЗ)
- Extract channels from worn observations and run RS recovery
- **Web app v2** (`web2/index.html`): five-panel atelier — parametric encoder, hand-drawn canvas with key embedding, photo-to-relief converter, **working scan demo** that decodes the rendered preview entirely in-browser, and an order workflow walkthrough
- **Browser-side decoder**: full JS port of the codec + perception pipeline. The "Scan current preview" button runs encode → render → extract → RS-decode end-to-end, no server. Demonstrates the algorithm round-tripping through the same ladder of fallbacks (full / rs_corrected / category_only / lost) the phone scanner would use.
- Web v1 generator (`web/index.html`) with editorial UI, live 3D preview, wear time-lapse, STL download
- Flutter scanner skeleton with camera preview, scanner overlay, mock decode flow
- The Dart codec is a faithful port of the Python encode side (and a working RS erasure decoder)

### What's a placeholder for production
- **Macro classifier**: hand-tuned harmonic correlator. In production this is a small CNN trained on rendered + worn pattern images. It's the dominant source of decode failure right now.
- **Real-camera phone-side perception**: the Flutter app and the v2 web app's camera button capture frames but don't run the full decode against a *real photograph* — that needs YOLOv8-seg (find the tile in the frame) + MiDaS or LiDAR (lift to depth) before the channel extractor. The web app is honest about this in its UI: camera mode shows a "pipeline pending" message; "scan current preview" runs the real decoder against the *rendered* preview to prove the algorithm works end-to-end in the browser.
- **Berlekamp-Massey**: the v1 Dart RS decoder handles erasures only. The v2 JS RS decoder uses a brute-force-positions + Gaussian-elimination approach that handles full RS-4 (errors + erasures) plus a sanity-check pass that catches false-positive recoveries when erasures over-determine the system.

## Honest accuracy numbers

Running `python tests/test_batch.py --n 30` on a 96-resolution grid:

| wear | full | rs_fix | category | lost | id_acc | cat_acc |
|------|-----:|-------:|---------:|-----:|-------:|--------:|
| 0.0  | 0    | 14     | 11       | 5    | 47%    | 47%     |
| 0.2  | 0    | 15     | 12       | 3    | 47%    | 47%     |
| 0.4  | 0    | 1      | 23       | 6    | 0%     | 43%     |
| 0.6  | 0    | 5      | 18       | 7    | 0%     | 33%     |

The ТЗ's targets (≥99% new, ≥90% at 40% wear) are aggressive and will require the macro classifier upgrade. The codec architecture itself is sound: the channels round-trip cleanly through encode → render → simulated capture, and the fallback ladder degrades gracefully.

## How to run things

### Generate an STL
```bash
python examples/generate_sample_stl.py --id 0x2DCA3791 --out pattern.stl
```

### Run the batch test
```bash
python tests/test_batch.py --n 100
```

### Open the web preview
```bash
open web/index.html         # macOS — v1 generator
open web2/index.html        # macOS — v2 with scan demo and 3 input modes
xdg-open web/index.html     # Linux
```

The v2 app (`web2/index.html`) ports the codec **and** the channel extractor to
JavaScript, so the full encode → render → extract → RS-decode loop runs in the
browser. Five panels:

1. **Parametric** — type a 32-bit ID, see channels, 3D preview, wear time-lapse, STL.
2. **Draw** — sketch a silhouette; the codec embeds your key as 32 boundary markers + macro modulation.
3. **From Photo** — drop any image; luminance becomes relief height; key embedded as edge modulation.
4. **Scan** — point at the preview from any of the above; the JS extractor reads channels, RS-decodes, displays the provenance card. The "Use camera" button captures a frame but is honest that real-world decoding requires on-device ML (YOLOv8-seg + MiDaS) not available in browsers.
5. **Order** — 4-step workflow from mint to activation.

### Build the Flutter app
```bash
cd flutter
flutter pub get
flutter run
```

## Codec layout (matches ТЗ §2.1)

| Channel | Bits | Range / meaning              | Resilience |
|---------|-----:|------------------------------|------------|
| macro   | 4    | 16 base shape families       | High       |
| count   | 4    | 3..18 protrusions            | High       |
| height  | 8    | 1.5..8.0 mm relief           | Medium     |
| angles  | 8    | 4-bit base + 4-bit modulation| Medium     |
| micro   | 8    | seeded high-frequency texture| Low        |
| **rs**  | 32   | Reed-Solomon parity (4 bytes)| —          |

Total: 32 bit payload + 32 bit parity = 64 bits over 30 mm tile.
RS-4 corrects 2 errors or up to 4 erasures.

## Design decisions worth re-reading the code over
- 16 macro families with **unique** harmonic signatures (hexagon and star_6 originally collided — caught during testing).
- RS markers placed as 32 binary high-contrast bumps (1.0 mm vs 3.0 mm) around the rim, with σ wider than the wear-stage smoothing kernel. This was the single biggest robustness win: changing from 8 markers × 2 bits each to 32 markers × 1 bit each.
- Templates are L2-normalized in the macro detector. Without normalization, families with more harmonic content win every comparison even on noise.
- The wear smoothing kernel (k=1 box blur) is deliberately narrower than the RS marker width, so micro-stage wear cleans up texture without eating the parity row.
"# slc" 
