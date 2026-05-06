# Privacy Policy — SLC Scanner

_Second-Life Couture Pattern Codec Scanner_  
_Last updated: 2026-05-02_

---

## What data the app processes

The SLC Scanner app processes the following data on your device:

| Data | Where it goes | Why |
|---|---|---|
| Camera frames | **On-device only** | Tile detection and depth estimation |
| Depth / height field | **On-device only** | Channel extraction from the tile |
| Decoded item ID (32-bit integer) | On-device; optionally sent to the cloud registry | Provenance lookup |
| Scan outcome counters | On-device only (by default) | Performance telemetry |

**No image data, no depth fields, and no raw sensor data ever leave your device** unless you explicitly choose to "Verify provenance" against the cloud registry — and even then, only the decoded 32-bit item ID is transmitted, never the image or depth field. This is a hard architectural constraint, not a configuration option.

---

## Provenance verification (optional, explicit)

When you tap "Verify provenance", the app sends:
```
POST https://registry.secondlifecouture.com/v1/lookup
Body: { "item_id": "0x2DCA3791" }
```

No camera image, no depth data, no location, no device identifier is included in this request. The server responds with the public provenance record for that item (maker, collection, garment, material, care instructions).

You can scan and read channels entirely offline. Provenance lookup requires a network connection and your explicit action.

---

## Telemetry (opt-in)

The app maintains in-app counters only:
- Total scan attempts
- Outcomes by fallback level (full / rs_corrected / category_only / lost)
- Stage latency (p50 and p95 in ms, per stage)
- Device capability flags (has LiDAR, has ARCore depth)

These counters live in `shared_preferences` on your device and are **never transmitted anywhere by default**.

You can opt in to share aggregate statistics with the atelier for ML retraining. If you opt in:
- Counters only — no images, no IDs, no location
- The opt-in is revocable at any time in Settings > Developer > Share diagnostics
- Transmitted once per week, in aggregate, via HTTPS

---

## Camera permission

The app requests camera access to scan pattern tiles. Frames are processed in memory and immediately discarded. No frame is written to your photo library or the app's document container unless you explicitly use the "Save last capture" feature in the developer screen.

---

## Data retention

The app stores nothing persistently about individual scans except the in-app counters described above. There is no scan history, no image cache, no ID log.

---

## Contact

For privacy questions: privacy@secondlifecouture.com
