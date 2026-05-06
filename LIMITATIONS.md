# Known Limitations — SLC Scanner

These are honest, documented constraints of the Second-Life Couture Pattern Codec system. They exist by design or are inherent to the physical medium. Do not claim to fix them without experimental evidence.

---

## 1. High-count IDs decode poorly

IDs whose `count` channel encodes 14 or 15 (i.e., 17 or 18 protrusions) have degraded decode accuracy. At that protrusion density the bumps crowd the rim and their peaks overlap, making reliable peak detection difficult with the current 360-sample ring scan.

The Python and JavaScript reference implementations have the same issue — this is not a Flutter-specific bug.

**Design constraint:** COUNT_TO_PROTRUSIONS maps 0..15 to 3..18 protrusions. The codec has no reserved range that avoids high counts; the RS parity row is the recovery path when count misfires.

**Practical impact:** IDs with `count ∈ {14, 15}` (i.e., `(item_id >> 16) & 0x0F ∈ {14, 15}`) have roughly 15–20 percentage points lower decode accuracy than the median. If you operate a registry, assign these ranges to low-priority catalog items or leave them unused.

---

## 2. Heavily worn tiles typically return category_only

Above ~50% simulated wear (fabric age ≈ 3–5 years of daily wear), the protrusion heights fall below the reliable measurement threshold and the angles channel degrades. The codec degrades gracefully through its fallback ladder:

```
full → rs_corrected → category_only → lost
```

`category_only` means you can recover the macro family and protrusion count (coarse garment category) but not the specific item ID. This is by design: the 32-bit RS parity row is wide enough to bridge moderate damage, but not extreme physical erosion.

**Communicated in UX:** The app displays "This garment has been worn extensively — that's why decode is harder" with the estimated wear factor from the wear estimator model.

---

## 3. Non-LiDAR Android depth estimation is significantly less accurate

Devices without ARCore Depth API support (or without a hardware time-of-flight sensor) fall back to MiDaS-small monocular depth estimation. Monocular depth:
- Gives relative depth, not absolute mm — requires calibration from the known 30mm tile diameter
- Has higher noise than structured-light or time-of-flight sensors
- Is affected by specular reflections on the TPU surface

**Expected accuracy delta:** 10–15 percentage points lower full-decode rate on these devices compared to iOS LiDAR.

**Communicated in UX:** On devices where MiDaS is the active depth source, the app shows:
> "Decode quality on this device is reduced. For best results, use a phone with a dedicated depth sensor."

The device's depth source is determined at startup and displayed in the developer screen (accessible via triple-tap on the version number).

---

## 4. Macro classifier accuracy at high wear

The harmonic correlator (current fallback when CNN confidence < 0.70) achieves only 47% accuracy at wear=0.0 and degrades to ~33% at wear=0.6, per the batch test results in `README.md`. The CNN macro classifier (Enhancement 1) is specifically designed to address this, but the trained model is not yet included in v1.

Until the CNN model is available, macro detection is the dominant failure mode.

---

## 5. Micro channel is not decoded in v1

The `micro` channel (seeded deterministic noise, 8 bits) is not recoverable from a depth scan without brute-forcing the seed against the observed high-frequency pattern — an expensive operation. In all current implementations the micro byte is declared absent and recovered by RS parity.

This means the RS system is always carrying at least 1 erasure (micro=null), reducing its error-correction headroom from 4 erasures to 3. This is acceptable given that micro is the least resilient channel by design.

---

## 6. Counterfeit detection is not implemented in v1

The `genuine_confidence` field in `DecodeResult` is hardcoded to `1.0` in v1. The binary counterfeit classifier (Enhancement 3) requires a trained model on genuine vs. replicated tile micro-textures. That training data doesn't exist yet.

Do not surface `genuine_confidence` to users as meaningful data in v1. The field is reserved for future use once the classifier ships.

---

## 7. Tile must be within 5–20 cm of the camera

The YOLOv8 detection model is trained on tiles at 0.5×–2.0× of the nominal 30mm diameter in the camera frame. Very close or very far positions are out-of-distribution and may not be detected.

**Practical guidance (surfaced in UI):** Move the camera until the tile fills roughly a quarter of the reticle — the inner pulsing circle guides this.
