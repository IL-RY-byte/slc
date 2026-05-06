"""
Pattern Codec — core encoding/decoding algorithm for Second-Life Couture.

Encodes a 32-bit item_id into 5 independent geometric channels with
Reed-Solomon error correction so the ID survives partial wear.

Channel layout (per ТЗ §2.1):
    macro    : 4 bits  — base shape (high resilience, read at 70% wear)
    count    : 4 bits  — number of protrusions (high resilience)
    height   : 8 bits  — relief height profile (medium resilience)
    angles   : 8 bits  — angles between elements (medium resilience)
    micro    : 8 bits  — micro-texture (low resilience, wears first)
    -----
    32 bits payload + 16 bits Reed-Solomon parity = 48 bits total

The RS parity is itself distributed across the geometry as redundant markers,
so even when micro is fully gone, we can recover from macro+count+height+angles+RS.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from typing import Optional

from reedsolo import RSCodec, ReedSolomonError


# --- Channel specification --------------------------------------------------

CHANNEL_BITS = {
    "macro":  4,
    "count":  4,
    "height": 8,
    "angles": 8,
    "micro":  8,
}
PAYLOAD_BITS = sum(CHANNEL_BITS.values())  # 32
PAYLOAD_BYTES = PAYLOAD_BITS // 8           # 4
RS_PARITY_BYTES = 4                         # 32 bits of parity → can fix 2 errors or 4 erasures
TOTAL_BYTES = PAYLOAD_BYTES + RS_PARITY_BYTES  # 8

# Channel resilience ranking (most to least durable). Used during decode
# to decide which channels we trust when there's a conflict.
CHANNEL_RESILIENCE = ["macro", "count", "height", "angles", "micro"]


# --- Data classes -----------------------------------------------------------

@dataclass
class Channels:
    """Five geometric channels + Reed-Solomon parity bytes."""
    macro:  int          # 0..15
    count:  int          # 0..15  (rendered as 3..18 protrusions)
    height: int          # 0..255
    angles: int          # 0..255
    micro:  int          # 0..255
    rs:     bytes        # 2 bytes of Reed-Solomon parity

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rs"] = self.rs.hex()
        return d


@dataclass
class DecodeResult:
    """Result of a decode attempt, with confidence diagnostics."""
    item_id: Optional[int]              # None if unrecoverable
    confidence: float                    # 0.0..1.0
    channels_read: dict                  # which channels were readable
    rs_corrected: bool                   # did RS have to fix errors?
    fallback_level: str                  # "full" | "rs_corrected" | "category_only" | "lost"


# --- Bit packing helpers ----------------------------------------------------

def _channels_to_payload(macro: int, count: int, height: int,
                         angles: int, micro: int) -> bytes:
    """Pack 5 channel values into 4 bytes (32 bits)."""
    if not (0 <= macro  < 16):  raise ValueError("macro out of range")
    if not (0 <= count  < 16):  raise ValueError("count out of range")
    if not (0 <= height < 256): raise ValueError("height out of range")
    if not (0 <= angles < 256): raise ValueError("angles out of range")
    if not (0 <= micro  < 256): raise ValueError("micro out of range")

    word = (macro << 28) | (count << 24) | (height << 16) | (angles << 8) | micro
    return word.to_bytes(4, "big")


def _payload_to_channels(payload: bytes) -> tuple[int, int, int, int, int]:
    """Unpack 4 bytes back into 5 channel values."""
    word = int.from_bytes(payload, "big")
    macro  = (word >> 28) & 0x0F
    count  = (word >> 24) & 0x0F
    height = (word >> 16) & 0xFF
    angles = (word >>  8) & 0xFF
    micro  =  word        & 0xFF
    return macro, count, height, angles, micro


# --- Public API -------------------------------------------------------------

class PatternCodec:
    """Encode/decode 32-bit item IDs through 5 geometric channels."""

    def __init__(self, rs_parity_bytes: int = RS_PARITY_BYTES):
        self.rs = RSCodec(rs_parity_bytes)
        self.rs_parity_bytes = rs_parity_bytes

    # -- encode --------------------------------------------------------------

    def encode(self, item_id: int) -> Channels:
        """Turn a 32-bit item_id into 5 channels + RS parity."""
        if not (0 <= item_id < 2**32):
            raise ValueError("item_id must fit in 32 bits")

        payload = item_id.to_bytes(4, "big")
        encoded = bytes(self.rs.encode(payload))   # 4 payload + 2 parity = 6 bytes
        parity  = encoded[PAYLOAD_BYTES:]

        macro, count, height, angles, micro = _payload_to_channels(payload)
        return Channels(macro, count, height, angles, micro, parity)

    # -- decode --------------------------------------------------------------

    def decode(self, channels: dict) -> DecodeResult:
        """
        Decode channels back into an item_id.

        `channels` is a dict where any channel may be:
            int / bytes  — a confidently read value
            None         — channel could not be read (worn off)

        The decoder tries:
          1. If all 5 channels read cleanly → reconstruct payload, verify with RS.
          2. If micro/angles/height are missing → use RS parity to recover them.
          3. If too much is missing for full recovery → return category only
             (macro + count, no full ID).
        """
        readable = {k: v for k, v in channels.items() if v is not None}
        n_read = len(readable)

        # Build payload bytes, marking erasures where channels are missing.
        # reedsolo can correct (parity / 2) errors OR (parity) erasures.
        # With parity=2 we can fix 1 error or 2 erasures.

        # Strategy: build the 6-byte codeword with placeholder zeros where
        # bytes are unknown, and pass erasure positions to RS.
        macro  = readable.get("macro")
        count  = readable.get("count")
        height = readable.get("height")
        angles = readable.get("angles")
        micro  = readable.get("micro")
        rs     = readable.get("rs")

        # Each payload byte depends on which channels:
        #   byte 0 (high) = macro<<4 | count       — needs both
        #   byte 1        = height                  — needs height
        #   byte 2        = angles                  — needs angles
        #   byte 3 (low)  = micro                   — needs micro
        #   bytes 4-5     = rs parity               — needs rs
        erasure_positions: list[int] = []
        b0 = ((macro & 0x0F) << 4) | (count & 0x0F) if (macro is not None and count is not None) else 0
        if macro is None or count is None:
            erasure_positions.append(0)
        b1 = height if height is not None else 0
        if height is None:
            erasure_positions.append(1)
        b2 = angles if angles is not None else 0
        if angles is None:
            erasure_positions.append(2)
        b3 = micro if micro is not None else 0
        if micro is None:
            erasure_positions.append(3)

        if rs is not None and len(rs) == self.rs_parity_bytes:
            rs_bytes_list = list(rs)
        else:
            rs_bytes_list = [0] * self.rs_parity_bytes
            for i in range(self.rs_parity_bytes):
                erasure_positions.append(PAYLOAD_BYTES + i)

        codeword = bytes([b0, b1, b2, b3] + rs_bytes_list)

        # Try RS recovery
        try:
            decoded, _, errata = self.rs.decode(codeword, erase_pos=erasure_positions)
            payload = bytes(decoded)
            recovered_macro, recovered_count, recovered_height, \
                recovered_angles, recovered_micro = _payload_to_channels(payload)
            item_id = int.from_bytes(payload, "big")
            rs_corrected = bool(errata)

            # Confidence: 6 readable "slots" total = macro, count, height, angles, micro, rs.
            # rs counts as one slot regardless of byte-length.
            slot_count = sum(1 for k in ("macro", "count", "height", "angles", "micro", "rs")
                             if channels.get(k) is not None)
            base_confidence = slot_count / 6.0
            if rs_corrected:
                base_confidence *= 0.9
            confidence = round(base_confidence, 3)

            if slot_count == 6 and not rs_corrected:
                level = "full"
            else:
                level = "rs_corrected"

            return DecodeResult(
                item_id=item_id,
                confidence=confidence,
                channels_read={k: (v is not None) for k, v in channels.items()},
                rs_corrected=rs_corrected,
                fallback_level=level,
            )
        except ReedSolomonError:
            # Too much damage — RS can't recover the full ID.
            # If we still have macro+count we can give the user a category hint.
            if macro is not None and count is not None:
                return DecodeResult(
                    item_id=None,
                    confidence=round(2.0 / 6.0, 3),
                    channels_read={k: (v is not None) for k, v in channels.items()},
                    rs_corrected=False,
                    fallback_level="category_only",
                )
            return DecodeResult(
                item_id=None,
                confidence=0.0,
                channels_read={k: (v is not None) for k, v in channels.items()},
                rs_corrected=False,
                fallback_level="lost",
            )


# --- Convenience ------------------------------------------------------------

def stl_hash(stl_bytes: bytes) -> str:
    """SHA-256 of a generated STL — used as `stl_hash` field in Pattern Service."""
    return hashlib.sha256(stl_bytes).hexdigest()


__all__ = ["PatternCodec", "Channels", "DecodeResult", "stl_hash",
           "CHANNEL_BITS", "PAYLOAD_BITS"]
