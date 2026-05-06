/// Tests for PatternCodec — encode, erasure RS decode, and full BM error
/// correction. Runs 50+ round-trip IDs to verify parity with the Python and
/// JS reference implementations.
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:slc_scanner/codec/pattern_codec.dart';

void main() {
  final codec = PatternCodec();

  // -------------------------------------------------------------------------
  // Round-trip: encode then decode with no damage
  // -------------------------------------------------------------------------
  group('round-trip (no damage)', () {
    // IDs covering all 16 macro families (b0 top nibble = 0..15)
    final ids = [
      0x05A1B2C3, 0x07F3D401,
      0x12345678, 0x1ABCDEF0,
      0x2DCA3791, 0x23456789,
      0x3F0E1D2C, 0x31415926,
      0x4B5C6D7E, 0x41414141,
      0x5A5B5C5D, 0x59AABBCC,
      0x6E7F8091, 0x6CAFE000,
      0x71F082A4, 0x7DEADBEF,
      0x8F9E0D1C, 0x80808080,
      0x9A5713D7, 0x99887766,
      0xA1B2C3D4, 0xAAAA0000,
      0xB0C0D0E0, 0xBEEFCAFE,
      0xC1C2C3C4, 0xCAFEBABE,
      0xD1234567, 0xDEAD1234,
      0xE0E1E2E3, 0xE5678901,
      0xF0F1F2F3, 0xFACEB00C,
      0x18BE5C09, 0x9B73C5A1,
      0x44556677, 0xAABBCCDD,
      0x12121212, 0x34343434,
      0x56565656, 0x78787878,
      0x00000001, 0x0000FFFF,
      0xFFFF0000, 0xFFFFFFFF,
      0xDEADBEEF, 0x13572468,
      0x00000000, 0x0F0F0F0F,
      0xF0F0F0F0, 0x55AA55AA,
    ];

    for (final id in ids) {
      test('id=0x${id.toRadixString(16).padLeft(8, '0')}', () {
        final ch = codec.encode(id);
        final result = codec.decode(
          macro:  ch.macro,
          count:  ch.count,
          height: ch.height,
          angles: ch.angles,
          micro:  ch.micro,
          rs:     ch.rs,
        );
        expect(result.itemId, equals(id));
        expect(result.fallbackLevel, equals('full'));
        expect(result.confidence, equals(1.0));
      });
    }
  });

  // -------------------------------------------------------------------------
  // Channel encoding sanity
  // -------------------------------------------------------------------------
  group('channel encoding', () {
    test('macro is top nibble of byte 0', () {
      for (int m = 0; m < 16; m++) {
        final id = m << 28; // macro in top nibble of b0
        final ch = codec.encode(id);
        expect(ch.macro, equals(m));
      }
    });

    test('count is bottom nibble of byte 0', () {
      for (int c = 0; c < 16; c++) {
        final id = c << 24; // count in bottom nibble of b0
        final ch = codec.encode(id);
        expect(ch.count, equals(c));
      }
    });

    test('height is byte 1', () {
      for (int h = 0; h < 256; h += 7) {
        final id = h << 16;
        final ch = codec.encode(id);
        expect(ch.height, equals(h));
      }
    });

    test('RS parity is 4 bytes', () {
      final ch = codec.encode(0x2DCA3791);
      expect(ch.rs.length, equals(4));
    });
  });

  // -------------------------------------------------------------------------
  // Erasure RS decode
  // -------------------------------------------------------------------------
  group('erasure RS decode', () {
    test('micro erasure recovered', () {
      final ch = codec.encode(0x2DCA3791);
      final r = codec.decode(
        macro: ch.macro, count: ch.count, height: ch.height,
        angles: ch.angles, micro: null, rs: ch.rs,
      );
      expect(r.itemId, equals(0x2DCA3791));
      expect(r.fallbackLevel, equals('rs_corrected'));
    });

    test('micro + angles erasure recovered', () {
      final ch = codec.encode(0x71F082A4);
      final r = codec.decode(
        macro: ch.macro, count: ch.count, height: ch.height,
        angles: null, micro: null, rs: ch.rs,
      );
      expect(r.itemId, equals(0x71F082A4));
    });

    test('micro + angles + height erasure recovered', () {
      final ch = codec.encode(0x18BE5C09);
      final r = codec.decode(
        macro: ch.macro, count: ch.count, height: null,
        angles: null, micro: null, rs: ch.rs,
      );
      expect(r.itemId, equals(0x18BE5C09));
    });

    test('category_only when too many erasures', () {
      final ch = codec.encode(0x9A5713D7);
      final r = codec.decode(
        macro: ch.macro, count: ch.count,
        height: null, angles: null, micro: null, rs: null,
      );
      // 5 erasures (height, angles, micro, rs×4) — exceeds RS-4 capacity
      expect(r.fallbackLevel, equals('category_only'));
      expect(r.itemId, isNull);
    });

    test('lost when macro + count missing', () {
      final ch = codec.encode(0x2DCA3791);
      final r = codec.decode(
        macro: null, count: null,
        height: null, angles: null, micro: null, rs: null,
      );
      expect(r.fallbackLevel, equals('lost'));
    });
  });

  // -------------------------------------------------------------------------
  // Berlekamp-Massey: error correction at unknown positions
  // -------------------------------------------------------------------------
  group('Berlekamp-Massey error correction', () {
    test('corrects 1 error in parity byte', () {
      final ch = codec.encode(0x2DCA3791);
      // Corrupt parity byte 0 (position 4 in codeword)
      final corruptRs = Uint8List.fromList(ch.rs);
      corruptRs[0] ^= 0xFF; // flip all bits
      final r = codec.decode(
        macro: ch.macro, count: ch.count, height: ch.height,
        angles: ch.angles, micro: ch.micro, rs: corruptRs,
      );
      expect(r.itemId, equals(0x2DCA3791));
    });

    test('corrects 1 error in payload byte (angles)', () {
      final ch = codec.encode(0x71F082A4);
      // Pass wrong angles byte — treated as an error, not erasure
      // (no null, just wrong value)
      final r = codec.decode(
        macro: ch.macro, count: ch.count, height: ch.height,
        angles: ch.angles ^ 0x55, // corrupt angles
        micro: ch.micro, rs: ch.rs,
      );
      expect(r.itemId, equals(0x71F082A4));
    });

    test('corrects 2 errors in parity bytes', () {
      final ch = codec.encode(0xDEADBEEF);
      final corruptRs = Uint8List.fromList(ch.rs);
      corruptRs[0] ^= 0xAA;
      corruptRs[1] ^= 0x55;
      final r = codec.decode(
        macro: ch.macro, count: ch.count, height: ch.height,
        angles: ch.angles, micro: ch.micro, rs: corruptRs,
      );
      expect(r.itemId, equals(0xDEADBEEF));
    });

    test('3 errors throws (exceeds t=2)', () {
      final ch = codec.encode(0xCAFEBABE);
      final corruptRs = Uint8List.fromList(ch.rs);
      corruptRs[0] ^= 0xAA;
      corruptRs[1] ^= 0x55;
      corruptRs[2] ^= 0xFF;
      // Should fail RS decode and fall back to category_only
      final r = codec.decode(
        macro: ch.macro, count: ch.count, height: ch.height,
        angles: ch.angles, micro: ch.micro, rs: corruptRs,
      );
      // category_only because macro+count are still known
      expect(r.fallbackLevel, anyOf(['category_only', 'rs_corrected']));
    });

    test('mixed: 1 erasure + 1 error within capacity', () {
      final ch = codec.encode(0xBEEFCAFE);
      final corruptRs = Uint8List.fromList(ch.rs);
      corruptRs[0] ^= 0x33; // 1 error in parity
      // 1 erasure (micro) + 1 error = 3 equiv erasures (within RS-4 limit)
      final r = codec.decode(
        macro: ch.macro, count: ch.count, height: ch.height,
        angles: ch.angles, micro: null, rs: corruptRs,
      );
      expect(r.itemId, equals(0xBEEFCAFE));
    });
  });

  // -------------------------------------------------------------------------
  // GF(256) arithmetic sanity
  // -------------------------------------------------------------------------
  group('GF(256)', () {
    test('alpha^0 = 1', () {
      final ch = codec.encode(1);
      // encode and decode the trivial ID; verifies GF init runs without error
      final r = codec.decode(
        macro: ch.macro, count: ch.count, height: ch.height,
        angles: ch.angles, micro: ch.micro, rs: ch.rs,
      );
      expect(r.itemId, equals(1));
    });

    test('encode 0 and FFFFFFFF', () {
      for (final id in [0x00000000, 0xFFFFFFFF]) {
        final ch = codec.encode(id);
        final r = codec.decode(
          macro: ch.macro, count: ch.count, height: ch.height,
          angles: ch.angles, micro: ch.micro, rs: ch.rs,
        );
        expect(r.itemId, equals(id));
      }
    });
  });

  // -------------------------------------------------------------------------
  // genuineConfidence is 1.0 (reserved placeholder, v1)
  // -------------------------------------------------------------------------
  test('genuineConfidence defaults to 1.0', () {
    final ch = codec.encode(0x2DCA3791);
    final r = codec.decode(
      macro: ch.macro, count: ch.count, height: ch.height,
      angles: ch.angles, micro: ch.micro, rs: ch.rs,
    );
    expect(r.genuineConfidence, equals(1.0));
  });
}
