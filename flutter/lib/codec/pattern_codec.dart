/// Pattern Codec — Dart port of codec/pattern_codec.py.
///
/// Mirrors the Python codec exactly so that an item_id encoded by the web/
/// generator decodes correctly on the phone.
///
/// RS decoder now supports full Berlekamp-Massey error correction (up to 2
/// errors at unknown positions) in addition to the existing erasure decoder.
/// This satisfies Stage 4 of the production pipeline spec.
library;

import 'dart:typed_data';

// ---------------------------------------------------------------------------
// Reed-Solomon over GF(256), primitive polynomial 0x11D (matches reedsolo).
// ---------------------------------------------------------------------------
class _GF256 {
  static final Uint8List exp = Uint8List(512);
  static final Uint8List log = Uint8List(256);
  static bool _initialized = false;

  static void _init() {
    if (_initialized) return;
    int x = 1;
    for (int i = 0; i < 255; i++) {
      exp[i] = x;
      log[x] = i;
      x <<= 1;
      if ((x & 0x100) != 0) x ^= 0x11D;
    }
    for (int i = 255; i < 512; i++) {
      exp[i] = exp[i - 255];
    }
    _initialized = true;
  }

  static int mul(int a, int b) {
    _init();
    if (a == 0 || b == 0) return 0;
    return exp[log[a] + log[b]];
  }

  static int div(int a, int b) {
    _init();
    if (a == 0) return 0;
    if (b == 0) throw ArgumentError('GF256: division by zero');
    return exp[(log[a] + 255 - log[b]) % 255];
  }

  static int pow(int a, int n) {
    _init();
    if (a == 0) return n == 0 ? 1 : 0;
    return exp[(log[a] * n) % 255];
  }

  static int eval(List<int> poly, int x) {
    _init();
    int result = 0;
    for (final coef in poly) {
      result = mul(result, x) ^ coef;
    }
    return result;
  }
}

class _ReedSolomon {
  final int nsym;
  late final Uint8List _generator;

  _ReedSolomon(this.nsym) {
    _GF256._init();
    List<int> g = [1];
    for (int i = 0; i < nsym; i++) {
      final next = List<int>.filled(g.length + 1, 0);
      for (int j = 0; j < g.length; j++) {
        next[j] ^= g[j];
        next[j + 1] ^= _GF256.mul(g[j], _GF256.exp[i]);
      }
      g = next;
    }
    _generator = Uint8List.fromList(g);
  }

  Uint8List encode(Uint8List msg) {
    final out = Uint8List(msg.length + nsym);
    for (int i = 0; i < msg.length; i++) {
      out[i] = msg[i];
    }
    for (int i = 0; i < msg.length; i++) {
      final coef = out[i];
      if (coef != 0) {
        for (int j = 1; j < _generator.length; j++) {
          out[i + j] ^= _GF256.mul(_generator[j], coef);
        }
      }
    }
    for (int i = 0; i < msg.length; i++) {
      out[i] = msg[i];
    }
    return out;
  }

  /// Decode with optional erasure positions.
  ///
  /// Supports three cases:
  ///   - No damage: syndromes all zero → return payload directly.
  ///   - Erasures only: modified syndromes are all zero → Forney on known positions.
  ///   - Mixed/errors: BM on (Forney-modified) syndromes finds unknown error positions;
  ///     combined Forney corrects everything. Throws [StateError] when capacity exceeded.
  Uint8List decode(Uint8List codeword, {List<int> erasures = const []}) {
    final n = codeword.length;

    // Compute syndromes S[i] = C(alpha^i) for i = 0..nsym-1
    final syndromes = Uint8List(nsym);
    bool allZero = true;
    for (int i = 0; i < nsym; i++) {
      int s = 0;
      for (int j = 0; j < n; j++) {
        s = _GF256.mul(s, _GF256.exp[i]) ^ codeword[j];
      }
      syndromes[i] = s;
      if (s != 0) allZero = false;
    }

    // Capacity check BEFORE allZero: an all-zero codeword with > nsym erasures
    // has trivially-zero syndromes but is not a valid decode.
    if (erasures.length > nsym) {
      throw StateError('too many erasures (${erasures.length} > $nsym)');
    }

    if (allZero) {
      return Uint8List.fromList(codeword.sublist(0, n - nsym));
    }

    List<int> allPositions;

    if (erasures.isEmpty) {
      // Error-only: BM → Chien → correct
      final sigma = _berlekampMassey(syndromes);
      final numErrors = sigma.length - 1;
      if (numErrors * 2 > nsym) {
        throw StateError('too many errors ($numErrors > ${nsym ~/ 2}) — uncorrectable');
      }
      allPositions = _chienSearch(sigma, n);
      if (allPositions.length != numErrors) {
        throw StateError('Chien search: found ${allPositions.length} roots, expected $numErrors');
      }
    } else {
      // Erasure or mixed: compute Forney-modified syndromes (which cancel the
      // erasure contribution), then run BM to find any additional errors.
      final fsynd = _modifiedSyndromes(syndromes, erasures, n);
      final usable  = nsym - erasures.length;
      final errSigma = _berlekampMassey(Uint8List.fromList(fsynd.sublist(0, usable)));
      final numErrors = errSigma.length - 1;
      if (2 * numErrors + erasures.length > nsym) {
        throw StateError('too many erasures+errors (2×$numErrors+${erasures.length} > $nsym)');
      }
      final errorPositions =
          numErrors > 0 ? _chienSearch(errSigma, n) : <int>[];
      if (errorPositions.length != numErrors) {
        throw StateError('Chien search mismatch: found ${errorPositions.length}, expected $numErrors');
      }
      allPositions = [...erasures, ...errorPositions];
    }

    return _correctAll(codeword, syndromes, allPositions, n);
  }

  // Forney-modified syndromes: folds each erasure locator into the syndrome
  // vector so that BM sees only the residual (unknown) error contribution.
  static List<int> _modifiedSyndromes(Uint8List syndromes, List<int> erasures, int n) {
    _GF256._init();
    final fsynd = List<int>.from(syndromes);
    for (final pos in erasures) {
      final x = _GF256.exp[(n - 1 - pos) % 255]; // alpha^(n-1-pos)
      for (int j = 0; j < fsynd.length - 1; j++) {
        fsynd[j] = _GF256.mul(fsynd[j], x) ^ fsynd[j + 1];
      }
    }
    return fsynd;
  }

  // Unified Forney correction for all positions (erasures + BM-found errors).
  // Uses the FCR=0 formula: e_pos = xi * omega(xInv) / sigma'(xInv)
  // where xi = alpha^(n-1-pos) and xInv = alpha^{-(n-1-pos)}.
  Uint8List _correctAll(
    Uint8List codeword,
    Uint8List syndromes,
    List<int> positions,
    int n,
  ) {
    // Build total error locator: sigma = prod(1 + alpha^(n-1-pos)*X)
    List<int> sigma = [1];
    for (final pos in positions) {
      final coef = _GF256.exp[(n - 1 - pos) % 255];
      final next = List<int>.filled(sigma.length + 1, 0);
      for (int j = 0; j < sigma.length; j++) {
        next[j] ^= sigma[j];
        next[j + 1] ^= _GF256.mul(sigma[j], coef);
      }
      sigma = next;
    }

    // Evaluator: omega = (S * sigma) mod x^nsym
    final omega = List<int>.filled(nsym, 0);
    for (int i = 0; i < nsym; i++) {
      int s = 0;
      for (int j = 0; j <= i && j < sigma.length; j++) {
        s ^= _GF256.mul(syndromes[i - j], sigma[j]);
      }
      omega[i] = s;
    }

    final corrected = Uint8List.fromList(codeword);
    for (final pos in positions) {
      final xi   = _GF256.exp[(n - 1 - pos) % 255];           // alpha^(n-1-pos)
      final xInv = _GF256.exp[(255 - (n - 1 - pos)) % 255];   // alpha^-(n-1-pos)

      // Formal derivative of sigma at xInv (odd-index terms only in GF(2^m))
      int sigmaPrime = 0;
      for (int j = 1; j < sigma.length; j += 2) {
        sigmaPrime ^= _GF256.mul(sigma[j], _GF256.pow(xInv, j - 1));
      }

      int omegaVal = 0;
      for (int j = 0; j < omega.length; j++) {
        omegaVal ^= _GF256.mul(omega[j], _GF256.pow(xInv, j));
      }

      if (sigmaPrime == 0) {
        throw StateError('Forney: zero derivative at position $pos — uncorrectable');
      }

      // FCR=0 Forney: multiply by xi before dividing by sigma'
      corrected[pos] ^= _GF256.mul(xi, _GF256.div(omegaVal, sigmaPrime));
    }

    return Uint8List.fromList(corrected.sublist(0, n - nsym));
  }

  // -------------------------------------------------------------------------
  // Berlekamp-Massey algorithm — returns error locator polynomial sigma.
  // sigma[0] = 1 always; len(sigma)-1 = number of errors found.
  // -------------------------------------------------------------------------
  static List<int> _berlekampMassey(Uint8List syndromes) {
    final n = syndromes.length;
    List<int> C = [1]; // current connection polynomial
    List<int> B = [1]; // previous connection polynomial
    int L = 0;
    int b = 1; // leading coefficient of B
    int m = 1; // shift register

    for (int i = 0; i < n; i++) {
      // Discrepancy
      int d = syndromes[i];
      for (int j = 1; j <= L && j < C.length; j++) {
        d ^= _GF256.mul(C[j], syndromes[i - j]);
      }

      if (d == 0) {
        m++;
      } else if (2 * L <= i) {
        final T = List<int>.from(C);
        final coeff = _GF256.div(d, b);
        // Extend C to accommodate shift
        while (C.length < B.length + m) C.add(0);
        for (int j = 0; j < B.length; j++) {
          C[j + m] ^= _GF256.mul(coeff, B[j]);
        }
        L = i + 1 - L;
        B = T;
        b = d;
        m = 1;
      } else {
        final coeff = _GF256.div(d, b);
        while (C.length < B.length + m) C.add(0);
        for (int j = 0; j < B.length; j++) {
          C[j + m] ^= _GF256.mul(coeff, B[j]);
        }
        m++;
      }
    }
    return C;
  }

  // -------------------------------------------------------------------------
  // Chien search — find all positions i where sigma(alpha^(-i)) == 0.
  // Returns error positions as indices into the codeword (0-based from left).
  // -------------------------------------------------------------------------
  static List<int> _chienSearch(List<int> sigma, int n) {
    _GF256._init();
    final positions = <int>[];
    for (int i = 0; i < n; i++) {
      // Evaluate sigma at alpha^(-(n-1-i)) = alpha^(i+1-n) = alpha^(255-(n-1-i))
      final xInv = _GF256.exp[(255 - (n - 1 - i)) % 255];
      int val = 0;
      for (int j = 0; j < sigma.length; j++) {
        val ^= _GF256.mul(sigma[j], _GF256.pow(xInv, j));
      }
      if (val == 0) positions.add(i);
    }
    return positions;
  }

}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Five geometric channels + Reed-Solomon parity bytes.
class Channels {
  final int macro;   // 0..15
  final int count;   // 0..15
  final int height;  // 0..255
  final int angles;  // 0..255
  final int micro;   // 0..255
  final Uint8List rs;

  const Channels({
    required this.macro,
    required this.count,
    required this.height,
    required this.angles,
    required this.micro,
    required this.rs,
  });

  Map<String, Object?> toMap() => {
        'macro': macro,
        'count': count,
        'height': height,
        'angles': angles,
        'micro': micro,
        'rs': rs.map((b) => b.toRadixString(16).padLeft(2, '0')).join(),
      };
}

/// Result of attempting to recover an item_id from observed channels.
class DecodeResult {
  final int? itemId;
  final double confidence;
  final bool rsCorrected;
  final String fallbackLevel; // 'full' | 'rs_corrected' | 'category_only' | 'lost'
  final double genuineConfidence; // reserved — hardcoded 1.0 until counterfeit CNN ships

  const DecodeResult({
    required this.itemId,
    required this.confidence,
    required this.rsCorrected,
    required this.fallbackLevel,
    this.genuineConfidence = 1.0,
  });
}

class PatternCodec {
  static const int payloadBytes  = 4;
  static const int rsParityBytes = 4;
  static const int totalBytes    = 8;

  final _ReedSolomon _rs = _ReedSolomon(rsParityBytes);

  Channels encode(int itemId) {
    if (itemId < 0 || itemId > 0xFFFFFFFF) {
      throw ArgumentError('itemId must be 0..2^32-1');
    }
    final payload = Uint8List(4)
      ..[0] = (itemId >> 24) & 0xFF
      ..[1] = (itemId >> 16) & 0xFF
      ..[2] = (itemId >> 8) & 0xFF
      ..[3] = itemId & 0xFF;
    final encoded = _rs.encode(payload);
    final parity  = encoded.sublist(payloadBytes);
    return Channels(
      macro:  (payload[0] >> 4) & 0x0F,
      count:  payload[0] & 0x0F,
      height: payload[1],
      angles: payload[2],
      micro:  payload[3],
      rs:     parity,
    );
  }

  /// Decode from a partially observed channel set.
  ///
  /// Pass null for any channel that was not readable. The decoder uses
  /// Reed-Solomon to recover missing bytes (erasures) or corrupted bytes
  /// (errors via Berlekamp-Massey). When RS cannot recover, falls back to
  /// category_only or lost.
  DecodeResult decode({
    int? macro,
    int? count,
    int? height,
    int? angles,
    int? micro,
    Uint8List? rs,
  }) {
    final erasures = <int>[];
    int b0 = 0, b1 = 0, b2 = 0, b3 = 0;

    if (macro != null && count != null) {
      b0 = ((macro & 0x0F) << 4) | (count & 0x0F);
    } else {
      erasures.add(0);
    }
    if (height != null) b1 = height; else erasures.add(1);
    if (angles != null) b2 = angles; else erasures.add(2);
    if (micro  != null) b3 = micro;  else erasures.add(3);

    Uint8List rsBytes;
    if (rs != null && rs.length == rsParityBytes) {
      rsBytes = rs;
    } else {
      rsBytes = Uint8List(rsParityBytes);
      erasures.addAll([4, 5, 6, 7]);
    }

    final cw = Uint8List(totalBytes)
      ..[0] = b0
      ..[1] = b1
      ..[2] = b2
      ..[3] = b3
      ..setRange(4, 8, rsBytes);

    try {
      final payload = _rs.decode(cw, erasures: erasures);
      final id = (payload[0] << 24) | (payload[1] << 16) |
                 (payload[2] << 8)  |  payload[3];
      final slotsRead = [macro, count, height, angles, micro, rs]
          .where((x) => x != null).length;
      final conf = slotsRead / 6.0;
      return DecodeResult(
        itemId:       id,
        confidence:   conf,
        rsCorrected:  erasures.isNotEmpty,
        fallbackLevel: slotsRead == 6 ? 'full' : 'rs_corrected',
      );
    } catch (_) {
      if (macro != null && count != null) {
        return DecodeResult(
          itemId:       null,
          confidence:   2.0 / 6.0,
          rsCorrected:  false,
          fallbackLevel: 'category_only',
        );
      }
      return const DecodeResult(
        itemId:       null,
        confidence:   0,
        rsCorrected:  false,
        fallbackLevel: 'lost',
      );
    }
  }
}

const macroFamilies = [
  'circle',   'triangle', 'hexagon',  'square',
  'star_5',   'star_6',   'star_8',   'rosette',
  'wave',     'pinwheel', 'diamond',  'octagon',
  'petal',    'gear',     'spiral',   'cross',
];
