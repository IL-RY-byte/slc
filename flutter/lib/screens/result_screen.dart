import 'package:flutter/material.dart';
import '../codec/pattern_codec.dart';
import '../ml/depth_estimator.dart';

class ResultScreen extends StatelessWidget {
  final int? decodedId;
  final Channels channels;
  final double confidence;
  final String fallbackLevel;
  final double estimatedWear;
  final DepthSource? depthSource;
  final Map<String, int>? diagnostics;

  const ResultScreen({
    super.key,
    required this.decodedId,
    required this.channels,
    required this.confidence,
    required this.fallbackLevel,
    this.estimatedWear = 0.0,
    this.depthSource,
    this.diagnostics,
  });

  static const _paper     = Color(0xFFF4EDE0);
  static const _paperDark = Color(0xFFEBE2D0);
  static const _ink       = Color(0xFF15110B);
  static const _inkSoft   = Color(0xFF4A4034);
  static const _rule      = Color(0xFFC4B89E);
  static const _accent    = Color(0xFFB8472B);

  @override
  Widget build(BuildContext context) {
    final family      = macroFamilies[channels.macro];
    final protrusions = 3 + (channels.count % 16);
    final heightMm    = (1.5 + (channels.height / 255.0) * 6.5).toStringAsFixed(2);

    return Scaffold(
      backgroundColor: _paper,
      appBar: AppBar(
        backgroundColor: _paper,
        foregroundColor: _ink,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.close, color: _ink),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: const Text(
          'PATTERN READ',
          style: TextStyle(
            fontFamily: 'JetBrainsMono',
            fontSize: 11,
            letterSpacing: 3.0,
            fontWeight: FontWeight.w500,
            color: _ink,
          ),
        ),
        centerTitle: true,
        bottom: const PreferredSize(
          preferredSize: Size.fromHeight(1.5),
          child: Divider(height: 1.5, thickness: 1.5, color: _ink),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(28, 24, 28, 32),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [

            _ConfidenceBadge(confidence: confidence, level: fallbackLevel),
            const SizedBox(height: 16),

            // Wear + depth source indicators
            if (estimatedWear > 0.05 || depthSource != null)
              _buildWearRow(estimatedWear, depthSource),

            const SizedBox(height: 20),

            // ----------------------------------------------------------------
            // Failure state: lost
            // ----------------------------------------------------------------
            if (fallbackLevel == 'lost') ...[
              _buildLostState(context),
            ] else ...[

              // Item title
              Text(
                decodedId != null
                    ? '0x${decodedId!.toRadixString(16).toUpperCase().padLeft(8, '0')}'
                    : 'Partial read',
                style: const TextStyle(
                  fontFamily: 'JetBrainsMono',
                  fontSize: 28,
                  fontWeight: FontWeight.w600,
                  color: _ink,
                  letterSpacing: 0.5,
                ),
              ),
              const SizedBox(height: 6),
              Row(
                children: [
                  Text(
                    family.toUpperCase(),
                    style: const TextStyle(
                      fontFamily: 'JetBrainsMono',
                      fontSize: 10,
                      letterSpacing: 2.4,
                      color: _accent,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(width: 10),
                  Container(width: 1, height: 10, color: _rule),
                  const SizedBox(width: 10),
                  Text(
                    '$protrusions-fold symmetry'.toUpperCase(),
                    style: const TextStyle(
                      fontFamily: 'JetBrainsMono',
                      fontSize: 10,
                      letterSpacing: 2.4,
                      color: _inkSoft,
                    ),
                  ),
                ],
              ),

              // ----------------------------------------------------------------
              // category_only: explain what was recovered
              // ----------------------------------------------------------------
              if (fallbackLevel == 'category_only') ...[
                const SizedBox(height: 20),
                _buildCategoryOnlyNote(family, protrusions),
              ],

              const SizedBox(height: 28),

              // Provenance card (mock data — wired to registry in production)
              if (decodedId != null) ...[
                _buildProvenanceCard(),
                const SizedBox(height: 28),
              ],

              // "What we read" — channel match status
              const _SectionHeader('What we read'),
              const SizedBox(height: 12),
              _buildChannelReadout(channels, fallbackLevel),
              const SizedBox(height: 36),

              // Action row
              _buildActionRow(context),
            ],
          ],
        ),
      ),
    );
  }

  // -------------------------------------------------------------------------

  Widget _buildWearRow(double wear, DepthSource? source) {
    final wearPct = (wear * 100).round();
    final items = <Widget>[];

    if (wear > 0.05) {
      items.add(_MetaChip(
        label: 'Wear ~$wearPct %',
        color: wear > 0.5 ? const Color(0xFF8A3621) : const Color(0xFF7A5811),
      ));
    }
    if (source != null && source != DepthSource.mock) {
      final label = switch (source) {
        DepthSource.lidar  => 'LiDAR depth',
        DepthSource.arCore => 'ARCore depth',
        DepthSource.midas  => 'MiDaS depth ⚠',
        _                  => '',
      };
      if (label.isNotEmpty) {
        if (items.isNotEmpty) items.add(const SizedBox(width: 8));
        items.add(_MetaChip(
          label: label,
          color: source == DepthSource.midas
              ? const Color(0xFF7A5811)
              : const Color(0xFF4A4034),
        ));
      }
    }

    return Wrap(children: items);
  }

  Widget _buildLostState(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(border: Border.all(color: _rule, width: 0.5)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Tile not detected',
            style: TextStyle(
              fontFamily: 'CormorantGaramond',
              fontStyle: FontStyle.italic,
              fontSize: 22,
              color: _ink,
              fontWeight: FontWeight.w500,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'Try better lighting, move closer, or hold the phone steady '
            'until the confidence meter fills.',
            style: TextStyle(
              fontFamily: 'CormorantGaramond',
              fontSize: 16,
              color: _inkSoft,
              height: 1.5,
            ),
          ),
          const SizedBox(height: 20),
          _Btn(
            text: 'Try again',
            onTap: () => Navigator.of(context).pop(),
          ),
        ],
      ),
    );
  }

  Widget _buildCategoryOnlyNote(String family, int protrusions) {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
      decoration: BoxDecoration(
        border: Border.all(color: const Color(0xFF7A5811), width: 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'This is a $protrusions-fold $family pattern, but the full '
            'identifier couldn\'t be recovered.',
            style: const TextStyle(
              fontFamily: 'CormorantGaramond',
              fontStyle: FontStyle.italic,
              fontSize: 16,
              color: _ink,
              height: 1.5,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'Try scanning from a cleaner angle or under better light.',
            style: TextStyle(
              fontFamily: 'JetBrainsMono',
              fontSize: 10,
              letterSpacing: 1.6,
              color: Color(0xFF7A5811),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildProvenanceCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: _paperDark.withOpacity(0.4),
        border: Border.all(color: _rule, width: 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: const [
          _ProvenanceLine(label: 'Maker',      value: 'Atelier No. 03 / Lviv'),
          SizedBox(height: 8),
          _ProvenanceLine(label: 'Collection', value: 'Spring · MMXXVI'),
          SizedBox(height: 8),
          _ProvenanceLine(label: 'Garment',    value: 'Linen pleated overcoat'),
          SizedBox(height: 8),
          _ProvenanceLine(label: 'Material',   value: '70 % linen · 30 % silk'),
          SizedBox(height: 8),
          _ProvenanceLine(label: 'Care',       value: 'Cool wash, line dry'),
        ],
      ),
    );
  }

  Widget _buildChannelReadout(Channels ch, String level) {
    final heightMm = (1.5 + (ch.height / 255.0) * 6.5).toStringAsFixed(2);
    final family   = macroFamilies[ch.macro];
    final protrusions = 3 + (ch.count % 16);

    // Channel "match" status — whether a channel was read vs recovered by RS
    final bool macroRead  = level != 'lost';
    final bool countRead  = level != 'lost';
    final bool heightRead = level == 'full' || level == 'rs_corrected';
    final bool anglesRead = level == 'full' || level == 'rs_corrected';
    final bool microRead  = false; // always RS-recovered in v1
    final bool rsRead     = level != 'lost';

    return _ChannelGrid(
      entries: [
        _ChannelEntry('macro',  '${ch.macro}', family,        _ResilienceTier.high,   read: macroRead),
        _ChannelEntry('count',  '${ch.count}', '$protrusions protrusions',
            _ResilienceTier.high,   read: countRead),
        _ChannelEntry('height', '${ch.height}', '$heightMm mm',
            _ResilienceTier.medium, read: heightRead),
        _ChannelEntry('angles', '${ch.angles}',
            '${((ch.angles >> 4) / 16 * 360).toStringAsFixed(0)}° base',
            _ResilienceTier.medium, read: anglesRead),
        _ChannelEntry('micro',  '${ch.micro}', 'RS-recovered · texture seed',
            _ResilienceTier.low, read: microRead),
        _ChannelEntry(
          'rs parity',
          ch.rs.map((b) => b.toRadixString(16).padLeft(2, '0')).join(),
          'Reed-Solomon (4 bytes)',
          null,
          fullWidth: true,
          read: rsRead,
        ),
      ],
    );
  }

  Widget _buildActionRow(BuildContext context) {
    return Row(
      children: [
        if (decodedId != null)
          Expanded(
            child: _Btn(text: 'Verify provenance', primary: true, onTap: () {}),
          ),
        if (decodedId != null) const SizedBox(width: 12),
        _Btn(text: 'New scan', onTap: () => Navigator.of(context).pop()),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Subcomponents
// ---------------------------------------------------------------------------

class _ConfidenceBadge extends StatelessWidget {
  final double confidence;
  final String level;
  const _ConfidenceBadge({required this.confidence, required this.level});

  @override
  Widget build(BuildContext context) {
    final (tone, label) = switch (level) {
      'full'          => (const Color(0xFF2D5A3D), 'Full ID · clean read'),
      'rs_corrected'  => (const Color(0xFFB8472B), 'RS-corrected · partial wear'),
      'category_only' => (const Color(0xFF7A5811), 'Category-only · heavily worn'),
      _               => (const Color(0xFF8A3621), 'Lost · pattern unreadable'),
    };
    final pct = level == 'lost' ? '—' : '${(confidence * 100).toStringAsFixed(0)}%';

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(border: Border.all(color: tone, width: 0.8)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 6, height: 6,
            decoration: BoxDecoration(color: tone, shape: BoxShape.circle),
          ),
          const SizedBox(width: 10),
          Text(
            label.toUpperCase(),
            style: TextStyle(
              fontFamily: 'JetBrainsMono',
              fontSize: 10,
              letterSpacing: 2.0,
              color: tone,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(width: 12),
          Text(
            pct,
            style: TextStyle(
              fontFamily: 'JetBrainsMono',
              fontSize: 11,
              letterSpacing: 1.0,
              color: tone,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

class _MetaChip extends StatelessWidget {
  final String label;
  final Color color;
  const _MetaChip({required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(border: Border.all(color: color, width: 0.5)),
      child: Text(
        label.toUpperCase(),
        style: TextStyle(
          fontFamily: 'JetBrainsMono',
          fontSize: 8.5,
          letterSpacing: 1.6,
          color: color,
        ),
      ),
    );
  }
}

class _ProvenanceLine extends StatelessWidget {
  final String label;
  final String value;
  const _ProvenanceLine({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        SizedBox(
          width: 110,
          child: Text(
            label.toUpperCase(),
            style: const TextStyle(
              fontFamily: 'JetBrainsMono',
              fontSize: 9,
              letterSpacing: 2.0,
              color: Color(0xFF4A4034),
            ),
          ),
        ),
        Expanded(
          child: Text(
            value,
            style: const TextStyle(
              fontFamily: 'CormorantGaramond',
              fontStyle: FontStyle.italic,
              fontSize: 16,
              color: Color(0xFF15110B),
              height: 1.2,
            ),
          ),
        ),
      ],
    );
  }
}

class _SectionHeader extends StatelessWidget {
  final String text;
  const _SectionHeader(this.text);
  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Text(
          text,
          style: const TextStyle(
            fontFamily: 'CormorantGaramond',
            fontStyle: FontStyle.italic,
            fontSize: 19,
            fontWeight: FontWeight.w500,
            color: Color(0xFF15110B),
          ),
        ),
        const SizedBox(width: 12),
        const Expanded(
          child: Divider(color: Color(0xFFC4B89E), height: 1, thickness: 0.5),
        ),
      ],
    );
  }
}

enum _ResilienceTier { high, medium, low }

class _ChannelEntry {
  final String name;
  final String value;
  final String unit;
  final _ResilienceTier? tier;
  final bool fullWidth;
  final bool read; // true = directly read; false = RS-recovered or absent

  const _ChannelEntry(
    this.name, this.value, this.unit, this.tier, {
    this.fullWidth = false,
    this.read = true,
  });
}

class _ChannelGrid extends StatelessWidget {
  final List<_ChannelEntry> entries;
  const _ChannelGrid({required this.entries});

  @override
  Widget build(BuildContext context) {
    final full   = entries.where((e) => e.fullWidth).toList();
    final paired = entries.where((e) => !e.fullWidth).toList();

    final rows = <Widget>[];
    for (int i = 0; i < paired.length; i += 2) {
      rows.add(Row(
        children: [
          Expanded(child: _ChannelTile(entry: paired[i])),
          const SizedBox(width: 1),
          if (i + 1 < paired.length)
            Expanded(child: _ChannelTile(entry: paired[i + 1]))
          else
            const Expanded(child: SizedBox()),
        ],
      ));
      if (i + 2 < paired.length) rows.add(const SizedBox(height: 1));
    }
    for (final e in full) {
      rows.add(const SizedBox(height: 1));
      rows.add(_ChannelTile(entry: e));
    }

    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFFC4B89E),
        border: Border.all(color: const Color(0xFFC4B89E)),
      ),
      child: Column(children: rows),
    );
  }
}

class _ChannelTile extends StatelessWidget {
  final _ChannelEntry entry;
  const _ChannelTile({required this.entry});

  @override
  Widget build(BuildContext context) {
    const paper   = Color(0xFFF4EDE0);
    const ink     = Color(0xFF15110B);
    const inkSoft = Color(0xFF4A4034);

    // Dim the tile slightly when the channel was RS-recovered (not directly read)
    final tileColor = entry.read ? paper : paper.withOpacity(0.7);
    final valueColor = entry.read ? ink : inkSoft;

    return Container(
      color: tileColor,
      padding: const EdgeInsets.fromLTRB(14, 10, 14, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  entry.name.toUpperCase(),
                  style: const TextStyle(
                    fontFamily: 'JetBrainsMono',
                    fontSize: 8.5,
                    letterSpacing: 2.4,
                    color: inkSoft,
                  ),
                ),
              ),
              if (!entry.read)
                const _RecoveredBadge()
              else if (entry.tier != null)
                _TierChip(tier: entry.tier!),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            entry.value,
            style: TextStyle(
              fontFamily: 'JetBrainsMono',
              fontSize: entry.fullWidth ? 13 : 16,
              fontWeight: FontWeight.w600,
              color: valueColor,
            ),
            overflow: TextOverflow.ellipsis,
          ),
          if (entry.unit.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 2),
              child: Text(
                entry.unit,
                style: const TextStyle(
                  fontFamily: 'CormorantGaramond',
                  fontStyle: FontStyle.italic,
                  fontSize: 13,
                  color: inkSoft,
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class _RecoveredBadge extends StatelessWidget {
  const _RecoveredBadge();
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
      decoration: BoxDecoration(
        border: Border.all(color: const Color(0xFF7A5811), width: 0.5),
      ),
      child: const Text(
        'RS',
        style: TextStyle(
          fontFamily: 'JetBrainsMono',
          fontSize: 8,
          letterSpacing: 1.0,
          color: Color(0xFF7A5811),
        ),
      ),
    );
  }
}

class _TierChip extends StatelessWidget {
  final _ResilienceTier tier;
  const _TierChip({required this.tier});

  @override
  Widget build(BuildContext context) {
    final (label, color) = switch (tier) {
      _ResilienceTier.high   => ('High', const Color(0xFF2D5A3D)),
      _ResilienceTier.medium => ('Med',  const Color(0xFF7A5811)),
      _ResilienceTier.low    => ('Low',  const Color(0xFF8A3621)),
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
      decoration: BoxDecoration(border: Border.all(color: color, width: 0.5)),
      child: Text(
        label.toUpperCase(),
        style: TextStyle(
          fontFamily: 'JetBrainsMono',
          fontSize: 8,
          letterSpacing: 1.6,
          color: color,
        ),
      ),
    );
  }
}

class _Btn extends StatelessWidget {
  final String text;
  final bool primary;
  final VoidCallback onTap;
  const _Btn({required this.text, this.primary = false, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 18),
        decoration: BoxDecoration(
          color: primary ? const Color(0xFF15110B) : const Color(0xFFF4EDE0),
          border: Border.all(color: const Color(0xFF15110B)),
        ),
        alignment: Alignment.center,
        child: Text(
          text.toUpperCase(),
          style: TextStyle(
            fontFamily: 'JetBrainsMono',
            fontSize: 10,
            letterSpacing: 2.4,
            fontWeight: FontWeight.w600,
            color: primary ? const Color(0xFFF4EDE0) : const Color(0xFF15110B),
          ),
        ),
      ),
    );
  }
}
