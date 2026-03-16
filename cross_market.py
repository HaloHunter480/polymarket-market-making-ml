"""
cross_market.py — Cross-Market Probability Constraint Engine
=============================================================

Markets in the same competition must obey probability constraints.
When they don't, there is a structural, model-free arbitrage.

──────────────────────────────────────────────────────────────────
THE CORE IDEA
──────────────────────────────────────────────────────────────────

Exactly 4 teams finish in the Bundesliga Top 4.  So across ALL
18 Bundesliga "top-4" markets, the probabilities MUST sum to 4.0:

  Bayern    0.95
  Dortmund  0.80
  Leipzig   0.62
  Leverkusen 0.56
  Frankfurt  0.48
  ...all 18 teams...
  ──────────────
  Observed sum  = 4.73   ← should be 4.0
  Violation     = +0.73  ← markets collectively overpriced

  Fair value of each market = market_mid × (4.0 / 4.73)

  Leverkusen fair = 0.34 × (4.0/4.73) = 0.287  → overpriced by 5.3c
  → SELL Leverkusen (post ask at fair_p, wait for mid to fall)

The constraint violation is a pure structural signal — no opinions
required about the sport.  The math is incontrovertible.

──────────────────────────────────────────────────────────────────
CONSTRAINT TYPES
──────────────────────────────────────────────────────────────────

  TOP_N          "Will X finish in top N?" — sum = N
  RELEGATED      "Will X be relegated?"    — sum = N_relegated
  WIN_OUTRIGHT   "Will X win [trophy]?"    — sum = 1.0
  MATCH_WINNER   "Will X win this match?"  — sum ≈ 1 (split 3-way for draw)
  PLAYOFFS       "Will X make playoffs?"   — sum = N_playoff_spots

──────────────────────────────────────────────────────────────────
OUTPUT
──────────────────────────────────────────────────────────────────

For every market in a detected group:
  cross_edge_cents   (p_fair - p_mid) * 100
                     NEGATIVE = overpriced (sell) — group sum > target
                     POSITIVE = underpriced (buy)  — group sum < target
  cross_group_id     identifier for the constraint group
  cross_group_size   number of markets in the group
  cross_violation    observed_sum - target_sum (the raw violation in price units)

When cross_edge_cents confirms prob_engine edge_cents (same sign),
market_maker.py upgrades alpha_mode one level:
  PASSIVE_MM  → SKEWED_MM
  SKEWED_MM   → DIRECTIONAL
"""

import re
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("cross_market")

# ── Minimum group size to trust the constraint ───────────────────────────────
MIN_GROUP_SIZE       = 3     # need at least 3 markets to enforce a constraint
MIN_VIOLATION_CENTS  = 1.0   # ignore groups with < 1 cent per-market violation

# ── Cross-market edge upgrade threshold ──────────────────────────────────────
# When cross_edge confirms prob_engine edge, upgrade alpha_mode if:
CROSS_CONFIRM_THRESHOLD = 3.0   # cents — both edges must be >= this to confirm


# ── Constraint pattern definitions ───────────────────────────────────────────
# Each entry: (regex, constraint_type, target_N_if_fixed, league_keywords)
#   target_N_if_fixed: int or None (None = auto-detect from regex group)
_PATTERNS = [
    # Top-N finish in a league/tournament
    (r'(?i)\btop[\s\-]*(\d+)\b',          "TOP_N",        None),
    # Relegated / promotion playoffs
    (r'(?i)\breleg(?:at)',                  "RELEGATED",    None),
    (r'(?i)\bpromot(?:ed|ion)',             "PROMOTED",     None),
    # Outright winner (one trophy, one winner)
    (r'(?i)\bwill [\w\s]+ win (?:the )?(?:championship|title|league|cup|series|trophy)',
                                            "WIN_OUTRIGHT", 1),
    # Match winner (within a single game — these share event_slug anyway)
    (r'(?i)\bwill [\w\s]+ win (?:this )?(?:game|match)',
                                            "MATCH_WINNER", None),
    # NBA / NFL / MLB playoff berths
    (r'(?i)(?:make|reach|qualify for).*?playoffs?',
                                            "PLAYOFFS",     None),
    # Make the finals / conference finals
    (r'(?i)(?:make|reach|advance to).*?finals?',
                                            "FINALS",       None),
    # Score-based (total-goals variants) — nested constraint handled separately
    (r'(?i)\bscore (?:over|more than) (\d+)',
                                            "GOALS_OVER_N", None),
]

# ── Competition / league keyword extractor ────────────────────────────────────
# Maps fragment → canonical competition key used as group prefix
_COMPETITION_KEYWORDS = [
    ("bundesliga",        "bundesliga"),
    ("premier league",    "premier_league"),
    ("la liga",           "la_liga"),
    ("ligue 1",           "ligue1"),
    ("serie a",           "serie_a"),
    ("champions league",  "ucl"),
    ("europa league",     "uel"),
    ("nba",               "nba"),
    ("nfl",               "nfl"),
    ("mlb",               "mlb"),
    ("nhl",               "nhl"),
    ("world cup",         "world_cup"),
    ("euro 20",           "euros"),
    ("copa america",      "copa_america"),
    ("super bowl",        "super_bowl"),
    ("march madness",     "march_madness"),
    ("ncaa",              "ncaa"),
]

# ── Number of relegation/promotion slots per competition ─────────────────────
_RELEGATION_SLOTS = {
    "bundesliga":    3,
    "premier_league": 3,
    "la_liga":       3,
    "ligue1":        3,
    "serie_a":       3,
}
_PROMOTION_SLOTS = {
    "bundesliga":    2,   # 2 automatic + 1 playoff
    "premier_league": 3,
}


@dataclass
class ConstraintGroup:
    group_id:        str            # e.g. "bundesliga:TOP_N:4"
    constraint_type: str            # "TOP_N" | "WIN_OUTRIGHT" | etc.
    target_sum:      float          # what the probabilities must sum to
    members:         list[str] = field(default_factory=list)   # slugs
    observed_sum:    float = 0.0    # actual sum of mids
    violation:       float = 0.0    # observed_sum - target_sum
    violation_pct:   float = 0.0    # violation / target_sum  (signed %)

    def per_market_edge(self, market_mid: float) -> float:
        """
        Fair-value adjusted cross-edge for one market (in price units).
        Negative = overpriced (group sum > target).
        """
        if self.observed_sum <= 0 or len(self.members) < MIN_GROUP_SIZE:
            return 0.0
        fair_p = market_mid * (self.target_sum / self.observed_sum)
        return round(fair_p - market_mid, 4)


def _extract_competition(text: str, slug: str) -> Optional[str]:
    """Return canonical competition key from question text or slug."""
    combined = (text + " " + slug).lower()
    for kw, key in _COMPETITION_KEYWORDS:
        if kw in combined:
            return key
    return None


def _detect_constraint(question: str, slug: str) -> Optional[tuple[str, float]]:
    """
    Returns (constraint_type, target_N_or_None) or None if no pattern matches.
    target_N is the fixed value for the sum constraint (e.g. 4 for top-4).
    None means it must be inferred from the group (e.g. relegation slots).
    """
    for pattern, ctype, fixed_n in _PATTERNS:
        m = re.search(pattern, question)
        if not m:
            m = re.search(pattern, slug)
        if m:
            if fixed_n is not None:
                return ctype, float(fixed_n)
            # Try to extract N from the regex match group
            try:
                n = int(m.group(1))
                return ctype, float(n)
            except (IndexError, ValueError):
                return ctype, None
    return None


def _infer_target(group_id: str, constraint_type: str,
                  competition: Optional[str]) -> Optional[float]:
    """Infer target_sum when the regex couldn't extract N directly."""
    if constraint_type == "RELEGATED":
        return float(_RELEGATION_SLOTS.get(competition or "", 3))
    if constraint_type == "PROMOTED":
        return float(_PROMOTION_SLOTS.get(competition or "", 2))
    if constraint_type in ("WIN_OUTRIGHT", "MATCH_WINNER"):
        return 1.0
    if constraint_type in ("PLAYOFFS", "FINALS"):
        # Can't infer without knowing bracket size — leave for group-size heuristic
        return None
    return None


def detect_constraint_groups(opps: list) -> dict[str, ConstraintGroup]:
    """
    Scan all passed MarketOpp objects and detect constraint groups.

    Groups markets by:
      1. event_slug  (same Polymarket event — most precise)
      2. competition + constraint_type + N  (cross-event, e.g. season-long standings)

    Returns dict: group_id → ConstraintGroup  (all members listed, no mids yet)
    """
    groups: dict[str, ConstraintGroup] = {}

    for opp in opps:
        q    = opp.question
        slug = opp.slug
        mid  = (opp.best_bid + opp.best_ask) / 2 if opp.best_bid > 0 else opp.mid_gamma

        detected = _detect_constraint(q, slug)
        if not detected:
            continue

        ctype, target_n = detected
        competition = _extract_competition(q, slug)

        # ── Group key ──────────────────────────────────────────────────────
        # Prefer event_slug for match-level grouping
        event_slug = getattr(opp, "event_slug", "") or ""
        if event_slug and ctype in ("MATCH_WINNER", "FINALS"):
            gid = f"event:{event_slug}:{ctype}"
        elif competition and target_n is not None:
            gid = f"{competition}:{ctype}:{int(target_n)}"
        elif competition and ctype in ("WIN_OUTRIGHT", "RELEGATED", "PROMOTED", "PLAYOFFS"):
            gid = f"{competition}:{ctype}"
        else:
            # Not enough context to group reliably
            continue

        # ── Target sum ─────────────────────────────────────────────────────
        target_sum = target_n
        if target_sum is None:
            target_sum = _infer_target(gid, ctype, competition)
        if target_sum is None:
            continue   # can't determine constraint — skip

        # ── Add to group ───────────────────────────────────────────────────
        if gid not in groups:
            groups[gid] = ConstraintGroup(
                group_id        = gid,
                constraint_type = ctype,
                target_sum      = target_sum,
            )
        groups[gid].members.append(slug)

    return groups


def compute_violations(
    groups: dict[str, ConstraintGroup],
    opp_map: dict[str, object],
) -> dict[str, ConstraintGroup]:
    """
    Fill in observed_sum and violation for each group using current market mids.

    opp_map: slug → MarketOpp
    Returns the same groups dict with violations filled in.
    """
    for gid, grp in list(groups.items()):
        if len(grp.members) < MIN_GROUP_SIZE:
            del groups[gid]
            continue

        total = 0.0
        valid = 0
        for slug in grp.members:
            opp = opp_map.get(slug)
            if not opp:
                continue
            mid = ((opp.best_bid + opp.best_ask) / 2
                   if opp.best_bid > 0 else opp.mid_gamma)
            if 0.02 < mid < 0.98:   # ignore locked markets
                total += mid
                valid += 1

        if valid < MIN_GROUP_SIZE:
            del groups[gid]
            continue

        grp.observed_sum  = round(total, 4)
        grp.violation     = round(total - grp.target_sum, 4)
        grp.violation_pct = round(grp.violation / grp.target_sum * 100, 2)

        if abs(grp.violation) > 0.01:
            direction = "OVER" if grp.violation > 0 else "UNDER"
            per_mkt = grp.violation / valid
            log.info(
                "CONSTRAINT %s [%s n=%d]: sum=%.3f target=%.1f "
                "violation=%+.3f (%s %.1fc/mkt)",
                gid, grp.constraint_type, valid,
                grp.observed_sum, grp.target_sum,
                grp.violation, direction, abs(per_mkt) * 100,
            )

    return groups


def apply_cross_market_edge(
    opps: list,
    groups: dict[str, ConstraintGroup],
) -> None:
    """
    In-place: set cross_edge_cents, cross_group_id, cross_group_size,
    cross_violation on each MarketOpp that belongs to a constraint group.

    Also upgrades alpha_mode when cross_edge confirms prob_engine edge:
      PASSIVE_MM  → SKEWED_MM   (when both edges ≥ CROSS_CONFIRM_THRESHOLD)
      SKEWED_MM   → DIRECTIONAL (when cross_edge ≥ 3c confirming prob_engine)
    """
    # Build reverse map: slug → group
    slug_to_group: dict[str, ConstraintGroup] = {}
    for grp in groups.values():
        for slug in grp.members:
            # A slug might be in multiple groups — keep the one with largest violation
            if slug not in slug_to_group or (
                abs(grp.violation) > abs(slug_to_group[slug].violation)
            ):
                slug_to_group[slug] = grp

    for opp in opps:
        grp = slug_to_group.get(opp.slug)
        if not grp:
            continue

        mid = ((opp.best_bid + opp.best_ask) / 2
               if opp.best_bid > 0 else opp.mid_gamma)

        cross_edge_price = grp.per_market_edge(mid)
        cross_edge_c     = round(cross_edge_price * 100, 2)

        opp.cross_edge_cents  = cross_edge_c
        opp.cross_group_id    = grp.group_id
        opp.cross_group_size  = len(grp.members)
        opp.cross_violation   = grp.violation

        if abs(cross_edge_c) >= MIN_VIOLATION_CENTS:
            log.info(
                "CROSS-EDGE %-45s  group=%s  cross_edge=%+.1fc  "
                "prob_edge=%+.1fc  mode=%s",
                opp.question[:45], grp.group_id[:40],
                cross_edge_c, opp.edge_cents, opp.alpha_mode,
            )

        # ── Upgrade alpha_mode when cross_edge confirms prob_engine edge ────
        both_positive = (cross_edge_c >= CROSS_CONFIRM_THRESHOLD
                         and opp.edge_cents >= CROSS_CONFIRM_THRESHOLD)
        both_negative = (cross_edge_c <= -CROSS_CONFIRM_THRESHOLD
                         and opp.edge_cents <= -CROSS_CONFIRM_THRESHOLD)

        if both_positive or both_negative:
            combined_edge = abs(cross_edge_c) + abs(opp.edge_cents)
            old_mode = opp.alpha_mode

            if opp.alpha_mode == "PASSIVE_MM" and combined_edge >= 3.0:
                opp.alpha_mode = "SKEWED_MM"
            elif opp.alpha_mode == "SKEWED_MM" and combined_edge >= 15.0:
                opp.alpha_mode = "DIRECTIONAL"

            if opp.alpha_mode != old_mode:
                log.info(
                    "CROSS-UPGRADE %-40s  %s → %s  "
                    "(prob=%+.1fc  cross=%+.1fc  combined=%.1fc)",
                    opp.question[:40], old_mode, opp.alpha_mode,
                    opp.edge_cents, cross_edge_c, combined_edge,
                )
            # Boost confidence to reflect the extra signal
            opp.p_confidence = min(1.0, opp.p_confidence + 0.15)


def run_cross_market_analysis(opps: list) -> dict[str, ConstraintGroup]:
    """
    Top-level entry point called from scanner Phase 3.

    1. Build opp_map
    2. Detect groups
    3. Compute violations
    4. Apply cross-edges to opps in-place
    Returns groups dict for logging/dashboard.
    """
    opp_map = {o.slug: o for o in opps}
    groups  = detect_constraint_groups(opps)

    if not groups:
        return {}

    groups = compute_violations(groups, opp_map)

    if not groups:
        return {}

    apply_cross_market_edge(opps, groups)
    return groups
