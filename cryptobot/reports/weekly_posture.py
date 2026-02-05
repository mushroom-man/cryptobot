# -*- coding: utf-8 -*-
"""
CryptoBot - Weekly Posture Report
==================================
Comprehensive weekly strategy posture with:
    Component A: Context Panel — this week vs 100-week history with histograms
    Component B: Posture Templates — 31 dynamic templates in 6 tiers

Usage:
    from cryptobot.reports.weekly_posture import generate_weekly_posture

    posture_text = generate_weekly_posture(db, config)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

# Timezone handling
try:
    from zoneinfo import ZoneInfo
    MELBOURNE_TZ = ZoneInfo('Australia/Melbourne')
except ImportError:
    import pytz
    MELBOURNE_TZ = pytz.timezone('Australia/Melbourne')


def melbourne_now() -> datetime:
    return datetime.now(MELBOURNE_TZ)


# =========================================================================
# DATA STRUCTURES
# =========================================================================

@dataclass
class WeekStats:
    """Aggregated stats for a single week."""
    week_start: datetime
    week_end: datetime
    n_trades: int = 0
    n_transitions: int = 0
    avg_quality: float = 0.0
    weekly_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    start_equity: float = 0.0
    end_equity: float = 0.0
    peak_equity: float = 0.0
    start_dd_pct: float = 0.0
    end_dd_pct: float = 0.0
    pct_long: float = 0.0
    pct_short: float = 0.0
    pct_flat: float = 0.0
    # Start/end direction counts
    start_n_long: int = 0
    start_n_short: int = 0
    start_n_flat: int = 0
    end_n_long: int = 0
    end_n_short: int = 0
    end_n_flat: int = 0
    # Quality trend
    first_half_quality: float = 0.0
    second_half_quality: float = 0.0
    # Pair-level
    pairs_transitioned: List[str] = field(default_factory=list)
    n_transitioned: int = 0
    # Time in unanimous direction
    pct_unanimous: float = 0.0
    pct_divergent: float = 0.0
    # Pair P&L (pair -> pnl)
    pair_pnl: Dict[str, float] = field(default_factory=dict)


@dataclass
class ContextMetric:
    """A single metric in the context panel."""
    name: str
    value: float
    format_str: str  # e.g. "{:.1f}%", "${:,.0f}", "{:.0f}"
    history: List[float] = field(default_factory=list)
    percentile: float = 0.0
    lower_is_better: bool = False

    @property
    def formatted_value(self) -> str:
        return self.format_str.format(self.value)

    @property
    def min_val(self) -> float:
        return min(self.history) if self.history else self.value

    @property
    def max_val(self) -> float:
        return max(self.history) if self.history else self.value

    @property
    def median_val(self) -> float:
        if not self.history:
            return self.value
        s = sorted(self.history)
        n = len(s)
        if n % 2 == 0:
            return (s[n // 2 - 1] + s[n // 2]) / 2
        return s[n // 2]


# =========================================================================
# DATA GATHERING
# =========================================================================

def _compute_week_stats(
    signals_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    week_start: datetime,
    week_end: datetime,
    pairs: List[str],
    initial_capital: float,
) -> WeekStats:
    """Compute aggregated stats for a single week from raw data."""

    ws = WeekStats(week_start=week_start, week_end=week_end)

    # --- Trades ---
    if trades_df is not None and len(trades_df) > 0:
        mask = (trades_df.index >= week_start) & (trades_df.index < week_end)
        week_trades = trades_df[mask]
        ws.n_trades = len(week_trades)

        # Pair P&L from trades (approximate from transaction values)
        if 'pair' in week_trades.columns and 'transaction_cost' in week_trades.columns:
            ws.pair_pnl = {}
            # Note: proper pair P&L requires mark-to-market, not just trade costs
            # This is a placeholder that will be refined with portfolio snapshot data

    # --- Signals ---
    if signals_df is not None and len(signals_df) > 0:
        mask = (signals_df.index >= week_start) & (signals_df.index < week_end)
        week_sigs = signals_df[mask]

        if len(week_sigs) > 0:
            total_records = len(week_sigs)

            # Direction percentages
            if 'signal' in week_sigs.columns:
                sig_col = week_sigs['signal']
                n_long = sig_col.isin(['LONG', 'BOOSTED_LONG']).sum()
                n_short = (sig_col == 'SHORT').sum()
                n_flat = (sig_col == 'FLAT').sum()

                ws.pct_long = n_long / total_records if total_records else 0
                ws.pct_short = n_short / total_records if total_records else 0
                ws.pct_flat = n_flat / total_records if total_records else 0

            # Average quality (confidence)
            if 'confidence' in week_sigs.columns:
                conf = week_sigs['confidence'].dropna()
                ws.avg_quality = conf.mean() if len(conf) > 0 else 0.0

                # Quality trend: first half vs second half
                mid = week_start + (week_end - week_start) / 2
                first_half = conf[conf.index < mid]
                second_half = conf[conf.index >= mid]
                ws.first_half_quality = first_half.mean() if len(first_half) > 0 else 0.0
                ws.second_half_quality = second_half.mean() if len(second_half) > 0 else 0.0

            # Start/end direction counts (first and last 24h)
            first_day_end = week_start + timedelta(hours=24)
            last_day_start = week_end - timedelta(hours=24)

            if 'signal' in week_sigs.columns:
                first_day = week_sigs[week_sigs.index < first_day_end]
                last_day = week_sigs[week_sigs.index >= last_day_start]

                if len(first_day) > 0:
                    # Use the most common signal per pair in first day
                    fd_latest = first_day.groupby('pair')['signal'].last()
                    ws.start_n_long = fd_latest.isin(['LONG', 'BOOSTED_LONG']).sum()
                    ws.start_n_short = (fd_latest == 'SHORT').sum()
                    ws.start_n_flat = (fd_latest == 'FLAT').sum()

                if len(last_day) > 0:
                    ld_latest = last_day.groupby('pair')['signal'].last()
                    ws.end_n_long = ld_latest.isin(['LONG', 'BOOSTED_LONG']).sum()
                    ws.end_n_short = (ld_latest == 'SHORT').sum()
                    ws.end_n_flat = (ld_latest == 'FLAT').sum()

            # Transitions: count state changes per pair
            if 'regime' in week_sigs.columns and 'pair' in week_sigs.columns:
                total_transitions = 0
                transitioned_pairs = set()

                for pair in pairs:
                    pair_sigs = week_sigs[week_sigs['pair'] == pair].sort_index()
                    if len(pair_sigs) > 1 and 'regime' in pair_sigs.columns:
                        states = pair_sigs['regime'].values
                        changes = sum(
                            1 for i in range(1, len(states))
                            if states[i] != states[i - 1]
                        )
                        if changes > 0:
                            total_transitions += changes
                            transitioned_pairs.add(pair)

                ws.n_transitions = total_transitions
                ws.pairs_transitioned = sorted(transitioned_pairs)
                ws.n_transitioned = len(transitioned_pairs)

            # Unanimous / divergent time
            if 'signal' in week_sigs.columns and 'pair' in week_sigs.columns:
                _compute_correlation_metrics(week_sigs, pairs, ws)

    # --- Equity ---
    if equity_df is not None and len(equity_df) > 0:
        mask = (equity_df.index >= week_start) & (equity_df.index < week_end)
        week_eq = equity_df[mask]

        if len(week_eq) > 0:
            ws.start_equity = week_eq['total_equity'].iloc[0]
            ws.end_equity = week_eq['total_equity'].iloc[-1]

            if 'peak_equity' in week_eq.columns:
                ws.peak_equity = week_eq['peak_equity'].max()

            if 'drawdown' in week_eq.columns:
                dd = week_eq['drawdown'].fillna(0)
                ws.start_dd_pct = float(dd.iloc[0]) * 100
                ws.end_dd_pct = float(dd.iloc[-1]) * 100
                ws.max_drawdown_pct = float(dd.max()) * 100

            if ws.start_equity > 0:
                ws.weekly_return_pct = (
                    (ws.end_equity - ws.start_equity) / ws.start_equity * 100
                )
        else:
            ws.start_equity = initial_capital
            ws.end_equity = initial_capital
    else:
        ws.start_equity = initial_capital
        ws.end_equity = initial_capital

    return ws


def _compute_correlation_metrics(
    week_sigs: pd.DataFrame,
    pairs: List[str],
    ws: WeekStats,
):
    """Compute unanimous/divergent time percentages."""
    # Group signals by approximate hour
    week_sigs = week_sigs.copy()
    week_sigs['hour_bucket'] = week_sigs.index.floor('h')

    hourly_groups = week_sigs.groupby('hour_bucket')
    n_hours = 0
    unanimous_hours = 0
    divergent_hours = 0

    for _, group in hourly_groups:
        if len(group) < 2:
            continue
        n_hours += 1
        latest_per_pair = group.groupby('pair')['signal'].last()
        active = latest_per_pair[latest_per_pair != 'FLAT']

        if len(active) >= 2:
            directions = set()
            for s in active:
                if s in ('LONG', 'BOOSTED_LONG'):
                    directions.add('LONG')
                elif s == 'SHORT':
                    directions.add('SHORT')

            if len(directions) == 1 and len(active) == len(latest_per_pair):
                unanimous_hours += 1
            if len(directions) == 2:
                n_long = sum(
                    1 for s in active if s in ('LONG', 'BOOSTED_LONG')
                )
                n_short = sum(1 for s in active if s == 'SHORT')
                if n_long >= 2 and n_short >= 2:
                    divergent_hours += 1

    ws.pct_unanimous = unanimous_hours / n_hours if n_hours > 0 else 0.0
    ws.pct_divergent = divergent_hours / n_hours if n_hours > 0 else 0.0


def gather_weekly_history(
    db,
    pairs: List[str],
    n_weeks: int = 100,
    initial_capital: float = 100000,
) -> List[WeekStats]:
    """
    Query DB and compute weekly stats for the last n_weeks.

    Returns list of WeekStats ordered chronologically (oldest first).
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # Calculate the start of the most recent completed week (Monday 00:00 UTC)
    days_since_monday = now.weekday()
    current_week_start = (now - timedelta(days=days_since_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    # Go back n_weeks from current week start
    history_start = current_week_start - timedelta(weeks=n_weeks)

    # Fetch all data in bulk (much faster than per-week queries)
    signals_df = db.get_signals(start=history_start.isoformat())
    trades_df = db.get_trades(start=history_start.isoformat())
    equity_df = db.get_equity_history(start=history_start.isoformat())

    # Ensure trades have datetime index
    if trades_df is not None and len(trades_df) > 0:
        if 'timestamp' in trades_df.columns:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df = trades_df.set_index('timestamp')
            trades_df.index = trades_df.index.tz_localize(None)

    # Build week boundaries
    weeks = []
    for i in range(n_weeks):
        ws = history_start + timedelta(weeks=i)
        we = ws + timedelta(weeks=1)
        if we > now:
            we = now
        weeks.append((ws, we))

    # Also include current (partial) week
    if current_week_start < now:
        weeks.append((current_week_start, now))

    # Compute stats per week
    results = []
    for ws, we in weeks:
        stats = _compute_week_stats(
            signals_df=signals_df,
            trades_df=trades_df,
            equity_df=equity_df,
            week_start=ws,
            week_end=we,
            pairs=pairs,
            initial_capital=initial_capital,
        )
        # Only include weeks with data
        has_signals = (
            signals_df is not None
            and len(signals_df) > 0
            and len(signals_df[(signals_df.index >= ws) & (signals_df.index < we)]) > 0
        )
        has_equity = (
            equity_df is not None
            and len(equity_df) > 0
            and len(equity_df[(equity_df.index >= ws) & (equity_df.index < we)]) > 0
        )
        if has_signals or has_equity:
            results.append(stats)

    return results


# =========================================================================
# PERCENTILE AND HISTOGRAM
# =========================================================================

def _percentile(value: float, history: List[float], lower_is_better: bool = False) -> float:
    """Calculate percentile rank of value within history."""
    if not history:
        return 50.0
    count_below = sum(1 for h in history if h < value)
    count_equal = sum(1 for h in history if h == value)
    pct = (count_below + 0.5 * count_equal) / len(history) * 100
    return pct


def _ascii_histogram(history: List[float], value: float, width: int = 30) -> str:
    """
    Generate ASCII histogram with marker for current value.

    Returns two lines:
        Line 1: histogram bars
        Line 2: marker showing where current value falls
    """
    if not history or len(history) < 3:
        return "  Insufficient history for histogram"

    # Create bins
    n_bins = min(width, 20)
    all_vals = history + [value]
    lo = min(all_vals)
    hi = max(all_vals)

    if lo == hi:
        return f"  All values identical: {lo}"

    # Small buffer to include max in last bin
    bin_edges = np.linspace(lo, hi + 1e-10, n_bins + 1)
    counts, _ = np.histogram(history, bins=bin_edges)

    # Normalize to block characters
    max_count = max(counts) if max(counts) > 0 else 1
    blocks = " ▁▂▃▄▅▆▇█"

    bar = ""
    for c in counts:
        idx = int(c / max_count * (len(blocks) - 1))
        bar += blocks[idx]

    # Find where current value falls in the histogram
    marker_pos = 0
    for i in range(n_bins):
        if value >= bin_edges[i] and value < bin_edges[i + 1]:
            marker_pos = i
            break
    else:
        marker_pos = n_bins - 1

    pointer = " " * marker_pos + "▲"

    return f"  {bar}\n  {pointer} You are here"


def build_context_metrics(
    current_week: WeekStats,
    history: List[WeekStats],
) -> List[ContextMetric]:
    """Build the context panel metrics with percentiles and history."""

    def _extract(field: str) -> List[float]:
        return [getattr(w, field, 0.0) for w in history]

    metrics = []

    # 1. Trades Executed
    trade_hist = _extract('n_trades')
    m = ContextMetric(
        name="Trades Executed",
        value=current_week.n_trades,
        format_str="{:.0f}",
        history=trade_hist,
    )
    m.percentile = _percentile(m.value, trade_hist)
    metrics.append(m)

    # 2. Regime Transitions
    trans_hist = _extract('n_transitions')
    m = ContextMetric(
        name="Regime Transitions",
        value=current_week.n_transitions,
        format_str="{:.0f}",
        history=trans_hist,
    )
    m.percentile = _percentile(m.value, trans_hist)
    metrics.append(m)

    # 3. Weekly Return
    ret_hist = _extract('weekly_return_pct')
    m = ContextMetric(
        name="Weekly Return",
        value=current_week.weekly_return_pct,
        format_str="{:+.2f}%",
        history=ret_hist,
    )
    m.percentile = _percentile(m.value, ret_hist)
    metrics.append(m)

    # 4. Avg Signal Quality
    qual_hist = _extract('avg_quality')
    m = ContextMetric(
        name="Avg Signal Quality",
        value=current_week.avg_quality,
        format_str="{:.2f}",
        history=qual_hist,
    )
    m.percentile = _percentile(m.value, qual_hist)
    metrics.append(m)

    # 5. Max Drawdown
    dd_hist = _extract('max_drawdown_pct')
    m = ContextMetric(
        name="Max Drawdown",
        value=current_week.max_drawdown_pct,
        format_str="{:.1f}%",
        history=dd_hist,
        lower_is_better=True,
    )
    m.percentile = _percentile(m.value, dd_hist, lower_is_better=True)
    metrics.append(m)

    return metrics


def format_context_panel(
    metrics: List[ContextMetric],
    n_weeks_available: int,
) -> str:
    """Format the context panel with histograms as text."""

    lines = [
        f"WEEKLY CONTEXT (vs {n_weeks_available} week{'s' if n_weeks_available != 1 else ''} of history)",
        "\u2501" * 50,
        "",
    ]

    for m in metrics:
        better_note = " (lower is better)" if m.lower_is_better else ""
        lines.append(
            f"{m.name}: {m.formatted_value}"
            f"       Percentile: {m.percentile:.0f}th{better_note}"
        )

        # Histogram
        if m.history and len(m.history) >= 3:
            lines.append(_ascii_histogram(m.history, m.value))
        else:
            lines.append("  Building history...")

        lines.append(
            f"  Min: {m.format_str.format(m.min_val)}  "
            f"Median: {m.format_str.format(m.median_val)}  "
            f"Max: {m.format_str.format(m.max_val)}"
        )
        lines.append("")

    return "\n".join(lines)


# =========================================================================
# TIER 1 — WEEKLY NARRATIVE (exactly one)
# =========================================================================

def _tier1_narrative(w: WeekStats) -> str:
    n_pairs = w.start_n_long + w.start_n_short + w.start_n_flat

    # #3 Consistently Cash
    if w.pct_flat > 0.70:
        return (
            f"The portfolio spent the majority of the week in cash "
            f"({w.pct_flat:.0%} of pair-hours). The market offered limited "
            f"directional opportunities — the system preserved capital by "
            f"staying sidelined."
        )

    # #6 Deployed to Cash
    start_active = w.start_n_long + w.start_n_short
    end_active = w.end_n_long + w.end_n_short
    if start_active >= 3 and w.end_n_flat >= 4:
        n_exited = start_active
        return (
            f"The week began with active positioning but the system moved "
            f"to cash as directional opportunities faded. {n_exited} "
            f"position(s) were closed as regime conditions weakened across "
            f"the portfolio."
        )

    # #7 Cash to Deployed
    start_flat = w.start_n_flat
    if start_flat >= 4 and end_active >= 3:
        end_direction = "bullish" if w.end_n_long > w.end_n_short else "bearish"
        return (
            f"The week began in cash but the system deployed capital as "
            f"new opportunities emerged. {end_active} position(s) were "
            f"established as regime conditions became directional in "
            f"{end_direction} favour."
        )

    # #4 Bullish to Bearish Shift
    if w.start_n_long >= 4 and w.end_n_short >= 4:
        return (
            f"The week began with bullish positioning but shifted to "
            f"bearish as regime conditions deteriorated. "
            f"{w.n_transitioned} asset(s) changed direction — the system "
            f"repositioned to the short side as downward momentum "
            f"established."
        )

    # #5 Bearish to Bullish Shift
    if w.start_n_short >= 4 and w.end_n_long >= 4:
        return (
            f"The week began with bearish positioning but shifted to "
            f"bullish as new uptrends emerged. {w.n_transitioned} "
            f"asset(s) rotated to the long side — the system is now "
            f"capturing upward momentum as conditions improve."
        )

    # #1 Consistently Bullish
    if w.pct_long > 0.70 and w.end_n_long >= 4:
        return (
            f"The portfolio maintained bullish positioning for most of "
            f"the week, with assets spending {w.pct_long:.0%} of the "
            f"time in confirmed uptrends. The system was actively "
            f"capturing upward momentum throughout."
        )

    # #2 Consistently Bearish
    if w.pct_short > 0.70 and w.end_n_short >= 4:
        return (
            f"The portfolio maintained bearish positioning for most of "
            f"the week, with assets spending {w.pct_short:.0%} of the "
            f"time in confirmed downtrends. The system was actively "
            f"positioned to profit from falling prices throughout."
        )

    # #8 Mixed — fallback
    return (
        f"The portfolio maintained mixed positioning throughout the "
        f"week, with {w.pct_long:.0%} of pair-hours long, "
        f"{w.pct_short:.0%} short, and {w.pct_flat:.0%} in cash. "
        f"Different assets were in different regimes — the system "
        f"traded both sides of the market."
    )


# =========================================================================
# TIER 2 — STABILITY (exactly one)
# =========================================================================

def _tier2_stability(w: WeekStats, history: List[WeekStats]) -> str:
    trans = w.n_transitions

    # Compute percentile for #13
    trans_hist = [h.n_transitions for h in history]
    trans_pct = _percentile(trans, trans_hist) if trans_hist else 50.0

    if trans <= 1:
        # #9 Very Stable
        return (
            f"Market regimes were highly stable this week with "
            f"{trans} transition(s). Once positioned, the system "
            f"held steady — conditions supported patient, "
            f"conviction-driven exposure."
        )
    elif trans <= 3:
        # #10 Stable
        return (
            f"Market regimes were stable this week with {trans} "
            f"transitions. The system made minor adjustments but "
            f"broadly maintained its positioning as conditions "
            f"evolved gradually."
        )
    elif trans <= 6:
        # #11 Moderately Active
        pairs_str = ", ".join(w.pairs_transitioned) if w.pairs_transitioned else "multiple assets"
        return (
            f"Market regimes were moderately active with {trans} "
            f"transitions across the portfolio. {pairs_str} shifted "
            f"to new states — the system adjusted positioning as "
            f"conditions evolved."
        )
    elif trans <= 10:
        # #12 Active
        return (
            f"Markets saw significant regime activity with {trans} "
            f"transitions. Multiple assets shifted states during the "
            f"week — the system repositioned actively to maintain "
            f"alignment with changing conditions."
        )
    else:
        # #13 Highly Choppy
        return (
            f"Markets were highly choppy this week with {trans} regime "
            f"transitions. This is in the {trans_pct:.0f}th percentile "
            f"of weekly activity. Frequent state changes create a "
            f"challenging environment for trend-following — the system "
            f"incurred additional trading costs from repositioning."
        )


# =========================================================================
# TIER 3 — SIGNAL QUALITY TREND (zero or one)
# =========================================================================

def _tier3_quality(w: WeekStats) -> List[str]:
    msgs = []
    diff = w.second_half_quality - w.first_half_quality

    if w.avg_quality > 0.65 and abs(diff) <= 0.05:
        # #16 Consistently Strong
        msgs.append(
            f"Signal confidence remained strong throughout the week "
            f"(avg {w.avg_quality:.0%}). The quality filter allowed "
            f"near-full position sizes — conditions supported "
            f"high-conviction trading."
        )
    elif w.avg_quality < 0.35 and abs(diff) <= 0.05:
        # #17 Consistently Weak
        msgs.append(
            f"Signal confidence remained below average throughout the "
            f"week (avg {w.avg_quality:.0%}). The quality filter "
            f"significantly reduced position sizes — the system "
            f"maintained directional views but limited risk exposure."
        )
    elif diff > 0.05:
        # #14 Improving
        msgs.append(
            f"Signal confidence improved through the week, rising "
            f"from an average of {w.first_half_quality:.0%} early in "
            f"the week to {w.second_half_quality:.0%} by the end. "
            f"Regime conditions are strengthening."
        )
    elif diff < -0.05:
        # #15 Declining
        msgs.append(
            f"Signal confidence declined through the week, falling "
            f"from {w.first_half_quality:.0%} to "
            f"{w.second_half_quality:.0%}. The system has reduced "
            f"position sizes as regime conditions weaken."
        )

    return msgs


# =========================================================================
# TIER 4 — TRADING ACTIVITY (exactly one)
# =========================================================================

def _tier4_activity(w: WeekStats, history: List[WeekStats]) -> str:
    trade_hist = [h.n_trades for h in history]
    pct = _percentile(w.n_trades, trade_hist) if trade_hist else 50.0

    if pct <= 20:
        # #18 Very Quiet
        return (
            f"A quiet week with only {w.n_trades} trade(s) — in the "
            f"bottom {pct:.0f}th percentile of weekly activity. The "
            f"system held existing positions with minimal adjustment "
            f"needed."
        )
    elif pct >= 80:
        # #20 Heavy Trading
        return (
            f"A busy week with {w.n_trades} trades — in the "
            f"{pct:.0f}th percentile of weekly activity. Regime "
            f"changes across multiple assets required significant "
            f"repositioning."
        )
    else:
        # #19 Normal
        return (
            f"Trading activity was typical this week with "
            f"{w.n_trades} trades — in the {pct:.0f}th percentile "
            f"of weekly activity."
        )


# =========================================================================
# TIER 5 — PERFORMANCE & RISK (zero or more)
# =========================================================================

def _tier5_performance(
    w: WeekStats,
    history: List[WeekStats],
    max_dd_pct: float,
    initial_capital: float,
) -> List[str]:
    msgs = []
    ret_hist = [h.weekly_return_pct for h in history]
    ret_pct = _percentile(w.weekly_return_pct, ret_hist) if ret_hist else 50.0

    # #21 Strong Positive
    if ret_pct >= 80:
        msgs.append(
            f"A strong week with a return of {w.weekly_return_pct:+.2f}% "
            f"— in the {ret_pct:.0f}th percentile of weekly performance. "
            f"The system's positioning aligned well with market movements."
        )
    # #22 Strong Negative
    elif ret_pct <= 20:
        msgs.append(
            f"A difficult week with a return of "
            f"{w.weekly_return_pct:+.2f}% — in the {ret_pct:.0f}th "
            f"percentile of weekly performance. Risk controls limited "
            f"the impact — the system's maximum drawdown design of "
            f"{max_dd_pct:.0f}% was not breached."
        )
    # #23 Flat
    elif abs(w.weekly_return_pct) <= 0.5:
        msgs.append(
            f"A flat week with a return of {w.weekly_return_pct:+.2f}%. "
            f"Limited market movement and/or light positioning resulted "
            f"in minimal portfolio change."
        )

    # #24 New All-Time High
    prev_peaks = [h.peak_equity for h in history if h.peak_equity > 0]
    if prev_peaks and w.peak_equity > max(prev_peaks):
        inception_return = (
            (w.end_equity - initial_capital) / initial_capital * 100
        )
        msgs.append(
            f"The portfolio reached a new all-time high of "
            f"${w.peak_equity:,.0f} this week. Cumulative return "
            f"since inception: {inception_return:+.1f}%."
        )

    # #25 Drawdown Deepened
    if w.end_dd_pct - w.start_dd_pct > 2.0:
        msgs.append(
            f"Drawdown increased from {w.start_dd_pct:.1f}% to "
            f"{w.end_dd_pct:.1f}% during the week. The system's risk "
            f"controls are active and position sizes have been reduced "
            f"proportionally."
        )

    # #26 Drawdown Recovered
    if w.start_dd_pct - w.end_dd_pct > 2.0:
        msgs.append(
            f"Drawdown improved from {w.start_dd_pct:.1f}% to "
            f"{w.end_dd_pct:.1f}% during the week. The portfolio is "
            f"recovering and the system is gradually restoring position "
            f"sizes."
        )

    # #27 Near Drawdown Limit
    if w.end_dd_pct > max_dd_pct * 0.8:
        msgs.append(
            f"\u26a0\ufe0f The portfolio ended the week at "
            f"{w.end_dd_pct:.1f}% drawdown, approaching the "
            f"{max_dd_pct:.0f}% limit. Position sizes are significantly "
            f"reduced. Capital preservation is the priority until equity "
            f"recovers."
        )

    return msgs


# =========================================================================
# TIER 6 — PAIR-LEVEL STANDOUTS (zero or more)
# =========================================================================

def _tier6_pair_standouts(w: WeekStats) -> List[str]:
    msgs = []

    # #30 Unanimous Direction
    if w.pct_unanimous > 0.80:
        msgs.append(
            f"All six assets moved in lockstep this week, spending "
            f"{w.pct_unanimous:.0%} of the time in the same direction. "
            f"Cryptocurrency correlations were elevated — the system's "
            f"risk parity allocation prevented concentration risk despite "
            f"the unified movement."
        )

    # #31 Divergent Behaviour
    if w.pct_divergent > 0.40:
        msgs.append(
            f"Assets showed significant divergence this week, with "
            f"long and short positions running simultaneously for "
            f"{w.pct_divergent:.0%} of the time. The portfolio benefited "
            f"from genuine diversification across different regime "
            f"conditions."
        )

    # Note: #28 Star Performer and #29 Biggest Detractor require
    # mark-to-market pair P&L which needs portfolio snapshot data.
    # These will be enabled when pair-level P&L tracking is implemented.
    # For now the pair P&L fields are placeholder.

    return msgs


# =========================================================================
# PUBLIC API
# =========================================================================

def get_weekly_posture_messages(
    current_week: WeekStats,
    history: List[WeekStats],
    max_dd_pct: float = 20.0,
    initial_capital: float = 100000,
) -> List[str]:
    """
    Generate all applicable weekly posture messages.

    Returns list of strings in tier order.
    """
    messages = []

    # Tier 1: exactly one narrative
    messages.append(_tier1_narrative(current_week))

    # Tier 2: exactly one stability
    messages.append(_tier2_stability(current_week, history))

    # Tier 3: zero or one quality
    messages.extend(_tier3_quality(current_week))

    # Tier 4: exactly one activity
    messages.append(_tier4_activity(current_week, history))

    # Tier 5: zero or more performance
    messages.extend(
        _tier5_performance(current_week, history, max_dd_pct, initial_capital)
    )

    # Tier 6: zero or more pair standouts
    messages.extend(_tier6_pair_standouts(current_week))

    return messages


def generate_weekly_posture(
    db,
    config: dict,
) -> str:
    """
    Full weekly posture report: context panel + posture messages.

    Args:
        db: Database instance
        config: Configuration dictionary

    Returns:
        Formatted text block ready for inclusion in weekly report.
    """
    pairs = config.get('pairs', [])
    initial_capital = config.get('position', {}).get('initial_capital', 100000)
    max_dd_pct = config.get('risk', {}).get('max_drawdown', 0.20) * 100

    # Gather history
    all_weeks = gather_weekly_history(
        db=db,
        pairs=pairs,
        n_weeks=100,
        initial_capital=initial_capital,
    )

    if not all_weeks:
        return (
            "WEEKLY POSTURE\n"
            + "=" * 50 + "\n"
            + "Insufficient data for weekly posture analysis. "
            + "The system needs at least one full week of signal history.\n"
        )

    current_week = all_weeks[-1]
    history = all_weeks[:-1]  # everything before current week

    # Build context panel
    context_metrics = build_context_metrics(current_week, history)
    context_text = format_context_panel(
        context_metrics,
        n_weeks_available=len(history),
    )

    # Build posture messages
    posture_msgs = get_weekly_posture_messages(
        current_week=current_week,
        history=history,
        max_dd_pct=max_dd_pct,
        initial_capital=initial_capital,
    )

    # Combine
    lines = [
        context_text,
        "",
        "STRATEGY POSTURE",
        "\u2501" * 50,
        "",
    ]

    for msg in posture_msgs:
        lines.append(msg)
        lines.append("")

    return "\n".join(lines)