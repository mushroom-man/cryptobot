# -*- coding: utf-8 -*-
"""
CryptoBot - Strategy Posture Templates
=======================================
Dynamic posture messages for daily reports and trade alerts.

Generates human-readable strategy commentary from run data.
Templates are organised into three tiers that stack:
    Tier 1 - Direction (exactly one per report)
    Tier 2 - Modifiers (zero or more)
    Tier 3 - Events (zero or more)

Usage:
    from cryptobot.reports.posture import get_posture_messages, build_posture_data

    # Build posture data from runner state
    data = build_posture_data(
        signals=signals_generated,
        trades_executed=trades_executed,
        vol_scalar=0.68,
        dd_scalar=1.0,
        equity=equity_dict,
        initial_capital=100000,
        previous_position_count=0,
    )

    # Get list of message strings
    messages = get_posture_messages(data)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# =========================================================================
# POSTURE DATA
# =========================================================================

@dataclass
class PostureData:
    """All data needed to evaluate posture templates."""

    # Direction counts
    n_long: int = 0
    n_short: int = 0
    n_flat: int = 0
    n_total: int = 6

    # Quality
    avg_quality: float = 0.0

    # Risk scalars
    vol_scalar: float = 1.0
    dd_scalar: float = 1.0

    # Drawdown
    drawdown_pct: float = 0.0
    max_dd_pct: float = 20.0

    # Equity
    total_equity: float = 0.0
    peak_equity: float = 0.0
    initial_capital: float = 100000.0
    has_equity_history: bool = True

    # Trades this run
    trades_executed: int = 0

    # Transitions this run
    n_transitions: int = 0
    transition_pairs: List[str] = field(default_factory=list)

    # Position context (from DB at start of run)
    had_positions_before: bool = False
    has_positions_now: bool = False

    @property
    def n_active(self) -> int:
        return self.n_long + self.n_short


def build_posture_data(
    signals: List[Dict[str, Any]],
    trades_executed: List[Dict[str, Any]],
    vol_scalar: float,
    dd_scalar: float,
    equity: Dict[str, Any],
    initial_capital: float,
    previous_position_count: int = 0,
    max_drawdown_pct: float = 20.0,
    transition_pairs: Optional[List[str]] = None,
) -> PostureData:
    """
    Build PostureData from runner state.

    Args:
        signals: List of signal dicts from TradingRunner.signals_generated
        trades_executed: List of trade dicts from TradingRunner.trades_executed
        vol_scalar: Current volatility scalar from sizer
        dd_scalar: Current drawdown scalar from sizer
        equity: Equity dict from database
        initial_capital: Starting capital
        previous_position_count: Number of non-zero positions at start of run
        max_drawdown_pct: Maximum allowed drawdown percentage
        transition_pairs: List of pair names that transitioned state this run
    """
    n_long = 0
    n_short = 0
    n_flat = 0
    quality_values = []

    for sig in signals:
        sig_type = sig.get('signal', 'FLAT')
        if sig_type in ('LONG', 'BOOSTED_LONG'):
            n_long += 1
        elif sig_type == 'SHORT':
            n_short += 1
        else:
            n_flat += 1

        qs = sig.get('quality_scalar')
        if qs is not None:
            quality_values.append(qs)

    avg_quality = sum(quality_values) / len(quality_values) if quality_values else 0.0

    total_equity = equity.get('total_equity', initial_capital)
    peak_equity = equity.get('peak_equity', total_equity)
    drawdown = equity.get('drawdown', 0.0) or 0.0
    drawdown_pct = drawdown * 100  # stored as decimal in DB

    tp = transition_pairs or []

    has_positions_now = (n_long + n_short) > 0

    return PostureData(
        n_long=n_long,
        n_short=n_short,
        n_flat=n_flat,
        n_total=len(signals),
        avg_quality=avg_quality,
        vol_scalar=vol_scalar,
        dd_scalar=dd_scalar,
        drawdown_pct=drawdown_pct,
        max_dd_pct=max_drawdown_pct,
        total_equity=total_equity,
        peak_equity=peak_equity,
        initial_capital=initial_capital,
        has_equity_history=equity.get('total_equity') is not None,
        trades_executed=len(trades_executed),
        n_transitions=len(tp),
        transition_pairs=tp,
        had_positions_before=previous_position_count > 0,
        has_positions_now=has_positions_now,
    )


# =========================================================================
# TIER 1 — DIRECTION (exactly one)
# =========================================================================

def _tier1_direction(d: PostureData) -> str:
    """Return exactly one direction template."""

    # #7 All flat
    if d.n_flat == d.n_total:
        return (
            "Fully in cash. No asset is showing a reliable directional "
            "edge — the system is preserving capital and waiting for "
            "clearer market conditions."
        )

    # #6 Majority flat (4+)
    if d.n_flat >= 4:
        return (
            f"Mostly in cash with selective positioning in {d.n_active} "
            f"asset(s). The market is offering limited opportunities — "
            f"the system is being patient and only trading where "
            f"conditions are clear."
        )

    # #1 All long
    if d.n_long == d.n_total:
        return (
            "Full bullish positioning across all six assets. Every pair "
            "is showing confirmed upward momentum — the system is fully "
            "deployed to capture rising prices."
        )

    # #2 All short
    if d.n_short == d.n_total:
        return (
            "Full bearish positioning across all six assets. All pairs "
            "are in confirmed downtrends — the system is actively "
            "positioned to profit from falling prices."
        )

    # #3 Majority long (4-5)
    if d.n_long >= 4:
        return (
            f"Bullish positioning in {d.n_long} of {d.n_total} assets. "
            f"The majority of the portfolio is capturing upward momentum, "
            f"with {d.n_short} short and {d.n_flat} in cash where "
            f"conditions are less favourable."
        )

    # #4 Majority short (4-5)
    if d.n_short >= 4:
        return (
            f"Bearish positioning in {d.n_short} of {d.n_total} assets. "
            f"The majority of the portfolio is positioned for falling "
            f"prices, with {d.n_long} long and {d.n_flat} in cash where "
            f"conditions differ."
        )

    # #5 Mixed
    return (
        f"Mixed positioning: {d.n_long} long, {d.n_short} short, "
        f"{d.n_flat} in cash. Different assets are in different market "
        f"regimes — the system is trading both sides where it sees "
        f"an edge."
    )


# =========================================================================
# TIER 2 — MODIFIERS (zero or more)
# =========================================================================

def _tier2_modifiers(d: PostureData) -> List[str]:
    """Return zero or more modifier templates."""
    msgs = []

    # Quality (#8, #9, #10) — mutually exclusive
    if d.n_active > 0:
        if d.avg_quality > 0.7:
            msgs.append(
                f"Signal confidence is strong across the portfolio "
                f"(avg {d.avg_quality:.0%}). The quality filter is "
                f"allowing near-full position sizes — current regime "
                f"conditions are well-established."
            )
        elif d.avg_quality >= 0.4:
            msgs.append(
                f"Signal confidence is moderate (avg {d.avg_quality:.0%}). "
                f"The quality filter is reducing position sizes as a "
                f"precaution — conditions are directional but not yet "
                f"at peak strength."
            )
        else:
            msgs.append(
                f"Signal confidence is below average across the portfolio "
                f"(avg {d.avg_quality:.0%}). Position sizes have been "
                f"significantly reduced — the system is maintaining "
                f"directional exposure but limiting risk until conditions "
                f"strengthen."
            )

    # Volatility compression (#11)
    if d.vol_scalar < 0.5:
        msgs.append(
            f"Market volatility is elevated. The system has automatically "
            f"compressed position sizes (vol scalar: {d.vol_scalar:.2f}) "
            f"to maintain consistent risk per trade despite turbulent "
            f"conditions."
        )

    # Drawdown (#12 and #13) — #13 takes priority if both match
    if d.drawdown_pct > d.max_dd_pct * 0.8:
        msgs.append(
            f"\u26a0\ufe0f Approaching risk limits. The portfolio is "
            f"{d.drawdown_pct:.1f}% below its peak "
            f"(limit: {d.max_dd_pct:.0f}%). Position sizes are "
            f"significantly reduced — the system is prioritising "
            f"capital preservation."
        )
    elif d.drawdown_pct > 10.0:
        msgs.append(
            f"Drawdown protection is active. The portfolio is "
            f"{d.drawdown_pct:.1f}% below its peak — the system has "
            f"reduced position sizes to limit further losses while "
            f"maintaining directional exposure."
        )

    # High correlation warning (#14)
    if (
        d.n_active > 0
        and (d.n_long == d.n_active or d.n_short == d.n_active)
        and d.avg_quality < 0.5
    ):
        msgs.append(
            "All active positions are moving in the same direction with "
            "below-average conviction. Cryptocurrency correlations are "
            "elevated — the system is limiting total exposure to manage "
            "concentration risk."
        )

    return msgs


# =========================================================================
# TIER 3 — EVENTS (zero or more)
# =========================================================================

def _tier3_events(d: PostureData) -> List[str]:
    """Return zero or more event templates."""
    msgs = []

    # First day (#22) — highest priority event
    if not d.has_equity_history:
        msgs.append(
            f"Paper trading has commenced with "
            f"${d.initial_capital:,.0f} starting capital. The system is "
            f"generating live signals and simulating trades to validate "
            f"performance. No real capital is at risk."
        )
        return msgs  # skip other events on first day

    # No trades — holding (#15) or all cash (#16)
    if d.trades_executed == 0:
        if d.has_positions_now:
            parts = []
            if d.n_long > 0:
                parts.append(f"{d.n_long} long")
            if d.n_short > 0:
                parts.append(f"{d.n_short} short")
            position_desc = " and ".join(parts) + " position(s)"
            msgs.append(
                f"No changes today. The system is holding {position_desc} "
                f"— current regime conditions remain stable and no "
                f"rebalancing is required."
            )
        else:
            msgs.append(
                "No changes today. The portfolio remains fully in cash "
                "— the system continues to monitor all six assets but "
                "none have met the threshold for entry."
            )

    # Transitions (#17, #18)
    if d.n_transitions == 1:
        pair_name = d.transition_pairs[0]
        msgs.append(
            f"{pair_name} has shifted to a new market regime. The system "
            f"has adjusted its positioning accordingly while all other "
            f"holdings remain unchanged."
        )
    elif d.n_transitions > 1:
        pairs_str = ", ".join(d.transition_pairs)
        msgs.append(
            f"{d.n_transitions} assets have shifted to new market regimes "
            f"({pairs_str}). The market is undergoing a broader structural "
            f"change — the system has repositioned accordingly."
        )

    # Re-deploying after cash (#19)
    if not d.had_positions_before and d.has_positions_now and d.trades_executed > 0:
        msgs.append(
            f"The system is re-entering the market after a period in "
            f"cash. New positions have been established in {d.n_active} "
            f"asset(s) where regime conditions now favour directional "
            f"trading."
        )

    # Exiting to cash (#20)
    if d.had_positions_before and not d.has_positions_now and d.trades_executed > 0:
        msgs.append(
            "The system has closed all positions and moved to cash. "
            "Regime conditions have deteriorated across the portfolio "
            "— capital is being preserved until directional "
            "opportunities return."
        )

    # New equity high (#21)
    if d.total_equity > d.peak_equity and d.total_equity > d.initial_capital:
        msgs.append(
            f"New portfolio high of ${d.total_equity:,.0f}. The system "
            f"is performing as designed — capturing momentum in "
            f"favourable regimes while managing risk during transitions."
        )

    return msgs


# =========================================================================
# PUBLIC API
# =========================================================================

def get_posture_messages(data: PostureData) -> List[str]:
    """
    Generate all applicable posture messages for a report.

    Returns a list of strings in tier order:
        [0]     = Tier 1 direction (always exactly one)
        [1..n]  = Tier 2 modifiers (zero or more)
        [n+1..] = Tier 3 events (zero or more)
    """
    messages = []

    # Tier 1: exactly one
    messages.append(_tier1_direction(data))

    # Tier 2: zero or more
    messages.extend(_tier2_modifiers(data))

    # Tier 3: zero or more
    messages.extend(_tier3_events(data))

    return messages


def format_posture_block(messages: List[str]) -> str:
    """
    Format posture messages into a text block for reports.

    Each message is separated by a blank line for readability.
    """
    return "\n\n".join(messages)