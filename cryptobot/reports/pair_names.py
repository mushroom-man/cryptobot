# -*- coding: utf-8 -*-
"""
CryptoBot - Pair Name Mapping
=============================
Maps trading pair codes to friendly names for reports.

Usage:
    from cryptobot.reports.pair_names import get_friendly_name, get_short_name
    
    get_friendly_name('XMRUSD')  # -> 'Monero (XMR)'
    get_short_name('XMRUSD')     # -> 'XMR'
"""

# Full friendly names for reports
PAIR_NAMES = {
    # Current trading pairs
    'XLMUSD': 'Stellar (XLM)',
    'ZECUSD': 'Zcash (ZEC)',
    'ETCUSD': 'Ethereum Classic (ETC)',
    'ETHUSD': 'Ethereum (ETH)',
    'XMRUSD': 'Monero (XMR)',
    'ADAUSD': 'Cardano (ADA)',
    
    # Common pairs for future expansion
    'XBTUSD': 'Bitcoin (BTC)',
    'BTCUSD': 'Bitcoin (BTC)',
    'SOLUSD': 'Solana (SOL)',
    'XRPUSD': 'Ripple (XRP)',
    'DOGEUSD': 'Dogecoin (DOGE)',
    'LINKUSD': 'Chainlink (LINK)',
    'MATICUSD': 'Polygon (MATIC)',
    'AVAXUSD': 'Avalanche (AVAX)',
    'DOTUSD': 'Polkadot (DOT)',
    'LTCUSD': 'Litecoin (LTC)',
    'UNIUSD': 'Uniswap (UNI)',
    'ATOMUSD': 'Cosmos (ATOM)',
    'NEARUSD': 'NEAR Protocol (NEAR)',
    'ALGOUSD': 'Algorand (ALGO)',
    'FTMUSD': 'Fantom (FTM)',
    'SANDUSD': 'The Sandbox (SAND)',
    'MANAUSD': 'Decentraland (MANA)',
    'AABORUSD': 'Aave (AAVE)',
}

# Short names (just the symbol)
PAIR_SHORT_NAMES = {
    pair: name.split('(')[1].rstrip(')') if '(' in name else pair.replace('USD', '')
    for pair, name in PAIR_NAMES.items()
}


def get_friendly_name(pair: str) -> str:
    """
    Get friendly name for a trading pair.
    
    Args:
        pair: Trading pair code (e.g., 'XMRUSD')
    
    Returns:
        Friendly name (e.g., 'Monero (XMR)') or original if not found
    """
    return PAIR_NAMES.get(pair, pair)


def get_short_name(pair: str) -> str:
    """
    Get short symbol name for a trading pair.
    
    Args:
        pair: Trading pair code (e.g., 'XMRUSD')
    
    Returns:
        Short name (e.g., 'XMR') or original if not found
    """
    return PAIR_SHORT_NAMES.get(pair, pair.replace('USD', ''))


def format_holding(pair: str, value: float, percentage: float) -> str:
    """
    Format a holding line for reports.
    
    Args:
        pair: Trading pair code
        value: Dollar value
        percentage: Percentage of portfolio
    
    Returns:
        Formatted string like 'Ethereum (ETH):     $8,892 (35%)'
    """
    name = get_friendly_name(pair)
    return f"   {name:24s} ${value:>10,.0f} ({percentage:>4.0f}%)"