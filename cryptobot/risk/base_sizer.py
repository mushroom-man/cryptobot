# -*- coding: utf-8 -*-
"""Base sizer classes."""

from abc import ABC


class BaseSizer(ABC):
    """Base class for single-asset sizers."""
    pass


class BasePortfolioSizer(ABC):
    """Base class for portfolio-level sizers."""
    pass


class PositionSizerProtocol:
    """Protocol for position sizing."""
    pass
