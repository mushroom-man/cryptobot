# -*- coding: utf-8 -*-
"""
CryptoBot - Backtest Metrics
=============================
Performance metrics and analytics for backtesting.

Usage:
    from cryptobot.research.backtest import calculate_metrics, compare_strategies
    
    metrics = calculate_metrics(equity_curve)
    comparison = compare_strategies([results1, results2], names=['Kelly', 'Fixed'])
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np


def calculate_metrics(
    equity_curve: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760,  # Hourly data
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        equity_curve: DataFrame with 'equity' column
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Number of periods per year (8760 for hourly)
    
    Returns:
        Dictionary of metrics
    """
    if 'equity' not in equity_curve.columns or len(equity_curve) < 2:
        return {}
    
    equity = equity_curve['equity']
    returns = equity.pct_change().dropna()
    
    if len(returns) == 0:
        return {}
    
    metrics = {}
    
    # Basic returns
    metrics['total_return'] = (equity.iloc[-1] / equity.iloc[0]) - 1
    
    # Annualized return
    n_periods = len(returns)
    years = n_periods / periods_per_year
    if years > 0:
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (1/years) - 1
    else:
        metrics['annualized_return'] = 0.0
    
    # Volatility
    metrics['volatility'] = returns.std() * np.sqrt(periods_per_year)
    
    # Sharpe Ratio
    excess_return = returns.mean() - (risk_free_rate / periods_per_year)
    if returns.std() > 0:
        metrics['sharpe_ratio'] = excess_return / returns.std() * np.sqrt(periods_per_year)
    else:
        metrics['sharpe_ratio'] = 0.0
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        metrics['sortino_ratio'] = excess_return / downside_returns.std() * np.sqrt(periods_per_year)
    else:
        metrics['sortino_ratio'] = metrics['sharpe_ratio']
    
    # Calmar Ratio (return / max drawdown)
    max_dd = calculate_max_drawdown(equity)
    if max_dd > 0:
        metrics['calmar_ratio'] = metrics['annualized_return'] / max_dd
    else:
        metrics['calmar_ratio'] = 0.0
    
    # Drawdown metrics
    metrics['max_drawdown'] = max_dd
    dd_series = calculate_drawdown_series(equity)
    metrics['avg_drawdown'] = dd_series.mean()
    
    # Win/Loss metrics
    winning_periods = returns[returns > 0]
    losing_periods = returns[returns < 0]
    
    metrics['win_rate'] = len(winning_periods) / len(returns) if len(returns) > 0 else 0
    metrics['avg_win'] = winning_periods.mean() if len(winning_periods) > 0 else 0
    metrics['avg_loss'] = losing_periods.mean() if len(losing_periods) > 0 else 0
    
    # Profit factor
    gross_profit = winning_periods.sum() if len(winning_periods) > 0 else 0
    gross_loss = abs(losing_periods.sum()) if len(losing_periods) > 0 else 1
    metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Skewness and Kurtosis
    metrics['skewness'] = returns.skew()
    metrics['kurtosis'] = returns.kurtosis()
    
    # Best/Worst periods
    metrics['best_period'] = returns.max()
    metrics['worst_period'] = returns.min()
    
    return metrics


def calculate_max_drawdown(equity: pd.Series) -> float:
    """Calculate maximum drawdown."""
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak
    return abs(drawdown.min())


def calculate_drawdown_series(equity: pd.Series) -> pd.Series:
    """Calculate drawdown series."""
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak
    return abs(drawdown)


def calculate_rolling_sharpe(
    returns: pd.Series,
    window: int = 720,  # 30 days of hourly data
    periods_per_year: int = 8760,
) -> pd.Series:
    """Calculate rolling Sharpe ratio."""
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    
    sharpe = (rolling_mean / rolling_std) * np.sqrt(periods_per_year)
    return sharpe


def calculate_monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """Calculate monthly returns table."""
    # Resample to monthly
    monthly = equity.resample('M').last()
    monthly_returns = monthly.pct_change()
    
    # Pivot to year x month format
    df = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values,
    })
    
    pivot = df.pivot(index='year', columns='month', values='return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Add yearly total
    pivot['Year'] = (1 + pivot.fillna(0)).prod(axis=1) - 1
    
    return pivot


def compare_to_benchmark(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
) -> Dict[str, float]:
    """
    Compare strategy to benchmark (e.g., buy-and-hold).
    
    Args:
        strategy_equity: Strategy equity curve
        benchmark_equity: Benchmark equity curve
    
    Returns:
        Comparison metrics
    """
    # Align indices
    aligned = pd.DataFrame({
        'strategy': strategy_equity,
        'benchmark': benchmark_equity,
    }).dropna()
    
    if len(aligned) < 2:
        return {}
    
    strategy_return = (aligned['strategy'].iloc[-1] / aligned['strategy'].iloc[0]) - 1
    benchmark_return = (aligned['benchmark'].iloc[-1] / aligned['benchmark'].iloc[0]) - 1
    
    # Calculate returns
    strat_returns = aligned['strategy'].pct_change().dropna()
    bench_returns = aligned['benchmark'].pct_change().dropna()
    
    # Alpha and Beta
    covariance = np.cov(strat_returns, bench_returns)[0, 1]
    benchmark_var = bench_returns.var()
    
    beta = covariance / benchmark_var if benchmark_var > 0 else 0
    
    # Annualized
    periods_per_year = 8760
    alpha = (strat_returns.mean() - beta * bench_returns.mean()) * periods_per_year
    
    # Information Ratio
    tracking_error = (strat_returns - bench_returns).std() * np.sqrt(periods_per_year)
    excess_return = strategy_return - benchmark_return
    info_ratio = excess_return / tracking_error if tracking_error > 0 else 0
    
    return {
        'strategy_return': strategy_return,
        'benchmark_return': benchmark_return,
        'excess_return': excess_return,
        'alpha': alpha,
        'beta': beta,
        'tracking_error': tracking_error,
        'information_ratio': info_ratio,
    }


def compare_strategies(
    results_list: List[Any],
    names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare multiple backtest results side by side.
    
    Args:
        results_list: List of BacktestResults objects
        names: Strategy names (optional)
    
    Returns:
        Comparison DataFrame
    """
    if names is None:
        names = [f"Strategy {i+1}" for i in range(len(results_list))]
    
    comparison = {}
    
    for name, results in zip(names, results_list):
        comparison[name] = {
            'Total Return': f"{results.total_return*100:.1f}%",
            'Annual Return': f"{results.annualized_return*100:.1f}%",
            'Sharpe': f"{results.sharpe_ratio:.2f}",
            'Sortino': f"{results.sortino_ratio:.2f}",
            'Max DD': f"{results.max_drawdown*100:.1f}%",
            'Trades': results.total_trades,
            'Win Rate': f"{results.win_rate*100:.0f}%",
            'Costs': f"${results.total_costs:,.0f}",
        }
    
    return pd.DataFrame(comparison)


def print_metrics(metrics: Dict[str, float], title: str = "Performance Metrics"):
    """Pretty print metrics dictionary."""
    print(f"\n{title}")
    print("=" * 40)
    
    format_map = {
        'total_return': '{:.2%}',
        'annualized_return': '{:.2%}',
        'volatility': '{:.2%}',
        'sharpe_ratio': '{:.2f}',
        'sortino_ratio': '{:.2f}',
        'calmar_ratio': '{:.2f}',
        'max_drawdown': '{:.2%}',
        'avg_drawdown': '{:.2%}',
        'win_rate': '{:.2%}',
        'avg_win': '{:.4f}',
        'avg_loss': '{:.4f}',
        'profit_factor': '{:.2f}',
        'skewness': '{:.2f}',
        'kurtosis': '{:.2f}',
        'best_period': '{:.4f}',
        'worst_period': '{:.4f}',
    }
    
    for key, value in metrics.items():
        fmt = format_map.get(key, '{:.4f}')
        label = key.replace('_', ' ').title()
        print(f"{label:20s}: {fmt.format(value)}")
