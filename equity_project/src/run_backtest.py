import logging
import os
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import vectorbt as vbt

from equity_project.src.utils import load_config, save_dict

project_path = Path(__file__).parent.parent
logger = logging.getLogger(__name__)
TARGET_CLASSES = [-1, 0, 1]


def _cap_and_renormalize(weights, max_weight):
    if max_weight is None or max_weight <= 0:
        return weights

    capped = weights.copy()
    for _ in range(10):
        over_cap = capped > max_weight
        if not over_cap.any().any():
            break
        capped = capped.clip(upper=max_weight)
        row_sum = capped.sum(axis=1)
        capped = (capped.T / row_sum.replace(0, np.nan)).T.fillna(0)
    return capped.clip(upper=max_weight)


def _apply_rebalance_frequency(weights, rebalance_frequency):
    if not rebalance_frequency or rebalance_frequency == "D":
        return weights

    rebalanced = weights.resample(rebalance_frequency).last()
    return rebalanced.reindex(weights.index).ffill().fillna(0)


def generate_weights(preds, close=None, cfg=None):
    """Превращаем скоры ML модели в веса бумаг в портфеле

    Args:
        preds (pd.DataFrame): Датафрейм скоров ML модели

    Returns:
        pd.DataFrame: Веса бумаг в портфеле
    """
    preds_unstack = preds.unstack(level="Ticker")

    cfg = cfg or {}
    portfolio_cfg = cfg.get("portfolio", {})

    # сигнал учитывает направление и уверенность, что будет не нейтральный исход
    direction_signal = preds_unstack[1] - preds_unstack[-1]
    confidence = 1 - preds_unstack[0]
    score = direction_signal * confidence

    min_signal = portfolio_cfg.get("min_signal", 0.0)
    score = score.where(score >= min_signal, 0)

    top_n = portfolio_cfg.get("top_n")
    if top_n:
        score = score.where(score.rank(axis=1, ascending=False) <= top_n, 0)

    if close is not None and portfolio_cfg.get("risk_scale", False):
        volatility_window = portfolio_cfg.get("volatility_window", 22)
        rolling_vol = close.pct_change().rolling(volatility_window).std().shift(1)
        score = score / rolling_vol.replace(0, np.nan)

    # распределяем веса пропорционально положительному скору
    weights = score.clip(lower=0).replace([np.inf, -np.inf], np.nan)
    weights = (weights.T / weights.sum(axis=1)).T
    weights = weights.fillna(0)
    weights = _cap_and_renormalize(weights, portfolio_cfg.get("max_weight"))
    weights = _apply_rebalance_frequency(
        weights, portfolio_cfg.get("rebalance_frequency", "D")
    )
    return weights


def _predict_full_class_proba(model, X_ticker, ticker, target_classes):
    """Returns probabilities aligned to the full target class set [-1, 0, 1]."""
    proba_values = np.asarray(model.predict_proba(X_ticker))
    if proba_values.ndim == 1:
        proba_values = proba_values.reshape(-1, 1)

    model_classes = pd.Index(getattr(model, "classes_", []))
    if len(model_classes) == 0:
        model_classes = pd.Index(target_classes[: proba_values.shape[1]])

    model_classes = model_classes.astype(int)
    if proba_values.shape[1] != len(model_classes):
        logger.warning(
            "%s: predict_proba returned %d columns, but model.classes_ has %d values; "
            "using the common prefix",
            ticker,
            proba_values.shape[1],
            len(model_classes),
        )
        model_classes = model_classes[: proba_values.shape[1]]
        proba_values = proba_values[:, : len(model_classes)]

    missing_classes = sorted(set(target_classes) - set(model_classes))
    if missing_classes:
        logger.info(
            "%s: classes %s were absent during training, filling probabilities with 0",
            ticker,
            missing_classes,
        )

    proba = pd.DataFrame(
        proba_values,
        index=pd.MultiIndex.from_product(
            [X_ticker.index, [ticker]], names=["Date", "Ticker"]
        ),
        columns=model_classes,
    )
    return proba.reindex(columns=target_classes, fill_value=0.0)


def _predict_meta_labeling_proba(model_pair, X_ticker, ticker):
    meta_proba = _predict_full_class_proba(
        model_pair["meta_model"], X_ticker, ticker, [0, 1]
    )
    side_proba = _predict_full_class_proba(
        model_pair["side_model"], X_ticker, ticker, [-1, 1]
    )

    p_trade = meta_proba[1]
    proba = pd.DataFrame(index=meta_proba.index, columns=TARGET_CLASSES, dtype=float)
    proba[-1] = side_proba[-1] * p_trade
    proba[1] = side_proba[1] * p_trade
    proba[0] = 1 - p_trade
    return proba[TARGET_CLASSES]


def predict_ticker_proba(model_bundle, X_backtest):
    """Делает инференс отдельной модели каждого тикера."""
    frames = []
    models = model_bundle["models"]
    mode = model_bundle.get("mode", "multiclass")
    logger.info(
        "Running model inference for %d ticker models, mode=%s",
        len(models),
        mode,
    )

    for i, (ticker, model) in enumerate(models.items(), start=1):
        logger.info("Predicting %s (%d/%d)", ticker, i, len(models))
        if ticker not in X_backtest.index.get_level_values("Ticker"):
            logger.info("Skipping %s: no backtest features", ticker)
            continue
        X_ticker = X_backtest.xs(ticker, level="Ticker").dropna()
        if X_ticker.empty:
            logger.info("Skipping %s: empty backtest features after dropna", ticker)
            continue

        if mode == "meta_labeling":
            frames.append(_predict_meta_labeling_proba(model, X_ticker, ticker))
        else:
            frames.append(
                _predict_full_class_proba(model, X_ticker, ticker, TARGET_CLASSES)
            )

    if not frames:
        raise ValueError("No ticker models produced predictions")

    return pd.concat(frames).sort_index()


def _build_portfolio(close, price, size, cfg):
    return vbt.Portfolio.from_orders(
        close=close,
        price=price,
        size=size,
        size_type="targetpercent",
        group_by=True,
        cash_sharing=True,
        freq="1d",
        init_cash=cfg["init_cash"],
        fees=cfg["fees"],
    )


def run_combinatorial_cv(close, price, size, cfg):
    """Оценивает бэктест на всех комбинациях out-of-sample блоков дат."""
    backtest_cfg = cfg.get("backtest", {})
    n_splits = backtest_cfg.get("cv_splits", 6)
    n_test_splits = backtest_cfg.get("cv_test_splits", 2)

    dates = np.array_split(close.index.sort_values().unique(), n_splits)
    metrics = {}
    combos = list(combinations(range(len(dates)), n_test_splits))
    logger.info("Running combinatorial CV: %d folds", len(combos))

    for i, combo in enumerate(combos, start=1):
        logger.info("Combinatorial CV fold %d/%d: test blocks=%s", i, len(combos), combo)
        test_dates = pd.Index(np.concatenate([dates[i] for i in combo]))
        if test_dates.empty:
            continue

        pf = _build_portfolio(
            close.loc[test_dates],
            price.loc[test_dates],
            size.loc[test_dates],
            cfg,
        )
        metrics["+".join(map(str, combo))] = pf.stats().to_dict()

    return metrics


def calculate_benchmark_metrics(strategy_metrics, benchmark_close, cfg):
    returns = benchmark_close.pct_change().dropna()
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1

    total_return = equity.iloc[-1] - 1 if not equity.empty else 0
    years = len(returns) / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    annualized_volatility = returns.std() * np.sqrt(252)
    sharpe = (
        annualized_return / annualized_volatility
        if annualized_volatility and annualized_volatility > 0
        else np.nan
    )

    strategy_total_return = strategy_metrics.get("Total Return [%]")
    if strategy_total_return is not None:
        strategy_total_return = strategy_total_return / 100

    return {
        "benchmark_ticker": cfg.get("backtest", {}).get("benchmark_ticker", "SPY"),
        "benchmark_total_return": total_return,
        "benchmark_annualized_return": annualized_return,
        "benchmark_annualized_volatility": annualized_volatility,
        "benchmark_sharpe": sharpe,
        "benchmark_max_drawdown": drawdown.min() if not drawdown.empty else np.nan,
        "strategy_total_return": strategy_total_return,
        "excess_total_return": (
            strategy_total_return - total_return
            if strategy_total_return is not None
            else np.nan
        ),
    }


def save_pnl_plot(pf):
    """Saves PnL plot as PNG when Chrome is available, otherwise as HTML."""
    fig = pf.plot()
    png_path = project_path / "artifacts/plots/pnl.png"
    html_path = project_path / "artifacts/plots/pnl.html"

    try:
        fig.write_image(png_path.as_posix())
        logger.info("Saved PnL PNG plot to %s", png_path)
    except RuntimeError as exc:
        logger.warning(
            "Could not save PnL plot as PNG: %s. Saving interactive HTML instead.",
            exc,
        )
        fig.write_html(html_path.as_posix())
        logger.info("Saved PnL HTML plot to %s", html_path)


def run_backtest():
    """
    Запускает бэктест на бэктестовых данных
    Сохраняет:
        - Основные бэктестовые метрики в /artifacts/backtest_metrics.json
        - График PnL стратегии в /artifacts/pnl.png
    """

    os.makedirs(project_path.as_posix() + "/artifacts/plots", exist_ok=True)
    os.makedirs(project_path.as_posix() + "/artifacts/metrics", exist_ok=True)

    cfg = load_config(project_path.parent.as_posix() + "/config.yaml")
    logger.info("Loading backtest datasets")

    # считываем бэктестовые данные и ML модель
    X_backtest = pd.read_parquet(
        project_path.as_posix() + "/data/processed/X_backtest.parquet"
    )

    backtest_data = pd.read_parquet(
        project_path.as_posix() + "/data/raw/backtest_data.parquet", engine="pyarrow"
    )
    benchmark_data = pd.read_parquet(project_path / "data/raw/benchmark_data.parquet")

    # производим инференс моделей
    model_bundle = joblib.load(project_path.as_posix() + "/models/model_bundle.joblib")
    preds = predict_ticker_proba(model_bundle, X_backtest)
    logger.info("Predictions prepared: shape=%s", preds.shape)

    # избавляемся от полностью пустых колонок котировок
    close = backtest_data.Close.dropna(axis=1, how="all")
    size = generate_weights(preds, close=close, cfg=cfg)
    columns = close.columns.intersection(size.columns)
    dates = close.index.intersection(size.index)
    price = backtest_data.shift(-1).Open.loc[dates, columns]
    close = close.loc[dates, columns]
    size = size.loc[dates, columns]
    logger.info(
        "Backtest matrices aligned: dates=%d, tickers=%d",
        len(dates),
        len(columns),
    )

    # формируем портфель на основе сигналов
    logger.info("Building vectorbt portfolio")
    pf = _build_portfolio(close, price, size, cfg)

    # сохраняем PnL график
    logger.info("Saving PnL plot")
    save_pnl_plot(pf)

    # сохраняем метрики бэктеста
    backtest_metrics = pf.stats().to_dict()
    save_dict(
        backtest_metrics,
        project_path.as_posix() + "/artifacts/metrics/backtest_metrics.json",
    )
    logger.info("Saved main backtest metrics")

    if cfg.get("backtest", {}).get("benchmark_metrics", True):
        benchmark_close = benchmark_data["Close"].reindex(dates).ffill().dropna()
        benchmark_metrics = calculate_benchmark_metrics(
            backtest_metrics, benchmark_close, cfg
        )
        save_dict(
            benchmark_metrics,
            project_path.as_posix() + "/artifacts/metrics/benchmark_metrics.json",
        )
        logger.info("Saved benchmark metrics")

    cv_metrics = run_combinatorial_cv(close, price, size, cfg)
    save_dict(
        cv_metrics,
        project_path.as_posix() + "/artifacts/metrics/backtest_cscv_metrics.json",
    )
    logger.info("Saved combinatorial CV metrics")


if __name__ == "__main__":
    run_backtest()
