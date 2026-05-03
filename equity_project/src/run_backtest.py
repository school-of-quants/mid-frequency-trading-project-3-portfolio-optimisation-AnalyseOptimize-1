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


def generate_weights(preds):
    """Превращаем скоры ML модели в веса бумаг в портфеле

    Args:
        preds (pd.DataFrame): Датафрейм скоров ML модели

    Returns:
        pd.DataFrame: Веса бумаг в портфеле
    """
    preds_unstack = preds.unstack(level="Ticker")

    # сигнал - преимущество вероятности роста над вероятностью падения
    long_prob_minus_short_prob = preds_unstack[1] - preds_unstack[-1]

    # распределяем веса пропорционально положительному ML-сигналу
    weights = long_prob_minus_short_prob.clip(lower=0)
    weights = (weights.T / weights.sum(axis=1)).T
    weights = weights.fillna(0)
    return weights


def predict_ticker_proba(model_bundle, X_backtest):
    """Делает инференс отдельной модели каждого тикера."""
    frames = []
    models = model_bundle["models"]
    logger.info("Running model inference for %d ticker models", len(models))

    for i, (ticker, model) in enumerate(models.items(), start=1):
        logger.info("Predicting %s (%d/%d)", ticker, i, len(models))
        if ticker not in X_backtest.index.get_level_values("Ticker"):
            logger.info("Skipping %s: no backtest features", ticker)
            continue
        X_ticker = X_backtest.xs(ticker, level="Ticker").dropna()
        if X_ticker.empty:
            logger.info("Skipping %s: empty backtest features after dropna", ticker)
            continue

        proba = pd.DataFrame(
            model.predict_proba(X_ticker),
            index=pd.MultiIndex.from_product(
                [X_ticker.index, [ticker]], names=["Date", "Ticker"]
            ),
            columns=model.classes_.astype(int),
        )
        for klass in (-1, 0, 1):
            if klass not in proba:
                proba[klass] = 0.0
        frames.append(proba[[-1, 0, 1]])

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

    # производим инференс моделей
    model_bundle = joblib.load(project_path.as_posix() + "/models/model_bundle.joblib")
    preds = predict_ticker_proba(model_bundle, X_backtest)
    logger.info("Predictions prepared: shape=%s", preds.shape)

    # избавляемся от полностью пустых колонок котировок
    close = backtest_data.Close.dropna(axis=1, how="all")
    size = generate_weights(preds)
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
    pf.plot().write_image(project_path.as_posix() + "/artifacts/plots/pnl.png")

    # сохраняем метрики бэктеста
    backtest_metrics = pf.stats().to_dict()
    save_dict(
        backtest_metrics,
        project_path.as_posix() + "/artifacts/metrics/backtest_metrics.json",
    )
    logger.info("Saved main backtest metrics")

    cv_metrics = run_combinatorial_cv(close, price, size, cfg)
    save_dict(
        cv_metrics,
        project_path.as_posix() + "/artifacts/metrics/backtest_cscv_metrics.json",
    )
    logger.info("Saved combinatorial CV metrics")


if __name__ == "__main__":
    run_backtest()
