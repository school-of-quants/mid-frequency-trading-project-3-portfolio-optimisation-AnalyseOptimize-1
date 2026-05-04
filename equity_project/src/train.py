import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from equity_project.src.utils import load_config, save_dict

project_path = Path(__file__).parent.parent
logger = logging.getLogger(__name__)


class PurgedKFold:
    """Time-series KFold with an embargo around every validation fold."""

    def __init__(self, n_splits=5, embargo_days=10):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.embargo = pd.Timedelta(days=embargo_days)

    def split(self, X):
        dates = pd.Index(X.index).sort_values().unique()
        fold_dates = np.array_split(dates, self.n_splits)

        for val_dates in fold_dates:
            if len(val_dates) == 0:
                continue

            val_start = pd.Timestamp(val_dates[0])
            val_end = pd.Timestamp(val_dates[-1])
            sample_dates = pd.Series(pd.Index(X.index), index=np.arange(len(X)))

            val_mask = sample_dates.isin(val_dates)
            purge_mask = sample_dates.between(
                val_start - self.embargo,
                val_end + self.embargo,
                inclusive="both",
            )
            train_idx = sample_dates.index[~purge_mask].to_numpy()
            val_idx = sample_dates.index[val_mask].to_numpy()

            if len(train_idx) and len(val_idx):
                yield train_idx, val_idx


def instantiate_model(cfg=None, use_best_model=True, iterations=None):
    """Задаем параметры модели для обучения

    Returns:
        CatBoostClassifier: Инициированная ML модель
    """
    cfg = cfg or {}
    model = CatBoostClassifier(
        auto_class_weights="Balanced",
        loss_function="MultiClass",
        use_best_model=use_best_model,
        eval_metric="MultiClass",
        early_stopping_rounds=cfg.get("early_stopping_rounds", 50),
        iterations=iterations or cfg.get("iterations", 1000),
        learning_rate=cfg.get("learning_rate", 0.05),
        depth=cfg.get("depth", 6),
        random_seed=cfg.get("random_seed", 42),
        verbose=cfg.get("verbose", False),
    )
    return model


def _prepare_ticker_data(X, y, ticker):
    X_ticker = X.xs(ticker, level="Ticker").copy()
    y_ticker = y.xs(ticker, level="Ticker")["target"].copy()
    data = X_ticker.join(y_ticker, how="inner").dropna()
    return data.drop(columns="target"), data["target"].astype(int)


def _cross_validate_ticker(X_ticker, y_ticker, model_cfg):
    splitter = PurgedKFold(
        n_splits=model_cfg.get("cv_splits", 5),
        embargo_days=model_cfg.get("embargo_days", 10),
    )
    fold_metrics = []
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_ticker), start=1):
        y_fold_train = y_ticker.iloc[train_idx]
        y_fold_val = y_ticker.iloc[val_idx]
        if y_fold_train.nunique() < 2:
            logger.info("Skipping CV fold %d: only one target class in train", fold)
            continue

        missing_val_classes = set(y_fold_val.unique()) - set(y_fold_train.unique())
        if missing_val_classes:
            logger.info(
                "Skipping CV fold %d: validation has classes absent in train: %s",
                fold,
                sorted(missing_val_classes),
            )
            continue

        model = instantiate_model(model_cfg, use_best_model=True)
        model.fit(
            X=X_ticker.iloc[train_idx],
            y=y_fold_train,
            eval_set=(X_ticker.iloc[val_idx], y_fold_val),
        )
        y_pred = model.predict(X_ticker.iloc[val_idx]).reshape(-1).astype(int)
        fold_metrics.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(y_fold_val, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_fold_val, y_pred),
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "best_iteration": model.get_best_iteration(),
            }
        )
        if model.get_best_iteration() is not None:
            best_iterations.append(model.get_best_iteration() + 1)

    return fold_metrics, best_iterations


def _fit_model_with_cv(X_ticker, y_ticker, model_cfg, ticker, model_name):
    if y_ticker.nunique() < 2:
        logger.info("Skipping %s %s: only one target class", ticker, model_name)
        return None, [], []

    fold_metrics, best_iterations = _cross_validate_ticker(
        X_ticker, y_ticker, model_cfg
    )
    iterations = (
        int(np.median(best_iterations))
        if best_iterations
        else model_cfg.get("iterations", 1000)
    )

    model = instantiate_model(
        model_cfg, use_best_model=False, iterations=max(iterations, 1)
    )
    model.fit(X=X_ticker, y=y_ticker)
    return model, fold_metrics, best_iterations


def _train_multiclass_ticker(X_ticker, y_ticker, model_cfg, ticker):
    model, fold_metrics, best_iterations = _fit_model_with_cv(
        X_ticker, y_ticker, model_cfg, ticker, "multiclass model"
    )
    if model is None:
        return None, fold_metrics

    logger.info(
        "Finished %s multiclass model: samples=%d, classes=%d, final_iterations=%d",
        ticker,
        len(X_ticker),
        y_ticker.nunique(),
        max(
            int(np.median(best_iterations))
            if best_iterations
            else model_cfg.get("iterations", 1000),
            1,
        ),
    )
    return model, fold_metrics


def _train_meta_labeling_ticker(X_ticker, y_ticker, model_cfg, ticker):
    meta_y = (y_ticker != 0).astype(int)
    meta_model, meta_metrics, _ = _fit_model_with_cv(
        X_ticker, meta_y, model_cfg, ticker, "meta model"
    )
    if meta_model is None:
        return None, {"meta": meta_metrics, "side": []}

    side_mask = y_ticker != 0
    side_X = X_ticker.loc[side_mask]
    side_y = y_ticker.loc[side_mask]
    side_model, side_metrics, _ = _fit_model_with_cv(
        side_X, side_y, model_cfg, ticker, "side model"
    )
    if side_model is None:
        return None, {"meta": meta_metrics, "side": side_metrics}

    logger.info(
        "Finished %s meta-labeling models: meta_samples=%d, side_samples=%d",
        ticker,
        len(X_ticker),
        len(side_X),
    )
    return {"meta_model": meta_model, "side_model": side_model}, {
        "meta": meta_metrics,
        "side": side_metrics,
    }


def train():
    """
    Запускаем обучение стратегии и сохраняем отдельную модель для каждого тикера.
    """
    os.makedirs(project_path.as_posix() + "/models", exist_ok=True)
    os.makedirs(project_path.as_posix() + "/artifacts/metrics", exist_ok=True)
    cfg = load_config(project_path.parent.as_posix() + "/config.yaml")
    model_cfg = cfg.get("model", {})
    meta_labeling_enabled = model_cfg.get("meta_labeling", {}).get("enabled", False)
    strategy_mode = "meta_labeling" if meta_labeling_enabled else "multiclass"

    # считываем обучающие данные
    X = pd.read_parquet(project_path.as_posix() + "/data/processed/X_train.parquet")
    y = pd.read_parquet(project_path.as_posix() + "/data/processed/y_train.parquet")
    logger.info("Loaded training data: X=%s, y=%s", X.shape, y.shape)

    models = {}
    cv_metrics = {}
    tickers = X.index.get_level_values("Ticker").unique()
    logger.info(
        "Training per-ticker CatBoost models for %d tickers, mode=%s",
        len(tickers),
        strategy_mode,
    )

    for i, ticker in enumerate(tickers, start=1):
        logger.info("Training ticker %s (%d/%d)", ticker, i, len(tickers))
        X_ticker, y_ticker = _prepare_ticker_data(X, y, ticker)

        if meta_labeling_enabled:
            model, metrics = _train_meta_labeling_ticker(
                X_ticker, y_ticker, model_cfg, ticker
            )
        else:
            model, metrics = _train_multiclass_ticker(
                X_ticker, y_ticker, model_cfg, ticker
            )
        if model is None:
            continue

        models[ticker] = model
        cv_metrics[ticker] = metrics

    # сохраняем обученные модели
    joblib.dump(
        {"models": models, "cv_metrics": cv_metrics, "mode": strategy_mode},
        project_path.as_posix() + "/models/model_bundle.joblib",
    )
    save_dict(
        cv_metrics,
        project_path.as_posix() + "/artifacts/metrics/train_cv_metrics.json",
    )
    logger.info("Training finished: %d models saved", len(models))


if __name__ == "__main__":
    train()
