import logging
import os
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

from equity_project.src.utils import load_config, three_barrier

warnings.filterwarnings("ignore")

project_path = Path(__file__).parent.parent
logger = logging.getLogger(__name__)


def generate_features(data):
    """Generate some features based on data

    Args:
        data (pd.DataFrame): raw OHLC dataset

    Returns:
        pd.DataFrame: features dataset
    """
    X = data.copy()

    close_col = "Close"

    # dealing with multiindex
    tickers = X[close_col].columns

    # price deviation from moving averages
    X[[(("dev5"), ticker) for ticker in tickers]] = (
        X[close_col] - X[close_col].rolling(5).mean()
    ) / X[close_col]
    X[[(("dev22"), ticker) for ticker in tickers]] = (
        X[close_col] - X[close_col].rolling(22).mean()
    ) / X[close_col]
    X[[(("dev252"), ticker) for ticker in tickers]] = (
        X[close_col] - X[close_col].rolling(252).mean()
    ) / X[close_col]
    X[[(("ma200vs50"), ticker) for ticker in tickers]] = (
        X[close_col].rolling(200).mean() - X[close_col].rolling(50).mean()
    ) / X[close_col]

    # price momentum
    X[[(("mom5"), ticker) for ticker in tickers]] = (
        X[close_col].pct_change(5).rank(axis=1)
    )
    X[[(("mom22"), ticker) for ticker in tickers]] = (
        X[close_col].pct_change(22).rank(axis=1)
    )
    X[[(("mom252"), ticker) for ticker in tickers]] = (
        X[close_col].pct_change(252).rank(axis=1)
    )

    # volatility
    X[[(("vol5"), ticker) for ticker in tickers]] = (X[close_col].rolling(5).std()) / X[
        close_col
    ].rolling(5).mean()
    X[[(("vol22"), ticker) for ticker in tickers]] = (
        X[close_col].rolling(22).std()
    ) / X[close_col].rolling(22).mean()
    X[[(("vol252"), ticker) for ticker in tickers]] = (
        X[close_col].rolling(252).std()
    ) / X[close_col].rolling(252).mean()

    # drop unnecessary сols
    X.drop(columns=["Close", "High", "Low", "Open", "Volume"], inplace=True)

    # avoid forward-looking
    X = X.shift(1)

    # avoid cold start
    X = X.iloc[260:, :]

    return X


def get_valid_tickers(data, min_coverage=0.9, coverage_field="Close"):
    """Возвращает тикеры, у которых достаточно непустых наблюдений."""
    data.columns.names = ["Price", "Ticker"]
    coverage = data[coverage_field].notna().mean().sort_values(ascending=False)
    valid_tickers = coverage[coverage >= min_coverage].index.tolist()
    return valid_tickers, coverage


def prepare_ohlc_data(data, min_coverage=0.9, coverage_field="Close"):
    """Фильтрует тикеры по покрытию, удаляет общие NaN-строки и делает ffill."""
    data = data.copy()
    data.index.name = "Date"
    data.columns.names = ["Price", "Ticker"]
    valid_tickers, coverage = get_valid_tickers(data, min_coverage, coverage_field)

    idx = pd.IndexSlice
    data = data.loc[:, idx[:, valid_tickers]].sort_index(axis=1)

    close = data[coverage_field]
    data = data.loc[~close.isna().all(axis=1)]
    data = data.ffill()
    data = data.dropna(axis=0, how="any")

    return data, valid_tickers, coverage


def get_label(train_data, cfg=None):
    """Создаем разметку для ML модели на основе тройного барьерного метода. Его параметры захардкожены, но при желании вы можете вынести их в конфиг

    Args:
        train_data (pd.DataFrame): raw OHLC dataset

    Returns:
        pd.DataFrame: target dataset
    """
    labeling_cfg = (cfg or {}).get("labeling", {})
    target = train_data.Close.apply(
        three_barrier,
        ptSl=labeling_cfg.get("pt_sl", [1, 1]),
        vertical_barrier_days=labeling_cfg.get("vertical_barrier_days", 10),
        target_return=labeling_cfg.get("target_return", 0.05),
    )
    return target


def get_raw_data():
    """Скачиваем OHLC данные, а также для каждого тикера определяем дату его первого вхождения в индекс

    Returns:
        pd.DataFrame: OHLC данные для всех акций, когда либо входивших в индекс S&P500
        dict: Словарь, где ключ - тикер, значение - дата первого вхождения в индекс
    """
    cfg = load_config(project_path.parent.as_posix() + "/config.yaml")
    TRAIN_START_DATE = cfg["train_start_date"]
    BACKTEST_END_DATE = cfg["backtest_end_date"]
    logger.info("Reading historical S&P 500 components")

    # будем включать в выборку бумаг не только актуальных состав индекса, но и все исторические вхождения
    # это должно убрать ошибку выжившего из датасета
    historical_components = pd.read_csv(
        project_path.as_posix() + "/data/pony/S&P_500_Historical_Components.csv",
        index_col=0,
    )

    historical_components = historical_components[
        (historical_components.index >= TRAIN_START_DATE)
        & (historical_components.index <= BACKTEST_END_DATE)
    ]

    first_appearance_dict = {}

    for index, row in historical_components.iterrows():
        for ticker in row.iloc[0].split(","):
            if ticker not in first_appearance_dict:
                first_appearance_dict[ticker] = index

    # поправляем название некоторых тикеров, чтобы yfinance их распознал
    if "BF.B" in first_appearance_dict:
        first_appearance_dict["BF-B"] = first_appearance_dict.pop("BF.B")
    if "BRK.B" in first_appearance_dict:
        first_appearance_dict["BRK-B"] = first_appearance_dict.pop("BRK.B")

    # для этих акций yfinance предоставлет битые данные (нулевые или околонулевые цены для некоторых периодов в прошлом, которые ломают алгоритм)
    # можете изучить их котировки и если yfinance цены истинны, то оставить тикеры в выборке, пока же мы их удалим
    for trash_ticker in ("DEC", "USBC", "CPWR", "TNB", "APP", "BMC", "SBNY"):
        first_appearance_dict.pop(trash_ticker, None)

    TICKERS = list(first_appearance_dict.keys())
    logger.info("Downloading OHLC data for %d tickers", len(TICKERS))

    data = yf.download(
        TICKERS,
        TRAIN_START_DATE,
        BACKTEST_END_DATE,
        group_by="column",
        auto_adjust=True,
    )

    # yahoo finance иногда выдает фантомные колонки по тикерам с неполными данными
    if "Adj Close" in data.columns:
        data.drop(columns="Adj Close", inplace=True)

    data.index = pd.to_datetime(data.index)
    data = data.astype(float)
    logger.info("Raw OHLC data downloaded: shape=%s", data.shape)

    return (
        data,
        first_appearance_dict,
    )


def get_data():
    """Скачиваем сырые данные, строим на их основе фичасеты и таргеты для наших тикеров и сохраняем сырые и обработанные данные"""
    cfg = load_config(project_path.parent.as_posix() + "/config.yaml")
    logger.info("Starting data preparation")

    (
        data,
        first_appearance_dict,
    ) = get_raw_data()

    data_cfg = cfg.get("data", {})
    data, valid_tickers, coverage = prepare_ohlc_data(
        data,
        min_coverage=data_cfg.get("min_ticker_coverage", 0.9),
        coverage_field=data_cfg.get("coverage_field", "Close"),
    )
    logger.info(
        "Filtered tickers by coverage: %d/%d kept",
        len(valid_tickers),
        len(coverage),
    )
    first_appearance_dict = {
        ticker: first_appearance_dt
        for ticker, first_appearance_dt in first_appearance_dict.items()
        if ticker in valid_tickers
    }

    os.makedirs(project_path.as_posix() + "/data/processed", exist_ok=True)
    os.makedirs(project_path.as_posix() + "/data/raw", exist_ok=True)
    coverage.to_frame("coverage").to_parquet(
        project_path.as_posix() + "/data/processed/ticker_coverage.parquet"
    )
    logger.info("Ticker coverage saved")

    # генерируем фичи для ML модели
    logger.info("Generating features")
    X = generate_features(data)

    # генерируем столбец таргета
    logger.info("Generating triple-barrier labels")
    y = get_label(data, cfg)

    X = X.stack(level=1)

    TRAIN_START_DATE = cfg["train_start_date"]
    TRAIN_END_DATE = cfg["train_end_date"]
    BACKTEST_START_DATE = cfg["backtest_start_date"]
    BACKTEST_END_DATE = cfg["backtest_end_date"]

    # для каждого тикера определяем дату первого вхождения в индекс
    for ticker, first_appearance_dt in first_appearance_dict.items():
        condition_to_drop = (X.index.get_level_values("Date") < first_appearance_dt) & (
            X.index.get_level_values("Ticker") == ticker
        )
        X = X[~condition_to_drop]

    idx = pd.IndexSlice
    data = data.loc[:, idx[:, X.index.get_level_values("Ticker").unique()]]
    y = y.stack(level=0).loc[X.index]
    y.name = "target"

    # разбиваем исходные данные на трейн и бэктест
    train_data = data[
        (data.index.get_level_values("Date") <= TRAIN_END_DATE)
        & (data.index.get_level_values("Date") >= TRAIN_START_DATE)
    ]
    train_data.to_parquet(project_path.as_posix() + "/data/raw/train_data.parquet")
    logger.info("Saved train raw data: shape=%s", train_data.shape)

    backtest_data = data[
        (data.index.get_level_values("Date") <= BACKTEST_END_DATE)
        & (data.index.get_level_values("Date") >= BACKTEST_START_DATE)
    ]

    backtest_data.to_parquet(
        project_path.as_posix() + "/data/raw/backtest_data.parquet", engine="pyarrow"
    )
    logger.info("Saved backtest raw data: shape=%s", backtest_data.shape)

    # разбиваем фичасеты и таргеты ML модели на трейн и бэктест
    X_train = X[
        (X.index.get_level_values("Date") <= TRAIN_END_DATE)
        & (X.index.get_level_values("Date") >= TRAIN_START_DATE)
    ]
    X_train.to_parquet(project_path.as_posix() + "/data/processed/X_train.parquet")
    logger.info("Saved X_train: shape=%s", X_train.shape)

    y_train = y.to_frame()[
        (y.index.get_level_values("Date") <= TRAIN_END_DATE)
        & (y.index.get_level_values("Date") >= TRAIN_START_DATE)
    ]
    y_train.to_parquet(project_path.as_posix() + "/data/processed/y_train.parquet")
    logger.info("Saved y_train: shape=%s", y_train.shape)

    X_backtest = X[
        (X.index.get_level_values("Date") <= BACKTEST_END_DATE)
        & (X.index.get_level_values("Date") >= BACKTEST_START_DATE)
    ]
    X_backtest.to_parquet(
        project_path.as_posix() + "/data/processed/X_backtest.parquet"
    )
    logger.info("Saved X_backtest: shape=%s", X_backtest.shape)

    y_backtest = y.to_frame()[
        (y.index.get_level_values("Date") <= BACKTEST_END_DATE)
        & (y.index.get_level_values("Date") >= BACKTEST_START_DATE)
    ]
    y_backtest.to_parquet(
        project_path.as_posix() + "/data/processed/y_backtest.parquet"
    )
    logger.info("Saved y_backtest: shape=%s", y_backtest.shape)
    logger.info("Data preparation finished")


if __name__ == "__main__":
    get_data()
