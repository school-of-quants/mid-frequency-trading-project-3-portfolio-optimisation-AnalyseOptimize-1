import datetime as dt
import json
from typing import Dict

import pandas as pd
from yaml import safe_load


def load_config(config_path: str) -> Dict:
    """Загружает yaml конфиг в виде python словаря

    Args:
        config_path (str): Путь до конфига

    Returns:
        Dict: Словарь с параметрами конфига
    """
    with open(config_path) as file:
        config = safe_load(file)
    return config


def save_dict(dict_, path):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(dict_, f, indent=4, default=str)


def applyPtSlOnT1(close, events, ptSl):
    """Tripple barrier method adjusted for Low and High prices

    Args:
        close (pd.Series): Close prices.
        events (pd.DataFrame): A pandas dataframe, with columns:
            - t1: The timestamp of vertical barrier. When the value is np.nan, there will not be a vertical barrier.
            - trgt: The unit width of the horizontal barriers.
        ptSl (list):
            - ptSl[0]: The factor that multiplies trgt to set the width of the upper barrier.
                If 0, there will not be an upper barrier.
            - ptSl[1]: The factor that multiplies trgt to set the width of the lower barrier.
                If 0, there will not be a lower barrier.

    Returns:
        pd.DataFrame: Timestamps of each barrier touch
    """

    # apply stop loss/profit taking, if it takes place before t1 (end of event)

    out = events[["t1"]].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0] * events["trgt"]
    else:
        pt = pd.Series(index=events.index)  # NaNs
    if ptSl[1] > 0:
        sl = -ptSl[1] * events["trgt"]
    else:
        sl = pd.Series(index=events.index)  # NaNs
    for loc, t1 in events["t1"].fillna(close.index[-1]).items():
        df0 = close[loc:t1]  # path prices
        df0 = df0 / close[loc] - 1  # path returns
        out.loc[loc, "sl"] = df0[df0 < sl[loc]].index.min()  # earliest stop loss.
        out.loc[loc, "pt"] = df0[df0 > pt[loc]].index.min()  # earliest profit taking.
    return out


def three_barrier(
    close,
    ptSl=None,
    rolling_n=50,
    scaling_factor=2.0,
    vertical_barrier_days=10,
    target_return=0.05,
    use_volatility_target=False,
):
    """Labeling based on tripple barrier method and historical volatility

    Args:
        close (pd.DataFrame): Assets close prices
        ptSl (list, optional): applyPtSlOnT1 ptSl parameter. Defaults to [1, 1].
        rolling_n (int, optional): Rolling window to calculate standart deviation. Defaults to 50.
        scaling_factor (float, optional): Multiplier of standart deviation to calculate horizontal barrier size. Defaults to 2.0.

    Returns:
        pd.Series: 3-class labels
    """
    if ptSl is None:
        ptSl = [1, 1]

    if use_volatility_target:
        trgt = close.pct_change().rolling(rolling_n).std() * scaling_factor
        trgt = trgt.shift(1).fillna(target_return)
        trgt = trgt.clip(lower=target_return)
    else:
        trgt = target_return

    events = pd.DataFrame(
        {
            "t1": close.index + dt.timedelta(days=vertical_barrier_days),
            "trgt": trgt,
        },
        index=close.index,
    )
    out = applyPtSlOnT1(close, events, ptSl)
    target = out.apply(
        lambda x: 1 if x.idxmin() == "pt" else -1 if x.idxmin() == "sl" else 0, axis=1
    )

    return target
