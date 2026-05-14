import pandas as pd


FEATURE_COLUMNS = [
    "temperature_k",
    "pressure_atm",
    "vibration_mms",
    "rolling_mean_16",
    "rolling_std_16",
    "z_score_16",
    "momentum_8",
    "lag_1",
    "volatility_16",
    "ema_12",
]


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    for column in FEATURE_COLUMNS:
        if column not in frame:
            frame[column] = 0.0
    return frame.ffill().fillna(0.0)
