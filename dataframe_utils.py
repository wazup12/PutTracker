import pandas as pd
from datetime import datetime


def ensure_dataframe(obj) -> pd.DataFrame:
    if isinstance(obj, pd.Series):
        return obj.to_frame().T
    elif isinstance(obj, pd.DataFrame):
        return obj
    else:
        return pd.DataFrame(obj)


def parse_symbol(df: pd.DataFrame, logger=None) -> pd.DataFrame:
    """Extracts Ticker, Expiration, Strike, and Type from the 'Symbol' column."""
    try:
        symbol_parts = df["Symbol"].str.split(" ", expand=True)
        df["Ticker"] = symbol_parts[0]
        df["Expiration"] = pd.to_datetime(symbol_parts[1], format="%m/%d/%Y")
        df["Strike"] = pd.to_numeric(symbol_parts[2])
        df["Type"] = symbol_parts[3]
    except Exception:
        df["Ticker"] = None
        df["Expiration"] = pd.NaT
        df["Strike"] = None
        df["Type"] = None
    return df


def clean_price_column(df, col_name="Price"):
    df[col_name] = pd.to_numeric(
        df[col_name].astype(str).replace(r"\$", "", regex=True), errors="coerce"
    )
    return df


def filter_options(df):
    return df[df["Security Type"] == "Option"].copy()


def drop_invalid_options(df):
    return df.dropna(subset=["Ticker", "Expiration", "Strike", "Type"])
