import numpy as np
import pandas as pd

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds advanced interaction features to the dataframe."""
    df['charging_efficiency'] = df['kWhDelivered'] / df['parsed_kWhRequested'].replace(0, np.nan)
    df['requested_gap'] = df['parsed_kWhRequested'] - df['kWhDelivered']
    return df
