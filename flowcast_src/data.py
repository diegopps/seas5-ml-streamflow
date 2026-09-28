"""
flowcast_src.data – Data loading and assembly, shared by every stage.

Loads the observed discharge record and the preprocessed SEAS5 feature
parquet, and assembles the analysis frame used by every model: features
merged with the observed target Q, missing-Q rows dropped, and skip_years
removed.
"""

import pandas as pd

GROUP_COL = 'hydro_year'


def load_obs_discharge(discharge_csv: str) -> pd.Series:
    """
    Load observed daily discharge as a Series indexed by normalised date.

    The CSV is expected to hold exactly two columns (date, discharge). Rows
    with missing discharge are dropped and the series is sorted by date.
    """
    df = pd.read_csv(discharge_csv)
    if df.shape[1] != 2:
        raise ValueError(
            f"Expected 2 columns (date, Q) in {discharge_csv}, "
            f"found {df.shape[1]}: {list(df.columns)}"
        )
    df.columns = ['date', 'Q']
    df['date'] = pd.to_datetime(df['date'], format='mixed').dt.normalize()
    df = df.dropna(subset=['Q']).set_index('date')['Q'].sort_index()
    return df


def load_features(in_path: str) -> pd.DataFrame:
    """Load the preprocessed feature parquet with normalised date columns."""
    df = pd.read_parquet(in_path)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df['init_date'] = pd.to_datetime(df['init_date']).dt.normalize()
    return df


def attach_observed_target(df_features: pd.DataFrame,
                           obs_daily_q: pd.Series) -> pd.DataFrame:
    """
    Merge the observed daily discharge onto the feature frame as target `Q`,
    joined on date. Any pre-existing Q column is replaced.
    """
    obs_q_df = obs_daily_q.rename('Q_obs').reset_index()
    obs_q_df.columns = ['date', 'Q_obs']
    df = df_features.drop(columns=['Q'], errors='ignore')
    df = df.merge(obs_q_df, on='date', how='left')
    return df.rename(columns={'Q_obs': 'Q'})


def build_analysis_frame(in_path: str,
                         obs_daily_q: pd.Series,
                         skip_years: set) -> pd.DataFrame:
    """
    Assemble the analysis frame shared by every model:
        load features -> attach observed Q -> drop missing-Q rows
        -> remove skip_years.

    Returns the frame plus the sorted list of hydro years it contains.
    """
    df = load_features(in_path)
    df = attach_observed_target(df, obs_daily_q)
    df = df.dropna(subset=['Q']).reset_index(drop=True)
    df = df[~df[GROUP_COL].isin(skip_years)].reset_index(drop=True)
    unique_years = sorted(df[GROUP_COL].unique())
    return df, unique_years
