"""
Dataset loader functions.

Each loader reads a raw dataset file and returns a pandas DataFrame
with a standardised schema: user_id, session_id, and feature columns.
"""

import pandas as pd


def load_cmu(filepath):
    """
    Load the CMU keystroke dynamics dataset.
    
    The dataset contains 51 users each typing the password '.tie5Roanl'
    approximately 400 times across 8 sessions. Each row in the raw file
    is one password typing attempt with pre-computed timing measurements.
    
    This function aggregates per-row features into a clean, standardised
    output with four numerical features per session.
    
    Parameters
    ----------
    filepath : str
        Path to the DSL-StrongPasswordData.csv file.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
            user_id (str), session_id (str),
            mean_hold_time (float), mean_inter_key_interval (float),
            std_inter_key_interval (float), total_duration (float).
    """
    # Read the CSV into a DataFrame
    df = pd.read_csv(filepath)
    
    # Identify columns by their prefix
    hold_cols = [c for c in df.columns if c.startswith('H.')]
    dd_cols = [c for c in df.columns if c.startswith('DD.')]
    
    # Compute aggregated features for each row
    df['mean_hold_time'] = df[hold_cols].mean(axis=1)
    df['mean_inter_key_interval'] = df[dd_cols].mean(axis=1)
    df['std_inter_key_interval'] = df[dd_cols].std(axis=1)
    df['total_duration'] = df[dd_cols].sum(axis=1)
    
    # Build a unique session identifier from session and repetition
    df['session_id'] = (
        df['sessionIndex'].astype(str) + '_' + df['rep'].astype(str)
    )
    
    # Rename 'subject' to 'user_id' for consistency across datasets
    df = df.rename(columns={'subject': 'user_id'})
    
    # Select only the columns we need
    columns_to_keep = [
        'user_id',
        'session_id',
        'mean_hold_time',
        'mean_inter_key_interval',
        'std_inter_key_interval',
        'total_duration',
    ]
    result = df[columns_to_keep].copy()
    
    # Remove any rows with missing values
    result = result.dropna()
    
    return result