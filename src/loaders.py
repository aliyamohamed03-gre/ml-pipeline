import pandas as pd


def load_cmu(filepath):
    
    #read the CSV into a DataFrame
    df = pd.read_csv(filepath)
    
   
    hold_cols = [c for c in df.columns if c.startswith('H.')]
    dd_cols = [c for c in df.columns if c.startswith('DD.')]
    

    df['mean_hold_time'] = df[hold_cols].mean(axis=1)
    df['mean_inter_key_interval'] = df[dd_cols].mean(axis=1)
    df['std_inter_key_interval'] = df[dd_cols].std(axis=1)
    df['total_duration'] = df[dd_cols].sum(axis=1)
    
    #build a unique session identifier from session and repetition
    df['session_id'] = (
        df['sessionIndex'].astype(str) + '_' + df['rep'].astype(str)
    )
    
    #rename 'subject' to 'user_id' for consistency across datasets
    df = df.rename(columns={'subject': 'user_id'})
    
    #only select needed columns 
    columns_to_keep = [
        'user_id',
        'session_id',
        'mean_hold_time',
        'mean_inter_key_interval',
        'std_inter_key_interval',
        'total_duration',
    ]
    result = df[columns_to_keep].copy()
    
    #remove any of the rows with missing values 
    result = result.dropna()
    
    return result

def load_aalto(test_sections_filepath, max_users=300, min_sessions_per_user=5):
    
    
    print(f"Loading session-level data from {test_sections_filepath}...")
    df = pd.read_csv(test_sections_filepath)
    print(f"Loaded {len(df)} sessions across {df['PARTICIPANT_ID'].nunique()} users")
    
    #filtering out the sessions with no keystrokes or zero duration
    df = df[df['KEYSTROKES'] > 0]
    df = df[df['INPUT_TIME'] > 0]
    
    #counting the sessions per user and filter users with not enough sessions 
    session_counts = df.groupby('PARTICIPANT_ID').size()
    eligible_users = session_counts[session_counts >= min_sessions_per_user].index
    df = df[df['PARTICIPANT_ID'].isin(eligible_users)]
    
    #pick only the the top users by session count
    top_users = df.groupby('PARTICIPANT_ID').size().sort_values(ascending=False).head(max_users).index
    df = df[df['PARTICIPANT_ID'].isin(top_users)].copy()
    print(f"Sampled to {len(top_users)} users with {len(df)} total sessions")
    
    #derive the features we need from what's available -- mean_inter_key_interval = total input time / number of keystrokes
    #as INPUT_TIME is in milliseconds, we will convert to seconds
    df['total_duration'] = df['INPUT_TIME'] / 1000
    df['mean_inter_key_interval'] = df['total_duration'] / df['KEYSTROKES']
    
    #as we don't have per-keystroke variance, we will approximate std using error rate
    #We wil assume igher error rate typically correlates with more variable typing rhythm
    df['std_inter_key_interval'] = df['mean_inter_key_interval'] * (1 + df['ERROR_RATE'])
    
    #as hold time is not available per-session in processed2020 data we will use WPM-derived estimate: faster typing = shorter holds on average
    #typical the hold time is 80-120ms so we scale inversely with WPM
    df['mean_hold_time'] = 0.1 * (50.0 / df['WPM'].clip(lower=10))
    df['mean_hold_time'] = df['mean_hold_time'].clip(upper=0.5)  #capping the unrealistic values
    
    #build the output 
    result = pd.DataFrame({
        'user_id': df['PARTICIPANT_ID'].astype(str),
        'session_id': df['TEST_SECTION_ID'].astype(str),
        'mean_hold_time': df['mean_hold_time'],
        'mean_inter_key_interval': df['mean_inter_key_interval'],
        'std_inter_key_interval': df['std_inter_key_interval'],
        'total_duration': df['total_duration'],
    })
    
    #not inxcluding any rows with missing or infinite values
    result = result.dropna()
    result = result[~result.isin([float('inf'), float('-inf')]).any(axis=1)]
    
    print(f"Final output: {len(result)} sessions from {result['user_id'].nunique()} users")
    return result.reset_index(drop=True)

