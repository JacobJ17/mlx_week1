import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

with open("user_lookup.pkl", "rb") as f:
    user_lookup = pickle.load(f)
# Load the saved encoder
with open("domain_encoder.pkl", "rb") as f:
    saved = pickle.load(f)
top_domains = saved["top_domains"]
le = saved["label_encoder"]

def get_user_features(username, post_time):
    user = user_lookup.get(username)
    if not user:
        return None, None, None
    age_at_post = (post_time - user["created"]).total_seconds()
    return user["karma"], user["submitted_count"], age_at_post

def get_day_group(dt):
    wd = dt.weekday()
    if wd < 5:
        return 'weekday'
    elif wd == 5:
        return 'saturday'
    else:
        return 'sunday'

def load_and_preprocess_data(parquet_file):
    # Read df from file
    df = pd.read_parquet(parquet_file)
    ### Process domain into embedding indices
    # If domain is categorical, add 'other' to categories
    if isinstance(df['domain'].dtype, pd.CategoricalDtype):
        if 'other' not in df['domain'].cat.categories:
            df['domain'] = df['domain'].cat.add_categories(['other'])
    df['domain_top100'] = df['domain'].where(df['domain'].isin(top_domains), 'other')
    df['domain_idx'] = le.transform(df['domain_top100'])
    
    ### Process time features
    # Convert post times in your main df
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'], unit='s')
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    # Time of day as a float: 0.0–24.0
    df['time_float'] = df['hour'] + df['minute'] / 60.0
    # Encode using sine and cosine (period = 24h)
    df['time_sin'] = np.sin(2 * np.pi * df['time_float'] / 24)
    df['time_cos'] = np.cos(2 * np.pi * df['time_float'] / 24)
    df.drop(columns=['hour', 'minute', 'time_float'], inplace=True)

    ### Process user features
    user_df = pd.DataFrame.from_dict(user_lookup, orient='index')
    user_df.index.name = 'by'
    user_df.reset_index(inplace=True)
    # Convert 'created' to datetime if it’s not already
    if not pd.api.types.is_datetime64_any_dtype(user_df['created']):
        user_df['created'] = pd.to_datetime(user_df['created'])
    df = df.merge(user_df, on='by', how='left')
    # Add user_exists column: 1 if user exists, 0 if not
    df['user_exists'] = df['karma'].notnull().astype(int)

    # Set user features to 0 if user does not exist
    for col in ['karma', 'submitted_count', 'created']:
        df[col] = df[col].fillna(0)

    # For created, if user doesn't exist, set to post time so age=0
    df['created'] = np.where(df['created'] == 0, df['time'], df['created'])
    df['created'] = pd.to_datetime(df['created'])

    df['user_age_at_post'] = (df['time'] - df['created']).dt.total_seconds()
    df['user_age_at_post'] = df['user_age_at_post'].clip(lower=0)
    df['log_user_age'] = np.log1p(df['user_age_at_post'])
    df['karma'] = df['karma'].clip(lower=0)
    df['log_karma'] = np.log1p(df['karma'])
    df['submitted_count'] = df['submitted_count'].clip(lower=0)
    df['log_submitted_count'] = np.log1p(df['submitted_count'])


    df['day_group'] = df['time'].apply(get_day_group)
    df = pd.get_dummies(df, columns=['day_group'])

    scaler = StandardScaler()
    df[['log_user_age', 'log_karma', 'log_submitted_count']] = scaler.fit_transform(
        df[['log_user_age', 'log_karma', 'log_submitted_count']]
    )

    df['log_score'] = np.log1p(df['score'])

    return df[[
        'title', 'domain_idx', 'log_user_age', 
        'log_karma', 'log_submitted_count', 'user_exists',
        'time_sin', 'time_cos', 
        'day_group_weekday', 'day_group_saturday', 'day_group_sunday',
        'score', 'log_score'
    ]]

def main():
    path = "hacker_news_ml_ready.parquet"
    df = load_and_preprocess_data(path)
    print(df.head())

if __name__ == "__main__":
    main()