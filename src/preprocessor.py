import pandas as pd

def normalize_data(df):
    normalized_df = df.copy()
    normalized_df.drop(columns='Class', axis=1, inplace=True)
    labels = df['Class']

    normalized_df = (normalized_df-normalized_df.min())/(normalized_df.max()-normalized_df.min())
    return normalized_df, labels

def preprocessing_data(df):
    num_columns = len(df.columns)

    df['Artist Name'] = pd.Categorical(df['Artist Name'])
    df['Artist Name'] = df['Artist Name'].cat.codes

    df['Track Name'] = pd.Categorical(df['Track Name'])
    df['Track Name'] = df['Track Name'].cat.codes

    df.fillna(0, inplace=True)

    #normalized_df = (df-df.min())/(df.max()-df.min())
    # print(normalized_df)
    normalized_df, labels = normalize_data(df)
    return normalized_df, labels, num_columns