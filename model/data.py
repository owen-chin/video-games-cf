import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    df.columns = ['user_id', 'title', 'action', 'hours', 'x']

    # del last col
    df = df.drop('x', axis=1)

    # remove all purchase actions
    df = df.drop(df[df['action'] == 'purchase'].index)

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df['user_id'] = user_encoder.fit_transform(df['user_id'].values)
    df['title'] = item_encoder.fit_transform(df['title'].values)

    return df, user_encoder, item_encoder
