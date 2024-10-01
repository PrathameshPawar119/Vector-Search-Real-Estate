import pandas as pd

def clean_data(df):
    df = df.dropna(subset=['street', 'city', 'state'])
    df = df[df['price'] > 0]  # Remove properties without price
    return df
