import numpy as np
import pandas as pd


def combineColumns(df):
    if df['description'].isnull().sum() > 0:
        for i in range(len(df['description'])):
            if df['description'][i] is None or df['description'][i] is np.NaN:
                df['description'][i] = ""

    df2 = pd.DataFrame()
    df2['title'] = df['title'].astype(str) + " " + df['description'].astype(str)
    return df2
