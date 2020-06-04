import pandas as pd


def read_data(PATH):
    df = pd.read_csv(PATH)
    df.fillna(" ", inplace=True)
    print("DATAFRAME LOADED!")

    return df
