import pandas as pd


def get_data():

    df = pd.read_csv('images/vehcle_img_data.csv')
    return df