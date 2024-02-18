import pandas as pd
import numpy as np
from notebooks import Feature as ft


def refactor_data(df):
    # Fill `attack_cat` to 'normal'
    df = fill_attack_cat(df)

    # Change port starting with '0x' to decimal
    df['sport'] = df['sport'].apply(hex_to_dec)
    df['dsport'] = df['dsport'].apply(hex_to_dec)

    # Fill 'ct_flw_http_mthd' to average of each feature
    # for feature in ['ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd']:
    #     df = fill_http_ftp(df, feature)

    df.replace([None, '', 'NaN', '-', ' '], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Change data type based on the project appendix
    # df = convert_feature_type(df, ft.port_, int)
    # df = convert_feature_type(df, ft.ip_, str)
    df = convert_feature_type(df, ft.object_, str)
    df = convert_feature_type(df, ft.integer_, int)
    df = convert_feature_type(df, ft.float_, float)
    df = convert_feature_type(df, ft.binary_, bool)

    # Factorize 4 features (proto, state, service, attack_cat)
    df = factorize_feature(df, ft.factorize_)
    return df


def factorize_feature(df, features):
    for feature in features:
        df[feature] = pd.factorize(df[feature])[0]
    return df


def fill_attack_cat(df):
    df['attack_cat'] = df.attack_cat.fillna(value='normal').apply(
        lambda x: x.strip().lower())
    return df


def hex_to_dec(port):
    if isinstance(port, str) and port.startswith('0x'):
        return int(port, 16)
    else:
        return port


def fill_http_ftp(df, feature):
    mean_val = df[feature].mean()
    df[feature].fillna(mean_val, inplace=True)
    return df


def convert_feature_type(df, features, dtype):
    for feature in features:
        df[feature] = df[feature].astype(dtype)
    return df
