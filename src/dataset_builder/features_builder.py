import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from .config import ORGANS, ORGAN_COL, ORGAN_COL_NUM, TARGET_PATH, SEGMENTS_PATH
from .frame_features import create_agg_df
from .frame_features_no_agg import create_dataset_keras


def build_dataset_agg(window_size=60, window_step=15):

    pop_df = pd.read_csv(TARGET_PATH)

    feats_df = create_agg_df(
        pop_df,
        SEGMENTS_PATH,
        window_size=window_size,
        window_step=window_step
    )

    # convertir órgano a número
    feats_df[ORGAN_COL_NUM] = OrdinalEncoder(
        categories=[ORGANS]
    ).fit_transform(feats_df[[ORGAN_COL]])

    # merge con los casos y labels
    final_df = pd.merge(
        feats_df,
        pop_df,
        left_on="case_id",
        right_on="case",
        how="left"
    )

    return final_df

def build_dataset_keras(window_size=60, window_step=15):

    pop_df = pd.read_csv(TARGET_PATH)

    X, meta_list = create_dataset_keras(
        pop_df,
        SEGMENTS_PATH,
        window_size=window_size,
        window_step=window_step
    )

    meta_df = pd.DataFrame(meta_list)

    # merge para obtener labels
    meta_df = pd.merge(
        meta_df,
        pop_df,
        left_on="case_id",
        right_on="case",
        how="left"
    )

    return X, meta_df