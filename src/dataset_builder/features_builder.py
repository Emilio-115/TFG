import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from dataset_builder.config import ORGANS, ORGAN_COL, ORGAN_COL_NUM, TARGET_PATH, SEGMENTS_PATH
from dataset_builder.frame_features import create_agg_df


def build_dataset(window_size=90, window_step=30):

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