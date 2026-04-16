from .features_builder import build_dataset_agg,build_dataset_keras
from .feature_definitions import COL_SELECTION, NEW_FEATS
import numpy as np

def main(agg=True):

    if agg:
        final_df = build_dataset_agg(
            window_size=60,
            window_step=15,
        )

        final_df = final_df[COL_SELECTION + NEW_FEATS]

        final_df.to_csv(
            "data/case_level_feats.csv",
            index=False
        )

    else:
        X, meta_df = build_dataset_keras(
            window_size=60,
            window_step=15,
        )

        np.save("data/case_level_feats_nn.npy", X)
        meta_df.to_csv("data/nn_meta.csv", index=False)

if __name__ == "__main__":
    main()