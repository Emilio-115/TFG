from features_builder import build_dataset
from feature_definitions import COL_SELECTION

def main():

    final_df = build_dataset(
        window_size=60,
        window_step=30
    )
    final_df = final_df[COL_SELECTION]
    final_df.to_csv(
        "data/case_level_feats_alltargets_w90_s30.csv",
        index=False
    )

if __name__ == "__main__":
    main()