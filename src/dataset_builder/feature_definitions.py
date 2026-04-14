import numpy as np
from itertools import combinations
from collections import namedtuple
from .config import ORGANS
from skimage.measure import label, regionprops

Point = namedtuple('Point', ['x','y'])

FRAME_CONF_STATS = {
    '1.1_frame_conf_mean': lambda frame: frame.mean(),
    '1.2_frame_conf_std': lambda frame: frame.std(),
    '1.3_frame_conf_max': lambda frame: frame.max(),
    '1.4_frame_conf_min': lambda frame: frame.min(),
}

REGION_CONF_STATS = {
    '2.1_region_conf_mean': lambda region: region.mean() if region.size > 0 else np.nan,
    '2.2_region_conf_std': lambda region: region.std() if region.size > 0 else np.nan,
    '2.3_region_conf_max': lambda region: region.max() if region.size > 0 else np.nan,
    '2.4_region_conf_min': lambda region: region.min() if region.size > 0 else np.nan,
}

REGION_COORDS_STATS = {
    '3.1_region_coords_centroid_Y': lambda region_coords: region_coords[0].mean() if region_coords[0].size > 0 else np.nan,
    '3.2_region_coords_centroid_X': lambda region_coords: region_coords[1].mean() if region_coords[1].size > 0 else np.nan,
    '3.3_region_coords_max_Y': lambda region_coords: region_coords[0].max() if region_coords[0].size > 0 else np.nan,
    '3.4_region_coords_max_X': lambda region_coords: region_coords[1].max() if region_coords[1].size > 0 else np.nan,
    '3.5_region_coords_min_Y': lambda region_coords: region_coords[0].min() if region_coords[0].size > 0 else np.nan,
    '3.6_region_coords_min_X': lambda region_coords: region_coords[1].min() if region_coords[1].size > 0 else np.nan,
    '3.7_region_coords_len_Y': lambda region_coords: (region_coords[0].max() - region_coords[0].min()) if region_coords[0].size > 0 else np.nan,
    '3.8_region_coords_len_X': lambda region_coords: (region_coords[1].max() - region_coords[1].min()) if region_coords[0].size > 0 else np.nan,
    '3.9_region_coords_bbox_area': lambda region_coords: (
        (region_coords[0].max() - region_coords[0].min()) *
        (region_coords[1].max() - region_coords[1].min())
    ) if region_coords[0].size > 0 else np.nan,
}



SPATIAL_PAIRS = list(combinations(ORGANS, 2))

SPATIAL_STATS = {
    f'4.{10+i:02d}_eucl_dist_{org1}_{org2}': 
        lambda centroids_coords: euclidean_distance(centroids_coords[0], centroids_coords[1])
    for i, (org1, org2) in enumerate(SPATIAL_PAIRS)
}


SEQUENCE_AGGREGATION_STATS = {
    '_mean': lambda feat_seq: feat_seq.mean(),
    '_std': lambda feat_seq: feat_seq.std(),
    '_max': lambda feat_seq: feat_seq.max(),
    '_min': lambda feat_seq: feat_seq.min(),
    # '_velocity_mean': lambda feat_seq: feat_seq.diff().abs().mean(),
    # '_velocity_max':  lambda feat_seq: feat_seq.diff().abs().max(),
}

def euclidean_distance(organ1: np.typing.NDArray,organ2: np.typing.NDArray):
    if np.any(np.isnan([organ1, organ2])):
        return np.nan
        
    return np.linalg.norm(organ1 - organ2)

spatial_col_selection = [
    f'{k}{agg_k}'
    for k in SPATIAL_STATS.keys()
    for agg_k in SEQUENCE_AGGREGATION_STATS.keys()
]

def _eccentricity_skimage(frame_organ_mask):
    binary = (frame_organ_mask > 0.5).astype(np.uint8)
    labeled = label(binary)
    props = regionprops(labeled)
    if not props:
        return np.nan
    largest = max(props, key=lambda r: r.area)
    return largest.eccentricity


FRAME_SHAPE_STATS = {
    '5.1_eccentricity_skimage': lambda frame: _eccentricity_skimage(frame),
}

frame_shape_col_selection = [
    f'{k}{agg_k}'
    for k in FRAME_SHAPE_STATS.keys()
    for agg_k in SEQUENCE_AGGREGATION_STATS.keys()
]

COL_SELECTION = ['case', 'nhc_final', 'start_frame', 'end_frame', 'organ', 'organ_num', 
                 'any_prolapse', 'cystocele', 'cystourethrocele', 'uterine_prolapse', 
                 'cervical_elongation', 'rectocele', 'enterocele',
                 '1.1_frame_conf_mean_mean',
                 '1.1_frame_conf_mean_std', '1.1_frame_conf_mean_max',
                 '1.1_frame_conf_mean_min', '1.2_frame_conf_std_mean',
                 '1.2_frame_conf_std_std', '1.2_frame_conf_std_max',
                 '1.2_frame_conf_std_min', '1.3_frame_conf_max_mean',
                 '1.3_frame_conf_max_std', '1.3_frame_conf_max_max',
                 '1.3_frame_conf_max_min', '1.4_frame_conf_min_mean',
                 '1.4_frame_conf_min_std', '1.4_frame_conf_min_max',
                 '1.4_frame_conf_min_min', '2.1_region_conf_mean_mean',
                 '2.1_region_conf_mean_std', '2.1_region_conf_mean_max',
                 '2.1_region_conf_mean_min', '2.2_region_conf_std_mean',
                 '2.2_region_conf_std_std', '2.2_region_conf_std_max',
                 '2.2_region_conf_std_min', '2.3_region_conf_max_mean',
                 '2.3_region_conf_max_std', '2.3_region_conf_max_max',
                 '2.3_region_conf_max_min', '2.4_region_conf_min_mean',
                 '2.4_region_conf_min_std', '2.4_region_conf_min_max',
                 '2.4_region_conf_min_min', '3.1_region_coords_centroid_Y_mean',
                 '3.1_region_coords_centroid_Y_std', '3.1_region_coords_centroid_Y_max',
                 '3.1_region_coords_centroid_Y_min', '3.2_region_coords_centroid_X_mean',
                 '3.2_region_coords_centroid_X_std', '3.2_region_coords_centroid_X_max',
                 '3.2_region_coords_centroid_X_min', '3.3_region_coords_max_Y_mean',
                 '3.3_region_coords_max_Y_std', '3.3_region_coords_max_Y_max',
                 '3.3_region_coords_max_Y_min', '3.4_region_coords_max_X_mean',
                 '3.4_region_coords_max_X_std', '3.4_region_coords_max_X_max',
                 '3.4_region_coords_max_X_min', '3.5_region_coords_min_Y_mean',
                 '3.5_region_coords_min_Y_std', '3.5_region_coords_min_Y_max',
                 '3.5_region_coords_min_Y_min', '3.6_region_coords_min_X_mean',
                 '3.6_region_coords_min_X_std', '3.6_region_coords_min_X_max',
                 '3.6_region_coords_min_X_min', '3.7_region_coords_len_Y_mean',
                 '3.7_region_coords_len_Y_std', '3.7_region_coords_len_Y_max',
                 '3.7_region_coords_len_Y_min', '3.8_region_coords_len_X_mean',
                 '3.8_region_coords_len_X_std', '3.8_region_coords_len_X_max',
                 '3.8_region_coords_len_X_min', '3.9_region_coords_bbox_area_mean',
                 '3.9_region_coords_bbox_area_std', '3.9_region_coords_bbox_area_max',
                 '3.9_region_coords_bbox_area_min']

NEW_FEATS = spatial_col_selection + frame_shape_col_selection


COLS_TO_DROP = [
    '1.1_frame_conf_mean_mean', '1.1_frame_conf_mean_std', '1.1_frame_conf_mean_max', '1.1_frame_conf_mean_min',
    '1.2_frame_conf_std_mean', '1.2_frame_conf_std_std', '1.2_frame_conf_std_max', '1.2_frame_conf_std_min',
    '1.3_frame_conf_max_mean', '1.3_frame_conf_max_std', '1.3_frame_conf_max_max', '1.3_frame_conf_max_min',
    '1.4_frame_conf_min_mean', '1.4_frame_conf_min_std', '1.4_frame_conf_min_max', '1.4_frame_conf_min_min',
    '2.1_region_conf_mean_std', '2.1_region_conf_mean_max', '2.1_region_conf_mean_min',
    '2.2_region_conf_std_mean', '2.2_region_conf_std_std', '2.2_region_conf_std_max', '2.2_region_conf_std_min',
    '2.3_region_conf_max_mean', '2.3_region_conf_max_std', '2.3_region_conf_max_max', '2.3_region_conf_max_min',
    '2.4_region_conf_min_mean', '2.4_region_conf_min_std', '2.4_region_conf_min_max', '2.4_region_conf_min_min',
    '3.1_region_coords_centroid_Y_mean', '3.1_region_coords_centroid_Y_std', '3.1_region_coords_centroid_Y_max', '3.1_region_coords_centroid_Y_min',
    '3.2_region_coords_centroid_X_mean', '3.2_region_coords_centroid_X_max', '3.2_region_coords_centroid_X_min',
    '3.3_region_coords_max_Y_mean', '3.3_region_coords_max_Y_std', '3.3_region_coords_max_Y_max', '3.3_region_coords_max_Y_min',
    '3.4_region_coords_max_X_mean', '3.4_region_coords_max_X_std', '3.4_region_coords_max_X_max', '3.4_region_coords_max_X_min',
    '3.5_region_coords_min_Y_mean', '3.5_region_coords_min_Y_std', '3.5_region_coords_min_Y_max', '3.5_region_coords_min_Y_min',
    '3.6_region_coords_min_X_mean', '3.6_region_coords_min_X_std', '3.6_region_coords_min_X_max', '3.6_region_coords_min_X_min',
    '3.7_region_coords_len_Y_mean', '3.7_region_coords_len_Y_std', '3.7_region_coords_len_Y_max', '3.7_region_coords_len_Y_min',
    '3.8_region_coords_len_X_mean', '3.8_region_coords_len_X_std', '3.8_region_coords_len_X_max', '3.8_region_coords_len_X_min',
    '3.9_region_coords_bbox_area_mean', '3.9_region_coords_bbox_area_std', '3.9_region_coords_bbox_area_max', '3.9_region_coords_bbox_area_min',
    '5.1_eccentricity_skimage_mean', '5.1_eccentricity_skimage_std', '5.1_eccentricity_skimage_max', '5.1_eccentricity_skimage_min',
    '4.22_eucl_dist_Bladder_Vagina_mean', '4.22_eucl_dist_Bladder_Vagina_min',
    '4.29_eucl_dist_Pubis_Urethra_min', '4.29_eucl_dist_Pubis_Urethra_std',
    '4.32_eucl_dist_Rectum_Urethra_min', '4.32_eucl_dist_Rectum_Urethra_std',
    '4.23_eucl_dist_Levator ani muscle_Pubis_min', '4.23_eucl_dist_Levator ani muscle_Pubis_std',
    '4.10_eucl_dist_Anus_Bladder_min',
    '4.21_eucl_dist_Bladder_Uterus_max',
    '4.19_eucl_dist_Bladder_Rectum_std',
    '4.35_eucl_dist_Urethra_Uterus_min',
    '1.4_frame_conf_min_min', '1.4_frame_conf_min_max',
]

TOP_10_FEATURES = [
    '4.36_eucl_dist_Urethra_Vagina_max',
    '4.16_eucl_dist_Anus_Vagina_max',
    '4.10_eucl_dist_Anus_Bladder_max',
    '4.13_eucl_dist_Anus_Rectum_mean',
    '4.14_eucl_dist_Anus_Urethra_max',
    '4.15_eucl_dist_Anus_Uterus_min',
    '4.20_eucl_dist_Bladder_Urethra_std',
    '4.37_eucl_dist_Uterus_Vagina_min',
    '4.18_eucl_dist_Bladder_Pubis_min',
    '4.26_eucl_dist_Levator ani muscle_Uterus_mean',
]

COLS_DROP_ALL_NEW_EXCEPT_TOP10 = [
    col for col in (
        [f'{k}{agg_k}' for k in SPATIAL_STATS.keys() for agg_k in SEQUENCE_AGGREGATION_STATS.keys()] +
        ['5.1_eccentricity_skimage_mean', '5.1_eccentricity_skimage_std',
         '5.1_eccentricity_skimage_max', '5.1_eccentricity_skimage_min']
    )
    if col not in TOP_10_FEATURES
]




DROP_TOP = [
    '1.1_frame_conf_mean_mean',
    '1.1_frame_conf_mean_std',
    '1.1_frame_conf_mean_max',
    '1.1_frame_conf_mean_min',
    '1.2_frame_conf_std_mean',
    '1.2_frame_conf_std_std',
    '1.2_frame_conf_std_max',
    '1.2_frame_conf_std_min',
    '1.3_frame_conf_max_mean',
    '1.3_frame_conf_max_std',
    '1.3_frame_conf_max_max',
    '1.3_frame_conf_max_min',
    '1.4_frame_conf_min_mean',
    '1.4_frame_conf_min_std',
    '1.4_frame_conf_min_max',
    '1.4_frame_conf_min_min',
    '2.1_region_conf_mean_std',
    '2.1_region_conf_mean_max',
    '2.2_region_conf_std_mean',
    '2.2_region_conf_std_std',
    '2.2_region_conf_std_max',
    '2.2_region_conf_std_min',
    '2.3_region_conf_max_mean',
    '2.3_region_conf_max_std',
    '2.3_region_conf_max_max',
    '2.3_region_conf_max_min',
    '2.4_region_conf_min_mean',
    '2.4_region_conf_min_std',
    '2.4_region_conf_min_max',
    '2.4_region_conf_min_min',
    '3.1_region_coords_centroid_Y_mean',
    '3.1_region_coords_centroid_Y_std',
    '3.1_region_coords_centroid_Y_max',
    '3.1_region_coords_centroid_Y_min',
    '3.2_region_coords_centroid_X_mean',
    '3.2_region_coords_centroid_X_std',
    '3.2_region_coords_centroid_X_max',
    '3.2_region_coords_centroid_X_min',
    '3.3_region_coords_max_Y_mean',
    '3.3_region_coords_max_Y_std',
    '3.3_region_coords_max_Y_max',
    '3.3_region_coords_max_Y_min',
    '3.4_region_coords_max_X_mean',
    '3.4_region_coords_max_X_std',
    '3.4_region_coords_max_X_max',
    '3.4_region_coords_max_X_min',
    '3.5_region_coords_min_Y_mean',
    '3.5_region_coords_min_Y_std',
    '3.5_region_coords_min_Y_max',
    '3.5_region_coords_min_Y_min',
    '3.6_region_coords_min_X_mean',
    '3.6_region_coords_min_X_std',
    '3.6_region_coords_min_X_max',
    '3.6_region_coords_min_X_min',
    '3.7_region_coords_len_Y_mean',
    '3.7_region_coords_len_Y_std',
    '3.7_region_coords_len_Y_max',
    '3.7_region_coords_len_Y_min',
    '3.8_region_coords_len_X_mean',
    '3.8_region_coords_len_X_std',
    '3.8_region_coords_len_X_max',
    '3.8_region_coords_len_X_min',
    '3.9_region_coords_bbox_area_mean',
    '3.9_region_coords_bbox_area_std',
    '3.9_region_coords_bbox_area_max',
    '3.9_region_coords_bbox_area_min',
    '4.10_eucl_dist_Anus_Bladder_std',
    '4.17_eucl_dist_Bladder_Levator ani muscle_mean',
    '4.17_eucl_dist_Bladder_Levator ani muscle_std',
    '4.22_eucl_dist_Bladder_Vagina_max',
    '4.23_eucl_dist_Levator ani muscle_Pubis_std',
    '4.25_eucl_dist_Levator ani muscle_Urethra_min',
    '4.26_eucl_dist_Levator ani muscle_Uterus_std',
    '4.26_eucl_dist_Levator ani muscle_Uterus_max',
    '4.29_eucl_dist_Pubis_Urethra_std',
    '4.31_eucl_dist_Pubis_Vagina_min',
    '4.35_eucl_dist_Urethra_Uterus_std',
    '4.35_eucl_dist_Urethra_Uterus_max',
    '4.36_eucl_dist_Urethra_Vagina_std',
    '4.36_eucl_dist_Urethra_Vagina_min',
    '5.1_eccentricity_skimage_mean',
    '5.1_eccentricity_skimage_std',
    '5.1_eccentricity_skimage_max',
    '5.1_eccentricity_skimage_min',
]
