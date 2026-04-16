import os
import pickle as pk
import numpy as np
import pandas as pd
from datetime import datetime
import pdb
from typing import Dict, List
from sklearn.preprocessing import OrdinalEncoder

from .config import ORGANS
from .feature_definitions import (
    FRAME_CONF_STATS,
    REGION_CONF_STATS,
    REGION_COORDS_STATS,
    SPATIAL_STATS,
    SPATIAL_PAIRS,
    FRAME_SHAPE_STATS,
    euclidean_distance
)

# Llena los valores NaN de las secuencias de características por frame usando forward fill y backward fill.
def ff_nan_values(organ_dict):
    ret = {}
    seqs_w_na = 0
    seqs_w_na_bfill = 0
    for organ, seq_dict in organ_dict.items():
        ret[organ] = {}
        for seq_name, seq in seq_dict.items():
            pdseq = pd.Series(seq)
            if pdseq.isnull().sum() > 0:
                # print(f'FIXING: {organ:20s}{seq_name:30s}{pdseq.isnull().sum()}')
                seqs_w_na += 1
                pdseq = pdseq.ffill()
                if pdseq.isnull().sum() > 0:
                    # print(f'FIXING NA W BFILL: {organ:20s}{seq_name:30s}{pdseq.isnull().sum()}')
                    seqs_w_na_bfill += 1
                    pdseq = pdseq.bfill()
            ret[organ][seq_name] = pdseq
    return ret, seqs_w_na, seqs_w_na_bfill


# Calcula las características por frame para una secuencia de frames de un caso.
def compute_frame_features_for_seq(case_id, data_path, start_frame=None, end_frame=None, organs=ORGANS, frame_conf_feat=None, region_conf_feat=None, region_coords_feat=None, spatial_feat=None, frame_shape_feat=None):
    base_path = f'{data_path}/{case_id:03d}/seg'
    nframes = len(os.listdir(base_path))
    start = start_frame if start_frame else 0
    end = end_frame if end_frame else nframes

    if start < 0 or start >= nframes or end <= start or end > nframes:
        print(f'WARNING incorrect frame range. case {case_id} start {start_frame} end{end_frame}.')
        start = 0
        end = nframes

    frame_conf_keys    = frame_conf_feat    or list(FRAME_CONF_STATS.keys())
    frame_shape_keys   = frame_shape_feat   or list(FRAME_SHAPE_STATS.keys()) 
    region_conf_keys   = region_conf_feat   or list(REGION_CONF_STATS.keys())
    region_coords_keys = region_coords_feat or list(REGION_COORDS_STATS.keys())
    spatial_keys       = spatial_feat       or list(SPATIAL_STATS.keys())

    case_stats_by_organ = {
        organ: {feat_name: [] for feat_name in frame_conf_keys + region_conf_keys + region_coords_keys + frame_shape_keys}
        for organ in organs
    }
    case_stats_spatial = {k: [] for k in spatial_keys}

    for frame_id in range(start, end):
        pkframe = pk.load(open(f'{data_path}/{case_id:03d}/seg/frame_{frame_id}.pk', 'rb'))

        for organ in organs:
            frame_raw = pkframe[:, :, ORGANS.index(organ)]
            frame_organ_region_coords = np.where(frame_raw > 0.5)
            frame_organ_region = frame_raw[frame_organ_region_coords]

            for k in frame_conf_keys:
                case_stats_by_organ[organ][k].append(FRAME_CONF_STATS[k](frame_raw))
            for k in frame_shape_keys:
                case_stats_by_organ[organ][k].append(FRAME_SHAPE_STATS[k](frame_raw))
            for k in region_conf_keys:
                case_stats_by_organ[organ][k].append(REGION_CONF_STATS[k](frame_organ_region))
            for k in region_coords_keys:
                case_stats_by_organ[organ][k].append(REGION_COORDS_STATS[k](frame_organ_region_coords))

        for k, (organ1, organ2) in zip(spatial_keys, SPATIAL_PAIRS):
            cy1 = case_stats_by_organ[organ1]['3.1_region_coords_centroid_Y'][-1]
            cx1 = case_stats_by_organ[organ1]['3.2_region_coords_centroid_X'][-1]
            cy2 = case_stats_by_organ[organ2]['3.1_region_coords_centroid_Y'][-1]
            cx2 = case_stats_by_organ[organ2]['3.2_region_coords_centroid_X'][-1]
            point1 = np.array([cy1, cx1])
            point2 = np.array([cy2, cx2])
            case_stats_spatial[k].append(euclidean_distance(point1, point2))

    return case_stats_by_organ, case_stats_spatial

def initialize_org_spatial_keys(organs, spatial_keys):
    organ_spatial_keys = {organ: [] for organ in organs}
    for k, (organ1, organ2) in zip(spatial_keys, SPATIAL_PAIRS):
        organ_spatial_keys[organ1].append(k)
        organ_spatial_keys[organ2].append(k)
    return organ_spatial_keys

def build_keras_tensor_per_organ(case_frame_feats_ff, case_spatial_feats, organs):
    result = []
    
    spatial_matrix = np.stack([
        np.array(seq) for seq in case_spatial_feats.values()
    ], axis=1)
    
    for organ_idx, organ in enumerate(organs):
        features = []
        for feat_name, seq in case_frame_feats_ff[organ].items():
            features.append(seq.values)

        organ_matrix = np.stack(features, axis=1)  

        organ_matrix = np.concatenate([organ_matrix, spatial_matrix], axis=1)  
        result.append((organ, organ_matrix))
    return result

def create_dataset_keras(cases_df, data_path, window_size=60, window_step=15, organs=ORGANS, frame_conf_feat=None, region_conf_feat=None, region_coords_feat=None, seq_agg_feat=None):
    seq_count = 0
    case_seq_count = 0
    cases_wless_fr_than_wz = []
    X_list = []
    meta_list = []


    print(f'BUILDING DF FOR SEGMENTATION IN PATH {data_path}.\nDate {datetime.now()}')
    for case_id in cases_df.case:
        try:
            base_path = f'{data_path}/{case_id:03d}/seg'
            nframes = len(os.listdir(base_path))
        except FileNotFoundError:
            print(f'ERROR segmentation not found for case {case_id}. Skkiping case.')
            continue
        case_seq_count = 0
        
        for start_frame in range(0, (nframes - window_size), window_step):
            seq_count += 1
            case_seq_count += 1
            end_frame = start_frame + window_size

            case_frame_feats, case_spatial_feats = compute_frame_features_for_seq(
                case_id, data_path, start_frame=start_frame, end_frame=end_frame
            )
            case_frame_feats_ff, seqs_w_na, seqs_w_na_bfill = ff_nan_values(case_frame_feats)
            if seqs_w_na:
                print(f'NAs ForwardFilled in {seqs_w_na} seqs in case {case_id} from {start_frame} till {end_frame} frames.')
            if seqs_w_na_bfill:
                print(f'NAs BackwardFilled in {seqs_w_na_bfill} seqs in case {case_id} from {start_frame} till {end_frame} frames.')

            organ_windows = build_keras_tensor_per_organ(case_frame_feats_ff, case_spatial_feats, organs)
            for organ, X_window in organ_windows:
                X_list.append(X_window)
                meta_list.append({
                    "case_id": case_id,
                    "organ": organ,
                    "start_frame": start_frame,
                    "end_frame": end_frame
                })
        print(f'Procesed {case_seq_count} secuences from case {case_id} with {nframes} frames')
        if case_seq_count == 0:
            cases_wless_fr_than_wz.append(case_id)

    print(f'Process finished with {seq_count} sequences from {len(cases_df.case)} cases')
    print(f'Cases excluded, having less frames than window size: {cases_wless_fr_than_wz}')
    print(f"Primer X_window shape: {X_list[0].shape}")
    print(f"Organs usados: {organs}")
    X = np.stack(X_list) 
    X = np.nan_to_num(X, nan=0.0)
    return X, meta_list