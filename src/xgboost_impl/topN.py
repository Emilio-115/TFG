import pandas as pd
from pathlib import Path

IMPORTANCE_DIR = Path('results/importance/rankings')
EXPERIMENT = 'w60_s15'  
TOP_N = 100

ranking = pd.read_csv(IMPORTANCE_DIR / f'global_ranking_{EXPERIMENT}.csv')
top_features = set(ranking.head(TOP_N)['feature'])


sample_df = pd.read_csv('data/case_level_feats_alltargets_w60_s30_v3.csv', nrows=1)


meta_cols = {'case', 'nhc_final', 'start_frame', 'end_frame', 'organ', 'organ_num',
             'any_prolapse', 'cystocele', 'cystourethrocele', 'uterine_prolapse',
             'cervical_elongation', 'rectocele', 'enterocele'}

feature_cols = [c for c in sample_df.columns if c not in meta_cols]
drop_cols = [c for c in feature_cols if c not in top_features]

print(f'Total features: {len(feature_cols)}')
print(f'Top {TOP_N} a conservar: {len(top_features & set(feature_cols))}')
print(f'A eliminar: {len(drop_cols)}')
print()
print('DROP_TOP134 = [')
for col in drop_cols:
    print(f"    '{col}',")
print(']')