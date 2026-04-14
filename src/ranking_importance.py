import pandas as pd
import numpy as np
from pathlib import Path

IMPORTANCE_DIR = Path('results/importance')
FILE_PATTERN = 'importance_todas_{experiment}_agg.csv'  
EXPERIMENT = 'w60_s15'  

def compute_score(df):
    """Score ajustado por consistencia: penaliza alta desviación relativa."""
    df = df[df['importance_mean'] > 0].copy()
    df['cv'] = df['importance_std'] / (df['importance_mean'] + 1e-9)
    df['score'] = df['importance_mean'] / (1 + df['cv'])
    return df.sort_values('score', ascending=False)


def load_prolapse_data(experiment: str):
    """Carga los CSVs de todas las carpetas de prolapso."""
    data = {}
    for prolapse_dir in IMPORTANCE_DIR.iterdir():
        if not prolapse_dir.is_dir():
            continue
        prolapse_name = prolapse_dir.name
        csv_path = prolapse_dir / FILE_PATTERN.format(experiment=experiment)
        if not csv_path.exists():
            print(f'WARNING: no encontrado {csv_path}')
            continue
        df = pd.read_csv(csv_path)
        data[prolapse_name] = compute_score(df)
    return data


def compute_global_ranking(data: dict):
    """Ranking global combinando todas las patologías."""
    rows = []
    all_features = set()
    for df in data.values():
        all_features.update(df['feature'])

    for feat in all_features:
        scores = []
        means = []
        stds = []
        n = 0
        for prolapse_name, df in data.items():
            row = df[df['feature'] == feat]
            if not row.empty and row['importance_mean'].values[0] > 0:
                scores.append(row['score'].values[0])
                means.append(row['importance_mean'].values[0])
                stds.append(row['importance_std'].values[0])
                n += 1

        if not scores:
            continue

        mean_imp = np.mean(means)
        mean_std = np.mean(stds)
        mean_score = np.mean(scores)
        cv = mean_std / (mean_imp + 1e-9)
        global_score = mean_score * n / (1 + cv)

        rows.append({
            'feature': feat,
            'importance_mean': round(mean_imp, 6),
            'importance_std': round(mean_std, 6),
            'cv': round(cv, 3),
            'n_diseases': n,
            'score': round(global_score, 6),
        })

    return pd.DataFrame(rows).sort_values('score', ascending=False)


def save_rankings(data: dict, global_ranking: pd.DataFrame, experiment: str):
    output_dir = IMPORTANCE_DIR / 'rankings'
    output_dir.mkdir(exist_ok=True)

    # Global
    global_path = output_dir / f'global_ranking_{experiment}.csv'
    global_ranking.to_csv(global_path, index=False)
    print(f'Global ranking guardado: {global_path}')

    # Por prolapso
    for prolapse_name, df in data.items():
        prolapse_out = output_dir / prolapse_name
        prolapse_out.mkdir(exist_ok=True)
        out_path = prolapse_out / f'ranking_{experiment}.csv'
        df[['feature', 'importance_mean', 'importance_std', 'cv', 'score']]\
            .to_csv(out_path, index=False)
        print(f'  {prolapse_name}: {out_path}')


if __name__ == '__main__':
    print(f'Cargando datos del experimento: {EXPERIMENT}')
    data = load_prolapse_data(EXPERIMENT)

    if not data:
        print('ERROR: no se encontraron archivos. Revisa EXPERIMENT y FILE_PATTERN.')
        exit(1)

    print(f'Patologías encontradas: {list(data.keys())}')

    global_ranking = compute_global_ranking(data)
    save_rankings(data, global_ranking, EXPERIMENT)

    print(f'\nTop 10 global:')
    print(global_ranking.head(10).to_string(index=False))