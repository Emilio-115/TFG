import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
import optuna
from src.dataset_builder.feature_definitions import COL_SELECTION
from src.utils.plots import plot_xgb_ap, plot_xgb_loss
from .register_data import generate_html_report
from .aggregations import group_predictions_by_case
from .schemas import Results
from sklearn.inspection import permutation_importance
from datetime import datetime
import os
from pathlib import Path

drop_selection = []
prolapses = ['any_prolapse', 'cystocele', 'cystourethrocele', 'uterine_prolapse',
             'cervical_elongation', 'rectocele', 'enterocele']
DF_PATH = 'data/case_level_feats_alltargets_w60_s15_v3.csv'
save_perm_name = "importance"


RANKINGS_DIR = 'results/importance/rankings'

def load_data(prolapse: int = 0, use_top_n: int = None,add_top_n: int = None):
    prolapse_name = prolapses[prolapse]
    ignored_obj = np.delete(prolapses, [prolapse])
    df = pd.read_csv(DF_PATH, usecols=lambda col: col not in ignored_obj)
    data, objective = df.drop(prolapse_name, axis=1), df[prolapse_name]
    data['organ'] = data['organ'].astype('category')
    

    if use_top_n is not None:
        ranking_path = Path(RANKINGS_DIR) / prolapse_name / 'ranking_w60_s15.csv'
        if ranking_path.exists():
            ranking = pd.read_csv(ranking_path)
            top_features = set(ranking.head(use_top_n)['feature'])

            meta_cols = {'case', 'organ'}
            cols_to_keep = [c for c in data.columns if c in top_features or c in meta_cols]
            data = data[cols_to_keep]
            print(f"    {prolapse_name}: {len(cols_to_keep) - len(meta_cols)} features cargadas desde ranking")
        else:
            print(f"    WARNING: ranking no encontrado en {ranking_path}, usando todas las features")
    elif add_top_n is not None:
        data = data[[c for c in COL_SELECTION if c not in prolapses]] #Añadida para coger originales más top 10
        ranking_path = Path(RANKINGS_DIR) / prolapse_name / 'ranking_w60_s15.csv'
        if ranking_path.exists():
            ranking = pd.read_csv(ranking_path)
            top_features = set(ranking.head(add_top_n)['feature'])
            cols_to_add = [c for c in top_features if c in df.columns and c not in data.columns]
            if cols_to_add:
                data = pd.concat([data, df[cols_to_add]], axis=1)
            print(f"    {prolapse_name}: {len(cols_to_add)} features nuevas añadidas desde ranking top{add_top_n}")
        else:
            print(f"    WARNING: ranking no encontrado en {ranking_path}, usando todas las features")
                
    data = data.drop(columns=["nhc_final", 'start_frame', 'end_frame', 'organ_num'] + drop_selection)
    return data, objective


def optimize_params(trial: optuna.Trial, data: pd.DataFrame = None, objective: pd.Series = None):

    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True)
    groups = data["case"]
    data = data.drop(columns=["case"])
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'enable_categorical': True
    }

    scores = []
    for train_index, test_index in sgkf.split(data, objective, groups):
        assert set(np.unique(groups[train_index])).isdisjoint(set(np.unique(groups[test_index])))
        train, train_obj = data.iloc[train_index], objective.iloc[train_index]
        test, test_obj = data.iloc[test_index], objective.iloc[test_index]

        pos_weight = 1.0
        if len(train_obj[train_obj == True]) > 0:
            pos_weight = max(1.0, len(train_obj[train_obj == False]) / len(train_obj[train_obj == True]))

        xgbc = XGBClassifier(**params, scale_pos_weight=pos_weight, base_score=0.5)
        xgbc.fit(X=train, y=train_obj)
        prediction = xgbc.predict_proba(X=test)[:, 1]
        scores.append(average_precision_score(y_true=test_obj, y_score=prediction))

    return np.mean(scores)


def run_experiment(target_prolapses: list[str]):
    res = dict()
    all_perm_importances = {p: [] for p in target_prolapses}
    
    for prolapse_name in target_prolapses:
        p = prolapses.index(prolapse_name)
        print(f"\n  · {prolapse_name} | {datetime.now()}")
        data, objective = load_data(p,add_top_n = 10)
        groups = data["case"]
        sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True)

        all_predictions = []
        all_true_labels = []
        all_eval_data = []
        
        for fold_idx, (train_index, eval_index) in enumerate(sgkf.split(data, objective, groups)):
            train = data.iloc[train_index].reset_index(drop=True)
            train_obj = objective.iloc[train_index].reset_index(drop=True)
            eval_data = data.iloc[eval_index].reset_index(drop=True)
            eval_obj = objective.iloc[eval_index].reset_index(drop=True)
            eval_data_no_case = eval_data.drop(columns=["case"])

            def obj(trial, _train=train, _train_obj=train_obj):
                return optimize_params(trial, data=_train, objective=_train_obj)

            study = optuna.create_study(direction='maximize')
            study.optimize(obj, n_trials=35, n_jobs=5)

            pos_weight = 1.0
            if len(train_obj[train_obj == True]) > 0:
                pos_weight = max(1.0, len(train_obj[train_obj == False]) / len(train_obj[train_obj == True]))

            train_no_case = train.drop(columns=["case"])
            params = study.best_params
            params["enable_categorical"] = True

            xgbc = XGBClassifier(**params, scale_pos_weight=pos_weight, base_score=0.5, eval_metric=['logloss', 'aucpr'])
            xgbc.fit(
                X=train_no_case, y=train_obj,
                eval_set=[
                    (train_no_case, train_obj),
                    (eval_data_no_case, eval_obj)
                ],
                
                verbose=False,
            )

            evals_result = xgbc.evals_result()

            evals_renamed = {
                'train': evals_result['validation_0'],
                'val':   evals_result['validation_1']
            }

            prediction = xgbc.predict_proba(X=eval_data_no_case)[:, 1]

            plot_xgb_loss(evals_renamed, prolapse_name, fold_idx)
            plot_xgb_ap(evals_renamed,   prolapse_name, fold_idx)
            
            perm_result = permutation_importance(
                xgbc, eval_data_no_case, eval_obj,
                n_repeats=10, random_state=42, scoring='average_precision'
            )
            fold_imp = pd.DataFrame({
                'feature': eval_data_no_case.columns,
                'importance_mean': perm_result.importances_mean,
                'importance_std': perm_result.importances_std
            })
            all_perm_importances[prolapse_name].append(fold_imp)

            all_predictions.append(prediction)
            all_eval_data.append(eval_data)
            all_true_labels.append(eval_obj)

        combined_predictions = np.concatenate(all_predictions)
        combined_eval_data = pd.concat(all_eval_data).reset_index(drop=True)
        combined_true_labels = pd.concat(all_true_labels).reset_index(drop=True)

        grouped_pred, grouped_y = group_predictions_by_case(
            combined_eval_data, combined_predictions, combined_true_labels
        )

        res[prolapse_name] = obtain_final_metrics(y_true=grouped_y, y_pred_probs=grouped_pred)
        print(f"    {prolapse_name}: {res[prolapse_name]}")

    save_permutation_importance(all_perm_importances)
    return res


def save_permutation_importance(all_perm_importances: dict):
    for prolapse_name, fold_imps in all_perm_importances.items():
        os.makedirs(f'results/importance/{prolapse_name}', exist_ok=True)

        all_folds_df = pd.concat(fold_imps).reset_index(drop=True)
        agg_imp = (
            all_folds_df.groupby('feature')
            .agg(importance_mean=('importance_mean', 'mean'),
                 importance_std=('importance_mean', 'std'))
            .sort_values('importance_mean', ascending=False)
            .reset_index()
        )
        agg_path = f'results/importance/{prolapse_name}/{save_perm_name}_agg.csv'
        agg_imp.to_csv(agg_path, index=False)
        print(f"  Permutation importance agregada guardada: {agg_path}")


def obtain_final_metrics(y_true, y_pred_probs):
    y_pred_classes = (y_pred_probs > 0.5).astype(int)
    report = classification_report(y_true=y_true, y_pred=y_pred_classes, output_dict=True)
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred_classes)
    ap_1 = average_precision_score(y_true=y_true, y_score=y_pred_probs)
    ap_0 = average_precision_score(1 - y_true, 1 - y_pred_probs)
    roc_auc_1 = roc_auc_score(y_true=y_true, y_score=y_pred_probs)
    roc_auc_0 = roc_auc_score(y_true=1 - y_true, y_score=1 - y_pred_probs)
    return Results(classif_report=report, conf_matrix=conf_matrix,
                   ap_0=ap_0, ap_1=ap_1, roc_auc_0=roc_auc_0, roc_auc_1=roc_auc_1)


def main(prolapses=prolapses, run_experiment=run_experiment):
    dfs = {
        #'w50_s25': 'data/case_level_feats_alltargets_w50_s25_v3.csv',
        'w60_s15': 'data/case_level_feats_alltargets_w60_s15_v3.csv',
        #'w60_s20': 'data/case_level_feats_alltargets_w60_s20_v3.csv',
        #'w60_s30': 'data/case_level_feats_alltargets_w60_s30_v3.csv',
        #'w70_s30': 'data/case_level_feats_alltargets_w70_s30_v3.csv'
    }
    
    drop_data = {
        #'top80': COLS_TO_DROP,
        #'original_and_top10': COLS_DROP_ALL_NEW_EXCEPT_TOP10,
        #'original': DROP_NEW,
        #'todas':[],
        'add_indiv_top10':[],
    }
    target_prolapses = prolapses  # o una sublista: ['cystocele', 'rectocele']

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for df_name, df_file in dfs.items():
        for drop_name, drop_cols in drop_data.items():
            print(f"\n--- Experimento: {df_name} | {drop_name} | {datetime.now()} ---")
            DF_PATH = df_file
            drop_selection = drop_cols
            save_perm_name = f"importance_{drop_name}_{df_name}"

            experiment = run_experiment(target_prolapses)
            context = f"Dataset: {df_name} | Features: {drop_name}"
            generate_html_report(experiment, f"exp_{df_name}_{drop_name}.html", context)

if __name__ == "__main__":
    main(prolapses, run_experiment)