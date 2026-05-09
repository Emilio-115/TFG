from datetime import datetime
from typing import List

import matplotlib
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from keras.models import Model
from keras.callbacks import History
from keras.backend import clear_session
from sklearn.model_selection import StratifiedGroupKFold

from src.cnn.models import make_model_res_net1D, make_model_tcn
from src.utils.plots import plot_auc_pr_evol, plot_loss
from src.utils.scaler import scale_data
from src.utils.tuner import get_keras_tuner, get_optuna_study
from src.xgboost_impl.aggregations import group_predictions_by_case
from src.xgboost_impl.register_data import generate_html_report
from src.xgboost_impl.schemas import Results
import optuna
from typing import Callable

matplotlib.use('Agg')

drop_selection = []
PROLAPSES = ['any_prolapse', 'cystocele', 'cystourethrocele', 'uterine_prolapse',
             'cervical_elongation', 'rectocele', 'enterocele']

DATA_PATH = 'data/case_level_feats_nn.npy'
META_DF_PATH = "data/nn_meta.csv"

RANKINGS_DIR = 'results/importance/rankings'

def load_data(prolapse: str = "any_prolapse", add_top_n: int = 10):
    #TODO Filtrar variables

    data: np.ndarray = np.load(DATA_PATH)

    meta_df = pd.read_csv(META_DF_PATH)
    objective = meta_df[prolapse].to_numpy()
    return data,objective,meta_df

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


def optuna_objective(
    trial: optuna.Trial,
    chosen_model: Callable[..., Model],
    input_shape: tuple,
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    fold_idx: int,
    inner_n_splits: int = 3,
) -> float:
    inner_sgkf = StratifiedGroupKFold(
        n_splits=inner_n_splits, shuffle=True,
        random_state=fold_idx * 10 + trial.number
    )

    scores = []
    for inner_train_idx, inner_val_idx in inner_sgkf.split(x_train, y_train, groups_train):
        x_tr, x_val = x_train[inner_train_idx], x_train[inner_val_idx]
        y_tr, y_val = y_train[inner_train_idx], y_train[inner_val_idx]

        num_pos = np.sum(y_tr)
        num_neg = len(y_tr) - num_pos
        inner_cw = {0: 1.0, 1: min(num_neg / num_pos, 6.0)}

        clear_session()
        model = chosen_model(trial, input_shape)
        model.fit(
            x_tr, y_tr,
            validation_data=(x_val, y_val),
            epochs=30,
            batch_size=64,
            class_weight=inner_cw,
            verbose=0,
            callbacks=[EarlyStopping(
                monitor="val_pr_auc", patience=3,
                restore_best_weights=True, mode="max"
            )],
        )
        val_probs = model.predict(x_val, verbose=0).flatten()
        scores.append(average_precision_score(y_true=y_val, y_score=val_probs))

    return np.mean(scores)

def run_experiment(target_prolapses: List[str], use_res_net: bool):
    res = dict()

    for prolapse_name in target_prolapses:
        print(f"\n · CNN: {prolapse_name} | {datetime.now()}")
        data, objective, meta_df = load_data(prolapse_name)
        experiment = "res_net" if use_res_net else "tcn"
        chosen_model = make_model_res_net1D if use_res_net else make_model_tcn
        groups = meta_df["case_id"].values
        sgkf   = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

        all_fold_probs = []
        all_fold_true  = []
        all_fold_meta  = []

        for fold_idx, (train_idx, eval_idx) in enumerate(sgkf.split(data, objective, groups)):
            print(f"\n · CNN: {prolapse_name} | Fold {fold_idx}")

            x_train, x_eval = scale_data(data[train_idx], data[eval_idx])
            y_train, y_eval = objective[train_idx], objective[eval_idx]

            num_pos = np.sum(y_train)
            num_neg = len(y_train) - num_pos
            cw = {0: 1.0, 1: num_neg / num_pos if num_pos > 0 else 1.0}
            input_shape = x_train.shape[1:]
            groups_train = meta_df.iloc[train_idx]["case_id"].values
            inner_splits = 2 if prolapse_name in {"cystourethrocele", "enterocele"} else 3

            def optimize_study(trial):
                return optuna_objective(trial, chosen_model, input_shape, x_train, y_train, groups_train, fold_idx, inner_splits)

            study = get_optuna_study(fold_idx, prolapse_name, experiment)
            study.optimize(optimize_study, n_trials=10, n_jobs=1)

            clear_session()
            best_trial = study.best_trial
            model = chosen_model(best_trial, input_shape)
            history = model.fit(
                x_train, y_train,
                validation_data=(x_eval, y_eval),
                epochs=25,
                batch_size=16,
                class_weight=cw,
                verbose=0,
                callbacks=[EarlyStopping(
                    monitor="val_pr_auc", patience=5,
                    restore_best_weights=True, mode="max"
                )],
            )

            probs = model.predict(x_eval, verbose=0).flatten()

            all_fold_probs.append(probs)
            all_fold_true.append(y_eval)
            all_fold_meta.append(meta_df.iloc[eval_idx])


            plot_auc_pr_evol(prolapse_name, fold_idx, history, experiment)
            plot_loss(history, prolapse_name, fold_idx, experiment)

            clear_session()

        combined_probs = np.concatenate(all_fold_probs)
        combined_true = np.concatenate(all_fold_true)
        combined_meta = pd.concat(all_fold_meta).reset_index(drop=True)

        grouped_pred, grouped_y = group_predictions_by_case(
            combined_meta, combined_probs, combined_true
        )

        res[prolapse_name] = obtain_final_metrics(
            y_true=grouped_y, y_pred_probs=grouped_pred
        )
        print(f"    {prolapse_name} finalizado. AP_1: {res[prolapse_name].ap_1:.4f}")

    return res



def main(res_net = True):
    df_name = "60w_15s"
    drop_name = "all"
    target_prolapses = PROLAPSES 
    experiment = "res_net" if res_net else "tcn"
    print(experiment)
    print(f"\n--- Iniciando Experimento CNN Comparativo: {df_name} | {drop_name} ---")
    
    experiment_results = run_experiment([target_prolapses[3]], res_net)
    
    context = f"{experiment.upper()} Model | Dataset: {df_name} | Features: {drop_name}"
    report_filename = f"exp_{df_name}_{drop_name}_{experiment}.html"
    
    generate_html_report(experiment_results, report_filename, context)
    print(f"\nReporte generado: {report_filename}")


if __name__ == "__main__":
    main()