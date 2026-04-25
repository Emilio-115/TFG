from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from src.lstm.models import make_model_bilstm
from src.utils.plots import plot_auc_pr_evol, plot_loss
from src.utils.scaler import scale_data
from src.utils.tuner import get_tuner
from src.xgboost_impl.aggregations import group_predictions_by_case
from src.xgboost_impl.register_data import generate_html_report
from src.xgboost_impl.schemas import Results

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

def run_experiment(target_prolapses: List[str]):
    res = dict()
    
    for prolapse_name in target_prolapses:
        print(f"\n · LSTM: {prolapse_name} | {datetime.now()}")
        data, objective, meta_df = load_data(prolapse_name)
        
        groups = meta_df["case_id"].values
        sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

        all_fold_probs = []
        all_fold_true = []
        all_fold_meta = []

        for fold_idx, (train_idx, eval_idx) in enumerate(sgkf.split(data, objective, groups)):
            print(f"\n · LSTM: {prolapse_name} | Fold {fold_idx}")
            x_train, x_eval = scale_data(data[train_idx], data[eval_idx])
            y_train, y_eval = objective[train_idx], objective[eval_idx]
            
            num_pos = np.sum(y_train)
            num_neg = len(y_train) - num_pos
            cw = {0: 1.0, 1: num_neg / num_pos if num_pos > 0 else 1.0}

            input_shape=x_train.shape[1:]


            tuner = get_tuner(lambda hp: make_model_bilstm(hp, input_shape),fold_idx=fold_idx, prolapse_name=prolapse_name,experiment="lstm")

            tuner.search(x_train, y_train,
                validation_data=(x_eval, y_eval),
                epochs=25,
                batch_size=16,
                class_weight=cw,
                verbose=0,
                callbacks=[EarlyStopping(monitor='val_pr_auc', patience=5, restore_best_weights=True, mode='max')])
            
            hp = tuner.get_best_hyperparameters()[0]

            model = make_model_bilstm(hp,input_shape=x_train.shape[1:])
            

            history = model.fit(
                x_train, y_train,
                validation_data=(x_eval, y_eval),
                epochs=25,
                batch_size=16,
                class_weight=cw,
                verbose=0,
                callbacks=[EarlyStopping(monitor='val_pr_auc', patience=5, restore_best_weights=True, mode='max')]
            )

            probs = model.predict(x_eval, verbose=0).flatten()
            
            all_fold_probs.append(probs)
            all_fold_true.append(y_eval)
            all_fold_meta.append(meta_df.iloc[eval_idx])

            plot_auc_pr_evol(prolapse_name, fold_idx, history, cnn=False)
            plot_loss(history, prolapse_name,fold_idx,cnn=False)

        combined_probs = np.concatenate(all_fold_probs)
        combined_true = np.concatenate(all_fold_true)
        combined_meta = pd.concat(all_fold_meta).reset_index(drop=True)

        grouped_pred, grouped_y = group_predictions_by_case(
            combined_meta, combined_probs, combined_true
        )

        res[prolapse_name] = obtain_final_metrics(y_true=grouped_y, y_pred_probs=grouped_pred)
        print(f"    {prolapse_name} finalizado. AP_1: {res[prolapse_name].ap_1:.4f}")

    return res



def main():
    df_name = "60w_15s"
    drop_name = "all"
    target_prolapses = PROLAPSES 

    print(f"\n--- Iniciando Experimento LSTM Comparativo: {df_name} | {drop_name} ---")
    
    experiment_results = run_experiment([target_prolapses[1]])
    
    context = f"LSTM Model | Dataset: {df_name} | Features: {drop_name}"
    report_filename = f"exp_{df_name}_{drop_name}_lstm.html"
    
    generate_html_report(experiment_results, report_filename, context)
    print(f"\nReporte generado: {report_filename}")


if __name__ == "__main__":
    main()