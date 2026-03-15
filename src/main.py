import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score,average_precision_score,classification_report, confusion_matrix, recall_score
import optuna
from dataset_builder.feature_definitions import COL_SELECTION
from register_data import generate_html_report
from typing import List, Dict
from aggregations import  group_predictions_by_case
from schemas import Results
from sklearn.inspection import permutation_importance

col_selection = COL_SELECTION

prolapses = ['any_prolapse', 'cystocele', 'cystourethrocele', 'uterine_prolapse', 
                 'cervical_elongation', 'rectocele', 'enterocele']

DF_PATH = 'data/case_level_feats_alltargets_v2.csv'
DF_PATH2 = 'data/case_level_feats_alltargets_w30.csv'
DF_PATH3 = 'data/case_level_feats_alltargets_w90.csv'
DF_PATH4 = "data/case_level_feats_alltargets_w60_s30_v3.csv"

def load_data(prolapse:int = 0):
    ignored_obj = np.delete(prolapses, [prolapse])
    df = pd.read_csv(DF_PATH4, usecols=lambda col: col not in ignored_obj)
    data,objective = df.drop(prolapses[prolapse],axis=1),df[prolapses[prolapse]]
    
    data['organ'] = data['organ'].astype('category')
    data = data.drop(columns=["nhc_final", 'start_frame', 'end_frame'])

    
    return data,objective

def optimize_params(trial: optuna.Trial, data: pd.DataFrame = None, objective: pd.Series = None):
    if data is None or objective is None:
        data,objective = load_data()
    
    sgkf = StratifiedGroupKFold(n_splits=4,shuffle=True)
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
        # 'random_state': 66,
        'enable_categorical':True
    }
    
    scores = []
    
    for i, (train_index, test_index) in enumerate(sgkf.split(data, objective, groups)):

        assert set(np.unique(groups[train_index])).isdisjoint(set(np.unique(groups[test_index])))
        
        train, train_obj = data.iloc[train_index],objective.iloc[train_index]
        test, test_obj = data.iloc[test_index], objective.iloc[test_index]
        
        pos_weight = max(1.0,len(train_obj[train_obj==False]) / len(train_obj[train_obj==True]))

        xgbc = XGBClassifier(**params, scale_pos_weight=pos_weight)
        xgbc.fit(X=train,y=train_obj)
        
        prediction = xgbc.predict_proba(X=test)[:,1]
        #prediction = xgbc.predict(X=test)
        scores.append(average_precision_score(y_true=test_obj,y_score=prediction))
        #scores.append(recall_score(y_true=test_obj,y_pred=prediction))
    return np.mean(scores)



def main_func():
    res = dict()
    for p in range(0, len(prolapses)):
        data, objective = load_data(p)
        groups = data["case"]
        sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True)

        all_predictions = []
        all_true_labels = []
        all_eval_data = []

        for i, (train_index, eval_index) in enumerate(sgkf.split(data, objective, groups)):
            train = data.iloc[train_index].reset_index(drop=True)
            train_obj = objective.iloc[train_index].reset_index(drop=True)
            eval_data = data.iloc[eval_index].reset_index(drop=True)
            eval_obj = objective.iloc[eval_index].reset_index(drop=True)


            def obj(trial, _train = train, _train_obj=train_obj):
                return optimize_params(trial, data=_train, objective=_train_obj)

            study = optuna.create_study(direction='maximize')
            study.optimize(obj, n_trials=25, n_jobs=5)

            pos_weight = max(1.0, len(train_obj[train_obj == False]) / len(train_obj[train_obj == True]))
            train_no_case = train.drop(columns=["case"])
            params = study.best_params
            params["enable_categorical"] = True

            xgbc = XGBClassifier(**params, scale_pos_weight=pos_weight)
            xgbc.fit(X=train_no_case, y=train_obj)

            eval_data_no_case = eval_data.drop(columns=["case"])
            prediction = xgbc.predict_proba(X=eval_data_no_case)[:, 1]

            all_predictions.append(prediction)
            all_eval_data.append(eval_data)
            all_true_labels.append(eval_obj)

        combined_predictions = np.concatenate(all_predictions)
        combined_eval_data = pd.concat(all_eval_data).reset_index(drop=True)
        combined_true_labels = pd.concat(all_true_labels).reset_index(drop=True)

        grouped_pred, grouped_y = group_predictions_by_case(
            combined_eval_data, combined_predictions, combined_true_labels
        )
        
        res[prolapses[p]] = obtain_final_metrics(y_true=grouped_y,y_pred=grouped_pred)
        print(f"{prolapses[p]}: {res[prolapses[p]]}")

    return res


def obtain_final_metrics(y_true,y_pred):
    report = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    ap_1 = average_precision_score(y_true=y_true,y_score=y_pred)
    ap_0 = average_precision_score(1 - y_true, 1 - y_pred)
    roc_auc_1 = roc_auc_score(y_true=y_true,y_score=y_pred)
    roc_auc_0 = roc_auc_score(y_true=1-y_true,y_score=1-y_pred)
    return Results(classif_report=report,conf_matrix=conf_matrix,ap_0=ap_0,ap_1=ap_1,roc_auc_0=roc_auc_0,roc_auc_1=roc_auc_1)


if __name__ == "__main__":
    experiment = main_func()

    context = "Reporte de resultados una repeticion con la columna de casos, nhc y frames eliminadas, usando average_precission_score para la optimización y aplicando scale_pos_weigth para aquellos prolapsos desbalanceados y los datos usando una ventana de 90 en su generación(paso 30)."
    generate_html_report(experiment, "test_all_metrics2.html", context)

