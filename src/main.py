import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score,average_precision_score,classification_report, confusion_matrix, recall_score
import optuna
from register_data import generate_html_report
from typing import List, Dict
from aggregations import aggregate_experiments, group_predictions_by_case
from schemas import Results

col_selection = ['case', 'nhc_final', 'start_frame', 'end_frame', 'organ', 'organ_num', 
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

prolapses = ['any_prolapse', 'cystocele', 'cystourethrocele', 'uterine_prolapse', 
                 'cervical_elongation', 'rectocele', 'enterocele']

DF_PATH = 'data/case_level_feats_alltargets_v2.csv'

def load_data(prolapse:int = 0):
    ignored_obj = np.delete(prolapses, [prolapse])
    df = pd.read_csv(DF_PATH, usecols=lambda col: col not in ignored_obj)
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
    
    for p in range(0,len(prolapses)):
        data,objective = load_data(p)
        groups = data["case"]
        sgkf = StratifiedGroupKFold(n_splits=3,shuffle=True)
        train_index, eval_index = next(sgkf.split(data, objective, groups))
        train, train_obj = data.iloc[train_index].reset_index(drop=True),objective.iloc[train_index].reset_index(drop=True)
        eval_data, eval_obj = data.iloc[eval_index].reset_index(drop=True), objective.iloc[eval_index].reset_index(drop=True)
        
        def obj(trial):
            return optimize_params(trial, data=train, objective=train_obj)

        study = optuna.create_study(direction='maximize')
        study.optimize(obj, n_trials=50, n_jobs=5)

        eval_data_no_case = eval_data.drop(columns=["case"])
        
        pos_weight = max(1.0,len(train_obj[train_obj==False]) / len(train_obj[train_obj==True]))
        
        train_no_case = train.drop(columns=["case"])
        params = study.best_params
        params["enable_categorical"] = True
        xgbc = XGBClassifier(**params,scale_pos_weight=pos_weight)
        xgbc.fit(X=train_no_case,y=train_obj)
        #prediction = xgbc.predict_proba(X=eval_data)[:,1]
        prediction = xgbc.predict_proba(X=eval_data_no_case)[:,1]

        grouped_pred, grouped_y = group_predictions_by_case(eval_data,prediction, eval_obj)
        #Descomentar estas dos
        # report = classification_report(y_true=eval_obj,y_pred=prediction,output_dict=True)
        # conf_matrix = confusion_matrix(y_true=eval_obj,y_pred=prediction) #[[TN FP],[FN TP]]
        
        report = classification_report(y_true=grouped_y,y_pred=grouped_pred,output_dict=True)
        conf_matrix = confusion_matrix(y_true=grouped_y,y_pred=grouped_pred) #[[TN FP],[FN TP]]

        res[prolapses[p]] = Results(report,conf_matrix)
    print(res)
    return res


if __name__ == "__main__":
    experiments = []
    for i in range(0,1):
        experiments.append(main_func())

    res = aggregate_experiments(experiments)
    context = "Reporte de resultados una repeticion con la columna de casos, nhc y frames eliminadas, usando average_precission_score para la optimización y sin aplicar scale_pos_weigth para aquellos prolapsos desbalanceados"
    generate_html_report(res, "test.html", context)

