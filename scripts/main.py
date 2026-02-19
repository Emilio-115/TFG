import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score,average_precision_score,classification_report, confusion_matrix, recall_score
import optuna
from register_data import generate_html_report
from typing import List, Dict
from collections import defaultdict
from aggregations import group_cases_index,average_classification_reports

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
    
    return data,objective

def optimal_params(trial: optuna.Trial, data = None, objective = None):
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

class Results:
    def __init__(self, classif_report, conf_matrix):
        self.classif_report = classif_report
        self.conf_matrix = conf_matrix
        
    def __str__(self):
        
        return f"Report:\n{self.classif_report}\nMatrix:\n{self.conf_matrix}"
    def __repr__(self):
        return self.__str__()

def main_func():
    res = dict()
    
    for p in range(0,2):  # len(prolapses)-1
        data,objective = load_data(p)
        groups = data["case"] #Eliminar esta columna en el entreno?
        sgkf = StratifiedGroupKFold(n_splits=4,shuffle=True)
        train_index, eval_index = next(sgkf.split(data, objective, groups))
        train, train_obj = data.iloc[train_index].reset_index(drop=True),objective.iloc[train_index].reset_index(drop=True)
        eval_data, eval_obj = data.iloc[eval_index].reset_index(drop=True), objective.iloc[eval_index].reset_index(drop=True)
        
        def obj(trial):
            return optimal_params(trial, data=train, objective=train_obj)

        study = optuna.create_study(direction='maximize')
        study.optimize(obj, n_trials=25, n_jobs=5)

        eval_grouped_by_index = group_cases_index(eval_data)
        eval_data_no_case = eval_data.drop(columns=["case"])
        

        train_no_case = train.drop(columns=["case"])
        params = study.best_params
        params["enable_categorical"] = True
        xgbc = XGBClassifier(**params)
        xgbc.fit(X=train_no_case,y=train_obj)
        #prediction = xgbc.predict_proba(X=eval_data)[:,1]
        prediction = xgbc.predict(X=eval_data_no_case)
        #final_score = average_precision_score(y_true=eval_obj,y_score=prediction)

        grouped_pred, grouped_y = group_predictions_by_case(eval_grouped_by_index,prediction, eval_obj)
        #Descomentar estas dos
        # report = classification_report(y_true=eval_obj,y_pred=prediction,output_dict=True)
        # conf_matrix = confusion_matrix(y_true=eval_obj,y_pred=prediction) #[[TN FP],[FN TP]]
        
        report = classification_report(y_true=grouped_y,y_pred=grouped_pred,output_dict=True)
        conf_matrix = confusion_matrix(y_true=grouped_y,y_pred=grouped_pred) #[[TN FP],[FN TP]]

        res[prolapses[p]] = Results(report,conf_matrix)
    print(res)
    return res


def aggregate_experiments(experiments_list):
    """
    Calcula la media de los resultados de múltiples experimentos.
    
    Args:
        experiments_list: Lista de diccionarios, cada uno con estructura 
                         {enfermedad: Results}
    
    Returns:
        dict: Diccionario con la media de los resultados por enfermedad
    """
    aggregated = defaultdict(lambda: {
        'classification_reports': [],
        'confusion_matrices': []
    })
    
    for experiment in experiments_list:
        for disease, results in experiment.items():
            aggregated[disease]['classification_reports'].append(results.classif_report)
            aggregated[disease]['confusion_matrices'].append(results.conf_matrix)
    
    final_results = {}
    
    for disease, data in aggregated.items():
        conf_matrices = np.array(data['confusion_matrices'])
        avg_conf_matrix = np.mean(conf_matrices, axis=0)
        
        reports = data['classification_reports']
        avg_report = average_classification_reports(reports)

        final_results[disease] = Results(avg_report, avg_conf_matrix)
    
    return final_results




def group_predictions_by_case(cases_indexes: Dict, prediction, y: pd.Series):

    indexes_values = cases_indexes.values()

    grouped_preds = pd.Series([np.mean(prediction[case_indexes]) for case_indexes in indexes_values])
    grouped_y = pd.Series([int(np.mean(y.iloc[case_indexes])) for case_indexes in indexes_values])
    # print(cases_indexes.keys())
    # print("#"*10)
    # print(grouped_preds)
    # print("#"*10)
    # print(grouped_y)

    return np.where(grouped_preds>=0.5,1,0), grouped_y


if __name__ == "__main__":
    experiments = []
    for i in range(0,2):
        experiments.append(main_func())

    res = aggregate_experiments(experiments)
    context = "Reporte de resultados con la columna de casos eliminada, usando average_precission_score para la optimización y scale_pos_weigth para aquellos prolapsos desbalanceados"
    generate_html_report(res, "3.current_experiment.html", context)

