
from collections import defaultdict

import numpy as np
import pandas as pd

from schemas import Results

def average_classification_reports(reports):
    """
    Calcula la media de múltiples classification reports.
    
    Args:
        reports: Lista de diccionarios con classification reports
    
    Returns:
        dict: Classification report con valores promediados
    """
    if not reports:
        return {}
    
    keys = reports[0].keys()
    avg_report = {}
    
    for key in keys:
        if key == 'accuracy':
            # La accuracy es un valor simple
            avg_report['accuracy'] = np.mean([r['accuracy'] for r in reports])
        else:
            # Para 'False', 'True', 'macro avg', 'weighted avg'
            avg_report[key] = {}
            metrics = reports[0][key].keys()
            
            for metric in metrics:
                values = [r[key][metric] for r in reports]
                avg_report[key][metric] = np.mean(values)
    
    return avg_report



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

def group_predictions_by_case(eval_data: pd.DataFrame, prediction, y: pd.Series):
    """
    Agrupa predicciones por caso.
    """
    df_temp = pd.DataFrame({
        'case': eval_data['case'],
        'pred': prediction,
        'target': y
    })
    print(df_temp.head())
    grouped = df_temp.groupby('case').agg({
        'pred': 'mean',
        'target': 'first' 
    })
    print(grouped.head())
    final_preds = (grouped['pred'] >= 0.5).astype(int).values
    final_y = grouped['target'].astype(int).values

    return final_preds, final_y