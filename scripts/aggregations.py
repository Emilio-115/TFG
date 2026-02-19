
import numpy as np
import pandas as pd
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


def group_cases_index(data: pd.DataFrame,):
    '''
    Metodo que devuelve un diccionario en el que para cada caso almacena una lista con los indices del
    frame en la tabla
    
    :param data: Datos con la columna case
    :param objetive: Columna objetivo
    '''
    
    return data.groupby("case").groups