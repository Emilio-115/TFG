import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_data(x_train: np.ndarray, x_eval: np.ndarray):
    """
    Normaliza train y eval usando estadísticas.
    
    Args:
        x_train: (n_train, n_frames, n_features)
        x_eval:  (n_eval,  n_frames, n_features)
    
    Returns:
        x_train_scaled, x_eval_scaled con la misma shape de entrada
    """
    _, n_frames, n_features = x_train.shape

    scaler = StandardScaler()
    
    x_train_scaled = scaler.fit_transform(
        x_train.reshape(-1, n_features)
    ).reshape(-1, n_frames, n_features)
    
    x_eval_scaled = scaler.transform(
        x_eval.reshape(-1, n_features)
    ).reshape(-1, n_frames, n_features)
    
    return x_train_scaled, x_eval_scaled