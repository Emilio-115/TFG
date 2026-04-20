import pandas as pd
import numpy as np
from keras.layers import Dense, GlobalAveragePooling1D, Input, Conv1D, BatchNormalization, ReLU, Dropout
from keras.models import Model
from keras.metrics import AUC
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from src.dataset_builder.feature_definitions import COL_SELECTION, NEW_FEATS
from datetime import datetime
from typing import List
import matplotlib.pyplot as plt
from src.xgboost_impl.register_data import generate_html_report
from src.xgboost_impl.schemas import Results
from src.xgboost_impl.aggregations import group_predictions_by_case


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
    breakpoint()
    return data,objective,meta_df


def make_model(input_shape):
    inputs = Input(shape=input_shape)

    # Bloque Convolucional 1 (Extracción de características de bajo nivel)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x) # Previene el sobreajuste apagando neuronas aleatoriamente

    # Bloque Convolucional 2 (Extracción de características complejas)
    x = Conv1D(filters=64, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)

    x = GlobalAveragePooling1D()(x)

    x = Dense(32, activation="relu")(x)
    x = Dropout(0.4)(x)     
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[AUC(name='pr_auc', curve='PR')] # Esto es similar al AP
    )

    return model

def obtain_final_metrics(y_true, y_pred_probs):
    # Convertimos probabilidades a clases (umbral 0.5) para el reporte
    y_pred_classes = (y_pred_probs > 0.5).astype(int)
    
    report = classification_report(y_true=y_true, y_pred=y_pred_classes, output_dict=True)
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred_classes)
    
    # Calculamos AP (Average Precision) y ROC AUC usando probabilidades
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
            # Split
            x_train, x_eval = data[train_idx], data[eval_idx]
            y_train, y_eval = objective[train_idx], objective[eval_idx]
            
            # Balanceo de carga (como el scale_pos_weight de XGBoost)
            num_pos = np.sum(y_train)
            num_neg = len(y_train) - num_pos
            cw = {0: 1.0, 1: num_neg / num_pos if num_pos > 0 else 1.0}

            # Modelo
            model = make_model(input_shape=x_train.shape[1:])
            
            # Entrenamiento (Silencioso para no ensuciar la consola)
            history = model.fit(
                x_train, y_train,
                validation_data=(x_eval, y_eval),
                epochs=50,
                batch_size=16,
                class_weight=cw,
                verbose=0,
                callbacks=[EarlyStopping(monitor='val_pr_auc', patience=10, restore_best_weights=True, mode='max')]
            )

            # Predicción de probabilidades (equivalente a predict_proba de XGB)
            probs = model.predict(x_eval, verbose=0).flatten()
            
            all_fold_probs.append(probs)
            all_fold_true.append(y_eval)
            all_fold_meta.append(meta_df.iloc[eval_idx])

            plt.figure()
            plt.plot(history.history['pr_auc'], label='Train PR-AUC')
            plt.plot(history.history['val_pr_auc'], label='Val PR-AUC')
            plt.legend()
            plt.title(f'Curvas de entrenamiento — {prolapse_name} | Fold {fold_idx}')
            plt.savefig(f'results/training_curve_{prolapse_name}_fold{fold_idx}.png')
            plt.close()


        # Concatenar resultados de todos los folds
        combined_probs = np.concatenate(all_fold_probs)
        combined_true = np.concatenate(all_fold_true)
        combined_meta = pd.concat(all_fold_meta).reset_index(drop=True)

        # Agrupación por caso (usando tu función original)
        # Asegúrate de que combined_meta tenga las columnas que espera group_predictions_by_case
        grouped_pred, grouped_y = group_predictions_by_case(
            combined_meta, combined_probs, combined_true
        )

        # Cálculo de métricas finales (Results)
        res[prolapse_name] = obtain_final_metrics(y_true=grouped_y, y_pred_probs=grouped_pred)
        print(f"    {prolapse_name} finalizado. AP_1: {res[prolapse_name].ap_1:.4f}")

    return res



def main():
    df_name = "60w_15s"
    drop_name = "all"
    target_prolapses = PROLAPSES 

    print(f"\n--- Iniciando Experimento LSTM Comparativo: {df_name} | {drop_name} ---")
    
    experiment_results = run_experiment(target_prolapses)
    
    # Generar reporte con tu formato
    context = f"LSTM Model | Dataset: {df_name} | Features: {drop_name}"
    report_filename = f"exp_{df_name}_{drop_name}_lstm.html"
    
    generate_html_report(experiment_results, report_filename, context)
    print(f"\nReporte generado: {report_filename}")


if __name__ == "__main__":
    main()