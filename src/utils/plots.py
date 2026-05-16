from pathlib import Path

import matplotlib.pyplot as plt
import os

expermients = ["res_net","tcn", "lstm"]

def plot_auc_pr_evol(prolapse_name, fold_idx, history, experiment="tcn"):
    if experiment not in expermients:
        raise Exception("Experimento no disponible")

    plt.figure()
    plt.plot(history.history['pr_auc'], label='Train PR-AUC')
    plt.plot(history.history['val_pr_auc'], label='Val PR-AUC')
    plt.legend()
    plt.title(f'Curvas de entrenamiento — {prolapse_name} | Fold {fold_idx}')

    save_path = Path(f'results/{experiment}/{prolapse_name}/training_curve_fold{fold_idx}.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path)
    plt.close()

def plot_loss(history, prolapse_name, fold_idx,experiment="tcn"):
    plt.figure(figsize=(10, 6))
    
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    # Dibujamos las líneas
    plt.plot(epochs, train_loss, 'bo-', label='Entrenamiento (Loss)')
    plt.plot(epochs, val_loss, 'r^-', label='Validación (Loss)')
    
    plt.title(f'Curva de Pérdida - {prolapse_name}')
    plt.xlabel('Épocas')
    plt.ylabel('Binary Crossentropy')
    plt.legend()
    plt.grid(True)
    
    save_path = Path(f'results/{experiment}/{prolapse_name}/loss_fold_{fold_idx}.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path)
    plt.close()


def plot_xgb_loss(evals_result: dict, prolapse_name: str, fold_idx: int):
    train_loss = evals_result['train']['logloss']
    val_loss   = evals_result['val']['logloss']
    plt.figure()
    plt.plot(train_loss, label='Entrenamiento (Loss)')
    plt.plot(val_loss,   label='Validación (Loss)')
    plt.xlabel('Árboles')
    plt.ylabel('Log Loss')
    plt.title(f'Curva de Pérdida - {prolapse_name} | Fold {fold_idx}')
    plt.legend()
    os.makedirs(f'results/xgboost/plots/{prolapse_name}', exist_ok=True)
    plt.savefig(f'results/xgboost/plots/{prolapse_name}/loss_fold{fold_idx}.png')
    plt.close()

def plot_xgb_ap(evals_result: dict, prolapse_name: str, fold_idx: int):
    train_ap = evals_result['train']['aucpr']
    val_ap   = evals_result['val']['aucpr']
    plt.figure()
    plt.plot(train_ap, label='Train PR-AUC')
    plt.plot(val_ap,   label='Val PR-AUC')
    plt.xlabel('Árboles')
    plt.ylabel('PR-AUC')
    plt.title(f'Curvas de entrenamiento — {prolapse_name} | Fold {fold_idx}')
    plt.legend()
    os.makedirs(f'results/xgboost/plots/{prolapse_name}', exist_ok=True)
    plt.savefig(f'results/xgboost/plots/{prolapse_name}/prauc_fold{fold_idx}.png')
    plt.close()