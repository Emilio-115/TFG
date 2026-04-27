from pathlib import Path

import matplotlib.pyplot as plt

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