from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def plot_auc_pr_evol(prolapse_name, fold_idx, history, cnn = True):
    folder = "cnn" if cnn else "lstm"

    plt.figure()
    plt.plot(history.history['pr_auc'], label='Train PR-AUC')
    plt.plot(history.history['val_pr_auc'], label='Val PR-AUC')
    plt.legend()
    plt.title(f'Curvas de entrenamiento — {prolapse_name} | Fold {fold_idx}')

    save_path = Path(f'results/{folder}/{fold_idx}/training_curve_{prolapse_name}_fold{fold_idx}.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path)
    plt.close()

def plot_pred_vs_y(prolapse_name, grouped_pred, grouped_y, cnn=True):
    folder = "cnn" if cnn else "lstm"

    plt.figure(figsize=(10,5))

    x = np.arange(len(grouped_y))

    plt.plot(x, grouped_y, label="Real", linewidth=2)
    plt.plot(x, grouped_pred, label="Predicción", linewidth=2)

    plt.xlabel("Muestra")
    plt.ylabel("Valor")
    plt.title(f"Predicción vs Real (línea) — {prolapse_name}")
    plt.legend()

    save_path = Path(f'results/{folder}/line_pred_vs_real_{prolapse_name}.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path)
    plt.close()

def plot_loss(history, prolapse_name, fold_idx,cnn=True):
    folder = "cnn" if cnn else "lstm"

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
    
    save_path = Path(f'results/{folder}/loss_{prolapse_name}_fold_{fold_idx}.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path)
    plt.close()