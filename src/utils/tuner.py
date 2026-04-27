import keras_tuner as kt
from typing import Callable
import os

def get_tuner(model_builder: Callable, fold_idx, prolapse_name, experiment):

    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    tuner_dir = os.path.join(BASE_DIR, "results", experiment, prolapse_name, "keras_tuner_results")
    os.makedirs(tuner_dir, exist_ok=True)
    project_name = f"tuner_fold_{fold_idx}"

    return kt.BayesianOptimization(
        hypermodel=model_builder,
        objective=kt.Objective("val_pr_auc", direction="max"),
        max_trials=10,
        directory=tuner_dir,
        project_name=project_name,
        overwrite=False
    )